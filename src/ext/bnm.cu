/*
Reduced Wong-Wang model (Deco 2014) simulation on GPU

This code includes kernels needed for the simulation of BOLD signal and calculation of 
FC and FCD, in addition to other GPU-related functions.

Each simulation (for a given set of parameters) is run on a single GPU block, and
each thread in the block simulates one region of the brain. The threads in the block
form a "cooperative group" and are synchronized after each integration time step.
Calculation of FC and FCD are also similarly parallelized across simulations and region or
window pairs.

Parts of this code are based on https://github.com/BrainModes/The-Hybrid-Virtual-Brain, 
https://github.com/murraylab/hbnm & https://github.com/decolab/cb-neuromod

Author: Amin Saberi, Feb 2023
*/
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <math.h>
#include <sys/stat.h>
#include <iostream>
#include <fstream>
#include <cmath>
#include <string>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <random>
#include <algorithm>
#include <map>
#include <gsl/gsl_matrix_double.h>
#include <gsl/gsl_vector_double.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_roots.h>

// TODO: clean up the includes
// and remove unncessary header files
// Note: 
#include "cubnm/defines.h"
#include "./utils.cu"
#include "cubnm/models/base.cuh"
__constant__ ModelConfigs d_conf;
#include "./fc.cu"
#include "./models/bw.cu"
#include "cubnm/bnm.cuh"
#include "./models/rww.cu"
// other models go here

cudaDeviceProp prop;

namespace bnm_gpu {
    bool is_initialized = false;
    int n_vols_remove, corr_len, n_windows, n_pairs, n_window_pairs, output_ts, max_delay;
    bool has_delay;
    u_real ***states_out, **BOLD, **mean_bold, **ssd_bold, **fc_trils, **windows_mean_bold, **windows_ssd_bold,
        **windows_fc_trils, **windows_mean_fc, **windows_ssd_fc, **fcd_trils, *noise,
        *d_SC, **d_global_params, **d_regional_params;
    int **global_out_int;
    bool **global_out_bool;
    int *pairs_i, *pairs_j, *window_starts, *window_ends, *window_pairs_i, *window_pairs_j;
    int last_time_steps = 0; // to avoid recalculating noise in subsequent calls of the function with force_reinit
    int last_nodes = 0;
    int last_rand_seed = 0;
    #ifdef NOISE_SEGMENT
    int *shuffled_nodes, *shuffled_ts;
    // set a default length of noise (msec)
    // (+1 to avoid having an additional repeat for a single time point
    // when time_steps can be divided by 30(000), as the actual duration of
    // simulation (in msec) is always user request time steps + 1)
    int noise_time_steps = 30001;
    int noise_repeats; // number of noise segment repeats for current simulations
    #endif
}
#ifdef USE_FLOATS
double **d_fc_trils, **d_fcd_trils;
#else
// use d_fc_trils and d_fcd_trils as aliases for fc_trils and fcd_trils
// which will later be used for GOF calculations
#define d_fc_trils fc_trils
#define d_fcd_trils fcd_trils
#endif

__device__ void calculateGlobalInput(
        u_real* tmp_globalinput, int* k_buff_idx,
        int* nodes, int* sim_idx, int* j, int *k, 
        int* buff_idx, int* max_delay, u_real* _SC, 
        int** delay, bool* has_delay, 
        u_real** conn_state_var_hist, u_real* conn_state_var_1
        ) {
    // calculates global input from other nodes `k` to current node `j`
    // In this and other device kernels I use a call by pointer design
    // which I hope in most cases will use up less memory + makes all
    // variables of bnm updatable in __device__ kernels
    // (though we don't need to update all)
    // TODO: consider passing some of the variables by value
    // (e.g. int and bool variables that are not updated)
    *tmp_globalinput = 0;
    if (*has_delay) {
        for (*k=0; *k<*nodes; (*k)++) {
            // calculate correct index of the other region in the buffer based on j-k delay
            *k_buff_idx = (*buff_idx + delay[*sim_idx][(*j)*(*nodes)+(*k)]) % *max_delay;
            *tmp_globalinput += _SC[*k] * conn_state_var_hist[*sim_idx][(*k_buff_idx)*(*nodes)+(*k)];
        }
    } else {
        for (*k=0; *k<*nodes; (*k)++) {
            *tmp_globalinput += _SC[*k] * conn_state_var_1[*k];
        }            
    }
}

template<typename Model>
__global__ void bnm(
    Model* model,
    u_real **BOLD,
    u_real ***states_out, 
    int **global_out_int,
    bool **global_out_bool,
    int n_vols_remove,
    u_real *SC, u_real **global_params, u_real **regional_params,
    u_real **conn_state_var_hist, int **delay, int max_delay,
    int N_SIMS, int nodes, int BOLD_TR, int time_steps, 
    u_real *noise, 
    bool extended_output,
#ifdef NOISE_SEGMENT
    int *shuffled_nodes, int *shuffled_ts,
    int noise_time_steps, int noise_repeats,
#endif
    int corr_len
    ) {
    // convert block to a cooperative group
    // get simulation and node indices
    int sim_idx = blockIdx.x;
    if (sim_idx >= N_SIMS) return;
    int j = threadIdx.x;
    if (j >= nodes) return;

    // set up noise shuffling if indicated
    #ifdef NOISE_SEGMENT
    /* 
    How noise shuffling works?
    At each time point we will have `ts_bold` which is the real time (in msec) 
    from the start of simulation, `ts_noise` which is the real time passed within 
    each repeat of the noise segment (`curr_noise_repeat`), `sh_ts_noise` which is 
    the shuffled timepoint (column of the noise segment) that will be used for getting 
    the noise of nodes * 10 int_i * 2 neurons for the current msec. 
    Similarly, in each thread we have `j` which is mapped to a `sh_j` which will 
    vary in each repeat.
    */
    // get position of the node
    // in shuffled nodes for the first
    // repeat of noise segment
    int sh_j = shuffled_nodes[j];
    int curr_noise_repeat = 0;
    int ts_noise = 0;
    // also get the shuffled ts of the
    // first time point
    int sh_ts_noise = shuffled_ts[ts_noise];
    #endif

    // determine the parameters of current simulation and node
    // use __shared__ for parameters that are shared
    // between regions in the same simulation, e.g. G, but
    // not for those that (may) vary, e.g. w_IE, w_EE and w_IE
    __shared__ u_real _global_params[Model::n_global_params];
    u_real _regional_params[Model::n_regional_params];
    int ii; // general-purpose index for parameters and varaiables
    for (ii=0; ii<Model::n_global_params; ii++) {
        _global_params[ii] = global_params[ii][sim_idx];
    }
    for (ii=0; ii<Model::n_regional_params; ii++) {
        _regional_params[ii] = regional_params[ii][sim_idx*nodes+j];
    }

    // initialize extended output sums
    if (extended_output & (!d_conf.extended_output_ts)) {
        for (ii=0; ii<Model::n_state_vars; ii++) {
            states_out[ii][sim_idx][j] = 0;
        }
    }

    // declare state variables, intermediate variables
    // and additional ints and bools
    u_real _state_vars[Model::n_state_vars];
    u_real* _intermediate_vars = (Model::n_intermediate_vars > 0) ? new u_real[Model::n_intermediate_vars] : NULL;
    int* _ext_int = (Model::n_ext_int > 0) ? new int[Model::n_ext_int] : NULL;
    bool* _ext_bool = (Model::n_ext_bool > 0) ? new bool[Model::n_ext_bool] : NULL;
    // __shared__ bool* _ext_bool_shared;
    // if (Model::n_ext_bool > 0) {
    //     _ext_bool_shared = new bool[Model::n_ext_bool];
    // }
    model->init(_state_vars, _intermediate_vars, _ext_int, _ext_bool, model);


    // Ballon-Windkessel model variables
    u_real bw_x, bw_f, bw_nu, bw_q, tmp_f;
    bw_x = 0.0;
    bw_f = 1.0;
    bw_nu = 1.0;
    bw_q = 1.0;

    // delay setup
    bool has_delay = (max_delay > 0);
    // if there is delay use a circular buffer (conn_state_var_hist)
    // and keep track of current buffer index (will be the same
    // in all nodes at each time point). Start from the end and
    // go backwards. 
    // Note that conn_state_var_hist is pseudo-2d
    int buff_idx, k_buff_idx;
    if (has_delay) {
        // initialize conn_state_var_hist in every time point at initial value
        for (buff_idx=0; buff_idx<max_delay; buff_idx++) {
            conn_state_var_hist[sim_idx][buff_idx*nodes+j] = _state_vars[Model::conn_state_var_idx];
        }
        buff_idx = max_delay-1;
    }

    // stored history of conn_state_var on extern shared memory
    // the memory is allocated dynamically based on the number of nodes
    // (see https://developer.nvidia.com/blog/using-shared-memory-cuda-cc/)
    extern __shared__ u_real conn_state_var_1[];
    conn_state_var_1[j] = _state_vars[Model::conn_state_var_idx];

//     #ifndef MANY_NODES
    // copy SC to thread memory to make it faster
    // (especially with lower N of simulations)
    u_real _SC[MAX_NODES];
    for (int k=0; k<nodes; k++) {
        _SC[k] = SC[j*nodes+k];
    }
//     #else
//     // use global memory for SC
//     u_real *_SC = &(SC[j*nodes]);
//     #endif

    // this will determine if the simulation should be restarted
    // (e.g. if FIC adjustment fails in rWW)
    __shared__ bool restart;
    restart = false;

    // integration loop
    u_real tmp_globalinput;
    int ts_bold, int_i, k;
    long noise_idx = 0;
    int BOLD_len_i = 0;
    ts_bold = 0;
    while (ts_bold <= time_steps) {
        if (d_conf.sync_msec) {
            // calculate global input every 1 ms
            // (will be fixed through 10 steps of the millisecond)
            // note that a sync call is needed before using
            // and updating S_i_1_E, otherwise the current
            // thread might access S_E of other nodes at wrong
            // times (t or t-2 instead of t-1)
            __syncthreads();
            calculateGlobalInput(
                &tmp_globalinput, &k_buff_idx,
                &nodes, &sim_idx, 
                &j, &k, &buff_idx, &max_delay, 
                _SC, delay, &has_delay,
                conn_state_var_hist, conn_state_var_1
            );
        }
        for (int_i = 0; int_i < 10; int_i++) {
            if (!d_conf.sync_msec) {
                // calculate global input every 0.1 ms
                __syncthreads();
                calculateGlobalInput(
                    &tmp_globalinput, &k_buff_idx,
                    &nodes, &sim_idx, 
                    &j, &k, &buff_idx, &max_delay, 
                    _SC, delay, &has_delay,
                    conn_state_var_hist, conn_state_var_1
                );
            }
            // equations
            #ifdef NOISE_SEGMENT
            noise_idx = (((sh_ts_noise * 10 + int_i) * nodes * Model::n_noise) + (sh_j * Model::n_noise));
            #else
            noise_idx = (((ts_bold * 10 + int_i) * nodes * Model::n_noise) + (j * Model::n_noise));
            #endif
            model->step(
                _state_vars, _intermediate_vars, 
                _global_params, _regional_params,
                &tmp_globalinput,
                noise, &noise_idx,
                model
            );
            if (!d_conf.sync_msec) {
                if (has_delay) {
                    // go 0.1 ms backward in the buffer and save most recent S_i_E
                    // wait for all regions to have correct (same) buff_idx
                    // and updated conn_state_var_hist
                    __syncthreads();
                    buff_idx = (buff_idx + max_delay - 1) % max_delay;
                    conn_state_var_hist[sim_idx][buff_idx*nodes+j] = _state_vars[Model::conn_state_var_idx];
                } else  {
                    // wait for other regions
                    // see note above on why this is needed before updating S_i_1_E
                    __syncthreads();
                    conn_state_var_1[j] = _state_vars[Model::conn_state_var_idx];
                }
            }
        }

        if (d_conf.sync_msec) {
            if (has_delay) {
                // go 1 ms backward in the buffer and save most recent S_i_E
                // wait for all regions to have correct (same) buff_idx
                // and updated conn_state_var_hist
                __syncthreads();
                buff_idx = (buff_idx + max_delay - 1) % max_delay;
                conn_state_var_hist[sim_idx][buff_idx*nodes+j] = _state_vars[Model::conn_state_var_idx];
            } else {
                // wait for other regions
                // see note above on why this is needed before updating S_i_1_E
                __syncthreads();
                conn_state_var_1[j] = _state_vars[Model::conn_state_var_idx];
            }
        }

        // Balloon-Windkessel model equations here since its
        // dt is 1 msec
        bw_step(&bw_x, &bw_f, &bw_nu, 
            &bw_q, &tmp_f,
            &(_state_vars[Model::bold_state_var_idx]));

        // Save BOLD and extended output to managed memory
        // every TR
        // TODO: make it possible to have different sampling rate
        // for BOLD and extended output
        // TODO: add option to return time series of extended output
        if (ts_bold % BOLD_TR == 0) {
            // calcualte and save BOLD
            BOLD[sim_idx][BOLD_len_i*nodes+j] = d_bwc.V_0_k1 * (1 - bw_q) + d_bwc.V_0_k2 * (1 - bw_q/bw_nu) + d_bwc.V_0_k3 * (1 - bw_nu);
            // save time series of extended output if indicated
            if (extended_output & d_conf.extended_output_ts) {
                for (ii=0; ii<Model::n_state_vars; ii++) {
                    states_out[ii][sim_idx][BOLD_len_i*nodes+j] = _state_vars[ii];
                }
            }
            // update sum (later mean) of extended
            // output only after n_vols_remove
            if ((BOLD_len_i>=n_vols_remove) & extended_output & (!d_conf.extended_output_ts)) {
                for (ii=0; ii<Model::n_state_vars; ii++) {
                    states_out[ii][sim_idx][j] += _state_vars[ii];
                }
            }
            BOLD_len_i++;
        }
        ts_bold++;

        #ifdef NOISE_SEGMENT
        // update noise segment time
        ts_noise++;
        // reset noise segment time 
        // and shuffle nodes if the segment
        // has reached to the end
        if (ts_noise % noise_time_steps == 0) {
            // at the last time point don't do this
            // to avoid going over the extent of shuffled_nodes
            if (ts_bold <= time_steps) {
                curr_noise_repeat++;
                sh_j = shuffled_nodes[curr_noise_repeat*nodes+j];
                ts_noise = 0;
            }
        }
        // get the shuffled timepoint corresponding to ts_noise 
        sh_ts_noise = shuffled_ts[ts_noise+curr_noise_repeat*noise_time_steps];
        #endif

        if (Model::has_post_bw_step) {
            model->post_bw_step(
                _state_vars, _intermediate_vars,
                _ext_int, _ext_bool, &restart,
                _global_params, _regional_params,
                &ts_bold, model
            );
        }

        // if restart is indicated (e.g. FIC failed in rWW)
        // reset the simulation and start from the beginning
        if (restart) {
            // model-specific restart
            model->restart(_state_vars, _intermediate_vars, _ext_int, _ext_bool, model);
            // reset indices
            BOLD_len_i = 0;
            ts_bold = 0;
            // reset Balloon-Windkessel model variables
            bw_x = 0.0;
            bw_f = 1.0;
            bw_nu = 1.0;
            bw_q = 1.0;
            // reset delay buffer index
            if (has_delay) {
                // initialize conn_state_var_hist in every time point at initial value
                for (buff_idx=0; buff_idx<max_delay; buff_idx++) {
                    conn_state_var_hist[sim_idx][buff_idx*nodes+j] = _state_vars[Model::conn_state_var_idx];
                }
                buff_idx = max_delay-1;
            }
            // reset conn_state_var_1
            conn_state_var_1[j] = _state_vars[Model::conn_state_var_idx];
            #ifdef NOISE_SEGMENT
            // reset the node shuffling
            sh_j = shuffled_nodes[j];
            curr_noise_repeat = 0;
            ts_noise = 0;
            sh_ts_noise = shuffled_ts[ts_noise];
            #endif
            restart = false; // restart is done
        }
    }
    if (Model::has_post_integration) {
        model->post_integration(
            BOLD, states_out, 
            global_out_int, global_out_bool,
            _state_vars, _intermediate_vars, 
            _ext_int, _ext_bool, 
            global_params, regional_params,
            _global_params, _regional_params,
            sim_idx, nodes, j,
            model
        );
    }
    if (extended_output & (!d_conf.extended_output_ts)) {
        // take average
        int extended_output_time_points = BOLD_len_i - n_vols_remove;
        for (ii=0; ii<Model::n_state_vars; ii++) {
            states_out[ii][sim_idx][j] /= extended_output_time_points;
        }
    }
}

template<typename Model>
void run_simulations_gpu(
    double * BOLD_out, double * fc_trils_out, double * fcd_trils_out,
    u_real ** global_params, u_real ** regional_params, u_real * v_list,
    u_real * SC, gsl_matrix * SC_gsl, u_real * SC_dist, bool do_delay,
    int nodes, int time_steps, int BOLD_TR, int window_size,
    int N_SIMS, bool extended_output, Model* model,
    ModelConfigs conf
)
// TODO: clean the order of args
{
    using namespace bnm_gpu;

    Model* d_model;
    CUDA_CHECK_RETURN(cudaMallocManaged(&d_model, sizeof(Model)));
    // new (d_model) Model();
    // copy model to managed memory
    CUDA_CHECK_RETURN(cudaMemcpy(d_model, model, sizeof(Model), cudaMemcpyHostToDevice));


    // copy SC to managed memory
    CUDA_CHECK_RETURN(cudaMemcpy(d_SC, SC, nodes*nodes * sizeof(u_real), cudaMemcpyHostToDevice));

    // copy parameters to managed memory
    for (int i=0; i<Model::n_global_params; i++) {
        CUDA_CHECK_RETURN(cudaMemcpy(d_global_params[i], global_params[i], N_SIMS * sizeof(u_real), cudaMemcpyHostToDevice));
    }
    for (int i=0; i<Model::n_regional_params; i++) {
        CUDA_CHECK_RETURN(cudaMemcpy(d_regional_params[i], regional_params[i], N_SIMS*nodes * sizeof(u_real), cudaMemcpyHostToDevice));
    }

    // if indicated, calculate delay matrix of each simulation and allocate
    // memory to conn_state_var_hist according to the max_delay among the current simulations
    // Note: unlike many other variables delay and conn_state_var_hist are not global variables
    // and are not initialized in init_gpu, in order to allow variable ranges of velocities
    // in each run_simulations_gpu call within a session
    u_real **conn_state_var_hist; 
    CUDA_CHECK_RETURN(cudaMallocManaged((void**)&conn_state_var_hist, sizeof(u_real*) * N_SIMS)); 
    int **delay;
    max_delay = 0; // msec or 0.1 msec depending on conf.sync_msec; this is a global variable that will be used in the kernel
    float min_velocity = 1e10; // only used for printing info
    float max_length;
    float curr_length, curr_delay, curr_velocity;
    if (do_delay) {
    // note that do_delay is user asking for delay to be considered, has_delay indicates
    // if user has asked for delay AND there would be any delay between nodes given
    // velocity and distance matrix
    // TODO: make it less complicated
        CUDA_CHECK_RETURN(cudaMallocManaged((void**)&delay, sizeof(int*) * N_SIMS));
        for (int sim_idx=0; sim_idx < N_SIMS; sim_idx++) {
            CUDA_CHECK_RETURN(cudaMallocManaged((void**)&delay[sim_idx], sizeof(int) * nodes * nodes));
            curr_velocity = v_list[sim_idx];
            if (!conf.sync_msec) {
                curr_velocity /= 10;
            }
            if (curr_velocity < min_velocity) {
                min_velocity = curr_velocity;
            }
            for (int i = 0; i < nodes; i++) {
                for (int j = 0; j < nodes; j++) {
                    curr_length = SC_dist[i*nodes+j];
                    if (i > j) {
                        curr_delay = (int)(((curr_length/curr_velocity))+0.5);
                        delay[sim_idx][i*nodes + j] = curr_delay;
                        delay[sim_idx][j*nodes + i] = curr_delay;
                        if (curr_delay > max_delay) {
                            max_delay = curr_delay;
                            max_length = curr_length;
                        }
                    }
                }
            }
        }
    }
    has_delay = (max_delay > 0); // this is a global variable that will be used in the kernel
    if (has_delay) {
        std::string velocity_unit = "m/s";
        std::string delay_unit = "msec";
        if (!conf.sync_msec) {
            velocity_unit = "m/0.1s";
            delay_unit = "0.1msec";
        }
        printf("Max distance %f (mm) with a minimum velocity of %f (%s) => Max delay: %d (%s)\n", max_length, min_velocity, velocity_unit.c_str(), max_delay, delay_unit.c_str());

        // allocate memory to conn_state_var_hist for N_SIMS * (nodes * max_delay)
        // TODO: make it possible to have variable max_delay per each simulation
        for (int sim_idx=0; sim_idx < N_SIMS; sim_idx++) {
            CUDA_CHECK_RETURN(cudaMallocManaged((void**)&conn_state_var_hist[sim_idx], sizeof(u_real) * nodes * max_delay));
        }
    } 
    #ifdef MANY_NODES
    // in case of large number of nodes also allocate memory to
    // conn_state_var_hist[sim_idx] for a length of nodes. This array
    // will contain the immediate history of S_i_E (at t-1)
    else {
        for (int sim_idx=0; sim_idx < N_SIMS; sim_idx++) {
            CUDA_CHECK_RETURN(cudaMallocManaged((void**)&conn_state_var_hist[sim_idx], sizeof(u_real) * nodes));
        }
    }
    #endif

    if (strcmp(Model::name, "rWW")==0 && d_model->conf.do_fic) {
        gsl_vector * curr_w_IE = gsl_vector_alloc(nodes);
        double *curr_w_EE = (double *)malloc(nodes * sizeof(double));
        double *curr_w_EI = (double *)malloc(nodes * sizeof(double));
        for (int sim_idx=0; sim_idx<N_SIMS; sim_idx++) {
            // make a copy of regional wEE and wEI
            for (int j=0; j<nodes; j++) {
                curr_w_EE[j] = (double)(d_regional_params[0][sim_idx*nodes+j]);
                curr_w_EI[j] = (double)(d_regional_params[1][sim_idx*nodes+j]);
            }
            // do FIC for the current particle
            global_out_bool[0][sim_idx] = false;
            // bool* _fic_unstable;
            analytical_fic_het(
                SC_gsl, d_global_params[0][sim_idx], curr_w_EE, curr_w_EI,
                // curr_w_IE, _fic_unstable);
                curr_w_IE, global_out_bool[0]+sim_idx);
            if (global_out_bool[0][sim_idx]) {
            // if (*_fic_unstable) {
                printf("In simulation #%d FIC solution is unstable. Setting wIE to 1 in all nodes\n", sim_idx);
                for (int j=0; j<nodes; j++) {
                    d_regional_params[2][sim_idx*nodes+j] = 1.0;
                }
            } else {
                // copy to w_IE_fic which will be passed on to the device
                for (int j=0; j<nodes; j++) {
                    d_regional_params[2][sim_idx*nodes+j] = (u_real)gsl_vector_get(curr_w_IE, j);
                }
            }
        }
    }

    // run simulations
    dim3 numBlocks(N_SIMS);
    dim3 threadsPerBlock(nodes);
    // (third argument is extern shared memory size for S_i_1_E)
    // provide NULL for extended output variables and FIC variables
    bool _extended_output = extended_output;
    if (strcmp(Model::name, "rWW")==0) {
        // for rWW extended output is needed if requested by user or FIC is done
        _extended_output = extended_output | model->conf.do_fic;
    }
    #ifndef MANY_NODES
    size_t shared_mem_extern = nodes*sizeof(u_real);
    #else
    size_t shared_mem_extern = 0;
    #endif 
    bnm<Model><<<numBlocks,threadsPerBlock,shared_mem_extern>>>(
        d_model,
        BOLD, states_out, 
        global_out_int,
        global_out_bool,
        n_vols_remove,
        d_SC, d_global_params, d_regional_params,
        conn_state_var_hist, delay, max_delay,
        N_SIMS, nodes, BOLD_TR, time_steps, 
        noise, _extended_output,
    #ifdef NOISE_SEGMENT
        shuffled_nodes, shuffled_ts, noise_time_steps, noise_repeats,
    #endif
        corr_len);
    CUDA_CHECK_LAST_ERROR();
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());

    // calculate mean and sd bold for FC calculation
    bold_stats<<<numBlocks, threadsPerBlock>>>(
        mean_bold, ssd_bold,
        BOLD, N_SIMS, nodes,
        output_ts, corr_len, n_vols_remove);
    CUDA_CHECK_LAST_ERROR();
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());
    // calculate window mean and sd bold for FCD calculations
    numBlocks.y = n_windows;
    window_bold_stats<<<numBlocks,threadsPerBlock>>>(
        BOLD, N_SIMS, nodes, 
        n_windows, window_size+1, window_starts, window_ends,
        windows_mean_bold, windows_ssd_bold);
    CUDA_CHECK_LAST_ERROR();
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());
    // calculate FC and window FCs
    int maxThreadsPerBlock = prop.maxThreadsPerBlock;
    numBlocks.y = ceil((float)n_pairs / (float)maxThreadsPerBlock);
    numBlocks.z = n_windows + 1; // +1 for total FC
    if (prop.maxThreadsPerBlock!=prop.maxThreadsDim[0]) {
        printf("Error: Code not implemented for GPUs in which maxThreadsPerBlock!=maxThreadsDim[0]\n");
        exit(1);
    }
    threadsPerBlock.x = maxThreadsPerBlock;
    threadsPerBlock.y = 1;
    threadsPerBlock.z = 1;
    fc<<<numBlocks, threadsPerBlock>>>(
        fc_trils, windows_fc_trils, BOLD, N_SIMS, nodes, n_pairs, pairs_i, pairs_j,
        output_ts, n_vols_remove, corr_len, mean_bold, ssd_bold,
        n_windows, window_size+1, windows_mean_bold, windows_ssd_bold,
        window_starts, window_ends,
        maxThreadsPerBlock
    );
    CUDA_CHECK_LAST_ERROR();
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());
    // calculate window mean and sd fc_tril for FCD calculations
    numBlocks.y = 1;
    numBlocks.z = 1;
    threadsPerBlock.x = n_windows;
    if (n_windows >= prop.maxThreadsPerBlock) {
        printf("Error: Mean/ssd FC tril of %d windows cannot be calculated on this device\n", n_windows);
        exit(1);
    }
    window_fc_stats<<<numBlocks,threadsPerBlock>>>(
        windows_mean_fc, windows_ssd_fc,
        NULL, NULL, NULL, NULL, // no need for L and R stats in CMAES
        windows_fc_trils, N_SIMS, n_windows, n_pairs,
        false, 0);
    CUDA_CHECK_LAST_ERROR();
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());
    // calculate FCD
    numBlocks.y = ceil((float)n_window_pairs / (float)maxThreadsPerBlock);
    numBlocks.z = 1;
    if (prop.maxThreadsPerBlock!=prop.maxThreadsDim[0]) {
        printf("Code not implemented for GPUs in which maxThreadsPerBlock!=maxThreadsDim[0]\n");
        exit(1);
    }
    threadsPerBlock.x = maxThreadsPerBlock;
    threadsPerBlock.y = 1;
    threadsPerBlock.z = 1;
    fcd<<<numBlocks, threadsPerBlock>>>(
        fcd_trils, NULL, NULL, // no need for L and R fcd in CMAES
        windows_fc_trils, 
        windows_mean_fc, windows_ssd_fc,
        NULL, NULL, NULL, NULL,
        N_SIMS, n_pairs, n_windows, n_window_pairs, 
        window_pairs_i, window_pairs_j, maxThreadsPerBlock,
        false, 0);
    CUDA_CHECK_LAST_ERROR();
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());

    #ifdef USE_FLOATS
    // Convert FC and FCD to doubles for GOF calculation
    numBlocks.y = n_pairs;
    numBlocks.z = 1;
    threadsPerBlock.x = 1;
    float2double<<<numBlocks, threadsPerBlock>>>(d_fc_trils, fc_trils, N_SIMS, n_pairs);
    CUDA_CHECK_LAST_ERROR();
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());
    numBlocks.y = n_window_pairs;
    float2double<<<numBlocks, threadsPerBlock>>>(d_fcd_trils, fcd_trils, N_SIMS, n_window_pairs);
    CUDA_CHECK_LAST_ERROR();
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());
    #endif

    // copy the output from managed memory to _out arrays (which can be numpy arrays)
    size_t bold_size = nodes * output_ts;
    size_t ext_out_size = nodes;
    if (conf.extended_output_ts) {
        ext_out_size *= output_ts;
    }
    // TODO: pass the managed arrays data directly
    // to the python arrays without copying
    for (int sim_idx=0; sim_idx<N_SIMS; sim_idx++) {
        memcpy(BOLD_out, BOLD[sim_idx], sizeof(u_real) * bold_size);
        BOLD_out+=bold_size;
        memcpy(fc_trils_out, fc_trils[sim_idx], sizeof(u_real) * n_pairs);
        fc_trils_out+=n_pairs;
        memcpy(fcd_trils_out, fcd_trils[sim_idx], sizeof(u_real) * n_window_pairs);
        fcd_trils_out+=n_window_pairs;
    }
    if (strcmp(Model::name, "rWW")==0 && model->conf.do_fic) {
        // copy wIE resulted from FIC to regional_params
        memcpy(regional_params[2], d_regional_params[2], N_SIMS*nodes*sizeof(u_real));
    }

    // free delay and conn_state_var_hist memories if allocated
    // Note: no need to clear memory of the other variables
    // as we'll want to reuse them in the next calls to run_simulations_gpu
    // within current session
    if (do_delay) {
        for (int sim_idx=0; sim_idx < N_SIMS; sim_idx++) {
            CUDA_CHECK_RETURN(cudaFree(delay[sim_idx]));
        }
        CUDA_CHECK_RETURN(cudaFree(delay));
    }
    if (has_delay) {
        for (int sim_idx=0; sim_idx < N_SIMS; sim_idx++) {
            CUDA_CHECK_RETURN(cudaFree(conn_state_var_hist[sim_idx]));
        }
    }
    // #ifdef MANY_NODES
    // else {
    //     for (int sim_idx=0; sim_idx < N_SIMS; sim_idx++) {
    //         CUDA_CHECK_RETURN(cudaFree(conn_state_var_hist[sim_idx]));
    //     }
    // } 
    // #endif   
    CUDA_CHECK_RETURN(cudaFree(conn_state_var_hist));

}

template<typename Model>
void init_gpu(
        int *output_ts_p, int *n_pairs_p, int *n_window_pairs_p,
        int N_SIMS, int nodes, bool extended_output, int rand_seed,
        int BOLD_TR, int time_steps, int window_size, int window_step,
        Model *model, BWConstants bwc, ModelConfigs conf, bool verbose
        )
    {
    using namespace bnm_gpu;
    // check CUDA device avaliability and properties
    prop = get_device_prop(verbose);

    // copy constants and configs from CPU
    CUDA_CHECK_RETURN(cudaMemcpyToSymbol(d_bwc, &bwc, sizeof(BWConstants)));
    CUDA_CHECK_RETURN(cudaMemcpyToSymbol(d_conf, &conf, sizeof(ModelConfigs)));
    if (strcmp(Model::name, "rWW")==0) {
        CUDA_CHECK_RETURN(cudaMemcpyToSymbol(d_rWWc, &Model::mc, sizeof(typename Model::Constants)));
    }

    // allocate device memory for SC
    CUDA_CHECK_RETURN(cudaMalloc(&d_SC, sizeof(u_real) * nodes*nodes));

    // allocate device memory for simulation parameters
    // size of global_params is (n_global_params, N_SIMS)
    // size of regional_params is (n_regional_params, N_SIMS * nodes)
    if (Model::n_global_params > 0) {
        CUDA_CHECK_RETURN(cudaMallocManaged((void**)&d_global_params, sizeof(u_real*) * Model::n_global_params));
        for (int param_idx=0; param_idx<Model::n_global_params; param_idx++) {
            CUDA_CHECK_RETURN(cudaMallocManaged((void**)&d_global_params[param_idx], sizeof(u_real) * N_SIMS));
        }
    }
    if (Model::n_regional_params > 0) {
        CUDA_CHECK_RETURN(cudaMallocManaged((void**)&d_regional_params, sizeof(u_real*) * Model::n_regional_params));
        for (int param_idx=0; param_idx<Model::n_regional_params; param_idx++) {
            CUDA_CHECK_RETURN(cudaMallocManaged((void**)&d_regional_params[param_idx], sizeof(u_real) * N_SIMS * nodes));
        }
    }

    // allocated memory for BOLD time-series of all simulations
    // BOLD will be a 2D array of size N_SIMS x (nodes x output_ts)
    u_real TR        = (u_real)BOLD_TR / 1000; // (s) TR of fMRI data
    // output_ts = (time_steps / (TR / mc.bw_dt))+1; // Length of BOLD time-series written to HDD
    output_ts = (time_steps / (TR / 0.001))+1; // Length of BOLD time-series written to HDD
    size_t bold_size = nodes * output_ts;
    CUDA_CHECK_RETURN(cudaMallocManaged((void**)&BOLD, sizeof(u_real*) * N_SIMS));


    // set up global int and bool outputs
    if (Model::n_global_out_int > 0) {
        CUDA_CHECK_RETURN(cudaMallocManaged((void**)&global_out_int, sizeof(int*) * Model::n_global_out_int));
        for (int i=0; i<Model::n_global_out_int; i++) {
            CUDA_CHECK_RETURN(cudaMallocManaged((void**)&global_out_int[i], sizeof(int) * N_SIMS));
        }
    }
    if (Model::n_global_out_bool > 0) {
        CUDA_CHECK_RETURN(cudaMallocManaged((void**)&global_out_bool, sizeof(bool*) * Model::n_global_out_bool));
        for (int i=0; i<Model::n_global_out_bool; i++) {
            CUDA_CHECK_RETURN(cudaMallocManaged((void**)&global_out_bool[i], sizeof(bool) * N_SIMS));
        }
    }

    // allocate memory for extended output
    bool _extended_output = extended_output;
    if (strcmp(Model::name, "rWW")==0) {
        // for rWW extended output is needed if requested by user or FIC is done
        _extended_output = extended_output | model->conf.do_fic;
    }
    size_t ext_out_size = nodes;
    if (conf.extended_output_ts) {
        ext_out_size *= output_ts;
    }
    if (_extended_output) {
        CUDA_CHECK_RETURN(cudaMallocManaged((void**)&states_out, sizeof(u_real**) * Model::n_state_vars));
        for (int var_idx=0; var_idx<Model::n_state_vars; var_idx++) {
            CUDA_CHECK_RETURN(cudaMallocManaged((void**)&states_out[var_idx], sizeof(u_real*) * N_SIMS));
            for (int sim_idx=0; sim_idx<N_SIMS; sim_idx++) {
                CUDA_CHECK_RETURN(cudaMallocManaged((void**)&states_out[var_idx][sim_idx], sizeof(u_real) * ext_out_size));
            }
        }
    }

    // specify n_vols_remove (for extended output and FC calculations)
    n_vols_remove = conf.bold_remove_s * 1000 / BOLD_TR; // 30 seconds

    // preparing FC calculations
    corr_len = output_ts - n_vols_remove;
    if (corr_len < 2) {
        printf("Number of volumes (after removing initial volumes) is too low for FC calculations\n");
        exit(1);
    }
    CUDA_CHECK_RETURN(cudaMallocManaged((void**)&mean_bold, sizeof(u_real*) * N_SIMS));
    CUDA_CHECK_RETURN(cudaMallocManaged((void**)&ssd_bold, sizeof(u_real*) * N_SIMS));
    n_pairs = ((nodes) * (nodes - 1)) / 2;
    int rh_idx;
    if (conf.exc_interhemispheric) {
        if ((nodes % 2) != 0) {
            printf("Error: exc_interhemispheric is set but number of nodes is not even\n");
            exit(1);
        }
        rh_idx = nodes / 2; // assumes symmetric number of parcels and L->R order
        n_pairs -= pow(rh_idx, 2); // exc the middle square
    }
    // if (n_pairs!=emp_FC_tril->size) { // TODO: do this check on Python side
    //     printf("Empirical and simulated FC size do not match\n");
    //     exit(1);
    // }
    // create a mapping between pair_idx and i and j
    int curr_idx = 0;
    CUDA_CHECK_RETURN(cudaMallocManaged(&pairs_i, sizeof(int) * n_pairs));
    CUDA_CHECK_RETURN(cudaMallocManaged(&pairs_j, sizeof(int) * n_pairs));
    for (int i=0; i < nodes; i++) {
        for (int j=0; j < nodes; j++) {
            if (i > j) {
                if (conf.exc_interhemispheric) {
                    // skip if each node belongs to a different hemisphere
                    if ((i < rh_idx) ^ (j < rh_idx)) {
                        continue;
                    }
                }
                pairs_i[curr_idx] = i;
                pairs_j[curr_idx] = j;
                curr_idx++;
            }
        }
    }
    // allocate memory for fc trils
    CUDA_CHECK_RETURN(cudaMallocManaged((void**)&fc_trils, sizeof(u_real*) * N_SIMS));

    // FCD preparation
    // calculate number of windows and window start/end indices
    int *_window_starts, *_window_ends; // are cpu integer arrays
    int _n_windows = get_dfc_windows(
        &_window_starts, &_window_ends, 
        corr_len, output_ts, n_vols_remove,
        window_step, window_size);
    n_windows = _n_windows;
    if (n_windows == 0) {
        printf("Error: Number of windows is 0\n");
        exit(1);
    }
    CUDA_CHECK_RETURN(cudaMallocManaged(&window_starts, sizeof(int) * n_windows));
    CUDA_CHECK_RETURN(cudaMallocManaged(&window_ends, sizeof(int) * n_windows));
    for (int i=0; i<n_windows; i++) {
        window_starts[i] = _window_starts[i];
        window_ends[i] = _window_ends[i];
    }
    // allocate memory for mean and ssd BOLD of each window
    // (n_sims x n_windows x nodes)
    CUDA_CHECK_RETURN(cudaMallocManaged((void**)&windows_mean_bold, sizeof(u_real*) * N_SIMS));
    CUDA_CHECK_RETURN(cudaMallocManaged((void**)&windows_ssd_bold, sizeof(u_real*) * N_SIMS));
    CUDA_CHECK_RETURN(cudaMallocManaged((void**)&windows_fc_trils, sizeof(u_real*) * N_SIMS));
    // allocate memory for mean and ssd fc_tril of each window
    // (n_sims x n_windows)
    CUDA_CHECK_RETURN(cudaMallocManaged((void**)&windows_mean_fc, sizeof(u_real*) * N_SIMS));
    CUDA_CHECK_RETURN(cudaMallocManaged((void**)&windows_ssd_fc, sizeof(u_real*) * N_SIMS));
    // create a mapping between window_pair_idx and i and j
    n_window_pairs = (n_windows * (n_windows-1)) / 2;
    curr_idx = 0;
    CUDA_CHECK_RETURN(cudaMallocManaged(&window_pairs_i, sizeof(int) * n_window_pairs));
    CUDA_CHECK_RETURN(cudaMallocManaged(&window_pairs_j, sizeof(int) * n_window_pairs));
    for (int i=0; i < n_windows; i++) {
        for (int j=0; j < n_windows; j++) {
            if (i > j) {
                window_pairs_i[curr_idx] = i;
                window_pairs_j[curr_idx] = j;
                curr_idx++;
            }
        }
    }
    // allocate memory for fcd trils
    CUDA_CHECK_RETURN(cudaMallocManaged((void**)&fcd_trils, sizeof(u_real*) * N_SIMS));

    #ifdef USE_FLOATS
    // allocate memory for double versions of fc and fcd trils on CPU
    CUDA_CHECK_RETURN(cudaMallocManaged((void**)&d_fc_trils, sizeof(double*) * N_SIMS));
    CUDA_CHECK_RETURN(cudaMallocManaged((void**)&d_fcd_trils, sizeof(double*) * N_SIMS));
    #endif



    // allocate memory per each simulation
    for (int sim_idx=0; sim_idx<N_SIMS; sim_idx++) {
        // allocate a chunk of BOLD to this simulation (not sure entirely if this is the best way to do it)
        CUDA_CHECK_RETURN(cudaMallocManaged((void**)&BOLD[sim_idx], sizeof(u_real) * bold_size));
        // allocate memory for fc calculations
        CUDA_CHECK_RETURN(cudaMallocManaged((void**)&mean_bold[sim_idx], sizeof(u_real) * nodes));
        CUDA_CHECK_RETURN(cudaMallocManaged((void**)&ssd_bold[sim_idx], sizeof(u_real) * nodes));
        CUDA_CHECK_RETURN(cudaMallocManaged((void**)&fc_trils[sim_idx], sizeof(u_real) * n_pairs));
        // allocate memory for window fc and fcd calculations
        CUDA_CHECK_RETURN(cudaMallocManaged((void**)&windows_mean_bold[sim_idx], sizeof(u_real) * n_windows * nodes));
        CUDA_CHECK_RETURN(cudaMallocManaged((void**)&windows_ssd_bold[sim_idx], sizeof(u_real) * n_windows * nodes));
        CUDA_CHECK_RETURN(cudaMallocManaged((void**)&windows_fc_trils[sim_idx], sizeof(u_real) * n_windows * n_pairs));
        CUDA_CHECK_RETURN(cudaMallocManaged((void**)&windows_mean_fc[sim_idx], sizeof(u_real) * n_windows));
        CUDA_CHECK_RETURN(cudaMallocManaged((void**)&windows_ssd_fc[sim_idx], sizeof(u_real) * n_windows));
        CUDA_CHECK_RETURN(cudaMallocManaged((void**)&fcd_trils[sim_idx], sizeof(u_real) * n_window_pairs));
        #ifdef USE_FLOATS
        // allocate memory for double copies of fc and fcd
        CUDA_CHECK_RETURN(cudaMallocManaged((void**)&d_fc_trils[sim_idx], sizeof(double) * n_pairs));
        CUDA_CHECK_RETURN(cudaMallocManaged((void**)&d_fcd_trils[sim_idx], sizeof(double) * n_window_pairs));
        #endif
    }

    // pre-calculate normally-distributed noise on CPU
    // this is necessary to ensure consistency of noise given the same seed
    // doing the same thing directly on the device is more challenging
    #ifndef NOISE_SEGMENT
    // precalculate the entire noise needed; can use up a lot of memory
    // with high N of nodes and longer durations leads maxes out the memory
    int noise_size = nodes * (time_steps+1) * 10 * Model::n_noise; // +1 for inclusive last time point, 2 for E and I
    #else
    // otherwise precalculate a noise segment and arrays of shuffled
    // nodes and time points and reuse-shuffle the noise segment
    // throughout the simulation for `noise_repeats`
    int noise_size = nodes * (noise_time_steps) * 10 * Model::n_noise;
    noise_repeats = ceil((float)(time_steps+1) / (float)noise_time_steps); // +1 for inclusive last time point
    #endif
    if ((rand_seed != last_rand_seed) || (time_steps != last_time_steps) || (nodes != last_nodes)) {
        printf("Precalculating %d noise elements...\n", noise_size);
        if (last_nodes != 0) {
            // noise is being recalculated, free the previous one
            CUDA_CHECK_RETURN(cudaFree(noise));
            #ifdef NOISE_SEGMENT
            CUDA_CHECK_RETURN(cudaFree(shuffled_nodes));
            CUDA_CHECK_RETURN(cudaFree(shuffled_ts));
            #endif
        }
        last_time_steps = time_steps;
        last_nodes = nodes;
        last_rand_seed = rand_seed;
        std::mt19937 rand_gen(rand_seed);
        std::normal_distribution<float> normal_dist(0, 1);
        CUDA_CHECK_RETURN(cudaMallocManaged(&noise, sizeof(u_real) * noise_size));
        for (int i = 0; i < noise_size; i++) {
            #ifdef USE_FLOATS
            noise[i] = normal_dist(rand_gen);
            #else
            noise[i] = (double)normal_dist(rand_gen);
            #endif
        }
        #ifdef NOISE_SEGMENT
        // create shuffled nodes and ts indices for each repeat of the 
        // precalculaed noise 
        printf("noise will be repeated %d times (nodes [rows] and timepoints [columns] will be shuffled in each repeat)\n", noise_repeats);
        CUDA_CHECK_RETURN(cudaMallocManaged(&shuffled_nodes, sizeof(int) * noise_repeats * nodes));
        CUDA_CHECK_RETURN(cudaMallocManaged(&shuffled_ts, sizeof(int) * noise_repeats * noise_time_steps));
        get_shuffled_nodes_ts(&shuffled_nodes, &shuffled_ts,
            nodes, noise_time_steps, noise_repeats, &rand_gen);
        #endif
    } else {
        printf("Noise already precalculated\n");
    }
    // pass on output_ts etc. to the run_simulations_interface
    *output_ts_p = output_ts;
    *n_pairs_p = n_pairs;
    *n_window_pairs_p = n_window_pairs;

    bnm_gpu::is_initialized = true;
}