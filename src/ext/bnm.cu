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
#include <thread>
#include <chrono>

// TODO: clean up the includes
// and remove unncessary header files
// TODO: only include headers and
// combine source code via compiler
#include "cubnm/defines.h"
#include "./utils.cu"
#include "./models/bw.cu"
#include "cubnm/models/base.cuh"
#include "./fc.cu"
#include "cubnm/bnm.cuh"
#include "./models/rww.cu"
#include "./models/rwwex.cu"
// other models go here

cudaDeviceProp prop;

namespace bnm_gpu {
    u_real ***states_out, **BOLD, **mean_bold, **ssd_bold, **fc_trils, **windows_mean_bold, **windows_ssd_bold,
        **windows_fc_trils, **windows_mean_fc, **windows_ssd_fc, **fcd_trils, *noise,
        *d_SC, **d_global_params, **d_regional_params;
    int **global_out_int;
    bool **global_out_bool;
    int *pairs_i, *pairs_j, *window_starts, *window_ends, *window_pairs_i, *window_pairs_j;
    #ifdef NOISE_SEGMENT
    int *shuffled_nodes, *shuffled_ts;
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
        u_real& tmp_globalinput, int& k_buff_idx,
        const int& nodes, const int& sim_idx, const int& j, 
        int& k, int& buff_idx, u_real* _SC, 
        int** delay, const bool& has_delay, const int& max_delay,
        u_real** conn_state_var_hist, u_real* conn_state_var_1
        ) {
    // calculates global input from other nodes `k` to current node `j`
    // Note: this will not skip over self-connections
    // if they should be ignored, their SC should be set to 0
    tmp_globalinput = 0;
    if (has_delay) {
        for (k=0; k<nodes; k++) {
            // calculate correct index of the other region in the buffer based on j-k delay
            k_buff_idx = (buff_idx + delay[sim_idx][j*nodes+k]) % max_delay;
            tmp_globalinput += _SC[k] * conn_state_var_hist[sim_idx][k_buff_idx*nodes+k];
        }
    } else {
        for (k=0; k<nodes; k++) {
            tmp_globalinput += _SC[k] * conn_state_var_1[k];
        }            
    }
}

template<typename Model>
__global__ void bnm(
        Model* model, u_real **BOLD, u_real ***states_out, 
        int **global_out_int, bool **global_out_bool,
        u_real *SC, u_real **global_params, u_real **regional_params,
        u_real **conn_state_var_hist, int **delay, int max_delay,
        #ifdef NOISE_SEGMENT
        int *shuffled_nodes, int *shuffled_ts,
        #endif
        u_real *noise, uint* progress
    ) {
    // convert block to a cooperative group
    // get simulation and node indices
    int sim_idx = blockIdx.x;
    if (sim_idx >= model->N_SIMS) return;
    int j = threadIdx.x;
    if (j >= model->nodes) return;

    // copy variables used in the loop to local memory
    const int nodes = model->nodes;
    const int time_steps = model->time_steps;
    const int BOLD_TR = model->BOLD_TR;
    const bool extended_output = model->base_conf.extended_output;
    const bool extended_output_ts = model->base_conf.extended_output_ts;
    const bool sync_msec = model->base_conf.sync_msec;
    #ifdef NOISE_SEGMENT
    const int noise_time_steps = model->base_conf.noise_time_steps;
    #endif

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
    if (extended_output && (!extended_output_ts)) {
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
    model->init(_state_vars, _intermediate_vars, _ext_int, _ext_bool);


    // Ballon-Windkessel model variables
    u_real bw_x, bw_f, bw_nu, bw_q, tmp_f;
    bw_x = 0.0;
    bw_f = 1.0;
    bw_nu = 1.0;
    bw_q = 1.0;

    // delay setup
    const bool has_delay = (max_delay > 0);
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
    u_real tmp_globalinput = 0.0;
    int int_i = 0;
    int k = 0;
    long noise_idx = 0;
    int BOLD_len_i = 0;
    int ts_bold = 0;
    while (ts_bold <= time_steps) {
        if (sync_msec) {
            // calculate global input every 1 ms
            // (will be fixed through 10 steps of the millisecond)
            // note that a sync call is needed before using
            // and updating S_i_1_E, otherwise the current
            // thread might access S_E of other nodes at wrong
            // times (t or t-2 instead of t-1)
            __syncthreads();
            calculateGlobalInput(
                tmp_globalinput, k_buff_idx,
                nodes, sim_idx, j, 
                k, buff_idx, _SC, 
                delay, has_delay, max_delay,
                conn_state_var_hist, conn_state_var_1
            );
        }
        for (int_i = 0; int_i < 10; int_i++) {
            if (!sync_msec) {
                // calculate global input every 0.1 ms
                __syncthreads();
                calculateGlobalInput(
                    tmp_globalinput, k_buff_idx,
                    nodes, sim_idx, j, 
                    k, buff_idx, _SC, 
                    delay, has_delay, max_delay,
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
                tmp_globalinput,
                noise, noise_idx
            );
            if (!sync_msec) {
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

        if (sync_msec) {
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
        bw_step(bw_x, bw_f, bw_nu, 
            bw_q, tmp_f,
            _state_vars[Model::bold_state_var_idx]);

        // Save BOLD and extended output to managed memory
        // every TR
        // TODO: make it possible to have different sampling rate
        // for BOLD and extended output
        // TODO: add option to return time series of extended output
        if (ts_bold % BOLD_TR == 0) {
            // calcualte and save BOLD
            BOLD[sim_idx][BOLD_len_i*nodes+j] = d_bwc.V_0_k1 * (1 - bw_q) + d_bwc.V_0_k2 * (1 - bw_q/bw_nu) + d_bwc.V_0_k3 * (1 - bw_nu);
            // save time series of extended output if indicated
            if (extended_output && extended_output_ts) {
                for (ii=0; ii<Model::n_state_vars; ii++) {
                    states_out[ii][sim_idx][BOLD_len_i*nodes+j] = _state_vars[ii];
                }
            }
            // update sum (later mean) of extended
            // output only after n_vols_remove
            if ((BOLD_len_i>=model->n_vols_remove) && extended_output && (!extended_output_ts)) {
                for (ii=0; ii<Model::n_state_vars; ii++) {
                    states_out[ii][sim_idx][j] += _state_vars[ii];
                }
            }
            BOLD_len_i++;
            if (model->base_conf.verbose && (j==0)) {
                atomicAdd(progress, 1);
            }

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
                _ext_int, _ext_bool, restart,
                _global_params, _regional_params,
                ts_bold
            );
        }

        // if restart is indicated (e.g. FIC failed in rWW)
        // reset the simulation and start from the beginning
        if (restart) {
            // model-specific restart
            model->restart(_state_vars, _intermediate_vars, _ext_int, _ext_bool);
            // subtract progress of current simulation
            if (model->base_conf.verbose && (j==0)) {
                atomicAdd(progress, -BOLD_len_i);
            }
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
            sim_idx, nodes, j
        );
    }
    if (extended_output && (!extended_output_ts)) {
        // take average across time points after n_vols_remove
        int extended_output_time_points = BOLD_len_i - model->n_vols_remove;
        for (ii=0; ii<Model::n_state_vars; ii++) {
            states_out[ii][sim_idx][j] /= extended_output_time_points;
        }
    }
}

template<typename Model>
void _run_simulations_gpu(
    double * BOLD_out, double * fc_trils_out, double * fcd_trils_out,
    u_real ** global_params, u_real ** regional_params, u_real * v_list,
    u_real * SC, u_real * SC_dist, BaseModel* m
)
{
    using namespace bnm_gpu;
    if (m->base_conf.verbose) {
        m->print_config();
    }

    // copy model to device 
    Model* h_model = (Model*)m; // cast BaseModel to its specific subclass, TODO: see if this is really needed
    Model* d_model;
    CUDA_CHECK_RETURN(cudaMallocManaged(&d_model, sizeof(Model)));
    CUDA_CHECK_RETURN(cudaMemcpy(d_model, h_model, sizeof(Model), cudaMemcpyHostToDevice));

    // copy SC to managed memory
    CUDA_CHECK_RETURN(cudaMemcpy(d_SC, SC, m->nodes*m->nodes * sizeof(u_real), cudaMemcpyHostToDevice));

    // copy parameters to managed memory
    for (int i=0; i<Model::n_global_params; i++) {
        CUDA_CHECK_RETURN(cudaMemcpy(d_global_params[i], global_params[i], m->N_SIMS * sizeof(u_real), cudaMemcpyHostToDevice));
    }
    for (int i=0; i<Model::n_regional_params; i++) {
        CUDA_CHECK_RETURN(cudaMemcpy(d_regional_params[i], regional_params[i], m->N_SIMS*m->nodes * sizeof(u_real), cudaMemcpyHostToDevice));
    }

    // The following currently only does analytical FIC for rWW
    // but in theory can be used for any model that requires
    // parameter modifications
    // TODO: consider doing this in a separate function
    // called from Python, therefore final params are passed
    // to _run_simulations_gpu (except that they might be
    // modified during the simulation, e.g. in numerical FIC)
    m->prep_params(d_global_params, d_regional_params, v_list, SC, SC_dist, global_out_bool, global_out_int);

    // if indicated, calculate delay matrix of each simulation and allocate
    // memory to conn_state_var_hist according to the max_delay among the current simulations
    // Note: unlike many other variables delay and conn_state_var_hist are not global variables
    // and are not initialized in init_gpu, in order to allow variable ranges of velocities
    // in each run_simulations_gpu call within a session
    u_real **conn_state_var_hist; 
    CUDA_CHECK_RETURN(cudaMallocManaged((void**)&conn_state_var_hist, sizeof(u_real*) * m->N_SIMS)); 
    int **delay;
    int max_delay = 0; // msec or 0.1 msec depending on base_conf.sync_msec; this is a global variable that will be used in the kernel
    float min_velocity = 1e10; // only used for printing info
    float max_length;
    float curr_length, curr_delay, curr_velocity;
    if (m->do_delay) {
    // note that do_delay is user asking for delay to be considered, has_delay indicates
    // if user has asked for delay AND there would be any delay between nodes given
    // velocity and distance matrix
    // TODO: make it less complicated
        CUDA_CHECK_RETURN(cudaMallocManaged((void**)&delay, sizeof(int*) * m->N_SIMS));
        for (int sim_idx=0; sim_idx < m->N_SIMS; sim_idx++) {
            CUDA_CHECK_RETURN(cudaMallocManaged((void**)&delay[sim_idx], sizeof(int) * m->nodes * m->nodes));
            curr_velocity = v_list[sim_idx];
            if (!m->base_conf.sync_msec) {
                curr_velocity /= 10;
            }
            if (curr_velocity < min_velocity) {
                min_velocity = curr_velocity;
            }
            for (int i = 0; i < m->nodes; i++) {
                for (int j = 0; j < m->nodes; j++) {
                    curr_length = SC_dist[i*m->nodes+j];
                    if (i > j) {
                        curr_delay = (int)(((curr_length/curr_velocity))+0.5);
                        delay[sim_idx][i*m->nodes + j] = curr_delay;
                        delay[sim_idx][j*m->nodes + i] = curr_delay;
                        if (curr_delay > max_delay) {
                            max_delay = curr_delay;
                            max_length = curr_length;
                        }
                    }
                }
            }
        }
    }
    bool has_delay = (max_delay > 0); // this is a global variable that will be used in the kernel
    if (has_delay) {
        std::string velocity_unit = "m/s";
        std::string delay_unit = "msec";
        if (!m->base_conf.sync_msec) {
            velocity_unit = "m/0.1s";
            delay_unit = "0.1msec";
        }
        printf("Max distance %f (mm) with a minimum velocity of %f (%s) => Max delay: %d (%s)\n", max_length, min_velocity, velocity_unit.c_str(), max_delay, delay_unit.c_str());

        // allocate memory to conn_state_var_hist for N_SIMS * (nodes * max_delay)
        // TODO: make it possible to have variable max_delay per each simulation
        for (int sim_idx=0; sim_idx < m->N_SIMS; sim_idx++) {
            CUDA_CHECK_RETURN(cudaMallocManaged((void**)&conn_state_var_hist[sim_idx], sizeof(u_real) * m->nodes * max_delay));
        }
    } 
    #ifdef MANY_NODES
    // in case of large number of nodes also allocate memory to
    // conn_state_var_hist[sim_idx] for a length of nodes. This array
    // will contain the immediate history of S_i_E (at t-1)
    else {
        for (int sim_idx=0; sim_idx < m->N_SIMS; sim_idx++) {
            CUDA_CHECK_RETURN(cudaMallocManaged((void**)&conn_state_var_hist[sim_idx], sizeof(u_real) * m->nodes));
        }
    }
    #endif

    // run simulations
    dim3 numBlocks(m->N_SIMS);
    dim3 threadsPerBlock(m->nodes);
    // (third argument is extern shared memory size for S_i_1_E)
    // provide NULL for extended output variables and FIC variables
    #ifndef MANY_NODES
    size_t shared_mem_extern = m->nodes*sizeof(u_real);
    #else
    size_t shared_mem_extern = 0;
    #endif 
    // keep track of progress
    // Note: based on BOLD TRs reached in the first node
    // of each simulation (therefore the progress will be
    // an approximation of the real progress)
    uint* progress;
    CUDA_CHECK_RETURN(cudaMallocManaged(&progress, sizeof(uint)));
    uint progress_final = m->output_ts * m->N_SIMS;
    *progress = 0;
    bnm<Model><<<numBlocks,threadsPerBlock,shared_mem_extern>>>(
        d_model,
        BOLD, states_out, 
        global_out_int,
        global_out_bool,
        d_SC, d_global_params, d_regional_params,
        conn_state_var_hist, delay, max_delay,
    #ifdef NOISE_SEGMENT
        shuffled_nodes, shuffled_ts,
    #endif
        noise, progress);
    // asynchroneously print out the progress
    // if verbose
    if (m->base_conf.verbose) {
        uint last_progress = 0;
        while (*progress < progress_final) {
            // Print progress as percentage
            printf("%.2f%%\r", ((double)*progress / progress_final) * 100);
            fflush(stdout);
            // Sleep for interval ms
            std::this_thread::sleep_for(std::chrono::milliseconds(m->base_conf.progress_interval));
            // make sure it doesn't get stuck
            // by checking if there has been any progress
            if (*progress == last_progress) {
                printf("No progress detected in the last %d ms.\n", m->base_conf.progress_interval);
                break;
            }
            last_progress = *progress;
        }
        if (*progress == progress_final) {
            printf("100.00%\n");
        } else {
            printf("If no errors are shown, the simulation is still running "
                "but the progress is not being updated as there was no progress in the "
                "last %d ms, which may be too fast for current GPU and simulations\n", 
                m->base_conf.progress_interval);
        }
    }
    CUDA_CHECK_LAST_ERROR();
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());
    if (m->base_conf.verbose) {
        printf("Simulation completed\n");
    }

    // calculate mean and sd bold for FC calculation
    bold_stats<<<numBlocks, threadsPerBlock>>>(
        mean_bold, ssd_bold,
        BOLD, m->N_SIMS, m->nodes,
        m->output_ts, m->corr_len, m->n_vols_remove);
    CUDA_CHECK_LAST_ERROR();
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());
    // calculate window mean and sd bold for FCD calculations
    numBlocks.x = m->N_SIMS;
    numBlocks.y = m->n_windows;
    window_bold_stats<<<numBlocks,threadsPerBlock>>>(
        BOLD, m->N_SIMS, m->nodes, 
        m->n_windows, m->window_size+1, window_starts, window_ends,
        windows_mean_bold, windows_ssd_bold);
    CUDA_CHECK_LAST_ERROR();
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());
    // calculate FC and window FCs
    int maxThreadsPerBlock = prop.maxThreadsPerBlock;
    numBlocks.x = m->N_SIMS;
    numBlocks.y = ceil((float)m->n_pairs / (float)maxThreadsPerBlock);
    numBlocks.z = m->n_windows + 1; // +1 for total FC
    if (prop.maxThreadsPerBlock!=prop.maxThreadsDim[0]) {
        printf("Error: Code not implemented for GPUs in which maxThreadsPerBlock!=maxThreadsDim[0]\n");
        exit(1);
    }
    threadsPerBlock.x = maxThreadsPerBlock;
    threadsPerBlock.y = 1;
    threadsPerBlock.z = 1;
    fc<<<numBlocks, threadsPerBlock>>>(
        fc_trils, windows_fc_trils, BOLD, m->N_SIMS, m->nodes, m->n_pairs, pairs_i, pairs_j,
        m->output_ts, m->n_vols_remove, m->corr_len, mean_bold, ssd_bold,
        m->n_windows, m->window_size+1, windows_mean_bold, windows_ssd_bold,
        window_starts, window_ends,
        maxThreadsPerBlock
    );
    CUDA_CHECK_LAST_ERROR();
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());
    // calculate window mean and sd fc_tril for FCD calculations
    numBlocks.x = m->N_SIMS;
    numBlocks.y = 1;
    numBlocks.z = 1;
    threadsPerBlock.x = m->n_windows;
    if (m->n_windows >= prop.maxThreadsPerBlock) {
        printf("Error: Mean/ssd FC tril of %d windows cannot be calculated on this device\n", m->n_windows);
        exit(1);
    }
    window_fc_stats<<<numBlocks,threadsPerBlock>>>(
        windows_mean_fc, windows_ssd_fc,
        NULL, NULL, NULL, NULL, // skipping L and R stats
        windows_fc_trils, m->N_SIMS, m->n_windows, m->n_pairs,
        false, 0);
    CUDA_CHECK_LAST_ERROR();
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());
    // calculate FCD
    numBlocks.x = m->N_SIMS;
    numBlocks.y = ceil((float)m->n_window_pairs / (float)maxThreadsPerBlock);
    numBlocks.z = 1;
    if (prop.maxThreadsPerBlock!=prop.maxThreadsDim[0]) {
        printf("Code not implemented for GPUs in which maxThreadsPerBlock!=maxThreadsDim[0]\n");
        exit(1);
    }
    threadsPerBlock.x = maxThreadsPerBlock;
    threadsPerBlock.y = 1;
    threadsPerBlock.z = 1;
    fcd<<<numBlocks, threadsPerBlock>>>(
        fcd_trils, NULL, NULL, // skipping separate L and R fcd
        windows_fc_trils, 
        windows_mean_fc, windows_ssd_fc,
        NULL, NULL, NULL, NULL,
        m->N_SIMS, m->n_pairs, m->n_windows, m->n_window_pairs, 
        window_pairs_i, window_pairs_j, maxThreadsPerBlock,
        false, 0);
    CUDA_CHECK_LAST_ERROR();
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());

    #ifdef USE_FLOATS
    // Convert FC and FCD to doubles for GOF calculation
    numBlocks.x = m->N_SIMS;
    numBlocks.y = m->n_pairs;
    numBlocks.z = 1;
    threadsPerBlock.x = 1;
    float2double<<<numBlocks, threadsPerBlock>>>(d_fc_trils, fc_trils, m->N_SIMS, m->n_pairs);
    CUDA_CHECK_LAST_ERROR();
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());
    numBlocks.x = m->N_SIMS;
    numBlocks.y = m->n_window_pairs;
    float2double<<<numBlocks, threadsPerBlock>>>(d_fcd_trils, fcd_trils, N_SIMS, n_window_pairs);
    CUDA_CHECK_LAST_ERROR();
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());
    #endif

    // copy the output from managed memory to _out arrays (which can be numpy arrays)
    size_t ext_out_size = m->nodes;
    if (m->base_conf.extended_output_ts) {
        ext_out_size *= m->output_ts;
    }
    // TODO: pass the managed arrays data directly
    // to the python arrays without copying
    for (int sim_idx=0; sim_idx<m->N_SIMS; sim_idx++) {
        memcpy(BOLD_out, BOLD[sim_idx], sizeof(u_real) * m->bold_size);
        BOLD_out+=m->bold_size;
        memcpy(fc_trils_out, fc_trils[sim_idx], sizeof(u_real) * m->n_pairs);
        fc_trils_out+=m->n_pairs;
        memcpy(fcd_trils_out, fcd_trils[sim_idx], sizeof(u_real) * m->n_window_pairs);
        fcd_trils_out+=m->n_window_pairs;
    }
    if (m->modifies_params) { // e.g. rWW with FIC
        // copy (potentially) modified parameters back to the original array
        for (int i=0; i<Model::n_global_params; i++) {
            memcpy(global_params[i], d_global_params[i], m->N_SIMS * sizeof(u_real));
        }
        for (int i=0; i<Model::n_regional_params; i++) {
            memcpy(regional_params[i], d_regional_params[i], m->N_SIMS*m->nodes * sizeof(u_real));
        }
    }

    // free delay and conn_state_var_hist memories if allocated
    // Note: no need to clear memory of the other variables
    // as we'll want to reuse them in the next calls to run_simulations_gpu
    // within current session
    if (m->do_delay) {
        for (int sim_idx=0; sim_idx < m->N_SIMS; sim_idx++) {
            CUDA_CHECK_RETURN(cudaFree(delay[sim_idx]));
        }
        CUDA_CHECK_RETURN(cudaFree(delay));
    }
    if (has_delay) {
        for (int sim_idx=0; sim_idx < m->N_SIMS; sim_idx++) {
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
void _init_gpu(BaseModel *m, BWConstants bwc) {
    using namespace bnm_gpu;
    // check CUDA device avaliability and properties
    prop = get_device_prop(m->base_conf.verbose);

    // copy constants and configs from CPU
    // TODO: make these members of the model class
    CUDA_CHECK_RETURN(cudaMemcpyToSymbol(d_bwc, &bwc, sizeof(BWConstants)));
    if (strcmp(Model::name, "rWW")==0) {
        CUDA_CHECK_RETURN(cudaMemcpyToSymbol(d_rWWc, &Model::mc, sizeof(typename Model::Constants)));
    } else if (strcmp(Model::name, "rWWEx")==0) {
        CUDA_CHECK_RETURN(cudaMemcpyToSymbol(d_rWWExc, &Model::mc, sizeof(typename Model::Constants)));
    }

    // allocate device memory for SC
    CUDA_CHECK_RETURN(cudaMalloc(&d_SC, sizeof(u_real) * m->nodes*m->nodes));

    // allocate device memory for simulation parameters
    // size of global_params is (n_global_params, N_SIMS)
    // size of regional_params is (n_regional_params, N_SIMS * nodes)
    if (Model::n_global_params > 0) {
        CUDA_CHECK_RETURN(cudaMallocManaged((void**)&d_global_params, sizeof(u_real*) * Model::n_global_params));
        for (int param_idx=0; param_idx<Model::n_global_params; param_idx++) {
            CUDA_CHECK_RETURN(cudaMallocManaged((void**)&d_global_params[param_idx], sizeof(u_real) * m->N_SIMS));
        }
    }
    if (Model::n_regional_params > 0) {
        CUDA_CHECK_RETURN(cudaMallocManaged((void**)&d_regional_params, sizeof(u_real*) * Model::n_regional_params));
        for (int param_idx=0; param_idx<Model::n_regional_params; param_idx++) {
            CUDA_CHECK_RETURN(cudaMallocManaged((void**)&d_regional_params[param_idx], sizeof(u_real) * m->N_SIMS * m->nodes));
        }
    }

    CUDA_CHECK_RETURN(cudaMallocManaged((void**)&BOLD, sizeof(u_real*) * m->N_SIMS));


    // set up global int and bool outputs
    if (Model::n_global_out_int > 0) {
        CUDA_CHECK_RETURN(cudaMallocManaged((void**)&global_out_int, sizeof(int*) * Model::n_global_out_int));
        for (int i=0; i<Model::n_global_out_int; i++) {
            CUDA_CHECK_RETURN(cudaMallocManaged((void**)&global_out_int[i], sizeof(int) * m->N_SIMS));
        }
    }
    if (Model::n_global_out_bool > 0) {
        CUDA_CHECK_RETURN(cudaMallocManaged((void**)&global_out_bool, sizeof(bool*) * Model::n_global_out_bool));
        for (int i=0; i<Model::n_global_out_bool; i++) {
            CUDA_CHECK_RETURN(cudaMallocManaged((void**)&global_out_bool[i], sizeof(bool) * m->N_SIMS));
        }
    }

    // allocate memory for extended output
    size_t ext_out_size = m->nodes;
    if (m->base_conf.extended_output_ts) {
        ext_out_size *= m->output_ts;
    }
    if (m->base_conf.extended_output) {
        CUDA_CHECK_RETURN(cudaMallocManaged((void**)&states_out, sizeof(u_real**) * Model::n_state_vars));
        for (int var_idx=0; var_idx<Model::n_state_vars; var_idx++) {
            CUDA_CHECK_RETURN(cudaMallocManaged((void**)&states_out[var_idx], sizeof(u_real*) * m->N_SIMS));
            for (int sim_idx=0; sim_idx<m->N_SIMS; sim_idx++) {
                CUDA_CHECK_RETURN(cudaMallocManaged((void**)&states_out[var_idx][sim_idx], sizeof(u_real) * ext_out_size));
            }
        }
    }

    // specify n_vols_remove (for extended output and FC calculations)
    m->n_vols_remove = m->base_conf.bold_remove_s * 1000 / m->BOLD_TR;

    // preparing FC calculations
    m->corr_len = m->output_ts - m->n_vols_remove;
    if (m->corr_len < 2) {
        printf("Number of BOLD volumes (after removing initial volumes) is too low for FC calculations\n");
        exit(1);
    }
    CUDA_CHECK_RETURN(cudaMallocManaged((void**)&mean_bold, sizeof(u_real*) * m->N_SIMS));
    CUDA_CHECK_RETURN(cudaMallocManaged((void**)&ssd_bold, sizeof(u_real*) * m->N_SIMS));
    m->n_pairs = ((m->nodes) * (m->nodes - 1)) / 2;
    int rh_idx;
    if (m->base_conf.exc_interhemispheric) {
        if ((m->nodes % 2) != 0) {
            printf("Error: exc_interhemispheric is set but number of nodes is not even\n");
            exit(1);
        }
        rh_idx = m->nodes / 2; // assumes symmetric number of parcels and L->R order
        m->n_pairs -= pow(rh_idx, 2); // exclude the middle square
    }
    // create a mapping between pair_idx and i and j
    int curr_idx = 0;
    CUDA_CHECK_RETURN(cudaMallocManaged(&pairs_i, sizeof(int) * m->n_pairs));
    CUDA_CHECK_RETURN(cudaMallocManaged(&pairs_j, sizeof(int) * m->n_pairs));
    for (int i=0; i < m->nodes; i++) {
        for (int j=0; j < m->nodes; j++) {
            if (i > j) {
                if (m->base_conf.exc_interhemispheric) {
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
    CUDA_CHECK_RETURN(cudaMallocManaged((void**)&fc_trils, sizeof(u_real*) * m->N_SIMS));

    // FCD preparation
    // calculate number of windows and window start/end indices
    int *_window_starts, *_window_ends; // are cpu integer arrays
    m->n_windows = get_dfc_windows(
        &_window_starts, &_window_ends, 
        m->corr_len, m->output_ts, m->n_vols_remove,
        m->window_step, m->window_size, m->base_conf.drop_edges);
    if (m->n_windows == 0) {
        printf("Error: Number of windows is 0\n");
        exit(1);
    }
    CUDA_CHECK_RETURN(cudaMallocManaged(&window_starts, sizeof(int) * m->n_windows));
    CUDA_CHECK_RETURN(cudaMallocManaged(&window_ends, sizeof(int) * m->n_windows));
    for (int i=0; i<m->n_windows; i++) {
        window_starts[i] = _window_starts[i];
        window_ends[i] = _window_ends[i];
    }
    // allocate memory for mean and ssd BOLD of each window
    // (n_sims x n_windows x nodes)
    CUDA_CHECK_RETURN(cudaMallocManaged((void**)&windows_mean_bold, sizeof(u_real*) * m->N_SIMS));
    CUDA_CHECK_RETURN(cudaMallocManaged((void**)&windows_ssd_bold, sizeof(u_real*) * m->N_SIMS));
    CUDA_CHECK_RETURN(cudaMallocManaged((void**)&windows_fc_trils, sizeof(u_real*) * m->N_SIMS));
    // allocate memory for mean and ssd fc_tril of each window
    // (n_sims x n_windows)
    CUDA_CHECK_RETURN(cudaMallocManaged((void**)&windows_mean_fc, sizeof(u_real*) * m->N_SIMS));
    CUDA_CHECK_RETURN(cudaMallocManaged((void**)&windows_ssd_fc, sizeof(u_real*) * m->N_SIMS));
    // create a mapping between window_pair_idx and i and j
    m->n_window_pairs = (m->n_windows * (m->n_windows-1)) / 2;
    curr_idx = 0;
    CUDA_CHECK_RETURN(cudaMallocManaged(&window_pairs_i, sizeof(int) * m->n_window_pairs));
    CUDA_CHECK_RETURN(cudaMallocManaged(&window_pairs_j, sizeof(int) * m->n_window_pairs));
    for (int i=0; i < m->n_windows; i++) {
        for (int j=0; j < m->n_windows; j++) {
            if (i > j) {
                window_pairs_i[curr_idx] = i;
                window_pairs_j[curr_idx] = j;
                curr_idx++;
            }
        }
    }
    // allocate memory for fcd trils
    CUDA_CHECK_RETURN(cudaMallocManaged((void**)&fcd_trils, sizeof(u_real*) * m->N_SIMS));

    #ifdef USE_FLOATS
    // allocate memory for double versions of fc and fcd trils on CPU
    CUDA_CHECK_RETURN(cudaMallocManaged((void**)&d_fc_trils, sizeof(double*) * m->N_SIMS));
    CUDA_CHECK_RETURN(cudaMallocManaged((void**)&d_fcd_trils, sizeof(double*) * m->N_SIMS));
    #endif



    // allocate memory per each simulation
    for (int sim_idx=0; sim_idx<m->N_SIMS; sim_idx++) {
        // allocate a chunk of BOLD to this simulation (not sure entirely if this is the best way to do it)
        CUDA_CHECK_RETURN(cudaMallocManaged((void**)&BOLD[sim_idx], sizeof(u_real) * m->bold_size));
        // allocate memory for fc calculations
        CUDA_CHECK_RETURN(cudaMallocManaged((void**)&mean_bold[sim_idx], sizeof(u_real) * m->nodes));
        CUDA_CHECK_RETURN(cudaMallocManaged((void**)&ssd_bold[sim_idx], sizeof(u_real) * m->nodes));
        CUDA_CHECK_RETURN(cudaMallocManaged((void**)&fc_trils[sim_idx], sizeof(u_real) * m->n_pairs));
        // allocate memory for window fc and fcd calculations
        CUDA_CHECK_RETURN(cudaMallocManaged((void**)&windows_mean_bold[sim_idx], sizeof(u_real) * m->n_windows * m->nodes));
        CUDA_CHECK_RETURN(cudaMallocManaged((void**)&windows_ssd_bold[sim_idx], sizeof(u_real) * m->n_windows * m->nodes));
        CUDA_CHECK_RETURN(cudaMallocManaged((void**)&windows_fc_trils[sim_idx], sizeof(u_real) * m->n_windows * m->n_pairs));
        CUDA_CHECK_RETURN(cudaMallocManaged((void**)&windows_mean_fc[sim_idx], sizeof(u_real) * m->n_windows));
        CUDA_CHECK_RETURN(cudaMallocManaged((void**)&windows_ssd_fc[sim_idx], sizeof(u_real) * m->n_windows));
        CUDA_CHECK_RETURN(cudaMallocManaged((void**)&fcd_trils[sim_idx], sizeof(u_real) * m->n_window_pairs));
        #ifdef USE_FLOATS
        // allocate memory for double copies of fc and fcd
        CUDA_CHECK_RETURN(cudaMallocManaged((void**)&d_fc_trils[sim_idx], sizeof(double) * m->n_pairs));
        CUDA_CHECK_RETURN(cudaMallocManaged((void**)&d_fcd_trils[sim_idx], sizeof(double) * m->n_window_pairs));
        #endif
    }

    // check if noise needs to be calculated
    if (
        (m->rand_seed != m->last_rand_seed) ||
        (m->time_steps != m->last_time_steps) ||
        (m->nodes != m->last_nodes) ||
        (m->base_conf.noise_time_steps != m->last_noise_time_steps)
        ) {
        // pre-calculate normally-distributed noise on CPU
        // this is necessary to ensure consistency of noise given the same seed
        // doing the same thing directly on the device is more challenging
        #ifndef NOISE_SEGMENT
        // precalculate the entire noise needed; can use up a lot of memory
        // with high N of nodes and longer durations leads maxes out the memory
        m->noise_size = m->nodes * (m->time_steps+1) * 10 * Model::n_noise; // +1 for inclusive last time point, 2 for E and I
        #else
        // otherwise precalculate a noise segment and arrays of shuffled
        // nodes and time points and reuse-shuffle the noise segment
        // throughout the simulation for `noise_repeats`
        m->noise_size = m->nodes * (m->base_conf.noise_time_steps) * 10 * Model::n_noise;
        m->noise_repeats = ceil((float)(m->time_steps+1) / (float)(m->base_conf.noise_time_steps)); // +1 for inclusive last time point
        #endif
        printf("Precalculating %d noise elements...\n", m->noise_size);
        if (m->last_nodes != 0) {
            // noise is being recalculated, free the previous one
            CUDA_CHECK_RETURN(cudaFree(noise));
            #ifdef NOISE_SEGMENT
            CUDA_CHECK_RETURN(cudaFree(shuffled_nodes));
            CUDA_CHECK_RETURN(cudaFree(shuffled_ts));
            #endif
        }
        m->last_time_steps = m->time_steps;
        m->last_nodes = m->nodes;
        m->last_rand_seed = m->rand_seed;
        m->last_noise_time_steps = m->base_conf.noise_time_steps;
        std::mt19937 rand_gen(m->rand_seed);
        std::normal_distribution<float> normal_dist(0, 1);
        CUDA_CHECK_RETURN(cudaMallocManaged(&noise, sizeof(u_real) * m->noise_size));
        for (int i = 0; i < m->noise_size; i++) {
            #ifdef USE_FLOATS
            noise[i] = normal_dist(rand_gen);
            #else
            noise[i] = (double)normal_dist(rand_gen);
            #endif
        }
        #ifdef NOISE_SEGMENT
        // create shuffled nodes and ts indices for each repeat of the 
        // precalculaed noise 
        printf("noise will be repeated %d times (nodes [rows] and timepoints [columns] will be shuffled in each repeat)\n", m->noise_repeats);
        CUDA_CHECK_RETURN(cudaMallocManaged(&shuffled_nodes, sizeof(int) * m->noise_repeats * m->nodes));
        CUDA_CHECK_RETURN(cudaMallocManaged(&shuffled_ts, sizeof(int) * m->noise_repeats * m->base_conf.noise_time_steps));
        get_shuffled_nodes_ts(&shuffled_nodes, &shuffled_ts,
            m->nodes, m->base_conf.noise_time_steps, m->noise_repeats, &rand_gen);
        #endif
    } else {
        printf("Noise already precalculated\n");
    }

    m->is_initialized = true;
}