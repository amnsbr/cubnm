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
#include "constants.hpp"
#include "utils.hpp"

#ifdef MANY_NODES
    #warning Compiling with -D MANY_NODES (S_E at t-1 is stored in device instead of shared memory)
#endif

// extern void analytical_fic_het(
//         gsl_matrix * sc, double G, double * w_EE, double * w_EI,
//         gsl_vector * w_IE_out, bool * _unstable);

namespace cg = cooperative_groups;

#define CUDA_CHECK_RETURN(value) {											\
	cudaError_t _m_cudaStat = value;										\
	if (_m_cudaStat != cudaSuccess) {										\
		fprintf(stderr, "Error %s at line %d in file %s\n",					\
				cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);		\
		exit(1);															\
	} }

#define CUDA_CHECK_LAST_ERROR() checkLast(__FILE__, __LINE__)
void checkLast(const char* const file, const int line)
{
    // from https://leimao.github.io/blog/Proper-CUDA-Error-Checking/
    cudaError_t err{cudaGetLastError()};
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line
                  << std::endl;
        std::cerr << cudaGetErrorString(err) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

__constant__ struct BWConstants d_bwc;
__constant__ struct ModelConfigs d_conf;

namespace bnm_gpu {
    bool is_initialized = false;
}
int n_vols_remove, corr_len, n_windows, n_pairs, n_window_pairs, output_ts, max_delay;
bool adjust_fic, has_delay;
cudaDeviceProp prop;
bool *fic_failed, *fic_unstable;
u_real ***state_vars_out,
    **BOLD, **mean_bold, **ssd_bold, **fc_trils, **windows_mean_bold, **windows_ssd_bold,
    **windows_fc_trils, **windows_mean_fc, **windows_ssd_fc, **fcd_trils, *noise, **w_IE_fic,
    *d_SC, **d_global_params, **d_regional_params;
int *fic_n_trials, *pairs_i, *pairs_j, *window_starts, *window_ends, *window_pairs_i, *window_pairs_j;
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
#ifdef USE_FLOATS
double **d_fc_trils, **d_fcd_trils;
#else
// use d_fc_trils and d_fcd_trils as aliases for fc_trils and fcd_trils
// which will later be used for GOF calculations
#define d_fc_trils fc_trils
#define d_fcd_trils fcd_trils
#endif
gsl_vector * emp_FCD_tril_copy;

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

__device__ void bw_step(
        u_real* bw_x, u_real* bw_f, u_real* bw_nu, 
        u_real* bw_q, u_real* tmp_f,
        u_real* S_i_E
        ) {
    // Balloon-Windkessel model integration step
    *bw_x  = (*bw_x)  +  d_bwc.bw_dt * ((*S_i_E) - d_bwc.kappa * (*bw_x) - d_bwc.y * ((*bw_f) - 1.0));
    *tmp_f = (*bw_f)  +  d_bwc.bw_dt * (*bw_x);
    *bw_nu = (*bw_nu) +  d_bwc.bw_dt_itau * ((*bw_f) - pow((*bw_nu), d_bwc.ialpha));
    *bw_q  = (*bw_q)  +  d_bwc.bw_dt_itau * ((*bw_f) * (1.0 - pow(d_bwc.oneminrho,(1.0/ (*bw_f)))) / d_bwc.rho  - pow(*bw_nu,d_bwc.ialpha) * (*bw_q) / (*bw_nu));
    *bw_f  = *tmp_f;
}


__device__ void rWWModel::init(
    u_real* _state_vars
) {
    _state_vars[0] = 0.0; // I_E
    _state_vars[1] = 0.0; // I_I
    _state_vars[2] = 0.0; // r_E
    _state_vars[3] = 0.0; // r_I
    _state_vars[4] = 0.001; // S_E
    _state_vars[5] = 0.001; // S_I
}

__device__ void rWWModel::step(
        u_real* _state_vars, u_real* _intermediate_vars,
        u_real* _global_params, u_real* _regional_params,
        u_real* tmp_globalinput,
        u_real* noise, long* noise_idx, rWWModel* model
        ) {
    _state_vars[0] = model->mc.w_E__I_0 + _regional_params[0] * _state_vars[4] + (*tmp_globalinput) * _global_params[0] * model->mc.J_NMDA - _regional_params[2] * _state_vars[5];
    // *tmp_I_E = model->mc.w_E__I_0 + (*w_EE) * (*S_i_E) + (*tmp_globalinput) * (*G_J_NMDA) - (*w_IE) * (*S_i_I);
    _state_vars[1] = model->mc.w_I__I_0 + _regional_params[1] * _state_vars[4] - _state_vars[5];
    // *tmp_I_I = model->mc.w_I__I_0 + (*w_EI) * (*S_i_E) - (*S_i_I);
    _intermediate_vars[0] = model->mc.a_E * _state_vars[0] - model->mc.b_E;
    // *tmp_aIb_E = model->mc.a_E * (*tmp_I_E) - model->mc.b_E;
    _intermediate_vars[1] = model->mc.a_I * _state_vars[1] - model->mc.b_I;
    // *tmp_aIb_I = model->mc.a_I * (*tmp_I_I) - model->mc.b_I;
    #ifdef USE_FLOATS
    // to avoid firing rate approaching infinity near I = b/a
    if (abs(_intermediate_vars[0]) < 1e-4) _intermediate_vars[0] = 1e-4;
    if (abs(_intermediate_vars[0]) < 1e-4) _intermediate_vars[0] = 1e-4;
    #endif
    _state_vars[2] = _intermediate_vars[0] / (1 - exp(-model->mc.d_E * _intermediate_vars[0]));
    // *tmp_r_E = *tmp_aIb_E / (1 - exp(-model->mc.d_E * (*tmp_aIb_E)));
    _state_vars[3] = _intermediate_vars[1] / (1 - exp(-model->mc.d_I * _intermediate_vars[1]));
    // *tmp_r_I = *tmp_aIb_I / (1 - exp(-model->mc.d_I * (*tmp_aIb_I)));
    _intermediate_vars[2] = noise[*noise_idx] * model->mc.sigma_model_sqrt_dt + model->mc.dt_gamma_E * ((1 - _state_vars[4]) * _state_vars[2]) - model->mc.dt_itau_E * _state_vars[4];
    // *dSdt_E = noise[*noise_idx] * model->mc.sigma_model_sqrt_dt + model->mc.dt_gamma_E * ((1 - (*S_i_E)) * (*tmp_r_E)) - model->mc.dt_itau_E * (*S_i_E);
    _intermediate_vars[3] = noise[(*noise_idx)+1] * model->mc.sigma_model_sqrt_dt + model->mc.dt_gamma_I * _state_vars[3] - model->mc.dt_itau_I * _state_vars[5];
    // *dSdt_I = noise[*noise_idx+1] * model->mc.sigma_model_sqrt_dt + model->mc.dt_gamma_I * (*tmp_r_I) - model->mc.dt_itau_I * (*S_i_I);
    _state_vars[4] += _intermediate_vars[2];
    // *S_i_E += *dSdt_E;
    _state_vars[5] += _intermediate_vars[3];
    // *S_i_I += *dSdt_I;
    // clip S to 0-1
    _state_vars[4] = max(0.0f, min(1.0f, _state_vars[4]));
    // *S_i_E = max(0.0f, min(1.0f, *S_i_E));
    _state_vars[5] = max(0.0f, min(1.0f, _state_vars[5]));
    // *S_i_I = max(0.0f, min(1.0f, *S_i_I));
}

// __device__ void fic_adjust(
//     u_real* mean_I_E, u_real* tmp_I_E, u_real* I_E_ba_diff,
//     u_real* w_IE, u_real* delta,
//     int* ts_bold, int* fic_trial, int* max_fic_trials,
//     bool* needs_fic_adjustment, bool* _adjust_fic,
//     bool* fic_failed, cg::thread_block* b, 
//     // following variables are needed for the reset
//     int* BOLD_len_i, int* nodes, int* sim_idx, int* j,
//     int* buff_idx, int* max_delay, int** delay, int* sh_j,
//     int* shuffled_nodes, int* shuffled_ts, 
//     int* curr_noise_repeat, int* ts_noise,
//     int* sh_ts_noise, bool* extended_output, bool* has_delay,
//     u_real* S_1_E, u_real* S_i_E, u_real* S_i_I, u_real* bw_x,
//     u_real* bw_f, u_real* bw_nu, u_real* bw_q,
//     u_real** S_E, u_real** I_E, u_real** r_E, u_real** S_I, u_real** I_I,
//     u_real** r_I, u_real** S_i_E_hist
// ) {

// }

// __device__ void rWWModel::post_bw_step(
//         u_real* _state_vars, u_real* _intermediate_vars,
//         u_real* _global_params, u_real* _regional_params
//         ) {
//     if ((*ts_bold >= d_conf.I_SAMPLING_START) & (*ts_bold <= d_conf.I_SAMPLING_END)) {
//         *mean_I_E += *tmp_I_E;
//     }
//     if (*ts_bold == d_conf.I_SAMPLING_END) {
//         *needs_fic_adjustment = false;
//         cg::sync(*b); // all threads must be at the same time point here given needs_fic_adjustment is shared
//         *mean_I_E /= d_conf.I_SAMPLING_DURATION;
//         *I_E_ba_diff = *mean_I_E - model->mc.b_a_ratio_E;
//         if (abs(*I_E_ba_diff + 0.026) > 0.005) {
//             *needs_fic_adjustment = true;
//             if (*fic_trial < *max_fic_trials) { // only do the adjustment if max trials is not exceeded
//                 // up- or downregulate inhibition
//                 if ((*I_E_ba_diff) < -0.026) {
//                     *w_IE -= *delta;
//                     // printf("sim %d node %d (trial %d): %f ==> adjusting w_IE by -%f ==> %f\n", sim_idx, j, fic_trial, I_E_ba_diff, delta, w_IE);
//                     *delta -= 0.001;
//                     *delta = CUDA_MAX(*delta, 0.001);
//                 } else {
//                     *w_IE += *delta;
//                     // printf("sim %d node %d (trial %d): %f ==> adjusting w_IE by +%f ==> %f\n", sim_idx, j, fic_trial, I_E_ba_diff, delta, w_IE);
//                 }
//             }
//         }
//         cg::sync(*b); // wait to see if needs_fic_adjustment in any node
//         // if needs_fic_adjustment in any node do another trial or declare fic failure and continue
//         // the simulation until the end
//         if (*needs_fic_adjustment) {
//             if (*fic_trial < *max_fic_trials) {
//                 // TODO: create a `init` kernel
//                 // than will be used here and at
//                 // the beginning of the simulation
//                 // reset time
//                 *ts_bold = 0;
//                 *BOLD_len_i = 0;
//                 *fic_trial++;
//                 // reset states
//                 S_1_E[*j] = 0.001;
//                 *S_i_E = 0.001;
//                 *S_i_I = 0.001;
//                 *bw_x = 0.0;
//                 *bw_f = 1.0;
//                 *bw_nu = 1.0;
//                 *bw_q = 1.0;
//                 *mean_I_E = 0;
//                 // reset extended output
//                 // normally this isn't needed because by default
//                 // bold_remove_s is 30s and FIC adjustment happens < 10s
//                 // so these variables have not been updated so far
//                 if (*extended_output & (!d_conf.extended_output_ts)) {
//                     S_E[*sim_idx][*j] = 0.0;
//                     I_E[*sim_idx][*j] = 0.0;
//                     r_E[*sim_idx][*j] = 0.0;
//                     S_I[*sim_idx][*j] = 0.0;
//                     I_I[*sim_idx][*j] = 0.0;
//                     r_I[*sim_idx][*j] = 0.0;
//                 }
//                 if (*has_delay) {
//                     // reset buffer
//                     // initialize S_i_E_hist in every time point at initial value
//                     for (*buff_idx=0; *buff_idx<*max_delay; (*buff_idx)++) {
//                         S_i_E_hist[*sim_idx][(*buff_idx)*(*nodes)+(*j)] = 0.001;
//                     }
//                     *buff_idx = *max_delay-1;
//                 }
//                 #ifdef NOISE_SEGMENT
//                 // reset the node shuffling
//                 *sh_j = shuffled_nodes[*j];
//                 *curr_noise_repeat = 0;
//                 *ts_noise = 0;
//                 *sh_ts_noise = shuffled_ts[*ts_noise];
//                 #endif
//             } else {
//                 // continue the simulation but
//                 // declare FIC failed
//                 fic_failed[*sim_idx] = true;
//                 *_adjust_fic = false;
//             }
//         } else {
//             // if no node needs fic adjustment don't run
//             // this block of code any more
//             *_adjust_fic = false;
//         }
//         cg::sync(*b);
//     }
// }

template<typename Model>
__global__ void bnm(
    u_real **BOLD,
    u_real ***state_vars_out, 
    Model* model,
    int n_vols_remove,
    u_real *SC, u_real **global_params, u_real **regional_params,
    u_real **conn_state_var_hist, int **delay, int max_delay,
    int N_SIMS, int nodes, int BOLD_TR, int time_steps, 
    u_real *noise, 
    // bool do_fic, u_real **w_IE_fic,
    // bool adjust_fic, int max_fic_trials, bool *fic_unstable, bool *fic_failed, int *fic_n_trials,
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
    cg::thread_block b = cg::this_thread_block();
    int j = b.thread_rank();
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

    // initialize state variables
    u_real _state_vars[Model::n_state_vars];
    model->init(_state_vars);
    // initialize extended output sums
    if (extended_output & (!d_conf.extended_output_ts)) {
        for (ii=0; ii<Model::n_state_vars; ii++) {
            state_vars_out[ii][sim_idx][j] = 0;
        }
    }
    // initialize intermediate variables
    u_real _intermediate_vars[Model::n_intermediate_vars];
    for (ii=0; ii<Model::n_intermediate_vars; ii++) {
        _intermediate_vars[ii] = 0;
    }

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

//     // initializations for FIC numerical adjustment
//     u_real mean_I_E, delta, I_E_ba_diff;
//     int fic_trial;
//     if (adjust_fic) {
//         mean_I_E = 0;
//         delta = d_conf.init_delta;
//         fic_trial = 0;
//         fic_failed[sim_idx] = false;
//     }
//     // copy adjust_fic to a local shared variable and use it to indicate
//     // if adjustment should be stopped after N trials
//     __shared__ bool _adjust_fic;
//     _adjust_fic = adjust_fic;
//     __shared__ bool needs_fic_adjustment;

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
            cg::sync(b);
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
                cg::sync(b);
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
                    cg::sync(b);
                    buff_idx = (buff_idx + max_delay - 1) % max_delay;
                    conn_state_var_hist[sim_idx][buff_idx*nodes+j] = _state_vars[Model::conn_state_var_idx];
                } else  {
                    // wait for other regions
                    // see note above on why this is needed before updating S_i_1_E
                    cg::sync(b);
                    conn_state_var_1[j] = _state_vars[Model::conn_state_var_idx];
                }
            }
        }

        if (d_conf.sync_msec) {
            if (has_delay) {
                // go 1 ms backward in the buffer and save most recent S_i_E
                // wait for all regions to have correct (same) buff_idx
                // and updated conn_state_var_hist
                cg::sync(b);
                buff_idx = (buff_idx + max_delay - 1) % max_delay;
                conn_state_var_hist[sim_idx][buff_idx*nodes+j] = _state_vars[Model::conn_state_var_idx];
            } else {
                // wait for other regions
                // see note above on why this is needed before updating S_i_1_E
                cg::sync(b);
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
                    state_vars_out[ii][sim_idx][BOLD_len_i*nodes+j] = _state_vars[ii];
                }
            }
            // update sum (later mean) of extended
            // output only after n_vols_remove
            if ((BOLD_len_i>=n_vols_remove) & extended_output & (!d_conf.extended_output_ts)) {
                for (ii=0; ii<Model::n_state_vars; ii++) {
                    state_vars_out[ii][sim_idx][j] += _state_vars[ii];
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
                // if ((j==0)&(sim_idx==0)) printf("t=%d curr_repeat->%d sh_j %d->", ts_noise, curr_noise_repeat, sh_j);
                sh_j = shuffled_nodes[curr_noise_repeat*nodes+j];
                // if ((j==0)&(sim_idx==0)) printf("%d\n", sh_j);
                ts_noise = 0;
            }
        }
        // get the shuffled timepoint corresponding to ts_noise 
        sh_ts_noise = shuffled_ts[ts_noise+curr_noise_repeat*noise_time_steps];
        #endif

        // model->post_step(
        //     _state_vars, _intermediate_vars, 
        //     _global_params, _regional_params,
        // );

//         if (_adjust_fic) {
//             fic_adjust(
//                 &mean_I_E, &tmp_I_E, &I_E_ba_diff,
//                 &w_IE, &delta,
//                 &ts_bold, &fic_trial, &max_fic_trials,
//                 &needs_fic_adjustment, &_adjust_fic,
//                 fic_failed, &b, 
//                 // following variables are needed for the reset
//                 &BOLD_len_i, &nodes, &sim_idx, &j,
//                 &buff_idx, &max_delay, delay, &sh_j,
//                 shuffled_nodes, shuffled_ts, 
//                 &curr_noise_repeat, &ts_noise,
//                 &sh_ts_noise, &extended_output, &has_delay,
//                 S_1_E, &S_i_E, &S_i_I, &bw_x,
//                 &bw_f, &bw_nu, &bw_q, 
//                 S_E, I_E, r_E, S_I, I_I,
//                 r_I, conn_state_var_hist
//             );
//         }
    }
//     if (adjust_fic) {
//         // save the final w_IE after adjustment
//         w_IE_fic[sim_idx][j] = w_IE;
//         // as well as the number of adjustment trials needed
//         fic_n_trials[sim_idx] = fic_trial;
//     }
    if (extended_output & (!d_conf.extended_output_ts)) {
        // take average
        int extended_output_time_points = BOLD_len_i - n_vols_remove;
        for (ii=0; ii<Model::n_state_vars; ii++) {
            state_vars_out[ii][sim_idx][j] /= extended_output_time_points;
        }
    }
}

__global__ void bold_stats(
    u_real **mean_bold, u_real **ssd_bold,
    u_real **BOLD, int N_SIMS, int nodes,
    int output_ts, int corr_len, int n_vols_remove) {
    // TODO: consider combining this with window_bold_stats
    // get simulation index
    int sim_idx = blockIdx.x;
    if (sim_idx >= N_SIMS) return;
    // get node index
    int j = threadIdx.x;
    if (j >= nodes) return;

    // mean
    u_real _mean_bold = 0;
    int vol;
    for (vol=n_vols_remove; vol<output_ts; vol++) {
        _mean_bold += BOLD[sim_idx][vol*nodes+j];
    }
    _mean_bold /= corr_len;
    // ssd
    u_real _ssd_bold = 0;
    for (vol=n_vols_remove; vol<output_ts; vol++) {
        _ssd_bold += POW(BOLD[sim_idx][vol*nodes+j] - _mean_bold, 2);
    }
    // save to memory
    mean_bold[sim_idx][j] = _mean_bold;
    ssd_bold[sim_idx][j] = SQRT(_ssd_bold);
}

__global__ void window_bold_stats(
    u_real **BOLD, int N_SIMS, int nodes,
    int n_windows, int window_size_1, int *window_starts, int *window_ends,
    u_real **windows_mean_bold, u_real **windows_ssd_bold) {
        // get simulation index
        int sim_idx = blockIdx.x;
        if (sim_idx >= N_SIMS) return;
        // get window index
        int w = blockIdx.y;
        if (w >= n_windows) return;
        // get node index
        int j = threadIdx.x;
        if (j >= nodes) return;
        // calculate mean of window
        u_real _mean_bold = 0;
        int vol;
        for (vol=window_starts[w]; vol<=window_ends[w]; vol++) {
            _mean_bold += BOLD[sim_idx][vol*nodes+j];
        }
        _mean_bold /= window_size_1;
        // calculate sd of window
        u_real _ssd_bold = 0;
        for (vol=window_starts[w]; vol<=window_ends[w]; vol++) {
            _ssd_bold += POW(BOLD[sim_idx][vol*nodes+j] - _mean_bold, 2);
        }
        // save to memory
        windows_mean_bold[sim_idx][w*nodes+j] = _mean_bold;
        windows_ssd_bold[sim_idx][w*nodes+j] = SQRT(_ssd_bold);
}

__global__ void fc(u_real **fc_trils, u_real **windows_fc_trils,
    u_real **BOLD, int N_SIMS, int nodes, int n_pairs, int *pairs_i,
    int *pairs_j, int output_ts, int n_vols_remove, 
    int corr_len, u_real **mean_bold, u_real **ssd_bold, 
    int n_windows, int window_size_1, u_real **windows_mean_bold, u_real **windows_ssd_bold,
    int *window_starts, int *window_ends,
    int maxThreadsPerBlock) {
        // get simulation index
        int sim_idx = blockIdx.x;
        if (sim_idx >= N_SIMS) return;
        // get pair index
        int pair_idx = threadIdx.x + (maxThreadsPerBlock * blockIdx.y);
        if (pair_idx >= n_pairs) return;
        int i = pairs_i[pair_idx];
        int j = pairs_j[pair_idx];
        // get window index
        int w = blockIdx.z - 1; // -1 indicates total FC
        if (w >= n_windows) return;
        int vol_start, vol_end;
        u_real _mean_bold_i, _mean_bold_j, _ssd_bold_i, _ssd_bold_j;
        if (w == -1) {
            vol_start = n_vols_remove;
            vol_end = output_ts;
            _mean_bold_i = mean_bold[sim_idx][i];
            _ssd_bold_i = ssd_bold[sim_idx][i];
            _mean_bold_j = mean_bold[sim_idx][j];
            _ssd_bold_j = ssd_bold[sim_idx][j];
        } else {
            vol_start = window_starts[w];
            vol_end = window_ends[w]+1; // +1 because end is non-inclusive
            _mean_bold_i = windows_mean_bold[sim_idx][w*nodes+i];
            _ssd_bold_i = windows_ssd_bold[sim_idx][w*nodes+i];
            _mean_bold_j = windows_mean_bold[sim_idx][w*nodes+j];
            _ssd_bold_j = windows_ssd_bold[sim_idx][w*nodes+j];
        }
        // calculate sigma(x_i * x_j)
        int vol;
        u_real cov = 0;
        for (vol=vol_start; vol<vol_end; vol++) {
            cov += (BOLD[sim_idx][vol*nodes+i] - _mean_bold_i) * (BOLD[sim_idx][vol*nodes+j] - _mean_bold_j);
        }
        // calculate corr(i, j)
        u_real corr = cov / (_ssd_bold_i * _ssd_bold_j);
        if (w == -1) {
            fc_trils[sim_idx][pair_idx] = corr;
        } else {
            windows_fc_trils[sim_idx][w*n_pairs+pair_idx] = corr;
        }
    }

__global__ void window_fc_stats(
    u_real **windows_mean_fc, u_real **windows_ssd_fc,
    u_real **L_windows_mean_fc, u_real **L_windows_ssd_fc,
    u_real **R_windows_mean_fc, u_real **R_windows_ssd_fc,
    u_real **windows_fc_trils, int N_SIMS, int n_windows, int n_pairs,
    bool save_hemis, int n_pairs_hemi) {
        // get simulation index
        int sim_idx = blockIdx.x;
        if (sim_idx >= N_SIMS) return;
        // get window index
        int w = threadIdx.x;
        if (w >= n_windows) return;
        // get hemi
        int hemi = blockIdx.z;
        if (!save_hemis) {
            if (hemi > 0) return;
        } else {
            if (hemi > 2) return;
        }
        // calculate mean fc of window
        u_real _mean_fc = 0;
        int pair_idx_start = 0;
        int pair_idx_end = n_pairs; // non-inclusive
        int pair_idx;
        int _curr_n_pairs = n_pairs;
        // for left and right specify start and end indices
        // that belong to current hemi. Note that this will work
        // regardless of exc_interhemispheric true or false
        if (hemi == 1) { // left
            pair_idx_end = n_pairs_hemi;
            _curr_n_pairs = n_pairs_hemi;
        } else if (hemi == 2) { // right
            pair_idx_start = n_pairs - n_pairs_hemi;
            _curr_n_pairs = n_pairs_hemi;
        }
        for (pair_idx=pair_idx_start; pair_idx<pair_idx_end; pair_idx++) {
            _mean_fc += windows_fc_trils[sim_idx][w*n_pairs+pair_idx];
        }
        _mean_fc /= _curr_n_pairs;
        // calculate ssd fc of window
        u_real _ssd_fc = 0;
        for (pair_idx=pair_idx_start; pair_idx<pair_idx_end; pair_idx++) {
            _ssd_fc += POW(windows_fc_trils[sim_idx][w*n_pairs+pair_idx] - _mean_fc, 2);
        }
        // save to memory
        if (hemi == 0) {
            windows_mean_fc[sim_idx][w] = _mean_fc;
            windows_ssd_fc[sim_idx][w] = SQRT(_ssd_fc);
        } else if (hemi == 1) {
            L_windows_mean_fc[sim_idx][w] = _mean_fc;
            L_windows_ssd_fc[sim_idx][w] = SQRT(_ssd_fc);
        } else if (hemi == 2) {
            R_windows_mean_fc[sim_idx][w] = _mean_fc;
            R_windows_ssd_fc[sim_idx][w] = SQRT(_ssd_fc);
        }
    }

__global__ void fcd(
    u_real **fcd_trils, u_real **L_fcd_trils, u_real **R_fcd_trils,
    u_real **windows_fc_trils,
    u_real **windows_mean_fc, u_real **windows_ssd_fc,
    u_real **L_windows_mean_fc, u_real **L_windows_ssd_fc,
    u_real **R_windows_mean_fc, u_real **R_windows_ssd_fc,
    int N_SIMS, int n_pairs, int n_windows, int n_window_pairs, 
    int *window_pairs_i, int *window_pairs_j, int maxThreadsPerBlock,
    bool save_hemis, int n_pairs_hemi) {
        // get simulation index
        int sim_idx = blockIdx.x;
        if (sim_idx >= N_SIMS) return;
        // get window pair index
        int window_pair_idx = threadIdx.x + (maxThreadsPerBlock * blockIdx.y);
        if (window_pair_idx >= n_window_pairs) return;
        int w_i = window_pairs_i[window_pair_idx];
        int w_j = window_pairs_j[window_pair_idx];
        // get hemi
        int hemi = blockIdx.z;
        if (!save_hemis) {
            if (hemi > 0) return;
        } else {
            if (hemi > 2) return;
        }
        // calculate cov
        int pair_idx;
        u_real cov = 0;
        // pair_idx_start = 0;
        // pair_idx_end = n_pairs; // non-inclusive
        // if (hemi == 1) { // left
        //     pair_idx_end = n_pairs_hemi;
        // } else if (hemi == 2) { // right
        //     pair_idx_start = n_pairs - n_pairs_hemi;
        // }
        if (hemi == 0) {
            for (pair_idx=0; pair_idx<n_pairs; pair_idx++) {
                cov += 
                    (windows_fc_trils[sim_idx][w_i*n_pairs+pair_idx] - windows_mean_fc[sim_idx][w_i]) 
                    * (windows_fc_trils[sim_idx][w_j*n_pairs+pair_idx] - windows_mean_fc[sim_idx][w_j]);
            }
            fcd_trils[sim_idx][window_pair_idx] = cov / (windows_ssd_fc[sim_idx][w_i] * windows_ssd_fc[sim_idx][w_j]);
        } else if (hemi == 1) {
            for (pair_idx=0; pair_idx<n_pairs_hemi; pair_idx++) {
                cov += 
                    (windows_fc_trils[sim_idx][w_i*n_pairs+pair_idx] - L_windows_mean_fc[sim_idx][w_i]) 
                    * (windows_fc_trils[sim_idx][w_j*n_pairs+pair_idx] - L_windows_mean_fc[sim_idx][w_j]);
            }
            L_fcd_trils[sim_idx][window_pair_idx] = cov / (L_windows_ssd_fc[sim_idx][w_i] * L_windows_ssd_fc[sim_idx][w_j]);
        } else if (hemi == 2) {
            for (pair_idx=n_pairs-n_pairs_hemi; pair_idx<n_pairs; pair_idx++) {
                cov += 
                    (windows_fc_trils[sim_idx][w_i*n_pairs+pair_idx] - R_windows_mean_fc[sim_idx][w_i]) 
                    * (windows_fc_trils[sim_idx][w_j*n_pairs+pair_idx] - R_windows_mean_fc[sim_idx][w_j]);
            }
            R_fcd_trils[sim_idx][window_pair_idx] = cov / (R_windows_ssd_fc[sim_idx][w_i] * R_windows_ssd_fc[sim_idx][w_j]);
        }
    }

__global__ void float2double(double **dst, float **src, size_t rows, size_t cols) {
    int row = blockIdx.x;
    int col = blockIdx.y;
    if ((row > rows) | (col > cols)) return;
    dst[row][col] = (float)(src[row][col]);
}

cudaDeviceProp get_device_prop(int verbose = 1) {
    /*
        Gets GPU device properties and prints them to the console.
        Also exits the program if no GPU is found.
    */
    int count;
    cudaError_t error = cudaGetDeviceCount(&count);
    struct cudaDeviceProp prop;
    if (error == cudaSuccess) {
        int device = 0;
        CUDA_CHECK_RETURN( cudaGetDeviceProperties(&prop, device) );
        if (verbose > 0) {
            std::cout << std::endl << "CUDA device #" << device << ": " << prop.name << std::endl;
        }
        if (verbose > 1) {
            std::cout << "totalGlobalMem: " << prop.totalGlobalMem << ", sharedMemPerBlock: " << prop.sharedMemPerBlock; 
            std::cout << ", regsPerBlock: " << prop.regsPerBlock << ", warpSize: " << prop.warpSize << ", memPitch: " << prop.memPitch << std::endl;
            std::cout << "maxThreadsPerBlock: " << prop.maxThreadsPerBlock << ", maxThreadsDim[0]: " << prop.maxThreadsDim[0] 
                << ", maxThreadsDim[1]: " << prop.maxThreadsDim[1] << ", maxThreadsDim[2]: " << prop.maxThreadsDim[2] << std::endl; 
            std::cout << "maxGridSize[0]: " << prop.maxGridSize[0] << ", maxGridSize[1]: " << prop.maxGridSize[1] << ", maxGridSize[2]: " 
                << prop.maxGridSize[2] << ", totalConstMem: " << prop.totalConstMem << std::endl;
            std::cout << "major: " << prop.major << ", minor: " << prop.minor << ", clockRate: " << prop.clockRate << ", textureAlignment: " 
                << prop.textureAlignment << ", deviceOverlap: " << prop.deviceOverlap << ", multiProcessorCount: " << prop.multiProcessorCount << std::endl; 
            std::cout << "kernelExecTimeoutEnabled: " << prop.kernelExecTimeoutEnabled << ", integrated: " << prop.integrated  
                << ", canMapHostMemory: " << prop.canMapHostMemory << ", computeMode: " << prop.computeMode <<  std::endl; 
            std::cout << "concurrentKernels: " << prop.concurrentKernels << ", ECCEnabled: " << prop.ECCEnabled << ", pciBusID: " << prop.pciBusID
                << ", pciDeviceID: " << prop.pciDeviceID << ", tccDriver: " << prop.tccDriver  << std::endl;
        }
    } else {
        std::cout << "No CUDA devices was found\n" << std::endl;
        exit(1);
    }
    return prop;
}

template<typename Model>
void run_simulations_gpu(
    double * BOLD_out, double * fc_trils_out, double * fcd_trils_out,
    double * S_E_out, double * S_I_out,
    double * r_E_out, double * r_I_out,
    double * I_E_out, double * I_I_out,
    bool * fic_unstable_out, bool * fic_failed_out,
    u_real * G_list, u_real * w_EE_list, u_real * w_EI_list, u_real * w_IE_list, u_real * v_list,
    u_real * SC, gsl_matrix * SC_gsl, u_real * SC_dist, bool do_delay,
    int nodes, int time_steps, int BOLD_TR, int _max_fic_trials, int window_size,
    int N_SIMS, bool do_fic, bool only_wIE_free, bool extended_output,
    struct ModelConfigs conf
)
// TODO: clean the order of args
{
    Model* d_model;
    CUDA_CHECK_RETURN(cudaMallocManaged(&d_model, sizeof(Model)));
    new (d_model) Model();

    // copy SC and parameter lists to device
    CUDA_CHECK_RETURN(cudaMemcpy(d_SC, SC, nodes*nodes * sizeof(u_real), cudaMemcpyHostToDevice));
    // TODO: make the following model-specific
    CUDA_CHECK_RETURN(cudaMemcpy(d_global_params[0], G_list, N_SIMS * sizeof(u_real), cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy(d_regional_params[0], w_EE_list, N_SIMS*nodes * sizeof(u_real), cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy(d_regional_params[1], w_EI_list, N_SIMS*nodes * sizeof(u_real), cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy(d_regional_params[2], w_IE_list, N_SIMS*nodes * sizeof(u_real), cudaMemcpyHostToDevice));

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

    // // do fic if indicated
    // if (do_fic) {
    //     gsl_vector * curr_w_IE = gsl_vector_alloc(nodes);
    //     double *curr_w_EE = (double *)malloc(nodes * sizeof(double));
    //     double *curr_w_EI = (double *)malloc(nodes * sizeof(double));
    //     for (int sim_idx=0; sim_idx<N_SIMS; sim_idx++) {
    //         // make a copy of regional wEE and wEI
    //         for (int j=0; j<nodes; j++) {
    //             curr_w_EE[j] = (double)(w_EE_list[sim_idx*nodes+j]);
    //             curr_w_EI[j] = (double)(w_EI_list[sim_idx*nodes+j]);
    //         }
    //         // do FIC for the current particle
    //         fic_unstable[sim_idx] = false;
    //         analytical_fic_het(
    //             SC_gsl, G_list[sim_idx], curr_w_EE, curr_w_EI,
    //             curr_w_IE, fic_unstable+sim_idx);

    //         if (fic_unstable[sim_idx]) {
    //             printf("In simulation #%d FIC solution is unstable. Setting wIE to 1 in all nodes\n", sim_idx);
    //             for (int j=0; j<nodes; j++) {
    //                 w_IE_fic[sim_idx][j] = 1.0;
    //             }
    //         } else {
    //             // copy to w_IE_fic which will be passed on to the device
    //             for (int j=0; j<nodes; j++) {
    //                 w_IE_fic[sim_idx][j] = (u_real)gsl_vector_get(curr_w_IE, j);
    //             }
    //         }
    //     }
    // }

    // run simulations
    dim3 numBlocks(N_SIMS);
    dim3 threadsPerBlock(nodes);
    // (third argument is extern shared memory size for S_i_1_E)
    // provide NULL for extended output variables and FIC variables
    bool _extended_output = (extended_output | do_fic); // extended output is needed if requested by user or FIC is done
    #ifndef MANY_NODES
    size_t shared_mem_extern = nodes*sizeof(u_real);
    #else
    size_t shared_mem_extern = 0;
    #endif 
    bnm<Model><<<numBlocks,threadsPerBlock,shared_mem_extern>>>(
        BOLD, state_vars_out, 
        d_model,
        n_vols_remove,
        d_SC, d_global_params, d_regional_params,
        conn_state_var_hist, delay, max_delay,
        N_SIMS, nodes, BOLD_TR, time_steps, 
        noise, 
        // do_fic, w_IE_fic, 
        // adjust_fic, _max_fic_trials, fic_unstable, fic_failed, fic_n_trials,
        _extended_output,
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
    for (int sim_idx=0; sim_idx<N_SIMS; sim_idx++) {
        memcpy(BOLD_out, BOLD[sim_idx], sizeof(u_real) * bold_size);
        BOLD_out+=bold_size;
        memcpy(fc_trils_out, fc_trils[sim_idx], sizeof(u_real) * n_pairs);
        fc_trils_out+=n_pairs;
        memcpy(fcd_trils_out, fcd_trils[sim_idx], sizeof(u_real) * n_window_pairs);
        fcd_trils_out+=n_window_pairs;
        if (extended_output) {
            memcpy(I_E_out, state_vars_out[0][sim_idx], sizeof(u_real) * ext_out_size);
            I_E_out+=ext_out_size;
            memcpy(I_I_out, state_vars_out[1][sim_idx], sizeof(u_real) * ext_out_size);
            I_I_out+=ext_out_size;
            memcpy(r_E_out, state_vars_out[2][sim_idx], sizeof(u_real) * ext_out_size);
            r_E_out+=ext_out_size;
            memcpy(r_I_out, state_vars_out[3][sim_idx], sizeof(u_real) * ext_out_size);
            r_I_out+=ext_out_size;
            memcpy(S_E_out, state_vars_out[4][sim_idx], sizeof(u_real) * ext_out_size);
            S_E_out+=ext_out_size;
            memcpy(S_I_out, state_vars_out[5][sim_idx], sizeof(u_real) * ext_out_size);
            S_I_out+=ext_out_size;
        }
        // if (do_fic) {
        //     memcpy(w_IE_list, w_IE_fic[sim_idx], sizeof(u_real) * nodes);
        //     w_IE_list+=nodes;
        // }
    }
    // if (do_fic) {
    //     memcpy(fic_unstable_out, fic_unstable, sizeof(bool) * N_SIMS);
    //     memcpy(fic_failed_out, fic_failed, sizeof(bool) * N_SIMS);
    // }

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
        int N_SIMS, int nodes, bool do_fic, bool extended_output, int rand_seed,
        int BOLD_TR, int time_steps, int window_size, int window_step,
        struct BWConstants bwc, struct ModelConfigs conf, bool verbose
        )
    {
    // check CUDA device avaliability and properties
    prop = get_device_prop(verbose);

    // copy constants and configs from CPU
    CUDA_CHECK_RETURN(cudaMemcpyToSymbol(d_bwc, &bwc, sizeof(BWConstants)));
    CUDA_CHECK_RETURN(cudaMemcpyToSymbol(d_conf, &conf, sizeof(ModelConfigs)));

    // allocate device memory for SC
    CUDA_CHECK_RETURN(cudaMalloc(&d_SC, sizeof(u_real) * nodes*nodes));

    // allocate device memory for simulation parameters
    // size of global_params is (n_global_params, N_SIMS)
    // size of regional_params is (n_regional_params, N_SIMS * nodes)
    CUDA_CHECK_RETURN(cudaMallocManaged((void**)&d_global_params, sizeof(u_real*) * Model::n_global_params));
    CUDA_CHECK_RETURN(cudaMallocManaged((void**)&d_regional_params, sizeof(u_real*) * Model::n_regional_params));
    for (int param_idx=0; param_idx<Model::n_global_params; param_idx++) {
        CUDA_CHECK_RETURN(cudaMallocManaged((void**)&d_global_params[param_idx], sizeof(u_real) * N_SIMS));
    }
    for (int param_idx=0; param_idx<Model::n_regional_params; param_idx++) {
        CUDA_CHECK_RETURN(cudaMallocManaged((void**)&d_regional_params[param_idx], sizeof(u_real) * N_SIMS * nodes));
    }

    // allocated memory for BOLD time-series of all simulations
    // BOLD will be a 2D array of size N_SIMS x (nodes x output_ts)
    u_real TR        = (u_real)BOLD_TR / 1000; // (s) TR of fMRI data
    // output_ts = (time_steps / (TR / mc.bw_dt))+1; // Length of BOLD time-series written to HDD
    output_ts = (time_steps / (TR / 0.001))+1; // Length of BOLD time-series written to HDD
    size_t bold_size = nodes * output_ts;
    CUDA_CHECK_RETURN(cudaMallocManaged((void**)&BOLD, sizeof(u_real*) * N_SIMS));

    // allocate memory for extended output
    // TODO: add FIC for rWW
    bool _extended_output = extended_output;
    size_t ext_out_size = nodes;
    if (conf.extended_output_ts) {
        ext_out_size *= output_ts;
    }
    if (_extended_output) {
        CUDA_CHECK_RETURN(cudaMallocManaged((void**)&state_vars_out, sizeof(u_real**) * Model::n_state_vars));
        for (int var_idx=0; var_idx<Model::n_state_vars; var_idx++) {
            CUDA_CHECK_RETURN(cudaMallocManaged((void**)&state_vars_out[var_idx], sizeof(u_real*) * N_SIMS));
            for (int sim_idx=0; sim_idx<N_SIMS; sim_idx++) {
                CUDA_CHECK_RETURN(cudaMallocManaged((void**)&state_vars_out[var_idx][sim_idx], sizeof(u_real) * ext_out_size));
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
        assert((nodes % 2) == 0);
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


template void run_simulations_gpu<rWWModel>(
    double*, double*, double*, double*, double*, double*, double*, double*, double*,
    bool*, bool*, u_real*, u_real*, u_real*, u_real*, u_real*, u_real*, gsl_matrix*, u_real*, bool,
    int, int, int, int, int, int, bool, bool, bool, ModelConfigs
);

template void init_gpu<rWWModel>(
        int *output_ts_p, int *n_pairs_p, int *n_window_pairs_p,
        int N_SIMS, int nodes, bool do_fic, bool extended_output, int rand_seed,
        int BOLD_TR, int time_steps, int window_size, int window_step,
        struct BWConstants bwc, struct ModelConfigs conf, bool verbose
);
