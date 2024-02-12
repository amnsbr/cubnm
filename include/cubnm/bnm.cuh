#ifndef BNM_CUH
#define BNM_CUH
#include "bnm.hpp"
__device__ void calculateGlobalInput(
        u_real* tmp_globalinput, int* k_buff_idx,
        int* nodes, int* sim_idx, int* j, int *k, 
        int* buff_idx, int* max_delay, u_real* _SC, 
        int** delay, bool* has_delay, 
        u_real** conn_state_var_hist, u_real* conn_state_var_1
        );

template<typename Model>
__global__ void bnm(
    Model* model,
    u_real **BOLD,
    u_real ***state_vars_out, 
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
    );

template<typename Model>
void _run_simulations_gpu(
    double * BOLD_out, double * fc_trils_out, double * fcd_trils_out,
    u_real ** global_params, u_real ** regional_params, u_real * v_list,
    u_real * SC, u_real * SC_dist, BaseModel *model
);

template<typename Model>
void _init_gpu(BaseModel *model, BWConstants bwc);

#endif