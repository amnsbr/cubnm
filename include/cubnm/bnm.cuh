#ifndef BNM_CUH
#define BNM_CUH
#include "bnm.hpp"

typedef void (*GlobalInputKernel)(
    u_real&, int&,
    const int&, const int&, const int&, 
    int&, int&, u_real*, 
    int**, const bool&, const int&,
    u_real**, u_real*
);

__device__ void global_input_cond(
    u_real& tmp_globalinput, int& k_buff_idx,
    const int& nodes, const int& sim_idx, const int& j, 
    int& k, int& buff_idx, u_real* _SC, 
    int** delay, const bool& has_delay, const int& max_delay,
    u_real** conn_state_var_hist, u_real* conn_state_var_1
);

__device__ void global_input_osc(
    u_real& tmp_globalinput, int& k_buff_idx,
    const int& nodes, const int& sim_idx, const int& j, 
    int& k, int& buff_idx, u_real* _SC, 
    int** delay, const bool& has_delay, const int& max_delay,
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

#endif