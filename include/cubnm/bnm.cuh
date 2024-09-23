#ifndef BNM_CUH
#define BNM_CUH
#include "bnm.hpp"

typedef void (*GlobalInputKernel)(
    u_real&, int&,
    const int&, const int&, const int&, 
    const int&, int&, int&, u_real**, 
    int**, const bool&, const int&,
    u_real**, u_real*
);

__device__ void global_input_cond(
    u_real& tmp_globalinput, int& k_buff_idx,
    const int& nodes, const int& sim_idx, const int& SC_idx,
    const int& j, int& k, int& buff_idx, u_real** SC, 
    int** delay, const bool& has_delay, const int& max_delay,
    u_real** conn_state_var_hist, u_real* conn_state_var_1
);

__device__ void global_input_osc(
    u_real& tmp_globalinput, int& k_buff_idx,
    const int& nodes, const int& sim_idx, const int& SC_idx,
    const int& j, int& k, int& buff_idx, u_real** SC, 
    int** delay, const bool& has_delay, const int& max_delay,
    u_real** conn_state_var_hist, u_real* conn_state_var_1
);

cudaDeviceProp prop;

#endif