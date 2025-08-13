#ifndef BNM_CUH
#define BNM_CUH
#include "bnm.hpp"

typedef void (*GlobalInputKernel)(
    double&, int&,
    const int&, const int&, const int&, 
    const int&, int&, int&, double**, 
    double**, const bool&, const int&,
    int&, const double&,
    double**, double*
);

__device__ void global_input_cond(
    double& tmp_globalinput, int& k_buff_idx,
    const int& nodes, const int& sim_idx, const int& SC_idx,
    const int& j, int& k, int& buff_idx, double** SC, 
    double** SC_dist, const bool& has_delay, const int& max_delay,
    int& curr_delay, const double& velocity,
    double** conn_state_var_hist, double* conn_state_var_1
);

__device__ void global_input_osc(
    double& tmp_globalinput, int& k_buff_idx,
    const int& nodes, const int& sim_idx, const int& SC_idx,
    const int& j, int& k, int& buff_idx, double** SC, 
    double** SC_dist, const bool& has_delay, const int& max_delay,
    int& curr_delay, const double& velocity,
    double** conn_state_var_hist, double* conn_state_var_1
);

cudaDeviceProp prop;

#endif