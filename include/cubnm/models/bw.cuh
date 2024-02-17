#ifndef BW_CUH
#define BW_CUH
#include "bw.hpp"
__constant__ BWConstants d_bwc;
__device__ void bw_step(
        u_real& bw_x, u_real& bw_f, u_real& bw_nu, 
        u_real& bw_q, u_real& tmp_f,
        u_real& n_state
        );
#endif