#ifndef BW_CUH
#define BW_CUH
#include "bw.hpp"
__constant__ BWConstants d_bwc;
__device__ void bw_step(
        double& bw_x, double& bw_f, double& bw_nu, 
        double& bw_q, double& tmp_f,
        double& n_state
        );
#endif