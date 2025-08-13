#include "cubnm/includes.cuh"
#include "cubnm/defines.h"
#include "cubnm/models/bw.cuh"
__device__ void bw_step(
        double& bw_x, double& bw_f, double& bw_nu, 
        double& bw_q, double& tmp_f,
        double& n_state
        ) {
    // Balloon-Windkessel model integration step
    bw_x  = bw_x  +  d_bwc.dt * (n_state - d_bwc.kappa * bw_x - d_bwc.y * (bw_f - 1.0));
    tmp_f = bw_f  +  d_bwc.dt * bw_x;
    bw_nu = bw_nu +  d_bwc.dt_itau * (bw_f - pow(bw_nu, d_bwc.ialpha));
    bw_q  = bw_q  +  d_bwc.dt_itau * (bw_f * (1.0 - pow(d_bwc.oneminrho,(1.0/ bw_f))) / d_bwc.rho  - pow(bw_nu,d_bwc.ialpha) * bw_q / bw_nu);
    bw_f  = tmp_f;
}
