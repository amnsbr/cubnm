#include "cubnm/defines.h"
#include "cubnm/models/bw.cuh"
__constant__ BWConstants d_bwc;
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
