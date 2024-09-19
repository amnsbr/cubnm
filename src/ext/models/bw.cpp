#include "cubnm/models/bw.hpp"
BWConstants bwc;

void init_bw_constants(BWConstants* bwc, u_real dt) {
    bwc->dt = dt; // Time-step of Balloon-Windkessel model (s)
    bwc->rho = 0.34;
    bwc->alpha = 0.32;
    bwc->tau = 0.98;
    bwc->y = 1.0/0.41;
    bwc->kappa = 1.0/0.65;
    bwc->V_0 = 0.02 * 100; // Resting blood volume fraction (as %) 
    bwc->k1 = 7 * bwc->rho;
    bwc->k2 = 2.0;
    bwc->k3 = 2 * bwc->rho - 0.2;
    bwc->ialpha = 1.0/bwc->alpha; // some pre-calculations
    bwc->itau = 1.0/bwc->tau;
    bwc->oneminrho = (1.0 - bwc->rho);
    bwc->dt_itau = bwc->dt * bwc->itau;
    bwc->V_0_k1 = bwc->V_0 * bwc->k1;
    bwc->V_0_k2 = bwc->V_0 * bwc->k2;
    bwc->V_0_k3 = bwc->V_0 * bwc->k3;
}

void h_bw_step(
        u_real& bw_x, u_real& bw_f, u_real& bw_nu, 
        u_real& bw_q, u_real& tmp_f,
        u_real& n_state
        ) {
    // Balloon-Windkessel model integration step
    bw_x  = bw_x  +  bwc.dt * (n_state - bwc.kappa * bw_x - bwc.y * (bw_f - 1.0));
    tmp_f = bw_f  +  bwc.dt * bw_x;
    bw_nu = bw_nu +  bwc.dt_itau * (bw_f - pow(bw_nu, bwc.ialpha));
    bw_q  = bw_q  +  bwc.dt_itau * (bw_f * (1.0 - pow(bwc.oneminrho,(1.0/ bw_f))) / bwc.rho  - pow(bw_nu,bwc.ialpha) * bw_q / bw_nu);
    bw_f  = tmp_f;
}