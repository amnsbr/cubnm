#include "cubnm/models/bw.hpp"
BWConstants bwc;

void init_bw_constants(BWConstants* bwc) {
    bwc->dt = 0.001; // Time-step of Balloon-Windkessel model (s)
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