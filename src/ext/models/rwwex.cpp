#include "cubnm/models/rwwex.hpp"

rWWExModel::Constants rWWExModel::mc;

void rWWExModel::init_constants() {
    mc.dt = 0.1; // Time-step of synaptic activity model (msec)
    mc.sqrt_dt = SQRT(mc.dt); 
    mc.J_N  = 0.2609; // (nA)
    mc.a = 270; // (n/C)
    mc.b = 108; // (Hz)
    mc.d = 0.154; // (s)
    mc.gamma = (u_real)0.641/(u_real)1000.0; // factor 1000 for expressing everything in ms
    mc.tau = 100; // (ms) Time constant of NMDA (excitatory)
    mc.itau = 1.0/mc.tau;
    mc.dt_itau = mc.dt * mc.itau;
    mc.dt_gamma = mc.dt * mc.gamma;
}