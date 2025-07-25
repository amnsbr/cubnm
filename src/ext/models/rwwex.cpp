#include "cubnm/models/rwwex.hpp"

rWWExModel::Constants rWWExModel::mc;

void rWWExModel::init_constants(double dt) {
    mc.dt = dt; // Time-step of synaptic activity model (msec)
    mc.sqrt_dt = SQRT(mc.dt); 
    mc.J_N  = 0.2609; // (nA)
    mc.a = 270; // (n/C)
    mc.b = 108; // (Hz)
    mc.d = 0.154; // (s)
    mc.gamma = (double)0.641/(double)1000.0; // factor 1000 for expressing everything in ms
    mc.tau = 100; // (ms) Time constant of NMDA (excitatory)
    mc.itau = 1.0/mc.tau;
    mc.dt_itau = mc.dt * mc.itau;
    mc.dt_gamma = mc.dt * mc.gamma;
}

void rWWExModel::h_init(
    double* _state_vars, double* _intermediate_vars,
    double* _global_params, double* _regional_params,
    int* _ext_int, bool* _ext_bool,
    int* _ext_int_shared, bool* _ext_bool_shared
) {
    _state_vars[2] = 0.001; // S
}

void rWWExModel::_j_restart(
    double* _state_vars, double* _intermediate_vars,
    double* _global_params, double* _regional_params,
    int* _ext_int, bool* _ext_bool,
    int* _ext_int_shared, bool* _ext_bool_shared
) {
    _state_vars[2] = 0.001; // S
}

void rWWExModel::h_step(
        double* _state_vars, double* _intermediate_vars,
        double* _global_params, double* _regional_params,
        double& tmp_globalinput,
        double* noise, long& noise_idx
        ) {
    // x = w * J_N * S + G * J_N * tmp_globalinput + I0
    _state_vars[0] = _regional_params[0] * rWWExModel::mc.J_N * _state_vars[2] + _global_params[0] * rWWExModel::mc.J_N * tmp_globalinput + _regional_params[1] ; 
    // axb = a * x - b
    _intermediate_vars[0] = rWWExModel::mc.a * _state_vars[0] - rWWExModel::mc.b;
    // r = axb / (1 - exp(-d * axb))
    _state_vars[1] = _intermediate_vars[0] / (1 - EXP(-rWWExModel::mc.d * _intermediate_vars[0]));
    // dSdt = dt * ((gamma * (1 - S) * r) - (S / tau)) + sigma * sqrt(dt) * noise
    _intermediate_vars[1] = rWWExModel::mc.dt_gamma * ((1 - _state_vars[2]) * _state_vars[1]) - rWWExModel::mc.dt_itau * _state_vars[2] + noise[noise_idx] * rWWExModel::mc.sqrt_dt * _regional_params[2];
    // S += dSdt
    _state_vars[2] += _intermediate_vars[1];
    // clip S to 0-1
    _state_vars[2] = fmax(0.0f, fmin(1.0f, _state_vars[2]));
}