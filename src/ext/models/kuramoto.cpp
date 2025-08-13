#include "cubnm/models/kuramoto.hpp"

KuramotoModel::Constants KuramotoModel::mc;

void KuramotoModel::init_constants(double dt) {
    mc.dt = dt; // Time-step (msec)
    mc.sqrt_dt = SQRT(mc.dt); 
    mc.twopi = 2 * PI;
}

void KuramotoModel::h_init(
    double* _state_vars, double* _intermediate_vars,
    double* _global_params, double* _regional_params,
    int* _ext_int, bool* _ext_bool,
    int* _ext_int_shared, bool* _ext_bool_shared
) {
    _state_vars[0] = _regional_params[0]; // theta set to initial theta
}

void KuramotoModel::_j_restart(
    double* _state_vars, double* _intermediate_vars,
    double* _global_params, double* _regional_params,
    int* _ext_int, bool* _ext_bool,
    int* _ext_int_shared, bool* _ext_bool_shared
) {
    _state_vars[0] = _regional_params[0]; // theta set to initial theta
}

void KuramotoModel::h_step(
        double* _state_vars, double* _intermediate_vars,
        double* _global_params, double* _regional_params,
        double& tmp_globalinput,
        double* noise, long& noise_idx
        ) {
    // dtheta/dt = omega + G * global_input
    _intermediate_vars[0] = _regional_params[1] + _global_params[0] * tmp_globalinput;
    // theta = theta + dt * (omega + G * global_input) + sqrt_dt * noise[noise_idx] * sigma
    _state_vars[0] = 
        _state_vars[0] + 
        KuramotoModel::mc.dt * _intermediate_vars[0] +
        KuramotoModel::mc.sqrt_dt * noise[noise_idx] * _regional_params[2];
    // phase reset
    _state_vars[0] = FMOD(_state_vars[0], KuramotoModel::mc.twopi);
}