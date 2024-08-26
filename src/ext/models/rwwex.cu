#include "cubnm/includes.cuh"
#include "cubnm/defines.h"
#include "cubnm/models/rwwex.cuh"
__device__ __NOINLINE__ void rWWExModel::init(
    u_real* _state_vars, u_real* _intermediate_vars,
    u_real* _global_params, u_real* _regional_params,
    int* _ext_int, bool* _ext_bool,
    int* _ext_int_shared, bool* _ext_bool_shared
) {
    _state_vars[2] = 0.001; // S
}

__device__ __NOINLINE__ void rWWExModel::restart(
    u_real* _state_vars, u_real* _intermediate_vars, 
    u_real* _global_params, u_real* _regional_params,
    int* _ext_int, bool* _ext_bool,
    int* _ext_int_shared, bool* _ext_bool_shared
) {
    _state_vars[2] = 0.001; // S
}

__device__ void rWWExModel::step(
        u_real* _state_vars, u_real* _intermediate_vars,
        u_real* _global_params, u_real* _regional_params,
        u_real& tmp_globalinput,
        u_real* noise, long& noise_idx
        ) {
    // x = w * J_N * S + G * J_N * tmp_globalinput + I0
    _state_vars[0] = _regional_params[0] * d_rWWExc.J_N * _state_vars[2] + _global_params[0] * d_rWWExc.J_N * tmp_globalinput + _regional_params[1] ; 
    // axb = a * x - b
    _intermediate_vars[0] = d_rWWExc.a * _state_vars[0] - d_rWWExc.b;
    #ifdef USE_FLOATS
    // to avoid firing rate approaching infinity near I = b/a
    if (abs(_intermediate_vars[0]) < 1e-4) _intermediate_vars[0] = 1e-4;
    #endif
    // r = axb / (1 - exp(-d * axb))
    _state_vars[1] = _intermediate_vars[0] / (1 - EXP(-d_rWWExc.d * _intermediate_vars[0]));
    // dSdt = dt * ((gamma * (1 - S) * r) - (S / tau)) + sigma * sqrt(dt) * noise
    _intermediate_vars[1] = d_rWWExc.dt_gamma * ((1 - _state_vars[2]) * _state_vars[1]) - d_rWWExc.dt_itau * _state_vars[2] + noise[noise_idx] * d_rWWExc.sqrt_dt * _regional_params[2];
    // S += dSdt
    _state_vars[2] += _intermediate_vars[1];
    // clip S to 0-1
    _state_vars[2] = max(0.0f, min(1.0f, _state_vars[2]));
}