#include "cubnm/includes.cuh"
#include "cubnm/defines.h"
#include "cubnm/models/kuramoto.cuh"
__device__ __NOINLINE__ void KuramotoModel::init(
    u_real* _state_vars, u_real* _intermediate_vars, 
    u_real* _global_params, u_real* _regional_params,
    int* _ext_int, bool* _ext_bool,
    int* _ext_int_shared, bool* _ext_bool_shared
) {
    _state_vars[0] = _regional_params[0]; // theta set to initial theta
}

__device__ __NOINLINE__ void KuramotoModel::restart(
    u_real* _state_vars, u_real* _intermediate_vars, 
    u_real* _global_params, u_real* _regional_params,
    int* _ext_int, bool* _ext_bool,
    int* _ext_int_shared, bool* _ext_bool_shared
) {
    _state_vars[0] = _regional_params[0]; // theta set to initial theta
}

__device__ void KuramotoModel::step(
        u_real* _state_vars, u_real* _intermediate_vars,
        u_real* _global_params, u_real* _regional_params,
        u_real& tmp_globalinput,
        u_real* noise, long& noise_idx
        ) {
    // dtheta/dt = omega + G * global_input
    _intermediate_vars[0] = _regional_params[1] + _global_params[0] * tmp_globalinput;
    // theta = theta + dt * (dtheta/dt) + sqrt_dt * noise[noise_idx] * sigma
    _state_vars[0] = 
        _state_vars[0] + 
        d_Kuramotoc.dt * _intermediate_vars[0] +
        d_Kuramotoc.sqrt_dt * noise[noise_idx] * _regional_params[2];
    // phase reset
    _state_vars[0] = FMOD(_state_vars[0], d_Kuramotoc.twopi);
}