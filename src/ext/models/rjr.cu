#include "cubnm/includes.cuh"
#include "cubnm/defines.h"
#include "cubnm/models/rjr.cuh"
__device__ __NOINLINE__ void rJRModel::init(
    u_real* _state_vars, u_real* _intermediate_vars, 
    u_real* _global_params, u_real* _regional_params,
    int* _ext_int, bool* _ext_bool,
    int* _ext_int_shared, bool* _ext_bool_shared
) {
    _state_vars[0] = d_rJRc.V0;
    _state_vars[1] = d_rJRc.V0;
    _state_vars[2] = 0.0;
    _state_vars[3] = 0.0;
    _state_vars[4] = d_rJRc.V0;

}

__device__ __NOINLINE__ void rJRModel::restart(
    u_real* _state_vars, u_real* _intermediate_vars, 
    u_real* _global_params, u_real* _regional_params,
    int* _ext_int, bool* _ext_bool,
    int* _ext_int_shared, bool* _ext_bool_shared
) {
    _state_vars[0] = d_rJRc.V0;
    _state_vars[1] = d_rJRc.V0;
    _state_vars[2] = 0.0;
    _state_vars[3] = 0.0;
    _state_vars[4] = d_rJRc.V0;
}

__device__ void rJRModel::step(
        u_real* _state_vars, u_real* _intermediate_vars,
        u_real* _global_params, u_real* _regional_params,
        u_real& tmp_globalinput,
        u_real* noise, long& noise_idx
        ) {
    // d2EPSP/dt2 = He * Ke * R^2 * Fe / (1.0 + exp(Re * (V50e - (x))) -  Dr * Ke * 2 * R * EPSC - Ke^2 * R^2 * EPSP
    // where x = G * tmp_globalinput - C_EI * IPSP
    _intermediate_vars[0] = 
        (
            d_rJRc.HeKe * POW(_regional_params[2], 2) 
            * (
                d_rJRc.Fe / (
                    1.0 + EXP(d_rJRc.Re * (
                        d_rJRc.V50e - (
                            _global_params[0] * tmp_globalinput - _regional_params[0] * _state_vars[1]
                        )
                    ))
                )
            )
        )
        - d_rJRc.DrKe2 * _regional_params[2] * _state_vars[2]
        - d_rJRc.Ke_sq * POW(_regional_params[2], 2) * _state_vars[0];
    // d2IPSP/dt2 = Hi * Ki * R^2 * (Fi / (1.0 + exp(Ri * (V50i - (x)))) -  Dr * Ki * 2 * R * IPSC - Ki^2 * R^2 * IPSP
    // where x = C_IE * EPSP
    _intermediate_vars[1] = 
        (    
            d_rJRc.HiKi * POW(_regional_params[2], 2) 
            * (
                d_rJRc.Fi / (
                    1.0 + EXP(d_rJRc.Ri * (
                        d_rJRc.V50i - (
                            _regional_params[1] * _state_vars[0]
                        )
                    ))
                )
            )
        )
        - d_rJRc.DrKi2 * _regional_params[2] * _state_vars[3]
        - d_rJRc.Ki_sq * POW(_regional_params[2], 2) * _state_vars[1];
    // EPSP = EPSP + dEPSP/dt*dt
    // where dEPSP/dt = EPSC (from the previous time point)
    _state_vars[0] += _state_vars[2] * d_rJRc.dt;
    // IPSP = IPSP + dIPSP/dt*dt
    // where dIPSP/dt = IPSC (from the previous time point)
    _state_vars[1] += _state_vars[3] * d_rJRc.dt;
    // EPSC += d2EPSP/dt2 * dt
    // where d2EPSP/dt2 is _intermediate_vars[0]
    _state_vars[2] += _intermediate_vars[0] * d_rJRc.dt;
    // IPSC += d2IPSP/dt2 * dt
    // where d2IPSP/dt2 is _intermediate_vars[1]
    _state_vars[3] += _intermediate_vars[1] * d_rJRc.dt;
    // adding noise
    // TODO: make it possible to create uniformly distributed noise
    // EPSC += sqrt(dt) * sigma * R**2 * noise
    _state_vars[2] += d_rJRc.sqrt_dt * _regional_params[3] * POW(_regional_params[2], 2) * noise[noise_idx];
    // IPSC += sqrt(dt) * sigma * R**2 * noise
    _state_vars[3] += d_rJRc.sqrt_dt * _regional_params[3] * POW(_regional_params[2], 2) * noise[noise_idx];
    // calculate normalized EPSP for BOLD calculation as
    // EPSP_norm = EPSP / He
    _state_vars[4] = _state_vars[0] / d_rJRc.He;
}