#include "cubnm/models/rjr.hpp"

rJRModel::Constants rJRModel::mc;

void rJRModel::init_constants() {
    mc.dt = 0.0001;
    mc.sqrt_dt = SQRT(mc.dt); 
    mc.Ke  = 100.0;
    mc.Ki = 50.0;
    mc.He = 3.25;
    mc.Hi = 22.0;
    mc.Fe = 100.0;
    mc.Fi = 50.0;
    mc.Re = 0.56;
    mc.Ri = 0.56;
    mc.V50e = 6.0;
    mc.V50i = 6.0;
    mc.Dr = 1.0;
    mc.V0 = 0.0;
    mc.HeKe = mc.He * mc.Ke;
    mc.HiKi = mc.Hi * mc.Ki;
    mc.DrKe2 = mc.Dr * mc.Ke * 2;
    mc.DrKi2 = mc.Dr * mc.Ki * 2;
    mc.Ke_sq = mc.Ke * mc.Ke;
    mc.Ki_sq = mc.Ki * mc.Ki;
}

void rJRModel::h_init(
    u_real* _state_vars, u_real* _intermediate_vars,
    u_real* _global_params, u_real* _regional_params,
    int* _ext_int, bool* _ext_bool,
    int* _ext_int_shared, bool* _ext_bool_shared
) {
    _state_vars[0] = rJRModel::mc.V0;
    _state_vars[1] = rJRModel::mc.V0;
    _state_vars[2] = 0.0;
    _state_vars[3] = 0.0;
    _state_vars[4] = rJRModel::mc.V0;
}

void rJRModel::_j_restart(
    u_real* _state_vars, u_real* _intermediate_vars,
    u_real* _global_params, u_real* _regional_params,
    int* _ext_int, bool* _ext_bool,
    int* _ext_int_shared, bool* _ext_bool_shared
) {
    _state_vars[0] = rJRModel::mc.V0;
    _state_vars[1] = rJRModel::mc.V0;
    _state_vars[2] = 0.0;
    _state_vars[3] = 0.0;
    _state_vars[4] = rJRModel::mc.V0;
}

void rJRModel::h_step(
        u_real* _state_vars, u_real* _intermediate_vars,
        u_real* _global_params, u_real* _regional_params,
        u_real& tmp_globalinput,
        u_real* noise, long& noise_idx
        ) {
    // d2EPSP/dt2 = He * Ke * R^2 * Fe / (1.0 + exp(Re * (V50e - (x))) -  Dr * Ke * 2 * R * EPSC - Ke^2 * R^2 * EPSP
    // where x = G * tmp_globalinput - C_EI * IPSP
    _intermediate_vars[0] = 
        rJRModel::mc.HeKe * POW(_regional_params[2], 2) 
        * (
            rJRModel::mc.Fe / (1.0 + EXP(rJRModel::mc.Re * (rJRModel::mc.V50e - (
            _global_params[0] * tmp_globalinput - _regional_params[0] * _state_vars[1]
            ))))
        ) 
        - rJRModel::mc.DrKe2 * _regional_params[2] * _state_vars[2]
        - rJRModel::mc.Ke_sq * POW(_regional_params[2], 2) * _state_vars[0];
    // d2IPSP/dt2 = Hi * Ki * R^2 * (Fi / (1.0 + exp(Ri * (V50i - (x)))) -  Dr * Ki * 2 * R * IPSC - Ki^2 * R^2 * IPSP
    // where x = C_IE * EPSP
    _intermediate_vars[1] = 
    rJRModel::mc.HiKi * POW(_regional_params[2], 2) 
        * (
            rJRModel::mc.Fi / (1.0 + EXP(rJRModel::mc.Ri * (rJRModel::mc.V50i - (
                _regional_params[1] * _state_vars[0]
            ))))
        )  
        - rJRModel::mc.DrKi2 * _regional_params[2] * _state_vars[3]
        - rJRModel::mc.Ki_sq * POW(_regional_params[2], 2) * _state_vars[1];
    // EPSP = EPSP + dEPSP/dt*dt
    // where dEPSP/dt = EPSC (from the previous time point)
    _state_vars[0] += _state_vars[2] * rJRModel::mc.dt;
    // IPSP = IPSP + dIPSP/dt*dt
    // where dIPSP/dt = IPSC (from the previous time point)
    _state_vars[1] += _state_vars[3] * rJRModel::mc.dt;
    // EPSC = EPSC + d2EPSP/dt2*dt
    // where d2EPSP/dt2 is _intermediate_vars[0]
    _state_vars[2] += _intermediate_vars[0] * rJRModel::mc.dt;
    // IPSC = IPSC + d2IPSP/dt2*dt
    // where d2IPSP/dt2 is _intermediate_vars[1]
    _state_vars[3] += _intermediate_vars[1] * rJRModel::mc.dt;
    // adding noise
    // TODO: make it possible to create uniformly distributed noise
    // EPSC = sqrt(dt) * sigma * R**2 * noise
    _state_vars[2] += rJRModel::mc.sqrt_dt * _regional_params[3] * POW(_regional_params[2], 2) * noise[noise_idx];
    // IPSC = sqrt(dt) * sigma * R**2 * noise
    _state_vars[3] += rJRModel::mc.sqrt_dt * _regional_params[3] * POW(_regional_params[2], 2) * noise[noise_idx];
    // EPSP_norm = EPSP / He
    _state_vars[4] = _state_vars[0] / rJRModel::mc.He;
}