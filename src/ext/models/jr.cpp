#include "cubnm/models/jr.hpp"

JRModel::Constants JRModel::mc;

void JRModel::init_constants(double dt) {
    mc.dt = dt;
    mc.sqrt_dt = SQRT(mc.dt);
    mc.A = 3.25;
    mc.B = 22.0;
    mc.a = 0.1;
    mc.b = 0.05;
    mc.v0 = 6.0;
    mc.nu_max = 0.0025;
    mc.r = 0.56;
    mc.p_min = 0.12;
    mc.p_max = 0.32;
    mc.mu = 0.22;
    // sigmoidal coupling constants
    mc.cmin = 0.0;
    mc.cmax = 0.005;
    mc.midpoint = 6.0;
}

void JRModel::h_init(
    double* _state_vars, double* _intermediate_vars,
    double* _global_params, double* _regional_params,
    int* _ext_int, bool* _ext_bool,
    int* _ext_int_shared, bool* _ext_bool_shared
) {
    // not implemented yet
}

void JRModel::_j_restart(
    double* _state_vars, double* _intermediate_vars,
    double* _global_params, double* _regional_params,
    int* _ext_int, bool* _ext_bool,
    int* _ext_int_shared, bool* _ext_bool_shared
) {
    // not implemented yet
}

void JRModel::h_step(
        double* _state_vars, double* _intermediate_vars,
        double* _global_params, double* _regional_params,
        double& tmp_globalinput,
        double* noise, long& noise_idx
        ) {
    // not implemented yet
}