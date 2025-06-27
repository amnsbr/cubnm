#include "cubnm/models/jr.hpp"

JRModel::Constants JRModel::mc;

void JRModel::init_constants(double dt) {
    // Note that the time step passed from the core is in milliseconds,
    // but in this model we need it in seconds.
    mc.dt = dt / (double)1000.0;
    mc.sqrt_dt = SQRT(mc.dt); 
    mc.a = 100.0;
    mc.ad = 50.0;
    mc.b = 50.0;
    mc.p = 5.4;
    mc.A = 3.25;
    mc.B = 22.0;
    mc.e0 = 2.5;
    mc.v0 = 6.0;
    mc.r0 = 0.56;
    mc.r1 = 0.56;
    mc.r2 = 0.56;
    // derived constants
    mc.a2 = mc.a * mc.a;
    mc.b2 = mc.b * mc.b;
    mc.ad2 = mc.ad * mc.ad;
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