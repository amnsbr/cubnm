#include "cubnm/includes.cuh"
#include "cubnm/defines.h"
#include "cubnm/models/jr.cuh"
__device__ __NOINLINE__ void JRModel::init(
    double* _state_vars, double* _intermediate_vars,
    double* _global_params, double* _regional_params,
    int* _ext_int, bool* _ext_bool,
    int* _ext_int_shared, bool* _ext_bool_shared
) {
    // set all to 0.0
    for (int i = 0; i < JRModel::n_state_vars; ++i) {
        _state_vars[i] = 0.0;
    }
}

__device__ __NOINLINE__ void JRModel::restart(
    double* _state_vars, double* _intermediate_vars, 
    double* _global_params, double* _regional_params,
    int* _ext_int, bool* _ext_bool,
    int* _ext_int_shared, bool* _ext_bool_shared
) {
    // set all to 0.0
    for (int i = 0; i < JRModel::n_state_vars; ++i) {
        _state_vars[i] = 0.0;
    }
}

__device__ void JRModel::step(
        double* _state_vars, double* _intermediate_vars,
        double* _global_params, double* _regional_params,
        double& tmp_globalinput,
        double* noise, long& noise_idx
        ) {
    /* 
    Based on TVB's JansenRit._numpy_dfun()
    ---
    Look-up table:
    (these are not assigned as variables
     as I fear [but am not sure] it might
     use more registers, hence reducing 
     performance)
    G: _global_params[0]
    J: _regional_params[0]
    a_1: _regional_params[1]
    a_2: _regional_params[2]
    a_3: _regional_params[3]
    a_4: _regional_params[4]
    sigma: _regional_params[5]
    y0: _state_vars[0]
    y1: _state_vars[1]
    y2: _state_vars[2]
    y3: _state_vars[3]
    y4: _state_vars[4]
    y5: _state_vars[5]
    y1_y2: _state_vars[6] (aka PSP)
    s_y1_y2: _state_vars[7] (only needed for connectivity)
    */
    // Calculate sigmoid terms
    // sigm_y1_y2 = 2.0 * self.nu_max / (1.0 + exp(self.r * (self.v0 - (y1 - y2))))
    _intermediate_vars[0] = 
        2.0 * d_JRc.nu_max /
        (1.0 + EXP(
            d_JRc.r * (d_JRc.v0 - (_state_vars[1] - _state_vars[2]))
        ));
    // sigm_y0_1  = 2.0 * self.nu_max / (1.0 + exp(self.r * (self.v0 - (self.a_1 * self.J * y0))))
    _intermediate_vars[1] = 
        2.0 * d_JRc.nu_max /
        (1.0 + EXP(
            d_JRc.r * (d_JRc.v0 - (_regional_params[1] * _regional_params[0] * _state_vars[0]))
        ));
    // sigm_y0_3  = 2.0 * self.nu_max / (1.0 + exp(self.r * (self.v0 - (self.a_3 * self.J * y0))))
    _intermediate_vars[2] = 
        2.0 * d_JRc.nu_max /
        (1.0 + EXP(
            d_JRc.r * (d_JRc.v0 - (_regional_params[3] * _regional_params[0] * _state_vars[0]))
        ));

    // Calculate derivatives (dX/dt)
    // y0_dot = y3
    _intermediate_vars[3] = _state_vars[3];
    // y1_dot = y4
    _intermediate_vars[4] = _state_vars[4];
    // y2_dot = y5
    _intermediate_vars[5] = _state_vars[5];
    // y3_dot = self.A * self.a * sigm_y1_y2 - 2.0 * self.a * y3 - self.a ** 2 * y0
    _intermediate_vars[6] = 
        d_JRc.A * d_JRc.a * _intermediate_vars[0] - 
        2.0 * d_JRc.a * _state_vars[3] - 
        d_JRc.a * d_JRc.a * _state_vars[0];
    // y4_dot = self.A * self.a * (self.mu + self.a_2 * self.J * sigm_y0_1 + lrc + short_range_coupling)
    //            - 2.0 * self.a * y4 - self.a ** 2 * y1
    _intermediate_vars[7] = 
        d_JRc.A * d_JRc.a * (
            d_JRc.mu + 
            _regional_params[2] * _regional_params[0] * _intermediate_vars[1] + 
            tmp_globalinput
        ) - 
        2.0 * d_JRc.a * _state_vars[4] - 
        d_JRc.a * d_JRc.a * _state_vars[1];
    // y5_dot = self.B * self.b * (self.a_4 * self.J * sigm_y0_3) - 2.0 * self.b * y5 - self.b ** 2 * y2
    _intermediate_vars[8] = 
        d_JRc.B * d_JRc.b * (_regional_params[4] * _regional_params[0] * _intermediate_vars[2]) - 
        2.0 * d_JRc.b * _state_vars[5] - 
        d_JRc.b * d_JRc.b * _state_vars[2];

    // Integration step (X = X + dX/dt * dt [+ noise_term])
    // y0
    _state_vars[0] += d_JRc.dt * _intermediate_vars[3];
    // y1
    _state_vars[1] += d_JRc.dt * _intermediate_vars[4];
    // y2
    _state_vars[2] += d_JRc.dt * _intermediate_vars[5];
    // y3 + sigma * noise * sqrt(dt)
    _state_vars[3] += d_JRc.dt * _intermediate_vars[6] + 
        d_JRc.sqrt_dt * _regional_params[5] * noise[noise_idx];
    // y4
    _state_vars[4] += d_JRc.dt * _intermediate_vars[7];
    // y5
    _state_vars[5] += d_JRc.dt * _intermediate_vars[8];

    // Calculate y1_y2 and s(y1_y2) which
    // are the non-integrated state variables
    // needed for BOLD and connectivity
    // y1_y2 = y1 - y2
    _state_vars[6] = _state_vars[1] - _state_vars[2];
    // s_y1_y2 is the sigmoid of y1_y2 which will be the
    // connectivity state variable, i.e., the output to
    // the other connected nodes. This is calculated based
    // on SigmoidalJansenRit.pre() method in TVB
    // s_y1_y2 = self.cmin +
    //         (self.cmax - self.cmin) / (1.0 + numpy.exp(self.r * (self.midpoint - (y1 - y2))))
    _state_vars[7] = d_JRc.cmin + 
        (d_JRc.cmax - d_JRc.cmin) / 
        (1.0 + EXP(d_JRc.r * (d_JRc.midpoint - _state_vars[6])));
}