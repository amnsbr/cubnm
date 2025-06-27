#include "cubnm/includes.cuh"
#include "cubnm/defines.h"
#include "cubnm/models/jr.cuh"
__device__ __NOINLINE__ void JRModel::init(
    double* _state_vars, double* _intermediate_vars,
    double* _global_params, double* _regional_params,
    int* _ext_int, bool* _ext_bool,
    int* _ext_int_shared, bool* _ext_bool_shared
) {
    // based on 
    // https://github.com/vandal-uv/Nicotine-Whole-Brain/blob/main/JansenRitModel.py
    _state_vars[0] = 0.131; // x0
    _state_vars[1] = 0.171; // x1
    _state_vars[2] = 0.343; // x2
    _state_vars[3] = 0.21;  // x3
    _state_vars[4] = 3.07;  // y1
    _state_vars[5] = 2.96;  // y2
    _state_vars[6] = 25.36; // z1
    _state_vars[7] = 2.42;  // z2

}

__device__ __NOINLINE__ void JRModel::restart(
    double* _state_vars, double* _intermediate_vars, 
    double* _global_params, double* _regional_params,
    int* _ext_int, bool* _ext_bool,
    int* _ext_int_shared, bool* _ext_bool_shared
) {
    // call init
    JRModel::init(
        _state_vars, _intermediate_vars,
        _global_params, _regional_params, 
        _ext_int, _ext_bool, 
        _ext_int_shared, _ext_bool_shared
    );
}

__device__ __FORCEINLINE__ double d_JR_sigmoid(const double& v, const double& r) {
    // Sigmoid function
    return (2.0 * d_JRc.e0) / (1.0 + EXP(r * (d_JRc.v0 - v)));
}

__device__ void JRModel::step(
        double* _state_vars, double* _intermediate_vars,
        double* _global_params, double* _regional_params,
        double& tmp_globalinput,
        double* noise, long& noise_idx
        ) {
    /* 
    Based on Coronel-Oliveros et al. (2023)
    ---
    Look-up table:
    (these are not assigned as variables
     as I fear [but am not sure] it might
     use more registers, hence reducing 
     performance)
    G: _global_params[0]
    C: _regional_params[0]
    C1c: _regional_params[1]
    C2c: _regional_params[2]
    C3c: _regional_params[3]
    C4c: _regional_params[4]
    sigma: _regional_params[5]
    x0: _state_vars[0]
    x1: _state_vars[1]
    x2: _state_vars[2]
    x3: _state_vars[3]
    y0: _state_vars[4]
    y1: _state_vars[5]
    y2: _state_vars[6]
    y3: _state_vars[7]
    r_E: _state_vars[8]
    x0_dot: _intermediate_vars[0]
    y0_dot_sigmoid: _intermediate_vars[1]
    y0_dot: _intermediate_vars[2]
    x1_dot: _intermediate_vars[3]
    y1_dot_sigmoid: _intermediate_vars[4]
    y1_dot: _intermediate_vars[5]
    x2_dot: _intermediate_vars[6]
    y2_dot_sigmoid: _intermediate_vars[7]
    y2_dot: _intermediate_vars[8]
    x3_dot: _intermediate_vars[9]
    y3_dot_sigmoid: _intermediate_vars[10]
    y3_dot: _intermediate_vars[11]
    noise_term: _intermediate_vars[12]
    ---
    Variations in the equations:
    C1 = C1c * C
    C2 = C2c * C
    C3 = C3c * C
    C4 = C4c * C
    alpha = G
    */

    // Calculate derivatives
    // Eq1
    // x0_dot = y0
    _intermediate_vars[0] = _state_vars[4];
    // Eq2
    // y0_dot = A * a * (sigmoid(C2c * C * x1 - C4c * C * x2 + C * G * x3, r0)) - \
    //          2 * a * y0 - a**2 * x0
    // first calculating the term inside the sigmoid() function
    _intermediate_vars[1] = d_JR_sigmoid(
        _regional_params[2] * _regional_params[0] * _state_vars[1] - 
        _regional_params[4] * _regional_params[0] * _state_vars[2] + 
        _regional_params[0] * _global_params[0] * _state_vars[3],
        d_JRc.r0
    );
    // then calculating y0_dot
    _intermediate_vars[2] = 
        d_JRc.A * d_JRc.a * _intermediate_vars[1] - 
        2.0 * d_JRc.a * _state_vars[4] - 
        d_JRc.a2 * _state_vars[0];
    // Eq3
    // x1_dot = y1
    _intermediate_vars[3] = _state_vars[5];
    // Eq4
    // y1_dot = A * a * (p + sigmoid(C1c * C * x0, r1)) - \
    //          2 * a * y1 - a**2 * x1
    // first calculating the term inside the sigmoid() function
    _intermediate_vars[4] = d_JR_sigmoid(
        _regional_params[1] * _regional_params[0] * _state_vars[0],
        d_JRc.r1
    );
    // then calculating y1_dot
    _intermediate_vars[5] = 
        d_JRc.A * d_JRc.a * (d_JRc.p + _intermediate_vars[4]) - 
        2.0 * d_JRc.a * _state_vars[5] - 
        d_JRc.a2 * _state_vars[1];
    // Eq5
    // x2_dot = y2
    _intermediate_vars[6] = _state_vars[6];
    // Eq6
    // y2_dot = B * b * (sigmoid(C3c * C * x0, r2)) - \
    //          2 * b * y2 - b**2 * x2
    // first calculating the term inside the sigmoid() function
    _intermediate_vars[7] = d_JR_sigmoid(
        _regional_params[3] * _regional_params[0] * _state_vars[0],
        d_JRc.r2
    );
    // then calculating y2_dot
    _intermediate_vars[8] = 
        d_JRc.B * d_JRc.b * _intermediate_vars[7] - 
        2.0 * d_JRc.b * _state_vars[6] - 
        d_JRc.b2 * _state_vars[2];
    // Eq7
    // x3_dot = y3
    _intermediate_vars[9] = _state_vars[7];
    // Eq8
    // y3_dot = A * ad * (M / norm @ sigmoid(C2c * C * x1 - C4c * C * x2 + C * G * x3, r0)) - \
    //          2 * ad * y3 - ad**2 * x3
    // where M / norm is already calculated and is available as tmp_globalinput
    // first calculating the term inside the sigmoid() function
    _intermediate_vars[10] = d_JR_sigmoid(
        _regional_params[2] * _regional_params[0] * _state_vars[1] - 
        _regional_params[4] * _regional_params[0] * _state_vars[2] + 
        _regional_params[0] * _global_params[0] * _state_vars[3],
        d_JRc.r0
    );
    // then calculating y3_dot
    _intermediate_vars[11] = 
        d_JRc.A * d_JRc.ad * (tmp_globalinput * _intermediate_vars[10]) - 
        2.0 * d_JRc.ad * _state_vars[7] - 
        d_JRc.ad2 * _state_vars[3];

    // noise term to be injected into y1
    // noise_term = sqrt(dt) * A * a * sigma * noise[noise_idx]
    _intermediate_vars[12] = d_JRc.sqrt_dt * d_JRc.A * d_JRc.a * 
        _regional_params[5] * noise[noise_idx];

    // Integration step
    // x0
    _state_vars[0] += d_JRc.dt * _intermediate_vars[0];
    // x1
    _state_vars[1] += d_JRc.dt * _intermediate_vars[3];
    // x2
    _state_vars[2] += d_JRc.dt * _intermediate_vars[6];
    // x3
    _state_vars[3] += d_JRc.dt * _intermediate_vars[9];
    // y0
    _state_vars[4] += d_JRc.dt * _intermediate_vars[2];
    // y1 + noise
    _state_vars[5] += d_JRc.dt * _intermediate_vars[5] + 
        _intermediate_vars[12];
    // y2
    _state_vars[6] += d_JRc.dt * _intermediate_vars[8];
    // y3
    _state_vars[7] += d_JRc.dt * _intermediate_vars[11];

    // Calculate r_E
    // pyrm = C2c * C * x1 - C4c * x2 + C * G * x3
    // r_E = sigmoid(pyrm, r0)
    _state_vars[8] = 
        d_JR_sigmoid(
            _regional_params[2] * _regional_params[0] * _state_vars[1] - 
            _regional_params[4] * _regional_params[0] * _state_vars[2] + 
            _regional_params[0] * _global_params[0] * _state_vars[3],
            d_JRc.r0
        );
}