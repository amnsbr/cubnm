#include "cubnm/includes.cuh"
#include "cubnm/defines.h"
#include "cubnm/models/rww.cuh"
__device__ __NOINLINE__ void rWWModel::init(
    double* _state_vars, double* _intermediate_vars, 
    double* _global_params, double* _regional_params,
    int* _ext_int, bool* _ext_bool,
    int* _ext_int_shared, bool* _ext_bool_shared
) {
    // Note that rather than harcoding the variable
    // indices it is also possible to do indexing via
    // strings, but that will be less efficient
    _state_vars[4] = 0.001; // S_E
    _state_vars[5] = 0.001; // S_I
    // numerical FIC initializations
    _intermediate_vars[4] = 0.0; // mean_I_E
    _intermediate_vars[5] = this->conf.init_delta; // delta
    _ext_int_shared[0] = 0; // fic_trial
    _ext_bool_shared[0] = this->conf.do_fic & (this->conf.max_fic_trials > 0); // _adjust_fic in current sim
    _ext_bool_shared[1] = false; // fic_failed
}

__device__ __NOINLINE__ void rWWModel::restart(
    double* _state_vars, double* _intermediate_vars, 
    double* _global_params, double* _regional_params,
    int* _ext_int, bool* _ext_bool,
    int* _ext_int_shared, bool* _ext_bool_shared
) {
    // this is different from init in that it doesn't
    // reset the numerical FIC variables delta, fic_trial
    // and _adjust_fic
    _state_vars[4] = 0.001; // S_E
    _state_vars[5] = 0.001; // S_I
    // numerical FIC reset
    _intermediate_vars[4] = 0.0; // mean_I_E
}

__device__ void rWWModel::step(
        double* _state_vars, double* _intermediate_vars,
        double* _global_params, double* _regional_params,
        double& tmp_globalinput,
        double* noise, long& noise_idx
        ) {
    // I_E = w_E * I_0 + (w_p * J_N) * S_E + global_input * G * J_N - w_IE * S_I
    _state_vars[0] =
        d_rWWc.w_E__I_0 
        + _regional_params[0] * _regional_params[1] * _state_vars[4] 
        + tmp_globalinput * _global_params[0] * _regional_params[1] 
        - _regional_params[2] * _state_vars[5];
    // I_I = w_I * I_0 + J_N * S_E - w_II * S_I
    _state_vars[1] = 
        d_rWWc.w_I__I_0 
        + _regional_params[1] * _state_vars[4] 
        - d_rWWc.w_II * _state_vars[5];
    // aIb_E = a_E * I_E - b_E
    _intermediate_vars[0] = d_rWWc.a_E * _state_vars[0] - d_rWWc.b_E;
    // aIb_I = a_I * I_I - b_I
    _intermediate_vars[1] = d_rWWc.a_I * _state_vars[1] - d_rWWc.b_I;
    // r_E = aIb_E / (1 - exp(-d_E * aIb_E))
    _state_vars[2] = _intermediate_vars[0] / (1 - EXP(-d_rWWc.d_E * _intermediate_vars[0]));
    // r_I = aIb_I / (1 - exp(-d_I * aIb_I))
    _state_vars[3] = _intermediate_vars[1] / (1 - EXP(-d_rWWc.d_I * _intermediate_vars[1]));
    // dS_E = noise * sigma * sqrt(dt) + dt * gamma_E * ((1 - S_E) * (r_E)) - dt * itau_E * S_E;
    _intermediate_vars[2] = 
        noise[noise_idx] * _regional_params[3] * d_rWWc.sqrt_dt 
        + d_rWWc.dt_gamma_E * ((1 - _state_vars[4]) * _state_vars[2]) 
        - d_rWWc.dt_itau_E * _state_vars[4];
    // dS_I = noise * sigma * sqrt(dt) + dt * gamma_I * r_I - dt * itau_I * S_I;
    _intermediate_vars[3] = 
        noise[noise_idx+1] * _regional_params[3] * d_rWWc.sqrt_dt 
        + d_rWWc.dt_gamma_I * _state_vars[3] 
        - d_rWWc.dt_itau_I * _state_vars[5];
    // S_E += dS_E;
    _state_vars[4] += _intermediate_vars[2];
    // S_I += dS_I;
    _state_vars[5] += _intermediate_vars[3];
    // clip S to 0-1
    _state_vars[4] = max(0.0f, min(1.0f, _state_vars[4]));
    _state_vars[5] = max(0.0f, min(1.0f, _state_vars[5]));
}

__device__ __NOINLINE__ void rWWModel::post_bw_step(
        double* _state_vars, double* _intermediate_vars,
        int* _ext_int, bool* _ext_bool, 
        int* _ext_int_shared, bool* _ext_bool_shared,
        bool& restart,
        double* _global_params, double* _regional_params,
        int& bw_i
        ) {
    if (_ext_bool_shared[0]) {
        if (((bw_i+1) >= this->conf.I_SAMPLING_START) & ((bw_i+1) <= this->conf.I_SAMPLING_END)) {
            _intermediate_vars[4] += _state_vars[0];
        }
        if ((bw_i+1) == this->conf.I_SAMPLING_END) {
            restart = false;
            __syncthreads(); // all threads must be at the same time point here given needs_fic_adjustment is shared
            _intermediate_vars[4] /= this->conf.I_SAMPLING_DURATION;
            _intermediate_vars[6] = _intermediate_vars[4] - d_rWWc.b_a_ratio_E;
            if (abs(_intermediate_vars[6] + 0.026) > 0.005) {
                restart = true;
                if (_ext_int_shared[0] < this->conf.max_fic_trials) { // only do the adjustment if max trials is not exceeded
                    // up- or downregulate inhibition
                    if ((_intermediate_vars[6]) < -0.026) {
                        _regional_params[2] -= _intermediate_vars[5];
                        // printf("sim %d node %d (trial %d): %f ==> adjusting w_IE by -%f ==> %f\n", sim_idx, j, fic_trial, I_E_ba_diff, delta, w_IE);
                        _intermediate_vars[5] -= 0.001;
                        _intermediate_vars[5] = CUDA_MAX(_intermediate_vars[5], 0.001);
                    } else {
                        _regional_params[2] += _intermediate_vars[5];
                        // printf("sim %d node %d (trial %d): %f ==> adjusting w_IE by +%f ==> %f\n", sim_idx, j, fic_trial, I_E_ba_diff, delta, w_IE);
                    }
                }
            }
            __syncthreads(); // wait to see if needs_fic_adjustment in any node
            // if needs_fic_adjustment in any node do another trial or declare fic failure and continue
            // the simulation until the end
            if (restart) {
                if (_ext_int_shared[0] < (this->conf.max_fic_trials)) {
                    if (threadIdx.x == 0) _ext_int_shared[0]++; // increment fic_trial
                } else {
                    // continue the simulation and
                    // declare FIC failed
                    restart = false;
                    _ext_bool_shared[0] = false; // _adjust_fic
                    _ext_bool_shared[1] = true; // fic_failed
                }
            } else {
                // if no node needs fic adjustment don't run
                // this block of code any more
                _ext_bool_shared[0] = false;
            }
            __syncthreads();
        }
    }
}

__device__ __NOINLINE__ void rWWModel::post_integration(
        double ***state_vars_out, 
        int **global_out_int, bool **global_out_bool,
        double* _state_vars, double* _intermediate_vars, 
        int* _ext_int, bool* _ext_bool,
        int* _ext_int_shared, bool* _ext_bool_shared,
        double** global_params, double** regional_params,
        double* _global_params, double* _regional_params,
        int& sim_idx, const int& nodes, int& j
    ) {
    if (this->conf.do_fic) {
        // save the wIE adjustment results
        // modified wIE array
        regional_params[2][sim_idx*nodes+j] = _regional_params[2];
        // number of trials and fic failure
        if (j == 0) {
            global_out_int[0][sim_idx] = _ext_int_shared[0];
            global_out_bool[1][sim_idx] = _ext_bool_shared[1];
        }
    }
}