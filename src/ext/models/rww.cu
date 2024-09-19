#include "cubnm/includes.cuh"
#include "cubnm/defines.h"
#include "cubnm/models/rww.cuh"
__device__ __NOINLINE__ void rWWModel::init(
    u_real* _state_vars, u_real* _intermediate_vars, 
    u_real* _global_params, u_real* _regional_params,
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
    u_real* _state_vars, u_real* _intermediate_vars, 
    u_real* _global_params, u_real* _regional_params,
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
        u_real* _state_vars, u_real* _intermediate_vars,
        u_real* _global_params, u_real* _regional_params,
        u_real& tmp_globalinput,
        u_real* noise, long& noise_idx
        ) {
    _state_vars[0] = d_rWWc.w_E__I_0 + _regional_params[0] * _state_vars[4] + tmp_globalinput * _global_params[0] * d_rWWc.J_NMDA - _regional_params[2] * _state_vars[5];
    // *tmp_I_E = d_rWWc.w_E__I_0 + (*w_EE) * (*S_i_E) + (*tmp_globalinput) * (*G_J_NMDA) - (*w_IE) * (*S_i_I);
    _state_vars[1] = d_rWWc.w_I__I_0 + _regional_params[1] * _state_vars[4] - _state_vars[5];
    // *tmp_I_I = d_rWWc.w_I__I_0 + (*w_EI) * (*S_i_E) - (*S_i_I);
    _intermediate_vars[0] = d_rWWc.a_E * _state_vars[0] - d_rWWc.b_E;
    // *tmp_aIb_E = d_rWWc.a_E * (*tmp_I_E) - d_rWWc.b_E;
    _intermediate_vars[1] = d_rWWc.a_I * _state_vars[1] - d_rWWc.b_I;
    // *tmp_aIb_I = d_rWWc.a_I * (*tmp_I_I) - d_rWWc.b_I;
    #ifdef USE_FLOATS
    // to avoid firing rate approaching infinity near I = b/a
    if (abs(_intermediate_vars[0]) < 1e-4) _intermediate_vars[0] = 1e-4;
    if (abs(_intermediate_vars[0]) < 1e-4) _intermediate_vars[0] = 1e-4;
    #endif
    _state_vars[2] = _intermediate_vars[0] / (1 - EXP(-d_rWWc.d_E * _intermediate_vars[0]));
    // *tmp_r_E = *tmp_aIb_E / (1 - exp(-d_rWWc.d_E * (*tmp_aIb_E)));
    _state_vars[3] = _intermediate_vars[1] / (1 - EXP(-d_rWWc.d_I * _intermediate_vars[1]));
    // *tmp_r_I = *tmp_aIb_I / (1 - exp(-d_rWWc.d_I * (*tmp_aIb_I)));
    _intermediate_vars[2] = noise[noise_idx] * d_rWWc.sigma_model_sqrt_dt + d_rWWc.dt_gamma_E * ((1 - _state_vars[4]) * _state_vars[2]) - d_rWWc.dt_itau_E * _state_vars[4];
    // *dSdt_E = noise[*noise_idx] * d_rWWc.sigma_model_sqrt_dt + d_rWWc.dt_gamma_E * ((1 - (*S_i_E)) * (*tmp_r_E)) - d_rWWc.dt_itau_E * (*S_i_E);
    _intermediate_vars[3] = noise[noise_idx+1] * d_rWWc.sigma_model_sqrt_dt + d_rWWc.dt_gamma_I * _state_vars[3] - d_rWWc.dt_itau_I * _state_vars[5];
    // *dSdt_I = noise[*noise_idx+1] * d_rWWc.sigma_model_sqrt_dt + d_rWWc.dt_gamma_I * (*tmp_r_I) - d_rWWc.dt_itau_I * (*S_i_I);
    _state_vars[4] += _intermediate_vars[2];
    // *S_i_E += *dSdt_E;
    _state_vars[5] += _intermediate_vars[3];
    // *S_i_I += *dSdt_I;
    // clip S to 0-1
    _state_vars[4] = max(0.0f, min(1.0f, _state_vars[4]));
    // *S_i_E = max(0.0f, min(1.0f, *S_i_E));
    _state_vars[5] = max(0.0f, min(1.0f, _state_vars[5]));
    // *S_i_I = max(0.0f, min(1.0f, *S_i_I));
}

__device__ __NOINLINE__ void rWWModel::post_bw_step(
        u_real* _state_vars, u_real* _intermediate_vars,
        int* _ext_int, bool* _ext_bool, 
        int* _ext_int_shared, bool* _ext_bool_shared,
        bool& restart,
        u_real* _global_params, u_real* _regional_params,
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
        u_real ***state_vars_out, 
        int **global_out_int, bool **global_out_bool,
        u_real* _state_vars, u_real* _intermediate_vars, 
        int* _ext_int, bool* _ext_bool,
        int* _ext_int_shared, bool* _ext_bool_shared,
        u_real** global_params, u_real** regional_params,
        u_real* _global_params, u_real* _regional_params,
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