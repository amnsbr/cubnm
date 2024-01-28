#ifndef BNM_CUH
#define BNM_CUH

cudaDeviceProp prop;

namespace bnm_gpu {
    bool is_initialized = false;
}
int n_vols_remove, corr_len, n_windows, n_pairs, n_window_pairs, output_ts, max_delay;
bool has_delay;
u_real ***state_vars_out,**BOLD, **mean_bold, **ssd_bold, **fc_trils, **windows_mean_bold, **windows_ssd_bold,
    **windows_fc_trils, **windows_mean_fc, **windows_ssd_fc, **fcd_trils, *noise,
    *d_SC, **d_global_params, **d_regional_params;
int **global_out_int;
bool **global_out_bool;
int *pairs_i, *pairs_j, *window_starts, *window_ends, *window_pairs_i, *window_pairs_j;
int last_time_steps = 0; // to avoid recalculating noise in subsequent calls of the function with force_reinit
int last_nodes = 0;
int last_rand_seed = 0;
#ifdef NOISE_SEGMENT
int *shuffled_nodes, *shuffled_ts;
// set a default length of noise (msec)
// (+1 to avoid having an additional repeat for a single time point
// when time_steps can be divided by 30(000), as the actual duration of
// simulation (in msec) is always user request time steps + 1)
int noise_time_steps = 30001;
int noise_repeats; // number of noise segment repeats for current simulations
#endif
#ifdef USE_FLOATS
double **d_fc_trils, **d_fcd_trils;
#else
// use d_fc_trils and d_fcd_trils as aliases for fc_trils and fcd_trils
// which will later be used for GOF calculations
#define d_fc_trils fc_trils
#define d_fcd_trils fcd_trils
#endif

__device__ void calculateGlobalInput(
        u_real* tmp_globalinput, int* k_buff_idx,
        int* nodes, int* sim_idx, int* j, int *k, 
        int* buff_idx, int* max_delay, u_real* _SC, 
        int** delay, bool* has_delay, 
        u_real** conn_state_var_hist, u_real* conn_state_var_1
        );

template<typename Model>
__global__ void bnm(
    Model* model,
    u_real **BOLD,
    u_real ***state_vars_out, 
    int **global_out_int,
    bool **global_out_bool,
    int n_vols_remove,
    u_real *SC, u_real **global_params, u_real **regional_params,
    u_real **conn_state_var_hist, int **delay, int max_delay,
    int N_SIMS, int nodes, int BOLD_TR, int time_steps, 
    u_real *noise, 
    bool extended_output,
#ifdef NOISE_SEGMENT
    int *shuffled_nodes, int *shuffled_ts,
    int noise_time_steps, int noise_repeats,
#endif
    int corr_len
    );

template<typename Model>
void run_simulations_gpu(
    double * BOLD_out, double * fc_trils_out, double * fcd_trils_out,
    double * S_E_out, double * S_I_out,
    double * r_E_out, double * r_I_out,
    double * I_E_out, double * I_I_out,
    bool * fic_unstable_out, bool * fic_failed_out,
    u_real * G_list, u_real * w_EE_list, u_real * w_EI_list, u_real * w_IE_list, u_real * v_list,
    u_real * SC, gsl_matrix * SC_gsl, u_real * SC_dist, bool do_delay,
    int nodes, int time_steps, int BOLD_TR, int _max_fic_trials, int window_size,
    int N_SIMS, bool do_fic, bool only_wIE_free, bool extended_output,
    ModelConfigs conf
);

template<typename Model, typename ModelConstants>
void init_gpu(
        int *output_ts_p, int *n_pairs_p, int *n_window_pairs_p,
        int N_SIMS, int nodes, bool do_fic, bool extended_output, int rand_seed,
        int BOLD_TR, int time_steps, int window_size, int window_step,
        BWConstants bwc, ModelConstants mc, ModelConfigs conf, bool verbose
        );

#endif