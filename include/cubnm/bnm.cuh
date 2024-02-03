#ifndef BNM_CUH
#define BNM_CUH
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
    u_real ** global_params, u_real ** regional_params, u_real * v_list,
    u_real * SC, gsl_matrix * SC_gsl, u_real * SC_dist, bool do_delay,
    int nodes, int time_steps, int BOLD_TR,int window_size,
    int N_SIMS, bool extended_output, Model *model,
    ModelConfigs conf
);

template<typename Model>
void init_gpu(
        int *output_ts_p, int *n_pairs_p, int *n_window_pairs_p,
        int N_SIMS, int nodes, bool extended_output, int rand_seed,
        int BOLD_TR, int time_steps, int window_size, int window_step,
        Model *model, BWConstants bwc, ModelConfigs conf, bool verbose
        );

#endif