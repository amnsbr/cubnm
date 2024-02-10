#include "cubnm/models/base.hpp"
#ifdef GPU_ENABLED
// declare gpu functions which will be provided by bnm.cu compiled library
template<typename Model>
extern void init_gpu(
        int *output_ts_p, int *n_pairs_p, int *n_window_pairs_p,
        int N_SIMS, int nodes, bool extended_output, int rand_seed,
        int BOLD_TR, int time_steps, int window_size, int window_step,
        BaseModel *model, BWConstants bwc, bool verbose
        );
template<typename Model>
extern void run_simulations_gpu(
    double * BOLD_ex_out, double * fc_trils_out, double * fcd_trils_out,
    u_real ** global_params, u_real ** regional_params, u_real * v_list,
    u_real * SC, gsl_matrix * SC_gsl, u_real * SC_dist, bool do_delay, int nodes,
    int time_steps, int BOLD_TR, int window_size,
    int N_SIMS, bool extended_output, BaseModel *model
);
#endif