#include "cubnm/models/base.hpp"
#ifdef GPU_ENABLED
// declare gpu functions which will be provided by bnm.cu compiled library
template<typename Model>
extern void _init_gpu(BaseModel *model, BWConstants bwc, bool force_reinit);

template<typename Model>
extern void _run_simulations_gpu(
    double * BOLD_ex_out, double * fc_trils_out, double * fcd_trils_out,
    u_real ** global_params, u_real ** regional_params, u_real * v_list,
    u_real ** SC, int * SC_indices, u_real * SC_dist, BaseModel *model
);
#endif
