#include "cubnm/models/base.hpp"
#ifdef GPU_ENABLED
// declare gpu functions which will be provided by bnm.cu compiled library
template<typename Model>
extern void _init_gpu(BaseModel *model, BWConstants bwc, bool force_reinit);

template<typename Model>
extern void _run_simulations_gpu(
    double * BOLD_ex_out, double * fc_trils_out, double * fcd_trils_out,
    double ** global_params, double ** regional_params, double * v_list,
    double ** SC, int * SC_indices, double * SC_dist, BaseModel *model
);
#endif
