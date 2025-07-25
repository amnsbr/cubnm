#ifndef BNM_HPP
#define BNM_HPP
template <typename Model>
void _run_simulations_cpu(
    double * BOLD_ex_out, double * fc_trils_out, double * fcd_trils_out,
    double ** global_params, double ** regional_params, double * v_list,
    double * SC, double * SC_dist, BaseModel* m
);

template <typename Model>
void _init_cpu(BaseModel *m, bool force_reinit);

typedef void (*HGlobalInputFunc)(
    double&, int&,
    const int&, const int&, 
    int&, int&, double*, 
    int*, const bool&, const int&,
    double*, double*
);

#endif