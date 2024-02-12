#ifndef BNM_HPP
#define BNM_HPP
template <typename Model>
void _run_simulations_cpu(
    double * BOLD_ex_out, double * fc_trils_out, double * fcd_trils_out,
    u_real ** global_params, u_real ** regional_params, u_real * v_list,
    u_real * SC, u_real * SC_dist, BaseModel* m
);

template <typename Model>
void _init_cpu(BaseModel *m);
#endif