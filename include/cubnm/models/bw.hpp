#ifndef BW_HPP
#define BW_HPP
struct BWConstants {
    u_real bw_dt;
    u_real rho;
    u_real alpha;
    u_real tau;
    u_real y;
    u_real kappa;
    u_real V_0;
    u_real k1;
    u_real k2;
    u_real k3;
    u_real ialpha;
    u_real itau;
    u_real oneminrho;
    u_real bw_dt_itau;
    u_real V_0_k1;
    u_real V_0_k2;
    u_real V_0_k3;
};

extern void init_bw_constants(BWConstants* bwc);
#endif