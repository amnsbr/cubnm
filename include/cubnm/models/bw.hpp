#ifndef BW_HPP
#define BW_HPP
struct BWConstants {
    double dt;
    double rho;
    double alpha;
    double tau;
    double y;
    double kappa;
    double V_0;
    double k1;
    double k2;
    double k3;
    double ialpha;
    double itau;
    double oneminrho;
    double dt_itau;
    double V_0_k1;
    double V_0_k2;
    double V_0_k3;
};

void init_bw_constants(BWConstants* bwc, double dt = 0.001);
#endif