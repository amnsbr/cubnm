#ifndef RWW_CUH
#define RWW_CUH
#include "rww.hpp"
extern void analytical_fic_het(
        gsl_matrix * sc, double G, double * w_EE, double * w_EI,
        gsl_vector * w_IE_out, bool * _unstable
        );

template void run_simulations_gpu<rWWModel>(
    double*, double*, double*, 
    u_real**, u_real**, u_real*, u_real*, gsl_matrix*, u_real*, bool,
    int, int, int, int, int, bool, rWWModel*, ModelConfigs
);

template void init_gpu<rWWModel>(
        int*, int*, int*,
        int, int, bool, int,
        int, int, int, int,
        rWWModel*, BWConstants, ModelConfigs, bool
);
#endif