#ifndef RWW_CUH
#define RWW_CUH
#include "rww.hpp"
extern void analytical_fic_het(
        gsl_matrix * sc, double G, double * w_EE, double * w_EI,
        gsl_vector * w_IE_out, bool * _unstable
        );

// the following are necessary as cuda code is compiled separately
// but the function is called from the main code that is compiled by g++
// therefore, from nvcc's perspective, there is no need to compile the
// function for this model, because it is not used in the cuda code
// TODO: remove them 
template void _run_simulations_gpu<rWWModel>(
    double*, double*, double*, 
    u_real**, u_real**, u_real*, 
    u_real*, gsl_matrix*, u_real*, 
    BaseModel*
);

template void _init_gpu<rWWModel>(BaseModel*, BWConstants);
#endif