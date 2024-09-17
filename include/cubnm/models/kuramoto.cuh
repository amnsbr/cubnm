#ifndef KURAMOTO_CUH
#define KURAMOTO_CUH
#include "kuramoto.hpp"
__constant__ KuramotoModel::Constants d_Kuramotoc;
template void _run_simulations_gpu<KuramotoModel>(
    double*, double*, double*, 
    u_real**, u_real**, u_real*, 
    u_real**, int*, u_real*, 
    BaseModel*
);

template void _init_gpu<KuramotoModel>(BaseModel*, BWConstants, bool);
#endif
