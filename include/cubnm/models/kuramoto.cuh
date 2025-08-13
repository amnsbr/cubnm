#ifndef KURAMOTO_CUH
#define KURAMOTO_CUH
#include "kuramoto.hpp"
__constant__ KuramotoModel::Constants d_Kuramotoc;
template void _run_simulations_gpu<KuramotoModel>(
    double*, double*, double*, 
    double**, double**, double*, 
    double**, int*, double*, 
    BaseModel*
);

template void _init_gpu<KuramotoModel>(BaseModel*, BWConstants, bool);
#endif
