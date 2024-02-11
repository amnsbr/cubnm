#ifndef RWWEX_CUH
#define RWWEX_CUH
#include "rwwex.hpp"
template void _run_simulations_gpu<rWWExModel>(
    double*, double*, double*, 
    u_real**, u_real**, u_real*, 
    u_real*, u_real*, 
    BaseModel*
);

template void _init_gpu<rWWExModel>(BaseModel*, BWConstants);
#endif