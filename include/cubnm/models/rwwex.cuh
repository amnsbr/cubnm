#ifndef RWWEX_CUH
#define RWWEX_CUH
#include "rwwex.hpp"
__constant__ rWWExModel::Constants d_rWWExc;
template void _run_simulations_gpu<rWWExModel>(
    double*, double*, double*, 
    double**, double**, double*, 
    double**, int*, double*, 
    BaseModel*
);

template void _init_gpu<rWWExModel>(BaseModel*, BWConstants, bool);
#endif
