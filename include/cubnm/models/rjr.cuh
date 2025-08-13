#ifndef RJR_CUH
#define RJR_CUH
#include "rjr.hpp"
__constant__ rJRModel::Constants d_rJRc;
template void _run_simulations_gpu<rJRModel>(
    double*, double*, double*, 
    double**, double**, double*, 
    double**, int*, double*, 
    BaseModel*
);

template void _init_gpu<rJRModel>(BaseModel*, BWConstants, bool);
#endif
