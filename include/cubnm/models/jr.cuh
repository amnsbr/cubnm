#ifndef JR_CUH
#define JR_CUH
#include "jr.hpp"
__constant__ JRModel::Constants d_JRc;
template void _run_simulations_gpu<JRModel>(
    double*, double*, double*, 
    double**, double**, double*, 
    double**, int*, double*, 
    BaseModel*
);

template void _init_gpu<JRModel>(BaseModel*, BWConstants, bool);
#endif
