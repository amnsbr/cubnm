#ifndef RWWEX_HPP
#define RWWEX_HPP
#include "cubnm/models/base.hpp"
class rWWExModel : public BaseModel {
public:
    // first define Constants and Config structs
    // they always must be defined even if empty
    struct Constants {
        u_real dt;
        u_real sqrt_dt;
        u_real J_N;
        u_real a;
        u_real b;
        u_real d;
        u_real gamma;
        u_real tau;
        u_real itau;
        u_real dt_itau;
        u_real dt_gamma;
    };
    struct Config {
    };

    // second, use the boilerplate macro to include
    // the repetitive elements of the class definition
    DEFINE_DERIVED_MODEL(
        rWWExModel, 
        "rWWEx", 
        3, 
        2, 
        1, 
        1, 
        3,
        2,
        2,
        false, 
        false, 
        0, 
        0, 
        0, 
        0, 
        0, 
        0, 
        0, 
        0, 
        0, 
        0
    )

    // additional functions that need to be overridden
    // (in addition to h_init, h_step, _j_restart
    // which are always overriden and have to be defined)
    // None in this model
};

#endif
