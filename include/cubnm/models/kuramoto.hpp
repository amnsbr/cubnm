#ifndef KURAMOTO_HPP
#define KURAMOTO_HPP
#include "cubnm/models/base.hpp"
class KuramotoModel : public BaseModel {
public:
    // first define Constants and Config structs
    // they always must be defined even if empty
    struct Constants {
        u_real dt;
        u_real sqrt_dt;
        u_real twopi;
    };
    struct Config {
    };

    // second, use the boilerplate macro to include
    // the repetitive elements of the class definition
    DEFINE_DERIVED_MODEL(
        KuramotoModel, 
        "Kuramoto", 
        1, 
        1, 
        1, 
        1, 
        3,
        0,
        0,
        false, 
        false, 
        true,
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
