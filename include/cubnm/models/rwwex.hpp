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
        rWWExModel, // CLASS_NAME
        "rWWEx", // NAME
        3, // STATE_VARS
        2, // INTER_VARS
        1, // NOISE
        1, // GLOBAL_PARAMS
        3, // REGIONAL_PARAMS
        2, // CONN_STATE_VAR_IDX
        2, // BOLD_STATE_VAR_IDX
        false, // HAS_POST_BW
        false, // HAS_POST_INT
        false, // IS_OSC
        0, // EXT_INT
        0, // EXT_BOOL
        0, // EXT_INT_SHARED
        0, // EXT_BOOL_SHARED
        0, // GLOBAL_OUT_INT
        0, // GLOBAL_OUT_BOOL
        0, // GLOBAL_OUT_UREAL
        0, // REGIONAL_OUT_INT
        0, // REGIONAL_OUT_BOOL
        0 // REGIONAL_OUT_UREAL
    )

    // additional functions that need to be overridden
    // (in addition to h_init, h_step, _j_restart
    // which are always overriden and have to be defined)
    // None in this model
};

#endif
