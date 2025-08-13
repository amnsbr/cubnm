#ifndef KURAMOTO_HPP
#define KURAMOTO_HPP
#include "cubnm/models/base.hpp"
class KuramotoModel : public BaseModel {
public:
    // first define Constants and Config structs
    // they always must be defined even if empty
    struct Constants {
        double dt;
        double sqrt_dt;
        double twopi;
    };
    struct Config {
    };

    // second, use the boilerplate macro to include
    // the repetitive elements of the class definition
    DEFINE_DERIVED_MODEL(
        KuramotoModel, // CLASS_NAME
        "Kuramoto", // NAME
        1, // STATE_VARS
        1, // INTER_VARS
        1, // NOISE
        1, // GLOBAL_PARAMS
        3, // REGIONAL_PARAMS
        0, // CONN_STATE_VAR_IDX
        0, // BOLD_STATE_VAR_IDX
        false, // HAS_POST_BW
        false, // HAS_POST_INT
        true, // IS_OSC
        0, // EXT_INT
        0, // EXT_BOOL
        0, // EXT_INT_SHARED
        0, // EXT_BOOL_SHARED
        0, // GLOBAL_OUT_INT
        0, // GLOBAL_OUT_BOOL
        0, // GLOBAL_OUT_DOUBLE
        0, // REGIONAL_OUT_INT
        0, // REGIONAL_OUT_BOOL
        0 // REGIONAL_OUT_DOUBLE
    )

    // additional functions that need to be overridden
    // (in addition to h_init, h_step, _j_restart
    // which are always overriden and have to be defined)
    // None in this model
};

#endif
