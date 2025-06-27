#ifndef JR_HPP
#define JR_HPP
#include "cubnm/models/base.hpp"
class JRModel : public BaseModel {
public:
    // first define Constants and Config structs
    // they always must be defined even if empty
    // TODO: use clearer names for the constants
    struct Constants {
        double dt; // time step in seconds
        double sqrt_dt; // square root of time step
        double a; // inverse of the characteristic time constant for EPSPs (1/sec)
        double ad; // inverse of the characteristic time constant for long-range EPSPs (1/sec)
        double b; // inverse of the characteristic time constant for IPSPs (1/sec)
        double p; // basal input to pyramidal population
        double A; // amplitude of EPSPs
        double B; // amplitude of IPSPs
        double e0; // half of the maximum firing rate
        double v0; // V1/2
        double r0, r1, r2; // slopes of sigmoid functions
        // derived
        double a2; // a squared
        double b2; // b squared
        double ad2; // ad squared
    };
    struct Config {
    };

    // second, use the boilerplate macro to include
    // the repetitive elements of the class definition
    DEFINE_DERIVED_MODEL(
        JRModel, // CLASS_NAME
        "JR", // NAME
        9, // STATE_VARS
        13, // INTER_VARS
        1, // NOISE
        1, // GLOBAL_PARAMS
        6, // REGIONAL_PARAMS
        3, // CONN_STATE_VAR_IDX
        8, // BOLD_STATE_VAR_IDX
        false, // HAS_POST_BW
        false, // HAS_POST_INT
        false, // IS_OSC
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
