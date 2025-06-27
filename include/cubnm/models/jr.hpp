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
        double A; // max amplitude of EPSPs (mV)
        double B; // max amplitude of IPSPs (mV)
        double a; // inverse of the time constant for EPSPs (1/msec)
        double b; // inverse of the time constant for IPSPs (1/msec)
        double v0; // Firing threshold (PSP) resulting in 50% firing rate (mV)
        double nu_max; // maximum firing rate (1/msec)
        double r; // steepness of the sigmoidal transformation (1/mV)
        double p_min; // minimum input firing rate
        double p_max; // maximum input firing rate
        double mu; // mean input firing rate
        // sigmoidal coupling constants
        double cmin; // minimum of sigmoid function
        double cmax; // maximum of sigmoid function
        double midpoint; // midpoint of the linear portion of the sigmoid function
    };
    struct Config {
    };

    // second, use the boilerplate macro to include
    // the repetitive elements of the class definition
    DEFINE_DERIVED_MODEL(
        JRModel, // CLASS_NAME
        "JR", // NAME
        8, // STATE_VARS
        9, // INTER_VARS
        1, // NOISE
        1, // GLOBAL_PARAMS
        6, // REGIONAL_PARAMS
        7, // CONN_STATE_VAR_IDX
        6, // BOLD_STATE_VAR_IDX
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
