#ifndef RJR_HPP
#define RJR_HPP
#include "cubnm/models/base.hpp"
class rJRModel : public BaseModel {
public:
    // first define Constants and Config structs
    // they always must be defined even if empty

    struct Constants {
        // Note: Following Jung 2023 units in this model
        // (unlike the others) are in sec or 1/sec rather 
        // than sec or 1/msec
        // TODO: harmonize it with the other models
        double dt;            // Time step (sec)
        double sqrt_dt;       // Square root of time step
        double Ke;            // Inversion of time constant of the EPSP kernel (1/sec)
        double Ki;            // Inversion of time constant of the IPSP kernel (1/sec)
        double He;            // Maximal EPSP (mV)
        double Hi;            // Maximal IPSP (mV)
        double Fe;            // Maximal firing rate of excitatory population (Hz)
        double Fi;            // Maximal firing rate of inhibitory population (Hz)
        double Re;            // Slope of the sigmoid activation function for excitatory population (1/mV)
        double Ri;            // Slope of the sigmoid activation function for inhibitory population (1/mV)
        double V50e;          // EPSP that achieves a 50% firing rate of a neural population (mV)
        double V50i;          // IPSP that achieves a 50% firing rate of a neural population (mV)
        double Dr;            // Damping ratio = 1 for critical damping (Dr > 1: exponential decaying, Dr < 1: Oscillating)
        double V0;            // Rest potential (mV) for PSP
        double HeKe;
        double HiKi;
        double DrKe2;
        double DrKi2;
        double Ke_sq;
        double Ki_sq;
    };
    struct Config {
    };

    // second, use the boilerplate macro to include
    // the repetitive elements of the class definition
    DEFINE_DERIVED_MODEL(
        rJRModel, 
        "rJR", 
        5, // STATE_VARS: EPSP, IPSP, EPSC, IPSC, EPSP/He
        2, // INTER_VARS: d2EPSP/dt2, d2IPSP/dt2 
        1, // NOISE
        1, // GLOBAL_PARAMS: G
        4, // REGIONAL_PARAMS: C_IE, C_EI, R, sigma
        0, // CONN_STATE_VAR_IDX
        4, // BOLD_STATE_VAR_IDX 
        false, 
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
