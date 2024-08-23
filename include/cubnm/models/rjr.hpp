#ifndef RJR_HPP
#define RJR_HPP
#include "cubnm/models/base.hpp"
class rJRModel : public BaseModel {
public:
    // first define Constants and Config structs
    // they always must be defined even if empty

    struct Constants {
        u_real dt;            // Time step (sec)
        u_real sqrt_dt;       // Square root of time step
        u_real Ke;            // Inversion of time constant of the EPSP kernel (1/sec)
        u_real Ki;            // Inversion of time constant of the IPSP kernel (1/sec)
        u_real He;            // Maximal EPSP (mV)
        u_real Hi;            // Maximal IPSP (mV)
        u_real Fe;            // Maximal firing rate of excitatory population (Hz)
        u_real Fi;            // Maximal firing rate of inhibitory population (Hz)
        u_real Re;            // Slope of the sigmoid activation function for excitatory population (1/mV)
        u_real Ri;            // Slope of the sigmoid activation function for inhibitory population (1/mV)
        u_real V50e;          // EPSP that achieves a 50% firing rate of a neural population (mV)
        u_real V50i;          // IPSP that achieves a 50% firing rate of a neural population (mV)
        u_real Dr;            // Damping ratio = 1 for critical damping (Dr > 1: exponential decaying, Dr < 1: Oscillating)
        u_real V0;            // Rest potential (mV) for PSP
        u_real C0;            // Connectivity weight of E to I
        u_real HeKe;
        u_real HiKi;
        u_real DrKe2;
        u_real DrKi2;
        u_real Ke_sq;
        u_real Ki_sq;
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
