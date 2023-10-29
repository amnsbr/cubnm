#ifndef CONSTANTS_HPP
#define CONSTANTS_HPP

// Using floats vs doubles in the simulations, pass -D USE_FLOATS=1 to compiler to use floats
// Note that this does not affect FIC, CMAES and GOF calculations (always using doubles)
// as well as the noise array (always generated as floats, but then casted to u_real)
#ifdef USE_FLOATS
    typedef float u_real;
    #define EXP expf
    #define POW powf
    #define CUDA_MAX fmaxf
    #define CUDA_MIN fminf
#else
    typedef double u_real;
    #define EXP exp
    #define POW pow
    #define CUDA_MAX max
    #define CUDA_MIN min
#endif

struct ModelConstants {
    u_real dt;
    u_real model_dt;
    u_real sqrt_dt;
    u_real J_NMDA;
    u_real a_E;
    u_real b_E;
    u_real b_a_ratio_E;
    u_real d_E;
    u_real a_I;
    u_real b_I;
    u_real d_I;
    u_real gamma_E;
    u_real gamma_I;
    u_real tau_E;
    u_real tau_I;
    u_real itau_E;
    u_real itau_I;
    u_real sigma_model;
    u_real I_0;
    u_real w_E;
    u_real w_I;
    u_real w_II;
    u_real I_ext;
    u_real w_E__I_0;
    u_real w_I__I_0;
    double tau_E_s;
    double tau_I_s;
    double gamma_E_s;
    double gamma_I_s;
    double r_I_ss;
    double r_E_ss;
    double I_I_ss;
    double I_E_ss;
    double S_I_ss;
    double S_E_ss;
    u_real rho;
    u_real alpha;
    u_real tau;
    u_real y;
    u_real kappa;
    u_real V_0;
    u_real k1;
    u_real k2;
    u_real k3;
    u_real ialpha;
    u_real itau;
    u_real oneminrho;
};


extern void init_constants(struct ModelConstants* mc);

struct ModelConfigs {
    // Simulation config
    int bold_remove_s;
    u_real MAX_COST;
    unsigned int I_SAMPLING_START;
    unsigned int I_SAMPLING_END;
    unsigned int I_SAMPLING_DURATION;
    bool numerical_fic;
    int max_fic_trials_cmaes;
    int max_fic_trials;
    u_real init_delta;
    bool use_fc_ks;
    bool use_fc_diff;
    bool exc_interhemispheric;

    // CMAES config
    bool sim_verbose;
    double sigma;
    double alphacov;
    int Variante;
    double gamma_scale;
    double bound_soft_edge;
    int early_stop_gens;
    double early_stop_tol;
    double fic_penalty_scale;
    bool fic_reject_failed;
    u_real scale_max_minmax;

    // GPU config
    bool grid_save_hemis;
    bool grid_save_dfc;
    bool w_IE_1;    
};

extern void init_conf(struct ModelConfigs* conf);

#endif