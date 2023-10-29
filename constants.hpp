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

// Global model and integration parameters
extern const u_real dt;
extern const u_real model_dt;
extern const u_real sqrt_dt;

// Local model: DMF-Parameters from Deco et al. JNeuro 2014
extern const u_real J_NMDA;
extern const u_real a_E;
extern const u_real b_E;
extern const u_real b_a_ratio_E;
extern const u_real d_E;
extern const u_real a_I;
extern const u_real b_I;
extern const u_real d_I;
extern const u_real gamma_E;
extern const u_real gamma_I;
extern const u_real tau_E;
extern const u_real tau_I;
extern const u_real itau_E;
extern const u_real itau_I;
extern const u_real sigma_model;
extern const u_real I_0;
extern const u_real w_E;
extern const u_real w_I;
extern const u_real w_II;
extern const u_real I_ext;
extern const u_real w_E__I_0;
extern const u_real w_I__I_0;

// Tau and gamma in seconds (for FIC)
extern const double tau_E_s;
extern const double tau_I_s;
extern const double gamma_E_s;
extern const double gamma_I_s;

// Steady-state solutions in isolated case (for FIC)
extern const double r_I_ss;
extern const double r_E_ss;
extern const double I_I_ss;
extern const double I_E_ss;
extern const double S_I_ss;
extern const double S_E_ss;

// Balloon-Windkessel
extern const u_real rho;
extern const u_real alpha;
extern const u_real tau;
extern const u_real y;
extern const u_real kappa;
extern const u_real V_0;
extern const u_real k1;
extern const u_real k2;
extern const u_real k3;
extern const u_real ialpha;
extern const u_real itau;
extern const u_real oneminrho;

// Simulation config
extern const int bold_remove_s;
extern const u_real MAX_COST;
extern const unsigned int I_SAMPLING_START;
extern const unsigned int I_SAMPLING_END;
extern const unsigned int I_SAMPLING_DURATION;
extern const bool numerical_fic;
extern const int max_fic_trials_cmaes;
extern const int max_fic_trials;
extern const u_real init_delta;
extern const bool use_fc_ks;
extern const bool use_fc_diff;
extern const bool exc_interhemispheric;

// CMAES config
extern const bool sim_verbose;
extern double sigma;
extern const double alphacov;
extern const int Variante;
extern const double gamma_scale;
extern const double bound_soft_edge;
extern const int early_stop_gens;
extern const double early_stop_tol;
extern const double fic_penalty_scale;
extern const bool fic_reject_failed;
extern const u_real scale_max_minmax;

// GPU config
extern const bool grid_save_hemis;
extern const bool grid_save_dfc;
extern const bool w_IE_1;
#endif