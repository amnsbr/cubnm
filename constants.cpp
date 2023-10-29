#ifndef CONSTANTS_CPP
#define CONSTANTS_CPP
#include "helpers.hpp"
#include "constants.hpp"

/*
Global model and integration parameters
*/
const u_real dt                  = 0.1;      // Integration step length dt = 0.1 ms
const u_real model_dt            = 0.001;    // Time-step of model (sampling-rate=1000 Hz)
const u_real sqrt_dt             = 0.316227; // sqrt(dt)
/*
Local model: DMF-Parameters from Deco et al. JNeuro 2014
*/
const u_real J_NMDA  = 0.15;
const u_real a_E     = 310;          // (n/C)
const u_real b_E     = 125;          // (Hz)
const u_real b_a_ratio_E   = b_E / a_E;
const u_real d_E     = 0.16;         // (s)
const u_real a_I     = 615;          // (n/C)
const u_real b_I     = 177;          // (Hz)
const u_real d_I     = 0.087;        // (s)
const u_real gamma_E   = (u_real)0.641/(u_real)1000.0; // factor 1000 for expressing everything in ms
const u_real gamma_I   = (u_real)1.0/(u_real)1000.0; // factor 1000 for expressing everything in ms
const u_real tau_E   = 100;          // (ms) Time constant of NMDA (excitatory)
const u_real tau_I   = 10;           // (ms) Time constant of GABA (inhibitory)
const u_real itau_E  = 1.0/tau_E;
const u_real itau_I  = 1.0/tau_I;
const u_real sigma_model   = 0.01;   // (nA) Noise amplitude (named sigma_model to avoid confusion with CMAES sigma)
const u_real I_0     = 0.382;        // (nA) overall effective external input
const u_real w_E     = 1.0;          // scaling of external input for excitatory pool
const u_real w_I     = 0.7;          // scaling of external input for inhibitory pool
const u_real w_II = 1.0;             // I->I self-coupling
const u_real I_ext = 0.0;            // [nA] external input
const u_real w_E__I_0      = w_E * I_0;
const u_real w_I__I_0      = w_I * I_0;

// tau and gamma in seconds (for FIC)
const double tau_E_s = 0.1;   // [s] (NMDA)
const double tau_I_s = 0.01;  // [s] (GABA)
const double gamma_E_s = 0.641;        // kinetic conversion factor (typo in text)
const double gamma_I_s = 1.0;

// Steady-state solutions in isolated case (for FIC)
const double r_I_ss = 3.9218448633;  // Hz
const double r_E_ss = 3.0773270642;  // Hz
const double I_I_ss = 0.2528951325;  // nA
const double I_E_ss = 0.3773805650;  // nA
const double S_I_ss = 0.0392184486;  // dimensionless
const double S_E_ss = 0.1647572075;  // dimensionless

/*
Balloon-Windkessel
*/
const u_real rho = 0.34;
const u_real alpha = 0.32;
const u_real tau = 0.98;
const u_real y = 1.0/0.41;
const u_real kappa = 1.0/0.65;
const u_real V_0 = 0.02;
const u_real k1 = 7 * rho;
const u_real k2 = 2.0;
const u_real k3 = 2 * rho - 0.2;
const u_real ialpha = 1.0/alpha;
const u_real itau = 1.0/tau;
const u_real oneminrho = (1.0 - rho);

// simulations config
const int bold_remove_s = get_env_or_default("BNM_BOLD_REMOVE", 30); // length of BOLD to remove before FC(D) calculation, in seconds
const u_real MAX_COST = get_env_or_default("BNM_MAX_COST", 2); // max cost returned if simulation/fitting is not feasible for set of parameters
const unsigned int I_SAMPLING_START = get_env_or_default("BNM_I_SAMPLING_START", 1000);
const unsigned int I_SAMPLING_END = get_env_or_default("BNM_I_SAMPLING_END", 10000);
const unsigned int I_SAMPLING_DURATION = I_SAMPLING_END - I_SAMPLING_START + 1;
const bool numerical_fic = get_env_or_default("BNM_NUMERICAL_FIC", 1);
const int max_fic_trials_cmaes = get_env_or_default("BNM_MAX_FIC_TRIALS_CMAES", 10);
const int max_fic_trials = get_env_or_default("BNM_MAX_FIC_TRIALS", 150);
const u_real init_delta = get_env_or_default("BNM_INIT_DELTA", 0.02); // initial delta in numerical adjustment of FIC
const bool use_fc_ks = get_env_or_default("BNM_USE_FC_KS", 0);
const bool use_fc_diff = get_env_or_default("BNM_USE_FC_DIFF", 1);
const bool exc_interhemispheric = get_env_or_default("BNM_EXC_INTERHEM", 1); // exclude interhemispheric connections from FC(D) calculation and fitting
// CMAES config
const bool sim_verbose = get_env_or_default("BNM_CMAES_SIM_VERBOSE", 0);
double sigma = get_env_or_default("BNM_CMAES_SIGMA", 0.5); // is updated during CMAES, should not be const
const double alphacov = get_env_or_default("BNM_CMAES_ALPHACOV", 2);
const int Variante = get_env_or_default("BNM_CMAES_VARIANTE", 6);
const double gamma_scale = get_env_or_default("BNM_CMAES_GAMMA_SCALE", 2.0);
const double bound_soft_edge = get_env_or_default("BNM_CMAES_BOUND_SOFT_EDGE", 0.0);
const int early_stop_gens = get_env_or_default("BNM_CMAES_EARLY_STOP_GENS", 30);
const double early_stop_tol = get_env_or_default("BNM_CMAES_EARLY_STOP_TOL", 1e-3);
const double fic_penalty_scale = get_env_or_default("BNM_CMAES_FIC_PENALTY_SCALE", 2.0);
const bool fic_reject_failed = get_env_or_default("BNM_CMAES_FIC_REJECT_FAILED", 0);
const u_real scale_max_minmax = get_env_or_default("BNM_SCALE_MAX_MINMAX", 1.0);
// GPU config
const bool grid_save_hemis = get_env_or_default("BNM_GPU_SAVE_HEMIS", 1);
const bool grid_save_dfc = get_env_or_default("BNM_GPU_SAVE_DFC", 0);
const bool w_IE_1 = false; // set initial w_IE to 1 in all nodes (similar to Deco 2014; ignores w_IE_fic)

#endif