#ifndef CONSTANTS_CPP
#define CONSTANTS_CPP
#include "helpers.cpp"
#include "constants.hpp"

struct ModelConstants mc;
struct ModelConfigs conf;

void init_constants(struct ModelConstants* mc) {
    mc->dt                  = 0.1;      // Integration step length dt = 0.1 ms
    mc->model_dt            = 0.001;    // Time-step of model (sampling-rate=1000 Hz)
    mc->sqrt_dt             = 0.316227; // sqrt(dt)
    /*
    Local model: DMF-Parameters from Deco et al. JNeuro 2014
    */
    mc->J_NMDA  = 0.15;
    mc->a_E     = 310;          // (n/C)
    mc->b_E     = 125;          // (Hz)
    mc->b_a_ratio_E   = mc->b_E / mc->a_E;
    mc->d_E     = 0.16;         // (s)
    mc->a_I     = 615;          // (n/C)
    mc->b_I     = 177;          // (Hz)
    mc->d_I     = 0.087;        // (s)
    mc->gamma_E   = (u_real)0.641/(u_real)1000.0; // factor 1000 for expressing everything in ms
    mc->gamma_I   = (u_real)1.0/(u_real)1000.0; // factor 1000 for expressing everything in ms
    mc->tau_E   = 100;          // (ms) Time constant of NMDA (excitatory)
    mc->tau_I   = 10;           // (ms) Time constant of GABA (inhibitory)
    mc->itau_E  = 1.0/mc->tau_E;
    mc->itau_I  = 1.0/mc->tau_I;
    mc->sigma_model   = 0.01;   // (nA) Noise amplitude (named sigma_model to avoid confusion with CMAES sigma)
    mc->I_0     = 0.382;        // (nA) overall effective external input
    mc->w_E     = 1.0;          // scaling of external input for excitatory pool
    mc->w_I     = 0.7;          // scaling of external input for inhibitory pool
    mc->w_II = 1.0;             // I->I self-coupling
    mc->I_ext = 0.0;            // [nA] external input
    mc->w_E__I_0      = mc->w_E * mc->I_0;
    mc->w_I__I_0      = mc->w_I * mc->I_0;

    // tau and gamma in seconds (for FIC)
    mc->tau_E_s = 0.1;   // [s] (NMDA)
    mc->tau_I_s = 0.01;  // [s] (GABA)
    mc->gamma_E_s = 0.641;        // kinetic conversion factor (typo in text)
    mc->gamma_I_s = 1.0;

    // Steady-state solutions in isolated case (for FIC)
    mc->r_I_ss = 3.9218448633;  // Hz
    mc->r_E_ss = 3.0773270642;  // Hz
    mc->I_I_ss = 0.2528951325;  // nA
    mc->I_E_ss = 0.3773805650;  // nA
    mc->S_I_ss = 0.0392184486;  // dimensionless
    mc->S_E_ss = 0.1647572075;  // dimensionless

    /*
    Balloon-Windkessel
    */
    mc->rho = 0.34;
    mc->alpha = 0.32;
    mc->tau = 0.98;
    mc->y = 1.0/0.41;
    mc->kappa = 1.0/0.65;
    mc->V_0 = 0.02;
    mc->k1 = 7 * mc->rho;
    mc->k2 = 2.0;
    mc->k3 = 2 * mc->rho - 0.2;
    mc->ialpha = 1.0/mc->alpha;
    mc->itau = 1.0/mc->tau;
    mc->oneminrho = (1.0 - mc->rho);
}

void init_conf(struct ModelConfigs* conf) {
    // simulations config
    conf->bold_remove_s = get_env_or_default("BNM_BOLD_REMOVE", 30); // length of BOLD to remove before FC(D) calculation, in seconds
    conf->MAX_COST = get_env_or_default("BNM_MAX_COST", 2); // max cost returned if simulation/fitting is not feasible for set of parameters
    conf->I_SAMPLING_START = get_env_or_default("BNM_I_SAMPLING_START", 1000);
    conf->I_SAMPLING_END = get_env_or_default("BNM_I_SAMPLING_END", 10000);
    conf->I_SAMPLING_DURATION = conf->I_SAMPLING_END - conf->I_SAMPLING_START + 1;
    conf->numerical_fic = get_env_or_default("BNM_NUMERICAL_FIC", 1);
    conf->max_fic_trials_cmaes = get_env_or_default("BNM_MAX_FIC_TRIALS_CMAES", 10);
    conf->max_fic_trials = get_env_or_default("BNM_MAX_FIC_TRIALS", 150);
    conf->init_delta = get_env_or_default("BNM_INIT_DELTA", 0.02); // initial delta in numerical adjustment of FIC
    conf->use_fc_ks = get_env_or_default("BNM_USE_FC_KS", 0);
    conf->use_fc_diff = get_env_or_default("BNM_USE_FC_DIFF", 1);
    conf->exc_interhemispheric = get_env_or_default("BNM_EXC_INTERHEM", 1); // exclude interhemispheric connections from FC(D) calculation and fitting
    conf->drop_edges = get_env_or_default("BNM_DROP_EDGES", 1); // drop the edges of the signal in dFC calculation
    // CMAES config
    conf->sim_verbose = get_env_or_default("BNM_CMAES_SIM_VERBOSE", 0);
    conf->sigma = get_env_or_default("BNM_CMAES_SIGMA", 0.5); // is updated during CMAES, should not be const
    conf->alphacov = get_env_or_default("BNM_CMAES_ALPHACOV", 2);
    conf->Variante = get_env_or_default("BNM_CMAES_VARIANTE", 6);
    conf->gamma_scale = get_env_or_default("BNM_CMAES_GAMMA_SCALE", 2.0);
    conf->bound_soft_edge = get_env_or_default("BNM_CMAES_BOUND_SOFT_EDGE", 0.0);
    conf->early_stop_gens = get_env_or_default("BNM_CMAES_EARLY_STOP_GENS", 30);
    conf->early_stop_tol = get_env_or_default("BNM_CMAES_EARLY_STOP_TOL", 1e-3);
    conf->fic_penalty_scale = get_env_or_default("BNM_CMAES_FIC_PENALTY_SCALE", 2.0);
    conf->fic_reject_failed = get_env_or_default("BNM_CMAES_FIC_REJECT_FAILED", 0);
    conf->scale_max_minmax = get_env_or_default("BNM_SCALE_MAX_MINMAX", 1.0);
    // GPU config
    conf->grid_save_hemis = get_env_or_default("BNM_GPU_SAVE_HEMIS", 1);
    conf->grid_save_dfc = get_env_or_default("BNM_GPU_SAVE_DFC", 0);
    conf->w_IE_1 = false; // set initial w_IE to 1 in all nodes (similar to Deco 2014; ignores w_IE_fic)
}

#endif