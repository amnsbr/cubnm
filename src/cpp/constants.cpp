#ifndef CONSTANTS_CPP
#define CONSTANTS_CPP
#include "constants.hpp"


float get_env_or_default(std::string key, double value_default) {
    const char* value = std::getenv(key.c_str());
    if (value != nullptr) {
        return atof(value);
    } else {
        return value_default;
    }
}

int get_env_or_default(std::string key, int value_default) {
    const char* value = std::getenv(key.c_str());
    if (value != nullptr) {
        return atoi(value);
    } else {
        return value_default;
    }
}

struct BWConstants bwc;
struct ModelConfigs conf;

void init_bw_constants(struct BWConstants* bwc) {
    bwc->bw_dt = 0.001; // Time-step of Balloon-Windkessel model (s)
    bwc->rho = 0.34;
    bwc->alpha = 0.32;
    bwc->tau = 0.98;
    bwc->y = 1.0/0.41;
    bwc->kappa = 1.0/0.65;
    bwc->V_0 = 0.02 * 100; // Resting blood volume fraction (as %) 
    bwc->k1 = 7 * bwc->rho;
    bwc->k2 = 2.0;
    bwc->k3 = 2 * bwc->rho - 0.2;
    bwc->ialpha = 1.0/bwc->alpha; // some pre-calculations
    bwc->itau = 1.0/bwc->tau;
    bwc->oneminrho = (1.0 - bwc->rho);
    bwc->bw_dt_itau = bwc->bw_dt * bwc->itau;
    bwc->V_0_k1 = bwc->V_0 * bwc->k1;
    bwc->V_0_k2 = bwc->V_0 * bwc->k2;
    bwc->V_0_k3 = bwc->V_0 * bwc->k3;
}

void init_conf(struct ModelConfigs* conf) {
    // TODO: consider making some variables (e.g. bold_remove_s) inputs to
    // run_simulation() instead
    conf->bold_remove_s = get_env_or_default("BNM_BOLD_REMOVE", 30); // length of BOLD to remove before FC(D) calculation, in seconds
    conf->I_SAMPLING_START = get_env_or_default("BNM_I_SAMPLING_START", 1000);
    conf->I_SAMPLING_END = get_env_or_default("BNM_I_SAMPLING_END", 10000);
    conf->I_SAMPLING_DURATION = conf->I_SAMPLING_END - conf->I_SAMPLING_START + 1;
    conf->numerical_fic = get_env_or_default("BNM_NUMERICAL_FIC", 1);
    conf->max_fic_trials = get_env_or_default("BNM_MAX_FIC_TRIALS", 5);
    conf->init_delta = get_env_or_default("BNM_INIT_DELTA", 0.02); // initial delta in numerical adjustment of FIC
    conf->exc_interhemispheric = get_env_or_default("BNM_EXC_INTERHEM", 1); // exclude interhemispheric connections from FC(D) calculation and fitting
    conf->drop_edges = get_env_or_default("BNM_DROP_EDGES", 1); // drop the edges of the signal in dFC calculation
    conf->sync_msec = get_env_or_default("BNM_SYNC_MSEC", 0); // sync nodes every msec vs every 0.1 msec
    conf->extended_output_ts = get_env_or_default("BNM_EXT_OUT_TS", 0); // record time series of extended output instead of their means; only in CPU
    conf->sim_verbose = get_env_or_default("BNM_SIM_VERBOSE", 0); // only in CPU
    conf->fic_verbose = get_env_or_default("BNM_FIC_VERBOSE", 0); // currently is not used
    conf->grid_save_hemis = get_env_or_default("BNM_GPU_SAVE_HEMIS", 1); // currently is not used
    conf->grid_save_dfc = get_env_or_default("BNM_GPU_SAVE_DFC", 0); // currently is not used
    conf->w_IE_1 = false; // set initial w_IE to 1 in all nodes (similar to Deco 2014; ignores w_IE_fic)
}

#endif