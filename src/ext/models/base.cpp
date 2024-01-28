#include "cubnm/models/base.hpp"
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

ModelConfigs conf;
void init_conf(ModelConfigs* conf) {
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