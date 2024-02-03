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
    conf->exc_interhemispheric = get_env_or_default("BNM_EXC_INTERHEM", 1); // exclude interhemispheric connections from FC(D) calculation and fitting
    conf->drop_edges = get_env_or_default("BNM_DROP_EDGES", 1); // drop the edges of the signal in dFC calculation
    conf->sync_msec = get_env_or_default("BNM_SYNC_MSEC", 0); // sync nodes every msec vs every 0.1 msec
    conf->extended_output_ts = get_env_or_default("BNM_EXT_OUT_TS", 0); // record time series of extended output instead of their means; only in CPU
    conf->sim_verbose = get_env_or_default("BNM_SIM_VERBOSE", 0); // only in CPU
}