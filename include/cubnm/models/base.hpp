#ifndef BASE_HPP
#define BASE_HPP

extern float get_env_or_default(std::string key, double value_default = 0.0);
extern int get_env_or_default(std::string key, int value_default = 0);

struct ModelConfigs {
    // Simulation config
    int bold_remove_s;
    bool exc_interhemispheric;
    bool drop_edges;
    bool sync_msec;
    bool extended_output_ts;
    bool sim_verbose;
};

extern void init_conf(ModelConfigs* conf); 
#endif