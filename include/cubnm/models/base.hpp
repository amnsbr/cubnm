#ifndef BASE_HPP
#define BASE_HPP

extern float get_env_or_default(std::string key, double value_default = 0.0);
extern int get_env_or_default(std::string key, int value_default = 0);

struct ModelConfigs {
    // Simulation config
    int bold_remove_s;
    unsigned int I_SAMPLING_START;
    unsigned int I_SAMPLING_END;
    unsigned int I_SAMPLING_DURATION;
    bool numerical_fic;
    int max_fic_trials;
    u_real init_delta;
    bool exc_interhemispheric;
    bool drop_edges;
    bool sync_msec;
    bool extended_output_ts;
    bool sim_verbose;
    bool fic_verbose;

    // GPU config
    bool grid_save_hemis;
    bool grid_save_dfc;
    bool w_IE_1;    
};

extern void init_conf(ModelConfigs* conf); 
#endif