#ifndef BASE_HPP
#define BASE_HPP

class BaseModel {
public:
    BaseModel() {};
    virtual ~BaseModel() = default;

    struct Config {
        int bold_remove_s{30};
        bool exc_interhemispheric{true};
        bool drop_edges{true};
        bool sync_msec{false};
        bool extended_output{true};
        bool extended_output_ts{false};
        bool sim_verbose{false};
    };
    
    Config base_conf;

    virtual void set_conf(std::map<std::string, std::string> config_map) {
        set_base_conf(config_map);
    }
    virtual void init_gpu_(
        int *output_ts_p, int *n_pairs_p, int *n_window_pairs_p,
        int N_SIMS, int nodes, int rand_seed,
        int BOLD_TR, int time_steps, int window_size, int window_step,
        BWConstants bwc, bool verbose
    ) = 0;
protected:
    void set_base_conf(std::map<std::string, std::string> config_map) {
        // Note: some of the base_conf members (extended_output, extended_output_ts) 
        // are set directly based on arguments passed from Python
        for (const auto& pair : config_map) {
            if (pair.first == "exc_interhemispheric") {
                this->base_conf.exc_interhemispheric = (bool)std::stoi(pair.second);
            } else if (pair.first == "sync_msec") {
                this->base_conf.sync_msec = (bool)std::stoi(pair.second);
            } else if (pair.first == "sim_verbose") {
                this->base_conf.sim_verbose = (bool)std::stoi(pair.second);
            } else if (pair.first == "bold_remove_s") {
                this->base_conf.bold_remove_s = std::stoi(pair.second);
            } else if (pair.first == "drop_edges") {
                this->base_conf.drop_edges = (bool)std::stoi(pair.second);
            }
        }
    }
};
#endif