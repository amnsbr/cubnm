#ifndef BASE_HPP
#define BASE_HPP

class BaseModel {
public:
    BaseModel(
        int nodes, int N_SIMS, int BOLD_TR, int time_steps, bool do_delay, 
        int window_size, int window_step, int rand_seed, bool verbose=false
        ) : nodes{nodes},
            N_SIMS{N_SIMS},
            BOLD_TR{BOLD_TR},
            time_steps{time_steps},
            do_delay{do_delay},
            window_size{window_size},
            window_step{window_step},
            rand_seed{rand_seed},
            verbose{verbose}
        {
            u_real TR = (u_real)BOLD_TR / 1000; // TR in seconds
            // calculate length of BOLD time-series
            // +1 to make it inclusive of the last vol
            output_ts = (time_steps / (TR / 0.001))+1;
            bold_size = output_ts * nodes;
        };
    virtual ~BaseModel() = default;

    static constexpr char *name = "Base";

    int nodes{}, N_SIMS{}, BOLD_TR{}, time_steps{}, window_size{}, window_step{}, 
        rand_seed{}, n_pairs{}, n_windows{}, n_window_pairs{}, output_ts{}, bold_size{},
        n_vols_remove{}, corr_len{}, noise_size{}, noise_repeats{},
        last_nodes{0}, last_time_steps{0}, last_rand_seed{0};
        // TODO: make some short or size_t
    bool verbose{false}, is_initialized{false}, do_delay{};

    struct Config {
        int bold_remove_s{30};
        bool exc_interhemispheric{true};
        bool drop_edges{true};
        bool sync_msec{false};
        bool extended_output{true};
        bool extended_output_ts{false};
        // set a default length of noise segment (msec)
        // (+1 to avoid having an additional repeat for a single time point
        // when time_steps can be divided by 30(000), as the actual duration of
        // simulation (in msec) is always user request time steps + 1)
        int noise_time_steps{30001};
        bool sim_verbose{false}; // not implemented in GPU
    };
    
    Config base_conf;

    void print_config() {
        std::cout << "nodes: " << nodes << std::endl;
        std::cout << "N_SIMS: " << N_SIMS << std::endl;
        std::cout << "BOLD_TR: " << BOLD_TR << std::endl;
        std::cout << "time_steps: " << time_steps << std::endl;
        std::cout << "do_delay: " << do_delay << std::endl;
        std::cout << "window_size: " << window_size << std::endl;
        std::cout << "window_step: " << window_step << std::endl;
        std::cout << "rand_seed: " << rand_seed << std::endl;
        std::cout << "verbose: " << verbose << std::endl;
        std::cout << "exc_interhemispheric: " << base_conf.exc_interhemispheric << std::endl;
        std::cout << "sync_msec: " << base_conf.sync_msec << std::endl;
        std::cout << "sim_verbose: " << base_conf.sim_verbose << std::endl;
        std::cout << "bold_remove_s: " << base_conf.bold_remove_s << std::endl;
        std::cout << "drop_edges: " << base_conf.drop_edges << std::endl;
        std::cout << "noise_time_steps: " << base_conf.noise_time_steps << std::endl;
    }

    virtual void set_conf(std::map<std::string, std::string> config_map) {
        set_base_conf(config_map);
    }
    virtual void init_gpu(BWConstants bwc) = 0;
    virtual void run_simulations_gpu(
        double * BOLD_ex_out, double * fc_trils_out, double * fcd_trils_out,
        u_real ** global_params, u_real ** regional_params, u_real * v_list,
        u_real * SC, gsl_matrix * SC_gsl, u_real * SC_dist) = 0;
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
            } else if (pair.first == "noise_time_steps") {
                this->base_conf.noise_time_steps = std::stoi(pair.second);
            }
        }
    }
};
#endif