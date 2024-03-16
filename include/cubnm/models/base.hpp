#ifndef BASE_HPP
#define BASE_HPP
#include "cubnm/models/bw.hpp" // will be used by all derived models

class BaseModel {
public:
    BaseModel(
        int nodes, int N_SIMS, int BOLD_TR, int time_steps, bool do_delay, 
        int window_size, int window_step, int rand_seed
        ) : nodes{nodes},
            N_SIMS{N_SIMS},
            BOLD_TR{BOLD_TR},
            time_steps{time_steps},
            do_delay{do_delay},
            window_size{window_size},
            window_step{window_step},
            rand_seed{rand_seed}
        {
            u_real TR = (u_real)BOLD_TR / 1000; // TR in seconds
            // calculate length of BOLD time-series
            // +1 to make it inclusive of the last vol
            output_ts = (time_steps / (TR / 0.001))+1;
            bold_size = output_ts * nodes;
        };
    // create virtual destructor and free
    // the memory allocated for the arrays
    virtual ~BaseModel() = default;

    static constexpr char *name = "Base";
    virtual void free_cpu();

    int nodes{}, N_SIMS{}, BOLD_TR{}, time_steps{}, window_size{}, window_step{}, 
        rand_seed{}, n_pairs{}, n_windows{}, n_window_pairs{}, output_ts{}, bold_size{},
        n_vols_remove{}, corr_len{}, noise_size{}, noise_repeats{},
        last_nodes{0}, last_time_steps{0}, last_rand_seed{0}, 
        last_noise_time_steps{0};
        // TODO: make some short or size_t
    bool cpu_initialized{false}, modifies_params{false}, do_delay{};
    
    #ifdef _GPU_ENABLED
    virtual void free_gpu();
    bool gpu_initialized{false};
    #endif

    u_real ***states_out;
    u_real *noise;
    int **global_out_int;
    bool **global_out_bool;
    int *window_starts, *window_ends;
    #ifdef NOISE_SEGMENT
    int *shuffled_nodes, *shuffled_ts;
    #endif
    #ifdef _GPU_ENABLED
    u_real **BOLD, **mean_bold, **ssd_bold, **fc_trils, **windows_mean_bold, **windows_ssd_bold,
        **windows_fc_trils, **windows_mean_fc, **windows_ssd_fc, **fcd_trils,
        *d_SC, **d_global_params, **d_regional_params;
    double **d_fc_trils, **d_fcd_trils;
    int *pairs_i, *pairs_j, *window_pairs_i, *window_pairs_j;
    #endif

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
        bool verbose{false}; // print simulation info + progress
        int progress_interval{500}; // msec; interval for updating progress
        bool serial{false};
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
        std::cout << "exc_interhemispheric: " << base_conf.exc_interhemispheric << std::endl;
        std::cout << "sync_msec: " << base_conf.sync_msec << std::endl;
        std::cout << "verbose: " << base_conf.verbose << std::endl;
        std::cout << "progress_interval: " << base_conf.progress_interval << std::endl;
        std::cout << "bold_remove_s: " << base_conf.bold_remove_s << std::endl;
        std::cout << "drop_edges: " << base_conf.drop_edges << std::endl;
        std::cout << "noise_time_steps: " << base_conf.noise_time_steps << std::endl;
    }

    virtual void set_conf(std::map<std::string, std::string> config_map) {
        set_base_conf(config_map);
    }
    #ifdef _GPU_ENABLED
    virtual void init_gpu(BWConstants bwc) = 0;
    virtual void run_simulations_gpu(
        double * BOLD_ex_out, double * fc_trils_out, double * fcd_trils_out,
        u_real ** global_params, u_real ** regional_params, u_real * v_list,
        u_real * SC, u_real * SC_dist) = 0;
    #endif
    virtual void init_cpu() = 0;
    virtual void run_simulations_cpu(
        double * BOLD_ex_out, double * fc_trils_out, double * fcd_trils_out,
        u_real ** global_params, u_real ** regional_params, u_real * v_list,
        u_real * SC, u_real * SC_dist) = 0;
    virtual void prep_params(u_real ** global_params, u_real ** regional_params, u_real * v_list,
        u_real * SC, u_real * SC_dist,
        bool ** global_out_bool, int ** global_out_int) {};
        // use if additional modification of parameters is needed (e.g. FIC in rWW)
    // declare getters for some of static variables
    // note that as these variables must be static
    // and known at compile time, we cannot define
    // the implementation in the base class
    // the only solution I could think of was to
    // define the (same) implementation in all
    // derived models to make sure they return
    // the static member of that particular model
    // TODO: see if there is a better solution
    virtual int get_n_state_vars() = 0;
    virtual int get_n_global_out_bool() = 0;
    virtual int get_n_global_out_int() = 0;
    virtual int get_n_global_params() = 0;
    virtual int get_n_regional_params() = 0;
    virtual char * get_name() {
        return name;
    }

    virtual void h_init(
        u_real* _state_vars, u_real* _intermediate_vars, 
        int* _ext_int, bool* _ext_bool,
        int* _ext_int_shared, bool* _ext_bool_shared
    ) = 0;
    virtual void h_step(
        u_real* _state_vars, u_real* _intermediate_vars,
        u_real* _global_params, u_real* _regional_params,
        u_real& tmp_globalinput,
        u_real* noise, long& noise_idx
    ) = 0;
    virtual void _j_post_bw_step(
        u_real* _state_vars, u_real* _intermediate_vars,
        int* _ext_int, bool* _ext_bool, 
        int* _ext_int_shared, bool* _ext_bool_shared,
        bool& restart,
        u_real* _global_params, u_real* _regional_params,
        int& ts_bold
    ) {};
    virtual void h_post_bw_step(
        u_real** _state_vars, u_real** _intermediate_vars,
        int** _ext_int, bool** _ext_bool, 
        int* _ext_int_shared, bool* _ext_bool_shared,
        bool& restart,
        u_real* _global_params, u_real** _regional_params,
        int& ts_bold
    ) {
        for (int j=0; j<this->nodes; j++) {
            bool j_restart = false;
            _j_post_bw_step(
                _state_vars[j], _intermediate_vars[j],
                _ext_int[j], _ext_bool[j], 
                _ext_int_shared, _ext_bool_shared,
                j_restart,
                _global_params, _regional_params[j],
                ts_bold
            );
            // restart if any node needs to restart
            restart = restart || j_restart;
        }
    };
    virtual void _j_restart(
        u_real* _state_vars, u_real* _intermediate_vars, 
        int* _ext_int, bool* _ext_bool,
        int* _ext_int_shared, bool* _ext_bool_shared
    ) {};
    virtual void h_restart(
        u_real** _state_vars, u_real** _intermediate_vars, 
        int** _ext_int, bool** _ext_bool,
        int* _ext_int_shared, bool* _ext_bool_shared
    ) {
        for (int j=0; j<this->nodes; j++) {
            _j_restart(
                _state_vars[j], _intermediate_vars[j], 
                _ext_int[j], _ext_bool[j], 
                _ext_int_shared, _ext_bool_shared
            );
        }
    };
    virtual void h_post_integration(
        u_real ***state_vars_out, 
        int **global_out_int, bool **global_out_bool,
        u_real* _state_vars, u_real* _intermediate_vars, 
        int* _ext_int, bool* _ext_bool, 
        int* _ext_int_shared, bool* _ext_bool_shared,
        u_real** global_params, u_real** regional_params,
        u_real* _global_params, u_real* _regional_params,
        int& sim_idx, const int& nodes, int& j
    ) {};

    // In addition to the above, each derived model
    // must also define the following __device__
    // kernels. Note that it is not possible to declare
    // them virtual, as CUDA does not work very well
    // with virtual functions: `init`, `step`, `post_bw_step`,
    // `restart`, `post_integration`. For their arguments see rww.hpp

    virtual gsl_vector * calculate_fc_tril(gsl_matrix * bold);
    virtual gsl_vector * calculate_fcd_tril(gsl_matrix * bold, int * window_starts, int * window_ends);

protected:
    void set_base_conf(std::map<std::string, std::string> config_map) {
        // Note: some of the base_conf members (extended_output, extended_output_ts) 
        // are set directly based on arguments passed from Python
        for (const auto& pair : config_map) {
            if (pair.first == "exc_interhemispheric") {
                this->base_conf.exc_interhemispheric = (bool)std::stoi(pair.second);
            } else if (pair.first == "sync_msec") {
                this->base_conf.sync_msec = (bool)std::stoi(pair.second);
            } else if (pair.first == "verbose") {
                this->base_conf.verbose = (bool)std::stoi(pair.second);
            } else if (pair.first == "bold_remove_s") {
                this->base_conf.bold_remove_s = std::stoi(pair.second);
            } else if (pair.first == "drop_edges") {
                this->base_conf.drop_edges = (bool)std::stoi(pair.second);
            } else if (pair.first == "noise_time_steps") {
                this->base_conf.noise_time_steps = std::stoi(pair.second);
            } else if (pair.first == "progress_interval") {
                this->base_conf.progress_interval = std::stoi(pair.second);
            } else if (pair.first == "serial") {
                this->base_conf.serial = (bool)std::stoi(pair.second);
            }
        }
    }
};

// following templates will be used by each derived model
#ifdef _GPU_ENABLED
template<typename Model>
extern void _run_simulations_gpu(
    double * BOLD_out, double * fc_trils_out, double * fcd_trils_out,
    u_real ** global_params, u_real ** regional_params, u_real * v_list,
    u_real * SC, u_real * SC_dist, BaseModel *model
);

template<typename Model>
extern void _init_gpu(BaseModel *model, BWConstants bwc);
#endif

template <typename Model>
void _run_simulations_cpu(
    double * BOLD_ex_out, double * fc_trils_out, double * fcd_trils_out,
    u_real ** global_params, u_real ** regional_params, u_real * v_list,
    u_real * SC, u_real * SC_dist, BaseModel* m
);

template <typename Model>
void _init_cpu(BaseModel *m);
#endif