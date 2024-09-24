#ifndef BASE_HPP
#define BASE_HPP
#include "cubnm/models/bw.hpp" // will be used by all derived models
#include "cubnm/models/boilerplate.hpp"

class BaseModel {
public:
    BaseModel(
        int nodes, int N_SIMS, int N_SCs, int BOLD_TR, int states_sampling,
        int time_steps, bool do_delay, int rand_seed,
        u_real dt, u_real bw_dt
        ) : nodes{nodes},
            N_SIMS{N_SIMS},
            N_SCs{N_SCs},
            BOLD_TR{BOLD_TR},
            states_sampling{states_sampling},
            time_steps{time_steps},
            do_delay{do_delay},
            rand_seed{rand_seed},
            dt{dt}, // in msec
            bw_dt{bw_dt / 1000.0} // input is in msec, but bw_dt is in seconds in the code
        {
            set_bold_states_len();
            set_loop_iters();
        };
    // create virtual destructor and free
    // the memory allocated for the arrays
    virtual ~BaseModel() = default;

    static constexpr char *name = "Base";

    virtual void update(
        int nodes, int N_SIMS, int N_SCs, int BOLD_TR, int states_sampling,
        int time_steps, bool do_delay, int rand_seed,
        u_real dt, u_real bw_dt
        ) {
            this->nodes = nodes;
            this->N_SIMS = N_SIMS;
            this->N_SCs = N_SCs;
            this->BOLD_TR = BOLD_TR;
            this->states_sampling = states_sampling;
            this->time_steps = time_steps;
            this->do_delay = do_delay;
            this->rand_seed = rand_seed;
            this->dt = dt; // msec
            this->bw_dt = bw_dt / 1000.0; // input is in msec, but bw_dt is in seconds in the code
            this->set_bold_states_len();
            this->set_loop_iters();
    }

    virtual void set_bold_states_len() {
        // bold samples (volumes) length
        // and total matrix size
        bold_len = time_steps / BOLD_TR;
        bold_size = bold_len * nodes;
        BOLD_TR_iters = ((u_real)BOLD_TR / 1000.0) / bw_dt;
        // states samples length and
        // total matrix size
        states_len = time_steps / states_sampling;
        states_size = states_len * nodes;
        states_sampling_iters = ((u_real)states_sampling / 1000.0) / bw_dt;
    }

    virtual void set_loop_iters() {
        // TODO: add checks in Python to make sure they are divisible
        bw_it = ((u_real)time_steps / 1000.0) / bw_dt;
        inner_it = (bw_dt * 1000) / dt;
    }

    virtual void free_cpu();

    int nodes{0}, N_SIMS{0}, N_SCs{0}, BOLD_TR{0}, states_sampling{0}, time_steps{0}, 
        rand_seed{0}, n_pairs{0}, n_windows{0}, 
        n_window_pairs{0}, bold_len{0}, bold_size{0}, states_len{0}, states_size{0},
        n_vols_remove{0}, n_states_samples_remove{0}, corr_len{0}, 
        noise_size{0}, noise_repeats{0}, max_delay{0},
        last_nodes{0}, last_time_steps{0}, last_rand_seed{0}, 
        last_noise_time_steps{0},
        bw_it{0}, inner_it{0}, BOLD_TR_iters{0}, states_sampling_iters{0};
        // TODO: make some short or size_t
    bool cpu_initialized{false}, modifies_params{false}, do_delay{false}, co_launch{false};
    u_real dt{0.1}, bw_dt{0.001};
    
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
    int noise_bw_it;
    #endif
    #ifdef _GPU_ENABLED
    u_real **BOLD, **mean_bold, **ssd_bold, **fc_trils, **windows_mean_bold, **windows_ssd_bold,
        **windows_fc_trils, **windows_mean_fc, **windows_ssd_fc, **fcd_trils,
        **d_SC, **d_global_params, **d_regional_params;
    double **d_fc_trils, **d_fcd_trils;
    int *pairs_i, *pairs_j, *window_pairs_i, *window_pairs_j, *d_SC_indices;
    #endif

    struct Config {
        bool verbose{false}; // print simulation info + progress
        bool do_fc{true};
        bool do_fcd{true};
        int bold_remove_s{30};
        bool exc_interhemispheric{true};
        int window_size{10};
        int window_step{2};
        bool drop_edges{true};
        bool ext_out{true};
        bool states_ts{false};
        int noise_time_steps{30000}; // msec
        int progress_interval{500}; // msec; real time interval for updating progress
        bool serial{false};
    };

    Config base_conf;

    void print_config() {
        std::cout << "nodes: " << nodes << std::endl;
        std::cout << "N_SIMS: " << N_SIMS << std::endl;
        std::cout << "N_SCs: " << N_SCs << std::endl;
        std::cout << "BOLD_TR: " << BOLD_TR << std::endl;
        std::cout << "states_sampling: " << states_sampling << std::endl;
        std::cout << "time_steps: " << time_steps << std::endl;
        std::cout << "do_delay: " << do_delay << std::endl;
        std::cout << "rand_seed: " << rand_seed << std::endl;
        std::cout << "exc_interhemispheric: " << base_conf.exc_interhemispheric << std::endl;
        std::cout << "verbose: " << base_conf.verbose << std::endl;
        std::cout << "progress_interval: " << base_conf.progress_interval << std::endl;
        std::cout << "bold_remove_s: " << base_conf.bold_remove_s << std::endl;
        std::cout << "drop_edges: " << base_conf.drop_edges << std::endl;
        std::cout << "ext_out: " << base_conf.ext_out << std::endl;
        std::cout << "do_fc: " << base_conf.do_fc << std::endl;
        std::cout << "do_fcd: " << base_conf.do_fcd << std::endl;
        std::cout << "states_ts: " << base_conf.states_ts << std::endl;
        std::cout << "noise_time_steps: " << base_conf.noise_time_steps << std::endl;
        std::cout << "serial: " << base_conf.serial << std::endl;
    }

    virtual void set_conf(std::map<std::string, std::string> config_map) {
        set_base_conf(config_map);
    }
    #ifdef _GPU_ENABLED
    virtual void init_gpu(BWConstants bwc, bool force_reinit) = 0;
    virtual void run_simulations_gpu(
        double * BOLD_ex_out, double * fc_trils_out, double * fcd_trils_out,
        u_real ** global_params, u_real ** regional_params, u_real * v_list,
        u_real ** SC, int * SC_indices, u_real * SC_dist) = 0;
    #endif
    virtual void init_cpu(bool force_reinit) = 0;
    virtual void run_simulations_cpu(
        double * BOLD_ex_out, double * fc_trils_out, double * fcd_trils_out,
        u_real ** global_params, u_real ** regional_params, u_real * v_list,
        u_real ** SC, int * SC_indices, u_real * SC_dist) = 0;
    virtual void prep_params(u_real ** global_params, u_real ** regional_params, u_real * v_list,
        u_real ** SC, int * SC_indices, u_real * SC_dist,
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
        u_real* _global_params, u_real* _regional_params,
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
        u_real* _global_params, u_real* _regional_params,
        int* _ext_int, bool* _ext_bool,
        int* _ext_int_shared, bool* _ext_bool_shared
    ) {};
    virtual void h_restart(
        u_real** _state_vars, u_real** _intermediate_vars, 
        u_real* _global_params, u_real** _regional_params,
        int** _ext_int, bool** _ext_bool,
        int* _ext_int_shared, bool* _ext_bool_shared
    ) {
        for (int j=0; j<this->nodes; j++) {
            _j_restart(
                _state_vars[j], _intermediate_vars[j],
                _global_params, _regional_params[j],
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
        // Note: some of the base_conf members (ext_out, states_ts) 
        // are set directly based on arguments passed from Python
        for (const auto& pair : config_map) {
            if (pair.first == "verbose") {
                this->base_conf.verbose = (bool)std::stoi(pair.second);
            } else if (pair.first == "do_fc") {
                this->base_conf.do_fc = (bool)std::stoi(pair.second);
            } else if (pair.first == "do_fcd") {
                this->base_conf.do_fcd = (bool)std::stoi(pair.second);
            } else if (pair.first == "bold_remove_s") {
                this->base_conf.bold_remove_s = std::stoi(pair.second);
            } else if (pair.first == "exc_interhemispheric") {
                this->base_conf.exc_interhemispheric = (bool)std::stoi(pair.second);
            } else if (pair.first == "window_size") {
                this->base_conf.window_size = std::stoi(pair.second);
            } else if (pair.first == "window_step") {
                this->base_conf.window_step = std::stoi(pair.second);
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
    u_real ** SC, int * SC_indices, u_real * SC_dist, BaseModel *model
);

template<typename Model>
extern void _init_gpu(BaseModel *model, BWConstants bwc, bool force_reinit);
#endif

template <typename Model>
void _run_simulations_cpu(
    double * BOLD_ex_out, double * fc_trils_out, double * fcd_trils_out,
    u_real ** global_params, u_real ** regional_params, u_real * v_list,
    u_real ** SC, int * SC_indices, u_real * SC_dist, BaseModel* m
);

template <typename Model>
void _init_cpu(BaseModel *m, bool force_reinit);
#endif
