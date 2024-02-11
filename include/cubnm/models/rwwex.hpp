#ifndef RWWEX_HPP
#define RWWEX_HPP
class rWWExModel : public BaseModel {
public:
    rWWExModel(
        int nodes, int N_SIMS, int BOLD_TR, int time_steps, bool do_delay, 
        int window_size, int window_step, int rand_seed
        ) : BaseModel(nodes, N_SIMS, BOLD_TR, time_steps, do_delay, window_size, window_step, rand_seed)
    {};

    static constexpr char *name = "rWWEx";
    static constexpr int n_state_vars = 3; // number of state variables (u_real)
    static constexpr int n_intermediate_vars = 2; // number of intermediate/extra u_real variables
    static constexpr int n_noise = 1; // number of noise elements needed for each node
    static constexpr int n_global_params = 1; // G
    static constexpr int n_regional_params = 3; // w, I0, sigma
    static constexpr char* state_var_names[] = {"x", "r", "S"};
    static constexpr char* intermediate_var_names[] = {"axb", "dSdt"};
    static constexpr char* conn_state_var_name = "S"; // name of the state variable which sends input to connected nodes
    static constexpr int conn_state_var_idx = 2;
    static constexpr char* bold_state_var_name = "S"; // name of the state variable which is used for BOLD calculation
    static constexpr int bold_state_var_idx = 2;
    // the following are needed for numerical FIC
    static constexpr int n_ext_int = 0; // number of additional int variables for each node
    static constexpr int n_ext_bool = 0; // number of additional bool variables for each node
    static constexpr int n_global_out_int = 0;
    static constexpr int n_global_out_bool = 0;
    static constexpr int n_global_out_u_real = 0;
    static constexpr int n_regional_out_int = 0;
    static constexpr int n_regional_out_bool = 0;
    static constexpr int n_regional_out_u_real = 0;
    static constexpr bool has_post_bw_step = false;
    static constexpr bool has_post_integration = false;

    struct Constants {
        u_real dt;
        u_real sqrt_dt;
        u_real J_N;
        u_real a;
        u_real b;
        u_real d;
        u_real gamma;
        u_real tau;
        u_real itau;
        u_real dt_itau;
        u_real dt_gamma;
    };

    struct Config {
    };

    static Constants mc;
    Config conf;

    static void init_constants();

    // in set_conf we have nothing to add beyond BaseModel
    // void set_conf(std::map<std::string, std::string> config_map) override;

    CUDA_CALLABLE_MEMBER void init(
        u_real* _state_vars, u_real* _intermediate_vars, 
        int* _ext_int, bool* _ext_bool
    );
    CUDA_CALLABLE_MEMBER void step(
        u_real* _state_vars, u_real* _intermediate_vars,
        u_real* _global_params, u_real* _regional_params,
        u_real& tmp_globalinput,
        u_real* noise, long& noise_idx
    );
    CUDA_CALLABLE_MEMBER void post_bw_step(
        u_real* _state_vars, u_real* _intermediate_vars,
        int* _ext_int, bool* _ext_bool, bool& restart,
        u_real* _global_params, u_real* _regional_params,
        int& ts_bold
    ); // does nothing
    CUDA_CALLABLE_MEMBER void restart(
        u_real* _state_vars, u_real* _intermediate_vars, 
        int* _ext_int, bool* _ext_bool
    );
    CUDA_CALLABLE_MEMBER void post_integration(
        u_real **BOLD, u_real ***state_vars_out, 
        int **global_out_int, bool **global_out_bool,
        u_real* _state_vars, u_real* _intermediate_vars, 
        int* _ext_int, bool* _ext_bool, 
        u_real** global_params, u_real** regional_params,
        u_real* _global_params, u_real* _regional_params,
        int& sim_idx, const int& nodes, int& j
    ); // does nothing

    void init_gpu(BWConstants bwc) override {
        _init_gpu<rWWExModel>(this, bwc);
    }

    void run_simulations_gpu(
        double * BOLD_ex_out, double * fc_trils_out, double * fcd_trils_out,
        u_real ** global_params, u_real ** regional_params, u_real * v_list,
        u_real * SC, u_real * SC_dist
    ) override {
        _run_simulations_gpu<rWWExModel>(
            BOLD_ex_out, fc_trils_out, fcd_trils_out, 
            global_params, regional_params, v_list,
            SC, SC_dist, this
        );
    }

    int get_n_state_vars() override {
        return n_state_vars;
    }
    int get_n_global_out_bool() override {
        return n_global_out_bool;
    }
    int get_n_global_out_int() override {
        return n_global_out_int;
    }
    int get_n_global_params() override {
        return n_global_params;
    }
    int get_n_regional_params() override {
        return n_regional_params;
    }
};

#endif