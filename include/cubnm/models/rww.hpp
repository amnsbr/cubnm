#ifndef RWW_HPP
#define RWW_HPP
class rWWModel {
public:
    static constexpr char *name = "rWW";
    static constexpr int n_state_vars = 6; // number of state variables (u_real)
    static constexpr int n_intermediate_vars = 7; // number of intermediate/extra u_real variables
    static constexpr int n_noise = 2; // number of noise elements needed for each node
    static constexpr int n_global_params = 1;
    static constexpr int n_regional_params = 3;
    static constexpr char* state_var_names[] = {"I_E", "I_I", "r_E", "r_I", "S_E", "S_I"};
    static constexpr char* intermediate_var_names[] = {"aIb_E", "aIb_I", "dSdt_E", "dSdt_I", "mean_I_E", "delta", "I_E_ba_diff"};
    static constexpr char* conn_state_var_name = "S_E"; // name of the state variable which sends input to connected nodes
    static constexpr int conn_state_var_idx = 4;
    static constexpr char* bold_state_var_name = "S_E"; // name of the state variable which is used for BOLD calculation
    static constexpr int bold_state_var_idx = 4;
    // the following are needed for numerical FIC
    static constexpr int n_ext_int = 1; // number of additional int variables for each node; fic_trial
    static constexpr int n_ext_bool = 2; // number of additional bool variables for each node; _adjust_fic, fic_failed
    // static constexpr int n_ext_int_shared = 1; // number of additional int variables shared; TODO: make the fic int and bools shared
    // static constexpr int n_ext_bool_shared = 2; // number of additional bool variables shared
    static constexpr int n_global_out_int = 1; // fic_trials
    static constexpr int n_global_out_bool = 2; // fic_unstable, fic_failed
    static constexpr int n_global_out_u_real = 0; // not implemented
    static constexpr int n_regional_out_int = 0; // not implemented
    static constexpr int n_regional_out_bool = 0; // not implemented
    static constexpr int n_regional_out_u_real = 0; // not implemented
    static constexpr bool has_post_bw_step = true;
    static constexpr bool has_post_integration = true;

    struct Constants {
        u_real dt;
        u_real sqrt_dt;
        u_real J_NMDA;
        u_real a_E;
        u_real b_E;
        u_real d_E;
        u_real a_I;
        u_real b_I;
        u_real d_I;
        u_real gamma_E;
        u_real gamma_I;
        u_real tau_E;
        u_real tau_I;
        u_real sigma_model;
        u_real I_0;
        u_real w_E;
        u_real w_I;
        u_real w_II;
        u_real I_ext;
        u_real w_E__I_0;
        u_real w_I__I_0;
        u_real b_a_ratio_E;
        u_real itau_E;
        u_real itau_I;
        u_real sigma_model_sqrt_dt;
        u_real dt_itau_E;
        u_real dt_gamma_E;
        u_real dt_itau_I;
        u_real dt_gamma_I;
        double tau_E_s;
        double tau_I_s;
        double gamma_E_s;
        double gamma_I_s;
        double r_I_ss;
        double r_E_ss;
        double I_I_ss;
        double I_E_ss;
        double S_I_ss;
        double S_E_ss;
        unsigned int I_SAMPLING_START;
        unsigned int I_SAMPLING_END;
        unsigned int I_SAMPLING_DURATION;
        u_real init_delta;
    };

    struct Config {
        bool do_fic{true};
        int max_fic_trials{5};
        bool fic_verbose{false};
    };

    static Constants mc;
    Config conf; 
    // conf cannot be made static without having a separate device copy
    // because device code cannot access non-static members

    static void init_constants();
    /* as a copy of Constants is going to be available
     in __constant__ memory of device, its values cannot
     be set in the constructor, so we need to have a separate
     function to set the values to the Constants struct copy
     on the host, which will then be copied to the device
     via cudaMemcpyToSymbol
     */

    void set_config(std::map<std::string, std::string> config_map, Config* conf);

    CUDA_CALLABLE_MEMBER void init(
        u_real* _state_vars, u_real* _intermediate_vars, 
        int* _ext_int, bool* _ext_bool
        );
    CUDA_CALLABLE_MEMBER void step(
        u_real* _state_vars, u_real* _intermediate_vars,
        u_real* _global_params, u_real* _regional_params,
        u_real* tmp_globalinput,
        u_real* noise, long* noise_idx
    );
    CUDA_CALLABLE_MEMBER void post_bw_step(
        u_real* _state_vars, u_real* _intermediate_vars,
        int* _ext_int, bool* _ext_bool, bool* restart,
        u_real* _global_params, u_real* _regional_params,
        int* ts_bold
    );
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
        int sim_idx, int nodes, int j
    );
};

#endif