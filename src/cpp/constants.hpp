#ifndef CONSTANTS_HPP
#define CONSTANTS_HPP

// Using floats vs doubles in the simulations, pass -D USE_FLOATS=1 to compiler to use floats
// Note that this does not affect FIC, CMAES and GOF calculations (always using doubles)
// as well as the noise array (always generated as floats, but then casted to u_real)
#ifdef USE_FLOATS
    typedef float u_real;
    #define EXP expf
    #define POW powf
    #define SQRT sqrtf
    #define CUDA_MAX fmaxf
    #define CUDA_MIN fminf
#else
    typedef double u_real;
    #define EXP exp
    #define POW pow
    #define SQRT sqrt
    #define CUDA_MAX max
    #define CUDA_MIN min
#endif

#ifndef MANY_NODES
    #define MAX_NODES 500
#else
    // this is just an arbitrary number
    // and it is not guaranteed that the code will work 
    // with this many nodes or won't work with more nodes
    #define MAX_NODES 10000
#endif

extern float get_env_or_default(std::string key, double value_default = 0.0);
extern int get_env_or_default(std::string key, int value_default = 0);

struct BWConstants {
    u_real bw_dt;
    u_real rho;
    u_real alpha;
    u_real tau;
    u_real y;
    u_real kappa;
    u_real V_0;
    u_real k1;
    u_real k2;
    u_real k3;
    u_real ialpha;
    u_real itau;
    u_real oneminrho;
    u_real bw_dt_itau;
    u_real V_0_k1;
    u_real V_0_k2;
    u_real V_0_k3;
};

struct rWWConstants {
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
};

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

extern void init_bw_constants(BWConstants* bwc);
extern void init_rWW_constants(rWWConstants* rWWc);
extern void init_conf(ModelConfigs* conf);

#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __device__
// #define CUDA_CONSTANT __constant__
#else
#define CUDA_CALLABLE_MEMBER
// #define CUDA_CONSTANT
#endif 

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

    bool do_fic;
    bool adjust_fic;
    int max_fic_trials;

    CUDA_CALLABLE_MEMBER void init(
        u_real* _state_vars, u_real* _intermediate_vars, 
        int* _ext_int, bool* _ext_bool, rWWModel* model
        );
    CUDA_CALLABLE_MEMBER void step(
        u_real* _state_vars, u_real* _intermediate_vars,
        u_real* _global_params, u_real* _regional_params,
        u_real* tmp_globalinput,
        u_real* noise, long* noise_idx, rWWModel* model
    );
    CUDA_CALLABLE_MEMBER void post_bw_step(
        u_real* _state_vars, u_real* _intermediate_vars,
        int* _ext_int, bool* _ext_bool, bool* restart,
        u_real* _global_params, u_real* _regional_params,
        int* ts_bold, rWWModel* model
    );
    CUDA_CALLABLE_MEMBER void restart(
        u_real* _state_vars, u_real* _intermediate_vars, 
        int* _ext_int, bool* _ext_bool, rWWModel* model
        );
    CUDA_CALLABLE_MEMBER void post_integration(
        u_real **BOLD, u_real ***state_vars_out, 
        int **global_out_int, bool **global_out_bool,
        u_real* _state_vars, u_real* _intermediate_vars, 
        int* _ext_int, bool* _ext_bool, 
        u_real** global_params, u_real** regional_params,
        u_real* _global_params, u_real* _regional_params,
        int sim_idx, int nodes, int j,
        rWWModel* model
    );
};

#endif