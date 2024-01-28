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


extern void init_bw_constants(struct BWConstants* bwc);

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

extern void init_conf(struct ModelConfigs* conf);

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
    static constexpr int n_state_vars = 6; // number of state variables
    static constexpr int n_intermediate_vars = 4; // number of intermediate variables
    static constexpr int n_noise = 2; // number of noise elements needed for each node
    static constexpr int n_global_params = 1;
    static constexpr int n_regional_params = 3;
    static constexpr char* state_var_names[] = {"I_E", "I_I", "r_E", "r_I", "S_E", "S_I"};
    static constexpr char* intermediate_var_names[] = {"aIb_E", "aIb_I", "dSdt_E", "dSdt_I"};
    static constexpr char* conn_state_var_name = "S_E"; // name of the state variable which sends input to connected nodes
    static constexpr int conn_state_var_idx = 4;
    static constexpr char* bold_state_var_name = "S_E"; // name of the state variable which is used for BOLD calculation
    static constexpr int bold_state_var_idx = 4;

    rWWConstants mc; // consider having it as a global variable simialr to BWConstants

    rWWModel() {
        init_constants(&mc);
    }

    void init_constants(rWWConstants* mc) {
        mc->dt  = 0.1; // Time-step of synaptic activity model (msec)
        mc->sqrt_dt = SQRT(mc->dt); 
        mc->J_NMDA  = 0.15;
        mc->a_E = 310; // (n/C)
        mc->b_E = 125; // (Hz)
        mc->d_E = 0.16; // (s)
        mc->a_I = 615; // (n/C)
        mc->b_I = 177; // (Hz)
        mc->d_I = 0.087; // (s)
        mc->gamma_E = (u_real)0.641/(u_real)1000.0; // factor 1000 for expressing everything in ms
        mc->gamma_I = (u_real)1.0/(u_real)1000.0; // factor 1000 for expressing everything in ms
        mc->tau_E = 100; // (ms) Time constant of NMDA (excitatory)
        mc->tau_I = 10; // (ms) Time constant of GABA (inhibitory)
        mc->sigma_model = 0.01; // (nA) Noise amplitude (named sigma_model to avoid confusion with CMAES sigma)
        mc->I_0 = 0.382; // (nA) overall effective external input
        mc->w_E = 1.0; // scaling of external input for excitatory pool
        mc->w_I = 0.7; // scaling of external input for inhibitory pool
        mc->w_II = 1.0; // I->I self-coupling
        mc->I_ext = 0.0; // [nA] external input
        mc->w_E__I_0 = mc->w_E * mc->I_0; // pre-calculating some multiplications/divisions
        mc->w_I__I_0 = mc->w_I * mc->I_0;
        mc->b_a_ratio_E = mc->b_E / mc->a_E;
        mc->itau_E = 1.0/mc->tau_E;
        mc->itau_I = 1.0/mc->tau_I;
        mc->sigma_model_sqrt_dt = mc->sigma_model * mc->sqrt_dt;
        mc->dt_itau_E = mc->dt * mc->itau_E;
        mc->dt_gamma_E = mc->dt * mc->gamma_E;
        mc->dt_itau_I = mc->dt * mc->itau_I;
        mc->dt_gamma_I = mc->dt * mc->gamma_I;

        /*
        FIC parameters
        */
        // tau and gamma in seconds (for FIC)
        mc->tau_E_s = 0.1; // [s] (NMDA)
        mc->tau_I_s = 0.01; // [s] (GABA)
        mc->gamma_E_s = 0.641; // kinetic conversion factor (typo in text)
        mc->gamma_I_s = 1.0;
        // Steady-state solutions in isolated case (for FIC)
        mc->r_I_ss = 3.9218448633; // Hz
        mc->r_E_ss = 3.0773270642; // Hz
        mc->I_I_ss = 0.2528951325; // nA
        mc->I_E_ss = 0.3773805650; // nA
        mc->S_I_ss = 0.0392184486; // dimensionless
        mc->S_E_ss = 0.1647572075; // dimensionless
    }

    CUDA_CALLABLE_MEMBER void init(u_real* _state_vars);
    CUDA_CALLABLE_MEMBER void step(
        u_real* _state_vars, u_real* _intermediate_vars,
        u_real* _global_params, u_real* _regional_params,
        u_real* tmp_globalinput,
        u_real* noise, long* noise_idx, rWWModel* model
    );
};

#endif