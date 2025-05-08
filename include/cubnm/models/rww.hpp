#ifndef RWW_HPP
#define RWW_HPP
#include "cubnm/models/base.hpp"

extern void analytical_fic_het(
        gsl_matrix * sc, double G, double * w_EE, double * w_EI,
        gsl_vector * w_IE_out, bool * _unstable
        );

class rWWModel : public BaseModel {
public:
    // first define Constants and Config structs
    // they always must be defined even if empty
    struct Constants {
        double dt;
        double sqrt_dt;
        double J_NMDA;
        double a_E;
        double b_E;
        double d_E;
        double a_I;
        double b_I;
        double d_I;
        double gamma_E;
        double gamma_I;
        double tau_E;
        double tau_I;
        double sigma_model;
        double I_0;
        double w_E;
        double w_I;
        double w_II;
        double I_ext;
        double w_E__I_0;
        double w_I__I_0;
        double b_a_ratio_E;
        double itau_E;
        double itau_I;
        double sigma_model_sqrt_dt;
        double dt_itau_E;
        double dt_gamma_E;
        double dt_itau_I;
        double dt_gamma_I;
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
    struct Config {
        bool do_fic{true};
        int max_fic_trials{5};
        bool fic_verbose{false};
        unsigned int I_SAMPLING_START{1000};
        unsigned int I_SAMPLING_END{10000};
        unsigned int I_SAMPLING_DURATION{I_SAMPLING_END - I_SAMPLING_START + 1};
        double init_delta{0.02};
    };

    // second, use the boilerplate macro to include
    // the repetitive elements of the class definition
    // this is the simplest solution (that I currently
    // know of) to the issue that CUDA kernels cannot be virtual
    // + some other functions e.g. free_cpu and free_gpu must
    // be redefined so that they operate on derived member variables
    DEFINE_DERIVED_MODEL(
        rWWModel, // CLASS_NAME
        "rWW", // NAME
        6, // STATE_VARS
        7, // INTER_VARS
        2, // NOISE
        1, // GLOBAL_PARAMS
        3, // REGIONAL_PARAMS
        4, // CONN_STATE_VAR_IDX
        4, // BOLD_STATE_VAR_IDX
        true, // HAS_POST_BW
        true, // HAS_POST_INT
        false, // IS_OSC
        0, // EXT_INT
        0, // EXT_BOOL
        1, // EXT_INT_SHARED
        2, // EXT_BOOL_SHARED
        1, // GLOBAL_OUT_INT
        2, // GLOBAL_OUT_BOOL
        0, // GLOBAL_OUT_DOUBLE
        0, // REGIONAL_OUT_INT
        0, // REGIONAL_OUT_BOOL
        0 // REGIONAL_OUT_DOUBLE
    )

    // additional functions that need to be overridden
    // (in addition to h_init, h_step, _j_restart
    // which are always overriden and have to be defined)
    void set_conf(std::map<std::string, std::string> config_map) override;
    void prep_params(double ** global_params, double ** regional_params, double * v_list,
        double ** SC, int * SC_indices, double * SC_dist,
        bool ** global_out_bool, int ** global_out_int) override final;
    void _j_post_bw_step(
        double* _state_vars, double* _intermediate_vars,
        int* _ext_int, bool* _ext_bool,
        int* _ext_int_shared, bool* _ext_bool_shared,
        bool& restart,
        double* _global_params, double* _regional_params,
        int& ts_bold) override final;
    void h_post_bw_step(
        double** _state_vars, double** _intermediate_vars,
        int** _ext_int, bool** _ext_bool,
        int* _ext_int_shared, bool* _ext_bool_shared,
        bool& restart,
        double* _global_params, double** _regional_params,
        int& ts_bold) override final;
    void h_post_integration(
        double ***state_vars_out, 
        int **global_out_int, bool **global_out_bool,
        double* _state_vars, double* _intermediate_vars,
        int* _ext_int, bool* _ext_bool,
        int* _ext_int_shared, bool* _ext_bool_shared,
        double** global_params, double** regional_params,
        double* _global_params, double* _regional_params,
        int& sim_idx, const int& nodes, int& j) override final;
};

#endif
