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
    struct Config {
        bool do_fic{true};
        int max_fic_trials{5};
        bool fic_verbose{false};
        unsigned int I_SAMPLING_START{1000};
        unsigned int I_SAMPLING_END{10000};
        unsigned int I_SAMPLING_DURATION{I_SAMPLING_END - I_SAMPLING_START + 1};
        u_real init_delta{0.02};
    };

    // second, use the boilerplate macro to include
    // the repetitive elements of the class definition
    // this is the simplest solution (that I currently
    // know of) to the issue that CUDA kernels cannot be virtual
    // + some other functions e.g. free_cpu and free_gpu must
    // be redefined so that they operate on derived member variables
    DEFINE_DERIVED_MODEL(
        rWWModel, 
        "rWW", 
        6, 
        7, 
        2, 
        1, 
        3,
        4,
        4,
        true, 
        true,
        false,
        0, 
        0, 
        1, 
        2, 
        1, 
        2, 
        0, 
        0, 
        0, 
        0
    )

    // additional functions that need to be overridden
    // (in addition to h_init, h_step, _j_restart
    // which are always overriden and have to be defined)
    void set_conf(std::map<std::string, std::string> config_map) override;
    void prep_params(u_real ** global_params, u_real ** regional_params, u_real * v_list,
        u_real ** SC, int * SC_indices, u_real * SC_dist,
        bool ** global_out_bool, int ** global_out_int) override final;
    void _j_post_bw_step(
        u_real* _state_vars, u_real* _intermediate_vars,
        int* _ext_int, bool* _ext_bool,
        int* _ext_int_shared, bool* _ext_bool_shared,
        bool& restart,
        u_real* _global_params, u_real* _regional_params,
        int& ts_bold) override final;
    void h_post_bw_step(
        u_real** _state_vars, u_real** _intermediate_vars,
        int** _ext_int, bool** _ext_bool,
        int* _ext_int_shared, bool* _ext_bool_shared,
        bool& restart,
        u_real* _global_params, u_real** _regional_params,
        int& ts_bold) override final;
    void h_post_integration(
        u_real ***state_vars_out, 
        int **global_out_int, bool **global_out_bool,
        u_real* _state_vars, u_real* _intermediate_vars,
        int* _ext_int, bool* _ext_bool,
        int* _ext_int_shared, bool* _ext_bool_shared,
        u_real** global_params, u_real** regional_params,
        u_real* _global_params, u_real* _regional_params,
        int& sim_idx, const int& nodes, int& j) override final;
};

#endif
