/*
This macro can be used to create the definition/declarations of a derived
model. Using a boilerplate macro is the simplest solution (that I currently
know of) to the issue that CUDA kernels cannot be virtual
+ some other functions e.g. free_cpu, free_gpu, etc. must
be redefined so that they operate on derived member variables.
Overall, the goal of using this boilerplat is to make addition of new
models easier and avoid code repetition to some extent.

Note: There are two macros defined with and without GPU
support and most of the code is repeated in the two, therefore
any changes must be made in both macros.

Usage:

// include/cubnm/models/derived_model.hpp

class derivedModel : public BaseModel {
    struct Constants {}; // this is required even if empty
    struct Config {}; // this is required even if empty

    DEFINE_DERIVED_MODEL(<args>) // see below the arguments

    // declare cpu functions that need to be overridden
    // (in addition to h_init, h_step, _j_restart
    // which are always overriden and have to be defined)
    ...
}

The imlpementation of device kernels must be in `src/ext/models/derived_model.cu`
and the implementation of CPU funcitons must be in `src/ext/models/derived_model.cpp`.
The latter must also include a decalartion of a Constants instance (which is static)
and the implementation of init_constants which will define the values of constants.

In addition for GPU usage the following must be defined in
`include/cubnm/models/derived_model.cuh`:

#include "derived_model.hpp"
__constant__ derivedModel::Constants d_derivedc;
template void _run_simulations_gpu<derivedModel>(
    double*, double*, double*, 
    u_real**, u_real**, u_real*, 
    u_real**, int*, u_real*, 
    BaseModel*
);

template void _init_gpu<derivedModel>(BaseModel*, BWConstants);

Lastly on Python side in `src/cubnm/sim.py` a class of `derivedSimGroup`
must be defined.

See rWWEx model as a simple example, and rWW as a more complex one.
*/

#ifdef _GPU_ENABLED
    #define DEFINE_DERIVED_MODEL(CLASS_NAME, NAME, STATE_VARS, INTER_VARS, NOISE, \
                                 GLOBAL_PARAMS, REGIONAL_PARAMS, CONN_STATE_VAR_IDX, \
                                 BOLD_STATE_VAR_IDX, HAS_POST_BW, HAS_POST_INT, IS_OSC, \
                                 EXT_INT, EXT_BOOL, EXT_INT_SHARED, EXT_BOOL_SHARED, \
                                 GLOBAL_OUT_INT, GLOBAL_OUT_BOOL, GLOBAL_OUT_UREAL, \
                                 REGIONAL_OUT_INT, REGIONAL_OUT_BOOL, REGIONAL_OUT_UREAL) \
    using BaseModel::BaseModel; \
    ~CLASS_NAME() { \
        if (cpu_initialized) { \
            this->free_cpu(); \
        } \
        if (gpu_initialized) { \
            this->free_gpu(); \
        } \
    } \
    static constexpr char* name = NAME; \
    static constexpr int n_state_vars = STATE_VARS; \
    static constexpr int n_intermediate_vars = INTER_VARS; \
    static constexpr int n_noise = NOISE; \
    static constexpr int n_global_params = GLOBAL_PARAMS; \
    static constexpr int n_regional_params = REGIONAL_PARAMS; \
    static constexpr int conn_state_var_idx = CONN_STATE_VAR_IDX; \
    static constexpr int bold_state_var_idx = BOLD_STATE_VAR_IDX; \
    static constexpr int n_ext_int = EXT_INT; \
    static constexpr int n_ext_bool = EXT_BOOL; \
    static constexpr int n_ext_int_shared = EXT_INT_SHARED; \
    static constexpr int n_ext_bool_shared = EXT_BOOL_SHARED; \
    static constexpr int n_global_out_int = GLOBAL_OUT_INT; \
    static constexpr int n_global_out_bool = GLOBAL_OUT_BOOL; \
    static constexpr int n_global_out_u_real = GLOBAL_OUT_UREAL; \
    static constexpr int n_regional_out_int = REGIONAL_OUT_INT; \
    static constexpr int n_regional_out_bool = REGIONAL_OUT_BOOL; \
    static constexpr int n_regional_out_u_real = REGIONAL_OUT_UREAL; \
    static constexpr bool has_post_bw_step = HAS_POST_BW; \
    static constexpr bool has_post_integration = HAS_POST_INT; \
    static constexpr bool is_osc = IS_OSC; \
    static Constants mc; \
    Config conf; \
    static void init_constants(u_real dt = 0.1); \
    CUDA_CALLABLE_MEMBER void init( \
        u_real* _state_vars, u_real* _intermediate_vars,  \
        u_real* _global_params, u_real* _regional_params, \
        int* _ext_int, bool* _ext_bool, \
        int* _ext_int_shared, bool* _ext_bool_shared); \
    CUDA_CALLABLE_MEMBER void step( \
        u_real* _state_vars, u_real* _intermediate_vars, \
        u_real* _global_params, u_real* _regional_params, \
        u_real& tmp_globalinput, \
        u_real* noise, long& noise_idx); \
    CUDA_CALLABLE_MEMBER void post_bw_step( \
        u_real* _state_vars, u_real* _intermediate_vars, \
        int* _ext_int, bool* _ext_bool,  \
        int* _ext_int_shared, bool* _ext_bool_shared, \
        bool& restart, \
        u_real* _global_params, u_real* _regional_params, \
        int& ts_bold); \
    CUDA_CALLABLE_MEMBER void restart( \
        u_real* _state_vars, u_real* _intermediate_vars,  \
        u_real* _global_params, u_real* _regional_params, \
        int* _ext_int, bool* _ext_bool, \
        int* _ext_int_shared, bool* _ext_bool_shared); \
    CUDA_CALLABLE_MEMBER void post_integration( \
        u_real ***state_vars_out,  \
        int **global_out_int, bool **global_out_bool, \
        u_real* _state_vars, u_real* _intermediate_vars,  \
        int* _ext_int, bool* _ext_bool,  \
        int* _ext_int_shared, bool* _ext_bool_shared, \
        u_real** global_params, u_real** regional_params, \
        u_real* _global_params, u_real* _regional_params, \
        int& sim_idx, const int& nodes, int& j); \
    void init_gpu(BWConstants bwc, bool force_reinit) override final { \
        _init_gpu<CLASS_NAME>(this, bwc, force_reinit); \
    } \
    void run_simulations_gpu( \
        double * BOLD_ex_out, double * fc_trils_out, double * fcd_trils_out, \
        u_real ** global_params, u_real ** regional_params, u_real * v_list, \
        u_real ** SC, int * SC_indices, u_real * SC_dist) override final { \
        _run_simulations_gpu<CLASS_NAME>( \
            BOLD_ex_out, fc_trils_out, fcd_trils_out,  \
            global_params, regional_params, v_list, \
            SC, SC_indices, SC_dist, this); \
    } \
    void h_init( \
        u_real* _state_vars, u_real* _intermediate_vars,  \
        u_real* _global_params, u_real* _regional_params, \
        int* _ext_int, bool* _ext_bool, \
        int* _ext_int_shared, bool* _ext_bool_shared) override final; \
    void h_step( \
        u_real* _state_vars, u_real* _intermediate_vars, \
        u_real* _global_params, u_real* _regional_params, \
        u_real& tmp_globalinput, \
        u_real* noise, long& noise_idx) override final; \
    void _j_restart( \
        u_real* _state_vars, u_real* _intermediate_vars,  \
        u_real* _global_params, u_real* _regional_params, \
        int* _ext_int, bool* _ext_bool, \
        int* _ext_int_shared, bool* _ext_bool_shared) override final; \
    void init_cpu(bool force_reinit) override final { \
        _init_cpu<CLASS_NAME>(this, force_reinit); \
    } \
    void run_simulations_cpu( \
        double * BOLD_ex_out, double * fc_trils_out, double * fcd_trils_out, \
        u_real ** global_params, u_real ** regional_params, u_real * v_list, \
        u_real ** SC, int * SC_indices, u_real * SC_dist) override final { \
        _run_simulations_cpu<CLASS_NAME>( \
            BOLD_ex_out, fc_trils_out, fcd_trils_out,  \
            global_params, regional_params, v_list, \
            SC, SC_indices, SC_dist, this); \
    } \
    int get_n_state_vars() override final { \
        return n_state_vars; \
    } \
    int get_n_global_out_bool() override final { \
        return n_global_out_bool; \
    } \
    int get_n_global_out_int() override final { \
        return n_global_out_int; \
    } \
    int get_n_global_params() override final { \
        return n_global_params; \
    } \
    int get_n_regional_params() override final { \
        return n_regional_params; \
    } \
    char * get_name() override final { \
        return name; \
    }
#else
    #define DEFINE_DERIVED_MODEL(CLASS_NAME, NAME, STATE_VARS, INTER_VARS, NOISE, \
                                 GLOBAL_PARAMS, REGIONAL_PARAMS, CONN_STATE_VAR_IDX, \
                                 BOLD_STATE_VAR_IDX, HAS_POST_BW, HAS_POST_INT, \
                                 EXT_INT, EXT_BOOL, EXT_INT_SHARED, EXT_BOOL_SHARED, \
                                 GLOBAL_OUT_INT, GLOBAL_OUT_BOOL, GLOBAL_OUT_UREAL, \
                                 REGIONAL_OUT_INT, REGIONAL_OUT_BOOL, REGIONAL_OUT_UREAL) \
    using BaseModel::BaseModel; \
    ~CLASS_NAME() { \
        if (cpu_initialized) { \
            this->free_cpu(); \
        } \
    } \
    static constexpr char* name = NAME; \
    static constexpr int n_state_vars = STATE_VARS; \
    static constexpr int n_intermediate_vars = INTER_VARS; \
    static constexpr int n_noise = NOISE; \
    static constexpr int n_global_params = GLOBAL_PARAMS; \
    static constexpr int n_regional_params = REGIONAL_PARAMS; \
    static constexpr int conn_state_var_idx = CONN_STATE_VAR_IDX; \
    static constexpr int bold_state_var_idx = BOLD_STATE_VAR_IDX; \
    static constexpr int n_ext_int = EXT_INT; \
    static constexpr int n_ext_bool = EXT_BOOL; \
    static constexpr int n_ext_int_shared = EXT_INT_SHARED; \
    static constexpr int n_ext_bool_shared = EXT_BOOL_SHARED; \
    static constexpr int n_global_out_int = GLOBAL_OUT_INT; \
    static constexpr int n_global_out_bool = GLOBAL_OUT_BOOL; \
    static constexpr int n_global_out_u_real = GLOBAL_OUT_UREAL; \
    static constexpr int n_regional_out_int = REGIONAL_OUT_INT; \
    static constexpr int n_regional_out_bool = REGIONAL_OUT_BOOL; \
    static constexpr int n_regional_out_u_real = REGIONAL_OUT_UREAL; \
    static constexpr bool has_post_bw_step = HAS_POST_BW; \
    static constexpr bool has_post_integration = HAS_POST_INT; \
    static Constants mc; \
    Config conf; \
    static void init_constants(u_real dt = 0.1); \
    void h_init( \
        u_real* _state_vars, u_real* _intermediate_vars,  \
        u_real* _global_params, u_real* _regional_params, \
        int* _ext_int, bool* _ext_bool, \
        int* _ext_int_shared, bool* _ext_bool_shared) override final; \
    void h_step( \
        u_real* _state_vars, u_real* _intermediate_vars, \
        u_real* _global_params, u_real* _regional_params, \
        u_real& tmp_globalinput, \
        u_real* noise, long& noise_idx) override final; \
    void _j_restart( \
        u_real* _state_vars, u_real* _intermediate_vars,  \
        u_real* _global_params, u_real* _regional_params, \
        int* _ext_int, bool* _ext_bool, \
        int* _ext_int_shared, bool* _ext_bool_shared) override final; \
    void init_cpu(bool force_reinit) override final { \
        _init_cpu<CLASS_NAME>(this, force_reinit); \
    } \
    void run_simulations_cpu( \
        double * BOLD_ex_out, double * fc_trils_out, double * fcd_trils_out, \
        u_real ** global_params, u_real ** regional_params, u_real * v_list, \
        u_real ** SC, int * SC_indices, u_real * SC_dist) override final { \
        _run_simulations_cpu<CLASS_NAME>( \
            BOLD_ex_out, fc_trils_out, fcd_trils_out,  \
            global_params, regional_params, v_list, \
            SC, SC_indices, SC_dist, this); \
    } \
    int get_n_state_vars() override final { \
        return n_state_vars; \
    } \
    int get_n_global_out_bool() override final { \
        return n_global_out_bool; \
    } \
    int get_n_global_out_int() override final { \
        return n_global_out_int; \
    } \
    int get_n_global_params() override final { \
        return n_global_params; \
    } \
    int get_n_regional_params() override final { \
        return n_regional_params; \
    } \
    char * get_name() override final { \
        return name; \
    }
#endif