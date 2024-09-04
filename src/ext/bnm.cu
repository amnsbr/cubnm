/*
Reduced Wong-Wang model (Deco 2014) simulation on GPU

This code includes kernels needed for the simulation of BOLD signal and calculation of 
FC and FCD, in addition to other GPU-related functions.

Each simulation (for a given set of parameters) is run on a single GPU block, and
each thread in the block simulates one region of the brain. The threads in the block
form a "cooperative group" and are synchronized after each integration time step.
Calculation of FC and FCD are also similarly parallelized across simulations and region or
window pairs.

Parts of this code are based on https://github.com/BrainModes/The-Hybrid-Virtual-Brain, 
https://github.com/murraylab/hbnm & https://github.com/decolab/cb-neuromod

Author: Amin Saberi, Feb 2023
*/
#include "cubnm/includes.cuh"
#include "cubnm/defines.h"
#include "cubnm/utils.cuh"
#include "cubnm/models/bw.cuh"
#include "cubnm/models/base.cuh"
#include "cubnm/fc.cuh"
#include "cubnm/bnm.cuh"
#include "cubnm/models/rww.cuh"
#include "cubnm/models/rwwex.cuh"
#include "cubnm/models/kuramoto.cuh"
// other models go here

cudaDeviceProp prop;

__device__ void global_input_cond(
        u_real& tmp_globalinput, int& k_buff_idx,
        const int& nodes, const int& sim_idx, const int& SC_idx,
        const int& j, int& k, int& buff_idx, u_real** SC, 
        int** delay, const bool& has_delay, const int& max_delay,
        u_real** conn_state_var_hist, u_real* conn_state_var_1
        ) {
    // calculates global input from other nodes `k` to current node `j`
    // Note: this will not skip over self-connections
    // if they should be ignored, their SC should be set to 0
    // Note 2: In SC and delay rows must be sources (k) and columns must
    // be targets (j), so that among threads of the same warp,
    // memory read is coalesced, such that at each k, SCs of all
    // k->j connections are adjacent in memory.
    // This is very important for performance especially
    // in higher number of nodes.
    tmp_globalinput = 0;
    if (has_delay) {
        for (k=0; k<nodes; k++) {
            // calculate correct index of the other region in the buffer based on j-k delay
            // buffer is moving backward, therefore the delay timesteps back in history
            // will be in +delay time steps in the buffer (then modulo max_delay as it is circular buffer)
            k_buff_idx = (buff_idx + delay[sim_idx][k*nodes+j]) % max_delay;
            tmp_globalinput += SC[SC_idx][k*nodes+j] * conn_state_var_hist[sim_idx][k_buff_idx*nodes+k];
        }
    } else {
        for (k=0; k<nodes; k++) {
            tmp_globalinput += SC[SC_idx][k*nodes+j] * conn_state_var_1[k];
        }            
    }
}

__device__ void global_input_osc(
        u_real& tmp_globalinput, int& k_buff_idx,
        const int& nodes, const int& sim_idx, const int& SC_idx,
        const int& j, int& k, int& buff_idx, u_real** SC, 
        int** delay, const bool& has_delay, const int& max_delay,
        u_real** conn_state_var_hist, u_real* conn_state_var_1
        ) {
    // calculates global input from other nodes `k` to current node `j`
    // See notes in global_input_cond
    tmp_globalinput = 0;
    if (has_delay) {
        for (k=0; k<nodes; k++) {
            // calculate correct index of the other region in the buffer based on j-k delay
            // buffer is moving backward, therefore the delay timesteps back in history
            // will be in +delay time steps in the buffer (then modulo max_delay as it is circular buffer)
            k_buff_idx = (buff_idx + delay[sim_idx][k*nodes+j]) % max_delay;
            tmp_globalinput += SC[SC_idx][k*nodes+j] * SIN(conn_state_var_hist[sim_idx][k_buff_idx*nodes+k] - conn_state_var_hist[sim_idx][buff_idx*nodes+j]);
        }
    } else {
        for (k=0; k<nodes; k++) {
            tmp_globalinput += SC[SC_idx][k*nodes+j] * SIN(conn_state_var_1[k] - conn_state_var_1[j]);
        }            
    }
}

template<typename Model>
__global__ void bnm(
        Model* model, u_real **BOLD, u_real ***states_out, 
        int **global_out_int, bool **global_out_bool,
        u_real **SC, int *SC_indices, 
        u_real **global_params, u_real **regional_params,
        u_real **conn_state_var_hist, int **delay, int max_delay,
        #ifdef NOISE_SEGMENT
        int *shuffled_nodes, int *shuffled_ts,
        #endif
        u_real *noise, uint* progress
    ) {
    // convert block to a cooperative group
    // get simulation and node indices
    int sim_idx = blockIdx.x;
    if (sim_idx >= model->N_SIMS) return;
    int j = threadIdx.x;
    if (j >= model->nodes) return;

    extern __shared__ u_real _shared_mem[];

    // copy variables used in the loop to local memory
    const int nodes = model->nodes;
    const int time_steps = model->time_steps;
    const int BOLD_TR = model->BOLD_TR;
    const int states_sampling = model->states_sampling;
    const bool ext_out = model->base_conf.ext_out;
    const bool states_ts = model->base_conf.states_ts;
    const bool sync_msec = model->base_conf.sync_msec;
    const int SC_idx = SC_indices[sim_idx];

    // set up noise shuffling if indicated
    #ifdef NOISE_SEGMENT
    /* 
    How noise shuffling works?
    At each time point we will have `ts_bold` which is the real time (in msec) 
    from the start of simulation, `ts_bold % noise_time_steps` which is the real 
    time passed within  each repeat of the noise segment (`curr_noise_repeat`), 
    `sh_ts_noise` which is the shuffled timepoint (column of the noise segment) 
    that will be used for getting the noise of nodes * 10 int_i * 2 neurons for 
    the current msec. 
    Similarly, in each thread we have `j` which is mapped to a `sh_j` which will 
    vary in each repeat.
    */
    const int noise_time_steps = model->base_conf.noise_time_steps;
    // get position of the node
    // in shuffled nodes for the first
    // repeat of noise segment
    int sh_j = shuffled_nodes[j];
    int curr_noise_repeat = 0;
    int sh_ts_noise = shuffled_ts[0];;
    #endif

    // determine the parameters of current simulation and node
    // use __shared__ for parameters that are shared
    // between regions in the same simulation, e.g. G, but
    // not for those that (may) vary, e.g. w_IE, w_EE and w_IE
    __shared__ u_real _global_params[Model::n_global_params];
    u_real _regional_params[Model::n_regional_params];
    int ii; // general-purpose index for parameters and varaiables
    for (ii=0; ii<Model::n_global_params; ii++) {
        _global_params[ii] = global_params[ii][sim_idx];
    }
    for (ii=0; ii<Model::n_regional_params; ii++) {
        _regional_params[ii] = regional_params[ii][sim_idx*nodes+j];
    }

    // initialize extended output sums
    if (ext_out && (!states_ts)) {
        for (ii=0; ii<Model::n_state_vars; ii++) {
            states_out[ii][sim_idx][j] = 0;
        }
    }

    // declare state variables, intermediate variables
    // and additional ints and bools
    u_real _state_vars[Model::n_state_vars];
    u_real _intermediate_vars[Model::n_intermediate_vars];
    // note: with this implementation n_state_vars and n_intermediate_vars
    // cannot be 0. Given they are frequently accessed it offers performance
    // improvement (as opposed to keeping them on heap). If a model does not
    // have any intermediate variables (having no states is undefined), 
    // n_intermediat_vars must be set to 1, but will be ignored in the model.
    // but keep the less frequently used variables on heap, while handling zero size
    int* _ext_int = (Model::n_ext_int > 0) ? (int*)malloc(Model::n_ext_int * sizeof(int)) : NULL;
    bool* _ext_bool = (Model::n_ext_bool > 0) ? (bool*)malloc(Model::n_ext_bool * sizeof(bool)) : NULL;
    int* _ext_int_shared = (int*)_shared_mem;
    bool* _ext_bool_shared = (bool*)(_shared_mem + Model::n_ext_int_shared*sizeof(int));
    // initialize model
    model->init(
        _state_vars, _intermediate_vars,
        _global_params, _regional_params,
        _ext_int, _ext_bool, 
        _ext_int_shared, _ext_bool_shared
    );


    // Ballon-Windkessel model variables
    u_real bw_x, bw_f, bw_nu, bw_q, tmp_f;
    bw_x = 0.0;
    bw_f = 1.0;
    bw_nu = 1.0;
    bw_q = 1.0;

    // delay setup
    const bool has_delay = (max_delay > 0);
    // if there is delay use a circular buffer (conn_state_var_hist)
    // and keep track of current buffer index (will be the same
    // in all nodes at each time point). Start from the end and
    // go backwards. 
    // Note that conn_state_var_hist is pseudo-2d
    int buff_idx, k_buff_idx;
    if (has_delay) {
        // initialize conn_state_var_hist in every time point at initial value
        for (buff_idx=0; buff_idx<max_delay; buff_idx++) {
            conn_state_var_hist[sim_idx][buff_idx*nodes+j] = _state_vars[Model::conn_state_var_idx];
        }
        buff_idx = max_delay-1;
    }

    // store immediate history of conn_state_var on extern shared memory
    // the memory is allocated dynamically based on the number of nodes
    // (see https://developer.nvidia.com/blog/using-shared-memory-cuda-cc/)
    u_real *_conn_state_var_1 = (u_real*)(_shared_mem + Model::n_ext_int_shared*sizeof(int)+Model::n_ext_bool_shared*sizeof(bool));
    u_real *conn_state_var_1;
    if (!(has_delay)) {
        // conn_state_var_1 is only used when
        // there is no delay
        if (nodes <= MAX_NODES_REG) {
            // for lower number of nodes
            // use shared memory for conn_state_var_1
            conn_state_var_1 = _conn_state_var_1;
        }
        #ifdef MANY_NODES
        else {
            // otherwise use global memory
            // allocated to conn_state_var_hist
            conn_state_var_1 = conn_state_var_hist[sim_idx];
        }
        // the else case only occurs if
        // MANY_NODES is defined but is also wrapped in
        // #ifdef MANY_NODES, as otherwise it'll hurt performance
        // of nodes <= MAX_NODES_REG simulations
        // (by making some compiler optimizations not possible)
        #endif
        conn_state_var_1[j] = _state_vars[Model::conn_state_var_idx];
    }

    // determine the global input function
    GlobalInputKernel global_input_kernel;
    if (Model::is_osc) {
        global_input_kernel = &global_input_osc;
    } else {
        global_input_kernel = &global_input_cond;
    }

    // this will determine if the simulation should be restarted
    // (e.g. if FIC adjustment fails in rWW)
    __shared__ bool restart;
    restart = false;

    // integration loop
    u_real tmp_globalinput = 0.0;
    int int_i = 0;
    int k = 0;
    long noise_idx = 0;
    int bold_i = 0;
    int states_i = 0;
    int ts_bold = 0;
    // outer loop for 1 msec steps
    // TODO: define number of steps for outer
    // and inner loops based on model dt and BW dt from user input 
    while (ts_bold < time_steps) {
        #ifdef NOISE_SEGMENT
        // get shuffled timepoint corresponding to
        // current noise repeat and the amount of time
        // past in the repeat
        sh_ts_noise = shuffled_ts[(ts_bold % noise_time_steps)+(curr_noise_repeat*noise_time_steps)];
        #endif
        if (sync_msec) {
            // calculate global input every 1 ms
            // (will be fixed through 10 steps of the millisecond)
            // note that a sync call is needed before using
            // and updating S_i_1_E, otherwise the current
            // thread might access S_E of other nodes at wrong
            // times (t or t-2 instead of t-1)
            __syncthreads();
            global_input_kernel(
                tmp_globalinput, k_buff_idx,
                nodes, sim_idx, SC_idx,
                j, k, buff_idx, SC, 
                delay, has_delay, max_delay,
                conn_state_var_hist, conn_state_var_1
            );
        }
        // inner loop for 0.1 msec steps
        for (int_i = 0; int_i < 10; int_i++) {
            if (!sync_msec) {
                // calculate global input every 0.1 ms
                __syncthreads();
                global_input_kernel(
                    tmp_globalinput, k_buff_idx,
                    nodes, sim_idx, SC_idx,
                    j, k, buff_idx, SC, 
                    delay, has_delay, max_delay,
                    conn_state_var_hist, conn_state_var_1
                );
            }
            // equations
            #ifdef NOISE_SEGMENT
            noise_idx = (((sh_ts_noise * 10 + int_i) * nodes * Model::n_noise) + (sh_j * Model::n_noise));
            #else
            noise_idx = (((ts_bold * 10 + int_i) * nodes * Model::n_noise) + (j * Model::n_noise));
            #endif
            model->step(
                _state_vars, _intermediate_vars, 
                _global_params, _regional_params,
                tmp_globalinput,
                noise, noise_idx
            );
            if (!sync_msec) {
                if (has_delay) {
                    // wait for other regions before updating so other
                    // nodes do not access S_i_E of current node at wrong times
                    __syncthreads();
                    conn_state_var_hist[sim_idx][buff_idx*nodes+j] = _state_vars[Model::conn_state_var_idx];
                    // go one time step backward in the buffer for the next time point
                    buff_idx = (buff_idx + max_delay - 1) % max_delay;
                    
                } else  {
                    // wait for other regions before updating so other
                    // nodes do not access S_i_E of current node at wrong times
                    __syncthreads();
                    conn_state_var_1[j] = _state_vars[Model::conn_state_var_idx];
                }
            }
        }

        if (sync_msec) {
            if (has_delay) {
                // wait for other regions before updating so other
                // nodes do not access S_i_E of current node at wrong times
                __syncthreads();
                conn_state_var_hist[sim_idx][buff_idx*nodes+j] = _state_vars[Model::conn_state_var_idx];
                // go one time step backward in the buffer for the next time point
                buff_idx = (buff_idx + max_delay - 1) % max_delay;
            } else {
                // wait for other regions before updating so other
                // nodes do not access S_i_E of current node at wrong times
                __syncthreads();
                conn_state_var_1[j] = _state_vars[Model::conn_state_var_idx];
            }
        }

        // Balloon-Windkessel model equations here since its
        // dt is 1 msec
        bw_step(bw_x, bw_f, bw_nu, 
            bw_q, tmp_f,
            _state_vars[Model::bold_state_var_idx]);

        // Calculate and write BOLD to managed memory every TR
        if ((ts_bold+1) % BOLD_TR == 0) {
            // calcualte and save BOLD
            BOLD[sim_idx][bold_i*nodes+j] = d_bwc.V_0_k1 * (1 - bw_q) + d_bwc.V_0_k2 * (1 - bw_q/bw_nu) + d_bwc.V_0_k3 * (1 - bw_nu);
            // update progress
            bold_i++;
            if (model->base_conf.verbose && (j==0)) {
                atomicAdd(progress, 1);
            }

        }

        if (ext_out) {
            if ((ts_bold+1) % states_sampling == 0) {
                // save time series of extended output if indicated
                if (states_ts) {
                    for (ii=0; ii<Model::n_state_vars; ii++) {
                        states_out[ii][sim_idx][states_i*nodes+j] = _state_vars[ii];
                    }
                } else {
                    // update sum (later mean) of extended
                    // output only after n_samples_remove_states
                    if (states_i >= model->n_states_samples_remove) {
                        for (ii=0; ii<Model::n_state_vars; ii++) {
                            states_out[ii][sim_idx][j] += _state_vars[ii];
                        }
                    }
                }
                states_i++;
            }
        }
        



        #ifdef NOISE_SEGMENT
        // reset noise segment time 
        // and shuffle nodes if the segment
        // has reached to the end
        if ((ts_bold+1) % noise_time_steps == 0) {
            // at the last time point don't do this
            // to avoid going over the extent of shuffled_nodes
            if (ts_bold+1 < time_steps) {
                curr_noise_repeat++;
                sh_j = shuffled_nodes[curr_noise_repeat*nodes+j];
            }
        }
        #endif

        if (Model::has_post_bw_step) {
            model->post_bw_step(
                _state_vars, _intermediate_vars,
                _ext_int, _ext_bool, 
                _ext_int_shared, _ext_bool_shared,
                restart,
                _global_params, _regional_params,
                ts_bold
            );
        }

        // move forward outer bw loop
        // this has to be before restart
        // because restart will reset ts_bold to 0
        ts_bold++;

        // if restart is indicated (e.g. FIC failed in rWW)
        // reset the simulation and start from the beginning
        if (restart) {
            // model-specific restart
            model->restart(
                _state_vars, _intermediate_vars, 
                _global_params, _regional_params,
                _ext_int, _ext_bool, 
                _ext_int_shared, _ext_bool_shared
            );
            // subtract progress of current simulation
            if (model->base_conf.verbose && (j==0)) {
                atomicAdd(progress, -bold_i);
            }
            // reset indices
            bold_i = 0;
            states_i = 0;
            ts_bold = 0;
            // reset Balloon-Windkessel model variables
            bw_x = 0.0;
            bw_f = 1.0;
            bw_nu = 1.0;
            bw_q = 1.0;
            if (has_delay) {
                // initialize conn_state_var_hist in every time point at initial value
                for (buff_idx=0; buff_idx<max_delay; buff_idx++) {
                    conn_state_var_hist[sim_idx][buff_idx*nodes+j] = _state_vars[Model::conn_state_var_idx];
                }
                // reset delay buffer index
                buff_idx = max_delay-1;
            } else {
                // reset conn_state_var_1
                conn_state_var_1[j] = _state_vars[Model::conn_state_var_idx];
            }
            #ifdef NOISE_SEGMENT
            // reset the node shuffling
            sh_j = shuffled_nodes[j];
            curr_noise_repeat = 0;
            #endif
            restart = false; // restart is done
            __syncthreads(); // make sure all threads are in sync after restart
        }
    }

    if (Model::has_post_integration) {
        model->post_integration(
            states_out, global_out_int, global_out_bool,
            _state_vars, _intermediate_vars, 
            _ext_int, _ext_bool, 
            _ext_int_shared, _ext_bool_shared,
            global_params, regional_params,
            _global_params, _regional_params,
            sim_idx, nodes, j
        );
    }
    if (ext_out && (!states_ts)) {
        // take average across time points after n_states_samples_remove
        int ext_out_time_points = states_i - model->n_states_samples_remove;
        for (ii=0; ii<Model::n_state_vars; ii++) {
            states_out[ii][sim_idx][j] /= ext_out_time_points;
        }
    }
    // free heap
    if (Model::n_ext_int > 0) {
        free(_ext_int);
    }
    if (Model::n_ext_bool > 0) {
        free(_ext_bool);
    }
}


template<typename Model>
__global__ void bnm_serial(
        Model* model, u_real **BOLD, u_real ***states_out, 
        int **global_out_int, bool **global_out_bool,
        u_real **SC, int *SC_indices, u_real **global_params, u_real **regional_params,
        u_real **conn_state_var_hist, int **delay, int max_delay,
        #ifdef NOISE_SEGMENT
        int *shuffled_nodes, int *shuffled_ts,
        #endif
        u_real *noise, uint* progress
    ) {
    // get simulation index
    int sim_idx = blockIdx.x;
    if (sim_idx >= model->N_SIMS) return;
    int j{0};

    // shared memory will be used for _ext_int and _ext_bool
    extern __shared__ u_real _shared_mem[];

    // copy variables used in the loop to local memory
    const int nodes = model->nodes;
    const int time_steps = model->time_steps;
    const int BOLD_TR = model->BOLD_TR;
    const bool ext_out = model->base_conf.ext_out;
    const bool states_ts = model->base_conf.states_ts;
    const bool sync_msec = model->base_conf.sync_msec;
    const int SC_idx = SC_indices[sim_idx];

    // set up noise shuffling if indicated
    #ifdef NOISE_SEGMENT
    const int noise_time_steps = model->base_conf.noise_time_steps;
    int sh_j{0};
    int curr_noise_repeat{0};
    int sh_ts_noise = shuffled_ts[0];
    #endif

    // copy parameters of current simulation to device heap memory
    u_real *_global_params = (u_real*)malloc(Model::n_global_params * sizeof(u_real));
    int ii; // general-purpose index for parameters and varaiables
    for (ii=0; ii<Model::n_global_params; ii++) {
        _global_params[ii] = global_params[ii][sim_idx];
    }
    u_real **_regional_params = (u_real**)malloc(nodes * sizeof(u_real*));
    for (j = 0; j < nodes; j++) {
        _regional_params[j] = (u_real*)malloc(Model::n_regional_params * sizeof(u_real));
        for (ii = 0; ii < Model::n_regional_params; ii++) {
            _regional_params[j][ii] = regional_params[ii][sim_idx * nodes + j];
        }
    }

    // initialize extended output sums
    if (ext_out && (!states_ts)) {
        for (ii=0; ii<Model::n_state_vars; ii++) {
            for (j=0; j<nodes; j++) {
                states_out[ii][sim_idx][j] = 0;
            }
        }
    }

    // allocated state variables, intermediate variables
    // and additional ints and bools on device heap
    u_real ** _state_vars = (u_real**)malloc(nodes * sizeof(u_real*));
    for (j=0; j<nodes; j++) {
        _state_vars[j] = (u_real*)malloc(Model::n_state_vars * sizeof(u_real));
    }
    u_real ** _intermediate_vars = (u_real**)malloc(nodes * sizeof(u_real*));
    for (j=0; j<nodes; j++) {
        _intermediate_vars[j] = (Model::n_intermediate_vars > 0) ? (u_real*)malloc(Model::n_intermediate_vars * sizeof(u_real)) : NULL;
    }
    int ** _ext_int = (int**)malloc(nodes * sizeof(int*));
    for (j=0; j<nodes; j++) {
        _ext_int[j] = (Model::n_ext_int > 0) ? (int*)malloc(Model::n_ext_int * sizeof(int)) : NULL;
    }
    bool ** _ext_bool = (bool**)malloc(nodes * sizeof(bool*));
    for (j=0; j<nodes; j++) {
        _ext_bool[j] = (Model::n_ext_bool > 0) ? (bool*)malloc(Model::n_ext_bool * sizeof(bool)) : NULL;
    }
    // shared variables    
    int* _ext_int_shared = (int*)_shared_mem;
    bool* _ext_bool_shared = (bool*)(_shared_mem + Model::n_ext_int_shared*sizeof(int));
    // initialize model
    for (j=0; j<nodes; j++) {
        model->init(
            _state_vars[j], _intermediate_vars[j],
            _global_params, _regional_params[j],
            _ext_int[j], _ext_bool[j], 
            _ext_int_shared, _ext_bool_shared
        );
    }


    // Ballon-Windkessel model variables
    u_real* bw_x = (u_real*)malloc(nodes * sizeof(u_real));
    u_real* bw_f = (u_real*)malloc(nodes * sizeof(u_real));
    u_real* bw_nu = (u_real*)malloc(nodes * sizeof(u_real));
    u_real* bw_q = (u_real*)malloc(nodes * sizeof(u_real));
    u_real tmp_f{0.0};
    for (j=0; j<nodes; j++) {
        bw_x[j] = 0.0;
        bw_f[j] = 1.0;
        bw_nu[j] = 1.0;
        bw_q[j] = 1.0;
    }

    // delay and conn_state history setup
    const bool has_delay = (max_delay > 0);
    // if there is delay use a circular buffer (conn_state_var_hist)
    // and keep track of current buffer index (will be the same
    // in all nodes at each time point). Start from the end and
    // go backwards. 
    // Note that conn_state_var_hist is pseudo-2d
    int buff_idx, k_buff_idx;
    u_real *conn_state_var_1; // immediate history when there is no delay
    if (has_delay) {
        for (j = 0; j < nodes; j++) {
            // initialize conn_state_var_hist in every time point at initial value
            for (buff_idx=0; buff_idx<max_delay; buff_idx++) {
                conn_state_var_hist[sim_idx][buff_idx*nodes+j] = _state_vars[j][Model::conn_state_var_idx];
            }
        }
        buff_idx = max_delay-1;
    } else {
        conn_state_var_1 = conn_state_var_hist[sim_idx];
        for (j = 0; j < nodes; j++) {
            conn_state_var_1[j] = _state_vars[j][Model::conn_state_var_idx];
        }
    }

    // determine the global input function
    GlobalInputKernel global_input_kernel;
    if (Model::is_osc) {
        global_input_kernel = &global_input_osc;
    } else {
        global_input_kernel = &global_input_cond;
    }
    
    // allocate memory for global input
    u_real * tmp_globalinput = (u_real*)malloc(nodes * sizeof(u_real));

    // this will determine if the simulation should be restarted
    // (e.g. if FIC adjustment fails in rWW)
    bool restart{false};

    // integration loop
    int int_i{0}, k{0}, bold_i{0}, ts_bold{0};
    long noise_idx{0};
    // outer loop for 1 msec steps
    // TODO: define number of steps for outer
    // and inner loops based on model dt and BW dt from user input 
    while (ts_bold < time_steps) {
        #ifdef NOISE_SEGMENT
        // get shuffled timepoint corresponding to
        // current noise repeat and the amount of time
        // past in the repeat
        sh_ts_noise = shuffled_ts[(ts_bold % noise_time_steps)+(curr_noise_repeat*noise_time_steps)];
        #endif
        if (sync_msec) {
            // calculate global input every 1 ms
            for (j=0; j<nodes; j++) {
                global_input_kernel(
                    tmp_globalinput[j], k_buff_idx,
                    nodes, sim_idx, SC_idx,
                    j, k, buff_idx, SC, 
                    delay, has_delay, max_delay,
                    conn_state_var_hist, conn_state_var_1
                );
            }
        }
        // inner loop for 0.1 msec steps
        for (int_i = 0; int_i < 10; int_i++) {
            if (!sync_msec) {
                // calculate global input every 0.1 ms
                for (j=0; j<nodes; j++) {
                    global_input_kernel(
                        tmp_globalinput[j], k_buff_idx,
                        nodes, sim_idx, SC_idx,
                        j, k, buff_idx, SC, 
                        delay, has_delay, max_delay,
                        conn_state_var_hist, conn_state_var_1
                    );
                }
            }
            // equations
            for (j=0; j<nodes; j++) {
                #ifdef NOISE_SEGMENT
                sh_j = shuffled_nodes[curr_noise_repeat*nodes+j];
                noise_idx = (((sh_ts_noise * 10 + int_i) * nodes * Model::n_noise) + (sh_j * Model::n_noise));
                #else
                noise_idx = (((ts_bold * 10 + int_i) * nodes * Model::n_noise) + (j * Model::n_noise));
                #endif
                model->step(
                    _state_vars[j], _intermediate_vars[j], 
                    _global_params, _regional_params[j],
                    tmp_globalinput[j],
                    noise, noise_idx
                );
            }
            if (!sync_msec) {
                if (has_delay) {
                    for (j=0; j<nodes; j++) {
                        conn_state_var_hist[sim_idx][buff_idx*nodes+j] = _state_vars[j][Model::conn_state_var_idx];
                    }
                    // go one time step backward in the buffer for the next time point
                    buff_idx = (buff_idx + max_delay - 1) % max_delay;
                    
                } else {
                    for (j=0; j<nodes; j++) {
                        conn_state_var_1[j] = _state_vars[j][Model::conn_state_var_idx];
                    }
                }
            }
        }

        if (sync_msec) {
            if (has_delay) {
                for (j=0; j<nodes; j++) {
                    conn_state_var_hist[sim_idx][buff_idx*nodes+j] = _state_vars[j][Model::conn_state_var_idx];
                }
                // go one time step backward in the buffer for the next time point
                buff_idx = (buff_idx + max_delay - 1) % max_delay;
                
            } else {
                for (j=0; j<nodes; j++) {
                    conn_state_var_1[j] = _state_vars[j][Model::conn_state_var_idx];
                }
            }
        }

        // Balloon-Windkessel model equations here since its
        // dt is 1 msec
        for (j=0; j<nodes; j++) {
            bw_step(bw_x[j], bw_f[j], bw_nu[j], 
                bw_q[j], tmp_f,
                _state_vars[j][Model::bold_state_var_idx]);
        }
        // Save BOLD and extended output to managed memory
        // every TR
        if ((ts_bold+1) % BOLD_TR == 0) {
            for (j=0; j<nodes; j++) {
                // calcualte and save BOLD
                BOLD[sim_idx][bold_i*nodes+j] = d_bwc.V_0_k1 * (1 - bw_q[j]) + d_bwc.V_0_k2 * (1 - bw_q[j]/bw_nu[j]) + d_bwc.V_0_k3 * (1 - bw_nu[j]);
                // save time series of extended output if indicated
                if (ext_out && states_ts) {
                    for (ii=0; ii<Model::n_state_vars; ii++) {
                        states_out[ii][sim_idx][bold_i*nodes+j] = _state_vars[j][ii];
                    }
                }
                // update sum (later mean) of extended
                // output only after n_vols_remove
                if ((bold_i>=model->n_vols_remove) && ext_out && (!states_ts)) {
                    for (ii=0; ii<Model::n_state_vars; ii++) {
                        states_out[ii][sim_idx][j] += _state_vars[j][ii];
                    }
                }
            }
            bold_i++;
            if (model->base_conf.verbose) {
                atomicAdd(progress, 1);
            }
        }

        #ifdef NOISE_SEGMENT
        // reset noise segment time 
        // and shuffle nodes if the segment
        // has reached to the end
        if ((ts_bold+1) % noise_time_steps == 0) {
            // at the last time point don't do this
            // to avoid going over the extent of shuffled_nodes
            if (ts_bold+1 < time_steps) {
                curr_noise_repeat++;
            }
        }
        #endif

        // Note: simulation restart
        // bool j_restart{false};
        // if (Model::has_post_bw_step) {
        //     for (j=0; j<nodes; j++) {
        //         j_restart = false;
        //         model->post_bw_step(
        //             _state_vars[j], _intermediate_vars[j],
        //             _ext_int[j], _ext_bool[j], 
        //             _ext_int_shared, _ext_bool_shared,
        //             j_restart,
        //             _global_params, _regional_params[j],
        //             ts_bold
        //         );
        //         restart = restart || j_restart;
        //     }
        // }

        // move forward outer bw loop
        // this has to be before restart
        // because restart will reset ts_bold to 0
        ts_bold++;

        // // if restart is indicated (e.g. FIC failed in rWW)
        // // reset the simulation and start from the beginning
        // if (restart) {
        //     // model-specific restart
        //     model->restart(_state_vars, _intermediate_vars, _ext_int, _ext_bool, _ext_int_shared, _ext_bool_shared);
        //     // subtract progress of current simulation
        //     if (model->base_conf.verbose && (j==0)) {
        //         atomicAdd(progress, -bold_i);
        //     }
        //     // reset indices
        //     bold_i = 0;
        //     ts_bold = 0;
        //     // reset Balloon-Windkessel model variables
        //     bw_x = 0.0;
        //     bw_f = 1.0;
        //     bw_nu = 1.0;
        //     bw_q = 1.0;
        //     if (has_delay) {
        //         // initialize conn_state_var_hist in every time point at initial value
        //         for (buff_idx=0; buff_idx<max_delay; buff_idx++) {
        //             conn_state_var_hist[sim_idx][buff_idx*nodes+j] = _state_vars[Model::conn_state_var_idx];
        //         }
        //         // reset delay buffer index
        //         buff_idx = max_delay-1;
        //     } else {
        //         // reset conn_state_var_1
        //         conn_state_var_1[j] = _state_vars[Model::conn_state_var_idx];
        //     }
        //     #ifdef NOISE_SEGMENT
        //     // reset the node shuffling
        //     sh_j = shuffled_nodes[j];
        //     curr_noise_repeat = 0;
        //     ts_noise = 0;
        //     sh_ts_noise = shuffled_ts[ts_noise];
        //     #endif
        //     restart = false; // restart is done
        //     __syncthreads(); // make sure all threads are in sync after restart
        // }
    }

    if (Model::has_post_integration) {
        for (j=0; j<nodes; j++) {
            model->post_integration(
                states_out, global_out_int, global_out_bool,
                _state_vars[j], _intermediate_vars[j], 
                _ext_int[j], _ext_bool[j], 
                _ext_int_shared, _ext_bool_shared,
                global_params, regional_params,
                _global_params, _regional_params[j],
                sim_idx, nodes, j
            );
        }
    }
    if (ext_out && (!states_ts)) {
        // take average across time points after n_vols_remove
        int ext_out_time_points = bold_i - model->n_vols_remove;
        for (ii=0; ii<Model::n_state_vars; ii++) {
            states_out[ii][sim_idx][j] /= ext_out_time_points;
        }
    }
    // free heap
    free(tmp_globalinput);
    free(bw_q);
    free(bw_nu);
    free(bw_f);
    free(bw_x);
    for (j=0; j<nodes; j++) {
        if (Model::n_ext_int > 0) {
            free(_ext_int[j]);
        }            
        if (Model::n_ext_bool > 0) {
            free(_ext_bool[j]);
        }
        free(_intermediate_vars[j]);
        free(_state_vars[j]);
        if (Model::n_regional_params > 0) {
            free(_regional_params[j]);
        }
    }
    free(_ext_int);
    free(_ext_bool);
    free(_intermediate_vars);
    free(_state_vars);
    free(_regional_params);
    if (Model::n_global_params > 0) {
        free(_global_params);
    }
}

template <typename Model>
void _run_simulations_gpu(
    double * BOLD_out, double * fc_trils_out, double * fcd_trils_out,
    u_real ** global_params, u_real ** regional_params, u_real * v_list,
    u_real **SC, int *SC_indices, u_real * SC_dist, BaseModel* m
)
{
    if (m->base_conf.verbose) {
        m->print_config();
    }

    // copy model to device 
    Model* h_model = (Model*)m; // cast BaseModel to its specific subclass, TODO: see if this is really needed
    Model* d_model;
    CUDA_CHECK_RETURN(cudaMallocManaged(&d_model, sizeof(Model)));
    CUDA_CHECK_RETURN(cudaMemcpy(d_model, h_model, sizeof(Model), cudaMemcpyHostToDevice));

    // copy SC to managed memory
    for (int SC_idx=0; SC_idx<m->N_SCs; SC_idx++) {
      CUDA_CHECK_RETURN(cudaMemcpy(m->d_SC[SC_idx], SC[SC_idx], m->nodes*m->nodes * sizeof(u_real), cudaMemcpyHostToDevice));
    }
    // copy SC_indices to managed memory
    CUDA_CHECK_RETURN(cudaMemcpy(m->d_SC_indices, SC_indices, m->N_SIMS * sizeof(int), cudaMemcpyHostToDevice));

    // copy parameters to managed memory
    for (int i=0; i<Model::n_global_params; i++) {
        CUDA_CHECK_RETURN(cudaMemcpy(m->d_global_params[i], global_params[i], m->N_SIMS * sizeof(u_real), cudaMemcpyHostToDevice));
    }
    for (int i=0; i<Model::n_regional_params; i++) {
        CUDA_CHECK_RETURN(cudaMemcpy(m->d_regional_params[i], regional_params[i], m->N_SIMS*m->nodes * sizeof(u_real), cudaMemcpyHostToDevice));
    }

    // The following currently only does analytical FIC for rWW
    // but in theory can be used for any model that requires
    // parameter modifications
    // TODO: consider doing this in a separate function
    // called from Python, therefore final params are passed
    // to _run_simulations_gpu (except that they might be
    // modified during the simulation, e.g. in numerical FIC)
    m->prep_params(m->d_global_params, m->d_regional_params, v_list, 
        SC, SC_indices, SC_dist, 
        m->global_out_bool, m->global_out_int);

    // if indicated, calculate delay matrix of each simulation and allocate
    // memory to conn_state_var_hist according to the max_delay among the current simulations
    // Note: unlike many other variables delay and conn_state_var_hist are not global variables
    // and are not initialized in init_gpu, in order to allow variable ranges of velocities
    // in each run_simulations_gpu call within a session
    u_real **conn_state_var_hist; 
    CUDA_CHECK_RETURN(cudaMallocManaged((void**)&conn_state_var_hist, sizeof(u_real*) * m->N_SIMS)); 
    int **delay;
    int max_delay{0}; // msec or 0.1 msec depending on base_conf.sync_msec; this is a global variable that will be used in the kernel
    float min_velocity{1e10}; // only used for printing info
    float max_length{0};
    float curr_length{0.0}, curr_velocity{0.0};
    int curr_delay{0};
    if (m->do_delay) {
    // note that do_delay is user asking for delay to be considered, has_delay indicates
    // if user has asked for delay AND there would be any delay between nodes given
    // velocity and distance matrix
    // TODO: make it less complicated
        CUDA_CHECK_RETURN(cudaMallocManaged((void**)&delay, sizeof(int*) * m->N_SIMS));
        for (int sim_idx=0; sim_idx < m->N_SIMS; sim_idx++) {
            CUDA_CHECK_RETURN(cudaMallocManaged((void**)&delay[sim_idx], sizeof(int) * m->nodes * m->nodes));
            curr_velocity = v_list[sim_idx];
            if (!m->base_conf.sync_msec) {
                curr_velocity /= 10;
            }
            if (curr_velocity < min_velocity) {
                min_velocity = curr_velocity;
            }
            for (int i = 0; i < m->nodes; i++) {
                for (int j = 0; j < m->nodes; j++) {
                    curr_length = SC_dist[i*m->nodes+j];
                    if (i > j) {
                        curr_delay = (int)round(curr_length/curr_velocity);
                        // set minimum delay to 1 because a node
                        // cannot access instantaneous states of 
                        // other nodes, as they might not have been
                        // calculated yet
                        curr_delay = std::max(curr_delay, 1);
                        delay[sim_idx][i*m->nodes + j] = curr_delay;
                        delay[sim_idx][j*m->nodes + i] = curr_delay;
                        if (curr_delay > max_delay) {
                            max_delay = curr_delay;
                            max_length = curr_length;
                        }
                    } else if (i == j) {
                        delay[sim_idx][i*m->nodes + j] = 1;
                    }
                }
            }
        }
    }
    bool has_delay = (max_delay > 0);
    if (has_delay) {
        if (m->base_conf.verbose) {
            std::string velocity_unit = "m/s";
            std::string delay_unit = "msec";
            if (!m->base_conf.sync_msec) {
                velocity_unit = "m/0.1s";
                delay_unit = "0.1msec";
            }
            std::cout << "Max distance " << max_length << " (mm) with a minimum velocity of " 
                << min_velocity << " (" << velocity_unit << ") => Max delay: " 
                << max_delay << " (" << delay_unit << ")" << std::endl;
        }
        // allocate memory to conn_state_var_hist for N_SIMS * (nodes * max_delay)
        // TODO: make it possible to have variable max_delay per each simulation
        for (int sim_idx=0; sim_idx < m->N_SIMS; sim_idx++) {
            CUDA_CHECK_RETURN(cudaMallocManaged((void**)&conn_state_var_hist[sim_idx], sizeof(u_real) * m->nodes * max_delay));
        }
    }
    else if (
        (m->base_conf.serial)
        #ifdef MANY_NODES
        || (m->nodes > MAX_NODES_REG)
        #endif
    )    
     {
        // if there is no delay and the number of nodes is large
        // or nodes are simulated serially
        // store immediate history to conn_state_var_hist
        // on global memory, instead of shared memory
        for (int sim_idx=0; sim_idx < m->N_SIMS; sim_idx++) {
            CUDA_CHECK_RETURN(cudaMallocManaged((void**)&conn_state_var_hist[sim_idx], sizeof(u_real) * m->nodes));
        }
    }

    // increase heap size as needed
    // start with initial heap size
    size_t heapSize = 0;
    CUDA_CHECK_RETURN(cudaDeviceGetLimit(&heapSize, cudaLimitMallocHeapSize));
    // add heap size required
    // in both serial and parallel nodes
    // _ext_int and _ext_bool are stored on heap
    heapSize += m->N_SIMS * (
        (Model::n_ext_int) * m->nodes * sizeof(int) +
        (Model::n_ext_bool) * m->nodes * sizeof(bool)
    );
    // in parallel case, these variables are also
    // on heap
    if (m->base_conf.serial) {
        heapSize += m->N_SIMS * (
            (Model::n_regional_params + Model::n_state_vars + Model::n_intermediate_vars) * m->nodes * sizeof(u_real) +
            (Model::n_global_params) * sizeof(u_real)
        );
    }
    CUDA_CHECK_RETURN(cudaDeviceSetLimit(cudaLimitMallocHeapSize, heapSize));

    // run simulations
    dim3 numBlocks(m->N_SIMS);
    dim3 threadsPerBlock(m->nodes);
    if (m->base_conf.serial) {
        threadsPerBlock.x = 1;
    }
    // calculate amount of required shared memory
    // used to store:
    // 1. _ext_int_shared
    // 2. _ext_bool_shared
    // 3. conn_state_var_1 if there is no delay
    // and the number of nodes is less than MAX_NODES_REG. When there is
    // delay this array is not needed. And with large number of
    // nodes there will be not enough shared memory available.
    size_t shared_mem_extern{0};
    shared_mem_extern += Model::n_ext_int_shared * sizeof(int) 
        + Model::n_ext_bool_shared * sizeof(bool);
    if ((!has_delay) && (m->nodes <= MAX_NODES_REG) && (!m->base_conf.serial)) {
        shared_mem_extern += m->nodes*sizeof(u_real);
    }
    // keep track of progress
    // Note: based on BOLD TRs reached in the first node
    // of each simulation (therefore the progress will be
    // an approximation of the real progress)
    uint* progress;
    CUDA_CHECK_RETURN(cudaMallocManaged(&progress, sizeof(uint)));
    uint progress_final = m->bold_len * m->N_SIMS;
    *progress = 0;
    if (m->base_conf.serial) {
        bnm_serial<Model><<<numBlocks,threadsPerBlock,shared_mem_extern>>>(
            d_model,
            m->BOLD, m->states_out, 
            m->global_out_int,
            m->global_out_bool,
            m->d_SC, m->d_SC_indices,
            m->d_global_params, m->d_regional_params,
            conn_state_var_hist, delay, max_delay,
        #ifdef NOISE_SEGMENT
            m->shuffled_nodes, m->shuffled_ts,
        #endif
            m->noise, progress);
    } else {
        bnm<Model><<<numBlocks,threadsPerBlock,shared_mem_extern>>>(
            d_model,
            m->BOLD, m->states_out, 
            m->global_out_int,
            m->global_out_bool,
            m->d_SC, m->d_SC_indices,
            m->d_global_params, m->d_regional_params,
            conn_state_var_hist, delay, max_delay,
        #ifdef NOISE_SEGMENT
            m->shuffled_nodes, m->shuffled_ts,
        #endif
            m->noise, progress);
    }
    // asynchroneously print out the progress
    // if verbose
    if (m->base_conf.verbose) {
        uint last_progress = 0;
        uint no_progress_count = 0;
        while (*progress < progress_final) {
            // Print progress as percentage
            std::cout << std::fixed << std::setprecision(2) 
                << ((double)*progress / progress_final) * 100 << "%\r";
            std::cout.flush();
            // Sleep for interval ms
            std::this_thread::sleep_for(std::chrono::milliseconds(m->base_conf.progress_interval));
            // make sure it doesn't get stuck
            // by checking if there has been any progress
            if (*progress == last_progress) {
                no_progress_count++;
            } else {
                no_progress_count = 0;
            }
            if (no_progress_count > 50) {
                std::cout << "No progress detected in the last " << 50 * m->base_conf.progress_interval << " ms." << std::endl;
                break;
            }
            last_progress = *progress;
        }
        if (*progress == progress_final) {
            std::cout << "100.00%" << std::endl;
        } else {
            std::cout << "If no errors are shown, the simulation is still running "
                "but the progress is not being updated as there was no progress in the "
                "last " << m->base_conf.progress_interval <<  " ms, which may be too "
                "fast for current GPU and simulations" << std::endl;
        }
    }
    CUDA_CHECK_LAST_ERROR();
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());
    if (m->base_conf.verbose) {
        std::cout << "Simulation completed" << std::endl;
    }
    // calculate mean and sd bold for FC calculation
    threadsPerBlock.x = m->nodes;
    bold_stats<<<numBlocks, threadsPerBlock>>>(
        m->mean_bold, m->ssd_bold,
        m->BOLD, m->N_SIMS, m->nodes,
        m->bold_len, m->corr_len, m->n_vols_remove);
    CUDA_CHECK_LAST_ERROR();
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());
    // calculate window mean and sd bold for FCD calculations
    numBlocks.x = m->N_SIMS;
    numBlocks.y = m->n_windows;
    window_bold_stats<<<numBlocks,threadsPerBlock>>>(
        m->BOLD, m->N_SIMS, m->nodes, 
        m->n_windows, m->window_size+1, m->window_starts, m->window_ends,
        m->windows_mean_bold, m->windows_ssd_bold);
    CUDA_CHECK_LAST_ERROR();
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());
    // calculate FC and window FCs
    int maxThreadsPerBlock = prop.maxThreadsPerBlock;
    numBlocks.x = m->N_SIMS;
    numBlocks.y = ceil((float)m->n_pairs / (float)maxThreadsPerBlock);
    numBlocks.z = m->n_windows + 1; // +1 for total FC
    if (prop.maxThreadsPerBlock!=prop.maxThreadsDim[0]) {
        std::cerr << "Error: Code not implemented for GPUs in which maxThreadsPerBlock!=maxThreadsDim[0]" << std::endl;
        exit(1);
    }
    threadsPerBlock.x = maxThreadsPerBlock;
    threadsPerBlock.y = 1;
    threadsPerBlock.z = 1;
    fc<<<numBlocks, threadsPerBlock>>>(
        m->fc_trils, m->windows_fc_trils, m->BOLD, m->N_SIMS, m->nodes, m->n_pairs, 
        m->pairs_i, m->pairs_j,
        m->bold_len, m->n_vols_remove, m->corr_len, m->mean_bold, m->ssd_bold,
        m->n_windows, m->window_size+1, m->windows_mean_bold, m->windows_ssd_bold,
        m->window_starts, m->window_ends,
        maxThreadsPerBlock
    );
    CUDA_CHECK_LAST_ERROR();
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());
    // calculate window mean and sd fc_tril for FCD calculations
    numBlocks.x = m->N_SIMS;
    numBlocks.y = 1;
    numBlocks.z = 1;
    threadsPerBlock.x = m->n_windows;
    if (m->n_windows >= prop.maxThreadsPerBlock) {
        std::cerr << "Error: Mean/ssd FC tril of " << m->n_windows 
            << " windows cannot be calculated on this device" << std::endl;
        exit(1);
    }
    window_fc_stats<<<numBlocks,threadsPerBlock>>>(
        m->windows_mean_fc, m->windows_ssd_fc,
        NULL, NULL, NULL, NULL, // skipping L and R stats
        m->windows_fc_trils, m->N_SIMS, m->n_windows, m->n_pairs,
        false, 0);
    CUDA_CHECK_LAST_ERROR();
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());
    // calculate FCD
    numBlocks.x = m->N_SIMS;
    numBlocks.y = ceil((float)m->n_window_pairs / (float)maxThreadsPerBlock);
    numBlocks.z = 1;
    if (prop.maxThreadsPerBlock!=prop.maxThreadsDim[0]) {
        std::cerr << "Code not implemented for GPUs in which maxThreadsPerBlock!=maxThreadsDim[0]" << std::endl;
        exit(1);
    }
    threadsPerBlock.x = maxThreadsPerBlock;
    threadsPerBlock.y = 1;
    threadsPerBlock.z = 1;
    fcd<<<numBlocks, threadsPerBlock>>>(
        m->fcd_trils, NULL, NULL, // skipping separate L and R fcd
        m->windows_fc_trils, 
        m->windows_mean_fc, m->windows_ssd_fc,
        NULL, NULL, NULL, NULL,
        m->N_SIMS, m->n_pairs, m->n_windows, m->n_window_pairs, 
        m->window_pairs_i, m->window_pairs_j, maxThreadsPerBlock,
        false, 0);
    CUDA_CHECK_LAST_ERROR();
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());

    #ifdef USE_FLOATS
    // Convert FC and FCD to doubles for GOF calculation
    numBlocks.x = m->N_SIMS;
    numBlocks.y = m->n_pairs;
    numBlocks.z = 1;
    threadsPerBlock.x = 1;
    float2double<<<numBlocks, threadsPerBlock>>>(m->d_fc_trils, m->fc_trils, m->N_SIMS, m->n_pairs);
    CUDA_CHECK_LAST_ERROR();
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());
    numBlocks.x = m->N_SIMS;
    numBlocks.y = m->n_window_pairs;
    float2double<<<numBlocks, threadsPerBlock>>>(m->d_fcd_trils, m->fcd_trils, m->N_SIMS, m->n_window_pairs);
    CUDA_CHECK_LAST_ERROR();
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());
    #endif

    // copy the output from managed memory to _out arrays (which can be numpy arrays)
    size_t ext_out_size = m->nodes;
    if (m->base_conf.states_ts) {
        ext_out_size *= m->bold_len;
    }
    // TODO: pass the managed arrays data directly
    // to the python arrays without copying
    for (int sim_idx=0; sim_idx<m->N_SIMS; sim_idx++) {
        memcpy(BOLD_out, m->BOLD[sim_idx], sizeof(u_real) * m->bold_size);
        BOLD_out+=m->bold_size;
        memcpy(fc_trils_out, m->fc_trils[sim_idx], sizeof(u_real) * m->n_pairs);
        fc_trils_out+=m->n_pairs;
        memcpy(fcd_trils_out, m->fcd_trils[sim_idx], sizeof(u_real) * m->n_window_pairs);
        fcd_trils_out+=m->n_window_pairs;
    }
    if (m->modifies_params) { // e.g. rWW with FIC
        // copy (potentially) modified parameters back to the original array
        for (int i=0; i<Model::n_global_params; i++) {
            memcpy(global_params[i], m->d_global_params[i], m->N_SIMS * sizeof(u_real));
        }
        for (int i=0; i<Model::n_regional_params; i++) {
            memcpy(regional_params[i], m->d_regional_params[i], m->N_SIMS*m->nodes * sizeof(u_real));
        }
    }

    // free delay and conn_state_var_hist memories if allocated
    // Note: no need to clear memory of the other variables
    // as we'll want to reuse them in the next calls to run_simulations_gpu
    // within current session
    if (m->do_delay) {
        for (int sim_idx=0; sim_idx < m->N_SIMS; sim_idx++) {
            CUDA_CHECK_RETURN(cudaFree(delay[sim_idx]));
        }
        CUDA_CHECK_RETURN(cudaFree(delay));
    }
    if (has_delay) {
        for (int sim_idx=0; sim_idx < m->N_SIMS; sim_idx++) {
            CUDA_CHECK_RETURN(cudaFree(conn_state_var_hist[sim_idx]));
        }
    }
    else if (
        (m->base_conf.serial)
        #ifdef MANY_NODES
        || (m->nodes > MAX_NODES_REG)
        #endif
    ) {
        for (int sim_idx=0; sim_idx < m->N_SIMS; sim_idx++) {
            CUDA_CHECK_RETURN(cudaFree(conn_state_var_hist[sim_idx]));
        }
    } 
    CUDA_CHECK_RETURN(cudaFree(conn_state_var_hist));
}

template <typename Model>
void _init_gpu(BaseModel *m, BWConstants bwc, bool force_reinit) {
    // check CUDA device avaliability and properties
    prop = get_device_prop(m->base_conf.verbose);

    // copy constants and configs from CPU
    // TODO: make these members of the model class
    CUDA_CHECK_RETURN(cudaMemcpyToSymbol(d_bwc, &bwc, sizeof(BWConstants)));
    if (strcmp(Model::name, "rWW")==0) {
        CUDA_CHECK_RETURN(cudaMemcpyToSymbol(d_rWWc, &Model::mc, sizeof(typename Model::Constants)));
    } 
    else if (strcmp(Model::name, "rWWEx")==0) {
        CUDA_CHECK_RETURN(cudaMemcpyToSymbol(d_rWWExc, &Model::mc, sizeof(typename Model::Constants)));
    }
    else if (strcmp(Model::name, "Kuramoto")==0) {
        CUDA_CHECK_RETURN(cudaMemcpyToSymbol(d_Kuramotoc, &Model::mc, sizeof(typename Model::Constants)));
    }

    // allocate device memory for SC
    CUDA_CHECK_RETURN(cudaMallocManaged((void**)&(m->d_SC), sizeof(u_real*) * m->N_SCs));
    for (int SC_idx=0; SC_idx<m->N_SCs; SC_idx++) {
        CUDA_CHECK_RETURN(cudaMallocManaged((void**)&(m->d_SC[SC_idx]), sizeof(u_real) * m->nodes*m->nodes));
    }
    CUDA_CHECK_RETURN(cudaMallocManaged(&(m->d_SC_indices), sizeof(int) * m->N_SIMS));
 
    // allocate device memory for simulation parameters
    // size of global_params is (n_global_params, N_SIMS)
    // size of regional_params is (n_regional_params, N_SIMS * nodes)
    if (Model::n_global_params > 0) {
        CUDA_CHECK_RETURN(cudaMallocManaged((void**)&(m->d_global_params), sizeof(u_real*) * Model::n_global_params));
        for (int param_idx=0; param_idx<Model::n_global_params; param_idx++) {
            CUDA_CHECK_RETURN(cudaMallocManaged((void**)&(m->d_global_params[param_idx]), sizeof(u_real) * m->N_SIMS));
        }
    }
    if (Model::n_regional_params > 0) {
        CUDA_CHECK_RETURN(cudaMallocManaged((void**)&(m->d_regional_params), sizeof(u_real*) * Model::n_regional_params));
        for (int param_idx=0; param_idx<Model::n_regional_params; param_idx++) {
            CUDA_CHECK_RETURN(cudaMallocManaged((void**)&(m->d_regional_params[param_idx]), sizeof(u_real) * m->N_SIMS * m->nodes));
        }
    }

    // set up global int and bool outputs
    if (Model::n_global_out_int > 0) {
        CUDA_CHECK_RETURN(cudaMallocManaged((void**)&(m->global_out_int), sizeof(int*) * Model::n_global_out_int));
        for (int i=0; i<Model::n_global_out_int; i++) {
            CUDA_CHECK_RETURN(cudaMallocManaged((void**)&(m->global_out_int[i]), sizeof(int) * m->N_SIMS));
        }
    }
    if (Model::n_global_out_bool > 0) {
        CUDA_CHECK_RETURN(cudaMallocManaged((void**)&(m->global_out_bool), sizeof(bool*) * Model::n_global_out_bool));
        for (int i=0; i<Model::n_global_out_bool; i++) {
            CUDA_CHECK_RETURN(cudaMallocManaged((void**)&(m->global_out_bool[i]), sizeof(bool) * m->N_SIMS));
        }
    }

    // allocate memory for extended output
    size_t ext_out_size = m->nodes;
    if (m->base_conf.states_ts) {
        ext_out_size *= m->states_len;
    }
    if (m->base_conf.ext_out) {
        CUDA_CHECK_RETURN(cudaMallocManaged((void**)&(m->states_out), sizeof(u_real**) * Model::n_state_vars));
        for (int var_idx=0; var_idx<Model::n_state_vars; var_idx++) {
            CUDA_CHECK_RETURN(cudaMallocManaged((void**)&(m->states_out[var_idx]), sizeof(u_real*) * m->N_SIMS));
            for (int sim_idx=0; sim_idx<m->N_SIMS; sim_idx++) {
                CUDA_CHECK_RETURN(cudaMallocManaged((void**)&(m->states_out[var_idx][sim_idx]), sizeof(u_real) * ext_out_size));
            }
        }
    }

    // specify n_vols_remove (for FC(D) calculations)
    m->n_vols_remove = m->base_conf.bold_remove_s * 1000 / m->BOLD_TR;

    // specifiy n_states_samples_remove (for states mean calculations)
    m->n_states_samples_remove = m->base_conf.bold_remove_s * 1000 / m->states_sampling;

    // preparing FC calculations
    m->corr_len = m->bold_len - m->n_vols_remove;
    if (m->corr_len < 2) {
        std::cerr << "Number of BOLD volumes (after removing initial volumes) is too low for FC calculations" << std::endl;
        exit(1);
    }
    CUDA_CHECK_RETURN(cudaMallocManaged((void**)&(m->BOLD), sizeof(u_real*) * m->N_SIMS));
    CUDA_CHECK_RETURN(cudaMallocManaged((void**)&(m->mean_bold), sizeof(u_real*) * m->N_SIMS));
    CUDA_CHECK_RETURN(cudaMallocManaged((void**)&(m->ssd_bold), sizeof(u_real*) * m->N_SIMS));
    m->n_pairs = ((m->nodes) * (m->nodes - 1)) / 2;
    int rh_idx;
    if (m->base_conf.exc_interhemispheric) {
        if ((m->nodes % 2) != 0) {
            std::cerr << "Error: exc_interhemispheric is set but number of nodes is not even" << std::endl;
            exit(1);
        }
        rh_idx = m->nodes / 2; // assumes symmetric number of parcels and L->R order
        m->n_pairs -= pow(rh_idx, 2); // exclude the middle square
    }
    // create a mapping between pair_idx and i and j
    int curr_idx = 0;
    CUDA_CHECK_RETURN(cudaMallocManaged(&(m->pairs_i), sizeof(int) * m->n_pairs));
    CUDA_CHECK_RETURN(cudaMallocManaged(&(m->pairs_j), sizeof(int) * m->n_pairs));
    for (int i=0; i < m->nodes; i++) {
        for (int j=0; j < m->nodes; j++) {
            if (i > j) {
                if (m->base_conf.exc_interhemispheric) {
                    // skip if each node belongs to a different hemisphere
                    if ((i < rh_idx) ^ (j < rh_idx)) {
                        continue;
                    }
                }
                m->pairs_i[curr_idx] = i;
                m->pairs_j[curr_idx] = j;
                curr_idx++;
            }
        }
    }
    // allocate memory for fc trils
    CUDA_CHECK_RETURN(cudaMallocManaged((void**)&(m->fc_trils), sizeof(u_real*) * m->N_SIMS));

    // FCD preparation
    // calculate number of windows and window start/end indices
    int *_window_starts, *_window_ends; // are cpu integer arrays
    m->n_windows = get_dfc_windows(
        &_window_starts, &_window_ends, 
        m->corr_len, m->bold_len, m->n_vols_remove,
        m->window_step, m->window_size, m->base_conf.drop_edges);
    if (m->n_windows == 0) {
        std::cerr << "Error: Number of windows is 0" << std::endl;
        exit(1);
    }
    CUDA_CHECK_RETURN(cudaMallocManaged(&(m->window_starts), sizeof(int) * m->n_windows));
    CUDA_CHECK_RETURN(cudaMallocManaged(&(m->window_ends), sizeof(int) * m->n_windows));
    for (int i=0; i<m->n_windows; i++) {
        m->window_starts[i] = _window_starts[i];
        m->window_ends[i] = _window_ends[i];
    }
    // allocate memory for mean and ssd BOLD of each window
    // (n_sims x n_windows x nodes)
    CUDA_CHECK_RETURN(cudaMallocManaged((void**)&(m->windows_mean_bold), sizeof(u_real*) * m->N_SIMS));
    CUDA_CHECK_RETURN(cudaMallocManaged((void**)&(m->windows_ssd_bold), sizeof(u_real*) * m->N_SIMS));
    CUDA_CHECK_RETURN(cudaMallocManaged((void**)&(m->windows_fc_trils), sizeof(u_real*) * m->N_SIMS));
    // allocate memory for mean and ssd fc_tril of each window
    // (n_sims x n_windows)
    CUDA_CHECK_RETURN(cudaMallocManaged((void**)&(m->windows_mean_fc), sizeof(u_real*) * m->N_SIMS));
    CUDA_CHECK_RETURN(cudaMallocManaged((void**)&(m->windows_ssd_fc), sizeof(u_real*) * m->N_SIMS));
    // create a mapping between window_pair_idx and i and j
    m->n_window_pairs = (m->n_windows * (m->n_windows-1)) / 2;
    curr_idx = 0;
    CUDA_CHECK_RETURN(cudaMallocManaged(&(m->window_pairs_i), sizeof(int) * m->n_window_pairs));
    CUDA_CHECK_RETURN(cudaMallocManaged(&(m->window_pairs_j), sizeof(int) * m->n_window_pairs));
    for (int i=0; i < m->n_windows; i++) {
        for (int j=0; j < m->n_windows; j++) {
            if (i > j) {
                m->window_pairs_i[curr_idx] = i;
                m->window_pairs_j[curr_idx] = j;
                curr_idx++;
            }
        }
    }
    // allocate memory for fcd trils
    CUDA_CHECK_RETURN(cudaMallocManaged((void**)&(m->fcd_trils), sizeof(u_real*) * m->N_SIMS));

    #ifdef USE_FLOATS
    // allocate memory for double versions of fc and fcd trils on CPU
    CUDA_CHECK_RETURN(cudaMallocManaged((void**)&(m->d_fc_trils), sizeof(double*) * m->N_SIMS));
    CUDA_CHECK_RETURN(cudaMallocManaged((void**)&(m->d_fcd_trils), sizeof(double*) * m->N_SIMS));
    #else
    // use d_fc_trils and d_fcd_trils as aliases for fc_trils and fcd_trils
    m->d_fc_trils = m->fc_trils;
    m->d_fcd_trils = m->fcd_trils;
    #endif



    // allocate memory per each simulation
    for (int sim_idx=0; sim_idx<m->N_SIMS; sim_idx++) {
        // allocate a chunk of BOLD to this simulation (not sure entirely if this is the best way to do it)
        CUDA_CHECK_RETURN(cudaMallocManaged((void**)&(m->BOLD[sim_idx]), sizeof(u_real) * m->bold_size));
        // allocate memory for fc calculations
        CUDA_CHECK_RETURN(cudaMallocManaged((void**)&(m->mean_bold[sim_idx]), sizeof(u_real) * m->nodes));
        CUDA_CHECK_RETURN(cudaMallocManaged((void**)&(m->ssd_bold[sim_idx]), sizeof(u_real) * m->nodes));
        CUDA_CHECK_RETURN(cudaMallocManaged((void**)&(m->fc_trils[sim_idx]), sizeof(u_real) * m->n_pairs));
        // allocate memory for window fc and fcd calculations
        CUDA_CHECK_RETURN(cudaMallocManaged((void**)&(m->windows_mean_bold[sim_idx]), sizeof(u_real) * m->n_windows * m->nodes));
        CUDA_CHECK_RETURN(cudaMallocManaged((void**)&(m->windows_ssd_bold[sim_idx]), sizeof(u_real) * m->n_windows * m->nodes));
        CUDA_CHECK_RETURN(cudaMallocManaged((void**)&(m->windows_fc_trils[sim_idx]), sizeof(u_real) * m->n_windows * m->n_pairs));
        CUDA_CHECK_RETURN(cudaMallocManaged((void**)&(m->windows_mean_fc[sim_idx]), sizeof(u_real) * m->n_windows));
        CUDA_CHECK_RETURN(cudaMallocManaged((void**)&(m->windows_ssd_fc[sim_idx]), sizeof(u_real) * m->n_windows));
        CUDA_CHECK_RETURN(cudaMallocManaged((void**)&(m->fcd_trils[sim_idx]), sizeof(u_real) * m->n_window_pairs));
        #ifdef USE_FLOATS
        // allocate memory for double copies of fc and fcd
        CUDA_CHECK_RETURN(cudaMallocManaged((void**)&(m->d_fc_trils[sim_idx]), sizeof(double) * m->n_pairs));
        CUDA_CHECK_RETURN(cudaMallocManaged((void**)&(m->d_fcd_trils[sim_idx]), sizeof(double) * m->n_window_pairs));
        #endif
    }

    // check if noise needs to be calculated
    if (
        (m->rand_seed != m->last_rand_seed) ||
        (m->time_steps != m->last_time_steps) ||
        (m->nodes != m->last_nodes) ||
        (m->base_conf.noise_time_steps != m->last_noise_time_steps) ||
        (!m->gpu_initialized) ||
        force_reinit
        ) {
        // pre-calculate normally-distributed noise on CPU
        // this is necessary to ensure consistency of noise given the same seed
        // doing the same thing directly on the device is more challenging
        #ifndef NOISE_SEGMENT
        // precalculate the entire noise needed; can use up a lot of memory
        // with high N of nodes and longer durations leads maxes out the memory
        m->noise_size = m->nodes * m->time_steps * 10 * Model::n_noise; // *10 for 0.1msec
        #else
        // otherwise precalculate a noise segment and arrays of shuffled
        // nodes and time points and reuse-shuffle the noise segment
        // throughout the simulation for `noise_repeats`
        m->noise_size = m->nodes * (m->base_conf.noise_time_steps) * 10 * Model::n_noise;
        m->noise_repeats = ceil((float)(m->time_steps) / (float)(m->base_conf.noise_time_steps));
        #endif
        if (m->base_conf.verbose) {
            std::cout << "Precalculating " << m->noise_size << " noise elements..." << std::endl;
        }
        if (m->last_nodes != 0) {
            // noise is being recalculated, free the previous one
            CUDA_CHECK_RETURN(cudaFree(m->noise));
            #ifdef NOISE_SEGMENT
            CUDA_CHECK_RETURN(cudaFree(m->shuffled_nodes));
            CUDA_CHECK_RETURN(cudaFree(m->shuffled_ts));
            #endif
        }
        m->last_time_steps = m->time_steps;
        m->last_nodes = m->nodes;
        m->last_rand_seed = m->rand_seed;
        m->last_noise_time_steps = m->base_conf.noise_time_steps;
        std::mt19937 rand_gen(m->rand_seed);
        std::normal_distribution<float> normal_dist(0, 1);
        CUDA_CHECK_RETURN(cudaMallocManaged(&(m->noise), sizeof(u_real) * m->noise_size));
        for (int i = 0; i < m->noise_size; i++) {
            #ifdef USE_FLOATS
            m->noise[i] = normal_dist(rand_gen);
            #else
            m->noise[i] = (double)normal_dist(rand_gen);
            #endif
        }
        #ifdef NOISE_SEGMENT
        // create shuffled nodes and ts indices for each repeat of the 
        // precalculaed noise 
        if (m->base_conf.verbose) {
            std::cout << "noise will be repeated " << m->noise_repeats << 
                " times (nodes [rows] and timepoints [columns] will be shuffled in each repeat)" << std::endl;
        }
        CUDA_CHECK_RETURN(cudaMallocManaged(&(m->shuffled_nodes), sizeof(int) * m->noise_repeats * m->nodes));
        CUDA_CHECK_RETURN(cudaMallocManaged(&(m->shuffled_ts), sizeof(int) * m->noise_repeats * m->base_conf.noise_time_steps));
        get_shuffled_nodes_ts(&(m->shuffled_nodes), &(m->shuffled_ts),
            m->nodes, m->base_conf.noise_time_steps, m->noise_repeats, &rand_gen);
        #endif
    } else {
        if (m->base_conf.verbose) {
            std::cout << "Noise already precalculated" << std::endl;
        }
    }

    m->gpu_initialized = true;
}

void BaseModel::free_gpu() {
    if (strcmp(this->get_name(), "Base")==0) {
        // skip freeing memory for BaseModel
        // though free_gpu normally is not called for BaseModel
        // but keeping it here for safety
        return;
    }
    if (!this->gpu_initialized) {
        // if gpu not initialized, skip freeing memory
        return;
    }
    if (this->base_conf.verbose) {
        std::cout << "Freeing GPU memory (" << this->get_name() << ")" << std::endl;
    }
    #ifdef NOISE_SEGMENT
    CUDA_CHECK_RETURN(cudaFree(this->shuffled_nodes));
    CUDA_CHECK_RETURN(cudaFree(this->shuffled_ts));
    #endif
    CUDA_CHECK_RETURN(cudaFree(this->noise));
    for (int sim_idx=0; sim_idx<this->N_SIMS; sim_idx++) {
        #ifdef USE_FLOATS
        CUDA_CHECK_RETURN(cudaFree(this->d_fcd_trils[sim_idx]));
        CUDA_CHECK_RETURN(cudaFree(this->d_fc_trils[sim_idx]));
        #endif
        CUDA_CHECK_RETURN(cudaFree(this->fcd_trils[sim_idx]));
        CUDA_CHECK_RETURN(cudaFree(this->windows_ssd_fc[sim_idx]));
        CUDA_CHECK_RETURN(cudaFree(this->windows_mean_fc[sim_idx]));
        CUDA_CHECK_RETURN(cudaFree(this->windows_fc_trils[sim_idx]));
        CUDA_CHECK_RETURN(cudaFree(this->windows_ssd_bold[sim_idx]));
        CUDA_CHECK_RETURN(cudaFree(this->windows_mean_bold[sim_idx]));
        CUDA_CHECK_RETURN(cudaFree(this->fc_trils[sim_idx]));
        CUDA_CHECK_RETURN(cudaFree(this->ssd_bold[sim_idx]));
        CUDA_CHECK_RETURN(cudaFree(this->mean_bold[sim_idx]));
        CUDA_CHECK_RETURN(cudaFree(this->BOLD[sim_idx]));
    }
    #ifdef USE_FLOATS
    CUDA_CHECK_RETURN(cudaFree(this->d_fcd_trils));
    CUDA_CHECK_RETURN(cudaFree(this->d_fc_trils));
    #endif
    CUDA_CHECK_RETURN(cudaFree(this->fcd_trils));
    CUDA_CHECK_RETURN(cudaFree(this->window_pairs_j));
    CUDA_CHECK_RETURN(cudaFree(this->window_pairs_i));
    CUDA_CHECK_RETURN(cudaFree(this->windows_ssd_fc));
    CUDA_CHECK_RETURN(cudaFree(this->windows_mean_fc));
    CUDA_CHECK_RETURN(cudaFree(this->windows_fc_trils));
    CUDA_CHECK_RETURN(cudaFree(this->windows_ssd_bold));
    CUDA_CHECK_RETURN(cudaFree(this->windows_mean_bold));
    CUDA_CHECK_RETURN(cudaFree(this->window_ends));
    CUDA_CHECK_RETURN(cudaFree(this->window_starts));
    CUDA_CHECK_RETURN(cudaFree(this->pairs_j));
    CUDA_CHECK_RETURN(cudaFree(this->pairs_i));
    CUDA_CHECK_RETURN(cudaFree(this->fc_trils));
    CUDA_CHECK_RETURN(cudaFree(this->ssd_bold));
    CUDA_CHECK_RETURN(cudaFree(this->mean_bold));
    CUDA_CHECK_RETURN(cudaFree(this->BOLD));
    if (this->base_conf.ext_out) {
        for (int var_idx=0; var_idx<this->get_n_state_vars(); var_idx++) {
            for (int sim_idx=0; sim_idx<this->N_SIMS; sim_idx++) {
                CUDA_CHECK_RETURN(cudaFree(this->states_out[var_idx][sim_idx]));
            }
            CUDA_CHECK_RETURN(cudaFree(this->states_out[var_idx]));
        }
        CUDA_CHECK_RETURN(cudaFree(this->states_out));
    }
    if (this->get_n_global_out_bool() > 0) {
        for (int i=0; i<this->get_n_global_out_bool(); i++) {
            CUDA_CHECK_RETURN(cudaFree(this->global_out_bool[i]));
        }
        CUDA_CHECK_RETURN(cudaFree(this->global_out_bool));
    }
    if (this->get_n_global_out_int() > 0) {
        for (int i=0; i<this->get_n_global_out_int(); i++) {
            CUDA_CHECK_RETURN(cudaFree(this->global_out_int[i]));
        }
        CUDA_CHECK_RETURN(cudaFree(this->global_out_int));
    }
    if (this->get_n_regional_params() > 0) {
        for (int i=0; i<this->get_n_regional_params(); i++) {
            CUDA_CHECK_RETURN(cudaFree(this->d_regional_params[i]));
        }
        CUDA_CHECK_RETURN(cudaFree(this->d_regional_params));
    }
    if (this->get_n_global_params() > 0) {
        for (int i=0; i<this->get_n_global_params(); i++) {
            CUDA_CHECK_RETURN(cudaFree(this->d_global_params[i]));
        }
        CUDA_CHECK_RETURN(cudaFree(this->d_global_params));
    }
    CUDA_CHECK_RETURN(cudaFree(this->d_SC_indices));
    for (int SC_idx=0; SC_idx<this->N_SCs; SC_idx++) {
        CUDA_CHECK_RETURN(cudaFree(this->d_SC[SC_idx]));
    }
    CUDA_CHECK_RETURN(cudaFree(this->d_SC));
}
