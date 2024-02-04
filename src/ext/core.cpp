#define PY_SSIZE_T_CLEAN
#define UMALLOC(var, type, size) var = (type *)malloc(sizeof(type) * size)
#ifdef OMP_ENABLED
    #include <omp.h>
#endif
#include <stdio.h>
#include <Python.h>
#include <numpy/arrayobject.h>
#include <vector>
#include <map>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_statistics.h>
#include <gsl/gsl_matrix_double.h>
#include <gsl/gsl_vector_double.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_roots.h>
#include <gsl/gsl_eigen.h>
#include <chrono>
#include <ctime>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <math.h>
#include <sys/stat.h>
#include <iostream>
#include <cmath>
#include <string>
#include <cstdlib>
#include <random>
#include <ios>
#include <fstream>
#include <iomanip>
#include <complex>
#include <algorithm>
#include <memory>
#include "cubnm/defines.h"
#include "./models/base.cpp"
#include "./models/bw.cpp"
#include "./models/rww.cpp"
#include "./models/rww_fic.cpp"
#include "./utils.cpp"
// #include "bnm.cpp"

#ifdef GPU_ENABLED
// declare gpu functions which will be provided by bnm.cu compiled library
template<typename Model>
extern void init_gpu(
        int *output_ts_p, int *n_pairs_p, int *n_window_pairs_p,
        int N_SIMS, int nodes, bool extended_output, int rand_seed,
        int BOLD_TR, int time_steps, int window_size, int window_step,
        Model *model, BWConstants bwc, ModelConfigs conf, bool verbose
        );
template<typename Model>
extern void run_simulations_gpu(
    double * BOLD_ex_out, double * fc_trils_out, double * fcd_trils_out,
    u_real ** global_params, u_real ** regional_params, u_real * v_list,
    u_real * SC, gsl_matrix * SC_gsl, u_real * SC_dist, bool do_delay, int nodes,
    int time_steps, int BOLD_TR, int window_size,
    int N_SIMS, bool extended_output, Model *model,
    ModelConfigs conf
);
namespace bnm_gpu {
    extern bool is_initialized;
    extern u_real *** states_out, **BOLD, **fc_trils, **fcd_trils;
    extern int **global_out_int;
    extern bool **global_out_bool;
}
#endif
// the following variables are calcualted during init functions and will
// be needed to determine the size of arrays
// they are global because init functions may be called only once
int _output_ts, _n_pairs, _n_window_pairs;


u_real ** np_to_array_2d(PyArrayObject * np_arr) {
    // converts a 2d numpy array to a 2d array of type u_real
    int rows = PyArray_DIM(np_arr, 0);
    int cols = PyArray_DIM(np_arr, 1);
    double* data = (double*)PyArray_DATA(np_arr);

    u_real** arr = (u_real**)malloc(rows * sizeof(u_real*));
    for (int i = 0; i < rows; i++) {
        arr[i] = (u_real*)malloc(cols * sizeof(u_real));
        for (int j = 0; j < cols; j++) {
            arr[i][j] = data[i*cols + j];
        }
    }
    return arr;
}

std::map<std::string, std::string> dict_to_map(PyObject *config_dict) {
    // Create a map to hold the config values
    std::map<std::string, std::string> config_map;
    if (!PyDict_Check(config_dict)) {
        PyErr_SetString(PyExc_TypeError, "Parameter must be a dictionary.");
        return config_map;
    }
    PyObject *key, *value;
    Py_ssize_t pos = 0;

    while (PyDict_Next(config_dict, &pos, &key, &value)) {
        // Ensure key and value are strings
        if (!PyUnicode_Check(key) || !PyUnicode_Check(value)) {
            PyErr_SetString(PyExc_TypeError, "Dictionary keys and values must be strings.");
            return config_map;
        }
        // Convert key and value to C++ types
        std::string key_str = PyUnicode_AsUTF8(key);
        std::string value_str = PyUnicode_AsUTF8(value);

        // Add the key-value pair to the map
        config_map[key_str] = value_str;
    }
    return config_map;
}

template<typename T>
void array_to_np_3d(T *** arr, PyObject * np_arr) {
    // converts a 3d array to a 3d numpy array
    int dim1 = PyArray_DIM(np_arr, 0);
    int dim2 = PyArray_DIM(np_arr, 1);
    int dim3 = PyArray_DIM(np_arr, 2);
    T* data = (T*)PyArray_DATA(np_arr);

    for (int i = 0; i < dim1; i++) {
        for (int j = 0; j < dim2; j++) {
            for (int k = 0; k < dim3; k++) {
                data[i*dim2*dim3 + j*dim3 + k] = arr[i][j][k];
            }
        }
    }
}

template<typename T>
void array_to_np_2d(T ** arr, PyObject * np_arr) {
    // converts a 2d array to a 2d numpy array
    int dim1 = PyArray_DIM(np_arr, 0);
    int dim2 = PyArray_DIM(np_arr, 1);
    T* data = (T*)PyArray_DATA(np_arr);

    for (int i = 0; i < dim1; i++) {
        for (int j = 0; j < dim2; j++) {
            data[i*dim2 + j] = arr[i][j];
        }
    }
}

// write a python extension function with no arguments named init which
// returns nothing
static PyObject* init(PyObject* self, PyObject* args) {
    // this function is called only once at the beginning of the session
    // (i.e. when core is imported)

    // initialize constants and configurations
    // with default values
    init_bw_constants(&bwc);
    rWWModel::init_constants(&(rWWModel::mc));
    init_conf(&conf);

    Py_RETURN_NONE;
}

static PyObject* set_const(PyObject* self, PyObject* args) {
    // sets model constants (currently only k1, k2, k3 and V0 of BW model)
    // TODO: consider getting the constants from the user
    // in run_simulations function
    const char* key;
    double value;

    if (!PyArg_ParseTuple(args, "sd", &key, &value)) {
        return NULL;
    }

    // update value of the constant
    if (strcmp(key, "k1") == 0) {
        bwc.k1 = value;
    }
    else if (strcmp(key, "k2") == 0) {
        bwc.k2 = value;
    }
    else if (strcmp(key, "k3") == 0) {
        bwc.k3 = value;
    }
    else if (strcmp(key, "V_0") == 0) {
        bwc.V_0 = value;
    }

    // update derived constants
    bwc.V_0_k1 = bwc.V_0 * bwc.k1;
    bwc.V_0_k2 = bwc.V_0 * bwc.k2;
    bwc.V_0_k3 = bwc.V_0 * bwc.k3;

    // set is_initialized to false so that the next time run_simulations
    // is called, the session will be reinitialized
    // (and new constants will be used)
    // bnm_cpu::is_initialized = false;
    #ifdef GPU_ENABLED
    bnm_gpu::is_initialized = false;
    #endif

    Py_RETURN_NONE;
}

static PyObject* set_conf(PyObject* self, PyObject* args) {
    // sets model configs
    const char* key;
    double value; // can be cast to int/bool if needed

    if (!PyArg_ParseTuple(args, "sd", &key, &value)) {
        return NULL;
    }

    if (strcmp(key, "bold_remove_s") == 0) {
        conf.bold_remove_s = (int)value;
    }
    else if (strcmp(key, "exc_interhemispheric") == 0) {
        conf.exc_interhemispheric = (bool)value;
    }
    else if (strcmp(key, "drop_edges") == 0) {
        conf.drop_edges = (bool)value;
    }
    else if (strcmp(key, "sync_msec") == 0) {
        conf.sync_msec = (bool)value;
    }
    else if (strcmp(key, "extended_output_ts") == 0) {
        conf.extended_output_ts = (bool)value;
    }
    else if (strcmp(key, "sim_verbose") == 0) {
        conf.sim_verbose = (bool)value;
    }

    // set is_initialized to false so that the next time run_simulations
    // is called, the session will be reinitialized
    // (and new constants will be used)
    // bnm_cpu::is_initialized = false;
    #ifdef GPU_ENABLED
    bnm_gpu::is_initialized = false;
    #endif

    Py_RETURN_NONE;
}

static PyObject* get_conf(PyObject* self, PyObject* args) {
    // Create a Python dictionary to store the configuration values
    PyObject* conf_dict = PyDict_New();
    if (conf_dict == NULL) {
        return NULL;
    }

    // Add the configuration values to the dictionary
    PyDict_SetItemString(conf_dict, "bold_remove_s", PyLong_FromLong(conf.bold_remove_s));
    PyDict_SetItemString(conf_dict, "exc_interhemispheric", PyBool_FromLong(conf.exc_interhemispheric));
    PyDict_SetItemString(conf_dict, "drop_edges", PyBool_FromLong(conf.drop_edges));
    PyDict_SetItemString(conf_dict, "sync_msec", PyBool_FromLong(conf.sync_msec));
    PyDict_SetItemString(conf_dict, "extended_output_ts", PyBool_FromLong(conf.extended_output_ts));
    PyDict_SetItemString(conf_dict, "sim_verbose", PyBool_FromLong(conf.sim_verbose));

    return conf_dict;
}

static PyObject* run_simulations(PyObject* self, PyObject* args) {
    char* model_name;
    PyArrayObject *SC, *SC_dist, *py_global_params, *py_regional_params, *v_list;
    PyObject* config_dict;
    bool extended_output, extended_output_ts, do_delay, force_reinit, use_cpu;
    int N_SIMS, nodes, time_steps, BOLD_TR, window_size, window_step, rand_seed;

    if (!PyArg_ParseTuple(args, "sO!O!O!O!O!Oiiiiiiiiiiii", 
            &model_name,
            &PyArray_Type, &SC,
            &PyArray_Type, &SC_dist,
            &PyArray_Type, &py_global_params,
            &PyArray_Type, &py_regional_params,
            &PyArray_Type, &v_list,
            &config_dict,
            &extended_output,
            &extended_output_ts,
            &do_delay,
            &force_reinit,
            &use_cpu,
            &N_SIMS,
            &nodes,
            &time_steps,
            &BOLD_TR,
            &window_size,
            &window_step,
            &rand_seed
            ))
        return NULL;

    py_global_params = (PyArrayObject*)PyArray_FROM_OTF((PyObject*)py_global_params, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    py_regional_params = (PyArrayObject*)PyArray_FROM_OTF((PyObject*)py_regional_params, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if ((py_global_params == NULL) | (py_regional_params == NULL)) return NULL;

    u_real ** global_params = np_to_array_2d(py_global_params);
    u_real ** regional_params = np_to_array_2d(py_regional_params);


    if (nodes > MAX_NODES) {
        printf("nodes must be less than %d\n", MAX_NODES);
        #ifndef MANY_NODES
        printf("To use more nodes, recompile the library with MANY_NODES flag.\n");
        #endif
        return NULL;
    }

    // initialize the model
    typedef rWWModel Model; // TODO: determine the model based on model_name
    Model* model = new Model();
    // set model configs
    std::map<std::string, std::string> config_map = dict_to_map(config_dict);
    model->init_config(&(model->conf)); // initialize to default values
    model->set_config(config_map, &(model->conf)); // update with user values if provided
   
    // printf("N_SIMS %d nodes %d time_steps %d BOLD_TR %d window_size %d window_step %d rand_seed %d extended_output %d do_delay %d use_cpu %d\n", 
    //     N_SIMS, nodes, time_steps, BOLD_TR, window_size, window_step, rand_seed, extended_output, do_delay, use_cpu);

    // update conf.extended_output_ts based on extended_output_ts
    // TODO: consider using this approach for other configs as well
    // (e.g. extended_output), as opposed to passing them as function arguments
    conf.extended_output_ts = extended_output_ts;

    // copy SC to SC_gsl if FIC is needed
    gsl_matrix *SC_gsl;
    if (strcmp(Model::name, "rWW")==0 && model->conf.do_fic) {
        SC_gsl = gsl_matrix_alloc(nodes, nodes);
        for (int i = 0; i < nodes; i++) {
            for (int j = 0; j < nodes; j++) {
                gsl_matrix_set(SC_gsl, i, j, ((double*)PyArray_DATA(SC))[i*nodes + j]);
            }
        }
    }


    // time_t start, end;
    std::chrono::time_point<std::chrono::system_clock> start, end;
    std::chrono::duration<double> elapsed_seconds;

    bool is_initialized;
    if (use_cpu) {
        // is_initialized = bnm_cpu::is_initialized;
    } else {
        #ifdef GPU_ENABLED
        is_initialized = bnm_gpu::is_initialized;
        #else
        // TODO: write a proper warning + instructions on what to do
        // if the system does have a CUDA-enabled GPU
        printf("Library not compiled with GPU support and cannot use GPU.\n");
        return NULL;
        #endif
    }

    // Initialize GPU/CPU if it's not already done in current session
    // it does memory allocation and noise precalculation among other things
    if ((!is_initialized) | (force_reinit)) {
        printf("Initializing the session...");
        start = std::chrono::high_resolution_clock::now();
        if (use_cpu) {
            printf("\nCurrently only GPU is supported...Exiting\n");
            return NULL;
            // init_cpu(
            //     &_output_ts, &_n_pairs, &_n_window_pairs,
            //     nodes,rand_seed, BOLD_TR, time_steps, window_size, window_step,
            //     mc, conf);
        } 
        #ifdef GPU_ENABLED
        else {
            init_gpu<Model>(
                &_output_ts, &_n_pairs, &_n_window_pairs,
                N_SIMS, nodes,
                extended_output, rand_seed, BOLD_TR, time_steps, window_size, window_step,
                model, bwc, conf, (!is_initialized));
        }
        #endif
        end = std::chrono::high_resolution_clock::now();
        elapsed_seconds = end - start;
        printf("took %lf s\n", elapsed_seconds.count());
    } else {
        printf("Current session is already initialized\n");
    }

    // Create NumPy arrays from BOLD_ex_out, fc_trils_out, and fcd_trils_out
    npy_intp bold_dims[2] = {N_SIMS, _output_ts*nodes};
    npy_intp fc_trils_dims[2] = {N_SIMS, _n_pairs};
    npy_intp fcd_trils_dims[2] = {N_SIMS, _n_window_pairs};
    npy_intp states_dims[3] = {Model::n_state_vars, N_SIMS, nodes};
    if (conf.extended_output_ts) {
        states_dims[2] *= _output_ts;
    }
    npy_intp global_bools_dims[2] = {Model::n_global_out_bool, N_SIMS};
    npy_intp global_ints_dims[2] = {Model::n_global_out_int, N_SIMS};

    PyObject* py_BOLD_ex_out = PyArray_SimpleNew(2, bold_dims, PyArray_DOUBLE);
    PyObject* py_fc_trils_out = PyArray_SimpleNew(2, fc_trils_dims, PyArray_DOUBLE);
    PyObject* py_fcd_trils_out = PyArray_SimpleNew(2, fcd_trils_dims, PyArray_DOUBLE);
    PyObject* py_states_out = PyArray_SimpleNew(3, states_dims, PyArray_DOUBLE);
    PyObject* py_global_bools_out = PyArray_SimpleNew(2, global_bools_dims, PyArray_BOOL);
    PyObject* py_global_ints_out = PyArray_SimpleNew(2, global_ints_dims, PyArray_INT);

    printf("Running %d simulations...\n", N_SIMS);
    start = std::chrono::high_resolution_clock::now();
    // TODO: cast the arrays to double if u_real is float
    if (use_cpu) {
        // run_simulations_cpu(
        //     (double*)PyArray_DATA(py_BOLD_ex_out), (double*)PyArray_DATA(py_fc_trils_out), 
        //     (double*)PyArray_DATA(py_fcd_trils_out),
        //     (double*)PyArray_DATA(py_S_E_out), (double*)PyArray_DATA(py_S_I_out),
        //     (double*)PyArray_DATA(py_r_E_out), (double*)PyArray_DATA(py_r_I_out),
        //     (double*)PyArray_DATA(py_I_E_out), (double*)PyArray_DATA(py_I_I_out),
        //     (bool*)PyArray_DATA(py_fic_unstable_out), (bool*)PyArray_DATA(py_fic_failed_out),
        //     (double*)PyArray_DATA(G_list), (double*)PyArray_DATA(w_EE_list), 
        //     (double*)PyArray_DATA(w_EI_list), (double*)PyArray_DATA(w_IE_list),
        //     (double*)PyArray_DATA(v_list),
        //     (double*)PyArray_DATA(SC), SC_gsl, (double*)PyArray_DATA(SC_dist), do_delay, nodes,
        //     time_steps, BOLD_TR, _max_fic_trials,
        //     window_size, window_step,
        //     N_SIMS, do_fic, only_wIE_free, extended_output, 
        //     mc, conf, rand_seed
        // );
    }
    #ifdef GPU_ENABLED
    else {
        run_simulations_gpu<Model>(
            (double*)PyArray_DATA(py_BOLD_ex_out), (double*)PyArray_DATA(py_fc_trils_out), 
            (double*)PyArray_DATA(py_fcd_trils_out),
            global_params, regional_params, 
            (double*)PyArray_DATA(v_list),
            (double*)PyArray_DATA(SC), SC_gsl, (double*)PyArray_DATA(SC_dist), do_delay, nodes,
            time_steps, BOLD_TR,
            window_size, N_SIMS, extended_output, model, conf
        );
    }
    #endif
    end = std::chrono::high_resolution_clock::now();
    elapsed_seconds = end - start;
    printf("took %lf s\n", elapsed_seconds.count());

    array_to_np_3d<u_real>(bnm_gpu::states_out, py_states_out);
    array_to_np_2d<bool>(bnm_gpu::global_out_bool, py_global_bools_out);
    array_to_np_2d<int>(bnm_gpu::global_out_int, py_global_ints_out);

    if (strcmp(Model::name, "rWW")==0 && model->conf.do_fic) {
        gsl_matrix_free(SC_gsl);
        // copy updated wIE back to py_regional_params
        int i = 2; // wIE index
        for (int j = 0; j < N_SIMS*nodes; j++) {
            ((double*)PyArray_DATA(py_regional_params))[i*N_SIMS*nodes + j] = regional_params[i][j];
        }
    }

    if (extended_output) {
        return Py_BuildValue("OOOOOO", 
            py_BOLD_ex_out, py_fc_trils_out, py_fcd_trils_out,
            py_states_out, py_global_bools_out, py_global_ints_out
        );
    } 
    else {
        return Py_BuildValue("OOO", py_BOLD_ex_out, py_fc_trils_out, py_fcd_trils_out);
    }
}


static PyMethodDef methods[] = {
    {"run_simulations", run_simulations, METH_VARARGS, 
        "run_simulations(model_name, SC, SC_dist, global_params, regional_params, \n"
        "v_list, model_config, extended_output, extended_output_ts do_delay, force_reinit, \n"
        "use_cpu, N_SIMS, nodes, time_steps, BOLD_TR, window_size, window_step, rand_seed)\n\n"
        "This function serves as an interface to run a group of simulations on GPU/CPU.\n\n"
        "Parameters:\n"
        "-----------\n"
        "model_name (str)\n"
            "\tname of the model to use\n"
            "\tcurrently only supports 'rWW'\n"
        "SC (np.ndarray) (nodes*nodes,)\n\tflattened strucutral connectivity matrix\n"
        "SC_dist (np.ndarray) (nodes*nodes,)\n"
            "\tflattened edge length matrix\n"
            "\twill be ignored if do_delay is False\n"
        "global_params (np.ndarray) (n_global_params, N_SIM)\n"
            "\tarray of global model parameters\n"
        "regional_params (np.ndarray) (n_regional_params, N_SIMS*nodes)\n"
            "\tflattened array of regional model parameters\n"
        "v_list (np.ndarray) (N_SIMS,)\n"
            "\tarray of conduction velocity values\n"
            "\twill be ignored if do_delay is False\n"
        "model_config (dict)\n"
            "\tmodel-specific configurations as a dictioary with\n"
            "\tstring keys and values. Provide an empty dictionary\n"
            "\tif no custom configurations are needed / to use defaults\n"
        "extended_output (bool)\n"
            "\twhether to return extended output (averaged across time)\n"
        "extended_output (bool)\n"
            "\twhether to return extended output as time series\n"
        "do_delay (bool)\n"
            "\twhether to consider inter-regional conduction delay \n"
        "force_reinit (bool)\n"
            "\twhether to force reinitialization of the session\n"
        "use_cpu (bool)\n"
            "\twhether to use CPU instead of GPU\n"
        "N_SIMS (int)\n"
            "\tnumber of simulations to run\n"
        "nodes (int)\n"
            "\tnumber of nodes in the network\n"
        "time_steps (int)\n"
            "\tduration of simulations (ms)\n"
        "BOLD_TR (int)\n"
            "\tBOLD repetition time (ms)\n"
            "\talso used as the sampling interval of extended output\n"
        "window_size (int)\n"
            "\tdynamic FC window size (number of TRs)\n"
        "window_step (int)\n"
            "\tdynamic FC window step (number of TRs)\n"
        "rand_seed (int)\n"
            "\tseed for random number generator\n\n"
        "Returns:\n"
        "--------\n"
        "sim_bold (np.ndarray) (N_SIMS, TRs*nodes)\n"
            "\tsimulated BOLD time series\n"
        "sim_fc (np.ndarray) (N_SIMS, edges)\n"
            "\tsimulated functional connectivity matrices\n"
        "sim_fcd (np.ndarray) (N_SIMS, n_window_pairs)\n"
            "\tsimulated functional connectivity dynamics matrices\n"
        "If extended_output is True, the function also returns "
        "the time-averaged model state variables, including:\n"
        "states_out (np.ndarray) (N_SIMS, nodes) or (N_SIMS, nodes*time)\n"
            "\tmodel state variables\n"
            "\tNote: if extended_output_ts is True, the time series "
            "of model state variables will be returned\n"
        "global_out_bool (np.ndarray) (n_global_out_bool, N_SIMS)\n"
            "\tglobal boolean outputs\n"
        "global_out_int (np.ndarray) (n_global_out_int, N_SIMS)\n"
            "\tglobal integer outputs\n"
    },
    {"init", init, METH_NOARGS, 
        "init()\n"
        "Initialize the session configs and constants"
    },
    {"set_conf", set_conf, METH_VARARGS, 
        "set_conf(key, value)\n"
        "Set the session configs.\n"
        "Parameters:\n"
        "-----------\n"
        "key (str)\n"
            "\tname of the config to set\n"
            "\tincluding: bold_remove_s, I_SAMPLING_START, I_SAMPLING_END, \n"
            "\tnumerical_fic, max_fic_trials, init_delta, exc_interhemispheric, \n"
            "\tdrop_edges, sync_msec, extended_output_ts, sim_verbose, fic_verbose\n"
        "value (float)\n"
            "\tvalue of the config to set\n"
            "\tfloat is converted to int/bool if needed\n"
    },
    {"set_const", set_const, METH_VARARGS, 
        "set_const(key, value)\n"
        "Set the value of a model constant.\n"
        "Currently only supports k1, k2, k3 and V0 of BW model\n\n"
        "Parameters:\n"
        "-----------\n"
        "key (str)\n"
            "\tname of the constant to set\n"
        "value (float)\n"
            "\tvalue of the constant to set\n\n"
    },
    {"get_conf", get_conf, METH_NOARGS, 
        "get_conf()\n"
        "Get the session configs.\n"
        "Returns:\n"
        "--------\n"
        "conf (dict)\n"
            "\tdictionary of session configs\n\n"
    },
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef coreModule = {
    PyModuleDef_HEAD_INIT, "core", // name of the module
    "core", -1, methods
};

PyMODINIT_FUNC PyInit_core(void) {
    import_array();
    return PyModule_Create(&coreModule);
}