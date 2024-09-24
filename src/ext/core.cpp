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
#include <iomanip>
#include "cubnm/defines.h"
#include "utils.cpp"
#include "./models/bw.cpp"
#include "./models/base.cpp"
#include "fc.cpp"
#include "bnm.cpp"
#include "./models/rww.cpp"
#include "./models/rww_fic.cpp"
#include "./models/rwwex.cpp"
#include "./models/kuramoto.cpp"

// create a pointer to the model object in current session
// so that it can be reused in subsequent calls to run_simulations
// this approach is an alternative to using `static` members
// which do not work very well with cuda
// but at any given time, only one model object 
// (and a copy of it on GPU) will exist
BaseModel *model;
char *last_model_name;


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

template<typename T>
void array_to_np_1d(T * arr, PyObject * np_arr) {
    // converts a 1d array to a nd numpy array
    // Note that unlike the other 2d and 3d functions
    // here np_arr can be n-dimensional
    int size = PyArray_Size(np_arr);
    T* data = (T*)PyArray_DATA(np_arr);

    for (int i = 0; i < size; i++) {
        data[i] = arr[i];
    }
}

static PyObject* init(PyObject* self, PyObject* args) {
    // this function is called only once at the beginning of the session
    // (i.e. when core is imported)

    // initialize ballon-windkessel model constants
    init_bw_constants(&bwc);

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

    // TODO: make sure that from Python side
    // reinitialization of the session is enforced
    // when the constants are updated

    Py_RETURN_NONE;
}


static PyObject* run_simulations(PyObject* self, PyObject* args) {
    char* model_name;
    PyArrayObject *py_SC, *py_SC_indices, *py_SC_dist, *py_global_params, *py_regional_params, *v_list;
    PyObject* config_dict;
    bool ext_out, states_ts, noise_out, do_delay, force_reinit, use_cpu;
    int N_SIMS, nodes, time_steps, BOLD_TR, states_sampling, rand_seed;
    double dt, bw_dt;

    if (!PyArg_ParseTuple(args, "sO!O!O!O!O!O!Oiiiiiiiiiiiidd", 
            &model_name,
            &PyArray_Type, &py_SC,
            &PyArray_Type, &py_SC_indices,
            &PyArray_Type, &py_SC_dist,
            &PyArray_Type, &py_global_params,
            &PyArray_Type, &py_regional_params,
            &PyArray_Type, &v_list,
            &config_dict,
            &ext_out,
            &states_ts,
            &noise_out,
            &do_delay,
            &force_reinit,
            &use_cpu,
            &N_SIMS,
            &nodes,
            &time_steps,
            &BOLD_TR,
            &states_sampling,
            &rand_seed,
            &dt,
            &bw_dt
            )) {
        std::cerr << "Error parsing arguments" << std::endl;
        Py_RETURN_NONE;
    }

    py_global_params = (PyArrayObject*)PyArray_FROM_OTF((PyObject*)py_global_params, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    py_regional_params = (PyArrayObject*)PyArray_FROM_OTF((PyObject*)py_regional_params, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    py_SC = (PyArrayObject*)PyArray_FROM_OTF((PyObject*)py_SC, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if ((py_global_params == NULL) | (py_regional_params == NULL) | (py_SC == NULL)) return NULL;

    u_real ** global_params = np_to_array_2d(py_global_params);
    u_real ** regional_params = np_to_array_2d(py_regional_params);
    u_real ** SC = np_to_array_2d(py_SC);
    // calcualte number of SCs as the max value of py_SC_idx
    int N_SCs = *std::max_element(
        (int*)PyArray_DATA(py_SC_indices), 
        (int*)PyArray_DATA(py_SC_indices) + PyArray_SIZE(py_SC_indices)
    ) + 1;

    #ifndef MANY_NODES
    if ((nodes > MAX_NODES)) {
        std::cerr << "Number of nodes must be less than " 
            << MAX_NODES << std::endl <<
            "To use more nodes, reinstall the package after" 
            << " `export CUBNM_MANY_NODES=1`" << std::endl;
        Py_RETURN_NONE;
    }
    #endif

    // initialize the model object if needed
    if (
            (model == nullptr) // first call
            || (strcmp(model_name, last_model_name)!=0) // different model
            #ifdef GPU_ENABLED
            || (model->cpu_initialized && (!use_cpu)) // CPU initialized but GPU is requested
            || (use_cpu && model->gpu_initialized) // GPU initialized but CPU is requested
            #endif 
        ) {
        last_model_name = model_name;
        if (model != nullptr) {
            delete model; // delete the old model to ensure only one model object exists
        }
        if (strcmp(model_name, "rWW")==0) {
            model = new rWWModel(
                nodes, N_SIMS, N_SCs, BOLD_TR, states_sampling, 
                time_steps, do_delay, rand_seed, dt, bw_dt
            );
        } 
        else if (strcmp(model_name, "rWWEx")==0) {
            model = new rWWExModel(
                nodes, N_SIMS, N_SCs, BOLD_TR, states_sampling, 
                time_steps, do_delay, rand_seed, dt, bw_dt
            );
        } 
        else if (strcmp(model_name, "Kuramoto")==0) {
            model = new KuramotoModel(
                nodes, N_SIMS, N_SCs, BOLD_TR, states_sampling, 
                time_steps, do_delay, rand_seed, dt, bw_dt
            );
        }
        else {
            std::cerr << "Model " << model_name << " not found" << std::endl;
            return NULL;
        }
    } else {
        // update model properties based on user data
        model->update(
            nodes, N_SIMS, N_SCs, BOLD_TR, states_sampling, 
            time_steps, do_delay, rand_seed, dt, bw_dt
        );
        // reset base_conf to defaults
        model->base_conf = BaseModel::Config();
    }
    // set model configs
    std::map<std::string, std::string> config_map = dict_to_map(config_dict);
    model->set_conf(config_map); // update with user values if provided
    model->base_conf.ext_out = ext_out;
    model->base_conf.states_ts = states_ts;

    std::chrono::time_point<std::chrono::system_clock> start, end;
    std::chrono::duration<double> init_seconds, run_seconds;

    if (!use_cpu) {
        #ifndef GPU_ENABLED
        // TODO: write a proper warning + instructions on what to do
        // if the system does have a CUDA-enabled GPU
        std::cerr << "Library not compiled with GPU support and cannot use GPU." << std::endl;
        return NULL;
        #endif
    }

    // Initialize GPU/CPU if it's not already done in current session
    // it does memory allocation and noise precalculation among other things
    start = std::chrono::high_resolution_clock::now();
    if (use_cpu & ((!model->cpu_initialized) | (force_reinit))) {
        std::cout << "Initializing CPU session..." << std::endl;
        model->init_cpu(force_reinit);
        end = std::chrono::high_resolution_clock::now();
        init_seconds = end - start;
        std::cout << "took " << std::fixed << std::setprecision(6)
            << init_seconds.count() << " s" << std::endl;
    } 
    #ifdef GPU_ENABLED
    else if (!use_cpu & ((!model->gpu_initialized) | (force_reinit))) {
        std::cout << "Initializing GPU session..." << std::endl;
        model->init_gpu(bwc, force_reinit);
        end = std::chrono::high_resolution_clock::now();
        init_seconds = end - start;
        std::cout << "took " << std::fixed << std::setprecision(6) 
            << init_seconds.count() << " s" << std::endl;
    }
    #endif
    else {
        std::cout << "Current session is already initialized" << std::endl;
    }

    // Create NumPy arrays for all the outputs
    // Note: some are directly passed to `run_simulations_*` functions
    // but for some (nd arrays and 1d arrays that need reshaping) the C array
    // including the output will be copied to the NumPy array after simulations
    // are done
    npy_intp bold_dims[2] = {N_SIMS, model->bold_len*nodes};
    npy_intp fc_trils_dims[2] = {N_SIMS, model->n_pairs};
    npy_intp fcd_trils_dims[2] = {N_SIMS, model->n_window_pairs};
    npy_intp states_dims[3] = {model->get_n_state_vars(), N_SIMS, nodes};
    if (model->base_conf.states_ts) {
        states_dims[2] *= model->states_len;
    }
    npy_intp global_bools_dims[2] = {model->get_n_global_out_bool(), N_SIMS};
    npy_intp global_ints_dims[2] = {model->get_n_global_out_int(), N_SIMS};
    npy_intp noise_dims[1] = {model->noise_size};
    #ifdef NOISE_SEGMENT
    npy_intp shuffled_nodes_dims[2] = {model->noise_repeats, model->nodes};
    npy_intp shuffled_ts_dims[2] = {model->noise_repeats, model->base_conf.noise_time_steps};
    #endif

    PyObject *py_BOLD_ex_out, *py_fc_trils_out, *py_fcd_trils_out,
        *py_states_out, *py_global_bools_out, *py_global_ints_out,
        *py_noise_out;
    #ifdef NOISE_SEGMENT
    PyObject *py_shuffled_nodes_out, *py_shuffled_ts_out;
    #endif

    // Allocate memory for the output arrays
    // for BOLD, fc_trils and fcd_trils cast the array
    // data to double pointer which will be passed on 
    // to run simulations
    // TODO: make the data transfer between GPU-C arrays-Python
    // consistent across variables and minimize the number of copies
    double *BOLD_ex_out, *fc_trils_out, *fcd_trils_out;
    py_BOLD_ex_out = PyArray_SimpleNew(2, bold_dims, PyArray_DOUBLE);
    BOLD_ex_out = (double*)PyArray_DATA(py_BOLD_ex_out);
    if (model->base_conf.do_fc) {
        py_fc_trils_out = PyArray_SimpleNew(2, fc_trils_dims, PyArray_DOUBLE);
        fc_trils_out = (double*)PyArray_DATA(py_fc_trils_out);
        if (model->base_conf.do_fcd) {
            py_fcd_trils_out = PyArray_SimpleNew(2, fcd_trils_dims, PyArray_DOUBLE);
            fcd_trils_out = (double*)PyArray_DATA(py_fcd_trils_out);
        }
    }
    if (ext_out) {
        py_states_out = PyArray_SimpleNew(3, states_dims, PyArray_DOUBLE);
        py_global_bools_out = PyArray_SimpleNew(2, global_bools_dims, PyArray_BOOL);
        py_global_ints_out = PyArray_SimpleNew(2, global_ints_dims, PyArray_INT);
    }
    if (noise_out) {
        py_noise_out = PyArray_SimpleNew(1, noise_dims, PyArray_DOUBLE);
        #ifdef NOISE_SEGMENT
        py_shuffled_nodes_out = PyArray_SimpleNew(2, shuffled_nodes_dims, PyArray_INT);
        py_shuffled_ts_out = PyArray_SimpleNew(2, shuffled_ts_dims, PyArray_INT);
        #endif
    }

    // Run simulations
    std::cout << "Running " << N_SIMS << " simulations..." << std::endl;
    start = std::chrono::high_resolution_clock::now();
    // TODO: cast the arrays to double if u_real is float
    if (use_cpu) {
        model->run_simulations_cpu(
            BOLD_ex_out, fc_trils_out, fcd_trils_out,
            global_params, regional_params, 
            (double*)PyArray_DATA(v_list),
            SC, (int*)PyArray_DATA(py_SC_indices), (double*)PyArray_DATA(py_SC_dist)
        );
    }
    #ifdef GPU_ENABLED
    else {
        model->run_simulations_gpu(
            BOLD_ex_out, fc_trils_out, fcd_trils_out,
            global_params, regional_params, 
            (double*)PyArray_DATA(v_list),
            SC, (int*)PyArray_DATA(py_SC_indices), (double*)PyArray_DATA(py_SC_dist)
        );
    }
    #endif
    end = std::chrono::high_resolution_clock::now();
    run_seconds = end - start;
    std::cout << "took " << std::fixed << std::setprecision(6)
        << run_seconds.count() << " s" << std::endl;

    // Copy the output C arrays to NumPy arrays
    if (ext_out) {
        array_to_np_3d<u_real>(model->states_out, py_states_out);
        array_to_np_2d<bool>(model->global_out_bool, py_global_bools_out);
        array_to_np_2d<int>(model->global_out_int, py_global_ints_out);
    }
    if (noise_out) {
        array_to_np_1d<u_real>(model->noise, py_noise_out);
        #ifdef NOISE_SEGMENT
        array_to_np_1d<int>(model->shuffled_nodes, py_shuffled_nodes_out);
        array_to_np_1d<int>(model->shuffled_ts, py_shuffled_ts_out);
        #endif
    }

    if (model->modifies_params) {
        // copy updated global_params back to py_global_params
        for (int i = 0; i < model->get_n_global_params(); i++) {
            for (int j = 0; j < N_SIMS; j++) {
                ((double*)PyArray_DATA(py_global_params))[i*N_SIMS + j] = global_params[i][j];
            }
        }
        // copy updated regional_params back to py_regional_params
        for (int i = 0; i < model->get_n_regional_params(); i++) {
            for (int j = 0; j < N_SIMS*nodes; j++) {
                ((double*)PyArray_DATA(py_regional_params))[i*N_SIMS*nodes + j] = regional_params[i][j];
            }
        }
    }

    // convert inti_seconds and run_seconds to Python floats
    PyObject* py_init_seconds = PyFloat_FromDouble(init_seconds.count());
    PyObject* py_run_seconds = PyFloat_FromDouble(run_seconds.count());
    
    // Return output as a list with varying number of elements
    // depending on ext_out and noise_out
    PyObject* out_list = PyList_New(3);
    PyList_SetItem(out_list, 0, py_init_seconds);
    PyList_SetItem(out_list, 1, py_run_seconds);
    PyList_SetItem(out_list, 2, py_BOLD_ex_out);
    if (model->base_conf.do_fc) {
        PyList_Append(out_list, py_fc_trils_out);
        if (model->base_conf.do_fcd) {
            PyList_Append(out_list, py_fcd_trils_out);
        }
    }
    if (ext_out) {
        PyList_Append(out_list, py_states_out);
        PyList_Append(out_list, py_global_bools_out);
        PyList_Append(out_list, py_global_ints_out);
    }
    if (noise_out) {
        PyList_Append(out_list, py_noise_out);
        #ifdef NOISE_SEGMENT
        PyList_Append(out_list, py_shuffled_nodes_out);
        PyList_Append(out_list, py_shuffled_ts_out);
        #endif
    }
    return out_list;
}


static PyMethodDef methods[] = {
    {"run_simulations", run_simulations, METH_VARARGS, 
        "run_simulations(model_name, SC, SC_indices, SC_dist, global_params, regional_params, \n"
        "v_list, model_config, ext_out, states_ts, noise_out, do_delay, force_reinit, \n"
        "use_cpu, N_SIMS, nodes, time_steps, BOLD_TR, rand_seed, dt, bw_dt)\n\n"
        "This function serves as an interface to run a group of simulations on GPU/CPU.\n\n"
        "Parameters:\n"
        "-----------\n"
        "model_name (str)\n"
            "\tname of the model to use\n"
            "\tcurrently only supports 'rWW'\n"
        "SC (np.ndarray) (n_SC, nodes*nodes)\n"
            "\tn_sc flattened strucutral connectivity matrices\n"
            "\if asymmetric, rows are sources and columns are targets\n"
        "SC_idx (np.ndarray) (N_SIMS,)\n"
            "\tindex of SC to use for each simulation"
        "SC_dist (np.ndarray) (nodes*nodes,)\n"
            "\tflattened edge length matrix\n"
            "\twill be ignored if do_delay is False\n"
            "\if asymmetric, rows are sources and columns are targets\n"
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
        "ext_out (bool)\n"
            "\twhether to return extended output (averaged across time)\n"
        "states_ts (bool)\n"
            "\twhether to return extended output as time series\n"
        "noise_out (bool)\n"
            "\twhether to return noise"
            #ifdef NOISE_SEGMENT
            " segment and ts and nodes shuffling\n"
            #else
            "\n"
            #endif
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
        "rand_seed (int)\n"
            "\tseed for random number generator\n\n"
        "dt (float)\n"
            "\tintegration time step (msec)\n"
        "bw_dt (float)\n"
            "\tintegration time step for the BOLD model (msec)\n\n"
        "Returns:\n"
        "--------\n"
        "sim_bold (np.ndarray) (N_SIMS, TRs*nodes)\n"
            "\tsimulated BOLD time series\n"
        "If config['do_fc'] is True, the function also returns:\n"
        "sim_fc (np.ndarray) (N_SIMS, edges)\n"
            "\tsimulated functional connectivity matrices\n"
        "If config['do_fcd'] is True, the function also returns:\n"
        "sim_fcd (np.ndarray) (N_SIMS, n_window_pairs)\n"
            "\tsimulated functional connectivity dynamics matrices\n"
        "If ext_out is True, the function also returns "
        "the time-averaged model state variables, including:\n"
        "states_out (np.ndarray) (N_SIMS, nodes) or (N_SIMS, nodes*time)\n"
            "\tmodel state variables\n"
            "\tNote: if states_ts is True, the time series "
            "of model state variables will be returned\n"
        "global_out_bool (np.ndarray) (n_global_out_bool, N_SIMS)\n"
            "\tglobal boolean outputs\n"
        "global_out_int (np.ndarray) (n_global_out_int, N_SIMS)\n"
            "\tglobal integer outputs\n"
        "If noise_out is True, the function also returns:\n"
        "noise (np.ndarray) (noise_size,)\n"
            "\tnoise array\n"
        #ifdef NOISE_SEGMENT
        "shuffled_nodes (np.ndarray) (noise_repeats, nodes)\n"
            "\tshuffled nodes\n"
        "shuffled_ts (np.ndarray) (noise_repeats, noise_time_steps)\n"
            "\tshuffled time series\n"
        #endif
    },
    {"init", init, METH_NOARGS, 
        "init()\n"
        "Initialize the session constants"
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
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef coreModule = {
    PyModuleDef_HEAD_INIT, "core",
    "core", -1, methods
};

PyMODINIT_FUNC PyInit_core(void) {
    import_array();
    return PyModule_Create(&coreModule);
}
