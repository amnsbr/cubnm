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
#include "constants.cpp"
#include "fic.cpp"
#include "utils.cpp"
#include "bnm.cpp"

#ifdef GPU_ENABLED
// declare gpu functions which will be provided by bnm.cu compiled library
extern void init_gpu(
        int *output_ts_p, int *n_pairs_p, int *n_window_pairs_p,
        int N_SIMS, int nodes, bool do_fic, bool extended_output, int rand_seed,
        int BOLD_TR, int time_steps, int window_size, int window_step,
        struct ModelConstants mc, struct ModelConfigs conf, bool verbose
        );
extern void run_simulations_gpu(
    double * BOLD_ex_out, double * fc_trils_out, double * fcd_trils_out,
    double * S_E_out, double * S_I_out, double * S_ratio_out,
    double * r_E_out, double * r_I_out, double * r_ratio_out,
    double * I_E_out, double * I_I_out, double * I_ratio_out,
    bool * fic_unstable_out, bool * fic_failed_out,
    u_real * G_list, u_real * w_EE_list, u_real * w_EI_list, u_real * w_IE_list, u_real * v_list,
    u_real * SC, gsl_matrix * SC_gsl, u_real * SC_dist, bool do_delay, int nodes,
    int time_steps, int BOLD_TR, int _max_fic_trials, int window_size,
    int N_SIMS, bool do_fic, bool only_wIE_free, bool extended_output,
    struct ModelConfigs conf
);
namespace bnm_gpu {
    extern bool is_initialized;
}
#endif
// the following variables are calcualted during init functions and will
// be needed to determine the size of arrays
// they are global because init functions may be called only once
int _output_ts, _n_pairs, _n_window_pairs;

// write a python extension function with no arguments named init which
// returns nothing
static PyObject* init(PyObject* self, PyObject* args) {
    // this function is called only once at the beginning of the session
    // (i.e. when core is imported)

    // initialize constants and configurations
    // with default values
    init_constants(&mc);
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
        mc.k1 = value;
    }
    else if (strcmp(key, "k2") == 0) {
        mc.k2 = value;
    }
    else if (strcmp(key, "k3") == 0) {
        mc.k3 = value;
    }
    else if (strcmp(key, "V_0") == 0) {
        mc.V_0 = value;
    }

    // update derived constants
    mc.V_0_k1 = mc.V_0 * mc.k1;
    mc.V_0_k2 = mc.V_0 * mc.k2;
    mc.V_0_k3 = mc.V_0 * mc.k3;

    // set is_initialized to false so that the next time run_simulations
    // is called, the session will be reinitialized
    // (and new constants will be used)
    bnm_cpu::is_initialized = false;
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
    else if (strcmp(key, "I_SAMPLING_START") == 0) {
        conf.I_SAMPLING_START = (int)value;
        conf.I_SAMPLING_DURATION = conf.I_SAMPLING_END - conf.I_SAMPLING_START + 1;
    }
    else if (strcmp(key, "I_SAMPLING_END") == 0) {
        conf.I_SAMPLING_END = (int)value;
        conf.I_SAMPLING_DURATION = conf.I_SAMPLING_END - conf.I_SAMPLING_START + 1;
    }
    else if (strcmp(key, "numerical_fic") == 0) {
        conf.numerical_fic = (bool)value;
    }
    else if (strcmp(key, "max_fic_trials") == 0) {
        conf.max_fic_trials = (int)value;
    }
    else if (strcmp(key, "init_delta") == 0) {
        conf.init_delta = value;
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
    else if (strcmp(key, "fic_verbose") == 0) {
        conf.fic_verbose = (bool)value;
    }

    // set is_initialized to false so that the next time run_simulations
    // is called, the session will be reinitialized
    // (and new constants will be used)
    bnm_cpu::is_initialized = false;
    #ifdef GPU_ENABLED
    bnm_gpu::is_initialized = false;
    #endif

    Py_RETURN_NONE;
}

static PyObject* run_simulations_interface(PyObject* self, PyObject* args) {
    PyArrayObject *SC, *SC_dist, *G_list, *w_EE_list, *w_EI_list, *w_IE_list, *v_list;
    bool do_fic, extended_output, do_delay, force_reinit, use_cpu;
    int N_SIMS, nodes, time_steps, BOLD_TR, window_size, window_step, rand_seed;
    double velocity;

    if (!PyArg_ParseTuple(args, "O!O!O!O!O!O!O!iiiiiiiiiiii", 
            &PyArray_Type, &SC,
            &PyArray_Type, &SC_dist,
            &PyArray_Type, &G_list,
            &PyArray_Type, &w_EE_list,
            &PyArray_Type, &w_EI_list,
            &PyArray_Type, &w_IE_list,
            &PyArray_Type, &v_list,
            &do_fic,
            &extended_output,
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

    // make sure arrays are 1D and of type double
    if (
        SC->nd != 1 || SC->descr->type_num != PyArray_DOUBLE ||
        SC_dist->nd != 1 || SC_dist->descr->type_num != PyArray_DOUBLE ||
        G_list->nd != 1 || G_list->descr->type_num != PyArray_DOUBLE ||
        w_EE_list->nd != 1 || w_EE_list->descr->type_num != PyArray_DOUBLE ||
        w_EE_list->nd != 1 || w_EE_list->descr->type_num != PyArray_DOUBLE ||
        w_IE_list->nd != 1 || w_IE_list->descr->type_num != PyArray_DOUBLE ||
        v_list->nd != 1 || v_list->descr->type_num != PyArray_DOUBLE
    ) {
        PyErr_SetString(PyExc_ValueError, "arrays must be one-dimensional and of type float");
        return NULL;
    }

    if (nodes > MAX_NODES) {
        printf("nodes must be less than %d\n", MAX_NODES);
        #ifndef MANY_NODES
        printf("To use more nodes, recompile the library with MANY_NODES flag.\n");
        #endif
        return NULL;
    }
   
    printf("do_fic %d N_SIMS %d nodes %d time_steps %d BOLD_TR %d window_size %d window_step %d rand_seed %d extended_output %d do_delay %d use_cpu %d\n", 
        do_fic, N_SIMS, nodes, time_steps, BOLD_TR, window_size, window_step, rand_seed, extended_output, do_delay, use_cpu);

    // TODO: these should be determined by the user
    // maybe via env variables
    bool only_wIE_free = false;
    int _max_fic_trials = conf.max_fic_trials;

    // copy SC to SC_gsl if FIC is needed
    gsl_matrix *SC_gsl;
    if (do_fic) {
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
        is_initialized = bnm_cpu::is_initialized;
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
            init_cpu(
                &_output_ts, &_n_pairs, &_n_window_pairs,
                nodes,rand_seed, BOLD_TR, time_steps, window_size, window_step,
                mc, conf);
        } 
        #ifdef GPU_ENABLED
        else {
            init_gpu(
                &_output_ts, &_n_pairs, &_n_window_pairs,
                N_SIMS, nodes,
                do_fic, extended_output, rand_seed, BOLD_TR, time_steps, window_size, window_step,
                mc, conf, (!is_initialized));
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
    npy_intp ext_var_dims[2] = {N_SIMS, nodes};
    PyObject* py_BOLD_ex_out = PyArray_SimpleNew(2, bold_dims, PyArray_DOUBLE);
    PyObject* py_fc_trils_out = PyArray_SimpleNew(2, fc_trils_dims, PyArray_DOUBLE);
    PyObject* py_fcd_trils_out = PyArray_SimpleNew(2, fcd_trils_dims, PyArray_DOUBLE);
    PyObject* py_S_E_out = PyArray_SimpleNew(2, ext_var_dims, PyArray_DOUBLE);
    PyObject* py_S_I_out = PyArray_SimpleNew(2, ext_var_dims, PyArray_DOUBLE);
    PyObject* py_S_ratio_out = PyArray_SimpleNew(2, ext_var_dims, PyArray_DOUBLE);
    PyObject* py_r_E_out = PyArray_SimpleNew(2, ext_var_dims, PyArray_DOUBLE);
    PyObject* py_r_I_out = PyArray_SimpleNew(2, ext_var_dims, PyArray_DOUBLE);
    PyObject* py_r_ratio_out = PyArray_SimpleNew(2, ext_var_dims, PyArray_DOUBLE);
    PyObject* py_I_E_out = PyArray_SimpleNew(2, ext_var_dims, PyArray_DOUBLE);
    PyObject* py_I_I_out = PyArray_SimpleNew(2, ext_var_dims, PyArray_DOUBLE);
    PyObject* py_I_ratio_out = PyArray_SimpleNew(2, ext_var_dims, PyArray_DOUBLE);
    npy_intp sims_dims[1] = {N_SIMS};
    PyObject* py_fic_unstable_out = PyArray_SimpleNew(1, sims_dims, PyArray_BOOL);
    PyObject* py_fic_failed_out = PyArray_SimpleNew(1, sims_dims, PyArray_BOOL);

    printf("Running %d simulations...\n", N_SIMS);
    start = std::chrono::high_resolution_clock::now();
    // TODO: cast the arrays to double if u_real is float
    if (use_cpu) {
        run_simulations_cpu(
            (double*)PyArray_DATA(py_BOLD_ex_out), (double*)PyArray_DATA(py_fc_trils_out), 
            (double*)PyArray_DATA(py_fcd_trils_out),
            (double*)PyArray_DATA(py_S_E_out), (double*)PyArray_DATA(py_S_I_out), (double*)PyArray_DATA(py_S_ratio_out),
            (double*)PyArray_DATA(py_r_E_out), (double*)PyArray_DATA(py_r_I_out), (double*)PyArray_DATA(py_r_ratio_out),
            (double*)PyArray_DATA(py_I_E_out), (double*)PyArray_DATA(py_I_I_out), (double*)PyArray_DATA(py_I_ratio_out),
            (bool*)PyArray_DATA(py_fic_unstable_out), (bool*)PyArray_DATA(py_fic_failed_out),
            (double*)PyArray_DATA(G_list), (double*)PyArray_DATA(w_EE_list), 
            (double*)PyArray_DATA(w_EI_list), (double*)PyArray_DATA(w_IE_list),
            (double*)PyArray_DATA(v_list),
            (double*)PyArray_DATA(SC), SC_gsl, (double*)PyArray_DATA(SC_dist), do_delay, nodes,
            time_steps, BOLD_TR, _max_fic_trials,
            window_size, window_step,
            N_SIMS, do_fic, only_wIE_free, extended_output, 
            mc, conf, rand_seed
        );
    }
    #ifdef GPU_ENABLED
    else {
        run_simulations_gpu(
            (double*)PyArray_DATA(py_BOLD_ex_out), (double*)PyArray_DATA(py_fc_trils_out), 
            (double*)PyArray_DATA(py_fcd_trils_out),
            (double*)PyArray_DATA(py_S_E_out), (double*)PyArray_DATA(py_S_I_out), (double*)PyArray_DATA(py_S_ratio_out),
            (double*)PyArray_DATA(py_r_E_out), (double*)PyArray_DATA(py_r_I_out), (double*)PyArray_DATA(py_r_ratio_out),
            (double*)PyArray_DATA(py_I_E_out), (double*)PyArray_DATA(py_I_I_out), (double*)PyArray_DATA(py_I_ratio_out),
            (bool*)PyArray_DATA(py_fic_unstable_out), (bool*)PyArray_DATA(py_fic_failed_out),
            (double*)PyArray_DATA(G_list), (double*)PyArray_DATA(w_EE_list), 
            (double*)PyArray_DATA(w_EI_list), (double*)PyArray_DATA(w_IE_list),
            (double*)PyArray_DATA(v_list),
            (double*)PyArray_DATA(SC), SC_gsl, (double*)PyArray_DATA(SC_dist), do_delay, nodes,
            time_steps, BOLD_TR, _max_fic_trials,
            window_size, N_SIMS, do_fic, only_wIE_free, extended_output, conf
        );
    }
    #endif
    end = std::chrono::high_resolution_clock::now();
    elapsed_seconds = end - start;
    printf("took %lf s\n", elapsed_seconds.count());

    if (extended_output) {
        return Py_BuildValue("OOOOOOOOOOOOO", py_BOLD_ex_out, py_fc_trils_out, py_fcd_trils_out,
            py_S_E_out, py_S_I_out, py_S_ratio_out, py_r_E_out, py_r_I_out, py_r_ratio_out,
            py_I_E_out, py_I_I_out, py_I_ratio_out, py_fic_unstable_out
        );
    } else {
        return Py_BuildValue("OOO", py_BOLD_ex_out, py_fc_trils_out, py_fcd_trils_out, py_fic_unstable_out);
    }
}


static PyMethodDef methods[] = {
    {"run_simulations", run_simulations_interface, METH_VARARGS, 
        "run_simulations(SC, SC_dist, G_list, w_EE_list, w_EI_list, w_IE_list, \n"
        "v_list, do_fic, extended_output, do_delay, force_reinit, use_cpu, \n"
        "N_SIMS, nodes, time_steps, BOLD_TR, window_size, window_step, rand_seed)\n\n"
        "This function serves as an interface to run a group of simulations on GPU/CPU.\n\n"
        "Parameters:\n"
        "-----------\n"
        "SC (np.ndarray) (nodes*nodes,)\n\tflattened strucutral connectivity matrix\n"
        "SC_dist (np.ndarray) (nodes*nodes,)\n"
            "\tflattened edge length matrix\n"
            "\twill be ignored if do_delay is False\n"
        "G_list (np.ndarray) (N_SIMS,)\n"
            "\tarray of global coupling values\n"
        "w_EE_list (np.ndarray) (N_SIMS*nodes,)\n"
            "\tflattened array of regional E-E synaptic weights\n"
        "w_EI_list (np.ndarray) (N_SIMS*nodes,)\n"
            "\tflattened array of regional E-I synaptic weights\n"
        "w_IE_list (np.ndarray) (N_SIMS*nodes,)\n"
            "\tflattened array of regional I-E synaptic weights\n"
        "v_list (np.ndarray) (N_SIMS,)\n"
            "\tarray of conduction velocity values\n"
            "\twill be ignored if do_delay is False\n"
        "do_fic (bool)\n"
            "\twhether to do feedback inhibition control\n"
        "extended_output (bool)\n"
            "\twhether to return extended output\n"
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
        "S_E (np.ndarray) (N_SIMS, nodes)\n"
        "S_I (np.ndarray) (N_SIMS, nodes)\n"
        "S_ratio (np.ndarray) (N_SIMS, nodes)\n"
        "r_E (np.ndarray) (N_SIMS, nodes)\n"
        "r_I (np.ndarray) (N_SIMS, nodes)\n"
        "r_ratio (np.ndarray) (N_SIMS, nodes)\n"
        "I_E (np.ndarray) (N_SIMS, nodes)\n"
        "I_I (np.ndarray) (N_SIMS, nodes)\n"
        "I_ratio (np.ndarray) (N_SIMS, nodes)\n"
        "fic_unstable (np.ndarray) (N_SIMS,)\n"
            "\tindicates whether analytical FIC led to unstable solution\n"
            "\tNote: if extended_output is True but do_fic is False," 
            "this array is still returned but will be empty\n"
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