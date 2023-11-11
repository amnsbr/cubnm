
#define PY_SSIZE_T_CLEAN
#include <stdio.h>
#include <Python.h>
#include <numpy/arrayobject.h>
#define UMALLOC(var, type, size) var = (type *)malloc(sizeof(type) * size)
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
#include <omp.h>
#include <ios>
#include <fstream>
#include <iomanip>
#include <complex>
#include <algorithm>
#include <memory>
#include "constants.cpp"
#include "fic.cpp"
// #include "bnm.cpp"

// declare gpu functions which will be provided by bnm.cu compiled library
extern void init_gpu(int N_SIMS, int nodes, bool do_fic, bool extended_output, int rand_seed,
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
    struct ModelConstants mc, struct ModelConfigs conf
);
extern bool gpu_initialized;
extern int n_pairs, n_window_pairs, output_ts; // will be defined by init_gpu

static PyObject* run_simulations_interface(PyObject* self, PyObject* args) {
    PyArrayObject *SC, *SC_dist, *G_list, *w_EE_list, *w_EI_list, *w_IE_list, *v_list;
    bool do_fic, extended_output, do_delay, force_reinit;
    int N_SIMS, nodes, time_steps, BOLD_TR, window_size, window_step, rand_seed;
    double velocity;

    if (!PyArg_ParseTuple(args, "O!O!O!O!O!O!O!iiiiiiiiiii", 
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
            &N_SIMS,
            &nodes,
            &time_steps,
            &BOLD_TR,
            &window_size,
            &window_step,
            &rand_seed
            ))
        return NULL;

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
        
    printf("do_fic %d N_SIMS %d nodes %d time_steps %d BOLD_TR %d window_size %d window_step %d rand_seed %d extended_output %d do_delay %d\n", 
        do_fic, N_SIMS, nodes, time_steps, BOLD_TR, window_size, window_step, rand_seed, extended_output, do_delay);

    // TODO: these should be determined by the user
    bool only_wIE_free = false;
    int _max_fic_trials = 5;

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

    // initialize constants and configurations
    init_constants(&mc);
    init_conf(&conf);

    // time_t start, end;
    std::chrono::time_point<std::chrono::system_clock> start, end;
    std::chrono::duration<double> elapsed_seconds;

    // Initialize GPU if it's not already done in current session
    // it does memory allocation and noise precalculation among other things
    if ((!gpu_initialized) | (force_reinit)) {
        printf("GPU initialization...");
        start = std::chrono::high_resolution_clock::now();
        init_gpu(N_SIMS, nodes,
            do_fic, extended_output, rand_seed, BOLD_TR, time_steps, window_size, window_step,
            mc, conf, (!gpu_initialized));
        end = std::chrono::high_resolution_clock::now();
        elapsed_seconds = end - start;
        printf("took %lf s\n", elapsed_seconds.count());
    } else {
        printf("GPU is already initialized\n");
    }

    // Create NumPy arrays from BOLD_ex_out, fc_trils_out, and fcd_trils_out
    npy_intp bold_dims[2] = {N_SIMS, output_ts*nodes};
    npy_intp fc_trils_dims[2] = {N_SIMS, n_pairs};
    npy_intp fcd_trils_dims[2] = {N_SIMS, n_window_pairs};
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
    // TODO: cast the arrays to float if u_real is float
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
        window_size, N_SIMS, do_fic, only_wIE_free, extended_output, 
        mc, conf
    );
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
    {"run_simulations", run_simulations_interface, METH_VARARGS, "runs simulations on GPU"},
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