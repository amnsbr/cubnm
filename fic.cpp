/*
Analytical Feedback Inhibition Control (FIC)
Calculates wIE needed in each node to maintain excitatory
firing rate of ~3 Hz.

Translated from Python code in https://github.com/murraylab/hbnm

Author: Amin Saberi, Feb 2023
*/
#ifndef FIC_CPP
#define FIC_CPP
// #include <iostream>
// #include <fstream>
// #include <cmath>
// #include <cstdlib>
// #include <sys/stat.h>
// #include <stdio.h>
// #include <gsl/gsl_matrix_double.h>
// #include <gsl/gsl_vector_double.h>
// #include <gsl/gsl_blas.h>
// #include <gsl/gsl_math.h>
// #include <gsl/gsl_errno.h>
// #include <gsl/gsl_roots.h>
// #include <vector>
// #include <memory>
// #include "constants.hpp"

// helper functions
void repeat(gsl_vector ** dest, double a, int size) {
    if (*dest == NULL || (*dest)->size != size) {
        *dest = gsl_vector_alloc(size);
    }
    gsl_vector_set_all(*dest, a);
}

void copy_array_to_vector(gsl_vector ** dest, double * src, int size) {
    if (*dest == NULL || (*dest)->size != size) {
        *dest = gsl_vector_alloc(size);
    }
    for (int i=0; i<size; i++) {
        gsl_vector_set(*dest, i, src[i]);
    }
}

void vector_scale(gsl_vector ** dest, gsl_vector * src, double a) {
    if (*dest == NULL || (*dest)->size != src->size) {
        *dest = gsl_vector_alloc(src->size);
    }
    gsl_vector_memcpy(*dest, src);
    gsl_vector_scale(*dest, a);
}

void mul_eye(gsl_matrix ** dest, double a, int size) {
    if (*dest == NULL || (*dest)->size1 != size) {
        *dest = gsl_matrix_alloc(size, size);
    }
    gsl_matrix_set_identity(*dest);
    gsl_matrix_scale(*dest, a);
}

void make_diag(gsl_matrix ** dest, gsl_vector * v) {
    int size = v->size;
    if (*dest == NULL || (*dest)->size1 != size) {
        *dest = gsl_matrix_calloc(size, size);
    } else {
        gsl_matrix_set_zero(*dest);
    }
    for (int i=0; i<size; i++) {
        gsl_matrix_set(*dest, i, i, gsl_vector_get(v, i));
    }
}

double gsl_fsolve(gsl_function F, double x_lo, double x_hi) {
    // Based on https://www.gnu.org/software/gsl/doc/html/roots.html#examples
    int status;
    int iter = 0, max_iter = 100;
    const gsl_root_fsolver_type *T;
    gsl_root_fsolver *s;
    double root = 0;

    T = gsl_root_fsolver_brent;
    s = gsl_root_fsolver_alloc(T);
    gsl_root_fsolver_set(s, &F, x_lo, x_hi);

    do
        {
        iter++;
        status = gsl_root_fsolver_iterate(s);
        root = gsl_root_fsolver_root(s);
        x_lo = gsl_root_fsolver_x_lower(s);
        x_hi = gsl_root_fsolver_x_upper(s);
        status = gsl_root_test_interval(x_lo, x_hi,
                                        0, 0.001);
        }
    while (status == GSL_CONTINUE && iter < max_iter);
    gsl_root_fsolver_free(s); 
    if (status != GSL_SUCCESS) {
        printf("Root solver did not converge\n");
        return -1;
    }
    return root;
}

// transfer function and derivatives for excitatory and inhibitory populations
double phi_E(double IE) {
    return ((mc.a_E * IE) - mc.b_E) / (1 - exp(-mc.d_E * ((mc.a_E * IE) - mc.b_E)));
}

double dphi_E(double IE) {
    return (
        (mc.a_E * (1 - exp(-1 * mc.d_E * ((mc.a_E * IE) - mc.b_E))))
        - (mc.a_E * mc.d_E * exp(-1 * mc.d_E * ((mc.a_E * IE) - mc.b_E)) * ((mc.a_E * IE) - mc.b_E))
        ) / pow((1 - exp(-1 * mc.d_E * ((mc.a_E * IE) - mc.b_E))), 2);
}

double phi_I(double II) {
    return ((mc.a_I * II) - mc.b_I) / (1 - exp(-1 * mc.d_I * ((mc.a_I * II) - mc.b_I)));
}

double dphi_I(double II) {
    return (
        mc.a_I * (1 - exp(-1 * mc.d_I * ((mc.a_I * II) - mc.b_I)))
        - mc.a_I * mc.d_I * exp(-1 * mc.d_I * ((mc.a_I * II) - mc.b_I)) * ((mc.a_I * II) - mc.b_I)
        ) / pow((1 - exp(-1 * mc.d_I * ((mc.a_I * II) - mc.b_I))), 2);
}

/* Eq.10 in Demirtas which would be used in `gsl_fsolve`
 to find the steady-state inhibitory synaptic gating variable
 and the suitable w_IE weight according to the FIC algorithm */

struct inh_curr_params {
    double _I0_I, _w_EI, _S_E_ss, _w_II, gamma_I_s, tau_I_s;
};

double _inh_curr_fixed_pts(double x, void * params) {
    struct inh_curr_params *p = (struct inh_curr_params *) params;
    return p->_I0_I + p->_w_EI * p->_S_E_ss -
            p->_w_II * p->gamma_I_s * p->tau_I_s * phi_I(x) - x;
}

// making most variables global to avoid re-allocation when calling from
// CMAES or grid search
// They are only allocated in the first call by checking if they are NULL
gsl_matrix *_K_EE, *_K_EI, *_w_EE_matrix, *sc;
int nc, curr_node_FIC;
gsl_vector *_w_II, *_w_IE, *_w_EI, *_w_EE, *_I0, *_I_ext,
            *_I0_E, *_I0_I, *_I_E_ss, *_I_I_ss, *_S_E_ss, *_S_I_ss,
            *_r_I_ss, *_K_EE_row, *w_IE_out;


void analytical_fic(
        gsl_matrix * sc, double G, double w_EE, double w_EI, double w_IE, 
        gsl_vector * het_map, double exc_scale, double inh_scale,
        gsl_vector * w_IE_out, bool * _unstable) {
    nc = sc->size1;

    // specify regional parameters
    repeat(&_w_II, mc.w_II, nc);
    repeat(&_w_IE, w_IE, nc);
    repeat(&_w_EI, w_EI, nc);
    repeat(&_w_EE, w_EE, nc);

    // apply heterogeneity if indicated
    double _curr_region_scale;
    if (exc_scale != 0.0) {
        for (int i=0; i < nc; i++) {
            _curr_region_scale = (1 + (gsl_vector_get(het_map, i) * exc_scale));
            if (_curr_region_scale <= 0) {
                printf("Heterogeneity scale lead to negative/zero in some regions\n");
                *_unstable = true;
                return;
            }
            gsl_vector_set(_w_EE, i, w_EE * _curr_region_scale);
        }
    }
    if (inh_scale != 0.0) {
        for (int i=0; i < nc; i++) {
            _curr_region_scale = (1 + (gsl_vector_get(het_map, i) * inh_scale));
            if (_curr_region_scale <= 0) {
                printf("Heterogeneity scale lead to negative/zero in some regions\n");
                *_unstable = true;
                return;
            }
            gsl_vector_set(_w_EI, i, w_EI * _curr_region_scale);
        }
    }

    repeat(&_I0, mc.I_0, nc);
    repeat(&_I_ext, mc.I_ext, nc);

    // Baseline input currents
    vector_scale(&_I0_E, _I0, mc.w_E);
    vector_scale(&_I0_I, _I0, mc.w_I);

    // Steady state values for isolated node
    repeat(&_I_E_ss, mc.I_E_ss, nc);
    repeat(&_I_I_ss, mc.I_I_ss, nc);
    repeat(&_S_E_ss, mc.S_E_ss, nc);
    repeat(&_S_I_ss, mc.S_I_ss, nc);
    // repeat(&_r_E_ss, r_E_ss, nc);
    repeat(&_r_I_ss, mc.r_I_ss, nc);
    
    // set K_EE and K_EI
    if (_K_EE==NULL) {
        _K_EE = gsl_matrix_alloc(nc, nc);
    }
    gsl_matrix_memcpy(_K_EE, sc);
    gsl_matrix_scale(_K_EE, G * mc.J_NMDA);
    make_diag(&_w_EE_matrix, _w_EE);
    gsl_matrix_add(_K_EE, _w_EE_matrix);
    // gsl_matrix_free(_w_EE_matrix);
    make_diag(&_K_EI, _w_EI);


    // analytic FIC
    gsl_function F;
    double curr_I_I, curr_r_I, _K_EE_dot_S_E_ss, J;
    if (_K_EE_row==NULL) {
        _K_EE_row = gsl_vector_alloc(nc);
    }
    for (int curr_node_FIC=0; curr_node_FIC<nc; curr_node_FIC++) {
        struct inh_curr_params params = {
            _I0_I->data[curr_node_FIC], _w_EI->data[curr_node_FIC],
            _S_E_ss->data[curr_node_FIC], _w_II->data[curr_node_FIC],
            mc.gamma_I_s, mc.tau_I_s
        };
        F.function = &_inh_curr_fixed_pts;
        F.params = &params;
        curr_I_I = gsl_fsolve(F, 0.0, 2.0);
        if (curr_I_I == -1) {
            *_unstable = true;
            return;
        }
        // gsl_vector_set(_I_I, curr_node_FIC, curr_I_I);
        gsl_vector_set(_I_I_ss, curr_node_FIC, curr_I_I);
        curr_r_I = phi_I(curr_I_I);
        // gsl_vector_set(_r_I, curr_node_FIC, curr_r_I);
        gsl_vector_set(_r_I_ss, curr_node_FIC, curr_r_I);
        gsl_vector_set(_S_I_ss, curr_node_FIC, 
                        curr_r_I * mc.tau_I_s * mc.gamma_I_s);
        gsl_matrix_get_row(_K_EE_row, _K_EE, curr_node_FIC);
        gsl_blas_ddot(_K_EE_row, _S_E_ss, &_K_EE_dot_S_E_ss);
        J = (-1 / _S_I_ss->data[curr_node_FIC]) *
                    (_I_E_ss->data[curr_node_FIC] - 
                    _I_ext->data[curr_node_FIC] - 
                    _I0_E->data[curr_node_FIC] -
                    _K_EE_dot_S_E_ss);
        if (J < 0) {
            *_unstable = true;
            return;
        }
        gsl_vector_set(_w_IE, curr_node_FIC, J);
    }    

    gsl_vector_memcpy(w_IE_out, _w_IE);
}

void analytical_fic_het(
        gsl_matrix * sc, double G, double * w_EE, double * w_EI,
        gsl_vector * w_IE_out, bool * _unstable) {
    // TODO: integrate anyltical_fic and analytical_fic_het into one function
    nc = sc->size1;

    // specify regional parameters
    repeat(&_w_II, mc.w_II, nc);
    repeat(&_w_IE, 0, nc);
    copy_array_to_vector(&_w_EI, w_EI, nc);
    copy_array_to_vector(&_w_EE, w_EE, nc);

    repeat(&_I0, mc.I_0, nc);
    repeat(&_I_ext, mc.I_ext, nc);

    // Baseline input currents
    vector_scale(&_I0_E, _I0, mc.w_E);
    vector_scale(&_I0_I, _I0, mc.w_I);

    // Steady state values for isolated node
    repeat(&_I_E_ss, mc.I_E_ss, nc);
    repeat(&_I_I_ss, mc.I_I_ss, nc);
    repeat(&_S_E_ss, mc.S_E_ss, nc);
    repeat(&_S_I_ss, mc.S_I_ss, nc);
    // repeat(&_r_E_ss, r_E_ss, nc);
    repeat(&_r_I_ss, mc.r_I_ss, nc);
    
    // set K_EE and K_EI
    if (_K_EE==NULL) {
        _K_EE = gsl_matrix_alloc(nc, nc);
    }
    gsl_matrix_memcpy(_K_EE, sc);
    gsl_matrix_scale(_K_EE, G * mc.J_NMDA);
    make_diag(&_w_EE_matrix, _w_EE);
    gsl_matrix_add(_K_EE, _w_EE_matrix);
    // gsl_matrix_free(_w_EE_matrix);
    make_diag(&_K_EI, _w_EI);


    // analytic FIC
    gsl_function F;
    double curr_I_I, curr_r_I, _K_EE_dot_S_E_ss, J;
    if (_K_EE_row==NULL) {
        _K_EE_row = gsl_vector_alloc(nc);
    }
    for (int curr_node_FIC=0; curr_node_FIC<nc; curr_node_FIC++) {
        struct inh_curr_params params = {
            _I0_I->data[curr_node_FIC], _w_EI->data[curr_node_FIC],
            _S_E_ss->data[curr_node_FIC], _w_II->data[curr_node_FIC],
            mc.gamma_I_s, mc.tau_I_s
        };
        F.function = &_inh_curr_fixed_pts;
        F.params = &params;
        curr_I_I = gsl_fsolve(F, 0.0, 2.0);
        if (curr_I_I == -1) {
            *_unstable = true;
            return;
        }
        // gsl_vector_set(_I_I, curr_node_FIC, curr_I_I);
        gsl_vector_set(_I_I_ss, curr_node_FIC, curr_I_I);
        curr_r_I = phi_I(curr_I_I);
        // gsl_vector_set(_r_I, curr_node_FIC, curr_r_I);
        gsl_vector_set(_r_I_ss, curr_node_FIC, curr_r_I);
        gsl_vector_set(_S_I_ss, curr_node_FIC, 
                        curr_r_I * mc.tau_I_s * mc.gamma_I_s);
        gsl_matrix_get_row(_K_EE_row, _K_EE, curr_node_FIC);
        gsl_blas_ddot(_K_EE_row, _S_E_ss, &_K_EE_dot_S_E_ss);
        J = (-1 / _S_I_ss->data[curr_node_FIC]) *
                    (_I_E_ss->data[curr_node_FIC] - 
                    _I_ext->data[curr_node_FIC] - 
                    _I0_E->data[curr_node_FIC] -
                    _K_EE_dot_S_E_ss);
        if (J < 0) {
            *_unstable = true;
            return;
        }
        gsl_vector_set(_w_IE, curr_node_FIC, J);
    }    

    gsl_vector_memcpy(w_IE_out, _w_IE);
}

gsl_vector * run_fic(gsl_matrix * sc, int n_regions, double G, double wee, 
        double wei, gsl_vector * het_map, double exc_scale, double inh_scale) {
    // only does FIC
    if (w_IE_out == NULL) {
        w_IE_out = gsl_vector_alloc(n_regions);
    }
    bool _unstable = false;
    analytical_fic(
        sc, G, wee, wei, 0, het_map, exc_scale, 
        inh_scale, w_IE_out, &_unstable);
    return w_IE_out;
}

void run_fic(float * J_i, std::string sc_path, int n_regions, double G, double wee, 
        double wei, gsl_vector * het_map, double exc_scale, double inh_scale) {
    // this is called by tvb.cpp
    // read sc
    if (sc == NULL) {
        sc = gsl_matrix_alloc(n_regions, n_regions);
    }
    FILE * sc_file = fopen(sc_path.c_str(), "r");
    if (sc_file==NULL) {
        printf( "\nERROR: Could not open SC matrix. Terminating... \n\n");
        exit(1);
    }
    gsl_matrix_fscanf(sc_file, sc);
    fclose(sc_file);  
    // do FIC
    gsl_vector * w_IE = run_fic(sc, n_regions, G, wee, wei, het_map, exc_scale, inh_scale);
    // save in the float array
    for (int i=0; i<n_regions; i++) {
        J_i[i] = (float)gsl_vector_get(w_IE, i);
    }
}
#endif