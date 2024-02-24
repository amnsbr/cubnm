/*
Analytical Feedback Inhibition Control (FIC)
Calculates wIE needed in each node to maintain excitatory
firing rate of ~3 Hz.

Translated from Python code in https://github.com/murraylab/hbnm

Author: Amin Saberi, Feb 2023
*/
#include "cubnm/models/rww_fic.hpp"

// helper functions
void repeat(gsl_vector ** dest, double a, int size) {
    *dest = gsl_vector_alloc(size);
    gsl_vector_set_all(*dest, a);
}

void copy_array_to_vector(gsl_vector ** dest, double * src, int size) {
    *dest = gsl_vector_alloc(size);
    for (int i=0; i<size; i++) {
        gsl_vector_set(*dest, i, src[i]);
    }
}

void vector_scale(gsl_vector ** dest, gsl_vector * src, double a) {
    *dest = gsl_vector_alloc(src->size);
    gsl_vector_memcpy(*dest, src);
    gsl_vector_scale(*dest, a);
}

void mul_eye(gsl_matrix ** dest, double a, int size) {
    *dest = gsl_matrix_alloc(size, size);
    gsl_matrix_set_identity(*dest);
    gsl_matrix_scale(*dest, a);
}

void make_diag(gsl_matrix ** dest, gsl_vector * v) {
    int size = v->size;
    *dest = gsl_matrix_calloc(size, size);
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
        std::cerr << "Root solver did not converge" << std::endl;
        return -1;
    }
    return root;
}

// transfer function and derivatives for excitatory and inhibitory populations
double phi_E(double IE) {
    return ((rWWModel::mc.a_E * IE) - rWWModel::mc.b_E) / (1 - exp(-rWWModel::mc.d_E * ((rWWModel::mc.a_E * IE) - rWWModel::mc.b_E)));
}

double dphi_E(double IE) {
    return (
        (rWWModel::mc.a_E * (1 - exp(-1 * rWWModel::mc.d_E * ((rWWModel::mc.a_E * IE) - rWWModel::mc.b_E))))
        - (rWWModel::mc.a_E * rWWModel::mc.d_E * exp(-1 * rWWModel::mc.d_E * ((rWWModel::mc.a_E * IE) - rWWModel::mc.b_E)) * ((rWWModel::mc.a_E * IE) - rWWModel::mc.b_E))
        ) / pow((1 - exp(-1 * rWWModel::mc.d_E * ((rWWModel::mc.a_E * IE) - rWWModel::mc.b_E))), 2);
}

double phi_I(double II) {
    return ((rWWModel::mc.a_I * II) - rWWModel::mc.b_I) / (1 - exp(-1 * rWWModel::mc.d_I * ((rWWModel::mc.a_I * II) - rWWModel::mc.b_I)));
}

double dphi_I(double II) {
    return (
        rWWModel::mc.a_I * (1 - exp(-1 * rWWModel::mc.d_I * ((rWWModel::mc.a_I * II) - rWWModel::mc.b_I)))
        - rWWModel::mc.a_I * rWWModel::mc.d_I * exp(-1 * rWWModel::mc.d_I * ((rWWModel::mc.a_I * II) - rWWModel::mc.b_I)) * ((rWWModel::mc.a_I * II) - rWWModel::mc.b_I)
        ) / pow((1 - exp(-1 * rWWModel::mc.d_I * ((rWWModel::mc.a_I * II) - rWWModel::mc.b_I))), 2);
}

/* Eq.10 in Demirtas which would be used in `gsl_fsolve`
 to find the steady-state inhibitory synaptic gating variable
 and the suitable w_IE weight according to the FIC algorithm */

double _inh_curr_fixed_pts(double x, void * params) {
    struct inh_curr_params *p = (struct inh_curr_params *) params;
    return p->_I0_I + p->_w_EI * p->_S_E_ss -
            p->_w_II * p->gamma_I_s * p->tau_I_s * phi_I(x) - x;
}


void analytical_fic_het(
        gsl_matrix * sc, double G, double * w_EE, double * w_EI,
        gsl_vector * w_IE_out, bool * _unstable) {
    int nodes = sc->size1;

    gsl_matrix *_K_EE, *_K_EI, *_w_EE_matrix;
    gsl_vector *_w_II, *_w_IE, *_w_EI, *_w_EE, *_I0, *_I_ext,
                *_I0_E, *_I0_I, *_I_E_ss, *_I_I_ss, *_S_E_ss, *_S_I_ss,
                *_r_I_ss, *_K_EE_row;

    // specify regional parameters
    repeat(&_w_II, rWWModel::mc.w_II, nodes);
    repeat(&_w_IE, 0, nodes);
    copy_array_to_vector(&_w_EI, w_EI, nodes);
    copy_array_to_vector(&_w_EE, w_EE, nodes);

    repeat(&_I0, rWWModel::mc.I_0, nodes);
    repeat(&_I_ext, rWWModel::mc.I_ext, nodes);

    // Baseline input currents
    vector_scale(&_I0_E, _I0, rWWModel::mc.w_E);
    vector_scale(&_I0_I, _I0, rWWModel::mc.w_I);

    // Steady state values for isolated node
    repeat(&_I_E_ss, rWWModel::mc.I_E_ss, nodes);
    repeat(&_I_I_ss, rWWModel::mc.I_I_ss, nodes);
    repeat(&_S_E_ss, rWWModel::mc.S_E_ss, nodes);
    repeat(&_S_I_ss, rWWModel::mc.S_I_ss, nodes);
    // repeat(&_r_E_ss, r_E_ss, nodes);
    repeat(&_r_I_ss, rWWModel::mc.r_I_ss, nodes);
    
    // set K_EE and K_EI
    _K_EE = gsl_matrix_alloc(nodes, nodes);

    gsl_matrix_memcpy(_K_EE, sc);
    gsl_matrix_scale(_K_EE, G * rWWModel::mc.J_NMDA);
    make_diag(&_w_EE_matrix, _w_EE);
    gsl_matrix_add(_K_EE, _w_EE_matrix);
    // gsl_matrix_free(_w_EE_matrix);
    make_diag(&_K_EI, _w_EI);


    // analytic FIC
    gsl_function F;
    double curr_I_I, curr_r_I, _K_EE_dot_S_E_ss, w_IE;

    _K_EE_row = gsl_vector_alloc(nodes);

    for (int j=0; j<nodes; j++) {
        struct inh_curr_params params = {
            _I0_I->data[j], _w_EI->data[j],
            _S_E_ss->data[j], _w_II->data[j],
            rWWModel::mc.gamma_I_s, rWWModel::mc.tau_I_s
        };
        F.function = &_inh_curr_fixed_pts;
        F.params = &params;
        curr_I_I = gsl_fsolve(F, 0.0, 2.0);
        if (curr_I_I == -1) {
            *_unstable = true;
            return;
        }
        // gsl_vector_set(_I_I, j, curr_I_I);
        gsl_vector_set(_I_I_ss, j, curr_I_I);
        curr_r_I = phi_I(curr_I_I);
        // gsl_vector_set(_r_I, j, curr_r_I);
        gsl_vector_set(_r_I_ss, j, curr_r_I);
        gsl_vector_set(_S_I_ss, j, 
                        curr_r_I * rWWModel::mc.tau_I_s * rWWModel::mc.gamma_I_s);
        gsl_matrix_get_row(_K_EE_row, _K_EE, j);
        gsl_blas_ddot(_K_EE_row, _S_E_ss, &_K_EE_dot_S_E_ss);
        w_IE = (-1 / _S_I_ss->data[j]) *
                    (_I_E_ss->data[j] - 
                    _I_ext->data[j] - 
                    _I0_E->data[j] -
                    _K_EE_dot_S_E_ss);
        if (w_IE < 0) {
            *_unstable = true;
            return;
        }
        gsl_vector_set(_w_IE, j, w_IE);
    }    

    gsl_vector_memcpy(w_IE_out, _w_IE);

    gsl_matrix_free(_K_EE); gsl_matrix_free(_K_EI); gsl_matrix_free(_w_EE_matrix);
    gsl_vector_free(_w_II); gsl_vector_free(_w_IE); gsl_vector_free(_w_EI); gsl_vector_free(_w_EE);
    gsl_vector_free(_I0); gsl_vector_free(_I_ext);
    gsl_vector_free(_I0_E); gsl_vector_free(_I0_I); gsl_vector_free(_I_E_ss); gsl_vector_free(_I_I_ss);
    gsl_vector_free(_S_E_ss); gsl_vector_free(_S_I_ss); gsl_vector_free(_r_I_ss); gsl_vector_free(_K_EE_row);
}