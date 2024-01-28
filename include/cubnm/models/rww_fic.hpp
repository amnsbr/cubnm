#ifndef RWW_FIC_HPP
#define RWW_FIC_HPP
void repeat(gsl_vector ** dest, double a, int size);
void copy_array_to_vector(gsl_vector ** dest, double * src, int size);
void vector_scale(gsl_vector ** dest, gsl_vector * src, double a);
void mul_eye(gsl_matrix ** dest, double a, int size);
void make_diag(gsl_matrix ** dest, gsl_vector * v);
double gsl_fsolve(gsl_function F, double x_lo, double x_hi);
double phi_E(double IE);
double dphi_E(double IE);
double phi_I(double II);
double dphi_I(double II);
struct inh_curr_params {
    double _I0_I, _w_EI, _S_E_ss, _w_II, gamma_I_s, tau_I_s;
};
void analytical_fic_het(
        gsl_matrix * sc, double G, double * w_EE, double * w_EI,
        gsl_vector * w_IE_out, bool * _unstable);
#endif