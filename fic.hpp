#ifndef FIC_HPP
#define FIC_HPP

extern void repeat(gsl_vector ** dest, double a, int size);
extern void copy_array_to_vector(gsl_vector ** dest, double * src, int size);
extern void vector_scale(gsl_vector ** dest, gsl_vector * src, double a);
extern void mul_eye(gsl_matrix ** dest, double a, int size);
extern void make_diag(gsl_matrix ** dest, gsl_vector * v);
extern double gsl_fsolve(gsl_function F, double x_lo, double x_hi);

extern double phi_E(double IE);
extern double dphi_E(double IE);
extern double phi_I(double II);
extern double dphi_I(double II);

extern double _inh_curr_fixed_pts(double x, void * params);
extern gsl_matrix *_K_EE, *_K_EI, *_w_EE_matrix, *sc;
extern int nc, curr_node_FIC;
extern gsl_vector *_w_II, *_w_IE, *_w_EI, *_w_EE, *_I0, *_I_ext,
            *_I0_E, *_I0_I, *_I_E_ss, *_I_I_ss, *_S_E_ss, *_S_I_ss,
            *_r_I_ss, *_K_EE_row, *w_IE_out;


extern void analytical_fic(
        gsl_matrix * sc, double G, double w_EE, double w_EI, double w_IE, 
        gsl_vector * het_map, double exc_scale, double inh_scale,
        gsl_vector * w_IE_out, bool * _unstable);
extern void analytical_fic_het(
        gsl_matrix * sc, double G, double * w_EE, double * w_EI,
        gsl_vector * w_IE_out, bool * _unstable);
extern gsl_vector * run_fic(gsl_matrix * sc, int n_regions, double G, double wee, 
        double wei, gsl_vector * het_map, double exc_scale, double inh_scale);
extern void run_fic(float * J_i, std::string sc_path, int n_regions, double G, double wee, 
        double wei, gsl_vector * het_map, double exc_scale, double inh_scale);
#endif