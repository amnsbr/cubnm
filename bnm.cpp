/*
Reduced Wong-Wang model (Deco 2014) simulation on CPU

Parts of this code are based on https://github.com/BrainModes/The-Hybrid-Virtual-Brain, 
https://github.com/murraylab/hbnm & https://github.com/decolab/cb-neuromod

Author: Amin Saberi, Feb 2023
*/

#include <stdio.h>
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
#include <chrono>
#include "fic.cpp"
#include "constants.h"
#include "check_fit.cpp"

double bnm(int nodes, u_real * SC, gsl_matrix * SC_gsl,
        u_real G, u_real * w_EE,  u_real * w_EI, u_real * w_IE, 
        u_real * fic_penalty, bool calculate_fic_penalty,
        int time_steps, int BOLD_TR, int rand_seed, int _max_fic_trials,
        gsl_vector * emp_FC_tril, gsl_vector * emp_FCD_tril, bool sim_only, 
        bool no_fcd, int step_TR, int window_TR, 
        std::string out_prefix, bool verbose, bool save_output, bool extended_output) {

    time_t start = time(NULL);

    /* random number generator */
    std::mt19937 rand_gen(rand_seed);
    // generating random numbers as floats regardless of USE_FLOATS
    // for better performance and consistency of the noise for the same
    // seed regardless of USE_FLOATS
    std::normal_distribution<float> normal_dist(0, 1);

    u_real TR        = (u_real)BOLD_TR / 1000; // (s) TR of fMRI data
    int   output_ts = (time_steps / (TR / model_dt))+1; // Length of BOLD time-series written to HDD    
    /*
     Allocate and Initialize arrays
     */
    size_t bold_size = nodes * output_ts;
    u_real *BOLD_ex                  = (u_real *)malloc((bold_size * sizeof(u_real)));
    u_real *S_i_E                    = (u_real *)malloc(nodes * sizeof(u_real));
    u_real *S_i_I                    = (u_real *)malloc(nodes * sizeof(u_real));
    u_real *_w_EE                    = (u_real *)malloc(nodes * sizeof(u_real));  // regional w_EE
    u_real *_w_EI                    = (u_real *)malloc(nodes * sizeof(u_real));  // regional w_EI
    u_real *_w_IE                    = (u_real *)malloc(nodes * sizeof(u_real));  // regional w_IE
    u_real *bw_x_ex                  = (u_real *)malloc(nodes * sizeof(u_real));  // State-variable 1 of BW-model (exc. pop.)
    u_real *bw_f_ex                  = (u_real *)malloc(nodes * sizeof(u_real));  // State-variable 2 of BW-model (exc. pop.)
    u_real *bw_nu_ex                 = (u_real *)malloc(nodes * sizeof(u_real));  // State-variable 3 of BW-model (exc. pop.)
    u_real *bw_q_ex                  = (u_real *)malloc(nodes * sizeof(u_real));  // State-variable 4 of BW-model (exc. pop.)
    u_real *r_i_E                    = (u_real *)malloc(nodes * sizeof(u_real));
    u_real *r_i_I                    = (u_real *)malloc(nodes * sizeof(u_real));
    u_real *I_i_E                    = (u_real *)malloc(nodes * sizeof(u_real));
    u_real *I_i_I                    = (u_real *)malloc(nodes * sizeof(u_real));

    if (S_i_E == NULL || S_i_I == NULL || _w_EE == NULL || _w_EI == NULL || _w_IE == NULL || 
        bw_x_ex == NULL || bw_f_ex == NULL || bw_nu_ex == NULL || bw_q_ex == NULL ||
        r_i_E == NULL || r_i_I == NULL || I_i_E == NULL || I_i_I == NULL) {
        printf("Error: Failed to allocate memory\n");
        exit(1);
    }

    for (int j = 0; j < nodes; j++) {
        S_i_E[j] = 0.001;
        S_i_I[j] = 0.001;
        _w_EE[j] = w_EE[j]; // this is redundant for wee and wei but needed for wie (in FIC+ case)
        _w_EI[j] = w_EI[j];
        _w_IE[j] = w_IE[j];
        bw_x_ex[j] = 0.0;
        bw_f_ex[j] = 1.0;
        bw_nu_ex[j] = 1.0;
        bw_q_ex[j] = 1.0;
        r_i_E[j] = 0;
        r_i_I[j] = 0;
        I_i_E[j] = 0;
        I_i_I[j] = 0;
    }

    // do FIC if wie (of the first region) is 0
    bool do_fic = false;
    double *_dw_EE, *_dw_EI;
    gsl_vector * _w_IE_vector;
    if (w_IE[0]==0) {
        if (verbose) std::cout << "Running FIC\n";
        do_fic = true;
        extended_output = true;
        _w_IE_vector = gsl_vector_alloc(nodes);
        // make a double copy of the wEE and wEI arrays (if needed in USE_FLOATS)
        #ifdef USE_FLOATS
            _dw_EE = (double *)malloc(nodes * sizeof(double));
            _dw_EI = (double *)malloc(nodes * sizeof(double));
            for (int j = 0; j < nodes; j++) {
                _dw_EE[j] = (double)_w_EE[j];
                _dw_EI[j] = (double)_w_EI[j];
            }
        #else
            #define _dw_EE _w_EE
            #define _dw_EI _w_EI
        #endif
        bool _unstable; // TODO: remove it as it is not used
        #pragma omp critical
        analytical_fic_het(
            SC_gsl, G, _dw_EE, _dw_EI,
            _w_IE_vector, &_unstable);
        // analytical fic is run as a critical step to avoid racing conditions
        // between the OMP threads
        // (this is a more cost-efficient solution than trying to
        // fix it with memory management)
        // copy the vector back to the _w_IE array
        for (int j = 0; j < nodes; j++) {
            _w_IE[j] = (u_real)(gsl_vector_get(_w_IE_vector, j));
        }
    }

    u_real *S_E_ex, *S_I_ex, *r_E_ex, *r_I_ex, *I_E_ex, *I_I_ex;
    if (extended_output) {
        S_E_ex                  = (u_real *)malloc((bold_size * sizeof(u_real)));
        S_I_ex                  = (u_real *)malloc((bold_size * sizeof(u_real)));
        r_E_ex                  = (u_real *)malloc((bold_size * sizeof(u_real)));
        r_I_ex                  = (u_real *)malloc((bold_size * sizeof(u_real)));
        I_E_ex                  = (u_real *)malloc((bold_size * sizeof(u_real)));
        I_I_ex                  = (u_real *)malloc((bold_size * sizeof(u_real)));
        if (S_E_ex == NULL || S_I_ex == NULL || r_E_ex == NULL || r_I_ex == NULL ||
            I_E_ex == NULL || I_I_ex == NULL) {
                printf("Error: Failed to allocate memory\n");
                exit(1);
        }
    }

    /*
    Integration
    */
    bool adjust_fic = (do_fic && numerical_fic);
    u_real *S_i_1_E = (u_real *)malloc(nodes * sizeof(u_real));
    u_real *mean_I_E = (u_real *)malloc(nodes * sizeof(u_real));
    u_real *delta = (u_real *)malloc(nodes * sizeof(u_real));
    if (S_i_1_E == NULL || mean_I_E == NULL || delta == NULL) {
        printf("Error: Failed to allocate memory\n");
        exit(1);
    }
    for (int j = 0; j < nodes; j++) {
        S_i_1_E[j] = S_i_E[j];
        mean_I_E[j] = 0;
        delta[j] = init_delta;
    }
    int fic_trial = 0;
    bool fic_failed = false;
    bool _adjust_fic = adjust_fic;
    u_real tmp_globalinput, tmp_rand_E, tmp_rand_I, 
        dSdt_E, dSdt_I, tmp_f, tmp_exp_E, tmp_exp_I,
        I_E_ba_diff;
    int j, k;
    int BOLD_len_i = 0;
    int ts_bold = 0;
    int bold_idx = 0;
    while (ts_bold <= time_steps) {
        if (verbose) printf("%.1f %% \r", ((u_real)ts_bold / (u_real)time_steps) * 100.0f );
        for (int int_i = 0; int_i < 10; int_i++) {
            // create a copy of S_E in the previous time point
            for (j=0; j<nodes; j++) {
                S_i_1_E[j] = S_i_E[j];
            }
            for (j=0; j<nodes; j++) {
                // calculate global input
                tmp_globalinput = 0;
                for (k=0; k<nodes; k++) {
                    tmp_globalinput += SC[j*nodes+k] * S_i_1_E[k];
                }
                // equations
                I_i_E[j] = w_E__I_0 + _w_EE[j] * S_i_E[j] + tmp_globalinput * G * J_NMDA - _w_IE[j]*S_i_I[j];
                I_i_I[j] = w_I__I_0 + _w_EI[j] * S_i_E[j] - S_i_I[j];
                tmp_exp_E = EXP(-d_E * (a_E * I_i_E[j] - b_E));
                tmp_exp_I = EXP(-d_I * (a_I * I_i_I[j] - b_I));
                r_i_E[j] = (a_E * I_i_E[j] - b_E) / (1 - tmp_exp_E);
                r_i_I[j] = (a_I * I_i_I[j] - b_I) / (1 - tmp_exp_I);
                tmp_rand_E = normal_dist(rand_gen);
                tmp_rand_I = normal_dist(rand_gen);
                dSdt_E = tmp_rand_E * sigma_model * sqrt_dt + dt * ((1 - S_i_E[j]) * gamma_E * r_i_E[j] - (S_i_E[j] / tau_E));
                dSdt_I = tmp_rand_I * sigma_model * sqrt_dt + dt * (gamma_I * r_i_I[j] - (S_i_I[j] / tau_I));
                S_i_E[j] += dSdt_E;
                S_i_I[j] += dSdt_I;
                // clip S to 0-1
                #ifdef USE_FLOATS
                S_i_E[j] = std::max(0.0f, std::min(S_i_E[j], 1.0f));
                S_i_I[j] = std::max(0.0f, std::min(S_i_I[j], 1.0f));
                #else
                S_i_E[j] = std::max(0.0, std::min(S_i_E[j], 1.0));
                S_i_I[j] = std::max(0.0, std::min(S_i_I[j], 1.0));
                #endif
            }
            // printf("\n");
        }

        /*
        Compute BOLD for that time-step (subsampled to 1 ms)
        save BOLD in addition to S_E, S_I, and r_E, r_I if requested
        */
        for (j=0; j<nodes; j++) {
            bw_x_ex[j]  = bw_x_ex[j]  +  model_dt * (S_i_E[j] - kappa * bw_x_ex[j] - y * (bw_f_ex[j] - 1.0));
            tmp_f       = bw_f_ex[j]  +  model_dt * bw_x_ex[j];
            bw_nu_ex[j] = bw_nu_ex[j] +  model_dt * itau * (bw_f_ex[j] - POW(bw_nu_ex[j], ialpha));
            bw_q_ex[j]  = bw_q_ex[j]  +  model_dt * itau * (bw_f_ex[j] * (1.0 - POW(oneminrho,(1.0/bw_f_ex[j]))) / rho  - POW(bw_nu_ex[j],ialpha) * bw_q_ex[j] / bw_nu_ex[j]);
            bw_f_ex[j]  = tmp_f;   
        }
        if (ts_bold % BOLD_TR == 0) {
            for (j = 0; j<nodes; j++) {
                bold_idx = BOLD_len_i*nodes+j;
                BOLD_ex[bold_idx] = 100 / rho * V_0 * (k1 * (1 - bw_q_ex[j]) + k2 * (1 - bw_q_ex[j]/bw_nu_ex[j]) + k3 * (1 - bw_nu_ex[j]));
                if (extended_output) {
                    S_E_ex[bold_idx] = S_i_E[j];
                    S_I_ex[bold_idx] = S_i_I[j];
                    r_E_ex[bold_idx] = r_i_E[j];
                    r_I_ex[bold_idx] = r_i_I[j];
                    I_E_ex[bold_idx] = I_i_E[j];
                    I_I_ex[bold_idx] = I_i_I[j];
                }
            }
            BOLD_len_i++;
        }
        ts_bold++;

        // adjust FIC according to Deco2014
        if (_adjust_fic) {
            if ((ts_bold >= I_SAMPLING_START) && (ts_bold <= I_SAMPLING_END)) {
                for (j = 0; j<nodes; j++) {
                    mean_I_E[j] += I_i_E[j];
                }
            }
            if (ts_bold == I_SAMPLING_END) {
                bool needs_fic_adjustment = false;
                if (verbose) printf("FIC adjustment trial %d\nnode\tIE_ba_diff\tdelta\tnew_w_IE\n", fic_trial);
                for (j = 0; j<nodes; j++) {
                    mean_I_E[j] /= I_SAMPLING_DURATION;
                    I_E_ba_diff = mean_I_E[j] - b_a_ratio_E;
                    if (abs(I_E_ba_diff + 0.026) > 0.005) {
                        needs_fic_adjustment = true;
                        if (fic_trial < _max_fic_trials) { // only do the adjustment if max trials is not exceeded
                            // up- or downregulate inhibition
                            if ((I_E_ba_diff) < -0.026) {
                                _w_IE[j] -= delta[j];
                                if (verbose) printf("%d\t%f\t-%f\t%f\n", j, I_E_ba_diff, delta[j], _w_IE[j]);
                                delta[j] -= 0.001;
                                delta[j] = fmaxf(delta[j], 0.001);
                            } else {
                                _w_IE[j] += delta[j];
                                if (verbose) printf("%d\t%f\t+%f\t%f\n", j, I_E_ba_diff, delta[j], _w_IE[j]);
                            }
                        }
                    }
                }
                if (needs_fic_adjustment) {
                    if (fic_trial < _max_fic_trials) {
                        // reset states
                        for (j = 0; j < nodes; j++) {
                            S_i_E[j] = 0.001;
                            S_i_I[j] = 0.001;
                            bw_x_ex[j] = 0.0;
                            bw_f_ex[j] = 1.0;
                            bw_nu_ex[j] = 1.0;
                            bw_q_ex[j] = 1.0;
                            mean_I_E[j] = 0;
                        }
                        // reset time and random sequence
                        ts_bold = 0;
                        BOLD_len_i = 0;
                        rand_gen = std::mt19937(rand_seed);
                        normal_dist = std::normal_distribution<float>(0, 1);
                        fic_trial++;
                    } else {
                        // continue the simulation but
                        // declare FIC failed
                        fic_failed = true;
                        _adjust_fic = false;
                    }
                }
                else {
                    // if no node needs fic adjustment don't run
                    // this block of code any more
                    _adjust_fic = false;
                }
            }
        }
    }

    // write to HDD
    FILE *BOLDout, *w_IE_file, *fic_trials_file, *fic_failed_file, *meanFRout, 
        *S_E_out, *S_I_out, *r_E_out, *r_I_out, *I_E_out, *I_I_out;
    if (save_output) {
        std::string bold_filename = out_prefix + "_bold.txt";
        BOLDout = fopen(bold_filename.c_str(), "w");
        if (do_fic) {
            std::string w_IE_filename = out_prefix + "_w_IE.txt";
            std::string fic_trials_filename = out_prefix + "_fic_ntrials.txt";
            std::string fic_failed_filename = out_prefix + "_fic_failed.txt";
            w_IE_file = fopen(w_IE_filename.c_str(), "w");
            fic_trials_file = fopen(fic_trials_filename.c_str(), "w");
            fic_failed_file = fopen(fic_failed_filename.c_str(), "w");
            for (j = 0; j < nodes; j++) {
                fprintf(w_IE_file, "%f\n", _w_IE[j]);
            }
            fprintf(fic_trials_file, "%d of max %d", fic_trial, _max_fic_trials);
            fprintf(fic_failed_file, "%d", fic_failed);
            fclose(w_IE_file);
            fclose(fic_trials_file);
            fclose(fic_failed_file);
        }
        // std::string fr_filename = out_prefix + "_meanFR.txt";
        // meanFRout = fopen(fr_filename.c_str(), "w");
        if (extended_output) {
            std::string S_E_filename = out_prefix + "_S_E.txt";
            S_E_out = fopen(S_E_filename.c_str(), "w");
            std::string S_I_filename = out_prefix + "_S_I.txt";
            S_I_out = fopen(S_I_filename.c_str(), "w");
            std::string r_E_filename = out_prefix + "_r_E.txt";
            r_E_out = fopen(r_E_filename.c_str(), "w");
            std::string r_I_filename = out_prefix + "_r_I.txt";
            r_I_out = fopen(r_I_filename.c_str(), "w");
            std::string I_E_filename = out_prefix + "_I_E.txt";
            I_E_out = fopen(I_E_filename.c_str(), "w");
            std::string I_I_filename = out_prefix + "_I_I.txt";
            I_I_out = fopen(I_I_filename.c_str(), "w");
        }
    }
    /*
        Print BOLD time series in addition to S_E, S_I, and r_E, r_I if requested
        Also save BOLD to gsl_matrix for using in check_fit
    */
    int n_vols_remove = bold_remove_s * 1000 / BOLD_TR; // // "BNM_BOLD_REMOVE" 30 seconds by default (only removed from gsl_matrix and not the saved txt)
    gsl_matrix * bold;
    if (!(sim_only)) {
        bold = gsl_matrix_alloc(BOLD_len_i-n_vols_remove, nodes);
    }
    u_real *mean_r_E;
    if (calculate_fic_penalty) {
        mean_r_E = (u_real *)malloc(nodes * sizeof(u_real)); // for fic_penalty calculation
        for (int j=0; j<nodes; j++) {
            mean_r_E[j] = 0;
        }
    }

    for (int i=0; i<output_ts; i++) {
        for (j=0; j<nodes; j++) {
            if (save_output) {
                fprintf(BOLDout, "%.7f ",BOLD_ex[i*nodes+j]);
                if (extended_output) {
                    fprintf(S_E_out, "%.7f ",S_E_ex[i*nodes+j]);
                    fprintf(S_I_out, "%.7f ",S_I_ex[i*nodes+j]);
                    fprintf(r_E_out, "%.7f ",r_E_ex[i*nodes+j]);
                    fprintf(r_I_out, "%.7f ",r_I_ex[i*nodes+j]);
                    fprintf(I_E_out, "%.7f ",I_E_ex[i*nodes+j]);
                    fprintf(I_I_out, "%.7f ",I_I_ex[i*nodes+j]);
                }

            }
            if ((i >= n_vols_remove) & !(sim_only)) {
                gsl_matrix_set(bold, i-n_vols_remove, j, BOLD_ex[i*nodes+j]);
                if (calculate_fic_penalty) {
                    mean_r_E[j] += r_E_ex[i*nodes+j];
                }
            }
        }
        if (save_output) {
            fprintf(BOLDout, "\n");
            if (extended_output) {
                fprintf(S_E_out, "\n");
                fprintf(S_I_out, "\n");
                fprintf(r_E_out, "\n");
                fprintf(r_I_out, "\n");
                fprintf(I_E_out, "\n");
                fprintf(I_I_out, "\n");
            }
        }
    }
    if (save_output) {
        fclose(BOLDout);
        if (extended_output) {
            fclose(S_E_out);
            fclose(S_I_out);
            fclose(r_E_out);
            fclose(r_I_out);
            fclose(I_E_out);
            fclose(I_I_out);
        }
    }
    // calculate FIC penalty
    if (calculate_fic_penalty) {
        *fic_penalty = 0;
        u_real diff_r_E;
        for (int j=0; j<nodes; j++) {
            mean_r_E[j] /= (BOLD_len_i - n_vols_remove);
            diff_r_E = abs(mean_r_E[j] - 3);
            if (diff_r_E > 1) {
                *fic_penalty += 1 - (EXP(-0.05 * diff_r_E));
            }
        }
        *fic_penalty *= fic_penalty_scale; // scale by BNM_CMAES_FIC_PENALTY_SCALE 
        *fic_penalty /= nodes; // average across nodes
        if ((fic_reject_failed) & (fic_failed)) {
            *fic_penalty += 1;
        }
    }

    // Free memory
    free(delta); free(mean_I_E); free(S_i_1_E);
    if (extended_output) {
        free(I_I_ex); free(I_E_ex); free(r_I_ex);  
        free(r_E_ex); free(S_I_ex); free(S_E_ex); 
    }
    free(I_i_I); free(I_i_E); free(r_i_I); free(r_i_E);
    free(bw_q_ex); free(bw_nu_ex); free(bw_f_ex); free(bw_x_ex); 
    free(_w_IE); free(_w_EI); free(_w_EE); 
    free(S_i_I); free(S_i_E); free(BOLD_ex); 
    


    if (verbose) printf("Simulation finished. Execution took %.2f s\n", (float)(time(NULL) - start));

    if (sim_only) {
        return MAX_COST;
    } else {
        double cost = check_fit_bold(bold, step_TR, window_TR, true, emp_FC_tril, emp_FCD_tril, no_fcd, verbose, exc_interhemispheric);
        gsl_matrix_free(bold);
        return cost;
    }
}

void run_simulations(
    double * corr, u_real * fic_penalties,
    u_real * G_list, u_real * w_EE_list, u_real * w_EI_list, u_real * w_IE_list,
    u_real * SC, gsl_matrix * SC_gsl, int nodes, int N_SIMS, bool calculate_fic_penalty,
    int time_steps, int BOLD_TR, int rand_seed, int _max_fic_trials,
    gsl_vector * emp_FC_tril, gsl_vector * emp_FCD_tril, bool sim_only, 
    bool no_fcd, int window_step, int window_size, 
    std::string sims_out_prefix, bool sim_verbose, bool save_output, bool extended_output
)
{
    #pragma omp parallel
	#pragma omp for
    for(int IndPar = 0; IndPar < N_SIMS; IndPar++) {
        // write thread info with the time
        std::chrono::system_clock::time_point now = std::chrono::system_clock::now();
        std::time_t current_time = std::chrono::system_clock::to_time_t(now);
        std::tm* timeinfo = std::localtime(&current_time);
        char time_str[9];
        std::strftime(time_str, sizeof(time_str), "%T", timeinfo);
        ::printf("Thread %d (of %d) is executing particle %d [%s]\n", omp_get_thread_num(), omp_get_num_threads(), IndPar, time_str);
        // run the simulation, calcualte FC and FCD and gof
        // (all in bnm function)
        corr[IndPar] = bnm(nodes, SC, SC_gsl,
            G_list[IndPar], 
            w_EE_list+(IndPar*nodes), // the starting index of the w_EE_list slice of current particle
            w_EI_list+(IndPar*nodes),
            w_IE_list+(IndPar*nodes),
            fic_penalties+IndPar, calculate_fic_penalty,
            time_steps, BOLD_TR, rand_seed, _max_fic_trials,
            emp_FC_tril, emp_FCD_tril, sim_only, no_fcd, window_step, window_size,
            sims_out_prefix, sim_verbose, save_output, extended_output);
    }
}
