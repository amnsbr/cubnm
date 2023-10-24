/* 
Include functions to calculate goodness of fit of a single CPU-based simulation
which is run standalone or as a particle of CMAES.

Author: Amin Saberi, Feb 2023
*/
#include <iostream>
#include <fstream>
#include <string>
#include <cmath>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_statistics.h>
#include <cassert>
#include "constants.h"
extern "C" {
#include "ks.h"
};


gsl_vector * fc_tril(gsl_matrix * bold, bool exc_interhemispheric) {
    /*
     Given empirical/simulated bold (n_vols x n_regions) returns
     the lower triangle of the FC
    */
    int n_regions = bold->size2;
    int n_vols = bold->size1;
    int n_pairs = ((n_regions) * (n_regions - 1)) / 2;
    int rh_idx;
    if (exc_interhemispheric) {
        assert((bold->size2 % 2) == 0);
        rh_idx = bold->size2 / 2; // assumes symmetric number of parcels and L->R order
        n_pairs -= pow(rh_idx, 2); // exc the middle square
    }
    int i, j;
    double corr;
    gsl_vector * FC_tril = gsl_vector_alloc(n_pairs);
    int curr_idx = 0;
    for (i = 0; i<(bold->size2); i++) {
        for (j = 0; j<(bold->size2); j++) {
            if (i > j) {
                if (exc_interhemispheric) {
                    // skip if each node belongs to a different hemisphere
                    if ((i < rh_idx) ^ (j < rh_idx)) {
                        continue;
                    }
                }
                gsl_vector_view ts_i = gsl_matrix_column(bold, i);
                gsl_vector_view ts_j = gsl_matrix_column(bold, j);
                corr = gsl_stats_correlation(
                    ts_i.vector.data, ts_i.vector.stride,
                    ts_j.vector.data, ts_j.vector.stride,
                    (bold->size1)
                );
                if (std::isnan(corr)) {
                    return NULL;
                }
                gsl_vector_set(FC_tril, curr_idx, corr);
                curr_idx ++;
            }
        }
    }
    return FC_tril;
}

gsl_vector * fcd_tril(gsl_matrix * bold, const int step, const int window_size, bool drop_edges, bool exc_interhemispheric) {
    /*
     Calculates the functional connectivity dynamics matrix (lower triangle)
     given BOLD, step and window size. Note that the actual window size is +1 higher.
     The FCD matrix shows similarity of FC patterns between the windows.
    */
    gsl_vector * FCD_tril;
    int n_vols = bold->size1;
    int n_regions = bold->size2;
    int n_pairs = ((n_regions) * (n_regions - 1)) / 2;
    if (exc_interhemispheric) {
        assert((bold->size2 % 2) == 0);
        int rh_idx = bold->size2 / 2; // assumes symmetric number of parcels and L->R order
        n_pairs -= pow(rh_idx, 2); // exc the middle square
    }
    int first_center, last_center, window_center, window_start, window_end;
    if (drop_edges) {
        first_center = window_size / 2;
        last_center = n_vols - 1 - (window_size / 2);
    } else {
        first_center = 0;
        last_center = n_vols - 1;
    }

    // too lazy to calculate the exact number of windows, but this should be the maximum bound (not sure though)
    gsl_matrix * window_FC_trils = gsl_matrix_alloc(n_pairs, ((n_vols + window_size) / step));
    // gsl_matrix_set_all(window_FC_trils, -1.5);
    // calculate the FC of each window
    int n_windows = 0;
    window_center = first_center;
    while (window_center <= last_center) {
        window_start = window_center - (window_size/2);
        if (window_start < 0)
            window_start = 0;
        window_end = window_center + (window_size/2);
        if (window_end >= n_vols)
            window_end = n_vols-1;
        gsl_matrix_view bold_window =  gsl_matrix_submatrix(bold, window_start, 0, window_end-window_start+1, n_regions);
        gsl_vector * window_FC_tril = fc_tril(&bold_window.matrix, exc_interhemispheric);
        if (window_FC_tril==NULL) {
            return NULL;
        }
        gsl_matrix_set_col(window_FC_trils, n_windows, window_FC_tril);
        window_center += step;
        n_windows ++;
    }
    if (n_windows < 10) {
        std::cout << "Warning: Too few FC windows: " << n_windows << "\n";
    }
    // calculate the FCD matrix (lower triangle)
    int window_i, window_j;
    double corr;
    int n_window_pairs = ((n_windows) * (n_windows - 1)) / 2;
    FCD_tril = gsl_vector_alloc(n_window_pairs);
    int curr_idx = 0;
    for (window_i=0; window_i<n_windows; window_i++) {
        for (window_j=0; window_j<n_windows; window_j++) {
            if (window_i > window_j) {
                gsl_vector_view FC_i = gsl_matrix_column(window_FC_trils, window_i);
                gsl_vector_view FC_j = gsl_matrix_column(window_FC_trils, window_j);
                // gsl_vector_fprintf(stdout, &FC_i.vector, "%f");
                corr = gsl_stats_correlation(
                    FC_i.vector.data, FC_i.vector.stride,
                    FC_j.vector.data, FC_j.vector.stride,
                    n_pairs
                );
                if (std::isnan(corr)) {
                    return NULL;
                }
                gsl_vector_set(FCD_tril, curr_idx, corr);
                curr_idx ++;
            }
        }
    }
    return FCD_tril;
}

double check_fit_bold(gsl_matrix * sim_bold, int FCD_step, int FCD_window_size, bool FCD_drop_edges,
        gsl_vector * emp_FC_tril, gsl_vector * emp_FCD_tril, bool no_fcd, bool verbose, 
        bool exc_interhemispheric) {
    const bool save = false; // TODO: ask user
    // check that size of emp_FC_tril is correct
    int n_pairs = ((sim_bold->size2) * (sim_bold->size2 - 1)) / 2;
    if (exc_interhemispheric) {
        assert((sim_bold->size2 % 2) == 0);
        int rh_idx = sim_bold->size2 / 2; // assumes symmetric number of parcels and L->R order
        n_pairs -= pow(rh_idx, 2); // exc the middle square
    }
    if (emp_FC_tril->size != n_pairs) {
        printf("The size of empirical FC tril (%d) is wrong for %d nodes with exc_interhemispheric = %d.\n", 
            emp_FC_tril->size, sim_bold->size2, exc_interhemispheric);
        exit(1);
    }
    double fc_ks, emp_sim_FC_corr, fcd_ks, fc_diff, cost;
    gsl_vector * sim_FC_tril = fc_tril(sim_bold, exc_interhemispheric);
    if (sim_FC_tril==NULL) {
        if (verbose)
            printf("NaN was found in FC. Aborting the simulation.\n");
        return MAX_COST;
    }
    if (save) {
        // It saves the files in another folder different from "sims"
        FILE * sim_FC_file = fopen("output/sim_FCtril.txt", "w");
        gsl_vector_fprintf(sim_FC_file, sim_FC_tril, "%f");
        fclose(sim_FC_file);
    }
    // Check fit to FC
    emp_sim_FC_corr = gsl_stats_correlation(
        emp_FC_tril->data, emp_FC_tril->stride,
        sim_FC_tril->data, sim_FC_tril->stride,
        emp_FC_tril->size
    );
    if (verbose) std::cout << " FC correlation: " << emp_sim_FC_corr << " ";
    cost = -emp_sim_FC_corr;
    if (use_fc_ks) {
        // Get KS distance of FC
        gsl_vector * emp_FC_tril_copy = gsl_vector_alloc(emp_FC_tril->size);
        gsl_vector_memcpy(emp_FC_tril_copy, emp_FC_tril);
        fc_ks = ks_stat(
            emp_FC_tril_copy->data, 
            sim_FC_tril->data, 
            emp_FC_tril_copy->size, 
            sim_FC_tril->size, 
            NULL);
        cost += fc_ks;
        if (verbose) std::cout << " FC KS: " << fc_ks << " ";
        gsl_vector_free(emp_FC_tril_copy);
    }
    if (use_fc_diff) {
        // get means of simulated and empirical fc
        fc_diff = abs(
            gsl_stats_mean(emp_FC_tril->data, emp_FC_tril->stride, emp_FC_tril->size) -
            gsl_stats_mean(sim_FC_tril->data, sim_FC_tril->stride, sim_FC_tril->size)
        );
        cost += fc_diff + 1;
        if (verbose) std::cout << " FC diff: " << fc_diff << " ";
    }
    gsl_vector_free(sim_FC_tril);
    if (!(no_fcd)) {
        // Calculate simulated FCD (lower triangle)
        gsl_vector * sim_FCD_tril = fcd_tril(sim_bold, FCD_step, FCD_window_size, FCD_drop_edges, exc_interhemispheric);
        if (sim_FCD_tril==NULL) {
            if (verbose)
                printf("NaN was found in FCD. Aborting the simulation.\n");
            return MAX_COST;
        }
        if (save) {
            FILE * sim_FCD_file = fopen("output/sim_FCDtril.txt", "w");
            gsl_vector_fprintf(sim_FCD_file, sim_FCD_tril, "%f");
            fclose(sim_FCD_file);
        }
        gsl_vector * emp_FCD_tril_copy = gsl_vector_alloc(emp_FCD_tril->size);
        gsl_vector_memcpy(emp_FCD_tril_copy, emp_FCD_tril);
        fcd_ks = ks_stat(
            emp_FCD_tril_copy->data, 
            sim_FCD_tril->data, 
            emp_FCD_tril_copy->size, 
            sim_FCD_tril->size, 
            NULL);
        cost += fcd_ks;
        gsl_vector_free(emp_FCD_tril_copy);
        gsl_vector_free(sim_FCD_tril);
        if (verbose) std::cout << " FCD KS: " << fcd_ks << "\n";
    }
    return cost;

}