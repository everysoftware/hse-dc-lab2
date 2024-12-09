#include <math.h>
#include <mpi.h>
#include <stdlib.h>
#include <stdio.h>

#include "dirichlet.h"

#define MAX_ITERATIONS 1000
#define OUTPUT_FILE "dirichlet.txt"

// Source function
double_t f(double_t x, double_t y) {
    return 0.0; // f(x, y) = 0
}

// Matrix functions
void allocate_matrix(double_t **matrix, const int32_t rows, const int32_t cols) {
    *matrix = (double *) calloc(rows * cols, sizeof(double));
    if (*matrix == NULL) {
        fprintf(stderr, "Memory allocation failed\n");
        exit(EXIT_FAILURE);
    }
}

void free_matrix(double_t *matrix) {
    free(matrix);
}

double_t *get_matrix_element(double_t *matrix, const int32_t cols, const int32_t row, const int32_t col) {
    return &matrix[row * cols + col];
}

void dirichlet(const int32_t n_points, const double_t eps, const double_t temperature, const double_t std,
               const int32_t rank,
               const int32_t comm_size) {
    const int32_t n_total = n_points + 2; // Including edge points
    double_t *F, *U;

    allocate_matrix(&F, n_points, n_points);
    allocate_matrix(&U, n_total, n_total);

    const double_t h = 1.0 / (n_points - 1); // Coordinate step
    const double_t h2 = h * h;

    if (rank == 0) {
        for (int32_t i = 0; i < n_points; ++i) {
            for (int32_t j = 0; j < n_points; ++j) {
                if (i == 0) {
                    *get_matrix_element(U, n_total, 0, j + 1) = temperature;
                    *get_matrix_element(U, n_total, n_points + 1, j + 1) = temperature;
                }
                *get_matrix_element(F, n_points, i, j) = f(i * h, j * h); // Источник
                *get_matrix_element(U, n_total, i + 1, j + 1) = ((double) rand() / RAND_MAX * 2 - 1) * std;
                // Started value
            }
            *get_matrix_element(U, n_total, i + 1, 0) = temperature;
            *get_matrix_element(U, n_total, i + 1, n_points + 1) = temperature;
        }
        *get_matrix_element(U, n_total, n_points, 0) = temperature;
        *get_matrix_element(U, n_total, n_points, n_points + 1) = temperature;
        *get_matrix_element(U, n_total, n_points + 1, 0) = temperature;
        *get_matrix_element(U, n_total, n_points + 1, n_points + 1) = temperature;
    }

    // Deliver matrices to processes
    MPI_Bcast(U, n_total * n_total, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(F, n_points * n_points, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    double_t d_max = 1.0;
    double_t dm = 0.0;
    const int32_t s_P = (int) round(sqrt(comm_size));
    const int32_t shift = n_points / s_P;
    const int32_t shift_sq = shift * shift;

    const int32_t i_P = rank / s_P;
    const int32_t j_P = rank % s_P;

    double_t *sendbuf = calloc(shift_sq, sizeof(double));
    double_t *recvbuf = calloc(n_points * n_points, sizeof(double));
    int32_t *recvcounts = calloc(comm_size, sizeof(int));
    int32_t *displs = calloc(comm_size, sizeof(int));

    if (sendbuf == NULL || recvbuf == NULL || recvcounts == NULL || displs == NULL) {
        fprintf(stderr, "Memory allocation failed\n");
        exit(EXIT_FAILURE);
    }

    // Allgatherv preparation
    for (int32_t i = 0; i < comm_size; ++i) {
        recvcounts[i] = shift_sq;
        displs[i] = i == 0 ? 0 : displs[i - 1] + recvcounts[i - 1];
    }

    // Until convergence
    for (int32_t iter = 0; iter < MAX_ITERATIONS && d_max > eps; ++iter) {
        d_max = 0.0;
        dm = 0.0;

        const int32_t i0 = i_P * shift + 1;
        const int32_t j0 = j_P * shift + 1;
        int32_t counter = 0;

        // Up by diagonal
        for (int32_t n = 0; n < shift; ++n) {
            for (int32_t i = i0; i <= i0 + n; ++i) {
                const int32_t j = j0 + n - (i - i0);

                const double_t tmp = *get_matrix_element(U, n_total, i, j);
                *get_matrix_element(U, n_total, i, j) =
                        sendbuf[counter] =
                        0.25 * (*get_matrix_element(U, n_total, i - 1, j) +
                                *get_matrix_element(U, n_total, i + 1, j) +
                                *get_matrix_element(U, n_total, i, j - 1) +
                                *get_matrix_element(U, n_total, i, j + 1) -
                                h2 * *get_matrix_element(F, n_points, i - 1, j - 1));

                const double_t d = fabs(tmp - sendbuf[counter++]);
                dm = dm < d ? d : dm;
            }
        }

        // Down by diagonal
        for (int32_t n = shift - 2; n >= 0; --n) {
            for (int32_t i = i0 + shift - 1; i >= i0 + shift - 1 - n; --i) {
                const int32_t j = j0 + 2 * shift - 2 - n - (i - i0);

                const double_t tmp = *get_matrix_element(U, n_total, i, j);
                *get_matrix_element(U, n_total, i, j) =
                        sendbuf[counter] =
                        0.25 * (*get_matrix_element(U, n_total, i - 1, j) +
                                *get_matrix_element(U, n_total, i + 1, j) +
                                *get_matrix_element(U, n_total, i, j - 1) +
                                *get_matrix_element(U, n_total, i, j + 1) -
                                h2 * *get_matrix_element(F, n_points, i - 1, j - 1));

                const double_t d = fabs(tmp - sendbuf[counter++]);
                dm = dm < d ? d : dm;
            }
        }

        // Collect data and calculate max error
        MPI_Allgatherv(sendbuf, counter, MPI_DOUBLE, recvbuf, recvcounts, displs, MPI_DOUBLE, MPI_COMM_WORLD);
        MPI_Allreduce(&dm, &d_max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    }

    if (rank == 0) {
        FILE *file = fopen(OUTPUT_FILE, "w");
        if (file != NULL) {
            for (int32_t i = 0; i < n_points; ++i) {
                for (int32_t j = 0; j < n_points; ++j) {
                    fprintf(file, "%e ", *get_matrix_element(U, n_total, i + 1, j + 1));
                }
                fprintf(file, "\n");
            }
            fclose(file);
        }
    }

    free_matrix(U);
    free_matrix(F);
    free(sendbuf);
    free(recvbuf);
    free(recvcounts);
    free(displs);
}
