#include <math.h>
#include <mpi.h>
#include <stdlib.h>
#include <stdio.h>

double f(double x, double y) {
    return 0.0;
}

void allocate_matrix(double **matrix, const int rows, const int cols) {
    *matrix = (double *) calloc(rows * cols, sizeof(double));
}

void free_matrix(double *matrix) {
    free(matrix);
}

double *get_matrix_element(double *matrix, const int cols, const int row, const int col) {
    return &matrix[row * cols + col];
}

void dirichlet(const int n_points, const double eps, const double temperature, const double std, const int rank,
               const int comm_size) {
    const int n_total = n_points + 2;
    double *F;
    double *U;

    allocate_matrix(&F, n_points, n_points);
    allocate_matrix(&U, n_total, n_total);

    const double h = 1.0 / (n_points - 1);
    const double h2 = h * h;

    if (rank == 0) {
        for (int i = 0; i < n_points; ++i) {
            for (int j = 0; j < n_points; ++j) {
                if (i == 0) {
                    *get_matrix_element(U, n_total, 0, j + 1) = temperature;
                    *get_matrix_element(U, n_total, n_points + 1, j + 1) = temperature;
                }
                *get_matrix_element(F, n_points, i, j) = f(i * h, j * h);
                *get_matrix_element(U, n_total, i + 1, j + 1) = ((double) rand() / RAND_MAX * 2 - 1) * std;
            }
            *get_matrix_element(U, n_total, i + 1, 0) = temperature;
            *get_matrix_element(U, n_total, i + 1, n_points + 1) = temperature;
        }
        *get_matrix_element(U, n_total, n_points, 0) = temperature;
        *get_matrix_element(U, n_total, n_points, n_points + 1) = temperature;
        *get_matrix_element(U, n_total, n_points + 1, 0) = temperature;
        *get_matrix_element(U, n_total, n_points + 1, n_points + 1) = temperature;
    }

    MPI_Bcast(U, n_total * n_total, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(F, n_points * n_points, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    double d_max = 1.0;
    double dm = 0.0;
    const int s_P = (int) round(sqrt(comm_size));
    const int shift = n_points / s_P;
    const int shift_sq = shift * shift;

    const int i_P = rank / s_P;
    const int j_P = rank % s_P;

    double *sendbuf = calloc(shift_sq, sizeof(double));
    double *recvbuf = calloc(n_points * n_points, sizeof(double));
    int *recvcounts = calloc(comm_size, sizeof(int));
    int *displs = calloc(comm_size, sizeof(int));

    for (int i = 0; i < comm_size; ++i) {
        recvcounts[i] = shift_sq;
        displs[i] = i == 0 ? 0 : displs[i - 1] + recvcounts[i - 1];
    }

    while (d_max > eps) {
        d_max = 0.0;
        dm = 0.0;

        const int i0 = i_P * shift + 1;
        const int j0 = j_P * shift + 1;
        int counter = 0;

        for (int n = 0; n < shift; ++n) {
            for (int i = i0; i <= i0 + n; ++i) {
                const int j = j0 + n - (i - i0);

                const double tmp = *get_matrix_element(U, n_total, i, j);
                *get_matrix_element(U, n_total, i, j) =
                        sendbuf[counter] =
                        0.25 * (*get_matrix_element(U, n_total, i - 1, j) +
                                *get_matrix_element(U, n_total, i + 1, j) +
                                *get_matrix_element(U, n_total, i, j - 1) +
                                *get_matrix_element(U, n_total, i, j + 1) -
                                h2 * *get_matrix_element(F, n_points, i - 1, j - 1));

                const double d = fabs(tmp - sendbuf[counter++]);
                dm = dm < d ? d : dm;
            }
        }

        for (int n = shift - 2; n >= 0; --n) {
            for (int i = i0 + shift - 1; i >= i0 + shift - 1 - n; --i) {
                const int j = j0 + 2 * shift - 2 - n - (i - i0);

                const double tmp = *get_matrix_element(U, n_total, i, j);
                *get_matrix_element(U, n_total, i, j) =
                        sendbuf[counter] =
                        0.25 * (*get_matrix_element(U, n_total, i - 1, j) +
                                *get_matrix_element(U, n_total, i + 1, j) +
                                *get_matrix_element(U, n_total, i, j - 1) +
                                *get_matrix_element(U, n_total, i, j + 1) -
                                h2 * *get_matrix_element(F, n_points, i - 1, j - 1));

                const double d = fabs(tmp - sendbuf[counter++]);
                dm = dm < d ? d : dm;
            }
        }

        MPI_Allgatherv(sendbuf, counter, MPI_DOUBLE, recvbuf, recvcounts, displs, MPI_DOUBLE, MPI_COMM_WORLD);
        MPI_Allreduce(&dm, &d_max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    }

    if (rank == 0) {
        FILE *file = fopen("output_matrix.txt", "w");
        for (int i = 0; i < n_points; ++i) {
            for (int j = 0; j < n_points; ++j) {
                fprintf(file, "%e ", *get_matrix_element(U, n_total, i + 1, j + 1));
            }
            fprintf(file, "\n");
        }
        fclose(file);
    }

    free_matrix(U);
    free_matrix(F);
    free(sendbuf);
    free(recvbuf);
    free(recvcounts);
    free(displs);
}
