//
// Created by pravi on 02.12.2024.
//
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <sys/types.h>
#include <mpi.h>
#include <time.h>

#include "cannon.h"

#define IS_TESTING 1

int32_t comm_size = -1, rank = -1;

void generate_vector(double_t *vector, const uint32_t size) {
    if (rank == 0) {
        for (uint32_t i = 0; i < size; i++) {
            vector[i] = (double_t) (rand() % 100) / 7.0;
        }
    }
    MPI_Bcast(vector, size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
}

void display_matrix(const char *matrix_label, const double_t *matrix, const uint32_t size) {
    printf("\n%s (%dx%d):", matrix_label, size, size);
    for (uint32_t i = 0; i < size * size; i++) {
        if (i % size == 0)
            putchar('\n');
        printf("%4.2lf ", matrix[i]);
    }
    putchar('\n');
}

void display_usage(const char *program_name) {
    fprintf(stderr,
            "Usage: mpiexec -n <num_processes> %s <num_size>\n\n"
            "Arguments:\n"
            "  <num_processes> - number of parallel processes to be used\n"
            "  <num_size>    - size of the square matrices (NxN)\n",
            program_name);
}

#ifdef IS_TESTING
void benchmark() {
    double_t start_time, end_time;
    double_t *mat_a;
    double_t *mat_b;
    double_t *result;
    uint32_t matrix_size;

    for (uint32_t i = 50; i <= 2800; i += 50) {
        matrix_size = i;

        mat_a = malloc(matrix_size * matrix_size * sizeof(double_t));
        mat_b = malloc(matrix_size * matrix_size * sizeof(double_t));
        result = malloc(matrix_size * matrix_size * sizeof(double_t));

        for (uint32_t row = 0; row < matrix_size; row++) {
            generate_vector(mat_a + (matrix_size * row), matrix_size);
            generate_vector(mat_b + (matrix_size * row), matrix_size);
        }

        MPI_Barrier(MPI_COMM_WORLD);

        start_time = MPI_Wtime();
        cannon(mat_a, mat_b, result, matrix_size, rank, comm_size);
        end_time = MPI_Wtime();

        if (rank == 0)
            printf("%d %4d %f\n", comm_size, matrix_size, end_time - start_time);

        free(result);
        free(mat_b);
        free(mat_a);
    }
}
#endif

int main(const int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

#ifndef IS_TESTING
    double_t *mat_a = NULL;
    double_t *mat_b = NULL;
    double_t *result = NULL;

    uint32_t matrix_size, total_elements;

    if (argc < 2) {
        if (rank == 0) display_usage(argv[0]);
        goto cleanup;
    }

    matrix_size = atoi(argv[1]);
    total_elements = matrix_size * matrix_size;

    if (rank == 0)
        srand(time(NULL));

    mat_a = malloc(total_elements * sizeof(double_t));
    mat_b = malloc(total_elements * sizeof(double_t));
    result = malloc(total_elements * sizeof(double_t));

    for (uint32_t row = 0; row < matrix_size; row++) {
        generate_vector(mat_a + (matrix_size * row), matrix_size);
        generate_vector(mat_b + (matrix_size * row), matrix_size);
    }
    memset(result, 0, total_elements * sizeof(double_t));

    if (rank == 0) {
        display_matrix("Matrix A", mat_a, matrix_size);
        display_matrix("Matrix B", mat_b, matrix_size);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    cannon(mat_a, mat_b, result, matrix_size, rank, comm_size);

    if (rank == 0)
        display_matrix("Resultant Matrix", result, matrix_size);

    free(mat_a);
    free(mat_b);
    free(result);

cleanup:
#else
    benchmark();
#endif

    MPI_Finalize();
    return EXIT_SUCCESS;
}
