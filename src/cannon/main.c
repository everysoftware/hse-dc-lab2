#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <sys/types.h>
#include <mpi.h>
#include <time.h>

#include "cannon.h"

#define RAND_N 100
#define RAND_M 7.0

#define BM_INIT_SIZE 50
#define BM_MAX_SIZE 2500
#define BM_SIZE_STEP 50
#define MB_FILE "cannon.csv"

int32_t comm_size = -1, rank = -1;

void generate_vector(double_t *vector, const uint32_t size) {
    if (rank == 0) {
        for (uint32_t i = 0; i < size; i++) {
            vector[i] = (double_t) (rand() % RAND_N) / RAND_M;
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
            "Usage: mpiexec -n <num_processes> %s <num_size> [--benchmark]\n\n", program_name);
}

void benchmark() {
    FILE *file = fopen(MB_FILE, "w");
    if (file == NULL) {
        fprintf(stderr, "Error opening file for writing benchmark results.\n");
        return;
    }

    if (rank == 0) {
        fprintf(file, "n,size,time\n");
    }

    for (uint32_t i = BM_INIT_SIZE; i <= BM_MAX_SIZE; i += BM_SIZE_STEP) {
        const uint32_t matrix_size = i;

        double_t *mat_a = malloc(matrix_size * matrix_size * sizeof(double_t));
        double_t *mat_b = malloc(matrix_size * matrix_size * sizeof(double_t));
        double_t *result = malloc(matrix_size * matrix_size * sizeof(double_t));

        for (uint32_t row = 0; row < matrix_size; row++) {
            generate_vector(mat_a + matrix_size * row, matrix_size);
            generate_vector(mat_b + matrix_size * row, matrix_size);
        }

        MPI_Barrier(MPI_COMM_WORLD);

        const double_t start_time = MPI_Wtime();
        cannon(mat_a, mat_b, result, matrix_size, rank, comm_size);
        const double_t end_time = MPI_Wtime();

        if (rank == 0) {
            fprintf(file, "%d,%4d,%f\n", comm_size, matrix_size, end_time - start_time);
        }

        free(result);
        free(mat_b);
        free(mat_a);
    }

    fclose(file);
}

int main(const int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int benchmark_mode = 0;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--benchmark") == 0) {
            benchmark_mode = 1;
            break;
        }
    }

    if (benchmark_mode) {
        benchmark();
    } else {
        if (argc < 2) {
            if (rank == 0) {
                display_usage(argv[0]);
            }
            goto ret;
        }

        const uint32_t matrix_size = atoi(argv[1]);
        const uint32_t total_elements = matrix_size * matrix_size;

        if (rank == 0) {
            srand(time(nullptr));
        }

        double_t *mat_a = malloc(total_elements * sizeof(double_t));
        double_t *mat_b = malloc(total_elements * sizeof(double_t));
        double_t *result = malloc(total_elements * sizeof(double_t));

        for (uint32_t row = 0; row < matrix_size; ++row) {
            generate_vector(mat_a + matrix_size * row, matrix_size);
            generate_vector(mat_b + matrix_size * row, matrix_size);
        }
        memset(result, 0, total_elements * sizeof(double_t));

        if (rank == 0) {
            display_matrix("Matrix A", mat_a, matrix_size);
            display_matrix("Matrix B", mat_b, matrix_size);
        }

        MPI_Barrier(MPI_COMM_WORLD);

        cannon(mat_a, mat_b, result, matrix_size, rank, comm_size);

        if (rank == 0) {
            display_matrix("Resultant Matrix", result, matrix_size);
        }

        free(mat_a);
        free(mat_b);
        free(result);
    }

ret:
    MPI_Finalize();
    return EXIT_SUCCESS;
}
