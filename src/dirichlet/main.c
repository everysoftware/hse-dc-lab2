//
// Created by pravi on 08.12.2024.
//
#include <math.h>
#include <mpi.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <string.h>

#include "dirichlet.h"

#define TEMPERATURE 100.0
#define EPSILON 1e-6
#define BM_FILE "dirichlet.csv"

int32_t comm_size = -1, rank = -1;

void display_usage(const char *program_name) {
    fprintf(stderr, "Usage: mpiexec -n <num_processes> %s <num_of_points> [--benchmark]\n", program_name);
}

void benchmark() {
    const int sizes[] = {100, 500, 1000, 2000, 3000};
    constexpr int num_sizes = sizeof(sizes) / sizeof(sizes[0]);

    FILE *file = fopen(BM_FILE, "w");
    if (file == NULL) {
        if (rank == 0) {
            fprintf(stderr, "Error opening file for writing benchmark results.\n");
        }
        return;
    }

    if (rank == 0) {
        fprintf(file, "n,size,time\n");
    }

    for (int i = 0; i < num_sizes; i++) {
        const int n_points = sizes[i];

        MPI_Barrier(MPI_COMM_WORLD);
        const double start_time = MPI_Wtime();

        dirichlet(n_points, EPSILON, TEMPERATURE, 5.0, rank, comm_size);

        MPI_Barrier(MPI_COMM_WORLD);
        const double end_time = MPI_Wtime();

        const double time_parallel = end_time - start_time;

        if (rank == 0) {
            fprintf(file, "%d,%d,%f\n", comm_size, n_points, time_parallel);
        }
    }

    fclose(file);
}

int main(const int argc, char *argv[]) {
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

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

        const int n_points = atoi(argv[1]);

        dirichlet(n_points, EPSILON, TEMPERATURE, 5.0, rank, comm_size);
    }

ret:
    MPI_Finalize();
    return EXIT_SUCCESS;
}
