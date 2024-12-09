//
// Created by pravi on 01.12.2024.
//
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <mpi.h>
#include <time.h>

#include "matvec.h"

#define RAND_N 100
#define RAND_M 7.0
#define BM_FILE "matvec.csv"

int32_t comm_size = -1, rank = -1;

void generate_vector(double_t *vector, const uint32_t size) {
    if (rank == 0) {
        for (uint32_t i = 0; i < size; i++) {
            vector[i] = (double_t) (rand() % RAND_N) / RAND_M;
        }
    }
    MPI_Bcast(vector, size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
}

void display_usage(const char *program_name) {
    fprintf(stderr, "Usage: mpiexec -n <num_processes> %s <num_rows> <num_columns> [--benchmark]\n", program_name);
}

void save_to_csv(const char *filename, const char *partition, const uint32_t rows, const uint32_t cols,
                 const double_t time) {
    FILE *file = fopen(filename, "a");
    if (!file) {
        if (rank == 0) {
            fprintf(stderr, "Opening file failed: %s\n", filename);
        }
        return;
    }

    if (rank == 0) {
        fprintf(file, "%d,%s,%u,%u,%.6f\n", comm_size, partition, rows, cols, time);
    }

    fclose(file);
}

void test_row_partition(const uint32_t min_rows, const uint32_t max_rows, const uint32_t step_rows,
                        const uint32_t min_cols, const uint32_t max_cols, const uint32_t step_cols,
                        const char *filename) {
    if (rank == 0) {
        printf("Testing row-wise distribution\n");
    }

    for (uint32_t rows = min_rows; rows <= max_rows; rows += step_rows) {
        for (uint32_t cols = min_cols; cols <= max_cols; cols += step_cols) {
            double_t *input_vector = malloc(cols * sizeof(double_t));
            generate_vector(input_vector, cols);

            double_t *matrix_data = malloc(rows * cols * sizeof(double_t));
            generate_vector(matrix_data, rows * cols);

            const uint32_t local_rows = rows / comm_size;
            double_t *partial_result = calloc(local_rows, sizeof(double_t));

            const double_t start_time = MPI_Wtime();
            for (uint32_t row = 0; row < local_rows; row++) {
                const double_t *current_row = matrix_data + (rank * local_rows + row) * cols;
                double_t row_result = 0.0;

                for (uint32_t col = 0; col < cols; col++) {
                    row_result += current_row[col] * input_vector[col];
                }
                partial_result[row] = row_result;
            }

            double_t *result_vector = nullptr;
            if (rank == 0) {
                result_vector = calloc(rows, sizeof(double_t));
            }

            MPI_Gather(partial_result, local_rows, MPI_DOUBLE, result_vector, local_rows, MPI_DOUBLE, 0,
                       MPI_COMM_WORLD);
            const double_t end_time = MPI_Wtime();

            if (rank == 0) {
                save_to_csv(filename, "row", rows, cols, end_time - start_time);
                free(result_vector);
            }

            free(partial_result);
            free(matrix_data);
            free(input_vector);
        }
    }
}

void test_column_partition(const uint32_t min_cols, const uint32_t max_cols, const uint32_t step_cols,
                           const uint32_t min_rows, const uint32_t max_rows, const uint32_t step_rows,
                           const char *filename) {
    if (rank == 0) {
        printf("Testing column-wise distribution\n");
    }

    for (uint32_t rows = min_rows; rows <= max_rows; rows += step_rows) {
        for (uint32_t cols = min_cols; cols <= max_cols; cols += step_cols) {
            double_t *input_vector = malloc(cols * sizeof(double_t));
            generate_vector(input_vector, cols);

            double_t *matrix_data = malloc(rows * cols * sizeof(double_t));
            generate_vector(matrix_data, rows * cols);

            double_t *partial_result = calloc(rows, sizeof(double_t));

            uint32_t local_cols = cols / comm_size;
            const uint32_t col_start = rank * local_cols;
            if (rank + 1 == comm_size) {
                local_cols += cols % comm_size;
            }
            const uint32_t col_end = col_start + local_cols;

            const double_t start_time = MPI_Wtime();
            for (uint32_t row = 0; row < rows; row++) {
                for (uint32_t col = col_start; col < col_end; col++) {
                    partial_result[row] += matrix_data[row * cols + col] * input_vector[col];
                }
            }
            const double_t end_time = MPI_Wtime();

            save_to_csv(filename, "column", rows, cols, end_time - start_time);

            free(partial_result);
            free(matrix_data);
            free(input_vector);
        }
    }
}

void test_block_partition(const uint32_t min_rows, const uint32_t max_rows, const uint32_t step_rows,
                          const uint32_t min_cols, const uint32_t max_cols, const uint32_t step_cols,
                          const char *filename) {
    if (rank == 0) {
        printf("Testing block-wise distribution\n");
    }

    for (uint32_t rows = min_rows; rows <= max_rows; rows += step_rows) {
        for (uint32_t cols = min_cols; cols <= max_cols; cols += step_cols) {
            double_t *input_vector = malloc(cols * sizeof(double_t));
            generate_vector(input_vector, cols);

            double_t *matrix_data = malloc(rows * cols * sizeof(double_t));
            generate_vector(matrix_data, rows * cols);

            double_t *partial_result = calloc(rows, sizeof(double_t));

            const uint32_t local_rows = rows / comm_size;
            const uint32_t local_cols = cols / comm_size;

            const double_t start_time = MPI_Wtime();
            for (uint32_t row = 0; row < local_rows; row++) {
                for (uint32_t col = 0; col < local_cols; col++) {
                    partial_result[row] += matrix_data[row * cols + col] * input_vector[col];
                }
            }
            const double_t end_time = MPI_Wtime();

            save_to_csv(filename, "block", rows, cols, end_time - start_time);

            free(partial_result);
            free(matrix_data);
            free(input_vector);
        }
    }
}

void benchmark() {
    const char *filename = BM_FILE;
    if (rank == 0) {
        FILE *file = fopen(filename, "w");
        if (file) {
            fprintf(file, "n,partition,rows,cols,time\n");
            fclose(file);
        }
    }

    test_row_partition(1000, 10000, 1000, 500, 5000, 500, filename);
    test_column_partition(1000, 10000, 1000, 500, 5000, 500, filename);
    test_block_partition(1000, 10000, 1000, 500, 5000, 500, filename);

    if (rank == 0) {
        printf("Check results in '%s'", filename);
    }
}

int main(const int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int is_benchmark = 0;

    // Parse command-line arguments
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--benchmark") == 0) {
            is_benchmark = 1;
            break;
        }
    }

    if (is_benchmark) {
        benchmark();
    } else {
        double_t *out_vec = nullptr;

        if (argc < 3) {
            if (rank == 0) {
                display_usage(argv[0]);
            }
            goto ret;
        }

        if (rank == 0) {
            srand(time(nullptr));
        }

        const uint32_t rows = atoi(argv[1]);
        const uint32_t cols = atoi(argv[2]);

        if (comm_size > rows) {
            if (rank == 0) {
                fprintf(stderr, "Error: number of processes should be less than number of rows\n");
            }
            goto ret;
        }

        double_t *vec = malloc(cols * sizeof(double_t));
        generate_vector(vec, cols);

        double_t *matrix = malloc(rows * cols * sizeof(double_t));
        generate_vector(matrix, cols * rows);

        if (rank == 0) {
            out_vec = malloc(rows * sizeof(double_t));
            memset(out_vec, 0, rows * sizeof(double_t));
        }

        if (rank == 0) {
            printf("\nMatrix (%dx%d):\n", rows, cols);
            for (uint32_t row = 0; row < rows; row++) {
                for (uint32_t col = 0; col < cols; col++) {
                    printf("%.2lf ", matrix[row * cols + col]);
                }
                putc('\n', stdout);
            }
            printf("Vector (%dx1):\n", cols);
            for (uint32_t col = 0; col < cols; col++) {
                printf("%.2lf ", vec[col]);
            }
            putc('\n', stdout);
        }
        MPI_Barrier(MPI_COMM_WORLD);

        matrix_vector_multiply(ROW_PARTITION, matrix, vec, out_vec, rows, cols, rank, comm_size);

        if (rank == 0) {
            printf("\nResult vector (%dx1):\n", rows);
            for (uint32_t row = 0; row < rows; row++) {
                printf("%.4lf ", out_vec[row]);
            }
        }

        free(matrix);
        free(vec);
        if (rank == 0) {
            free(out_vec);
        }
    }

ret:
    MPI_Finalize();
    return EXIT_SUCCESS;
}
