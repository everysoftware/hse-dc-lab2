//
// Created by pravi on 01.12.2024.
//
#include "matvec.h"

#include <mpi.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

double_t vector_sum(const double_t *const vec, const uint32_t size) {
    double_t total = 0.0;

    for (uint32_t i = 0; i < size; ++i) {
        total += vec[i];
    }

    return total;
}

void matrix_column_multiply(const double_t *const mat, double_t *result, const double_t *vec, const uint32_t rows,
                            const uint32_t cols, const int32_t rank, const int32_t comm_size) {
    uint32_t local_col_count = cols / comm_size;
    const uint32_t start_col = rank * local_col_count;
    if (rank + 1 == comm_size) {
        local_col_count += cols % comm_size;
    }
    const uint32_t end_col = start_col + local_col_count;

    for (uint32_t row = 0; row < rows; ++row) {
        for (uint32_t col = start_col; col < end_col; ++col) {
            result[row] += mat[row * cols + col] * vec[col];
        }
    }
}

double_t row_vector_dot_product(const double_t *const row, const double_t *const col, const uint32_t size) {
    double_t result = 0.0;
    for (uint32_t i = 0; i < size; ++i) {
        result += row[i] * col[i];
    }
    return result;
}

void row_partition_multiply(const double_t *const mat, const double_t *const vec, double_t *result,
                            const uint32_t rows, const uint32_t cols, const int32_t rank, const int32_t comm_size) {
    const uint32_t partition_size = rows / comm_size;

    double_t *local_result = malloc(partition_size * sizeof(double_t));
    memset(local_result, 0, partition_size * sizeof(double_t));

    for (uint32_t i = 0; i < partition_size; ++i) {
        local_result[i] = row_vector_dot_product(mat + i * cols, vec, cols);
    }

    MPI_Reduce(local_result, result, partition_size, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    free(local_result);
}

void column_partition_multiply(const double_t *const mat, const double_t *const vec, double_t *result,
                               const uint32_t rows, const uint32_t cols, const int32_t rank,
                               const int32_t total_procs) {
    double_t *temp_result = malloc(rows * sizeof(double_t));
    memset(temp_result, 0, rows * sizeof(double_t));

    matrix_column_multiply(mat, temp_result, vec, rows, cols, rank, total_procs);

    MPI_Reduce(temp_result, result, rows, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    free(temp_result);
}

void block_partition_multiply(const double_t *const mat, const double_t *const vec, double_t *result,
                              const uint32_t rows, const uint32_t cols, const int32_t rank) {
    const uint32_t half_rows = rows / 2;
    const uint32_t half_cols = cols / 2;

    double_t *temp_result = malloc(rows * sizeof(double_t));
    memset(temp_result, 0, rows * sizeof(double_t));

    uint32_t row_start = 0, row_end = half_rows, col_offset = 0;

    switch (rank) {
        case 0:
            col_offset = half_cols;
            break;
        case 1:
            break;
        case 2:
            row_start = half_rows;
            row_end = rows;
            break;
        case 3:
            row_start = half_rows;
            row_end = rows;
            col_offset = half_cols;
            break;
        default:
            exit(EXIT_FAILURE);
    }

    for (uint32_t i = row_start; i < row_end; ++i) {
        temp_result[i] += row_vector_dot_product(mat + i * cols + col_offset, vec + col_offset, half_cols);
    }

    MPI_Reduce(temp_result, result, rows, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    free(temp_result);
}

void matrix_vector_multiply(const enum PARTITION mode, const double_t *const mat, const double_t *const vec,
                            double_t *result, const uint32_t rows, const uint32_t cols, const int32_t rank,
                            const int32_t comm_size) {
    switch (mode) {
        case ROW_PARTITION:
            row_partition_multiply(mat, vec, result, rows, cols, rank, comm_size);
            break;
        case COLUMN_PARTITION:
            column_partition_multiply(mat, vec, result, rows, cols, rank, comm_size);
            break;
        case BLOCK_PARTITION:
            if (comm_size == 4)
                block_partition_multiply(mat, vec, result, rows, cols, rank);
            break;
    }
}
