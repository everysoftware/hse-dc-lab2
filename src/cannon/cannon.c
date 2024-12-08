//
// Created by pravi on 02.12.2024.
//

#include "cannon.h"
#include <mpi.h>
#include <stdlib.h>
#include <math.h>

void matrix_multiply_local(const double_t *const mat_a, const double_t *const mat_b, double_t *result,
                           const uint32_t size) {
    for (uint32_t row = 0; row < size; row++) {
        for (uint32_t col = 0; col < size; col++) {
            for (uint32_t idx = 0; idx < size; idx++) {
                result[row * size + col] += mat_a[row * size + idx] * mat_b[idx * size + col];
            }
        }
    }
}

void cannon(const double_t *const global_mat_a, const double_t *const global_mat_b, double_t *global_result,
            const uint32_t mat_size, int32_t rank, const int32_t comm_size) {
    uint32_t local_size, local_area;

    double_t *local_mat_a;
    double_t *local_mat_b;
    double_t *local_result;

    local_size = mat_size / sqrt(comm_size);
    local_area = local_size * local_size;

    local_mat_a = malloc(local_area * sizeof(double_t));
    local_mat_b = malloc(local_area * sizeof(double_t));
    local_result = malloc(local_area * sizeof(double_t));

    for (uint32_t i = 0; i < local_area; i++) {
        local_result[i] = 0.0;
    }

    MPI_Scatter(global_mat_a, local_area, MPI_DOUBLE, local_mat_a, local_area, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatter(global_mat_b, local_area, MPI_DOUBLE, local_mat_b, local_area, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    matrix_multiply_local(local_mat_a, local_mat_b, local_result, local_size);

    MPI_Gather(local_result, local_area, MPI_DOUBLE, global_result, local_area, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    free(local_result);
    free(local_mat_b);
    free(local_mat_a);
}
