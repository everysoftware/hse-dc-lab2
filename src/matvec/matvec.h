//
// Created by pravi on 01.12.2024.
//
#ifndef MATVEC_H

#define MATVEC_H

#include <math.h>
#include <stdint.h>

enum PARTITION {
    ROW_PARTITION,
    COLUMN_PARTITION,
    BLOCK_PARTITION
};

void matrix_vector_multiply(enum PARTITION mode, const double_t *mat, const double_t *vec, double_t *result, uint32_t rows,
                    uint32_t cols, int32_t rank, int32_t comm_size);

#endif
