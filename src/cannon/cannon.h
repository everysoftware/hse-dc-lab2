//
// Created by pravi on 02.12.2024.
//

#ifndef CANNON_H
#define CANNON_H

#include <math.h>
#include <stdint.h>

void cannon(double_t *global_mat_a, double_t *global_mat_b, double_t *global_result, uint32_t mat_size,
            int32_t rank, int32_t comm_size);

#endif //CANNON_H
