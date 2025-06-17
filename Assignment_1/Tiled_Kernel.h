#ifndef TILED_KERNEL_H
#define TILED_KERNEL_H

#include <cuda_runtime.h>

__global__ void MatrixMulCUDA(const double *A, const double *B, double *C, int A_Row, int A_Col, int B_Col);

#endif