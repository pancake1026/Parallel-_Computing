#ifndef MATRIX_MULTIPLIER_H
#define MATRIX_MULTIPLIER_H

#define TILE_SIZE 16

#include <cuda_runtime.h>
#include <cuda_profiler_api.h>

class Matrix_Multiplier
{
public:
    Matrix_Multiplier(int A_Row, int A_Col, int B_Col);
    ~Matrix_Multiplier();

    void Matrix_Multiply();

private:
    void initial_Matrix();
    void cpu_Matrix();
    void gpu_Matrix();
    void verify_Result();
    int A_Row, A_Col, B_Col;
    size_t size_A, size_B, size_C;
    double *host_A, *host_B, *host_C, *host_cpu_C;
};
#endif
