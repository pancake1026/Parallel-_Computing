#include "Tiled_Kernel.h"

#define TILE_SIZE 16

__global__ void MatrixMulCUDA(const double *A, const double *B, double *C, int A_Row, int A_Col, int B_Col)
{
    __shared__ double subTileA[TILE_SIZE][TILE_SIZE];
    __shared__ double subTileB[TILE_SIZE][TILE_SIZE];

    int Row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int Col = blockIdx.x * TILE_SIZE + threadIdx.x;

    double C_value = 0;

    for (int i = 0; i < (A_Col + TILE_SIZE - 1) / TILE_SIZE; i++)
    {
        subTileA[threadIdx.y][threadIdx.x] = (Row < A_Row && (threadIdx.x + (TILE_SIZE * i)) < A_Col) ? A[Row * A_Col + i * TILE_SIZE + threadIdx.x] : 0;
        subTileB[threadIdx.y][threadIdx.x] = (Col < B_Col && (threadIdx.y + (TILE_SIZE * i)) < A_Col) ? B[(threadIdx.y + (TILE_SIZE * i)) * B_Col + Col] : 0;
        __syncthreads();
        for (int j = 0; j < TILE_SIZE; j++)
        {
            C_value += subTileA[threadIdx.y][j] * subTileB[j][threadIdx.x];
        }
        __syncthreads();
    }
    if (Row < A_Row && Col < B_Col)
    {
        C[Row * B_Col + Col] = C_value;
    }
}