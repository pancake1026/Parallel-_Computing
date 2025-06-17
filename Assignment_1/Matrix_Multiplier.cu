#include "Matrix_Multiplier.h"
#include "Tiled_Kernel.h"
#include <cstdlib>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <sys/time.h>
using namespace std;

#define nIter 1

Matrix_Multiplier::Matrix_Multiplier(int A_Row, int A_Col, int B_Col)
{
    this->A_Row = A_Row;
    this->A_Col = A_Col;
    this->B_Col = B_Col;
    size_A = A_Row * A_Col * sizeof(double);
    size_B = A_Col * B_Col * sizeof(double);
    size_C = A_Row * B_Col * sizeof(double);
    // CPU allocate
    host_A = new double[A_Row * A_Col];
    host_B = new double[A_Col * B_Col];
    host_C = new double[A_Row * B_Col];
    host_cpu_C = new double[A_Row * B_Col];
}

Matrix_Multiplier::~Matrix_Multiplier()
{
    delete[] host_A;
    delete[] host_B;
    delete[] host_C;
    delete[] host_cpu_C;
}

void Matrix_Multiplier::initial_Matrix()
{
    srand(time(0));
    for (int i = 0; i < A_Row * A_Col; i++)
    {
        host_A[i] = static_cast<double>(rand()) / RAND_MAX;
    }
    for (int j = 0; j < A_Col * B_Col; j++)
    {
        host_B[j] = static_cast<double>(rand()) / RAND_MAX;
    }
}

void Matrix_Multiplier::cpu_Matrix()
{
    for (int i = 0; i < A_Row; i++)
    {
        for (int j = 0; j < B_Col; j++)
        {
            double value = 0;
            for (int k = 0; k < A_Col; k++)
            {
                value += host_A[i * A_Col + k] * host_B[k * B_Col + j];
            }
            host_cpu_C[i * B_Col + j] = value;
        }
    }
}

void Matrix_Multiplier::gpu_Matrix()
{
    double *device_A, *device_B, *device_C;
    cudaError_t error;

    error = cudaMalloc((void **)&device_A, size_A);
    if (error != cudaSuccess)
    {
        cerr << "Failed to allocate device vector A (error code " << cudaGetErrorString(error) << ")!" << endl;
        exit(EXIT_FAILURE);
    }
    error = cudaMalloc((void **)&device_B, size_B);
    if (error != cudaSuccess)
    {
        cerr << "Failed to allocate device vector B (error code " << cudaGetErrorString(error) << ")!" << endl;
        exit(EXIT_FAILURE);
    }
    error = cudaMalloc((void **)&device_C, size_C);
    if (error != cudaSuccess)
    {
        cerr << "Failed to allocate device vector C (error code " << cudaGetErrorString(error) << ")!" << endl;
        exit(EXIT_FAILURE);
    }

    cudaMemcpy(device_A, host_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(device_B, host_B, size_B, cudaMemcpyHostToDevice);

    dim3 DimGrid((int)ceil((double)B_Col / TILE_SIZE), (int)ceil((double)A_Row / TILE_SIZE));
    dim3 DimBlock(TILE_SIZE, TILE_SIZE);

    double total_time = 0;
    struct timeval start, end;

    for (int j = 0; j < nIter; j++)
    {
        cudaDeviceSynchronize();
        cudaProfilerStart();
        gettimeofday(&start, NULL);

        // Call the kernel
        MatrixMulCUDA<<<DimGrid, DimBlock>>>(device_A, device_B, device_C, A_Row, A_Col, B_Col);

        cudaDeviceSynchronize();
        gettimeofday(&end, NULL);
        cudaProfilerStop();
        total_time += (end.tv_sec - start.tv_sec) * 1000 + (end.tv_usec - start.tv_usec) * 0.001;
    }

    cout << fixed << setprecision(3) << "GPU execution time(latency): " << (total_time / nIter) << " ms" << endl;

    cudaMemcpy(host_C, device_C, size_C, cudaMemcpyDeviceToHost);

    cudaFree(device_A);
    cudaFree(device_B);
    cudaFree(device_C);
}

void Matrix_Multiplier::verify_Result()
{
    bool correct = true;
    for (int i = 0; i < A_Row * B_Col; i++)
    {
        if (fabs(host_C[i] - host_cpu_C[i]) > 1e-4)
        {
            correct = false;

            int row = i / B_Col;
            int col = i % B_Col;

            cout << "host_C element at (" << row << ", " << col << ") is " << host_C[i] << endl;
            cout << "host_cpu_C element at (" << row << ", " << col << ") is " << host_cpu_C[i] << endl;
            cout << "Result verification failed at element (" << row << ", " << col << ")" << endl;
        }
    }
    correct ? cout << "Test passed!" << endl : cout << "Test failed!" << endl;
}

void Matrix_Multiplier::Matrix_Multiply()
{
    initial_Matrix();

    // Measure CPU time
    struct timeval cpu_start, cpu_end;
    gettimeofday(&cpu_start, NULL);

    cpu_Matrix();

    gettimeofday(&cpu_end, NULL);
    double cpu_time = (cpu_end.tv_sec - cpu_start.tv_sec) * 1000 + (cpu_end.tv_usec - cpu_start.tv_usec) * 0.001; // CPU time in milliseconds
    cout << "CPU execution time(latency): " << cpu_time << " ms" << endl;

    gpu_Matrix();
    verify_Result();
}
