#include "Cuda_Convolution.h"
#include <iostream>
#include <iomanip>
#include <ctime>
#include <cstdlib>
#include <sys/time.h>
using namespace std;

__constant__ float Device_Mask[MASK_WIDTH];

__global__ void tiled_1D_convolution(float *input, float *output, int size)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ float data[TILE_SIZE + MASK_WIDTH - 1];

    int halo_left = (blockIdx.x - 1) * blockDim.x + threadIdx.x;
    int halo_right = (blockIdx.x + 1) * blockDim.x + threadIdx.x;

    if (threadIdx.x >= (blockDim.x - RADIUS))
    {
        data[threadIdx.x - (blockDim.x - RADIUS)] = (halo_left < 0) ? 0 : input[halo_left];
    }
    data[RADIUS + threadIdx.x] = (index < size) ? input[index] : 0;
    if (threadIdx.x < RADIUS)
    {
        data[RADIUS + blockDim.x + threadIdx.x] = (halo_right >= size) ? 0 : input[halo_right];
    }
    __syncthreads();

    if (index < size)
    {
        float result = 0;
        for (int i = 0; i < MASK_WIDTH; i++)
        {
            result += data[threadIdx.x + i] * Device_Mask[i];
        }
        output[index] = result;
    }
}

Cuda_Convolution ::Cuda_Convolution()
{
    size_t bytes = ARRAY_LEN * sizeof(float);
    cudaError_t error;

    error = cudaMalloc((void **)&Device_Input, bytes);
    if (error != cudaSuccess)
    {
        std::cerr << "Failed to allocate device vector Device_Input (error code " << cudaGetErrorString(error) << ")!" << std::endl;
        exit(EXIT_FAILURE);
    }
    error = cudaMalloc((void **)&Device_Output, bytes);
    if (error != cudaSuccess)
    {
        std::cerr << "Failed to allocate device vector Device_Output (error code " << cudaGetErrorString(error) << ")!" << std::endl;
        exit(EXIT_FAILURE);
    }
}

Cuda_Convolution ::~Cuda_Convolution()
{
    cudaFree(Device_Input);
    cudaFree(Device_Output);
}

void Cuda_Convolution ::BasicConvolution(float *input, float *output, float *mask)
{
    size_t bytes = ARRAY_LEN * sizeof(float);

    cudaError_t error;
    error = cudaMemcpy(Device_Input, input, bytes, cudaMemcpyHostToDevice);
    if (error != cudaSuccess)
    {
        std::cerr << "Failed to copy input data to device (error code " << cudaGetErrorString(error) << ")!" << endl;
        exit(EXIT_FAILURE);
    }
    error = cudaMemcpyToSymbol(Device_Mask, mask, MASK_WIDTH * sizeof(float));
    if (error != cudaSuccess)
    {
        cerr << "Failed to copy mask to constant memory (error code " << cudaGetErrorString(error) << ")!" << endl;
        exit(EXIT_FAILURE);
    }

    dim3 DimGrid((ARRAY_LEN + TILE_SIZE - 1) / TILE_SIZE);
    dim3 DimBlock(TILE_SIZE);

    struct timeval GPU_Start, GPU_End;

    gettimeofday(&GPU_Start, NULL);
    tiled_1D_convolution<<<DimGrid, DimBlock>>>(Device_Input, Device_Output, ARRAY_LEN);
    cudaDeviceSynchronize();
    gettimeofday(&GPU_End, NULL);
    double Total_Time = (GPU_End.tv_sec - GPU_Start.tv_sec) * 1000 + (GPU_End.tv_usec - GPU_Start.tv_usec) * 0.001;

    cout << fixed << setprecision(3) << "GPU execution time(latency): " << Total_Time << " ms" << endl;

    error = cudaMemcpy(output, Device_Output, bytes, cudaMemcpyDeviceToHost);
    if (error != cudaSuccess)
    {
        cerr << "Failed to copy output data to host (error code " << cudaGetErrorString(error) << ")!" << endl;
        exit(EXIT_FAILURE);
    }
}

void Cuda_Convolution ::Random_Input(float *data, int size)
{
    static bool seed_set = false;
    if (!seed_set)
    {
        srand(time(0));
        seed_set = true;
    }

    for (int i = 0; i < size; i++)
    {
        data[i] = static_cast<float>(rand()) / RAND_MAX;
    }
}

void Cuda_Convolution ::Golden_Reference(float *input, float *output, float *mask, int size)
{
    struct timeval CPU_Start, CPU_End;
    gettimeofday(&CPU_Start, NULL);
    for (int i = 0; i < size; i++)
    {
        float temp = 0;
        for (int j = 0; j < MASK_WIDTH; j++)
        {
            int index = i + j - RADIUS;
            if (index >= 0 && index < size)
            {
                temp += input[index] * mask[j];
            }
        }
        output[i] = temp;
    }
    gettimeofday(&CPU_End, NULL);
    double CPU_Time = (CPU_End.tv_sec - CPU_Start.tv_sec) * 1000 + (CPU_End.tv_usec - CPU_Start.tv_usec) * 0.001;
    cout << "CPU executation time(latency): " << CPU_Time << " ms" << endl;
}

void Cuda_Convolution ::Verify(float *CPU_Result, float *GPU_Result, int size)
{
    bool correct = true;
    for (int i = 0; i < size; i++)
    {
        if (fabs(CPU_Result[i] - GPU_Result[i] > 1e-3))
        {
            correct = false;
            cout << "CPU_Result at " << i << " is " << CPU_Result[i] << endl;
            cout << "GPU_Result at " << i << " is " << GPU_Result[i] << endl;
            cout << "Verfication failed at element " << i << endl;
            break;
        }
    }
    correct ? cout << "Test passed!" << endl : cout << "Test failed!" << endl;
}