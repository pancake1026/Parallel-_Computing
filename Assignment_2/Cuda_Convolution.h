#ifndef CUDA_CONVOLUTION_H
#define CUDA_CONVOLUTION_H

#define ARRAY_LEN 2048
#define MASK_WIDTH 5
#define RADIUS (MASK_WIDTH / 2)
#define TILE_SIZE 512

#include <cuda_runtime.h>
#include <cuda_profiler_api.h>

class Cuda_Convolution
{
public:
    Cuda_Convolution();
    ~Cuda_Convolution();
    void BasicConvolution(float *input, float *output, float *mask);
    void Random_Input(float *data, int size);
    void Golden_Reference(float *input, float *output, float *mask, int size);
    void Verify(float *CPU_Result, float *GPU_Result, int size);

private:
    float *Device_Input, *Device_Output;
};

__global__ void tiled_1D_convolution(float *input, float *output, float *mask, int size);

#endif