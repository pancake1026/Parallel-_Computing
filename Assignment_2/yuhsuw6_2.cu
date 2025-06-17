#include "Cuda_Convolution.h"
#include <iostream>
using namespace std;

int main(int argc, char *argv[])
{
    if (argc != 2)
    {
        cerr << "Usage: <file> <Test Time>" << endl;
        return 1;
    }

    int testSample = atoi(argv[1]);

    Cuda_Convolution convolution;

    float Input[ARRAY_LEN];
    float Mask[MASK_WIDTH];
    float GPU_Output[ARRAY_LEN];
    float CPU_Output[ARRAY_LEN];

    for (int i = 0; i < testSample; i++)
    {
        cout << ">>>>>>>>>>>>>>>>> test No. " << i + 1 << " >>>>>>>>>>>>>>>>>" << endl;
        convolution.Random_Input(Input, ARRAY_LEN);
        convolution.Random_Input(Mask, MASK_WIDTH);
        convolution.Golden_Reference(Input, CPU_Output, Mask, ARRAY_LEN);
        convolution.BasicConvolution(Input, GPU_Output, Mask);
        convolution.Verify(CPU_Output, GPU_Output, ARRAY_LEN);
    }

    return 0;
}
