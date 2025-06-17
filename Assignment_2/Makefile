# define variable
CUDA_PATH = /usr/local/cuda
NVCC = $(CUDA_PATH)/bin/nvcc
TARGET = 2
UCINETID = yuhsuw6

#compile option
NVCCFLAGS = -arch=sm_53 -O2

#source file and target file
SRCS = yuhsuw6_2.cu Cuda_Convolution.cu
OBJS = $(SRCS:.cu=.o)

#default rule
all: $(TARGET)

#genertate target file
$(TARGET): $(OBJS)
	$(NVCC) $(NVCCFLAGS) -o $(UCINETID)_2 $(OBJS)

#compile each .cu to .o
yuhsuw6_2.o: yuhsuw6_2.cu Cuda_Convolution.h
	$(NVCC) $(NVCCFLAGS) -c yuhsuw6_2.cu -o yuhsuw6_2.o
Cuda_Convolution.o: Cuda_Convolution.cu Cuda_Convolution.h
	$(NVCC) $(NVCCFLAGS) -c Cuda_Convolution.cu -o Cuda_Convolution.o

#clean object
clean:
	rm -f $(OBJS) $(UCINETID)_2
