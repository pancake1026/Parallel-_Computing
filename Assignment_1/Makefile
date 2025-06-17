# define variable
CUDA_PATH = /usr/local/cuda
NVCC = $(CUDA_PATH)/bin/nvcc
TARGET = lab1
UCINETID = yuhsuw6

#compile option
NVCCFLAGS = -arch=sm_86 -O2

#source file and target file
SRCS = yuhsuw6_lab1.cu Tiled_Kernel.cu Matrix_Multiplier.cu
OBJS = $(SRCS:.cu=.o)

#default rule
all: $(TARGET)

#genertate target file
$(TARGET): $(OBJS)
	$(NVCC) $(NVCCFLAGS) -o $(UCINETID)_lab1 $(OBJS)

#compile each .cu to .o
yuhsuw6_lab1.o: yuhsuw6_lab1.cu Tiled_Kernel.h Matrix_Multiplier.h
	$(NVCC) $(NVCCFLAGS) -c yuhsuw6_lab1.cu -o yuhsuw6_lab1.o
Tiled_Kernel.o: Tiled_Kernel.cu Tiled_Kernel.h
	$(NVCC) $(NVCCFLAGS) -c Tiled_Kernel.cu -o Tiled_Kernel.o
Matrix_Multiplier.o: Matrix_Multiplier.cu Matrix_Multiplier.h Tiled_Kernel.h
	$(NVCC) $(NVCCFLAGS) -c Matrix_Multiplier.cu -o Matrix_Multiplier.o

#clean object
clean:
	rm -f $(OBJS) $(UCINETID)_lab1
