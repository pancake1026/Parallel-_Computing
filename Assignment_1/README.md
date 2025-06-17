Lab 1: Tiled Matrix Multiplication

-Overview

    This project provides a CUDA-based solution for tiled matrix multiplication, capable of handling matrices of any size. It leverages both CPU and GPU to perform the multiplication and validate the results. The implementation is based on the provided template, with modifications to modularize the code. The original functionality of generating random matrices as test cases and evaluating the results has been preserved.

-Files Included

    yuhsuw6_lab1.cu: The main source file that contains the host code and CUDA kernel calls.
    Tiled_Kernel.cu: The CUDA kernel file that performs tiled matrix multiplication.
    Matrix_Multiplier.cu: Additional functions for handling matrix multiplication logic.
    Tiled_Kernel.h and Matrix_Multiplier.h: Header files containing function prototypes and shared definitions.
    Makefile: Makefile for compiling the CUDA source files.

-Compilation Instructions

    Navigate to the project directory on the server:
        cd ~/snap/snapd-desktop-integration/current/Documents/Assignment_1
    
    Then, to compile the project, run the following command in the terminal:
        make

    This will compile all `.cu` files and generate an executable named `yuhsuw6_lab1`.

-Running the Program

    To run the program, specify the number of test iterations as an argument:
        ./yuhsuw6_lab1 <Test Time>
        
    For example, to run 5 test iterations:
        ./yuhsuw6_lab1 5

-Cleaning Up
    To remove all compiled files and the executable, use the following command:
        make clean