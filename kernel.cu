#include <iostream>
#include "kernel.h"

__global__ void myKernel() {
    printf("Hello from CUDA Kernel!\n");
}

void launchKernel() {
    myKernel << <1, 1 >> > ();
    cudaDeviceSynchronize(); // Ensure kernel execution completes
}
