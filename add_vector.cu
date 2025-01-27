
// This is a simple example of a CUDA program that adds two vectors in parallel.
// It is a modified version of the example provided by NVIDIA in the CUDA Toolkit.

//#define EXECUTION_PARAMETERS  // define this macro to include the execution parameters in the output

#ifdef EXECUTION_PARAMETERS
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);

__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

int main()
{
    const int arraySize = 5;
    const int a[arraySize] = { 1, 2, 3, 4, 5 };
    const int b[arraySize] = { 10, 20, 30, 40, 50 };
	int c[arraySize] = { 0 };  // the result of the addition will be placed here

    // Add vectors in parallel.
    cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize);
	if (cudaStatus != cudaSuccess) { // if the function addWithCuda() returns an error
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }

	printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",  // print the result
        c[0], c[1], c[2], c[3], c[4]);

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaStatus = cudaDeviceReset();  // reset the device
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
// This helper 
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
{
	int* dev_a = 0; // pointer to the device memory, allocated by cudaMalloc()
    int *dev_b = 0;
    int *dev_c = 0;
	cudaError_t cudaStatus;  // the status of the CUDA functions

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

	// Allocate GPU buffers for three vectors (two input, one output), like malloc() but for the GPU
	cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));  // allocate memory for the output vector
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
    // Cudamalloc:
    // Input_
	//       1. Pointer to the pointer to the memory to be allocated
	//       2. Size of the memory to be allocated
	// Output_
	//       1. cudaSuccess if the memory was allocated successfully
	//       2. cudaErrorMemoryAllocation if the memory could not be allocated
	//       3. cudaErrorInvalidValue if the size of the memory to be allocated is 0
	//       4. cudaErrorMemoryAllocation if the memory could not be allocated
	//       5. cudaErrorInitializationError if the driver is not initialized
	//       6. cudaErrorInvalidDevice if the device is invalid

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    size_t size_free, size_total;
    cudaMemGetInfo(&size_free, &size_total);  // get the amount of free and total memory on the device

    printf("Free memory after allocation: %lu bytes\n", size_free);
    printf("Total memory after allocation: %lu bytes\n", size_total);
    printf("Free memory after allocation: %lu KB\n", size_free / 1024);
    printf("Total memory after allocation: %lu KB\n", size_total / 1024);
    printf("Free memory after allocation: %lu MB\n", size_free / (1024 * 1024));
    printf("Total memory after allocation: %lu MB\n", size_total / (1024 * 1024));
    printf("Free memory after allocation: %lu GB\n", size_free / (1024 * 1024 * 1024));
    printf("Total memory after allocation: %lu GB\n", size_total / (1024 * 1024 * 1024));

    // Launch a kernel on the GPU with one thread for each element.
    addKernel<<<1, size>>>(dev_c, dev_a, dev_b);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    
    return cudaStatus;
}
#endif