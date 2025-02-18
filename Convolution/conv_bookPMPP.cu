#include <cuda_runtime.h>
#include <iostream>
#include <device_launch_parameters.h>
#include "main_header.cuh"
#include <chrono>

// basic convolution kernel
// @param N input image
// @param F filter
// @param P output image
// @param r filter radius
// @param width image width
// @param height image height

__global__ void convolution_2d_basic_kernel(float* N, float* F, float* P, int r, int width, int height) {
	int outCol = blockIdx.x * blockDim.x + threadIdx.x;
	int outRow = blockIdx.y * blockDim.y + threadIdx.y;
	float Pvalue = 0.0f;
	for (int fRow = 0; fRow < 2 * r + 1; fRow++) {
		for (int fCol = 0; fCol < 2 * r + 1; fCol++) {
			int inRow = outRow - r + fRow;
			int inCol = outCol - r + fCol;
			if (inRow >= 0 && inRow < height && inCol >= 0 && inCol < width) {
				Pvalue += N[inRow * width + inCol] * F[fRow * (2 * r + 1) + fCol];   // 2D convolution main operation done for each pixel of the output image
			}
		}
	}
	P[outRow * width + outCol] = Pvalue;
}

// COnstant memory is avaiavble for all threads in the block, but cannot be modified during kernel execution
#define FILTER_RADIUS 2
__constant__ float F[2 * FILTER_RADIUS + 1][2 * FILTER_RADIUS + 1];   // constant memory for filter

// convolution kernel using constant memory
// @param N input image
// @param P output image
// @param r filter radius
// @param width image width
// @param height image height

__global__ void convolution_2d_const_mem_kernel(float* N, float* P, int r, int width, int height) {
	int outCol = blockIdx.x * blockDim.x + threadIdx.x;
	int outRow = blockIdx.y * blockDim.y + threadIdx.y;
	float Pvalue = 0.0f;
	for (int fRow = 0; fRow < 2 * r + 1; fRow++) {
		for (int fCol = 0; fCol < 2 * r + 1; fCol++) {
			int inRow = outRow - r + fRow;
			int inCol = outCol - r + fCol;
			if (inRow >= 0 && inRow < height && inCol >= 0 && inCol < width) {
				Pvalue += N[inRow * width + inCol] * F[fRow][fCol];   // 2D convolution main operation done for each pixel of the output image
			}
		}
	}
	P[outRow * width + outCol] = Pvalue;
}


int launchConvolution2DBasicKernel(int width, int height, int r, float* h_N, float* h_F, float* h_P, dim3 blockDim, dim3 gridDim) {
    // Initialize default input data if not provided
    int fixed_value = 5;
    float default_h_N[25];
    float default_h_F[9];
    float default_h_P[25] = { 0 }; // Output image

    for (int i = 0; i < 25; ++i) {
        default_h_N[i] = fixed_value;
    }
    for (int i = 0; i < 9; ++i) {
        default_h_F[i] = fixed_value;
    }

    if (h_N == nullptr) h_N = default_h_N;
    if (h_F == nullptr) h_F = default_h_F;
    if (h_P == nullptr) h_P = default_h_P;

    // Allocate device memory
    float* d_N, * d_F, * d_P;
    size_t imageSize = width * height * sizeof(float);
    size_t filterSize = (2 * r + 1) * (2 * r + 1) * sizeof(float);

    cudaMalloc(&d_N, imageSize);
    cudaMalloc(&d_F, filterSize);
    cudaMalloc(&d_P, imageSize);

    // Copy data from host to device
    cudaMemcpy(d_N, h_N, imageSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_F, h_F, filterSize, cudaMemcpyHostToDevice);

    // Define block and grid sizes if not provided
    if (gridDim.x == 0 && gridDim.y == 0) {
        gridDim = dim3((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y);
    }

    // Measure execution time of basic kernel
    auto start = std::chrono::high_resolution_clock::now();
    convolution_2d_basic_kernel << <gridDim, blockDim >> > (d_N, d_F, d_P, r, width, height);
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    std::cout << "Execution time of basic kernel: " << duration.count() << " seconds\n";


    // Copy the result back to host
    cudaMemcpy(h_P, d_P, imageSize, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_N);
    cudaFree(d_F);
    cudaFree(d_P);

    // Print the output image
    std::cout << "Output Image Basic Kernel Convolution:\n" << std::endl;
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            std::cout << h_P[i * width + j] << " ";
        }
        std::cout << std::endl;
    }

	// ********************************************************************************************************************
    // CONSTANT KERNEL
    for (int i = 0; i < 25; ++i) {
        default_h_N[i] = fixed_value;
    }
    for (int i = 0; i < 9; ++i) {
        default_h_F[i] = fixed_value;
    }

    if (h_N == nullptr) h_N = default_h_N;
    if (h_F == nullptr) h_F = default_h_F;
    if (h_P == nullptr) h_P = default_h_P;

    // Allocate device memory
    //float* d_N, * d_F, * d_P;
    //size_t imageSize = width * height * sizeof(float);
    //size_t filterSize = (2 * r + 1) * (2 * r + 1) * sizeof(float);

    cudaMalloc(&d_N, imageSize);
    cudaMalloc(&d_F, filterSize);
    cudaMalloc(&d_P, imageSize);

    // Copy data from host to device
    cudaMemcpy(d_N, h_N, imageSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_F, h_F, filterSize, cudaMemcpyHostToDevice);

    // Define block and grid sizes if not provided
    if (gridDim.x == 0 && gridDim.y == 0) {
        gridDim = dim3((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y);
    }

    //cudaMemcpyToSymbol(F, F_h, (2 * FILTER_RADIUS + 1)* (2 * FILTER_RADIUS + 1) * sizeof(float));  // copy filter to constant memory

    // Measure execution time of constant memory kernel
    start = std::chrono::high_resolution_clock::now();
    convolution_2d_const_mem_kernel << <gridDim, blockDim >> > (d_N, d_P, r, width, height);
    cudaDeviceSynchronize();
    end = std::chrono::high_resolution_clock::now();
    duration = end - start;
    std::cout << "Execution time of constant memory kernel: " << duration.count() << " seconds\n";


    // Copy the result back to host
    cudaMemcpy(h_P, d_P, imageSize, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_N);
    cudaFree(d_F);
    cudaFree(d_P);

    // Print the output image
    std::cout << "Output Image COnstant Memory Kernel Convolution:\n" << std::endl;
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            std::cout << h_P[i * width + j] << " ";
        }
        std::cout << std::endl;
    }
    return 0;
}
