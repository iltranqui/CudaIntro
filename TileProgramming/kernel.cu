#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cuda_device_runtime_api.h>
#include <device_launch_parameters.h>
#include <iostream>
#define CCCL_IGNORE_DEPRECATED_CPP_DIALECT
#include <cccl/thrust/device_vector.h>  // Resolves to cccl/thrust/
#include <thrust/device_vector.h>
#include <cccl/thrust/transform.h>
#include <thrust/transform.h>
#include <cccl/thrust/functional.h>
#include <thrust/functional.h>
#include "tensor_utils.h"

// Diagnostic kernel to print C++ version
__global__ void printVersion() {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("__cplusplus=%ld\n", (long)__cplusplus);
        printf("_MSVC_LANG=%ld\n", (long)_MSVC_LANG);
        if (__cplusplus == 199711L) {
            printf("C++98/03 detected (199711L)\n");
        }
        else if (__cplusplus == 201103L) {
            printf("C++11 detected (201103L)\n");
        }
        else if (__cplusplus == 201402L) {
            printf("C++14 detected (201402L)\n");
        }
        else if (__cplusplus == 201703L) {
            printf("C++17 detected (201703L)\n");
        }
        else if (__cplusplus == 202002L) {
            printf("C++20 detected (202002L)\n");
        }
        else {
            printf("Unknown or future C++ version: %ld\n", (long)__cplusplus);
        }
    }
}

void thrustexample() {
    thrust::device_vector<float> a(2, 1.0f); // 2 elements, value 1.0
    thrust::device_vector<float> b(2, 2.0f); // 2 elements, value 2.0
    thrust::device_vector<float> c(2);
    thrust::transform(a.begin(), a.end(), b.begin(), c.begin(), thrust::plus<float>());
    std::cout << "Result: " << c[0] << " " << c[1] << std::endl;   // Expected: 3.0 3.0
}

// Tile size (block size for shared memory)
#define TILE_WIDTH 32

// Matrix multiplication kernel using shared memory tiling
__global__ void matrixMulTiled(float* C, float* A, float* B, int width) {
    // Shared memory for tiles
    __shared__ float sA[TILE_WIDTH][TILE_WIDTH];
    __shared__ float sB[TILE_WIDTH][TILE_WIDTH];

    // Global indices
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int idx;

    // Accumulator for partial sum
    float sum = 0.0f;

    // Iterate over tiles along the k-dimension
    for (int t = 0; t < (width / TILE_WIDTH); ++t) {
        // Load tile of A into shared memory (row-major for A)
        if (row < width && (t * TILE_WIDTH + threadIdx.x) < width) {
            sA[threadIdx.y][threadIdx.x] = A[row * width + t * TILE_WIDTH + threadIdx.x];
        }
        else {
            sA[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // Load tile of B into shared memory (column-major access)
        if (col < width && (t * TILE_WIDTH + threadIdx.y) < width) {
            sB[threadIdx.y][threadIdx.x] = B[(t * TILE_WIDTH + threadIdx.y) * width + col];
        }
        else {
            sB[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // Synchronize to ensure tiles are loaded
        __syncthreads();

        // Compute partial dot product for this tile
        for (int k = 0; k < TILE_WIDTH; ++k) {
            sum += sA[threadIdx.y][k] * sB[k][threadIdx.x];
        }

        // Synchronize before loading next tile
        __syncthreads();
    }

    // Write result to global memory
    if (row < width && col < width) {
        idx = row * width + col;
        C[idx] = sum;
    }
}


// Confronto tra risultati GPU e CPU (con tolleranza)
inline bool checkResult(const float* GPU, const float* CPU, int width, float eps = 1e-5f) {
    int size = width * width;
    for (int i = 0; i < size; ++i) {
        if (std::fabs(GPU[i] - CPU[i]) > eps) {
            printf("Mismatch at %d: GPU=%f, CPU=%f\n", i, GPU[i], CPU[i]);
            return false;
        }
    }
    return true;
}

int main() {

    int width = 1024;  // Matrix size (N x N)
    int size = width * width * sizeof(float);
    float* h_A = (float*)malloc(size);
    float* h_B = (float*)malloc(size);
    float* h_C_GPU = (float*)malloc(size);
    float* h_C_CPU = (float*)malloc(size);

    // Initialize input matrices
    randomInit(h_A, width * width);
    randomInit(h_B, width * width);

    // Allocate GPU memory
    float* d_A, * d_B, * d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    // Copy to GPU
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    printVersion << <1, 1 >> > ();
    cudaDeviceSynchronize();
    thrustexample();

    // Launch kernel: grid/block dims based on tile size
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 dimGrid(width / TILE_WIDTH, width / TILE_WIDTH);
    matrixMulTiled << <dimGrid, dimBlock >> > (d_C, d_A, d_B, width);

    // Copy result back
    cudaMemcpy(h_C_GPU, d_C, size, cudaMemcpyDeviceToHost);

    // Verify with CPU
    matrixMulCPU(h_C_CPU, h_A, h_B, width);
    if (checkResult(h_C_GPU, h_C_CPU, width)) {
        printf("Matrix multiplication successful! (Tiled GPU matches CPU)\n");
    }
    else {
        printf("Error: Results do not match!\n");
    }

    // Cleanup
    free(h_A); free(h_B); free(h_C_GPU); free(h_C_CPU);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);

    // Performance note: For timing, use cudaEventRecord() around kernel launch.

    int r1 = thrustensor();
    int r2 = thrustcudacore();
    int r3 = thrustTensorBfloat16();
    return 0;
}