#pragma once
#include "tensor_utils.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cmath>
#include <stdio.h>
#include <stdlib.h>

void* Loadtensortogpu(const void* host_tensor, size_t bytes) {
    void* device_tensor = nullptr;
    cudaError_t status = cudaMalloc(&device_tensor, bytes);
    if (status != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!\n");
        return nullptr;
    }
    status = cudaMemcpy(device_tensor, host_tensor, bytes, cudaMemcpyHostToDevice);
    if (status != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy to device failed!\n");
        cudaFree(device_tensor);
        return nullptr;
    }
    return device_tensor;
}

void Unloadtensortogpu(void* device_tensor, void* host_tensor, size_t bytes) {
    cudaError_t status = cudaMemcpy(host_tensor, device_tensor, bytes, cudaMemcpyDeviceToHost);
    if (status != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy to host failed!\n");
    }
    cudaFree(device_tensor);
}

// Moltiplicazione matriciale CPU (naive, row-major)
void matrixMulCPU(float* C, float* A, float* B, int width) {
    for (int i = 0; i < width; ++i) {
        for (int j = 0; j < width; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < width; ++k) {
                sum += A[i * width + k] * B[k * width + j];
            }
            C[i * width + j] = sum;
        }
    }   
}
/*
// Confronto tra risultati GPU e CPU (con tolleranza)
bool checkResult(float* GPU, float* CPU, int width) {
    int size = width * width;
	float eps = 1e-5; // Tolleranza per il confronto
    for (int i = 0; i < size; ++i) {
        if (std::fabs(GPU[i] - CPU[i]) > eps) {
            printf("Mismatch at %d: GPU=%f, CPU=%f\n", i, GPU[i], CPU[i]);
            return false;
        }
    }
    return true;
}
*/