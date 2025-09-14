#pragma once
#include <memory>
#include <vector>
#include <random>
#include <cuda_runtime.h>
#include <cuda_fp16.h> // aggiunto per __half
#include <cuda_bf16.h> // aggiunto per __nv_bfloat16
#include <cstdlib>

/*
Example usage:

// 1D tensor of 100 ints between 0 and 10
auto tensor = generatetensors<int>(100, 0, 10);

// 2D tensor (flattened) of 5x5 floats between -1.0 and 1.0
size_t rows = 5, cols = 5;
auto tensor2d = generatetensors<float>(rows * cols, -1.0f, 1.0f);

// Load tensor to GPU
void* device_ptr = Loadtensortogpu(tensor->data(), tensor->size() * sizeof(int));

// Unload tensor from GPU back to host and free device memory
Unloadtensortogpu(device_ptr, tensor->data(), tensor->size() * sizeof(int));
*/

// Template function to generate a tensor of arbitrary dimension and type
// Usage: auto tensor = generatetensors<T>(dim, dim1, dim2, ..., lower, upper);
template<typename T>
std::unique_ptr<std::vector<T>> generatetensors(size_t total_size, T lower, T upper) {
    auto tensor = std::make_unique<std::vector<T>>(total_size);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<T> dis(lower, upper);
    for (size_t i = 0; i < total_size; ++i) {
        (*tensor)[i] = dis(gen);
    }
    return tensor;
}

// Loads tensor to GPU, returns device pointer
void* Loadtensortogpu(const void* host_tensor, size_t bytes);

// Unloads tensor from GPU to host, frees device memory
void Unloadtensortogpu(void* device_tensor, void* host_tensor, size_t bytes);

void matrixMulCPU(float* C, float* A, float* B, int width);

// Generic template: fill float[] and converted[]
template<typename T, typename ConvertFunc>
inline void randomInit(float* data, T* data_converted, int size, ConvertFunc convert) {
    for (int i = 0; i < size; ++i) {
        float v = rand() / static_cast<float>(RAND_MAX);
        data[i] = v;
        data_converted[i] = convert(v);
    }
}

// Special case: only float (no conversion needed)
inline void randomInit(float* data, int size) {
    randomInit<float>(data, data, size, [](float v) { return v; });
}

// Wrapper for __half
inline void randomInit(float* data, __half* data_half, int size) {
    randomInit<__half>(data, data_half, size, __float2half);
}

// Wrapper for __nv_bfloat16
inline void randomInit(float* data, __nv_bfloat16* data_bf16, int size) {
    randomInit<__nv_bfloat16>(data, data_bf16, size, __float2bfloat16);
}

// Prototypes for functions implemented in matmulTensor.cu
int thrustensor();
int thrustcudacore();
int thrustTensorBfloat16();