// Perform 1D convolution on CPU and GPU and compare the execution time and speedup

#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <chrono> // For measuring execution time

// Function to perform 1D convolution on CPU
void conv1d_cpu(const float* input, const float* kernel, float* output, int input_size, int kernel_size) {
    int output_size = input_size - kernel_size + 1;

    for (int i = 0; i < output_size; i++) {
        float sum = 0.0f;
        for (int j = 0; j < kernel_size; j++) {
            sum += input[i + j] * kernel[j];
        }
        output[i] = sum;
    }
}

// CUDA kernel to perform 1D convolution on GPU
__global__
void conv1d_kernel(float* input, float* kernel, float* output, int input_size, int kernel_size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int output_size = input_size - kernel_size + 1;

    if (i < output_size) {
        float sum = 0.0f;
        for (int j = 0; j < kernel_size; j++) {
            sum += input[i + j] * kernel[j];
        }
        output[i] = sum;
    }
}

int conv_1d_host() {
    // Define input size and kernel size
    int input_size = 1000000; // Large input for performance testing
    int kernel_size = 5;
    int output_size = input_size - kernel_size + 1;

    // Allocate and initialize host memory (CPU)
    std::vector<float> h_input(input_size);
    std::vector<float> h_kernel(kernel_size);
    std::vector<float> h_output_cpu(output_size, 0.0f);
    std::vector<float> h_output_gpu(output_size, 0.0f);

    // Initialize input and kernel with random values
    for (int i = 0; i < input_size; i++) h_input[i] = static_cast<float>(rand()) / RAND_MAX;
    for (int i = 0; i < kernel_size; i++) h_kernel[i] = static_cast<float>(rand()) / RAND_MAX;

    // ============================
    // CPU Execution & Timing
    // ============================
    auto start_cpu = std::chrono::high_resolution_clock::now();
    conv1d_cpu(h_input.data(), h_kernel.data(), h_output_cpu.data(), input_size, kernel_size);
    auto end_cpu = std::chrono::high_resolution_clock::now();
    double cpu_time = std::chrono::duration<double, std::milli>(end_cpu - start_cpu).count();

    std::cout << "CPU Time: " << cpu_time << " ms\n";

    // ============================
    // GPU Execution & Timing
    // ============================

    // Allocate device memory
    float* d_input, * d_kernel, * d_output;
    cudaMalloc((void**)&d_input, input_size * sizeof(float));
    cudaMalloc((void**)&d_kernel, kernel_size * sizeof(float));
    cudaMalloc((void**)&d_output, output_size * sizeof(float));

    // Copy data from host to device
    cudaMemcpy(d_input, h_input.data(), input_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, h_kernel.data(), kernel_size * sizeof(float), cudaMemcpyHostToDevice);

    // Define kernel execution configuration
    int blockSize = 256;
    int gridSize = (output_size + blockSize - 1) / blockSize;

    // GPU Timing Start
    cudaDeviceSynchronize(); // Ensure GPU is ready
    auto start_gpu = std::chrono::high_resolution_clock::now();

    // Launch kernel
    conv1d_kernel << <gridSize, blockSize >> > (d_input, d_kernel, d_output, input_size, kernel_size);

    // GPU Timing End
    cudaDeviceSynchronize();
    auto end_gpu = std::chrono::high_resolution_clock::now();

    // Copy result back to host
    cudaMemcpy(h_output_gpu.data(), d_output, output_size * sizeof(float), cudaMemcpyDeviceToHost);

    // Compute GPU execution time
    double gpu_time = std::chrono::duration<double, std::milli>(end_gpu - start_gpu).count();

    std::cout << "GPU Time: " << gpu_time << " ms\n";

    // ============================
    // Speedup Calculation
    // ============================
    double speedup = cpu_time / gpu_time;
    std::cout << "Speedup (CPU/GPU): " << speedup << "x\n";

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_kernel);
    cudaFree(d_output);

    return 0;
}
