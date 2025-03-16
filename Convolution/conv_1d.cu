// Perform 1D convolution on CPU and GPU and compare the execution time and speedup

#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <chrono> // For measuring execution time
#include <cassert>

__global__ void warmup_kernel() {
    // Simple empty kernel just to wake up the GPU
}

extern "C" void gpu_warmup() {
    cudaError_t err;

    // Launch a small warmup kernel
    warmup_kernel << <1, 1 >> > ();

    // Synchronize to ensure execution completes
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "CUDA warmup failed: " << cudaGetErrorString(err) << std::endl;
    }
}

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

void conv1d_cpu_padded(const float* input, const float* kernel, float* output, int input_size, int kernel_size) {
    int pad = kernel_size / 2; // Zero-padding size on both ends
    int output_size = input_size;  // Output size is the same as input size with padding

    for (int i = 0; i < output_size; i++) {
        float sum = 0.0f;

        // Convolution loop, taking padding into account
        for (int j = 0; j < kernel_size; j++) {
            int input_idx = i + j - pad;  // Shifted index to account for padding

            // Handle boundary conditions: pad with zeros if out of bounds
            if (input_idx >= 0 && input_idx < input_size) {
                sum += input[input_idx] * kernel[j];
            }
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

// CUDA kernel for 1D convolution with zero padding
__global__ void conv1d_kernel_padded(const float* input, const float* kernel, float* output, size_t N, size_t K) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int pad = K / 2; // Zero-padding size on both ends

    if (i < N) {
        float sum = 0.0f;
        for (int j = 0; j < K; j++) {
            int input_idx = i + j - pad; // Shifted index to account for padding
            if (input_idx >= 0 && input_idx < N) {
                sum += input[input_idx] * kernel[j];
            }
        }
        output[i] = sum;
    }
}

#define BLOCK_SIZE 256  // Number of threads per block

// Optimized CUDA kernel for 1D convolution with zero padding using shared memory
// The second function (conv1d_kernel_padded_shared) is better than the first (conv1d_kernel_padded) because it optimizes memory access using shared memory, reducing global memory transactions and improving performance.
// Each thread fetches input values directly from global memory -> function above
// Uses shared memory (shared_input[]), which is much faster (~100x lower latency) than global memory.-> function below
__global__ void conv1d_kernel_padded_shared(const float* input, const float* kernel, float* output, size_t N, size_t K) {
    extern __shared__ float shared_input[];  // Dynamic shared memory allocation

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int pad = K / 2;  // Zero-padding size on both ends
    int tid = threadIdx.x;

    // Global memory index for input
    int global_idx = i - pad;

    // Load input into shared memory (zero if out of bounds)
    shared_input[tid] = (global_idx >= 0 && global_idx < N) ? input[global_idx] : 0.0f;

    // Load extra elements for boundary handling (each block loads extra K-1 elements)
    if (tid < K - 1) {
        int extra_idx = global_idx + BLOCK_SIZE;
        shared_input[tid + BLOCK_SIZE] = (extra_idx >= 0 && extra_idx < N) ? input[extra_idx] : 0.0f;
    }

    __syncthreads();  // Synchronize to ensure all threads load their data

    // Perform convolution only for valid output indices
    if (i < N) {
        float sum = 0.0f;

        // Use loop unrolling for better performance (when K is small)
        #pragma unroll
        for (int j = 0; j < K; j++) {
            sum += shared_input[tid + j] * kernel[j];
        }

        output[i] = sum;
    }
}

// Wrapper function to launch the CUDA kernel
extern "C" void solution(const float* A, const float* B, float* C, size_t N, size_t K) {
    assert(A != nullptr && "Error: Input pointer A is null");
    assert(B != nullptr && "Error: Kernel pointer B is null");
    assert(C != nullptr && "Error: Output pointer C is null");

    // Check that input, kernel, and output sizes are consistent
    assert(N > 0 && "Error: Input size N must be greater than 0");
    assert(K > 0 && "Error: Kernel size K must be greater than 0");
    assert(N >= K && "Error: Input size N must be at least kernel size K");

    // Configure CUDA kernel launch parameters
    int threads_per_block = 256;
    int blocks_per_grid = (N + threads_per_block - 1) / threads_per_block;

    // Launch CUDA kernel
    conv1d_kernel_padded << <blocks_per_grid, threads_per_block >> > (A, B, C, N, K);

	assert(A.size() == B.size() && B.size() == C.size() && "Error: Input, kernel, and output sizes must match");

    // Synchronize to ensure kernel execution completes
    cudaDeviceSynchronize();
}

// Wrapper function to launch the optimized CUDA kernel
extern "C" void solution_shared(const float* A, const float* B, float* C, size_t N, size_t K) {
    assert(A != nullptr && "Error: Input pointer A is null");
    assert(B != nullptr && "Error: Kernel pointer B is null");
    assert(C != nullptr && "Error: Output pointer C is null");

    // Check that input, kernel, and output sizes are consistent
    assert(N > 0 && "Error: Input size N must be greater than 0");
    assert(K > 0 && "Error: Kernel size K must be greater than 0");
    assert(N >= K && "Error: Input size N must be at least kernel size K");
    int threads_per_block = BLOCK_SIZE;
    int blocks_per_grid = (N + threads_per_block - 1) / threads_per_block;

    // Launch optimized kernel with shared memory allocation
    size_t shared_mem_size = (BLOCK_SIZE + K - 1) * sizeof(float);
    conv1d_kernel_padded_shared << <blocks_per_grid, threads_per_block, shared_mem_size >> > (A, B, C, N, K);

    assert(A.size() == B.size() && B.size() == C.size() && "Error: Input, kernel, and output sizes must match");

    cudaDeviceSynchronize();
}

int conv_1d_host() {
    // Define input size and kernel size
	gpu_warmup();
    int input_size = 1000; // Large input for performance testing
    int kernel_size = 3;
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

	// ============================
	// Zero-padding NOT SHARED
	// ============================

    gpu_warmup();

	// Allocate device memory
	float* d_input_padded, * d_kernel_padded, * d_output_padded;
	cudaMalloc((void**)&d_input_padded, input_size * sizeof(float));
	cudaMalloc((void**)&d_kernel_padded, kernel_size * sizeof(float));
	cudaMalloc((void**)&d_output_padded, output_size * sizeof(float));

	// Copy data from host to device
	cudaMemcpy(d_input_padded, h_input.data(), input_size * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_kernel_padded, h_kernel.data(), kernel_size * sizeof(float), cudaMemcpyHostToDevice);

	// GPU Timing Start
	cudaDeviceSynchronize(); // Ensure GPU is ready
	auto start_gpu_padded = std::chrono::high_resolution_clock::now();

	// Launch kernel
	solution(d_input_padded, d_kernel_padded, d_output_padded, input_size, kernel_size);

    // GPU Timing End
    cudaDeviceSynchronize();

    auto end_gpu_padded = std::chrono::high_resolution_clock::now();

    // Compute GPU execution time
    double gpu_time_padded = std::chrono::duration<double, std::milli>(end_gpu_padded - start_gpu_padded).count();

    std::cout << "GPU Time with Zero-padding NOT SHARED: " << gpu_time_padded << " ms\n";

    // Free device memory
    cudaFree(d_input_padded);
    cudaFree(d_kernel_padded);
    cudaFree(d_output_padded);

    // ============================
    // Zero-padding SHARED
    // ============================

    gpu_warmup();

    cudaMalloc((void**)&d_input_padded, input_size * sizeof(float));
    cudaMalloc((void**)&d_kernel_padded, kernel_size * sizeof(float));
	cudaMalloc((void**)&d_output_padded, output_size * sizeof(float));

    // Copy data from host to device
    cudaMemcpy(d_input_padded, h_input.data(), input_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel_padded, h_kernel.data(), kernel_size * sizeof(float), cudaMemcpyHostToDevice);

    // GPU Timing Start
    cudaDeviceSynchronize(); // Ensure GPU is ready
    start_gpu_padded = std::chrono::high_resolution_clock::now();

    solution_shared(d_input_padded, d_kernel_padded, d_output_padded, input_size, kernel_size);
        
	// GPU Timing End
	cudaDeviceSynchronize();

	end_gpu_padded = std::chrono::high_resolution_clock::now();
        
	// Copy result back to host
	cudaMemcpy(h_output_gpu.data(), d_output_padded, output_size * sizeof(float), cudaMemcpyDeviceToHost);

	// Compute GPU execution time
	gpu_time_padded = std::chrono::duration<double, std::milli>(end_gpu_padded - start_gpu_padded).count();

	std::cout << "GPU Time with Zero-padding SHARED: " << gpu_time_padded << " ms\n";
    
    // Free device memory
	cudaFree(d_input_padded);
	cudaFree(d_kernel_padded);
	cudaFree(d_output_padded);

    return 0;
}

int conv_1d_padded() {
    // ============================
    // Zero-padding NOT SHARED
    // ============================

    // Define input size and kernel size
    int input_size = 1000; // Large input for performance testing
    int kernel_size = 3;
    int output_size = input_size;

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
    conv1d_cpu_padded(h_input.data(), h_kernel.data(), h_output_cpu.data(), input_size, kernel_size);
    auto end_cpu = std::chrono::high_resolution_clock::now();
    double cpu_time = std::chrono::duration<double, std::milli>(end_cpu - start_cpu).count();

    std::cout << "CPU Time: " << cpu_time << " ms\n";

    gpu_warmup();

    // Allocate device memory
    float* d_input_padded, * d_kernel_padded, * d_output_padded;
    cudaMalloc((void**)&d_input_padded, input_size * sizeof(float));
    cudaMalloc((void**)&d_kernel_padded, kernel_size * sizeof(float));
    cudaMalloc((void**)&d_output_padded, output_size * sizeof(float));

    // Copy data from host to device
    cudaMemcpy(d_input_padded, h_input.data(), input_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel_padded, h_kernel.data(), kernel_size * sizeof(float), cudaMemcpyHostToDevice);

    // GPU Timing Start
    cudaDeviceSynchronize(); // Ensure GPU is ready
    auto start_gpu_padded = std::chrono::high_resolution_clock::now();

    // Launch kernel
    solution(d_input_padded, d_kernel_padded, d_output_padded, input_size, kernel_size);

    // GPU Timing End
    cudaDeviceSynchronize();

    auto end_gpu_padded = std::chrono::high_resolution_clock::now();

    // Copy the output results to Host
    cudaMemcpy(h_output_gpu.data(), d_output_padded, input_size * sizeof(float), cudaMemcpyDeviceToHost);

    // Compute GPU execution time
    double gpu_time_padded = std::chrono::duration<double, std::milli>(end_gpu_padded - start_gpu_padded).count();
    // Check size match

    std::cout << "GPU Time with Zero-padding NOT SHARED: " << gpu_time_padded << " ms\n";

	for (int i = 0; i < output_size; i++) {
		assert(fabs(h_output_cpu[i] - h_output_gpu[i]) < 1e-5f && "Error: GPU output does not match CPU output");
		//std::cout << "Output[" << i << "] = " << h_output_gpu[i] << std::endl;
	}

    for (int i = 0; i < output_size; i++) {
        if (fabs(h_output_cpu[i] - h_output_gpu[i]) >= 1e-5f) {  // Increased precision tolerance
            std::cerr << "Error: GPU output does not match CPU output at index " << i
                << " (CPU: " << h_output_cpu[i] << ", GPU: " << h_output_gpu[i] << ")\n";
            exit(EXIT_FAILURE);  // Stop execution
        }
        // std::cout << "Output[" << i << "] = " << h_output_gpu[i] << std::endl;
    }

    // Free device memory
    cudaFree(d_input_padded);
    cudaFree(d_kernel_padded);
    cudaFree(d_output_padded);
}

int conv_1d_padded_shared() {
    // ============================
    // Zero-padding SHARED
    // ============================

    // Define input size and kernel size
    int input_size = 1000; // Large input for performance testing
    int kernel_size = 3;
    int output_size = input_size;

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
    conv1d_cpu_padded(h_input.data(), h_kernel.data(), h_output_cpu.data(), input_size, kernel_size);
    auto end_cpu = std::chrono::high_resolution_clock::now();
    double cpu_time = std::chrono::duration<double, std::milli>(end_cpu - start_cpu).count();

    std::cout << "CPU Time: " << cpu_time << " ms\n";

    gpu_warmup();

    // Allocate device memory
    float* d_input_padded, * d_kernel_padded, * d_output_padded;
    cudaMalloc((void**)&d_input_padded, input_size * sizeof(float));
    cudaMalloc((void**)&d_kernel_padded, kernel_size * sizeof(float));
    cudaMalloc((void**)&d_output_padded, output_size * sizeof(float));

    // Copy data from host to device
    cudaMemcpy(d_input_padded, h_input.data(), input_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel_padded, h_kernel.data(), kernel_size * sizeof(float), cudaMemcpyHostToDevice);

    // GPU Timing Start
    cudaDeviceSynchronize(); // Ensure GPU is ready
    auto start_gpu_padded = std::chrono::high_resolution_clock::now();

    // Launch kernel
    solution_shared(d_input_padded, d_kernel_padded, d_output_padded, input_size, kernel_size);

    // GPU Timing End
    cudaDeviceSynchronize();

    auto end_gpu_padded = std::chrono::high_resolution_clock::now();

    // Copy the output results to Host
    cudaMemcpy(h_output_gpu.data(), d_output_padded, input_size * sizeof(float), cudaMemcpyDeviceToHost);

    // Compute GPU execution time
    double gpu_time_padded = std::chrono::duration<double, std::milli>(end_gpu_padded - start_gpu_padded).count();
    // Check size match

    std::cout << "GPU Time with Zero-padding SHARED: " << gpu_time_padded << " ms\n";

    for (int i = 0; i < output_size; i++) {
        assert(fabs(h_output_cpu[i] - h_output_gpu[i]) < 1e-5f && "Error: GPU output does not match CPU output");
        //std::cout << "Output[" << i << "] = " << h_output_gpu[i] << std::endl;
    }

    for (int i = 0; i < output_size; i++) {
        if (fabs(h_output_cpu[i] - h_output_gpu[i]) >= 1e-5f) {  // Increased precision tolerance
            std::cerr << "Error: GPU output does not match CPU output at index " << i
                << " (CPU: " << h_output_cpu[i] << ", GPU: " << h_output_gpu[i] << ")\n";
            exit(EXIT_FAILURE);  // Stop execution
        }
        // std::cout << "Output[" << i << "] = " << h_output_gpu[i] << std::endl;
    }

    // Free device memory
    cudaFree(d_input_padded);
    cudaFree(d_kernel_padded);
    cudaFree(d_output_padded);
}