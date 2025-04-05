#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>
#include <iostream>
// #include "main_header.cuh"  -> don't need 
#include "max_pooling_1d.cu"

int deive_proprieties() {
	int deviceCount;
	cudaGetDeviceCount(&deviceCount);
	std::cout << "Number of CUDA devices: " << deviceCount << std::endl;
	for (int i = 0; i < deviceCount; ++i) {
		cudaDeviceProp prop;
		cudaGetDeviceProperties(&prop, i);
		std::cout << "Device " << i << ": " << prop.name << std::endl;
		std::cout << "  Total global memory: " << prop.totalGlobalMem / (1024 * 1024) << " MB" << std::endl;
		std::cout << "  Multiprocessor count: " << prop.multiProcessorCount << std::endl;
		std::cout << "  Max threads per block: " << prop.maxThreadsPerBlock << std::endl;
	}
	return 0;
}

/*
static void BM_CudaDeviceProperties(benchmark::State& state) {
    for (auto _ : state) {
        int deviceCount;
        cudaGetDeviceCount(&deviceCount);
        for (int i = 0; i < deviceCount; ++i) {
            cudaDeviceProp prop;
            cudaGetDeviceProperties(&prop, i);
        }
    }
}

BENCHMARK(BM_CudaDeviceProperties);
*/

#include <gtest/gtest.h>
#include <cmath>
#include <cstdlib>
#include <vector>
#include <limits>

// Helper function to compute output length
int compute_output_length(size_t H, int kernel_size, int stride, int padding, int dilation) {
    return ((H + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1;
}

// CPU implementation of 1D max pooling for verification
std::vector<float> cpu_max_pooling_1d(const std::vector<float>& input, int kernel_size, int stride,
    int padding, int dilation) {
    int H = input.size();
    int H_out = compute_output_length(H, kernel_size, stride, padding, dilation);
    std::vector<float> output(H_out, -INFINITY);
    for (int i = 0; i < H_out; ++i) {
        float max_val = -INFINITY;
        for (int m = 0; m < kernel_size; ++m) {
            int index = stride * i + dilation * m - padding;
            float value = (index < 0 || index >= H) ? -INFINITY : input[index];
            max_val = fmaxf(max_val, value);
        }
        output[i] = max_val;
    }
    return output;
}

// A basic test case for the CUDA max pooling implementation
TEST(MaxPooling1DTest, BasicFunctionality) {
    // Define parameters for the test
    const size_t H = 5;
    const int kernel_size = 3;
    const int stride = 1;
    const int padding = 1;
    const int dilation = 1;

    // Prepare a known input vector
    std::vector<float> h_input = { 0.1f, 0.5f, 0.3f, 0.8f, 0.2f };

    // Compute expected output using the CPU version
    std::vector<float> expected_output = cpu_max_pooling_1d(h_input, kernel_size, stride, padding, dilation);
    int H_out = expected_output.size();

    // Allocate device memory for input and output.
    float* d_input, * d_output;
    cudaMalloc(&d_input, H * sizeof(float));
    // Allocate output buffer large enough to hold the computed H_out values.
    cudaMalloc(&d_output, H_out * sizeof(float));

    // Copy input to device.
    cudaMemcpy(d_input, h_input.data(), H * sizeof(float), cudaMemcpyHostToDevice);

    // Call the CUDA max pooling function.
    solution_max_pooling_1d(d_input, kernel_size, stride, padding, dilation, d_output, H);

    // Copy the output from device to host.
    std::vector<float> h_output(H_out, 0);
    cudaMemcpy(h_output.data(), d_output, H_out * sizeof(float), cudaMemcpyDeviceToHost);

    // Compare the expected and the actual output.
    for (int i = 0; i < H_out; ++i) {
        EXPECT_NEAR(h_output[i], expected_output[i], 1e-5)
            << "Mismatch at index " << i << ": expected " << expected_output[i]
            << ", got " << h_output[i];
    }

    // Free device memory.
    cudaFree(d_input);
    cudaFree(d_output);
}

// Additional test cases can be added below to test different parameter sets or input data.

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    // Optionally initialize CUDA if needed.
    cudaSetDevice(0);
    return RUN_ALL_TESTS();
}

/* GBENCHMARK_MAIN 
int main(int argc, char** argv) {
    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();
    return 0;
}
*/
