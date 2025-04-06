#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <tuple>
#include <cmath>     // For floor, fmaxf (though not needed for avg), INFINITY (for cpu init)
#include <algorithm> // For std::max (used in H_out calc)
#include <limits>    // For std::numeric_limits
#include <cstdlib>   // For size_t
#include <cstdio>    // For printf formatting
#include <numeric>   // For std::accumulate (useful for CPU version)

// Include the file containing the CUDA kernel implementation for average pooling
// This file should define:
// extern "C" void solution_average_pooling_1d(const float* input, int kernel_size, int stride, int padding, float* output, size_t H);
// and contain the CORRECT average pooling kernel logic.
#include "average_pool_1d.cu" // <<< IMPORTANT: Assumes this contains the correct avg pool solution
#include <gtest/gtest.h> // Include Google Test header

// Helper function to compute output length for Average Pooling (based on provided formula)
// H_out = floor((H + 2*P - k) / S) + 1
int compute_output_length_avg(size_t H, int kernel_size, int stride, int padding) {
    if (kernel_size <= 0 || stride <= 0 || H == 0) {
        return 0;
    }
    // Ensure kernel_size doesn't exceed padded input size in a way that makes numerator negative before floor
    // Although mathematically floor handles it, it clarifies intent.
    double numerator = static_cast<double>(H) + 2.0 * padding - static_cast<double>(kernel_size);
    // If numerator is negative, the window is larger than the padded input extent, output size is 0
    if (numerator < 0) {
        // Check edge case: if stride allows exactly one placement starting before input
        // Example H=1, k=3, p=0, s=1 => num = 1+0-3 = -2 => floor(-2/1)+1 = -1 -> 0. Correct.
        // Example H=1, k=3, p=1, s=1 => num = 1+2-3 = 0 => floor(0/1)+1 = 1. Correct.
    }

    int H_out = static_cast<int>(std::floor(numerator / static_cast<double>(stride))) + 1;
    return std::max(0, H_out); // Ensure output size is not negative
}

// CPU implementation of 1D average pooling for verification
std::vector<float> cpu_average_pooling_1d(const std::vector<float>& input, int kernel_size, int stride, int padding) {
    int H = static_cast<int>(input.size());
    int H_out = compute_output_length_avg(H, kernel_size, stride, padding);

    if (H_out <= 0) {
        return {}; // Return empty vector if output size is non-positive
    }

    std::vector<float> output(H_out);

    for (int i = 0; i < H_out; ++i) {
        float current_sum = 0.0f;
        // Calculate the starting position of the window in the potentially padded input space
        int start_index_in_padded = i * stride - padding;

        // Iterate over the kernel window
        for (int m = 0; m < kernel_size; ++m) {
            // Calculate the index in the *original* input tensor H
            int input_index = start_index_in_padded + m;

            // Check if the index is within the valid bounds of the original input H
            if (input_index >= 0 && input_index < H) {
                current_sum += input[input_index];
            }
            // Otherwise, the value is implicitly zero (padding), so we add nothing to the sum.
        }

        // Divide by the kernel size k (always), handle potential division by zero
        if (kernel_size > 0) {
            output[i] = current_sum / static_cast<float>(kernel_size);
        }
        else {
            output[i] = 0.0f; // Or handle as error / NaN
        }
    }
    return output;
}


// --- Parameterized GTest Fixture for Correctness ---
// Parameters: <InputSize H, kernel_size, stride, padding>
class AveragePooling1DParamTest : public ::testing::TestWithParam<std::tuple<size_t, int, int, int>> {
protected:
    size_t H;
    int kernel_size;
    int stride;
    int padding;
    std::vector<float> h_input;
    std::vector<float> expected_output;
    float* d_input = nullptr;
    float* d_output = nullptr;
    int H_out;

    void SetUp() override {
        std::tie(H, kernel_size, stride, padding) = GetParam();

        // Prepare input data
        h_input.resize(H);
        if (H > 0) {
            for (size_t i = 0; i < H; ++i) {
                // Use a simple predictable pattern for easier debugging if needed
                h_input[i] = static_cast<float>(i % 10 + 1) * 0.5f;
            }
        }

        // Compute expected output using the CPU version
        expected_output = cpu_average_pooling_1d(h_input, kernel_size, stride, padding);
        H_out = expected_output.size();

        // Allocate CUDA memory only if valid computation is possible
        if (H > 0 && H_out > 0) {
            cudaError_t err;
            err = cudaMalloc(&d_input, H * sizeof(float));
            ASSERT_EQ(err, cudaSuccess) << "Failed to allocate device input memory (d_input)";
            err = cudaMalloc(&d_output, H_out * sizeof(float));
            if (err != cudaSuccess) { cudaFree(d_input); d_input = nullptr; }
            ASSERT_EQ(err, cudaSuccess) << "Failed to allocate device output memory (d_output)";

            err = cudaMemcpy(d_input, h_input.data(), H * sizeof(float), cudaMemcpyHostToDevice);
            if (err != cudaSuccess) { cudaFree(d_input); cudaFree(d_output); d_input = nullptr; d_output = nullptr; }
            ASSERT_EQ(err, cudaSuccess) << "Failed to copy input data from host to device";
        }
        else {
            d_input = nullptr; d_output = nullptr;
        }
        cudaError_t setupErr = cudaGetLastError();
        ASSERT_EQ(setupErr, cudaSuccess) << "CUDA error during SetUp: " << cudaGetErrorString(setupErr);
    }

    void TearDown() override {
        if (d_input) cudaFree(d_input);
        if (d_output) cudaFree(d_output);
        d_input = nullptr; d_output = nullptr;
        cudaError_t tearDownErr = cudaGetLastError();
        ASSERT_EQ(tearDownErr, cudaSuccess) << "CUDA error during TearDown: " << cudaGetErrorString(tearDownErr);
    }
};

// --- Parameterized Test Case for Correctness ---
TEST_P(AveragePooling1DParamTest, CUDAVerification) {
    if (H_out <= 0) {
        ASSERT_TRUE(expected_output.empty()) << "CPU output should be empty for H_out <= 0";
        SUCCEED() << "Test skipped: Calculated output size H_out (" << H_out << ") is non-positive.";
        return;
    }
    ASSERT_NE(d_input, nullptr) << "Device input pointer is null";
    ASSERT_NE(d_output, nullptr) << "Device output pointer is null";

    // Call the CUDA average pooling function (ensure it calls the correct kernel)
    solution_average_pooling_1d(d_input, kernel_size, stride, padding, d_output, H);

    cudaError_t kernelErr = cudaGetLastError();
    ASSERT_EQ(kernelErr, cudaSuccess) << "CUDA kernel launch/execution failed: " << cudaGetErrorString(kernelErr);
    // No explicit sync needed here if solution_average_pooling_1d already syncs, but good practice:
    cudaError_t syncErr = cudaDeviceSynchronize();
    ASSERT_EQ(syncErr, cudaSuccess) << "CUDA device synchronization failed: " << cudaGetErrorString(syncErr);

    // Copy result back
    std::vector<float> h_output(H_out);
    cudaError_t copyErr = cudaMemcpy(h_output.data(), d_output, H_out * sizeof(float), cudaMemcpyDeviceToHost);
    ASSERT_EQ(copyErr, cudaSuccess) << "Failed to copy output data from device to host";

    // Compare results
    ASSERT_EQ(h_output.size(), expected_output.size()) << "Output vector size mismatch";
    for (int i = 0; i < H_out; ++i) {
        EXPECT_NEAR(h_output[i], expected_output[i], 1e-5f)
            << "Mismatch at index " << i << " for params (H=" << H
            << ", k=" << kernel_size << ", s=" << stride << ", p=" << padding
            << "). Expected: " << expected_output[i]
            << ", Got: " << h_output[i];
    }
}

// --- Test Instantiation for Correctness ---
INSTANTIATE_TEST_SUITE_P(
    AvgPool1DKernelParams,
    AveragePooling1DParamTest,
    ::testing::Values(
        // Basic cases
        std::make_tuple(5, 3, 1, 1),   // H=5, K=3, S=1, P=1 -> H_out = floor((5+2-3)/1)+1 = 5
        std::make_tuple(10, 3, 1, 0),  // H=10, K=3, S=1, P=0 -> H_out = floor((10+0-3)/1)+1 = 8
        std::make_tuple(10, 3, 2, 0),  // H=10, K=3, S=2, P=0 -> H_out = floor((10+0-3)/2)+1 = floor(3.5)+1 = 4
        std::make_tuple(10, 3, 2, 1),  // H=10, K=3, S=2, P=1 -> H_out = floor((10+2-3)/2)+1 = floor(4.5)+1 = 5
        std::make_tuple(12, 4, 3, 1),  // H=12, K=4, S=3, P=1 -> H_out = floor((12+2-4)/3)+1 = floor(10/3)+1 = 3+1 = 4
        std::make_tuple(8, 2, 2, 0),   // H=8, K=2, S=2, P=0 -> H_out = floor((8+0-2)/2)+1 = floor(3)+1 = 4

        // Edge cases
        std::make_tuple(5, 1, 1, 0),   // H=5, K=1, S=1, P=0 -> H_out = floor((5+0-1)/1)+1 = 5 (Identity?)
        std::make_tuple(5, 1, 2, 0),   // H=5, K=1, S=2, P=0 -> H_out = floor((5+0-1)/2)+1 = floor(2)+1 = 3
        std::make_tuple(7, 7, 1, 0),   // H=7, K=7, S=1, P=0 -> H_out = floor((7+0-7)/1)+1 = 1
        std::make_tuple(7, 7, 7, 0),   // H=7, K=7, S=7, P=0 -> H_out = floor((7+0-7)/7)+1 = 1
        std::make_tuple(5, 3, 1, 3),   // H=5, K=3, S=1, P=3 -> H_out = floor((5+6-3)/1)+1 = 9
        std::make_tuple(3, 5, 1, 1),   // H=3, K=5, S=1, P=1 -> H_out = floor((3+2-5)/1)+1 = 1

        // Cases potentially resulting in zero output size
        // H=2, K=3, S=1, P=0 => H_out = floor((2+0-3)/1)+1 = floor(-1)+1 = 0
        std::make_tuple(2, 3, 1, 0)
    )
);


// --- Parameterized GTest Fixture for Performance ---
// Parameters: <InputSize H, kernel_size, stride, padding>
class AveragePooling1DPerformanceTest : public ::testing::TestWithParam<std::tuple<size_t, int, int, int>> {
protected:
    size_t H;
    int kernel_size;
    int stride;
    int padding;
    float* d_input = nullptr;
    float* d_output = nullptr;
    int H_out;
    const int num_runs = 10; // Number of times to run the kernel for averaging time
    const int warm_up_runs = 2; // Number of warm-up runs before timing

    cudaEvent_t start_event, stop_event;

    void SetUp() override {
        std::tie(H, kernel_size, stride, padding) = GetParam();
        H_out = compute_output_length_avg(H, kernel_size, stride, padding);

        ASSERT_GT(H, 0) << "Input size H must be positive for performance tests.";
        // Allow H_out=0 for calculation, but kernel won't run. Perf test should handle H_out > 0.
        ASSERT_GT(H_out, 0) << "Output size H_out must be positive for meaningful performance tests.";

        cudaError_t err;
        err = cudaEventCreate(&start_event); ASSERT_EQ(err, cudaSuccess);
        err = cudaEventCreate(&stop_event); if (err != cudaSuccess) cudaEventDestroy(start_event); ASSERT_EQ(err, cudaSuccess);

        err = cudaMalloc(&d_input, H * sizeof(float)); ASSERT_EQ(err, cudaSuccess);
        err = cudaMalloc(&d_output, H_out * sizeof(float)); if (err != cudaSuccess) cudaFree(d_input); ASSERT_EQ(err, cudaSuccess);

        err = cudaMemset(d_input, 0, H * sizeof(float)); ASSERT_EQ(err, cudaSuccess); // Initialize on device

        cudaError_t setupErr = cudaGetLastError(); ASSERT_EQ(setupErr, cudaSuccess);

        // Warm-up runs
        for (int i = 0; i < warm_up_runs; ++i) {
            solution_average_pooling_1d(d_input, kernel_size, stride, padding, d_output, H);
        }
        cudaError_t syncErr = cudaDeviceSynchronize(); ASSERT_EQ(syncErr, cudaSuccess); // Ensure warm-up is finished
    }

    void TearDown() override {
        if (d_input) cudaFree(d_input);
        if (d_output) cudaFree(d_output);
        cudaEventDestroy(start_event);
        cudaEventDestroy(stop_event);
        d_input = nullptr; d_output = nullptr;
        cudaError_t tearDownErr = cudaGetLastError(); ASSERT_EQ(tearDownErr, cudaSuccess);
    }

    // Helper to estimate GFLOPS (using add/div count)
    double calculate_gflops(float milliseconds) {
        if (milliseconds <= 0 || H_out <= 0 || kernel_size <= 0) return 0.0;
        // Estimate FLOPs: H_out outputs. Each needs (k-1) adds + 1 div = k FLOPs.
        double total_ops = static_cast<double>(H_out) * static_cast<double>(kernel_size);
        double seconds = milliseconds / 1000.0;
        double gflops = total_ops / seconds / 1e9; // Giga-FLOPs per second
        return gflops;
    }
};

// --- Parameterized Test Case for Performance ---
TEST_P(AveragePooling1DPerformanceTest, MeasureRuntimeAndGFLOPS) {
    ASSERT_NE(d_input, nullptr);
    ASSERT_NE(d_output, nullptr);
    ASSERT_GT(H_out, 0); // Should be guaranteed by fixture SetUp assertion

    cudaError_t err;

    err = cudaEventRecord(start_event, 0); ASSERT_EQ(err, cudaSuccess);
    for (int i = 0; i < num_runs; ++i) {
        solution_average_pooling_1d(d_input, kernel_size, stride, padding, d_output, H);
    }
    err = cudaEventRecord(stop_event, 0); ASSERT_EQ(err, cudaSuccess);
    err = cudaEventSynchronize(stop_event); ASSERT_EQ(err, cudaSuccess);

    float milliseconds = 0;
    err = cudaEventElapsedTime(&milliseconds, start_event, stop_event); ASSERT_EQ(err, cudaSuccess);
    float avg_milliseconds = milliseconds / num_runs;
    double gflops = calculate_gflops(avg_milliseconds);

    printf("[ PERF INFO ] H=%-10zu K=%d S=%d P=%d | Avg Runtime: %8.2f ms | Est. Performance: %8.2f GFLOPS\n",
        H, kernel_size, stride, padding, avg_milliseconds, gflops);
}


// --- Test Instantiation for Performance ---
// Using the same parameter sets provided for the previous max_pooling example,
// but performance results will differ for average pooling.
INSTANTIATE_TEST_SUITE_P(
    AvgPool1DPerformanceRuns,
    AveragePooling1DPerformanceTest,
    ::testing::Values(
        // Parameter sets adapted from the previous request (Dilation=1 implied)
        std::make_tuple(2097152, 7, 4, 3), // ~2M
        std::make_tuple(4194304, 2, 1, 0), // ~4M
        std::make_tuple(8388608, 3, 2, 1), // ~8M
        // std::make_tuple(16777216,  4, 2, 1), // ~16M (Original had D=2, not applicable here) -> Replacing D=1
        std::make_tuple(16777216, 4, 2, 1), // ~16M, D=1 assumed
        std::make_tuple(33554432, 3, 1, 1), // ~33M
        std::make_tuple(67108864, 5, 3, 2)  // ~67M
    )
);
