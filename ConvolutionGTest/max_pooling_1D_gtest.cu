#include <gtest/gtest.h>
#include <vector>
#include <tuple>
#include <cmath> // For std::ceil
#include <algorithm> // For std::max
#include <cuda_runtime.h> // Include CUDA runtime header
#include <iostream>
#include <cstdlib>   // For size_t
#include <cstdio>    // For printf formatting
#include <numeric>   // For std::accumulate
#include "max_pooling_1d.cu"
#include <device_launch_parameters.h>
#include "gtest_header.cuh"

// Assume cpu_max_pooling_1d and solution_max_pooling_1d are defined elsewhere.
// Example placeholder for cpu_max_pooling_1d for completeness:
std::vector<float> cpu_max_pooling_1d(const std::vector<float>& input,
    int kernel_size, int stride, int padding, int dilation) {
    int H = input.size();
    // Calculate output size based on PyTorch formula:
    // H_out = floor((H + 2*padding - dilation*(kernel_size-1) - 1)/stride + 1)
    int H_out = std::floor(static_cast<float>(H + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1);
    if (H_out <= 0) {
        return {}; // Return empty vector if output size is non-positive
    }

    std::vector<float> output(H_out);

    for (int i = 0; i < H_out; ++i) {
        float max_val = -std::numeric_limits<float>::infinity();
        int start_index = i * stride - padding;
        for (int k = 0; k < kernel_size; ++k) {
            int input_index = start_index + k * dilation;
            if (input_index >= 0 && input_index < H) {
                max_val = std::max(max_val, input[input_index]);
            }
        }
        // If the window didn't overlap with any valid input (e.g., padding > kernel_size),
        // the max_val might still be -infinity. Handle appropriately (e.g., set to 0 or another default).
        // For simplicity here, we keep it, assuming valid pooling windows. If no valid input
        // was found in the window, behavior might depend on the exact definition required.
        output[i] = max_val;
    }
    return output;
}

// Assume solution_max_pooling_1d is the CUDA kernel launcher function
// extern "C" void solution_max_pooling_1d(const float* d_input, int kernel_size, int stride, int padding, int dilation, float* d_output, int H_in);
// Placeholder for the CUDA function definition if needed for compilation
void solution_max_pooling_1d(const float* d_input, int kernel_size, int stride, int padding, int dilation, float* d_output, int H) {
    // In a real scenario, this would launch the CUDA kernel.
    // For this example, we'll copy a dummy result or leave it empty.
    // The actual comparison logic is in the test.
    int H_out = std::floor(static_cast<float>(H + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1);
    if (H_out <= 0) return; // No output to compute

    // *** NOTE: This is NOT a real CUDA implementation. ***
    // *** It's just to make the GTest structure runnable.    ***
    // *** You MUST replace this with your actual CUDA kernel launch. ***
    /*
    std::vector<float> h_input(H_in);
    cudaMemcpy(h_input.data(), d_input, H_in * sizeof(float), cudaMemcpyDeviceToHost);
    std::vector<float> h_output = cpu_max_pooling_1d(h_input, kernel_size, stride, padding, dilation);
    if (!h_output.empty()) {
        cudaMemcpy(d_output, h_output.data(), h_output.size() * sizeof(float), cudaMemcpyHostToDevice);
    }
    */
    // Calculate output size using:
    // H_out = floor((H + 2*padding - dilation*(kernel_size-1) - 1) / stride) + 1/
    // int H_out = ((H + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1;

    // Set up grid and block dimensions
    int defaultThreadsPerBlock = 1024;
    int* threadsPerBlock = &defaultThreadsPerBlock;
    int blocksPerGrid = (H_out + *threadsPerBlock - 1) / *threadsPerBlock;

    // Launch the CUDA kernel
    maxpool1d_kernel << <blocksPerGrid, *threadsPerBlock >> > (d_input, kernel_size, stride, padding, dilation, d_output, H, H_out);

    // Wait for the kernel to finish before returning
    cudaDeviceSynchronize();
}

// Custom listener to print errors in red
class RedErrorPrinter : public ::testing::EmptyTestEventListener {
public:
    void OnTestPartResult(const ::testing::TestPartResult& part_result) override {
        if (part_result.failed()) {
            std::cerr << ANSI_RED << part_result.file_name() << ":" << part_result.line_number() << ": Failure: "
                << part_result.message() << ANSI_RESET << std::endl;
        }
    }
};

// Define a test fixture class that is parameterized
class MaxPooling1DTest : public ::testing::TestWithParam<std::tuple<size_t, int, int, int, int>> {
protected:
    // Member variables to store parameters for each test instance
    size_t H;
    int kernel_size;
    int stride;
    int padding;
    int dilation;
    std::vector<float> h_input;
    std::vector<float> expected_output;
    float* d_input = nullptr;
    float* d_output = nullptr;
    int H_out;

    // SetUp method to initialize resources before each test
    void SetUp() override {
        // Get parameters for the current test case
        std::tie(H, kernel_size, stride, padding, dilation) = GetParam();

        // Prepare input data (e.g., sequential values)
        h_input.resize(H);
        for (size_t i = 0; i < H; ++i) {
            // Using a simple pattern, but can be randomized or made more complex
            h_input[i] = static_cast<float>(i % 10) * 0.1f + static_cast<float>(i);
        }

        // Compute expected output using the CPU version
        expected_output = cpu_max_pooling_1d(h_input, kernel_size, stride, padding, dilation);
        H_out = expected_output.size();

        // Only proceed with CUDA allocation if output is expected
        if (H_out > 0) {
            // Allocate device memory
            cudaError_t err = cudaMalloc(&d_input, H * sizeof(float));
            ASSERT_EQ(err, cudaSuccess) << "Failed to allocate device input memory";
            err = cudaMalloc(&d_output, H_out * sizeof(float));
            ASSERT_EQ(err, cudaSuccess) << "Failed to allocate device output memory";


            // Copy input to device
            err = cudaMemcpy(d_input, h_input.data(), H * sizeof(float), cudaMemcpyHostToDevice);
            ASSERT_EQ(err, cudaSuccess) << "Failed to copy input data to device";
        }
        else {
            d_input = nullptr;
            d_output = nullptr;
        }
    }

    // TearDown method to release resources after each test
    void TearDown() override {
        if (d_input) {
            cudaFree(d_input);
            d_input = nullptr;
        }
        if (d_output) {
            cudaFree(d_output);
            d_output = nullptr;
        }
        // It's good practice to check for CUDA errors during cleanup too
        cudaError_t err = cudaGetLastError();
        ASSERT_EQ(err, cudaSuccess) << "CUDA error during TearDown";
    }
};

// Define the parameterized test case
TEST_P(MaxPooling1DTest, CUDAVsCPUComparison) {
    // If H_out is non-positive, the CPU function returns an empty vector,
    // and no CUDA execution or comparison is needed.
    if (H_out <= 0) {
        ASSERT_TRUE(expected_output.empty());
        SUCCEED() << "Test skipped as output size is non-positive.";
        return; // Nothing to test
    }

    ASSERT_NE(d_input, nullptr);
    ASSERT_NE(d_output, nullptr);


    // Call the CUDA max pooling function.
    solution_max_pooling_1d(d_input, kernel_size, stride, padding, dilation, d_output, H);
    // Check for kernel launch errors
    cudaError_t kernelErr = cudaGetLastError();
    ASSERT_EQ(kernelErr, cudaSuccess) << "CUDA kernel launch failed: " << cudaGetErrorString(kernelErr);
    // Synchronize to ensure kernel completion before copying results
    cudaError_t syncErr = cudaDeviceSynchronize();
    ASSERT_EQ(syncErr, cudaSuccess) << "CUDA device synchronization failed: " << cudaGetErrorString(syncErr);


    // Copy the output from device to host.
    std::vector<float> h_output(H_out);
    cudaError_t copyErr = cudaMemcpy(h_output.data(), d_output, H_out * sizeof(float), cudaMemcpyDeviceToHost);
    ASSERT_EQ(copyErr, cudaSuccess) << "Failed to copy output data from device";


    // Compare the expected and the actual output.
    ASSERT_EQ(h_output.size(), expected_output.size()) << "Output size mismatch";
    for (int i = 0; i < H_out; ++i) {
        EXPECT_NEAR(h_output[i], expected_output[i], 1e-5)
            << "Mismatch at index " << i << " for params (H=" << H
            << ", k=" << kernel_size << ", s=" << stride << ", p=" << padding
            << ", d=" << dilation << "): expected " << expected_output[i]
            << ", got " << h_output[i];
    }
}

// Instantiate the test suite with different sets of parameters
INSTANTIATE_TEST_SUITE_P(
    MaxPool1DParameterSet, // A name for the instantiation
    MaxPooling1DTest,        // The test fixture class
    ::testing::Values(        // The parameter values to test
        // Basic cases
        std::make_tuple(5, 3, 1, 1, 1),    // Original test case
        std::make_tuple(10, 3, 1, 0, 1),   // No padding
        std::make_tuple(10, 3, 2, 0, 1),   // Stride 2, no padding
        std::make_tuple(10, 3, 2, 1, 1),   // Stride 2, padding 1
        std::make_tuple(10, 2, 2, 0, 1),   // Kernel 2, Stride 2
        // Larger input
        std::make_tuple(50, 5, 2, 2, 1),
        // Cases with dilation
        std::make_tuple(10, 3, 1, 1, 2),   // Dilation 2
        std::make_tuple(20, 3, 1, 2, 3),   // Dilation 3, padding 2
        // Edge cases
        std::make_tuple(5, 1, 1, 0, 1),    // Kernel size 1 (identity if stride=1, padding=0)
        std::make_tuple(5, 5, 1, 0, 1),    // Kernel size equals input size
        std::make_tuple(5, 5, 5, 0, 1),    // Kernel size and stride equal input size
        std::make_tuple(5, 3, 1, 3, 1),    // Large padding
        std::make_tuple(4, 3, 2, 0, 2),    // Dilation causes window larger than input effectively
        std::make_tuple(3, 5, 1, 1, 1),    // Kernel larger than input (with padding)
        std::make_tuple(2, 3, 1, 0, 1)     // Kernel larger than input (no padding - potentially empty output)
        // Case leading to zero output size ( H + 2p - d(k-1) - 1 ) / s + 1 <= 0
        // Example: H=5, k=3, s=1, p=0, d=3 => (5 + 0 - 3(2) - 1)/1 + 1 = (5-6-1)/1 + 1 = -2 + 1 = -1 <= 0
         // std::make_tuple(5, 3, 1, 0, 3) // Check if H_out calculation handles negative results
    )
);

// --- Helper Functions (compute_output_length, cpu_max_pooling_1d) ---
// Assume these are defined correctly as in the previous versions
// Helper function to compute output length based on PyTorch formula
int compute_output_length(size_t H, int kernel_size, int stride, int padding, int dilation) {
    if (kernel_size <= 0 || stride <= 0 || dilation <= 0 || H == 0) {
        return 0;
    }
    int dilated_kernel_size = dilation * (kernel_size - 1) + 1;
    float numerator = static_cast<float>(H + 2 * padding - dilated_kernel_size);
    int H_out = static_cast<int>(std::floor(numerator / stride)) + 1;
    return std::max(0, H_out);
}

// --- Parameterized GTest Fixture for Performance ---
class MaxPooling1DPerformanceTest : public ::testing::TestWithParam<std::tuple<size_t, int, int, int, int>> {
protected:
    size_t H;
    int kernel_size;
    int stride;
    int padding;
    int dilation;
    float* d_input = nullptr;
    float* d_output = nullptr;
    int H_out;
    const int num_runs = 10; // Number of times to run the kernel for averaging time
    const int warm_up_runs = 2; // Number of warm-up runs before timing

    cudaEvent_t start_event, stop_event;

    void SetUp() override {
        std::tie(H, kernel_size, stride, padding, dilation) = GetParam();
        H_out = compute_output_length(H, kernel_size, stride, padding, dilation);

        ASSERT_GT(H, 0) << "Input size H must be positive for performance tests.";
        ASSERT_GT(H_out, 0) << "Output size H_out must be positive for performance tests.";

        cudaError_t err;

        // Create CUDA events for timing
        err = cudaEventCreate(&start_event);
        ASSERT_EQ(err, cudaSuccess) << "Failed to create start CUDA event";
        err = cudaEventCreate(&stop_event);
        if (err != cudaSuccess) { cudaEventDestroy(start_event); } // Cleanup if stop fails
        ASSERT_EQ(err, cudaSuccess) << "Failed to create stop CUDA event";

        // Allocate device memory
        err = cudaMalloc(&d_input, H * sizeof(float));
        ASSERT_EQ(err, cudaSuccess) << "Failed to allocate device input memory (d_input)";
        err = cudaMalloc(&d_output, H_out * sizeof(float));
        if (err != cudaSuccess) { cudaFree(d_input); d_input = nullptr; }
        ASSERT_EQ(err, cudaSuccess) << "Failed to allocate device output memory (d_output)";

        // Initialize input data on device (e.g., with a pattern or random, copying is slow for large H)
        // For perf test, just filling it on device might be sufficient/faster if kernel exists
        // If you need specific data, copy a smaller block and repeat, or use a fill kernel.
        // For simplicity, let's copy zeros (actual data pattern often doesn't dominate maxpool perf)
        err = cudaMemset(d_input, 0, H * sizeof(float));
        ASSERT_EQ(err, cudaSuccess) << "Failed to memset device input memory";

        // Initial check for CUDA errors
        cudaError_t setupErr = cudaGetLastError();
        ASSERT_EQ(setupErr, cudaSuccess) << "CUDA error during SetUp: " << cudaGetErrorString(setupErr);

        // Warm-up runs (important!)
        for (int i = 0; i < warm_up_runs; ++i) {
            solution_max_pooling_1d(d_input, kernel_size, stride, padding, dilation, d_output, H);
        }
        // Ensure warm-up is finished
        cudaError_t syncErr = cudaDeviceSynchronize();
        ASSERT_EQ(syncErr, cudaSuccess) << "CUDA sync failed after warm-up.";
    }

    void TearDown() override {
        if (d_input) cudaFree(d_input);
        if (d_output) cudaFree(d_output);
        cudaEventDestroy(start_event);
        cudaEventDestroy(stop_event);
        d_input = nullptr; d_output = nullptr;
        cudaError_t tearDownErr = cudaGetLastError();
        ASSERT_EQ(tearDownErr, cudaSuccess) << "CUDA error during TearDown: " << cudaGetErrorString(tearDownErr);
    }

    // Helper to estimate GFLOPS (using comparison count)
    double calculate_gflops(float milliseconds) {
        if (milliseconds <= 0) return 0.0;
        // Estimate operations: H_out outputs, K comparisons per output
        double total_ops = static_cast<double>(H_out) * static_cast<double>(kernel_size);
        double seconds = milliseconds / 1000.0;
        double gflops = total_ops / seconds / 1e9; // Giga-Ops per second
        return gflops;
    }
};

// --- Parameterized Test Case for Performance ---  
TEST_P(MaxPooling1DPerformanceTest, MeasureRuntimeAndGFLOPS) {  
    ASSERT_NE(d_input, nullptr);  
    ASSERT_NE(d_output, nullptr);  
  
    cudaError_t err;  
    std::vector<float> run_times(num_runs);  
  
    // Start timing  
    for (int i = 0; i < num_runs; ++i) {  
        err = cudaEventRecord(start_event, 0); // Use default stream 0  
        ASSERT_EQ(err, cudaSuccess) << "Failed to record start event";  
  
        // Run the kernel  
        solution_max_pooling_1d(d_input, kernel_size, stride, padding, dilation, d_output, H);  
  
        // Stop timing  
        err = cudaEventRecord(stop_event, 0);  
        ASSERT_EQ(err, cudaSuccess) << "Failed to record stop event";  
  
        // Wait for the stop event to complete  
        err = cudaEventSynchronize(stop_event);  
        ASSERT_EQ(err, cudaSuccess) << "Failed to synchronize stop event";  
  
        // Calculate elapsed time  
        float milliseconds = 0;  
        err = cudaEventElapsedTime(&milliseconds, start_event, stop_event);  
        ASSERT_EQ(err, cudaSuccess) << "Failed to get elapsed time";  
  
        run_times[i] = milliseconds;  
    }  
  
    // Calculate average, best, and worst time  
    float avg_milliseconds = std::accumulate(run_times.begin(), run_times.end(), 0.0f) / num_runs;  
    float best_milliseconds = *std::min_element(run_times.begin(), run_times.end());  
    float worst_milliseconds = *std::max_element(run_times.begin(), run_times.end());  
  
    // Calculate estimated GFLOPS  
    double gflops = calculate_gflops(avg_milliseconds);  
  
    // Print the results in the desired format  
    // Using printf for potentially better alignment and floating point formatting  
    printf("[ PERF INFO ] H=%-10zu K=%d S=%d P=%d D=%d | Avg Runtime: %8.2f ms | Best Runtime: %8.2f ms | Worst Runtime: %8.2f ms | Est. Performance: %8.2f GFLOPS\n",  
        H, kernel_size, stride, padding, dilation, avg_milliseconds, best_milliseconds, worst_milliseconds, gflops);  
  
    // Optionally, add assertions for performance bounds (makes it a pass/fail test)  
    // EXPECT_LT(avg_milliseconds, 10.0) << "Runtime exceeded threshold";  
    // EXPECT_GT(gflops, 10.0) << "Performance below threshold";  
}


// --- Test Instantiation for Performance ---
INSTANTIATE_TEST_SUITE_P(
    MaxPool1DPerformanceRuns, // Different name from the correctness suite
    MaxPooling1DPerformanceTest,
    ::testing::Values(
        // Parameter sets from the user request
        std::make_tuple(2097152, 7, 4, 3, 1), // ~2M
        std::make_tuple(4194304, 2, 1, 0, 1), // ~4M
        std::make_tuple(8388608, 3, 2, 1, 1), // ~8M
        std::make_tuple(16777216, 4, 2, 1, 2), // ~16M - Note D=2
        std::make_tuple(33554432, 3, 1, 1, 1), // ~33M
        std::make_tuple(67108864, 5, 3, 2, 1)  // ~67M

        // Add other sizes if needed
        // std::make_tuple(1024*1024, 3, 1, 1, 1) // 1M example
    )
);

/*
// Main function to run the tests
int main(int argc, char** argv) {
	printDeviceInfo(); // Print device info before running tests
    ::testing::InitGoogleTest(&argc, argv);

    // Add our custom listener to print errors in red
    ::testing::TestEventListeners& listeners =
        ::testing::UnitTest::GetInstance()->listeners();
    // By default, GTest prints basic output to stdout. We don't want that here.
    // listeners.Release(listeners.default_result_printer()); // This line would remove the default printer

    // Add our custom error printer
    listeners.Append(new RedErrorPrinter);

    // Optionally initialize CUDA once if needed globally, or ensure it's handled per test/fixture.
    // cudaSetDevice(0) might be better in SetUpTestCase if using a test fixture class that needs it once.
    // For simplicity, putting it here is common if all tests need the same device.
    cudaError_t err = cudaSetDevice(0);
    if (err != cudaSuccess) {
        std::cerr << "Failed to set CUDA device 0: " << cudaGetErrorString(err) << std::endl;
        return 1; // Indicate failure
    }
    return RUN_ALL_TESTS();
}
*/