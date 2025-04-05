#include <gtest/gtest.h>
#include <vector>
#include <tuple>
#include <cmath> // For std::ceil
#include <algorithm> // For std::max
#include <cuda_runtime.h> // Include CUDA runtime header


// --- ANSI Color Codes ---
#ifdef _WIN32 // Basic check for Windows console (might need more robust detection)
    // Windows console needs specific API calls or might support ANSI via WT, etc.
    // For simplicity, disable color on basic Windows detection.
const char* const ANSI_RED = "";
const char* const ANSI_GREEN = "";
const char* const ANSI_YELLOW = "";
const char* const ANSI_RESET = "";
#else
    // ANSI escape codes for colors (common on Linux/macOS)
const char* const ANSI_RED = "\033[1;31m"; // Bold Red
const char* const ANSI_GREEN = "\033[1;32m"; // Bold Green
const char* const ANSI_YELLOW = "\033[1;33m"; // Bold Yellow
const char* const ANSI_RESET = "\033[0m";   // Reset color
#endif

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
void solution_max_pooling_1d(const float* d_input, int kernel_size, int stride, int padding, int dilation, float* d_output, int H_in) {
    // In a real scenario, this would launch the CUDA kernel.
    // For this example, we'll copy a dummy result or leave it empty.
    // The actual comparison logic is in the test.
    int H_out = std::floor(static_cast<float>(H_in + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1);
    if (H_out <= 0) return; // No output to compute

    // *** NOTE: This is NOT a real CUDA implementation. ***
    // *** It's just to make the GTest structure runnable.   ***
    // *** You MUST replace this with your actual CUDA kernel launch. ***
    std::vector<float> h_input(H_in);
    cudaMemcpy(h_input.data(), d_input, H_in * sizeof(float), cudaMemcpyDeviceToHost);
    std::vector<float> h_output = cpu_max_pooling_1d(h_input, kernel_size, stride, padding, dilation);
    if (!h_output.empty()) {
        cudaMemcpy(d_output, h_output.data(), h_output.size() * sizeof(float), cudaMemcpyHostToDevice);
    }
}


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
    MaxPooling1DTest,      // The test fixture class
    ::testing::Values(     // The parameter values to test
        // Basic cases
        std::make_tuple(5, 3, 1, 1, 1),   // Original test case
        std::make_tuple(10, 3, 1, 0, 1),  // No padding
        std::make_tuple(10, 3, 2, 0, 1),  // Stride 2, no padding
        std::make_tuple(10, 3, 2, 1, 1),  // Stride 2, padding 1
        std::make_tuple(10, 2, 2, 0, 1),  // Kernel 2, Stride 2
        // Larger input
        std::make_tuple(50, 5, 2, 2, 1),
        // Cases with dilation
        std::make_tuple(10, 3, 1, 1, 2),  // Dilation 2
        std::make_tuple(20, 3, 1, 2, 3),  // Dilation 3, padding 2
        // Edge cases
        std::make_tuple(5, 1, 1, 0, 1),   // Kernel size 1 (identity if stride=1, padding=0)
        std::make_tuple(5, 5, 1, 0, 1),   // Kernel size equals input size
        std::make_tuple(5, 5, 5, 0, 1),   // Kernel size and stride equal input size
        std::make_tuple(5, 3, 1, 3, 1),   // Large padding
        std::make_tuple(4, 3, 2, 0, 2),   // Dilation causes window larger than input effectively
        std::make_tuple(3, 5, 1, 1, 1),    // Kernel larger than input (with padding)
        std::make_tuple(2, 3, 1, 0, 1)    // Kernel larger than input (no padding - potentially empty output)
        // Case leading to zero output size ( H + 2p - d(k-1) - 1 ) / s + 1 <= 0
        // Example: H=5, k=3, s=1, p=0, d=3 => (5 + 0 - 3(2) - 1)/1 + 1 = (5-6-1)/1 + 1 = -2 + 1 = -1 <= 0
         // std::make_tuple(5, 3, 1, 0, 3) // Check if H_out calculation handles negative results
    )
);


// Main function to run the tests
int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
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