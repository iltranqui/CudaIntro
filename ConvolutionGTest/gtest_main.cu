#include <cuda_runtime.h> // Include CUDA runtime header
#include <device_launch_parameters.h>
#include <gtest/gtest.h>
#include <vector>
#include <tuple>
#include <cmath> // For std::ceil
#include "gtest_header.cuh"

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

__global__ void dummyKernel() {} // A minimal kernel for device initialization

// CUDA function to print device information
void printDeviceInfo() {
    int deviceCount = 0;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);
    if (error != cudaSuccess) {
        std::cerr << "CUDA error: Failed to get device count: " << cudaGetErrorString(error) << std::endl;
        return;
    }

    if (deviceCount == 0) {
        std::cout << "No CUDA-enabled devices found." << std::endl;
        return;
    }

    for (int device = 0; device < deviceCount; ++device) {
        cudaDeviceProp deviceProp;
        error = cudaGetDeviceProperties(&deviceProp, device);
        if (error != cudaSuccess) {
            std::cerr << "CUDA error: Failed to get device properties for device " << device << ": " << cudaGetErrorString(error) << std::endl;
            continue;
        }

        std::cout << "--------------------------------------------------" << std::endl;
        std::cout << "Device ID: " << device << std::endl;
        std::cout << "Device Name: " << deviceProp.name << std::endl;
        std::cout << "CUDA Capability: " << deviceProp.major << "." << deviceProp.minor << std::endl;
        std::cout << "Maximum threads per block: " << deviceProp.maxThreadsPerBlock << std::endl;
        std::cout << "Maximum dimensions of a thread block (x,y,z): "
            << deviceProp.maxThreadsDim[0] << ", "
            << deviceProp.maxThreadsDim[1] << ", "
            << deviceProp.maxThreadsDim[2] << std::endl;
        std::cout << "Maximum dimensions of a grid size (x,y,z): "
            << deviceProp.maxGridSize[0] << ", "
            << deviceProp.maxGridSize[1] << ", "
            << deviceProp.maxGridSize[2] << std::endl;
        std::cout << "Total global memory: " << deviceProp.totalGlobalMem / (1024 * 1024) << " MB" << std::endl;
        std::cout << "Shared memory per block: " << deviceProp.sharedMemPerBlock / 1024 << " KB" << std::endl;
        std::cout << "Registers per block: " << deviceProp.regsPerBlock << std::endl;
        // Add more properties as needed

        // Optionally, launch a minimal kernel to ensure the device is ready
        dummyKernel << <1, 1 >> > ();
        cudaDeviceSynchronize();
        cudaGetLastError(); // Clear any potential errors from the dummy kernel
    }
    std::cout << "--------------------------------------------------" << std::endl;
}

/*
// CUDA kernel for the warmup
__global__ __host__ void warmupKernel() {
    // This kernel does a minimal amount of work
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // Access a small amount of global memory (read and write)
    float* dummy_ptr;
    cudaMalloc((void**)&dummy_ptr, sizeof(float));
    float value = 1.0f;
    cudaMemcpy(dummy_ptr, &value, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(&value, dummy_ptr, sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(dummy_ptr);
}

// C++ void function to perform GPU warmup
void gpuWarmup(int numBlocks, int threadsPerBlock) {
    // Launch the warmup kernel on the GPU
    warmupKernel << <numBlocks, threadsPerBlock >> > ();

    // Wait for the kernel to complete
    cudaDeviceSynchronize();

    // Check for any CUDA errors
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cerr << "CUDA error after warmup: " << cudaGetErrorString(error) << std::endl;
    }
}
*/

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