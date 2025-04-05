#include <device_launch_parameters.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <random>

// https://tensara.org/problems/max-pool-1d/

// CUDA error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error in " << __FILE__ << " at line " << __LINE__ << ": " << cudaGetErrorString(err) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)



// CUDA kernel performing 1D max pooling
__global__ void maxpool1d_kernel(const float* input, int kernel_size, int stride, int padding, int dilation, float* output, size_t H, int H_out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < H_out) {
        // Start with negative infinity for the max value
        float max_val = -INFINITY;
        for (int m = 0; m < kernel_size; m++) {
            // Calculate the input index for this kernel element
            int index = stride * i + dilation * m - padding;
            // If the index is out-of-bound, treat the value as -infinity
            float value = (index < 0 || index >= H) ? -INFINITY : input[index];
            max_val = fmaxf(max_val, value);
        }
        output[i] = max_val;
    }
}

// Note: input and output are device pointers to float arrays
extern "C" void solution_max_pooling_1d(const float* input, int kernel_size, int stride, int padding, int dilation, float* output, size_t H) {
    // Calculate output size using:
    // H_out = floor((H + 2*padding - dilation*(kernel_size-1) - 1) / stride) + 1
    int H_out = ((H + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1;

    // Set up grid and block dimensions
    int threadsPerBlock = 256;
    int blocksPerGrid = (H_out + threadsPerBlock - 1) / threadsPerBlock;

    // Launch the CUDA kernel
    maxpool1d_kernel << <blocksPerGrid, threadsPerBlock >> > (input, kernel_size, stride, padding, dilation, output, H, H_out);

    // Wait for the kernel to finish before returning
    cudaDeviceSynchronize();
}

void generate_random_data(std::vector<float>& data) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (auto& val : data) {
        val = dist(gen);
    }
}

int max_pooling_1d() {
    size_t H = 16;  // Length of the 1D input tensor
    int kernel_size = 3;
    int stride = 1;
    int padding = 1;
    int dilation = 1;

    std::vector<float> h_input(H);
    generate_random_data(h_input);

    float* d_input, * d_output;
    CUDA_CHECK(cudaMalloc(&d_input, H * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, H * sizeof(float))); // Assuming output is same size
    CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), H * sizeof(float), cudaMemcpyHostToDevice));

    // Call the CUDA function
    solution_max_pooling_1d(d_input, kernel_size, stride, padding, dilation, d_output, H);

    // Copy result back to host
    std::vector<float> h_output(H);
    CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, H * sizeof(float), cudaMemcpyDeviceToHost));

    std::cout << "Input: ";
    for (float val : h_input) std::cout << val << " ";
    std::cout << "\nOutput: ";
    for (float val : h_output) std::cout << val << " ";
    std::cout << std::endl;

    // Free device memory
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));

    return 0;
}