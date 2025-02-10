// conv2d_infer.cu with inference
// conv2d.cu
#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cuda_runtime.h>              // necessary to use cudaMalloc, cudaMemcpy, etc. and CUDA runtime APIs
#include <device_launch_parameters.h>  // necessary to use threadIdx, blockIdx, blockDim, gridDim

// Define the image dimensions and kernel dimensions
#define IMAGE_WIDTH 512
#define IMAGE_HEIGHT 512
#define KERNEL_WIDTH 3
#define KERNEL_HEIGHT 3

// The convolution kernel is stored in constant memory for efficiency. -> constant memory is stored
__constant__ float d_kernel[KERNEL_WIDTH * KERNEL_HEIGHT];

// CUDA kernel for performing 2D convolution on the input matrix.
__global__ void conv2D(const float* input, float* output, int width, int height) {
    // Compute the (x,y) coordinate of the element this thread will process.
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Compute the radius (offset) of the kernel.
    int kCenterX = KERNEL_WIDTH / 2;
    int kCenterY = KERNEL_HEIGHT / 2;

    if (x < width && y < height) {
        float sum = 0.0f;
        // Iterate over the kernel window.
        for (int m = 0; m < KERNEL_HEIGHT; m++) {
            for (int n = 0; n < KERNEL_WIDTH; n++) {
                // Compute the corresponding input pixel location.
                int row = y + m - kCenterY;
                int col = x + n - kCenterX;
                // Check for boundary conditions. Here, we use zero-padding.
                if (row >= 0 && row < height && col >= 0 && col < width) {
                    sum += input[row * width + col] * d_kernel[m * KERNEL_WIDTH + n];
                }
            }
        }
        output[y * width + x] = sum;
    }
}

int conv2d_inf() {
    // Set seed for random number generation
    std::srand(static_cast<unsigned int>(std::time(0)));

    // Calculate the total size of the image.
    const int imageSize = IMAGE_WIDTH * IMAGE_HEIGHT;
    const int imageBytes = imageSize * sizeof(float);

    // Allocate host memory for input and output matrices.
    float* h_input = new float[imageSize];
    float* h_output = new float[imageSize];

    // Fill the input matrix with random values.
    for (int i = 0; i < imageSize; i++) {
        h_input[i] = static_cast<float>(std::rand() % 256); // Values in [0, 255]
    }

    // Define a 3x3 averaging kernel (blur filter).
    float h_kernel[KERNEL_WIDTH * KERNEL_HEIGHT] = {
        1.0f / 9, 1.0f / 9, 1.0f / 9,
        1.0f / 9, 1.0f / 9, 1.0f / 9,
        1.0f / 9, 1.0f / 9, 1.0f / 9
    };

    // Copy the kernel to the device constant memory.
    cudaMemcpyToSymbol(d_kernel, h_kernel, KERNEL_WIDTH * KERNEL_HEIGHT * sizeof(float));

    // Allocate device memory for the input and output.
    float* d_input, * d_output;
    cudaMalloc((void**)&d_input, imageBytes);
    cudaMalloc((void**)&d_output, imageBytes);

    // Copy the input matrix from the host to the device.
    cudaMemcpy(d_input, h_input, imageBytes, cudaMemcpyHostToDevice);

    // Define the block and grid dimensions.
    dim3 blockDim(16, 16);
    dim3 gridDim((IMAGE_WIDTH + blockDim.x - 1) / blockDim.x, (IMAGE_HEIGHT + blockDim.y - 1) / blockDim.y);

    // Launch the convolution kernel.
    conv2D << <gridDim, blockDim >> > (d_input, d_output, IMAGE_WIDTH, IMAGE_HEIGHT);
    cudaDeviceSynchronize();

    // Copy the output matrix back to the host.
    cudaMemcpy(h_output, d_output, imageBytes, cudaMemcpyDeviceToHost);

    // Print a small portion of the output for verification.
    std::cout << "Sample output values:" << std::endl;
    for (int y = 250; y < 255; y++) {
        for (int x = 250; x < 255; x++) {
            std::cout << h_output[y * IMAGE_WIDTH + x] << "\t";
        }
        std::cout << std::endl;
    }

    // Clean up host and device memory.
    delete[] h_input;
    delete[] h_output;
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}

