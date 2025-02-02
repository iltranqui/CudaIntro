// conv2d_backprop.cu with backpropagation
// #define MAIN
#ifdef MAIN
#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define IMAGE_WIDTH 512
#define IMAGE_HEIGHT 512
#define KERNEL_WIDTH 3
#define KERNEL_HEIGHT 3

// -------------------------------------------------------------------------
// Forward Convolution Kernel (for context/testing)
// -------------------------------------------------------------------------
__global__ void conv2D_forward(const float* input, const float* kernel, float* output,
    int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int kCenterX = KERNEL_WIDTH / 2;
    int kCenterY = KERNEL_HEIGHT / 2;

    if (x < width && y < height) {
        float sum = 0.0f;
        // Iterate over kernel elements.
        for (int m = 0; m < KERNEL_HEIGHT; m++) {
            for (int n = 0; n < KERNEL_WIDTH; n++) {
                int row = y + m - kCenterY;
                int col = x + n - kCenterX;
                // Use zero-padding if indices are out-of-bound.
                if (row >= 0 && row < height && col >= 0 && col < width) {
                    sum += input[row * width + col] * kernel[m * KERNEL_WIDTH + n];
                }
            }
        }
        output[y * width + x] = sum;
    }
}

// -------------------------------------------------------------------------
// Backpropagation Kernel 1: Compute gradient with respect to input.
// -------------------------------------------------------------------------
// This computes: d_input = conv2D_backprop_input(d_output, kernel)
// where the kernel is flipped (rotated 180°)
__global__ void conv2D_backprop_input(const float* d_output, const float* kernel,
    float* d_input, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int kCenterX = KERNEL_WIDTH / 2;
    int kCenterY = KERNEL_HEIGHT / 2;

    if (x < width && y < height) {
        float sum = 0.0f;
        // For each input pixel, sum contributions from all d_output positions
        // that used this pixel in the forward pass.
        for (int m = 0; m < KERNEL_HEIGHT; m++) {
            for (int n = 0; n < KERNEL_WIDTH; n++) {
                // In the forward pass: output(y,x) got contribution from input(y + m - kCenterY, x + n - kCenterX)
                // In backprop, we “reverse” the operation:
                int out_x = x - n + kCenterX;
                int out_y = y - m + kCenterY;
                if (out_x >= 0 && out_x < width && out_y >= 0 && out_y < height) {
                    // Use the rotated kernel (flip vertically and horizontally).
                    int r_m = KERNEL_HEIGHT - 1 - m;
                    int r_n = KERNEL_WIDTH - 1 - n;
                    sum += d_output[out_y * width + out_x] * kernel[r_m * KERNEL_WIDTH + r_n];
                }
            }
        }
        d_input[y * width + x] = sum;
    }
}

// -------------------------------------------------------------------------
// Backpropagation Kernel 2: Compute gradient with respect to the kernel.
// -------------------------------------------------------------------------
// This computes: d_kernel = conv2D_backprop_kernel(input, d_output)
// Each kernel element is computed by summing over all spatial positions.
__global__ void conv2D_backprop_kernel(const float* input, const float* d_output,
    float* d_kernel, int width, int height) {
    // Each thread computes one element of the kernel gradient.
    int k = threadIdx.x; // k ranges from 0 to KERNEL_WIDTH*KERNEL_HEIGHT - 1
    if (k < KERNEL_WIDTH * KERNEL_HEIGHT) {
        int kernel_row = k / KERNEL_WIDTH;
        int kernel_col = k % KERNEL_WIDTH;
        int kCenterX = KERNEL_WIDTH / 2;
        int kCenterY = KERNEL_HEIGHT / 2;
        float sum = 0.0f;

        // Loop over every output pixel.
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                // In the forward pass, output(y,x) used:
                // input(y + kernel_row - kCenterY, x + kernel_col - kCenterX)
                int in_x = x + kernel_col - kCenterX;
                int in_y = y + kernel_row - kCenterY;
                if (in_x >= 0 && in_x < width && in_y >= 0 && in_y < height) {
                    sum += input[in_y * width + in_x] * d_output[y * width + x];
                }
            }
        }
        d_kernel[k] = sum;
    }
}

// -------------------------------------------------------------------------
// Main: Setup data, run forward conv and backpropagation.
// -------------------------------------------------------------------------
int main() {
    // Seed for reproducibility.
    std::srand(static_cast<unsigned int>(std::time(0)));

    int imageSize = IMAGE_WIDTH * IMAGE_HEIGHT;
    int imageBytes = imageSize * sizeof(float);
    int kernelSize = KERNEL_WIDTH * KERNEL_HEIGHT;
    int kernelBytes = kernelSize * sizeof(float);

    // Allocate host memory.
    float* h_input = new float[imageSize];
    float* h_output = new float[imageSize];
    float* h_d_input = new float[imageSize];
    float* h_d_output = new float[imageSize];
    float* h_kernel = new float[kernelSize];
    float* h_d_kernel = new float[kernelSize];

    // Initialize the input image with random values.
    for (int i = 0; i < imageSize; i++) {
        h_input[i] = static_cast<float>(std::rand() % 256);
        // For simplicity, assume the gradient from the next layer is all ones.
        h_d_output[i] = 1.0f;
    }
    // Initialize a simple averaging (blur) kernel.
    for (int i = 0; i < kernelSize; i++) {
        h_kernel[i] = 1.0f / kernelSize;
    }

    // Allocate device memory.
    float* d_input, * d_output, * d_kernel, * d_d_input, * d_d_output, * d_d_kernel;
    cudaMalloc((void**)&d_input, imageBytes);
    cudaMalloc((void**)&d_output, imageBytes);
    cudaMalloc((void**)&d_kernel, kernelBytes);
    cudaMalloc((void**)&d_d_input, imageBytes);
    cudaMalloc((void**)&d_d_output, imageBytes);
    cudaMalloc((void**)&d_d_kernel, kernelBytes);

    // Copy data from host to device.
    cudaMemcpy(d_input, h_input, imageBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, h_kernel, kernelBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_d_output, h_d_output, imageBytes, cudaMemcpyHostToDevice);

    // Define grid and block dimensions for the 2D operations.
    dim3 blockDim(16, 16);
    dim3 gridDim((IMAGE_WIDTH + blockDim.x - 1) / blockDim.x, (IMAGE_HEIGHT + blockDim.y - 1) / blockDim.y);

    // ---------------------------------------------------------------
    // (1) Forward pass: Compute the convolution output.
    // ---------------------------------------------------------------
    conv2D_forward << < gridDim, blockDim >> > (d_input, d_kernel, d_output, IMAGE_WIDTH, IMAGE_HEIGHT);

    // ---------------------------------------------------------------
    // (2) Backpropagation pass:
    //   (a) Compute gradient with respect to input.
    // ---------------------------------------------------------------
    conv2D_backprop_input << < gridDim, blockDim >> > (d_d_output, d_kernel, d_d_input, IMAGE_WIDTH, IMAGE_HEIGHT);
    cudaDeviceSynchronize();

    // ---------------------------------------------------------------
    // (2) Backpropagation pass:
    //   (b) Compute gradient with respect to the kernel.
    //       Launch one block with one thread per kernel element.
    // ---------------------------------------------------------------
    conv2D_backprop_kernel << <1, kernelSize >> > (d_input, d_d_output, d_d_kernel, IMAGE_WIDTH, IMAGE_HEIGHT);
    cudaDeviceSynchronize();

    // Copy the results back to host memory.
    cudaMemcpy(h_output, d_output, imageBytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_d_input, d_d_input, imageBytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_d_kernel, d_d_kernel, kernelBytes, cudaMemcpyDeviceToHost);

    // Display a small sample of the results.
    std::cout << "Forward convolution sample output:" << std::endl;
    for (int y = 250; y < 255; y++) {
        for (int x = 250; x < 255; x++) {
            std::cout << h_output[y * IMAGE_WIDTH + x] << "\t";
        }
        std::cout << std::endl;
    }

    std::cout << "\nBackpropagated input gradient sample:" << std::endl;
    for (int y = 250; y < 255; y++) {
        for (int x = 250; x < 255; x++) {
            std::cout << h_d_input[y * IMAGE_WIDTH + x] << "\t";
        }
        std::cout << std::endl;
    }

    std::cout << "\nKernel gradient:" << std::endl;
    for (int i = 0; i < kernelSize; i++) {
        std::cout << h_d_kernel[i] << "\t";
    }
    std::cout << std::endl;

    // Clean up device and host memory.
    delete[] h_input;
    delete[] h_output;
    delete[] h_d_input;
    delete[] h_d_output;
    delete[] h_kernel;
    delete[] h_d_kernel;
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_kernel);
    cudaFree(d_d_input);
    cudaFree(d_d_output);
    cudaFree(d_d_kernel);

    return 0;
}

#endif // MAIN