#include <iostream>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include "device_launch_parameters.h"
#include "main_header.cuh"
#include <cstdlib>

#define BLOCK_SIZE 16

// ----------------------------------------------------------
// Kernel to initialize a matrix with random values using curand
// ----------------------------------------------------------
template <typename T>
__global__ void init_matrix(T* mat, int size, float scale = 1.0f) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        curandState state;
        curand_init(1234, idx, 0, &state);
        mat[idx] = scale * (curand_uniform(&state) - 0.5f);
    }
}

// ----------------------------------------------------------
// Forward Pass: Basic 2D Convolution Kernel
// ----------------------------------------------------------
// N: input image
// F: filter (kernel) of size (2*r+1)x(2*r+1)
// P: output image
// r: filter radius (e.g., r=1 => 3x3 filter)
// width, height: dimensions of the input/output image
__global__ void convolution_2d_basic_kernel_forward(float* N, float* F, float* P, int r, int width, int height) {
    int outCol = blockIdx.x * blockDim.x + threadIdx.x;
    int outRow = blockIdx.y * blockDim.y + threadIdx.y;
    if (outCol < width && outRow < height) {
        float Pvalue = 0.0f;
        // Loop over the filter window
        for (int fRow = 0; fRow < 2 * r + 1; fRow++) {
            for (int fCol = 0; fCol < 2 * r + 1; fCol++) {
                int inRow = outRow - r + fRow;
                int inCol = outCol - r + fCol;
                // Check bounds for the input image
                if (inRow >= 0 && inRow < height && inCol >= 0 && inCol < width) {
                    Pvalue += N[inRow * width + inCol] * F[fRow * (2 * r + 1) + fCol];
                }
            }
        }
        P[outRow * width + outCol] = Pvalue;
    }
}

// ----------------------------------------------------------
// Backward Pass: Compute Gradient w.r.t. Filter Weights
// ----------------------------------------------------------
// N: input image
// grad_output: gradient of loss w.r.t. output P (assumed pre-computed, here set to ones)
// grad_F: gradient for filter F (to be computed)
// r: filter radius, width and height: dimensions of image
//
// For each filter element F[fRow, fCol], the gradient is the sum over
// all output pixels (grad_output * corresponding input pixel).
__global__ void convolution_2d_backward_kernel(float* N, float* grad_output, float* grad_F, int r, int width, int height) {
    // Each thread computes one filter element gradient.
    int fRow = threadIdx.y;
    int fCol = threadIdx.x;
    int filterSize = 2 * r + 1;
    if (fRow < filterSize && fCol < filterSize) {
        float grad = 0.0f;
        // Loop over all output positions
        for (int outRow = 0; outRow < height; outRow++) {
            for (int outCol = 0; outCol < width; outCol++) {
                // The corresponding input pixel for this filter element:
                int inRow = outRow - r + fRow;
                int inCol = outCol - r + fCol;
                if (inRow >= 0 && inRow < height && inCol >= 0 && inCol < width) {
                    grad += grad_output[outRow * width + outCol] * N[inRow * width + inCol];
                }
            }
        }
        grad_F[fRow * filterSize + fCol] = grad;
    }
}

// ----------------------------------------------------------
// Weight Update Kernel: Simple Gradient Descent Step
// ----------------------------------------------------------
// F: filter weights
// grad_F: computed gradient for filter weights
// filterSize: total number of weights in the filter
// learning_rate: step size for gradient descent
__global__ void update_weights_kernel(float* F, float* grad_F, int filterSize, float learning_rate) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < filterSize) {
        F[idx] -= learning_rate * grad_F[idx];
    }
}

int conv2d_backpass() {
    // Parameters for the image and convolution
    int width = 32;
    int height = 32;
	int r = 3; // Filter radius: r=1 gives a 3x3 filter # 2 gives 5x5 filter 
    int filterSize = (2 * r + 1) * (2 * r + 1);
    int imageSize = width * height;

    // Allocate unified memory for input image, filter, output, and gradients.
    float* N, * F, * P, * grad_output, * grad_F, *target;
    cudaMallocManaged(&N, imageSize * sizeof(float));
    cudaMallocManaged(&F, filterSize * sizeof(float));
    cudaMallocManaged(&P, imageSize * sizeof(float));
    cudaMallocManaged(&grad_output, imageSize * sizeof(float));
    cudaMallocManaged(&grad_F, filterSize * sizeof(float));
    cudaMallocManaged(&target, imageSize * sizeof(float));

    // Initialize input image N and filter F with random values.
    int numThreads = 256;
    int numBlocks_N = (imageSize + numThreads - 1) / numThreads;
    int numBlocks_F = (filterSize + numThreads - 1) / numThreads;
    init_matrix << <numBlocks_N, numThreads >> > (N, imageSize, 1.0f);
    init_matrix << <numBlocks_F, numThreads >> > (F, filterSize, 1.0f);
    cudaDeviceSynchronize();

    // For simplicity, set grad_output to ones (simulate gradient from next layer)
    for (int i = 0; i < imageSize; i++) {
        grad_output[i] = 1.0f;
    }

    // Initialize target output (for example, a constant value of 1.0 for each pixel)
    for (int i = 0; i < imageSize; i++) {
        target[i] = 0.0f;
    }


    // Perform an initial forward pass and print the first output value
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((width + BLOCK_SIZE - 1) / BLOCK_SIZE, (height + BLOCK_SIZE - 1) / BLOCK_SIZE);
    convolution_2d_basic_kernel_forward << <gridDim, blockDim >> > (N, F, P, r, width, height);
    cudaDeviceSynchronize();
    std::cout << "Initial forward pass: First output value = " << P[0] << std::endl;
    // Perform an initial forward pass and print the initial filter weights as a 2D matrix
    std::cout << "Initial filter weights:" << std::endl;
    for (int i = 0; i < 2 * r + 1; i++) {
        for (int j = 0; j < 2 * r + 1; j++) {
            std::cout << F[i * (2 * r + 1) + j] << " ";
        }
        std::cout << std::endl;
    }

    // Training loop parameters
    int numEpochs = 1000;
    float learning_rate = 0.1f;

    float loss = 0.0f;
    // Training Loop
    for (int epoch = 0; epoch < numEpochs; epoch++) {
        // ----------------------------- Forward Pass -----------------------------
        convolution_2d_basic_kernel_forward << <gridDim, blockDim >> > (N, F, P, r, width, height);
        cudaDeviceSynchronize();
        //std::cout << "Epoch " << epoch << ": First output value = " << P[0] << std::endl;

        // --------------------------- Loss Computation ---------------------------
       // Compute Mean Squared Error (MSE) loss and also update grad_output.
        for (int i = 0; i < imageSize; i++) {
            float diff = P[i] - target[i];
            loss += diff * diff;
            // Derivative of MSE loss: dL/dP = 2*(P - target) / imageSize
            grad_output[i] = 2.0f * diff / imageSize;
        }
        loss /= imageSize;
        //std::cout << "Epoch " << epoch << ": Loss = " << loss << std::endl;

        // -------------------------- Backward Pass -------------------------------
        // Zero out grad_F before computing new gradients
        for (int i = 0; i < filterSize; i++) {
            grad_F[i] = 0.0f;
        }
        // Launch backward kernel with one block of size (2*r+1)x(2*r+1)
        dim3 blockDimBackward(2 * r + 1, 2 * r + 1);
        dim3 gridDimBackward(1, 1);
        convolution_2d_backward_kernel << <gridDimBackward, blockDimBackward >> > (N, grad_output, grad_F, r, width, height);
        cudaDeviceSynchronize();

        // -------------------------- Update Weights ------------------------------
        update_weights_kernel << <numBlocks_F, numThreads >> > (F, grad_F, filterSize, learning_rate);
        cudaDeviceSynchronize();

        // Optionally, print the first output value and the first filter weight after the update
        //std::cout << "Epoch " << epoch << ": First output value = " << P[0] << ", First filter weight = " << F[0] << std::endl;
    }

    // Perform an initial forward pass and print the first output value
    convolution_2d_basic_kernel_forward << <gridDim, blockDim >> > (N, F, P, r, width, height);
    cudaDeviceSynchronize();
    std::cout << "Initial forward pass: FINAL output value = " << P[0] << std::endl;
    // Perform an initial forward pass and print the initial filter weights as a 2D matrix
    std::cout << "Initial filter weights AFTER TRAINING:" << std::endl;
    for (int i = 0; i < 2 * r + 1; i++) {
        for (int j = 0; j < 2 * r + 1; j++) {
            std::cout << F[i * (2 * r + 1) + j] << " ";
        }
        std::cout << std::endl;
    }

    // Printing final loss
	std::cout << "Final loss = " << loss << std::endl;

    // Free unified memory
    cudaFree(N);
    cudaFree(F);
    cudaFree(P);
    cudaFree(grad_output);
    cudaFree(grad_F);

    return 0;
}
