#include <iostream>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <device_launch_parameters.h>
#include "main_header.cuh"

#define BLOCK_SIZE 16

// CUDA kernel to initialize random matrices
template <typename T>
__global__ void init_matrix(T* mat, int rows, int cols, float scale = 1.0f) {
    // Calculate the global thread indices
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int index = idy * cols + idx;

    // Check if the thread is within matrix bounds
    if (idx < cols && idy < rows) {
        // Initialize the random state
        curandState state;
		curand_init(1234, index, 0, &state);  // Use index for unique seed

        // Assign a random value to the matrix element, scaled and shifted to [-0.5, 0.5]
        mat[index] = scale * (curand_uniform(&state) - 0.5f);
    }
}

// Forward pass for deformable convolution
/* Deformable Convolutional Networks https://arxiv.org/abs/1703.06211 have 2 type of parameters: 
		- weight: the convolutional kernel weights
		- offset: the learned offsets to the sampling positions

*/
__global__ void deform_conv2d_forward(float* input, float* offset, float* weight, float* output,
    int in_h, int in_w, int out_h, int out_w, int ksize, int stride) {
    // Compute output pixel coordinates
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Ensure the thread is within output dimensions
    if (x < out_w && y < out_h) {
        float sum = 0.0f; // Accumulate convolution result

        // Iterate over the kernel window
        for (int i = 0; i < ksize; i++) {
            for (int j = 0; j < ksize; j++) {
                // Compute the index for the offset values
                int offset_idx = (y * out_w + x) * 2;

                // Compute the sampling positions with learned offsets
                int px = x * stride + j + offset[offset_idx];  // Horizontal position
                int py = y * stride + i + offset[offset_idx + 1];  // Vertical position

                // Ensure sampling positions are within input bounds
                if (px >= 0 && px < in_w && py >= 0 && py < in_h) {
                    sum += input[py * in_w + px] * weight[i * ksize + j]; // Apply convolution
                }
            }
        }

        // Store the computed output value
        output[y * out_w + x] = sum;
    }
}

// Custom atomicAdd for float (only needed for old architectures)
__device__ float atomicAddFloat(float* address, float val) {
    return atomicAdd(address, val);
}

// Backward pass for deformable convolution
__global__ void deform_conv2d_backward(float* grad_output, float* grad_input, float* grad_weight, float* grad_offset,
    float* input, float* weight, float* offset,
    int in_h, int in_w, int out_h, int out_w, int ksize, int stride) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < out_w && y < out_h) {
        float grad = grad_output[y * out_w + x];
        for (int i = 0; i < ksize; i++) {
            for (int j = 0; j < ksize; j++) {
                int offset_idx = (y * out_w + x) * 2;
                int px = x * stride + j + offset[offset_idx];
                int py = y * stride + i + offset[offset_idx + 1];
                if (px >= 0 && px < in_w && py >= 0 && py < in_h) {
                    atomicAddFloat(&grad_input[py * in_w + px], grad * weight[i * ksize + j]);
                    atomicAddFloat(&grad_weight[i * ksize + j], grad * input[py * in_w + px]);
                    atomicAddFloat(&grad_offset[offset_idx], grad * input[py * in_w + px] * j);
                    atomicAddFloat(&grad_offset[offset_idx + 1], grad * input[py * in_w + px] * i);
                }
            }
        }
    }
}

int conv2d_deform() {
    int in_h = 32, in_w = 32, out_h = 30, out_w = 30, ksize = 3, stride = 1;
    float* input, * offset, * weight, * output, * grad_output, * grad_input, * grad_weight, * grad_offset;
    size_t size_in = in_h * in_w * sizeof(float);
    size_t size_out = out_h * out_w * sizeof(float);
    size_t size_k = ksize * ksize * sizeof(float);

    cudaMallocManaged(&input, size_in);
    cudaMallocManaged(&offset, size_out * 2);
    cudaMallocManaged(&weight, size_k);
    cudaMallocManaged(&output, size_out);
    cudaMallocManaged(&grad_output, size_out);
    cudaMallocManaged(&grad_input, size_in);
    cudaMallocManaged(&grad_weight, size_k);
    cudaMallocManaged(&grad_offset, size_out * 2);

    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((in_w + block.x - 1) / block.x, (in_h + block.y - 1) / block.y);

    init_matrix << <grid, block >> > (input, in_h, in_w);
    init_matrix << <grid, block >> > (offset, out_h, out_w * 2);
    init_matrix << <grid, block >> > (weight, ksize, ksize);
    init_matrix << <grid, block >> > (grad_output, out_h, out_w);
    cudaDeviceSynchronize();

    deform_conv2d_forward << <grid, block >> > (input, offset, weight, output, in_h, in_w, out_h, out_w, ksize, stride);
    cudaDeviceSynchronize();

    std::cout << "First output value: " << output[0] << std::endl;

    deform_conv2d_backward << <grid, block >> > (grad_output, grad_input, grad_weight, grad_offset,
        input, weight, offset, in_h, in_w, out_h, out_w, ksize, stride);
    cudaDeviceSynchronize();

    std::cout << "First gradient value (input): " << grad_input[0] << std::endl;

    cudaFree(input);
    cudaFree(offset);
    cudaFree(weight);
    cudaFree(output);
    cudaFree(grad_output);
    cudaFree(grad_input);
    cudaFree(grad_weight);
    cudaFree(grad_offset);
    return 0;
    return 1;
}
