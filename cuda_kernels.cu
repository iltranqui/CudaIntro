#include "cuda_kernels.cuh"
#include <cuda_runtime.h>
#include <iostream>

__global__ void forward_kernel(float* input, float* weights, float* biases, float* output, int in_size, int out_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= out_size) return;

    float sum = biases[idx];
    for (int i = 0; i < in_size; ++i) {
        sum += input[i] * weights[i * out_size + idx];
    }
    output[idx] = sum;
}

__global__ void backward_kernel(float* grad_output, float* weights, float* biases, int in_size, int out_size, float learning_rate) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= out_size) return;

    for (int i = 0; i < in_size; ++i) {
        weights[i * out_size + idx] -= learning_rate * grad_output[idx];
    }
    biases[idx] -= learning_rate * grad_output[idx];
}

void forward_cuda(float* input, float* weights, float* biases, float* output, int in_size, int out_size) {
    dim3 blockDim(128);
    dim3 gridDim((out_size + blockDim.x - 1) / blockDim.x);
    forward_kernel << <gridDim, blockDim >> > (input, weights, biases, output, in_size, out_size);
    cudaDeviceSynchronize();
}

void backward_cuda(float* grad_output, float* weights, float* biases, int in_size, int out_size, float learning_rate) {
    dim3 blockDim(128);
    dim3 gridDim((out_size + blockDim.x - 1) / blockDim.x);
    backward_kernel << <gridDim, blockDim >> > (grad_output, weights, biases, in_size, out_size, learning_rate);
    cudaDeviceSynchronize();
}
