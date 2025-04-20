/**
 * @file deform_conv2d_gtest.cu
 * @brief Google Test file for testing deformable convolution implementation
 *
 * This file contains tests for both forward and backward passes of deformable convolution,
 * including correctness tests, gradient checking, and performance benchmarks.
 */

// Standard library includes
#include <gtest/gtest.h>  // Google Test framework
#include <vector>        // For std::vector
#include <tuple>         // For std::tuple
#include <cmath>         // For mathematical functions
#include <algorithm>     // For std::fill
#include <iostream>      // For std::cout
#include <cstdlib>       // For rand()
#include <cstdio>        // For printf
#include <numeric>       // For std::accumulate

// CUDA includes
#include <cuda_runtime.h>             // CUDA runtime API
#include <device_launch_parameters.h> // For CUDA kernel launch parameters

// Project includes
#include "gtest_header.cuh"           // Custom GTest header

// Include the deformable convolution implementation
// We need to include the file that contains the CUDA kernels
#include "../Convolution/conv2_deform.cu"

/**
 * @brief CPU implementation of bilinear interpolation for testing
 *
 * This function performs bilinear interpolation on the input tensor at the specified
 * coordinates (y, x). It handles boundary conditions by zero-padding.
 *
 * @param input The input tensor as a flattened vector
 * @param in_h Height of the input tensor
 * @param in_w Width of the input tensor
 * @param y Vertical coordinate for sampling (can be fractional)
 * @param x Horizontal coordinate for sampling (can be fractional)
 * @return Interpolated value at the specified coordinates
 */
float cpu_bilinear_interpolate(const std::vector<float>& input, int in_h, int in_w, float y, float x) {
    // Get the four nearest integer coordinates
    int x1 = floor(x);
    int x2 = x1 + 1;
    int y1 = floor(y);
    int y2 = y1 + 1;

    // Calculate the interpolation weights
    float wx1 = x2 - x; // Weight for x1
    float wx2 = x - x1; // Weight for x2
    float wy1 = y2 - y; // Weight for y1
    float wy2 = y - y1; // Weight for y2

    // Boundary check
    bool valid_x1 = (x1 >= 0 && x1 < in_w);
    bool valid_x2 = (x2 >= 0 && x2 < in_w);
    bool valid_y1 = (y1 >= 0 && y1 < in_h);
    bool valid_y2 = (y2 >= 0 && y2 < in_h);

    // Get pixel values (with zero padding for out-of-bounds)
    float v11 = (valid_y1 && valid_x1) ? input[y1 * in_w + x1] : 0.0f;
    float v12 = (valid_y1 && valid_x2) ? input[y1 * in_w + x2] : 0.0f;
    float v21 = (valid_y2 && valid_x1) ? input[y2 * in_w + x1] : 0.0f;
    float v22 = (valid_y2 && valid_x2) ? input[y2 * in_w + x2] : 0.0f;

    // Bilinear interpolation
    float value = wy1 * wx1 * v11 + wy1 * wx2 * v12 + wy2 * wx1 * v21 + wy2 * wx2 * v22;

    return value;
}

/**
 * @brief CPU implementation of deformable convolution forward pass for testing
 *
 * This function computes the forward pass of deformable convolution on the CPU.
 * It uses bilinear interpolation to sample from the input at deformed positions.
 *
 * @param input The input tensor as a flattened vector
 * @param offset The offset tensor for deforming the sampling grid
 * @param weight The convolution kernel weights
 * @param in_h Height of the input tensor
 * @param in_w Width of the input tensor
 * @param out_h Height of the output tensor
 * @param out_w Width of the output tensor
 * @param ksize Size of the kernel (assumed square)
 * @param stride Stride for the convolution
 * @return Output tensor as a flattened vector
 */
std::vector<float> cpu_deform_conv2d_forward(
    const std::vector<float>& input,
    const std::vector<float>& offset,
    const std::vector<float>& weight,
    int in_h, int in_w,
    int out_h, int out_w,
    int ksize, int stride) {

    std::vector<float> output(out_h * out_w, 0.0f);

    for (int y = 0; y < out_h; y++) {
        for (int x = 0; x < out_w; x++) {
            float sum = 0.0f;

            // Calculate the center of the receptive field
            int center_h = y * stride;
            int center_w = x * stride;

            // Loop over kernel window
            for (int i = 0; i < ksize; i++) {
                for (int j = 0; j < ksize; j++) {
                    // Each kernel position (i,j) has its own offset (dx, dy)
                    int offset_idx = ((y * out_w + x) * ksize * ksize + i * ksize + j) * 2;

                    // Deformed sampling position on input
                    float px = center_w + j - (ksize - 1) / 2.0f + offset[offset_idx];      // adjusted x position
                    float py = center_h + i - (ksize - 1) / 2.0f + offset[offset_idx + 1];  // adjusted y position

                    // Use bilinear interpolation to sample from the input
                    float val = cpu_bilinear_interpolate(input, in_h, in_w, py, px);

                    // Apply convolution
                    sum += val * weight[i * ksize + j];
                }
            }

            output[y * out_w + x] = sum;
        }
    }

    return output;
}

/**
 * @brief CPU implementation of deformable convolution backward pass for testing
 *
 * This function computes the gradients for the input, weights, and offsets
 * in the backward pass of deformable convolution on the CPU.
 *
 * @param grad_output Gradient of the loss with respect to the output
 * @param grad_input Output: Gradient of the loss with respect to the input
 * @param grad_weight Output: Gradient of the loss with respect to the weights
 * @param grad_offset Output: Gradient of the loss with respect to the offsets
 * @param input The original input tensor
 * @param weight The convolution kernel weights
 * @param offset The offset tensor for deforming the sampling grid
 * @param in_h Height of the input tensor
 * @param in_w Width of the input tensor
 * @param out_h Height of the output tensor
 * @param out_w Width of the output tensor
 * @param ksize Size of the kernel (assumed square)
 * @param stride Stride for the convolution
 */
void cpu_deform_conv2d_backward(
    const std::vector<float>& grad_output,
    std::vector<float>& grad_input,
    std::vector<float>& grad_weight,
    std::vector<float>& grad_offset,
    const std::vector<float>& input,
    const std::vector<float>& weight,
    const std::vector<float>& offset,
    int in_h, int in_w,
    int out_h, int out_w,
    int ksize, int stride) {

    // Initialize gradients to zero
    std::fill(grad_input.begin(), grad_input.end(), 0.0f);
    std::fill(grad_weight.begin(), grad_weight.end(), 0.0f);
    std::fill(grad_offset.begin(), grad_offset.end(), 0.0f);

    for (int y = 0; y < out_h; y++) {
        for (int x = 0; x < out_w; x++) {
            float grad = grad_output[y * out_w + x];

            // Calculate the center of the receptive field
            int center_h = y * stride;
            int center_w = x * stride;

            for (int i = 0; i < ksize; i++) {
                for (int j = 0; j < ksize; j++) {
                    // Compute the index for the offset values
                    int offset_idx = ((y * out_w + x) * ksize * ksize + i * ksize + j) * 2;

                    // Compute the sampling positions with learned offsets
                    float px = center_w + j - (ksize - 1) / 2.0f + offset[offset_idx];      // Horizontal position
                    float py = center_h + i - (ksize - 1) / 2.0f + offset[offset_idx + 1];  // Vertical position

                    // Get the interpolated value
                    float val = cpu_bilinear_interpolate(input, in_h, in_w, py, px);

                    // Gradient for the weight
                    grad_weight[i * ksize + j] += grad * val;

                    // Get the four nearest integer coordinates for gradient calculation
                    int x1 = floor(px);
                    int x2 = x1 + 1;
                    int y1 = floor(py);
                    int y2 = y1 + 1;

                    // Calculate the interpolation weights
                    float wx1 = x2 - px; // Weight for x1
                    float wx2 = px - x1; // Weight for x2
                    float wy1 = y2 - py; // Weight for y1
                    float wy2 = py - y1; // Weight for y2

                    // Boundary check
                    bool valid_x1 = (x1 >= 0 && x1 < in_w);
                    bool valid_x2 = (x2 >= 0 && x2 < in_w);
                    bool valid_y1 = (y1 >= 0 && y1 < in_h);
                    bool valid_y2 = (y2 >= 0 && y2 < in_h);

                    // Gradient for the input
                    float grad_val = grad * weight[i * ksize + j];
                    if (valid_y1 && valid_x1) grad_input[y1 * in_w + x1] += grad_val * wy1 * wx1;
                    if (valid_y1 && valid_x2) grad_input[y1 * in_w + x2] += grad_val * wy1 * wx2;
                    if (valid_y2 && valid_x1) grad_input[y2 * in_w + x1] += grad_val * wy2 * wx1;
                    if (valid_y2 && valid_x2) grad_input[y2 * in_w + x2] += grad_val * wy2 * wx2;

                    // Get pixel values for gradient calculation
                    float v11 = (valid_y1 && valid_x1) ? input[y1 * in_w + x1] : 0.0f;
                    float v12 = (valid_y1 && valid_x2) ? input[y1 * in_w + x2] : 0.0f;
                    float v21 = (valid_y2 && valid_x1) ? input[y2 * in_w + x1] : 0.0f;
                    float v22 = (valid_y2 && valid_x2) ? input[y2 * in_w + x2] : 0.0f;

                    // Gradient for the offsets
                    float grad_offset_x = grad_val * (wy1 * (v12 - v11) + wy2 * (v22 - v21));
                    float grad_offset_y = grad_val * (wx1 * (v21 - v11) + wx2 * (v22 - v12));

                    grad_offset[offset_idx] += grad_offset_x;
                    grad_offset[offset_idx + 1] += grad_offset_y;
                }
            }
        }
    }
}

/**
 * @brief Test fixture class for testing the forward pass of deformable convolution
 *
 * This class sets up the test environment for testing the forward pass of deformable convolution.
 * It initializes input, offset, and weight tensors, and computes the expected output using
 * the CPU implementation for comparison with the CUDA implementation.
 */
class DeformConv2DForwardTest : public ::testing::TestWithParam<std::tuple<int, int, int, int, int>> {
protected:
    // Member variables
    int in_h, in_w; // Input height and width
    int ksize; // Kernel size
    int stride; // Stride
    int out_h, out_w; // Output height and width

    std::vector<float> h_input;
    std::vector<float> h_offset;
    std::vector<float> h_weight;
    std::vector<float> h_output;
    std::vector<float> expected_output;

    float* d_input = nullptr;
    float* d_offset = nullptr;
    float* d_weight = nullptr;
    float* d_output = nullptr;

    void SetUp() override {
        // Get parameters for the current test case
        std::tie(in_h, in_w, ksize, stride, out_h) = GetParam();

        // Calculate output width
        out_w = out_h; // Assuming square output for simplicity

        // Initialize input data
        h_input.resize(in_h * in_w);
        for (int i = 0; i < in_h * in_w; i++) {
            h_input[i] = static_cast<float>(i % 10) * 0.1f;
        }

        // Initialize offset data (small random values)
        h_offset.resize(out_h * out_w * 2 * ksize * ksize);
        for (int i = 0; i < h_offset.size(); i++) {
            h_offset[i] = (static_cast<float>(rand()) / RAND_MAX - 0.5f) * 0.2f;
        }

        // Initialize weight data
        h_weight.resize(ksize * ksize);
        for (int i = 0; i < ksize * ksize; i++) {
            h_weight[i] = static_cast<float>(i % 5) * 0.1f + 0.1f;
        }

        // Allocate device memory
        cudaMalloc(&d_input, h_input.size() * sizeof(float));
        cudaMalloc(&d_offset, h_offset.size() * sizeof(float));
        cudaMalloc(&d_weight, h_weight.size() * sizeof(float));
        cudaMalloc(&d_output, out_h * out_w * sizeof(float));

        // Copy data to device
        cudaMemcpy(d_input, h_input.data(), h_input.size() * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_offset, h_offset.data(), h_offset.size() * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_weight, h_weight.data(), h_weight.size() * sizeof(float), cudaMemcpyHostToDevice);

        // Compute expected output using CPU implementation
        expected_output = cpu_deform_conv2d_forward(h_input, h_offset, h_weight, in_h, in_w, out_h, out_w, ksize, stride);

        // Resize output buffer
        h_output.resize(out_h * out_w);
    }

    void TearDown() override {
        if (d_input) cudaFree(d_input);
        if (d_offset) cudaFree(d_offset);
        if (d_weight) cudaFree(d_weight);
        if (d_output) cudaFree(d_output);

        d_input = nullptr;
        d_offset = nullptr;
        d_weight = nullptr;
        d_output = nullptr;
    }
};

/**
 * @brief Test case for verifying the correctness of the forward pass
 *
 * This test launches the CUDA kernel for the forward pass of deformable convolution
 * and compares the results with the expected output computed by the CPU implementation.
 */
TEST_P(DeformConv2DForwardTest, ForwardPassCorrectness) {
    // Define grid and block dimensions
    dim3 block(16, 16);
    dim3 grid((out_w + block.x - 1) / block.x, (out_h + block.y - 1) / block.y);

    // Launch the kernel
    deform_conv2d_forward<<<grid, block>>>(d_input, d_offset, d_weight, d_output, in_h, in_w, out_h, out_w, ksize, stride);

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    ASSERT_EQ(err, cudaSuccess) << "Kernel launch failed: " << cudaGetErrorString(err);

    // Synchronize to ensure kernel completion
    err = cudaDeviceSynchronize();
    ASSERT_EQ(err, cudaSuccess) << "Kernel execution failed: " << cudaGetErrorString(err);

    // Copy result back to host
    err = cudaMemcpy(h_output.data(), d_output, out_h * out_w * sizeof(float), cudaMemcpyDeviceToHost);
    ASSERT_EQ(err, cudaSuccess) << "Failed to copy result from device: " << cudaGetErrorString(err);

    // Compare with expected output
    for (int i = 0; i < out_h * out_w; i++) {
        EXPECT_NEAR(h_output[i], expected_output[i], 1e-4) << "Mismatch at index " << i;
    }
}

/**
 * @brief Test fixture class for testing the backward pass of deformable convolution
 *
 * This class sets up the test environment for testing the backward pass of deformable convolution.
 * It initializes input, offset, weight, and gradient output tensors, and computes the expected
 * gradients using the CPU implementation for comparison with the CUDA implementation.
 */
class DeformConv2DBackwardTest : public ::testing::TestWithParam<std::tuple<int, int, int, int, int>> {
protected:
    // Member variables
    int in_h, in_w;
    int ksize;
    int stride;
    int out_h, out_w;

    std::vector<float> h_input;
    std::vector<float> h_offset;
    std::vector<float> h_weight;
    std::vector<float> h_grad_output;
    std::vector<float> h_grad_input;
    std::vector<float> h_grad_weight;
    std::vector<float> h_grad_offset;

    std::vector<float> expected_grad_input;
    std::vector<float> expected_grad_weight;
    std::vector<float> expected_grad_offset;

    float* d_input = nullptr;
    float* d_offset = nullptr;
    float* d_weight = nullptr;
    float* d_grad_output = nullptr;
    float* d_grad_input = nullptr;
    float* d_grad_weight = nullptr;
    float* d_grad_offset = nullptr;

    void SetUp() override {
        // Get parameters for the current test case
        std::tie(in_h, in_w, ksize, stride, out_h) = GetParam();

        // Calculate output width
        out_w = out_h; // Assuming square output for simplicity

        // Initialize input data
        h_input.resize(in_h * in_w);
        for (int i = 0; i < in_h * in_w; i++) {
            h_input[i] = static_cast<float>(i % 10) * 0.1f;
        }

        // Initialize offset data (small random values)
        h_offset.resize(out_h * out_w * 2 * ksize * ksize);
        for (int i = 0; i < h_offset.size(); i++) {
            h_offset[i] = (static_cast<float>(rand()) / RAND_MAX - 0.5f) * 0.2f;
        }

        // Initialize weight data
        h_weight.resize(ksize * ksize);
        for (int i = 0; i < ksize * ksize; i++) {
            h_weight[i] = static_cast<float>(i % 5) * 0.1f + 0.1f;
        }

        // Initialize gradient output data
        h_grad_output.resize(out_h * out_w);
        for (int i = 0; i < out_h * out_w; i++) {
            h_grad_output[i] = static_cast<float>(i % 7) * 0.1f + 0.05f;
        }

        // Initialize gradient buffers
        h_grad_input.resize(in_h * in_w, 0.0f);
        h_grad_weight.resize(ksize * ksize, 0.0f);
        h_grad_offset.resize(h_offset.size(), 0.0f);

        // Compute expected gradients using CPU implementation
        expected_grad_input.resize(in_h * in_w, 0.0f);
        expected_grad_weight.resize(ksize * ksize, 0.0f);
        expected_grad_offset.resize(h_offset.size(), 0.0f);

        cpu_deform_conv2d_backward(
            h_grad_output, expected_grad_input, expected_grad_weight, expected_grad_offset,
            h_input, h_weight, h_offset, in_h, in_w, out_h, out_w, ksize, stride
        );

        // Allocate device memory
        cudaMalloc(&d_input, h_input.size() * sizeof(float));
        cudaMalloc(&d_offset, h_offset.size() * sizeof(float));
        cudaMalloc(&d_weight, h_weight.size() * sizeof(float));
        cudaMalloc(&d_grad_output, h_grad_output.size() * sizeof(float));
        cudaMalloc(&d_grad_input, h_grad_input.size() * sizeof(float));
        cudaMalloc(&d_grad_weight, h_grad_weight.size() * sizeof(float));
        cudaMalloc(&d_grad_offset, h_grad_offset.size() * sizeof(float));

        // Copy data to device
        cudaMemcpy(d_input, h_input.data(), h_input.size() * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_offset, h_offset.data(), h_offset.size() * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_weight, h_weight.data(), h_weight.size() * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_grad_output, h_grad_output.data(), h_grad_output.size() * sizeof(float), cudaMemcpyHostToDevice);

        // Initialize gradient buffers on device to zero
        cudaMemset(d_grad_input, 0, h_grad_input.size() * sizeof(float));
        cudaMemset(d_grad_weight, 0, h_grad_weight.size() * sizeof(float));
        cudaMemset(d_grad_offset, 0, h_grad_offset.size() * sizeof(float));
    }

    void TearDown() override {
        if (d_input) cudaFree(d_input);
        if (d_offset) cudaFree(d_offset);
        if (d_weight) cudaFree(d_weight);
        if (d_grad_output) cudaFree(d_grad_output);
        if (d_grad_input) cudaFree(d_grad_input);
        if (d_grad_weight) cudaFree(d_grad_weight);
        if (d_grad_offset) cudaFree(d_grad_offset);

        d_input = nullptr;
        d_offset = nullptr;
        d_weight = nullptr;
        d_grad_output = nullptr;
        d_grad_input = nullptr;
        d_grad_weight = nullptr;
        d_grad_offset = nullptr;
    }
};

/**
 * @brief Test case for verifying the correctness of the backward pass
 *
 * This test launches the CUDA kernel for the backward pass of deformable convolution
 * and compares the computed gradients with the expected gradients computed by the CPU implementation.
 */
TEST_P(DeformConv2DBackwardTest, BackwardPassCorrectness) {
    // Define grid and block dimensions
    dim3 block(16, 16);
    dim3 grid((out_w + block.x - 1) / block.x, (out_h + block.y - 1) / block.y);

    // Launch the kernel
    deform_conv2d_backward<<<grid, block>>>(
        d_grad_output, d_grad_input, d_grad_weight, d_grad_offset,
        d_input, d_weight, d_offset, in_h, in_w, out_h, out_w, ksize, stride
    );

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    ASSERT_EQ(err, cudaSuccess) << "Kernel launch failed: " << cudaGetErrorString(err);

    // Synchronize to ensure kernel completion
    err = cudaDeviceSynchronize();
    ASSERT_EQ(err, cudaSuccess) << "Kernel execution failed: " << cudaGetErrorString(err);

    // Copy results back to host
    err = cudaMemcpy(h_grad_input.data(), d_grad_input, h_grad_input.size() * sizeof(float), cudaMemcpyDeviceToHost);
    ASSERT_EQ(err, cudaSuccess) << "Failed to copy grad_input from device: " << cudaGetErrorString(err);

    err = cudaMemcpy(h_grad_weight.data(), d_grad_weight, h_grad_weight.size() * sizeof(float), cudaMemcpyDeviceToHost);
    ASSERT_EQ(err, cudaSuccess) << "Failed to copy grad_weight from device: " << cudaGetErrorString(err);

    err = cudaMemcpy(h_grad_offset.data(), d_grad_offset, h_grad_offset.size() * sizeof(float), cudaMemcpyDeviceToHost);
    ASSERT_EQ(err, cudaSuccess) << "Failed to copy grad_offset from device: " << cudaGetErrorString(err);

    // Compare with expected gradients
    // For grad_input
    for (int i = 0; i < in_h * in_w; i++) {
        EXPECT_NEAR(h_grad_input[i], expected_grad_input[i], 1e-3) << "grad_input mismatch at index " << i;
    }

    // For grad_weight
    for (int i = 0; i < ksize * ksize; i++) {
        EXPECT_NEAR(h_grad_weight[i], expected_grad_weight[i], 1e-3) << "grad_weight mismatch at index " << i;
    }

    // For grad_offset
    for (int i = 0; i < h_offset.size(); i++) {
        EXPECT_NEAR(h_grad_offset[i], expected_grad_offset[i], 1e-3) << "grad_offset mismatch at index " << i;
    }
}

/**
 * @brief Instantiate the forward pass test suite with different parameter sets
 *
 * This creates multiple test cases with different input sizes, kernel sizes, and strides
 * to thoroughly test the forward pass of deformable convolution.
 */
INSTANTIATE_TEST_SUITE_P(
    DeformConv2DForwardTests,
    DeformConv2DForwardTest,
    ::testing::Values(
        // in_h, in_w, ksize, stride, out_h
        std::make_tuple(7, 7, 3, 1, 5),     // Small input, 3x3 kernel
        std::make_tuple(16, 16, 3, 1, 14),  // Medium input, 3x3 kernel
        std::make_tuple(16, 16, 5, 1, 12),  // Medium input, 5x5 kernel
        std::make_tuple(32, 32, 3, 2, 15),  // Larger input, stride 2
        std::make_tuple(32, 32, 5, 2, 14)   // Larger input, 5x5 kernel, stride 2
    )
);

/**
 * @brief Instantiate the backward pass test suite with different parameter sets
 *
 * This creates multiple test cases with different input sizes, kernel sizes, and strides
 * to thoroughly test the backward pass of deformable convolution.
 */
INSTANTIATE_TEST_SUITE_P(
    DeformConv2DBackwardTests,
    DeformConv2DBackwardTest,
    ::testing::Values(
        // in_h, in_w, ksize, stride, out_h
        std::make_tuple(7, 7, 3, 1, 5),     // Small input, 3x3 kernel
        std::make_tuple(8, 8, 3, 1, 6),     // Small input, 3x3 kernel, different size
        std::make_tuple(9, 9, 3, 1, 7),     // Small input, 3x3 kernel, different size
        std::make_tuple(10, 10, 3, 1, 8),   // Small input, 3x3 kernel, different size
        std::make_tuple(12, 12, 3, 1, 10),  // Medium input, 3x3 kernel
        std::make_tuple(16, 16, 3, 1, 14),  // Medium input, 3x3 kernel
        std::make_tuple(16, 16, 5, 1, 12),  // Medium input, 5x5 kernel
        std::make_tuple(20, 20, 3, 1, 18),  // Medium input, 3x3 kernel, different size
        std::make_tuple(20, 20, 5, 1, 16),  // Medium input, 5x5 kernel, different size
        std::make_tuple(24, 24, 3, 2, 11),  // Medium input, stride 2
        std::make_tuple(24, 24, 5, 2, 10),  // Medium input, 5x5 kernel, stride 2
        std::make_tuple(32, 32, 3, 2, 15),  // Larger input, stride 2
        std::make_tuple(32, 32, 5, 2, 14)   // Larger input, 5x5 kernel, stride 2
    )
);

/**
 * @brief Test fixture class for gradient checking of deformable convolution
 *
 * This class sets up the test environment for gradient checking of deformable convolution.
 * It uses finite differences to numerically verify the correctness of the analytical gradients
 * computed by the backward pass.
 */
class DeformConv2DGradientCheckTest : public ::testing::TestWithParam<std::tuple<int, int, int, int, int>> {
protected:
    // Member variables
    int in_h, in_w;
    int ksize;
    int stride;
    int out_h, out_w;

    std::vector<float> h_input;
    std::vector<float> h_offset;
    std::vector<float> h_weight;
    std::vector<float> h_output;
    std::vector<float> h_grad_output;

    float* d_input = nullptr;
    float* d_offset = nullptr;
    float* d_weight = nullptr;
    float* d_output = nullptr;
    float* d_grad_output = nullptr;
    float* d_grad_input = nullptr;
    float* d_grad_weight = nullptr;
    float* d_grad_offset = nullptr;

    // Epsilon for finite difference approximation
    const float epsilon = 1e-4f;

    void SetUp() override {
        // Get parameters for the current test case
        std::tie(in_h, in_w, ksize, stride, out_h) = GetParam();

        // Calculate output width
        out_w = out_h; // Assuming square output for simplicity

        // Initialize input data
        h_input.resize(in_h * in_w);
        for (int i = 0; i < in_h * in_w; i++) {
            h_input[i] = static_cast<float>(i % 10) * 0.1f;
        }

        // Initialize offset data (small random values)
        h_offset.resize(out_h * out_w * 2 * ksize * ksize);
        for (int i = 0; i < h_offset.size(); i++) {
            h_offset[i] = (static_cast<float>(rand()) / RAND_MAX - 0.5f) * 0.2f;
        }

        // Initialize weight data
        h_weight.resize(ksize * ksize);
        for (int i = 0; i < ksize * ksize; i++) {
            h_weight[i] = static_cast<float>(i % 5) * 0.1f + 0.1f;
        }

        // Initialize gradient output data
        h_grad_output.resize(out_h * out_w);
        for (int i = 0; i < out_h * out_w; i++) {
            h_grad_output[i] = static_cast<float>(i % 7) * 0.1f + 0.05f;
        }

        // Allocate device memory
        cudaMalloc(&d_input, h_input.size() * sizeof(float));
        cudaMalloc(&d_offset, h_offset.size() * sizeof(float));
        cudaMalloc(&d_weight, h_weight.size() * sizeof(float));
        cudaMalloc(&d_output, out_h * out_w * sizeof(float));
        cudaMalloc(&d_grad_output, h_grad_output.size() * sizeof(float));
        cudaMalloc(&d_grad_input, h_input.size() * sizeof(float));
        cudaMalloc(&d_grad_weight, h_weight.size() * sizeof(float));
        cudaMalloc(&d_grad_offset, h_offset.size() * sizeof(float));

        // Copy data to device
        cudaMemcpy(d_input, h_input.data(), h_input.size() * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_offset, h_offset.data(), h_offset.size() * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_weight, h_weight.data(), h_weight.size() * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_grad_output, h_grad_output.data(), h_grad_output.size() * sizeof(float), cudaMemcpyHostToDevice);
    }

    void TearDown() override {
        if (d_input) cudaFree(d_input);
        if (d_offset) cudaFree(d_offset);
        if (d_weight) cudaFree(d_weight);
        if (d_output) cudaFree(d_output);
        if (d_grad_output) cudaFree(d_grad_output);
        if (d_grad_input) cudaFree(d_grad_input);
        if (d_grad_weight) cudaFree(d_grad_weight);
        if (d_grad_offset) cudaFree(d_grad_offset);

        d_input = nullptr;
        d_offset = nullptr;
        d_weight = nullptr;
        d_output = nullptr;
        d_grad_output = nullptr;
        d_grad_input = nullptr;
        d_grad_weight = nullptr;
        d_grad_offset = nullptr;
    }

    // Helper function to compute output with perturbed input
    std::vector<float> computeOutputWithPerturbedInput(int idx, float epsilon) {
        std::vector<float> perturbed_input = h_input;
        perturbed_input[idx] += epsilon;

        // Copy perturbed input to device
        cudaMemcpy(d_input, perturbed_input.data(), perturbed_input.size() * sizeof(float), cudaMemcpyHostToDevice);

        // Compute output
        dim3 block(16, 16);
        dim3 grid((out_w + block.x - 1) / block.x, (out_h + block.y - 1) / block.y);

        // Initialize output to zero
        cudaMemset(d_output, 0, out_h * out_w * sizeof(float));

        // Launch the kernel
        deform_conv2d_forward<<<grid, block>>>(d_input, d_offset, d_weight, d_output, in_h, in_w, out_h, out_w, ksize, stride);
        cudaDeviceSynchronize();

        // Copy result back to host
        std::vector<float> output(out_h * out_w);
        cudaMemcpy(output.data(), d_output, output.size() * sizeof(float), cudaMemcpyDeviceToHost);

        return output;
    }

    // Helper function to compute output with perturbed weight
    std::vector<float> computeOutputWithPerturbedWeight(int idx, float epsilon) {
        std::vector<float> perturbed_weight = h_weight;
        perturbed_weight[idx] += epsilon;

        // Copy perturbed weight to device
        cudaMemcpy(d_weight, perturbed_weight.data(), perturbed_weight.size() * sizeof(float), cudaMemcpyHostToDevice);

        // Compute output
        dim3 block(16, 16);
        dim3 grid((out_w + block.x - 1) / block.x, (out_h + block.y - 1) / block.y);

        // Initialize output to zero
        cudaMemset(d_output, 0, out_h * out_w * sizeof(float));

        // Launch the kernel
        deform_conv2d_forward<<<grid, block>>>(d_input, d_offset, d_weight, d_output, in_h, in_w, out_h, out_w, ksize, stride);
        cudaDeviceSynchronize();

        // Copy result back to host
        std::vector<float> output(out_h * out_w);
        cudaMemcpy(output.data(), d_output, output.size() * sizeof(float), cudaMemcpyDeviceToHost);

        return output;
    }

    // Helper function to compute output with perturbed offset
    std::vector<float> computeOutputWithPerturbedOffset(int idx, float epsilon) {
        std::vector<float> perturbed_offset = h_offset;
        perturbed_offset[idx] += epsilon;

        // Copy perturbed offset to device
        cudaMemcpy(d_offset, perturbed_offset.data(), perturbed_offset.size() * sizeof(float), cudaMemcpyHostToDevice);

        // Compute output
        dim3 block(16, 16);
        dim3 grid((out_w + block.x - 1) / block.x, (out_h + block.y - 1) / block.y);

        // Initialize output to zero
        cudaMemset(d_output, 0, out_h * out_w * sizeof(float));

        // Launch the kernel
        deform_conv2d_forward<<<grid, block>>>(d_input, d_offset, d_weight, d_output, in_h, in_w, out_h, out_w, ksize, stride);
        cudaDeviceSynchronize();

        // Copy result back to host
        std::vector<float> output(out_h * out_w);
        cudaMemcpy(output.data(), d_output, output.size() * sizeof(float), cudaMemcpyDeviceToHost);

        return output;
    }

    // Helper function to compute loss
    float computeLoss(const std::vector<float>& output) {
        float loss = 0.0f;
        for (int i = 0; i < output.size(); i++) {
            loss += output[i] * h_grad_output[i];
        }
        return loss;
    }
};

/**
 * @brief Test case for checking the correctness of input gradients
 *
 * This test verifies that the analytical gradients computed by the backward pass
 * match the numerical gradients computed using finite differences.
 */
TEST_P(DeformConv2DGradientCheckTest, InputGradientCheck) {
    // Skip large tests to save time
    if (in_h > 16 || in_w > 16) {
        GTEST_SKIP() << "Skipping large test case for gradient checking";
    }

    // Compute gradients using backward pass
    dim3 block(16, 16);
    dim3 grid((out_w + block.x - 1) / block.x, (out_h + block.y - 1) / block.y);

    // Initialize gradients to zero
    cudaMemset(d_grad_input, 0, h_input.size() * sizeof(float));
    cudaMemset(d_grad_weight, 0, h_weight.size() * sizeof(float));
    cudaMemset(d_grad_offset, 0, h_offset.size() * sizeof(float));

    // Launch the backward kernel
    deform_conv2d_backward<<<grid, block>>>(
        d_grad_output, d_grad_input, d_grad_weight, d_grad_offset,
        d_input, d_weight, d_offset, in_h, in_w, out_h, out_w, ksize, stride
    );
    cudaDeviceSynchronize();

    // Copy gradients back to host
    std::vector<float> grad_input(h_input.size());
    cudaMemcpy(grad_input.data(), d_grad_input, grad_input.size() * sizeof(float), cudaMemcpyDeviceToHost);

    // Check a subset of gradients using finite differences
    int num_checks = std::min(10, static_cast<int>(h_input.size()));
    for (int i = 0; i < num_checks; i++) {
        int idx = rand() % h_input.size();

        // Compute output with positive perturbation
        std::vector<float> output_plus = computeOutputWithPerturbedInput(idx, epsilon);
        float loss_plus = computeLoss(output_plus);

        // Compute output with negative perturbation
        std::vector<float> output_minus = computeOutputWithPerturbedInput(idx, -epsilon);
        float loss_minus = computeLoss(output_minus);

        // Compute numerical gradient
        float numerical_grad = (loss_plus - loss_minus) / (2 * epsilon);

        // Compare with analytical gradient
        EXPECT_NEAR(grad_input[idx], numerical_grad, 1e-2) << "Input gradient mismatch at index " << idx;
    }
}

/**
 * @brief Test case for checking the correctness of weight gradients
 *
 * This test verifies that the analytical gradients for weights computed by the backward pass
 * match the numerical gradients computed using finite differences.
 */
TEST_P(DeformConv2DGradientCheckTest, WeightGradientCheck) {
    // Skip large tests to save time
    if (in_h > 16 || in_w > 16) {
        GTEST_SKIP() << "Skipping large test case for gradient checking";
    }

    // Compute gradients using backward pass
    dim3 block(16, 16);
    dim3 grid((out_w + block.x - 1) / block.x, (out_h + block.y - 1) / block.y);

    // Initialize gradients to zero
    cudaMemset(d_grad_input, 0, h_input.size() * sizeof(float));
    cudaMemset(d_grad_weight, 0, h_weight.size() * sizeof(float));
    cudaMemset(d_grad_offset, 0, h_offset.size() * sizeof(float));

    // Launch the backward kernel
    deform_conv2d_backward<<<grid, block>>>(
        d_grad_output, d_grad_input, d_grad_weight, d_grad_offset,
        d_input, d_weight, d_offset, in_h, in_w, out_h, out_w, ksize, stride
    );
    cudaDeviceSynchronize();

    // Copy gradients back to host
    std::vector<float> grad_weight(h_weight.size());
    cudaMemcpy(grad_weight.data(), d_grad_weight, grad_weight.size() * sizeof(float), cudaMemcpyDeviceToHost);

    // Check all weight gradients using finite differences
    for (int i = 0; i < h_weight.size(); i++) {
        // Compute output with positive perturbation
        std::vector<float> output_plus = computeOutputWithPerturbedWeight(i, epsilon);
        float loss_plus = computeLoss(output_plus);

        // Compute output with negative perturbation
        std::vector<float> output_minus = computeOutputWithPerturbedWeight(i, -epsilon);
        float loss_minus = computeLoss(output_minus);

        // Compute numerical gradient
        float numerical_grad = (loss_plus - loss_minus) / (2 * epsilon);

        // Compare with analytical gradient
        EXPECT_NEAR(grad_weight[i], numerical_grad, 1e-2) << "Weight gradient mismatch at index " << i;
    }
}

/**
 * @brief Test case for checking the correctness of offset gradients
 *
 * This test verifies that the analytical gradients for offsets computed by the backward pass
 * match the numerical gradients computed using finite differences.
 */
TEST_P(DeformConv2DGradientCheckTest, OffsetGradientCheck) {
    // Skip large tests to save time
    if (in_h > 16 || in_w > 16) {
        GTEST_SKIP() << "Skipping large test case for gradient checking";
    }

    // Compute gradients using backward pass
    dim3 block(16, 16);
    dim3 grid((out_w + block.x - 1) / block.x, (out_h + block.y - 1) / block.y);

    // Initialize gradients to zero
    cudaMemset(d_grad_input, 0, h_input.size() * sizeof(float));
    cudaMemset(d_grad_weight, 0, h_weight.size() * sizeof(float));
    cudaMemset(d_grad_offset, 0, h_offset.size() * sizeof(float));

    // Launch the backward kernel
    deform_conv2d_backward<<<grid, block>>>(
        d_grad_output, d_grad_input, d_grad_weight, d_grad_offset,
        d_input, d_weight, d_offset, in_h, in_w, out_h, out_w, ksize, stride
    );
    cudaDeviceSynchronize();

    // Copy gradients back to host
    std::vector<float> grad_offset(h_offset.size());
    cudaMemcpy(grad_offset.data(), d_grad_offset, grad_offset.size() * sizeof(float), cudaMemcpyDeviceToHost);

    // Check a subset of offset gradients using finite differences
    int num_checks = std::min(20, static_cast<int>(h_offset.size()));
    for (int i = 0; i < num_checks; i++) {
        int idx = rand() % h_offset.size();

        // Compute output with positive perturbation
        std::vector<float> output_plus = computeOutputWithPerturbedOffset(idx, epsilon);
        float loss_plus = computeLoss(output_plus);

        // Compute output with negative perturbation
        std::vector<float> output_minus = computeOutputWithPerturbedOffset(idx, -epsilon);
        float loss_minus = computeLoss(output_minus);

        // Compute numerical gradient
        float numerical_grad = (loss_plus - loss_minus) / (2 * epsilon);

        // Compare with analytical gradient
        EXPECT_NEAR(grad_offset[idx], numerical_grad, 1e-2) << "Offset gradient mismatch at index " << idx;
    }
}

/**
 * @brief Instantiate the gradient check test suite with a subset of parameters
 *
 * This creates test cases for gradient checking with smaller input sizes to keep
 * test execution time reasonable.
 */
INSTANTIATE_TEST_SUITE_P(
    DeformConv2DGradientCheckTests,
    DeformConv2DGradientCheckTest,
    ::testing::Values(
        // in_h, in_w, ksize, stride, out_h
        std::make_tuple(7, 7, 3, 1, 5),     // Small input, 3x3 kernel
        std::make_tuple(10, 10, 3, 1, 8),   // Small input, 3x3 kernel
        std::make_tuple(12, 12, 5, 1, 8)    // Small input, 5x5 kernel
    )
);

/**
 * @brief Test fixture class for performance testing of deformable convolution
 *
 * This class sets up the test environment for measuring the performance of
 * both forward and backward passes of deformable convolution with different
 * input sizes, kernel sizes, and strides.
 */
class DeformConv2DPerformanceTest : public ::testing::TestWithParam<std::tuple<int, int, int, int, int>> {
protected:
    // Member variables
    int in_h, in_w;
    int ksize;
    int stride;
    int out_h, out_w;

    std::vector<float> h_input;
    std::vector<float> h_offset;
    std::vector<float> h_weight;
    std::vector<float> h_output;
    std::vector<float> h_grad_output;
    std::vector<float> h_grad_input;
    std::vector<float> h_grad_weight;
    std::vector<float> h_grad_offset;

    float* d_input = nullptr;
    float* d_offset = nullptr;
    float* d_weight = nullptr;
    float* d_output = nullptr;
    float* d_grad_output = nullptr;
    float* d_grad_input = nullptr;
    float* d_grad_weight = nullptr;
    float* d_grad_offset = nullptr;

    // CUDA events for timing
    cudaEvent_t start_event;
    cudaEvent_t stop_event;

    // Number of iterations for timing
    const int num_iterations = 100;

    void SetUp() override {
        // Get parameters for the current test case
        std::tie(in_h, in_w, ksize, stride, out_h) = GetParam();

        // Calculate output width
        out_w = out_h; // Assuming square output for simplicity

        // Initialize input data
        h_input.resize(in_h * in_w);
        for (int i = 0; i < in_h * in_w; i++) {
            h_input[i] = static_cast<float>(i % 10) * 0.1f;
        }

        // Initialize offset data (small random values)
        h_offset.resize(out_h * out_w * 2 * ksize * ksize);
        for (int i = 0; i < h_offset.size(); i++) {
            h_offset[i] = (static_cast<float>(rand()) / RAND_MAX - 0.5f) * 0.2f;
        }

        // Initialize weight data
        h_weight.resize(ksize * ksize);
        for (int i = 0; i < ksize * ksize; i++) {
            h_weight[i] = static_cast<float>(i % 5) * 0.1f + 0.1f;
        }

        // Initialize gradient output data
        h_grad_output.resize(out_h * out_w);
        for (int i = 0; i < out_h * out_w; i++) {
            h_grad_output[i] = static_cast<float>(i % 7) * 0.1f + 0.05f;
        }

        // Initialize gradient buffers
        h_grad_input.resize(in_h * in_w, 0.0f);
        h_grad_weight.resize(ksize * ksize, 0.0f);
        h_grad_offset.resize(h_offset.size(), 0.0f);

        // Allocate device memory
        cudaMalloc(&d_input, h_input.size() * sizeof(float));
        cudaMalloc(&d_offset, h_offset.size() * sizeof(float));
        cudaMalloc(&d_weight, h_weight.size() * sizeof(float));
        cudaMalloc(&d_output, out_h * out_w * sizeof(float));
        cudaMalloc(&d_grad_output, h_grad_output.size() * sizeof(float));
        cudaMalloc(&d_grad_input, h_grad_input.size() * sizeof(float));
        cudaMalloc(&d_grad_weight, h_grad_weight.size() * sizeof(float));
        cudaMalloc(&d_grad_offset, h_grad_offset.size() * sizeof(float));

        // Copy data to device
        cudaMemcpy(d_input, h_input.data(), h_input.size() * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_offset, h_offset.data(), h_offset.size() * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_weight, h_weight.data(), h_weight.size() * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_grad_output, h_grad_output.data(), h_grad_output.size() * sizeof(float), cudaMemcpyHostToDevice);

        // Create CUDA events for timing
        cudaEventCreate(&start_event);
        cudaEventCreate(&stop_event);
    }

    void TearDown() override {
        if (d_input) cudaFree(d_input);
        if (d_offset) cudaFree(d_offset);
        if (d_weight) cudaFree(d_weight);
        if (d_output) cudaFree(d_output);
        if (d_grad_output) cudaFree(d_grad_output);
        if (d_grad_input) cudaFree(d_grad_input);
        if (d_grad_weight) cudaFree(d_grad_weight);
        if (d_grad_offset) cudaFree(d_grad_offset);

        cudaEventDestroy(start_event);
        cudaEventDestroy(stop_event);

        d_input = nullptr;
        d_offset = nullptr;
        d_weight = nullptr;
        d_output = nullptr;
        d_grad_output = nullptr;
        d_grad_input = nullptr;
        d_grad_weight = nullptr;
        d_grad_offset = nullptr;
    }
};

/**
 * @brief Test case for measuring the performance of the forward pass
 *
 * This test measures the average execution time of the forward pass of deformable convolution
 * over multiple iterations with different input sizes, kernel sizes, and strides.
 */
TEST_P(DeformConv2DPerformanceTest, ForwardPassPerformance) {
    // Define grid and block dimensions
    dim3 block(16, 16);
    dim3 grid((out_w + block.x - 1) / block.x, (out_h + block.y - 1) / block.y);

    // Warm-up run
    deform_conv2d_forward<<<grid, block>>>(d_input, d_offset, d_weight, d_output, in_h, in_w, out_h, out_w, ksize, stride);
    cudaDeviceSynchronize();

    // Start timing
    cudaEventRecord(start_event);

    // Run multiple iterations
    for (int i = 0; i < num_iterations; i++) {
        deform_conv2d_forward<<<grid, block>>>(d_input, d_offset, d_weight, d_output, in_h, in_w, out_h, out_w, ksize, stride);
    }

    // Stop timing
    cudaEventRecord(stop_event);
    cudaEventSynchronize(stop_event);

    // Calculate elapsed time
    float elapsed_time_ms;
    cudaEventElapsedTime(&elapsed_time_ms, start_event, stop_event);

    // Calculate average time per iteration
    float avg_time_ms = elapsed_time_ms / num_iterations;

    // Print performance results
    std::cout << "Forward Pass Performance (" << in_h << "x" << in_w << ", ksize=" << ksize << ", stride=" << stride << "): "
              << avg_time_ms << " ms per iteration" << std::endl;

    // Ensure the kernel executed successfully
    cudaError_t err = cudaGetLastError();
    ASSERT_EQ(err, cudaSuccess) << "Kernel execution failed: " << cudaGetErrorString(err);
}

/**
 * @brief Test case for measuring the performance of the backward pass
 *
 * This test measures the average execution time of the backward pass of deformable convolution
 * over multiple iterations with different input sizes, kernel sizes, and strides.
 */
TEST_P(DeformConv2DPerformanceTest, BackwardPassPerformance) {
    // Define grid and block dimensions
    dim3 block(16, 16);
    dim3 grid((out_w + block.x - 1) / block.x, (out_h + block.y - 1) / block.y);

    // Initialize gradients to zero
    cudaMemset(d_grad_input, 0, h_input.size() * sizeof(float));
    cudaMemset(d_grad_weight, 0, h_weight.size() * sizeof(float));
    cudaMemset(d_grad_offset, 0, h_offset.size() * sizeof(float));

    // Warm-up run
    deform_conv2d_backward<<<grid, block>>>(
        d_grad_output, d_grad_input, d_grad_weight, d_grad_offset,
        d_input, d_weight, d_offset, in_h, in_w, out_h, out_w, ksize, stride
    );
    cudaDeviceSynchronize();

    // Start timing
    cudaEventRecord(start_event);

    // Run multiple iterations
    for (int i = 0; i < num_iterations; i++) {
        // Reset gradients for each iteration
        cudaMemset(d_grad_input, 0, h_input.size() * sizeof(float));
        cudaMemset(d_grad_weight, 0, h_weight.size() * sizeof(float));
        cudaMemset(d_grad_offset, 0, h_offset.size() * sizeof(float));

        deform_conv2d_backward<<<grid, block>>>(
            d_grad_output, d_grad_input, d_grad_weight, d_grad_offset,
            d_input, d_weight, d_offset, in_h, in_w, out_h, out_w, ksize, stride
        );
    }

    // Stop timing
    cudaEventRecord(stop_event);
    cudaEventSynchronize(stop_event);

    // Calculate elapsed time
    float elapsed_time_ms;
    cudaEventElapsedTime(&elapsed_time_ms, start_event, stop_event);

    // Calculate average time per iteration
    float avg_time_ms = elapsed_time_ms / num_iterations;

    // Print performance results
    std::cout << "Backward Pass Performance (" << in_h << "x" << in_w << ", ksize=" << ksize << ", stride=" << stride << "): "
              << avg_time_ms << " ms per iteration" << std::endl;

    // Ensure the kernel executed successfully
    cudaError_t err = cudaGetLastError();
    ASSERT_EQ(err, cudaSuccess) << "Kernel execution failed: " << cudaGetErrorString(err);
}

/**
 * @brief Instantiate the performance test suite with different parameters
 *
 * This creates test cases for performance testing with various input sizes, kernel sizes,
 * and strides to measure the performance characteristics of the implementation.
 */
INSTANTIATE_TEST_SUITE_P(
    DeformConv2DPerformanceTests,
    DeformConv2DPerformanceTest,
    ::testing::Values(
        // in_h, in_w, ksize, stride, out_h
        std::make_tuple(32, 32, 3, 1, 30),    // Medium input, 3x3 kernel
        std::make_tuple(64, 64, 3, 1, 62),    // Medium input, 3x3 kernel
        std::make_tuple(128, 128, 3, 1, 126), // Large input, 3x3 kernel
        std::make_tuple(32, 32, 5, 1, 28),    // Medium input, 5x5 kernel
        std::make_tuple(64, 64, 5, 1, 60),    // Medium input, 5x5 kernel
        std::make_tuple(32, 32, 3, 2, 15),    // Medium input, stride 2
        std::make_tuple(64, 64, 3, 2, 31),    // Medium input, stride 2
        std::make_tuple(128, 128, 3, 2, 63)   // Large input, stride 2
    )
);
