// File: cudnn_training.cu
#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include "cuda_device_runtime_api.h"
#include "device_launch_parameters.h"
#include <cudnn.h>
#include <random>
#include <chrono>
#include <iomanip>

// Error checking macro for cuDNN
#define CUDNN_CHECK(call) do { \
    cudnnStatus_t status = call; \
    if (status != CUDNN_STATUS_SUCCESS) { \
        std::cerr << "cuDNN error at " << __FILE__ << ":" << __LINE__ << ": " \
                  << cudnnGetErrorString(status) << std::endl; \
        return -1; \
    } \
} while(0)

// Error checking macro for CUDA
#define CUDA_CHECK(call) do { \
    cudaError_t error = call; \
    if (error != cudaSuccess) { \
        std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": " \
                  << cudaGetErrorString(error) << std::endl; \
        return -1; \
    } \
} while(0)

// Kernel for initializing tensors with random values
__global__ void initialize_tensor(float* data, int size, float mean = 0.0f, float stddev = 0.01f) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        // Use a deterministic seeded approach for reproducibility
        unsigned int seed = idx;
        seed = (seed * 0x5DEECE66DL + 0xBL) & ((1L << 48) - 1);
        float random_value = (seed / (float)(1L << 48));
        // Convert to normal distribution using Box-Muller transform
        if (idx % 2 == 0 && idx + 1 < size) {
            float u1 = random_value;
            float u2 = (((seed * 0x5DEECE66DL + 0xBL) & ((1L << 48) - 1)) / (float)(1L << 48));
            float z0 = sqrtf(-2.0f * logf(u1)) * cosf(2.0f * 3.14159f * u2);
            data[idx] = mean + stddev * z0;
        }
        else if (idx % 2 == 1) {
            float u1 = (((seed * 0x5DEECE66DL + 0xBL) & ((1L << 48) - 1)) / (float)(1L << 48));
            float u2 = random_value;
            float z1 = sqrtf(-2.0f * logf(u1)) * sinf(2.0f * 3.14159f * u2);
            data[idx] = mean + stddev * z1;
        }
        else {
            data[idx] = mean + stddev * (random_value * 2.0f - 1.0f);
        }
    }
}

// Kernel to compute Mean Squared Error (MSE) loss
__global__ void mse_loss_kernel(const float* output, const float* target, float* loss, float* grad_output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float diff = output[idx] - target[idx];
        atomicAdd(loss, diff * diff / (2 * size)); // Sum of squared differences / 2N
        grad_output[idx] = diff / size; // Gradient of MSE loss is (y - t) / N
    }
}

// Kernel to update weights using gradients (SGD)
__global__ void sgd_update_kernel(float* weights, const float* gradients, int size, float learning_rate) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        weights[idx] -= learning_rate * gradients[idx];
    }
}

int cudnn_loop() {
    // Define tensor dimensions
    const int batch_size = 32;
    const int channels = 3;
    const int height = 32;
    const int width = 32;

    // Define convolution parameters
    const int filter_count = 16;     // Number of filters
    const int kernel_size = 3;       // 3x3 kernel
    const int padding = kernel_size / 2;  // Same padding
    const int stride = 1;

    // Training parameters
    const int epochs = 5000;
    const float learning_rate = 0.01f;

    std::cout << "=== 2D Convolution Training with cuDNN ===" << std::endl;
    std::cout << "Batch size: " << batch_size << ", Input: " << height << "x" << width
        << "x" << channels << ", Filters: " << filter_count << "x" << kernel_size << "x" << kernel_size
        << std::endl;

    // Calculate sizes for memory allocation
    const int input_size = batch_size * channels * height * width;
    const int filter_size = filter_count * channels * kernel_size * kernel_size;
    const int output_height = (height + 2 * padding - kernel_size) / stride + 1;
    const int output_width = (width + 2 * padding - kernel_size) / stride + 1;
    const int output_size = batch_size * filter_count * output_height * output_width;

    // Allocate device memory
    float* d_input, * d_filters, * d_output, * d_target;
    float* d_dinput, * d_dfilters, * d_doutput; // Gradients, gradients require double the memoery fro training
    float* d_loss;

	// CUDA_CHECK: macro for error checking in CUDA runtime API
    CUDA_CHECK(cudaMalloc(&d_input, input_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_filters, filter_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, output_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_target, output_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dinput, input_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dfilters, filter_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_doutput, output_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_loss, sizeof(float)));

    // Initialize input tensor and filters with random values
    dim3 block(256);
    dim3 grid_input((input_size + block.x - 1) / block.x);
    dim3 grid_filters((filter_size + block.x - 1) / block.x);
    dim3 grid_output((output_size + block.x - 1) / block.x);

    // Initialize input and filters
    initialize_tensor << <grid_input, block >> > (d_input, input_size, 0.0f, 0.1f);
    initialize_tensor << <grid_filters, block >> > (d_filters, filter_size, 0.0f, 0.01f);
    initialize_tensor << <grid_output, block >> > (d_target, output_size, 0.5f, 0.1f); // Target values

    CUDA_CHECK(cudaDeviceSynchronize());

    // Initialize cuDNN
    cudnnHandle_t cudnn;  // cudNN Handle: 
	CUDNN_CHECK(cudnnCreate(&cudnn));  // creating a handle for cuDNN

    // Create tensor descriptors
    cudnnTensorDescriptor_t input_desc, output_desc, doutput_desc, dinput_desc;
    cudnnFilterDescriptor_t filter_desc, dfilter_desc;
    cudnnConvolutionDescriptor_t conv_desc;

    CUDNN_CHECK(cudnnCreateTensorDescriptor(&input_desc));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&output_desc));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&doutput_desc));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&dinput_desc));
    CUDNN_CHECK(cudnnCreateFilterDescriptor(&filter_desc));
    CUDNN_CHECK(cudnnCreateFilterDescriptor(&dfilter_desc));
    CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&conv_desc));

    // Set tensor descriptors
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
        batch_size, channels, height, width));

    CUDNN_CHECK(cudnnSetFilter4dDescriptor(filter_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
        filter_count, channels, kernel_size, kernel_size));

    CUDNN_CHECK(cudnnSetConvolution2dDescriptor(conv_desc, padding, padding, stride, stride, 1, 1,
        CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

    // Get output dimensions
    int out_n, out_c, out_h, out_w;
    CUDNN_CHECK(cudnnGetConvolution2dForwardOutputDim(conv_desc, input_desc, filter_desc,
        &out_n, &out_c, &out_h, &out_w));

    // Confirm output dimensions match our expectations
    if (out_n != batch_size || out_c != filter_count ||
        out_h != output_height || out_w != output_width) {
        std::cerr << "Output dimensions mismatch!" << std::endl;
        return -1;
    }

    // Set output tensor descriptor
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(output_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
        out_n, out_c, out_h, out_w));

    // Set gradient descriptors (same as corresponding forward pass descriptors)
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(doutput_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
        out_n, out_c, out_h, out_w));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(dinput_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
        batch_size, channels, height, width));
    CUDNN_CHECK(cudnnSetFilter4dDescriptor(dfilter_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
        filter_count, channels, kernel_size, kernel_size));

    // Find algorithm for forward convolution
    cudnnConvolutionFwdAlgo_t fwd_algo;
    int returnedAlgoCount;
    cudnnConvolutionFwdAlgoPerf_t fwd_algo_perf;
    CUDNN_CHECK(cudnnFindConvolutionForwardAlgorithm(cudnn, input_desc, filter_desc,
        conv_desc, output_desc, 1,
        &returnedAlgoCount, &fwd_algo_perf));
    fwd_algo = fwd_algo_perf.algo;

    // Find algorithms for backward convolution
    cudnnConvolutionBwdFilterAlgo_t bwd_filter_algo;
    cudnnConvolutionBwdFilterAlgoPerf_t bwd_filter_algo_perf;
    CUDNN_CHECK(cudnnFindConvolutionBackwardFilterAlgorithm(cudnn, input_desc, doutput_desc,
        conv_desc, dfilter_desc, 1,
        &returnedAlgoCount, &bwd_filter_algo_perf));
    bwd_filter_algo = bwd_filter_algo_perf.algo;

    cudnnConvolutionBwdDataAlgo_t bwd_data_algo;
    cudnnConvolutionBwdDataAlgoPerf_t bwd_data_algo_perf;
    CUDNN_CHECK(cudnnFindConvolutionBackwardDataAlgorithm(cudnn, filter_desc, doutput_desc,
        conv_desc, dinput_desc, 1,
        &returnedAlgoCount, &bwd_data_algo_perf));
    bwd_data_algo = bwd_data_algo_perf.algo;

    // Allocate workspace memory for cuDNN algorithms
    size_t fwd_workspace_size, bwd_filter_workspace_size, bwd_data_workspace_size;
    CUDNN_CHECK(cudnnGetConvolutionForwardWorkspaceSize(cudnn, input_desc, filter_desc,
        conv_desc, output_desc, fwd_algo,
        &fwd_workspace_size));

    CUDNN_CHECK(cudnnGetConvolutionBackwardFilterWorkspaceSize(cudnn, input_desc, doutput_desc,
        conv_desc, dfilter_desc, bwd_filter_algo,
        &bwd_filter_workspace_size));

    CUDNN_CHECK(cudnnGetConvolutionBackwardDataWorkspaceSize(cudnn, filter_desc, doutput_desc,
        conv_desc, dinput_desc, bwd_data_algo,
        &bwd_data_workspace_size));

    // Use the largest workspace size
    size_t max_workspace_size = std::max({ fwd_workspace_size, bwd_filter_workspace_size, bwd_data_workspace_size });
    void* workspace;
    CUDA_CHECK(cudaMalloc(&workspace, max_workspace_size));

    // Scaling factors for cuDNN operations
    float alpha = 1.0f;
    float beta = 0.0f;

    // Host memory for reporting loss
    float h_loss;

    // Training loop
    std::cout << "\nTraining for " << epochs << " epochs..." << std::endl;
    for (int epoch = 0; epoch < epochs; epoch++) {
        // Reset loss for this epoch
        CUDA_CHECK(cudaMemset(d_loss, 0, sizeof(float)));

        // Forward pass
        CUDNN_CHECK(cudnnConvolutionForward(cudnn, &alpha, input_desc, d_input,
            filter_desc, d_filters, conv_desc, fwd_algo,
            workspace, fwd_workspace_size, &beta,
            output_desc, d_output));

        // Compute loss and gradient of output
        mse_loss_kernel << <grid_output, block >> > (d_output, d_target, d_loss, d_doutput, output_size);

        // Backward pass for filter gradients
        CUDNN_CHECK(cudnnConvolutionBackwardFilter(cudnn, &alpha, input_desc, d_input,
            doutput_desc, d_doutput, conv_desc, bwd_filter_algo,
            workspace, bwd_filter_workspace_size, &beta,
            dfilter_desc, d_dfilters));

        // Backward pass for input gradients (needed if this is part of a larger network)
        CUDNN_CHECK(cudnnConvolutionBackwardData(cudnn, &alpha, filter_desc, d_filters,
            doutput_desc, d_doutput, conv_desc, bwd_data_algo,
            workspace, bwd_data_workspace_size, &beta,
            dinput_desc, d_dinput));

        // Update filter weights with SGD
        sgd_update_kernel << <grid_filters, block >> > (d_filters, d_dfilters, filter_size, learning_rate);

        // Get loss value for reporting
        CUDA_CHECK(cudaMemcpy(&h_loss, d_loss, sizeof(float), cudaMemcpyDeviceToHost));

        if (epoch % 5 == 0 || epoch == epochs - 1) {
            std::cout << "Epoch " << std::setw(3) << epoch << ", Loss: " << std::fixed << std::setprecision(6) << h_loss << std::endl;
        }
    }

    // Clean up
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(input_desc));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(output_desc));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(doutput_desc));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(dinput_desc));
    CUDNN_CHECK(cudnnDestroyFilterDescriptor(filter_desc));
    CUDNN_CHECK(cudnnDestroyFilterDescriptor(dfilter_desc));
    CUDNN_CHECK(cudnnDestroyConvolutionDescriptor(conv_desc));
    CUDNN_CHECK(cudnnDestroy(cudnn));

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_filters));
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaFree(d_target));
    CUDA_CHECK(cudaFree(d_dinput));
    CUDA_CHECK(cudaFree(d_dfilters));
    CUDA_CHECK(cudaFree(d_doutput));
    CUDA_CHECK(cudaFree(d_loss));
    CUDA_CHECK(cudaFree(workspace));

    std::cout << "\nTraining complete!" << std::endl;
    return 0;
}
