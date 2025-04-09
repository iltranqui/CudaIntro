#pragma once

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <random>
#include <iostream>
#include <fstream>
#include <cuda_runtime.h>

// Constants
constexpr int INPUT_CHANNELS = 1;      // Grayscale images
constexpr int INPUT_HEIGHT = 28;       // MNIST image height
constexpr int INPUT_WIDTH = 28;        // MNIST image width
constexpr int CONV1_FILTERS = 32;      // First convolutional layer filters
constexpr int CONV1_KERNEL_SIZE = 5;   // 5x5 kernel
constexpr int CONV1_PADDING = 2;       // Same padding
constexpr int CONV1_STRIDE = 1;        // Stride of 1
constexpr int POOL1_SIZE = 2;          // 2x2 max pooling
constexpr int POOL1_STRIDE = 2;        // Stride of 2
constexpr int CONV2_FILTERS = 64;      // Second convolutional layer filters
constexpr int CONV2_KERNEL_SIZE = 3;   // 3x3 kernel
constexpr int CONV2_PADDING = 1;       // Same padding
constexpr int CONV2_STRIDE = 1;        // Stride of 1
constexpr int POOL2_SIZE = 2;          // 2x2 max pooling
constexpr int POOL2_STRIDE = 2;        // Stride of 2
constexpr int FC1_SIZE = 1024;         // First fully connected layer
constexpr int OUTPUT_SIZE = 10;        // 10 digits (0-9)
constexpr int BATCH_SIZE = 128;        // Batch size
constexpr float LEARNING_RATE = 0.001f; // Learning rate
constexpr int NUM_EPOCHS = 10;         // Number of epochs

// CUDA Error checking
#define CHECK_CUDA_ERROR(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error in " << __FILE__ << " at line " << __LINE__ << ": " << \
            cudaGetErrorString(err) << " (" << err << ")" << std::endl; \
        exit(1); \
    } \
}

// Forward declarations for CUDA kernels
__global__ void conv2dKernel(float* input, float* output, float* weights, float* bias, 
                            int batch_size, int in_channels, int out_channels,
                            int in_height, int in_width, int kernel_size, 
                            int out_height, int out_width, int stride, int padding);
                            
__global__ void maxPoolKernel(float* input, float* output, 
                             int batch_size, int channels,
                             int in_height, int in_width, 
                             int pool_size, int stride);
                             
__global__ void reluActivationKernel(float* data, int size);

__global__ void fullyConnectedForwardKernel(float* input, float* output, float* weights, float* bias,
                                          int batch_size, int in_features, int out_features);
                                          
__global__ void softmaxKernel(float* output, int batch_size, int features);

__global__ void computeLossKernel(float* output, int* labels, float* loss, int batch_size, int num_classes);

__global__ void softmaxGradientKernel(float* output, int* labels, float* grad_output, 
                                     int batch_size, int num_classes);
                                     
__global__ void fullyConnectedBackwardKernel(float* grad_output, float* input, float* weights,
                                           float* grad_weights, float* grad_bias, float* grad_input,
                                           int batch_size, int in_features, int out_features);
                                           
__global__ void maxPoolBackwardKernel(float* grad_output, float* input, float* output, float* grad_input,
                                    int batch_size, int channels, 
                                    int in_height, int in_width,
                                    int pool_size, int stride);
                                    
__global__ void conv2dBackwardKernel(float* grad_output, float* input, float* weights,
                                   float* grad_weights, float* grad_bias, float* grad_input,
                                   int batch_size, int in_channels, int out_channels,
                                   int in_height, int in_width, int kernel_size,
                                   int out_height, int out_width, int stride, int padding);
                                   
__global__ void reluBackwardKernel(float* grad_output, float* input, float* grad_input, int size);

__global__ void updateParametersKernel(float* weights, float* grad_weights, float* bias, float* grad_bias,
                                     int size_weights, int size_bias, float learning_rate);

// MNIST Dataset class
class MNISTDataset {
public:
    MNISTDataset(const std::string& csv_file, const std::string& base_path);

    int getNumSamples() const { return static_cast<int>(image_paths.size()); }
    
    void getBatch(int batch_idx, int batch_size, float* images, int* labels);
    
    int getNumBatches(int batch_size) const {
        return (getNumSamples() + batch_size - 1) / batch_size;
    }

private:
    std::vector<std::string> image_paths;
    std::vector<int> labels;
    std::string base_path;
};

// CNN class
class ConvolutionalNeuralNetwork {
public:
    ConvolutionalNeuralNetwork();
    ~ConvolutionalNeuralNetwork();

    void train(MNISTDataset& train_dataset, MNISTDataset& val_dataset);
    float evaluate(MNISTDataset& dataset);

private:
    // Host memory for weights and biases
    std::vector<float> h_conv1_weights;
    std::vector<float> h_conv1_bias;
    std::vector<float> h_conv2_weights;
    std::vector<float> h_conv2_bias;
    std::vector<float> h_fc1_weights;
    std::vector<float> h_fc1_bias;
    std::vector<float> h_fc2_weights;
    std::vector<float> h_fc2_bias;
    
    // Device memory for weights and biases
    float* d_conv1_weights;
    float* d_conv1_bias;
    float* d_conv2_weights;
    float* d_conv2_bias;
    float* d_fc1_weights;
    float* d_fc1_bias;
    float* d_fc2_weights;
    float* d_fc2_bias;
    
    // Temporary buffers for forward and backward pass
    float* d_input;
    float* d_conv1_output;
    float* d_pool1_output;
    float* d_conv2_output;
    float* d_pool2_output;
    float* d_fc1_output;
    float* d_output;
    int* d_labels;
    float* d_loss;
    
    // Gradients
    float* d_grad_conv1_weights;
    float* d_grad_conv1_bias;
    float* d_grad_conv2_weights;
    float* d_grad_conv2_bias;
    float* d_grad_fc1_weights;
    float* d_grad_fc1_bias;
    float* d_grad_fc2_weights;
    float* d_grad_fc2_bias;
    float* d_grad_output;
    float* d_grad_fc1_output;
    float* d_grad_pool2_output;
    float* d_grad_conv2_output;
    float* d_grad_pool1_output;
    float* d_grad_conv1_output;
    
    // Helper methods
    void initializeWeights();
    void allocateMemory();
    void freeMemory();
    void forwardPass(float* d_input, int* d_labels);
    void backwardPass();
    void updateParameters();
    
    // Calculated dimensions
    int conv1_output_height;
    int conv1_output_width;
    int pool1_output_height;
    int pool1_output_width;
    int conv2_output_height;
    int conv2_output_width;
    int pool2_output_height;
    int pool2_output_width;
    int flattened_size;
};
