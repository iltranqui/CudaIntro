#pragma once

#include <opencv4/opencv2/opencv_modules.hpp>
#include <opencv4/opencv2/core.hpp>
#include <opencv4/opencv2/imgproc.hpp>
#include <opencv4/opencv2/highgui.hpp>
#include <opencv4/opencv2/imgcodecs.hpp>
#include <vector>
#include <string>
#include <random>
#include <iostream>
#include <fstream>
#include <cuda_runtime.h>
#include "mnist_kernels.cuh"

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

// New third convolutional layer
constexpr int CONV3_FILTERS = 128;     // Third convolutional layer filters
constexpr int CONV3_KERNEL_SIZE = 3;   // 3x3 kernel
constexpr int CONV3_PADDING = 1;       // Same padding
constexpr int CONV3_STRIDE = 1;        // Stride of 1
constexpr int POOL3_SIZE = 2;          // 2x2 max pooling
constexpr int POOL3_STRIDE = 2;        // Stride of 2

// New fourth convolutional layer
constexpr int CONV4_FILTERS = 256;     // Fourth convolutional layer filters
constexpr int CONV4_KERNEL_SIZE = 3;   // 3x3 kernel
constexpr int CONV4_PADDING = 1;       // Same padding
constexpr int CONV4_STRIDE = 1;        // Stride of 1
constexpr int POOL4_SIZE = 2;          // 2x2 max pooling
constexpr int POOL4_STRIDE = 2;        // Stride of 2

// New fifth convolutional layer
constexpr int CONV5_FILTERS = 512;     // Fifth convolutional layer filters
constexpr int CONV5_KERNEL_SIZE = 3;   // 3x3 kernel
constexpr int CONV5_PADDING = 1;       // Same padding
constexpr int CONV5_STRIDE = 1;        // Stride of 1

constexpr int FC1_SIZE = 2048;         // First fully connected layer (increased)
constexpr int FC2_SIZE = 1024;         // Second fully connected layer (increased)
constexpr int FC3_SIZE = 512;          // Third fully connected layer (increased)
constexpr int FC4_SIZE = 256;          // Fourth fully connected layer (new)
constexpr int FC5_SIZE = 128;          // Fifth fully connected layer (new)
constexpr int OUTPUT_SIZE = 10;        // 10 digits (0-9)
constexpr int BATCH_SIZE = 512;        // Batch size (adjusted for deeper network)
constexpr float LEARNING_RATE = 0.0005f; // Learning rate (reduced for deeper network)
constexpr int NUM_EPOCHS = 10;         // Number of epochs (increased for better training)

// CUDA Error checking
#define CHECK_CUDA_ERROR(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error in " << __FILE__ << " at line " << __LINE__ << ": " << \
            cudaGetErrorString(err) << " (" << err << ")" << std::endl; \
        exit(1); \
    } \
}

// CUDA kernels are now defined in mnist_kernels.cuh

// MNIST Dataset class
class MNISTDataset {
public:
    MNISTDataset(const std::string& csv_file, const std::string& base_path);

    int getNumSamples() const { return static_cast<int>(image_paths.size()); }

    // Alias for getNumSamples for compatibility
    int size() const { return getNumSamples(); }

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
    float evaluate(MNISTDataset& dataset, float* val_loss = nullptr);

private:
    // Host memory for weights and biases
    std::vector<float> h_conv1_weights;
    std::vector<float> h_conv1_bias;
    std::vector<float> h_conv2_weights;
    std::vector<float> h_conv2_bias;
    std::vector<float> h_conv3_weights;
    std::vector<float> h_conv3_bias;
    std::vector<float> h_conv4_weights;
    std::vector<float> h_conv4_bias;
    std::vector<float> h_conv5_weights;
    std::vector<float> h_conv5_bias;
    std::vector<float> h_fc1_weights;
    std::vector<float> h_fc1_bias;
    std::vector<float> h_fc2_weights;
    std::vector<float> h_fc2_bias;
    std::vector<float> h_fc3_weights;
    std::vector<float> h_fc3_bias;
    std::vector<float> h_fc4_weights;
    std::vector<float> h_fc4_bias;
    std::vector<float> h_fc5_weights;
    std::vector<float> h_fc5_bias;

    // Device memory for weights and biases
    float* d_conv1_weights;
    float* d_conv1_bias;
    float* d_conv2_weights;
    float* d_conv2_bias;
    float* d_conv3_weights;
    float* d_conv3_bias;
    float* d_conv4_weights;
    float* d_conv4_bias;
    float* d_conv5_weights;
    float* d_conv5_bias;
    float* d_fc1_weights;
    float* d_fc1_bias;
    float* d_fc2_weights;
    float* d_fc2_bias;
    float* d_fc3_weights;
    float* d_fc3_bias;
    float* d_fc4_weights;
    float* d_fc4_bias;
    float* d_fc5_weights;
    float* d_fc5_bias;

    // Temporary buffers for forward and backward pass
    float* d_input;
    float* d_conv1_output;
    float* d_pool1_output;
    int* d_pool1_max_indices;
    float* d_conv2_output;
    float* d_pool2_output;
    int* d_pool2_max_indices;
    float* d_conv3_output;
    float* d_pool3_output;
    int* d_pool3_max_indices;
    float* d_conv4_output;
    float* d_pool4_output;
    int* d_pool4_max_indices;
    float* d_conv5_output;
    float* d_fc1_output;
    float* d_fc2_output;
    float* d_fc3_output;
    float* d_fc4_output;
    float* d_fc5_output;
    float* d_output;
    int* d_labels;
    float* d_loss;

    // Gradients
    float* d_grad_conv1_weights;
    float* d_grad_conv1_bias;
    float* d_grad_conv2_weights;
    float* d_grad_conv2_bias;
    float* d_grad_conv3_weights;
    float* d_grad_conv3_bias;
    float* d_grad_conv4_weights;
    float* d_grad_conv4_bias;
    float* d_grad_conv5_weights;
    float* d_grad_conv5_bias;
    float* d_grad_fc1_weights;
    float* d_grad_fc1_bias;
    float* d_grad_fc2_weights;
    float* d_grad_fc2_bias;
    float* d_grad_fc3_weights;
    float* d_grad_fc3_bias;
    float* d_grad_fc4_weights;
    float* d_grad_fc4_bias;
    float* d_grad_fc5_weights;
    float* d_grad_fc5_bias;
    float* d_grad_output;
    float* d_grad_fc5_output;
    float* d_grad_fc4_output;
    float* d_grad_fc3_output;
    float* d_grad_fc2_output;
    float* d_grad_fc1_output;
    float* d_grad_pool4_output;
    float* d_grad_conv5_output;
    float* d_grad_pool3_output;
    float* d_grad_conv4_output;
    float* d_grad_pool2_output;
    float* d_grad_conv3_output;
    float* d_grad_pool1_output;
    float* d_grad_conv2_output;
    float* d_grad_pool0_output;
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
    int conv3_output_height;
    int conv3_output_width;
    int pool3_output_height;
    int pool3_output_width;
    int conv4_output_height;
    int conv4_output_width;
    int pool4_output_height;
    int pool4_output_width;
    int conv5_output_height;
    int conv5_output_width;
    int flattened_size;
};
