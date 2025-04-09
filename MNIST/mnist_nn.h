#pragma once

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <random>
#include <iostream>
#include <fstream>
#include <cuda_runtime.h>

// Constants
constexpr int INPUT_SIZE = 784;  // 28x28 images
constexpr int HIDDEN_SIZE = 512;
constexpr int OUTPUT_SIZE = 10;  // 10 digits (0-9)
constexpr int BATCH_SIZE = 20000;
constexpr float LEARNING_RATE = 0.005f;
constexpr int NUM_EPOCHS = 20;

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
__global__ void forwardInputToHiddenKernel(float* d_input, float* d_hidden, float* d_weights_ih, float* d_bias_h, int batch_size);
__global__ void reluActivationKernel(float* d_hidden, int size);
__global__ void forwardHiddenToOutputKernel(float* d_hidden, float* d_output, float* d_weights_ho, float* d_bias_o, int batch_size);
__global__ void softmaxKernel(float* d_output, int batch_size);
__global__ void computeLossKernel(float* d_output, int* d_labels, float* d_loss, int batch_size);
__global__ void backpropOutputToHiddenKernel(float* d_output, int* d_labels, float* d_hidden, float* d_weights_ho, float* d_grad_weights_ho, float* d_grad_bias_o, int batch_size);
__global__ void backpropHiddenToInputKernel(float* d_grad_hidden, float* d_input, float* d_weights_ih, float* d_grad_weights_ih, float* d_grad_bias_h, int batch_size);
__global__ void updateParametersKernel(float* d_weights, float* d_grad_weights, float* d_bias, float* d_grad_bias, int rows, int cols, float learning_rate);

// MNIST Dataset class
class MNISTDataset {
public:
    MNISTDataset(const std::string& csv_file, const std::string& base_path);

    int getNumSamples() const { return static_cast<int>(image_paths.size()); }

    void getBatch(int batch_idx, std::vector<float>& batch_images, std::vector<int>& batch_labels);

private:
    std::vector<std::string> image_paths;
    std::vector<int> labels;
    std::string base_path;
};

// Neural Network class
class NeuralNetwork {
public:
    NeuralNetwork();
    ~NeuralNetwork();

    void train(MNISTDataset& train_dataset, MNISTDataset& val_dataset);
    float evaluate(MNISTDataset& dataset);

private:
    // Host memory
    std::vector<float> h_weights_ih;  // Input to hidden weights
    std::vector<float> h_bias_h;      // Hidden bias
    std::vector<float> h_weights_ho;  // Hidden to output weights
    std::vector<float> h_bias_o;      // Output bias

    // Device memory
    float* d_weights_ih;
    float* d_bias_h;
    float* d_weights_ho;
    float* d_bias_o;

    // Temporary buffers for forward and backward pass
    float* d_input;
    float* d_hidden;
    float* d_output;
    int* d_labels;
    float* d_loss;

    // Gradients
    float* d_grad_weights_ih;
    float* d_grad_bias_h;
    float* d_grad_weights_ho;
    float* d_grad_bias_o;
    float* d_grad_hidden;

    void initializeParameters();
    void allocateDeviceMemory();
    void freeDeviceMemory();

    void forwardPass(float* d_batch_input, int* d_batch_labels, int batch_size);
    float backwardPass(int batch_size);
    void updateParameters();
};
