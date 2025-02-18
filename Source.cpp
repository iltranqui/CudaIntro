#include "network.h"
#include "cuda_kernels.cuh"
#include <iostream>

NeuralNet::NeuralNet() {
    weights_.resize(784 * 128); // Example layer (784 input, 128 neurons)
    biases_.resize(128, 0.1f);  // Biases
}

void NeuralNet::forward(std::vector<float>& input, std::vector<float>& output) {
    forward_cuda(input.data(), weights_.data(), biases_.data(), output.data(), 784, 128);
}

void NeuralNet::backward(std::vector<float>& grad_output, float learning_rate) {
    backward_cuda(grad_output.data(), weights_.data(), biases_.data(), 784, 128, learning_rate);
}
