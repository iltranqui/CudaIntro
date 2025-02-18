#ifndef NETWORK_H
#define NETWORK_H

#include <vector>

class NeuralNet {
public:
    NeuralNet();
    void forward(std::vector<float>& input, std::vector<float>& output);
    void backward(std::vector<float>& grad_output, float learning_rate);

private:
    std::vector<float> weights_;
    std::vector<float> biases_;
};

#endif // NETWORK_H
#pragma once
