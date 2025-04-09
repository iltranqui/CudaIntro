#pragma once

// Convolution forward pass
__global__ void convolutionForwardKernel(float* d_input, float* d_output, float* d_filters, float* d_bias,
                                       int batch_size, int in_channels, int out_channels,
                                       int in_height, int in_width, int out_height, int out_width,
                                       int kernel_size, int padding, int stride);

// Max pooling forward pass
__global__ void maxPoolingForwardKernel(float* d_input, float* d_output, int* d_max_indices,
                                      int batch_size, int channels,
                                      int in_height, int in_width, int out_height, int out_width,
                                      int pool_size, int stride);

// ReLU activation
__global__ void reluActivationKernel(float* d_input, float* d_output, int size);

// Fully connected forward pass
__global__ void fullyConnectedForwardKernel(float* d_input, float* d_output, float* d_weights, float* d_bias,
                                          int batch_size, int in_features, int out_features);

// Softmax activation
__global__ void softmaxKernel(float* d_output, int batch_size, int output_size);

// Cross-entropy loss
__global__ void crossEntropyLossKernel(float* d_output, int* d_labels, float* d_loss, int batch_size, int output_size);

// Softmax gradient
__global__ void softmaxGradientKernel(float* d_output, int* d_labels, float* d_grad_output,
                                     int batch_size, int output_size);

// Fully connected backward pass
__global__ void fullyConnectedBackwardKernel(float* d_input, float* d_grad_output, 
                                           float* d_weights, float* d_grad_weights,
                                           float* d_grad_bias, float* d_grad_input,
                                           int batch_size, int in_features, int out_features);

// ReLU backward pass
__global__ void reluBackwardKernel(float* d_grad_output, float* d_input, float* d_grad_input, int size);

// Max pooling backward pass
__global__ void maxPoolingBackwardKernel(float* d_grad_output, int* d_max_indices, float* d_grad_input,
                                       int batch_size, int channels,
                                       int out_height, int out_width, int in_size);

// Convolution backward pass
__global__ void conv2dBackwardKernel(float* d_grad_output, float* d_input, float* d_weights,
                                   float* d_grad_weights, float* d_grad_bias, float* d_grad_input,
                                   int batch_size, int in_channels, int out_channels,
                                   int in_height, int in_width, int kernel_size,
                                   int out_height, int out_width, int padding, int stride);

// Update parameters kernel
__global__ void updateParametersKernel(float* d_weights, float* d_grad_weights, 
                                     float* d_bias, float* d_grad_bias,
                                     int weights_size, int bias_size, float learning_rate);
