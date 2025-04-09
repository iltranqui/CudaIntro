#include "mnist_cnn.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// CUDA kernel for 2D convolution
__global__ void conv2dKernel(float* input, float* output, float* weights, float* bias,
                           int batch_size, int in_channels, int out_channels,
                           int in_height, int in_width, int kernel_size,
                           int out_height, int out_width, int stride, int padding) {
    int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    int out_y = blockIdx.y * blockDim.y + threadIdx.y;
    int batch_idx = blockIdx.z / out_channels;
    int out_channel = blockIdx.z % out_channels;
    
    if (out_x < out_width && out_y < out_height && batch_idx < batch_size) {
        float sum = bias[out_channel];
        
        for (int in_channel = 0; in_channel < in_channels; in_channel++) {
            for (int ky = 0; ky < kernel_size; ky++) {
                for (int kx = 0; kx < kernel_size; kx++) {
                    int in_x = out_x * stride - padding + kx;
                    int in_y = out_y * stride - padding + ky;
                    
                    if (in_x >= 0 && in_x < in_width && in_y >= 0 && in_y < in_height) {
                        int input_idx = ((batch_idx * in_channels + in_channel) * in_height + in_y) * in_width + in_x;
                        int weight_idx = ((out_channel * in_channels + in_channel) * kernel_size + ky) * kernel_size + kx;
                        
                        sum += input[input_idx] * weights[weight_idx];
                    }
                }
            }
        }
        
        int output_idx = ((batch_idx * out_channels + out_channel) * out_height + out_y) * out_width + out_x;
        output[output_idx] = sum;
    }
}

// CUDA kernel for max pooling
__global__ void maxPoolKernel(float* input, float* output,
                            int batch_size, int channels,
                            int in_height, int in_width,
                            int pool_size, int stride) {
    int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    int out_y = blockIdx.y * blockDim.y + threadIdx.y;
    int batch_channel_idx = blockIdx.z;
    int batch_idx = batch_channel_idx / channels;
    int channel = batch_channel_idx % channels;
    
    int out_height = (in_height - pool_size) / stride + 1;
    int out_width = (in_width - pool_size) / stride + 1;
    
    if (out_x < out_width && out_y < out_height && batch_idx < batch_size) {
        float max_val = -FLT_MAX;
        int max_idx_x = -1;
        int max_idx_y = -1;
        
        for (int ky = 0; ky < pool_size; ky++) {
            for (int kx = 0; kx < pool_size; kx++) {
                int in_x = out_x * stride + kx;
                int in_y = out_y * stride + ky;
                
                if (in_x < in_width && in_y < in_height) {
                    int input_idx = ((batch_idx * channels + channel) * in_height + in_y) * in_width + in_x;
                    float val = input[input_idx];
                    
                    if (val > max_val) {
                        max_val = val;
                        max_idx_x = in_x;
                        max_idx_y = in_y;
                    }
                }
            }
        }
        
        int output_idx = ((batch_idx * channels + channel) * out_height + out_y) * out_width + out_x;
        output[output_idx] = max_val;
    }
}

// CUDA kernel for ReLU activation
__global__ void reluActivationKernel(float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        output[idx] = fmaxf(0.0f, input[idx]);
    }
}

// CUDA kernel for softmax activation
__global__ void softmaxKernel(float* input, int batch_size, int num_classes) {
    int batch_idx = blockIdx.x;
    
    if (batch_idx < batch_size) {
        float max_val = -FLT_MAX;
        
        // Find max value for numerical stability
        for (int i = 0; i < num_classes; i++) {
            int idx = batch_idx * num_classes + i;
            max_val = fmaxf(max_val, input[idx]);
        }
        
        float sum = 0.0f;
        
        // Compute exponentials and sum
        for (int i = 0; i < num_classes; i++) {
            int idx = batch_idx * num_classes + i;
            input[idx] = expf(input[idx] - max_val);
            sum += input[idx];
        }
        
        // Normalize
        for (int i = 0; i < num_classes; i++) {
            int idx = batch_idx * num_classes + i;
            input[idx] /= sum;
        }
    }
}

// CUDA kernel for computing cross-entropy loss
__global__ void computeLossKernel(float* predictions, int* labels, float* loss, int batch_size, int num_classes) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < batch_size) {
        int label = labels[idx];
        float pred = predictions[idx * num_classes + label];
        
        // Clip prediction to avoid log(0)
        pred = fmaxf(pred, 1e-15f);
        
        loss[idx] = -logf(pred);
    }
}

// CUDA kernel for fully connected forward pass
__global__ void fullyConnectedForwardKernel(float* input, float* output, float* weights, float* bias,
                                          int batch_size, int in_features, int out_features) {
    int out_feature = blockIdx.x * blockDim.x + threadIdx.x;
    int batch_idx = blockIdx.y;
    
    if (out_feature < out_features && batch_idx < batch_size) {
        float sum = bias[out_feature];
        
        for (int in_feature = 0; in_feature < in_features; in_feature++) {
            sum += input[batch_idx * in_features + in_feature] * weights[out_feature * in_features + in_feature];
        }
        
        output[batch_idx * out_features + out_feature] = sum;
    }
}

// CUDA kernel for softmax gradient
__global__ void softmaxGradientKernel(float* output, int* labels, float* grad_output, int batch_size, int num_classes) {
    int out_feature = threadIdx.x;
    int batch_idx = blockIdx.y;
    
    if (out_feature < num_classes && batch_idx < batch_size) {
        int idx = batch_idx * num_classes + out_feature;
        int label = labels[batch_idx];
        
        if (out_feature == label) {
            grad_output[idx] = output[idx] - 1.0f;
        } else {
            grad_output[idx] = output[idx];
        }
    }
}

// CUDA kernel for fully connected backward pass
__global__ void fullyConnectedBackwardKernel(float* grad_output, float* input, float* weights,
                                           float* grad_weights, float* grad_bias, float* grad_input,
                                           int batch_size, int in_features, int out_features) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < in_features * out_features) {
        int out_feature = idx / in_features;
        int in_feature = idx % in_features;
        
        float grad_weight_sum = 0.0f;
        
        for (int batch_idx = 0; batch_idx < batch_size; batch_idx++) {
            grad_weight_sum += grad_output[batch_idx * out_features + out_feature] * input[batch_idx * in_features + in_feature];
            
            if (grad_input != nullptr && in_feature == 0) {
                float grad_input_sum = 0.0f;
                
                for (int o = 0; o < out_features; o++) {
                    grad_input_sum += grad_output[batch_idx * out_features + o] * weights[o * in_features + in_feature];
                }
                
                grad_input[batch_idx * in_features + in_feature] = grad_input_sum;
            }
        }
        
        grad_weights[out_feature * in_features + in_feature] = grad_weight_sum;
        
        if (in_feature == 0) {
            float grad_bias_sum = 0.0f;
            
            for (int batch_idx = 0; batch_idx < batch_size; batch_idx++) {
                grad_bias_sum += grad_output[batch_idx * out_features + out_feature];
            }
            
            grad_bias[out_feature] = grad_bias_sum;
        }
    }
}

// CUDA kernel for ReLU backward pass
__global__ void reluBackwardKernel(float* grad_output, float* input, float* grad_input, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        grad_input[idx] = (input[idx] > 0.0f) ? grad_output[idx] : 0.0f;
    }
}

// CUDA kernel for max pooling backward pass
__global__ void maxPoolBackwardKernel(float* grad_output, float* input, float* output, float* grad_input,
                                    int batch_size, int channels,
                                    int in_height, int in_width,
                                    int pool_size, int stride) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int out_height = (in_height - pool_size) / stride + 1;
    int out_width = (in_width - pool_size) / stride + 1;
    int total_elements = batch_size * channels * out_height * out_width;
    
    if (idx < total_elements) {
        int out_x = idx % out_width;
        int out_y = (idx / out_width) % out_height;
        int channel = (idx / (out_width * out_height)) % channels;
        int batch_idx = idx / (out_width * out_height * channels);
        
        float max_val = -FLT_MAX;
        int max_idx_x = -1;
        int max_idx_y = -1;
        
        // Find the position of the maximum value in the input patch
        for (int ky = 0; ky < pool_size; ky++) {
            for (int kx = 0; kx < pool_size; kx++) {
                int in_x = out_x * stride + kx;
                int in_y = out_y * stride + ky;
                
                if (in_x < in_width && in_y < in_height) {
                    int input_idx = ((batch_idx * channels + channel) * in_height + in_y) * in_width + in_x;
                    float val = input[input_idx];
                    
                    if (val > max_val) {
                        max_val = val;
                        max_idx_x = in_x;
                        max_idx_y = in_y;
                    }
                }
            }
        }
        
        // Propagate gradient only to the position of the maximum value
        if (max_idx_x >= 0 && max_idx_y >= 0) {
            int grad_input_idx = ((batch_idx * channels + channel) * in_height + max_idx_y) * in_width + max_idx_x;
            int grad_output_idx = ((batch_idx * channels + channel) * out_height + out_y) * out_width + out_x;
            
            atomicAdd(&grad_input[grad_input_idx], grad_output[grad_output_idx]);
        }
    }
}

// CUDA kernel for convolution backward pass
__global__ void conv2dBackwardKernel(float* grad_output, float* input, float* weights,
                                   float* grad_weights, float* grad_bias, float* grad_input,
                                   int batch_size, int in_channels, int out_channels,
                                   int in_height, int in_width, int kernel_size,
                                   int out_height, int out_width, int stride, int padding) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_weights = out_channels * in_channels * kernel_size * kernel_size;
    
    if (idx < total_weights) {
        int kx = idx % kernel_size;
        int ky = (idx / kernel_size) % kernel_size;
        int in_channel = (idx / (kernel_size * kernel_size)) % in_channels;
        int out_channel = idx / (kernel_size * kernel_size * in_channels);
        
        float grad_weight_sum = 0.0f;
        
        for (int batch_idx = 0; batch_idx < batch_size; batch_idx++) {
            for (int out_y = 0; out_y < out_height; out_y++) {
                for (int out_x = 0; out_x < out_width; out_x++) {
                    int in_x = out_x * stride - padding + kx;
                    int in_y = out_y * stride - padding + ky;
                    
                    if (in_x >= 0 && in_x < in_width && in_y >= 0 && in_y < in_height) {
                        int input_idx = ((batch_idx * in_channels + in_channel) * in_height + in_y) * in_width + in_x;
                        int grad_output_idx = ((batch_idx * out_channels + out_channel) * out_height + out_y) * out_width + out_x;
                        
                        grad_weight_sum += input[input_idx] * grad_output[grad_output_idx];
                        
                        if (grad_input != nullptr && kx == 0 && ky == 0 && out_channel == 0) {
                            float grad_input_sum = 0.0f;
                            
                            for (int oc = 0; oc < out_channels; oc++) {
                                for (int ky_i = 0; ky_i < kernel_size; ky_i++) {
                                    for (int kx_i = 0; kx_i < kernel_size; kx_i++) {
                                        int out_x_i = (in_x + padding - kx_i) / stride;
                                        int out_y_i = (in_y + padding - ky_i) / stride;
                                        
                                        if (out_x_i >= 0 && out_x_i < out_width && out_y_i >= 0 && out_y_i < out_height &&
                                            (in_x + padding - kx_i) % stride == 0 && (in_y + padding - ky_i) % stride == 0) {
                                            int weight_idx = ((oc * in_channels + in_channel) * kernel_size + ky_i) * kernel_size + kx_i;
                                            int grad_output_idx_i = ((batch_idx * out_channels + oc) * out_height + out_y_i) * out_width + out_x_i;
                                            
                                            grad_input_sum += weights[weight_idx] * grad_output[grad_output_idx_i];
                                        }
                                    }
                                }
                            }
                            
                            grad_input[input_idx] = grad_input_sum;
                        }
                    }
                }
            }
        }
        
        grad_weights[idx] = grad_weight_sum;
        
        // Compute gradient for bias
        if (kx == 0 && ky == 0 && in_channel == 0) {
            float grad_bias_sum = 0.0f;
            
            for (int batch_idx = 0; batch_idx < batch_size; batch_idx++) {
                for (int out_y = 0; out_y < out_height; out_y++) {
                    for (int out_x = 0; out_x < out_width; out_x++) {
                        int grad_output_idx = ((batch_idx * out_channels + out_channel) * out_height + out_y) * out_width + out_x;
                        grad_bias_sum += grad_output[grad_output_idx];
                    }
                }
            }
            
            grad_bias[out_channel] = grad_bias_sum;
        }
    }
}

// CUDA kernel for parameter update
__global__ void updateParametersKernel(float* weights, float* grad_weights, float* bias, float* grad_bias,
                                     int weights_size, int bias_size, float learning_rate) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < weights_size) {
        weights[idx] -= learning_rate * grad_weights[idx];
    }
    
    if (idx < bias_size) {
        bias[idx] -= learning_rate * grad_bias[idx];
    }
}
