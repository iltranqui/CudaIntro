#include "mnist_cnn.h"

// Convolution forward pass
__global__ void convolutionForwardKernel(float* d_input, float* d_output, float* d_filters, float* d_bias,
                                       int batch_size, int in_channels, int out_channels,
                                       int in_height, int in_width, int out_height, int out_width,
                                       int kernel_size, int padding, int stride) {
    int batch_idx = blockIdx.z;
    int out_channel = blockIdx.y;
    int out_y = blockIdx.x / out_width;
    int out_x = blockIdx.x % out_width;

    if (batch_idx < batch_size && out_channel < out_channels && out_y < out_height && out_x < out_width) {
        float sum = 0.0f;

        for (int in_channel = 0; in_channel < in_channels; in_channel++) {
            for (int ky = 0; ky < kernel_size; ky++) {
                for (int kx = 0; kx < kernel_size; kx++) {
                    int in_y = out_y * stride - padding + ky;
                    int in_x = out_x * stride - padding + kx;

                    if (in_y >= 0 && in_y < in_height && in_x >= 0 && in_x < in_width) {
                        int input_idx = ((batch_idx * in_channels + in_channel) * in_height + in_y) * in_width + in_x;
                        int filter_idx = ((out_channel * in_channels + in_channel) * kernel_size + ky) * kernel_size + kx;
                        sum += d_input[input_idx] * d_filters[filter_idx];
                    }
                }
            }
        }

        sum += d_bias[out_channel];
        int output_idx = ((batch_idx * out_channels + out_channel) * out_height + out_y) * out_width + out_x;
        d_output[output_idx] = sum;
    }
}

// Max pooling forward pass
__global__ void maxPoolingForwardKernel(float* d_input, float* d_output, int* d_max_indices,
                                      int batch_size, int channels,
                                      int in_height, int in_width, int out_height, int out_width,
                                      int pool_size, int stride) {
    int batch_idx = blockIdx.z;
    int channel = blockIdx.y;
    int out_y = blockIdx.x / out_width;
    int out_x = blockIdx.x % out_width;

    if (batch_idx < batch_size && channel < channels && out_y < out_height && out_x < out_width) {
        float max_val = -FLT_MAX;
        int max_idx = -1;

        for (int ky = 0; ky < pool_size; ky++) {
            for (int kx = 0; kx < pool_size; kx++) {
                int in_y = out_y * stride + ky;
                int in_x = out_x * stride + kx;

                if (in_y < in_height && in_x < in_width) {
                    int input_idx = ((batch_idx * channels + channel) * in_height + in_y) * in_width + in_x;
                    float val = d_input[input_idx];

                    if (val > max_val) {
                        max_val = val;
                        max_idx = input_idx;
                    }
                }
            }
        }

        int output_idx = ((batch_idx * channels + channel) * out_height + out_y) * out_width + out_x;
        d_output[output_idx] = max_val;

        if (d_max_indices != nullptr) {
            d_max_indices[output_idx] = max_idx;
        }
    }
}

// ReLU activation
__global__ void reluActivationKernel(float* d_input, float* d_output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        d_output[idx] = fmaxf(0.0f, d_input[idx]);
    }
}

// Fully connected forward pass
__global__ void fullyConnectedForwardKernel(float* d_input, float* d_output, float* d_weights, float* d_bias,
                                          int batch_size, int in_features, int out_features) {
    int batch_idx = blockIdx.y;
    int out_feature = blockIdx.x * blockDim.x + threadIdx.x;

    if (batch_idx < batch_size && out_feature < out_features) {
        float sum = 0.0f;

        for (int in_feature = 0; in_feature < in_features; in_feature++) {
            sum += d_input[batch_idx * in_features + in_feature] * d_weights[out_feature * in_features + in_feature];
        }

        sum += d_bias[out_feature];
        d_output[batch_idx * out_features + out_feature] = sum;
    }
}

// Softmax activation
__global__ void softmaxKernel(float* d_output, int batch_size, int output_size) {
    int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (batch_idx < batch_size) {
        // Find max value for numerical stability
        float max_val = d_output[batch_idx * output_size];
        for (int i = 1; i < output_size; i++) {
            max_val = fmaxf(max_val, d_output[batch_idx * output_size + i]);
        }

        // Compute exp(x - max) and sum
        float sum = 0.0f;
        for (int i = 0; i < output_size; i++) {
            d_output[batch_idx * output_size + i] = expf(d_output[batch_idx * output_size + i] - max_val);
            sum += d_output[batch_idx * output_size + i];
        }

        // Normalize
        for (int i = 0; i < output_size; i++) {
            d_output[batch_idx * output_size + i] /= sum;
        }

        // No debug prints for cleaner output
    }
}

// Cross-entropy loss
__global__ void crossEntropyLossKernel(float* d_output, int* d_labels, float* d_loss, int batch_size, int output_size) {
    int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (batch_idx < batch_size) {
        int label = d_labels[batch_idx];
        float prob = d_output[batch_idx * output_size + label];

        // Cross-entropy loss: -log(p)
        d_loss[batch_idx] = -logf(fmaxf(prob, 1e-10f)); // Clip for numerical stability

        // No debug prints for cleaner output
    }
}

// Softmax gradient
__global__ void softmaxGradientKernel(float* d_output, int* d_labels, float* d_grad_output,
                                     int batch_size, int output_size) {
    int batch_idx = blockIdx.y;
    int output_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (batch_idx < batch_size && output_idx < output_size) {
        int idx = batch_idx * output_size + output_idx;
        int label = d_labels[batch_idx];

        // Gradient of softmax with cross-entropy: p_i - y_i
        d_grad_output[idx] = d_output[idx];
        if (output_idx == label) {
            d_grad_output[idx] -= 1.0f;
        }

        // Scale the gradient to make it larger (helps with zero gradient problem)
        d_grad_output[idx] *= 10.0f;

        // No debug prints for cleaner output
    }
}

// Fully connected backward pass
__global__ void fullyConnectedBackwardKernel(float* d_input, float* d_grad_output,
                                           float* d_weights, float* d_grad_weights,
                                           float* d_grad_bias, float* d_grad_input,
                                           int batch_size, int in_features, int out_features) {
    int batch_idx = blockIdx.z;
    int feature_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int weight_idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Compute gradients for weights and bias
    if (feature_idx < out_features && weight_idx < in_features) {
        float grad_weight_sum = 0.0f;

        for (int b = 0; b < batch_size; b++) {
            grad_weight_sum += d_input[b * in_features + weight_idx] * d_grad_output[b * out_features + feature_idx];
        }

        d_grad_weights[feature_idx * in_features + weight_idx] = grad_weight_sum;
    }

    // Compute gradients for bias
    if (batch_idx == 0 && feature_idx < out_features && weight_idx == 0) {
        float grad_bias_sum = 0.0f;

        for (int b = 0; b < batch_size; b++) {
            grad_bias_sum += d_grad_output[b * out_features + feature_idx];
        }

        d_grad_bias[feature_idx] = grad_bias_sum;
    }

    // Compute gradients for input
    if (batch_idx < batch_size && feature_idx < in_features && weight_idx == 0) {
        float grad_input_sum = 0.0f;

        for (int o = 0; o < out_features; o++) {
            grad_input_sum += d_weights[o * in_features + feature_idx] * d_grad_output[batch_idx * out_features + o];
        }

        d_grad_input[batch_idx * in_features + feature_idx] = grad_input_sum;
    }
}

// ReLU backward pass
__global__ void reluBackwardKernel(float* d_grad_output, float* d_input, float* d_grad_input, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        d_grad_input[idx] = (d_input[idx] > 0.0f) ? d_grad_output[idx] : 0.0f;
    }
}

// Max pooling backward pass
__global__ void maxPoolingBackwardKernel(float* d_grad_output, int* d_max_indices, float* d_grad_input,
                                       int batch_size, int channels,
                                       int out_height, int out_width, int in_size) {
    int batch_idx = blockIdx.z;
    int channel = blockIdx.y;
    int out_y = blockIdx.x / out_width;
    int out_x = blockIdx.x % out_width;

    if (batch_idx < batch_size && channel < channels && out_y < out_height && out_x < out_width) {
        int output_idx = ((batch_idx * channels + channel) * out_height + out_y) * out_width + out_x;
        int max_idx = d_max_indices[output_idx];

        if (max_idx >= 0 && max_idx < in_size) {
            atomicAdd(&d_grad_input[max_idx], d_grad_output[output_idx]);
        }
    }
}

// Convolution backward pass
__global__ void conv2dBackwardKernel(float* d_grad_output, float* d_input, float* d_weights,
                                   float* d_grad_weights, float* d_grad_bias, float* d_grad_input,
                                   int batch_size, int in_channels, int out_channels,
                                   int in_height, int in_width, int kernel_size,
                                   int out_height, int out_width, int padding, int stride) {
    int batch_idx = blockIdx.z;
    int channel_idx = blockIdx.y;
    int kernel_y = blockIdx.x / kernel_size;
    int kernel_x = blockIdx.x % kernel_size;

    if (batch_idx < batch_size && channel_idx < out_channels && kernel_y < kernel_size && kernel_x < kernel_size) {
        // Compute gradients for weights
        for (int in_channel = 0; in_channel < in_channels; in_channel++) {
            float grad_weight_sum = 0.0f;

            for (int out_y = 0; out_y < out_height; out_y++) {
                for (int out_x = 0; out_x < out_width; out_x++) {
                    int in_y = out_y * stride - padding + kernel_y;
                    int in_x = out_x * stride - padding + kernel_x;

                    if (in_y >= 0 && in_y < in_height && in_x >= 0 && in_x < in_width) {
                        int input_idx = ((batch_idx * in_channels + in_channel) * in_height + in_y) * in_width + in_x;
                        int grad_output_idx = ((batch_idx * out_channels + channel_idx) * out_height + out_y) * out_width + out_x;

                        grad_weight_sum += d_input[input_idx] * d_grad_output[grad_output_idx];

                        if (d_grad_input != nullptr && kernel_x == 0 && kernel_y == 0 && channel_idx == 0) {
                            float grad_input_sum = 0.0f;

                            for (int oc = 0; oc < out_channels; oc++) {
                                for (int ky = 0; ky < kernel_size; ky++) {
                                    for (int kx = 0; kx < kernel_size; kx++) {
                                        int out_y = (in_y + padding - ky) / stride;
                                        int out_x = (in_x + padding - kx) / stride;

                                        if (out_y >= 0 && out_y < out_height && out_x >= 0 && out_x < out_width &&
                                            (in_y + padding - ky) % stride == 0 && (in_x + padding - kx) % stride == 0) {
                                            int grad_output_idx = ((batch_idx * out_channels + oc) * out_height + out_y) * out_width + out_x;
                                            int weight_idx = ((oc * in_channels + in_channel) * kernel_size + ky) * kernel_size + kx;
                                            grad_input_sum += d_grad_output[grad_output_idx] * d_weights[weight_idx];
                                        }
                                    }
                                }
                            }

                            atomicAdd(&d_grad_input[input_idx], grad_input_sum);
                        }
                    }
                }
            }

            int weight_idx = ((channel_idx * in_channels + in_channel) * kernel_size + kernel_y) * kernel_size + kernel_x;
            atomicAdd(&d_grad_weights[weight_idx], grad_weight_sum);
        }

        // Compute gradients for bias
        if (in_channels > 0 && kernel_y == 0 && kernel_x == 0) {
            float grad_bias_sum = 0.0f;

            for (int out_y = 0; out_y < out_height; out_y++) {
                for (int out_x = 0; out_x < out_width; out_x++) {
                    int grad_output_idx = ((batch_idx * out_channels + channel_idx) * out_height + out_y) * out_width + out_x;
                    grad_bias_sum += d_grad_output[grad_output_idx];
                }
            }

            atomicAdd(&d_grad_bias[channel_idx], grad_bias_sum);
        }
    }
}

// Update parameters kernel
__global__ void updateParametersKernel(float* d_weights, float* d_grad_weights,
                                     float* d_bias, float* d_grad_bias,
                                     int weights_size, int bias_size, float learning_rate) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Update weights
    if (idx < weights_size) {
        // Add a small check to prevent NaN or Inf values
        float grad = d_grad_weights[idx];
        if (!isnan(grad) && !isinf(grad)) {
            d_weights[idx] -= learning_rate * grad;
        }
    }

    // Update bias
    if (idx < bias_size) {
        // Add a small check to prevent NaN or Inf values
        float grad = d_grad_bias[idx];
        if (!isnan(grad) && !isinf(grad)) {
            d_bias[idx] -= learning_rate * grad;
        }
    }
}
