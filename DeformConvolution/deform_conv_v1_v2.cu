#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>

/**
 * @file deform_conv_v1_v2.cu
 * @brief Example implementation of deformable convolution v1 and v2
 * 
 * This file contains simplified implementations of standard 2D convolution,
 * deformable convolution v1, and deformable convolution v2 to illustrate
 * the key differences between them.
 */

// Bilinear interpolation function for sampling at non-integer locations
__device__ float bilinear_interpolate(float* input, int height, int width, float y, float x) {
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
    bool valid_x1 = (x1 >= 0 && x1 < width);
    bool valid_x2 = (x2 >= 0 && x2 < width);
    bool valid_y1 = (y1 >= 0 && y1 < height);
    bool valid_y2 = (y2 >= 0 && y2 < height);
    
    // Get pixel values (with zero padding for out-of-bounds)
    float v11 = (valid_y1 && valid_x1) ? input[y1 * width + x1] : 0.0f;
    float v12 = (valid_y1 && valid_x2) ? input[y1 * width + x2] : 0.0f;
    float v21 = (valid_y2 && valid_x1) ? input[y2 * width + x1] : 0.0f;
    float v22 = (valid_y2 && valid_x2) ? input[y2 * width + x2] : 0.0f;
    
    // Bilinear interpolation
    float value = wy1 * wx1 * v11 + wy1 * wx2 * v12 + wy2 * wx1 * v21 + wy2 * wx2 * v22;
    
    return value;
}

/**
 * Standard 2D Convolution
 * 
 * In standard convolution, the sampling grid is fixed.
 * For each output position, we sample from a regular grid in the input.
 */
__global__ void standard_conv2d(
    float* input,    // Input feature map
    float* weight,   // Convolution weights
    float* output,   // Output feature map
    int in_h, int in_w,
    int out_h, int out_w,
    int ksize, int stride, int padding
) {
    // Compute output pixel coordinates
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Ensure the thread is within output dimensions
    if (x < out_w && y < out_h) {
        float sum = 0.0f; // Accumulate convolution result
        
        // Compute the top-left corner of the receptive field
        int in_y = y * stride - padding;
        int in_x = x * stride - padding;
        
        // Iterate over the kernel window
        for (int ky = 0; ky < ksize; ky++) {
            for (int kx = 0; kx < ksize; kx++) {
                // Compute input position
                int iy = in_y + ky;
                int ix = in_x + kx;
                
                // Check if the input position is valid
                if (iy >= 0 && iy < in_h && ix >= 0 && ix < in_w) {
                    // Get input value and weight
                    float in_val = input[iy * in_w + ix];
                    float w_val = weight[ky * ksize + kx];
                    
                    // Accumulate weighted value
                    sum += in_val * w_val;
                }
            }
        }
        
        // Store the result
        output[y * out_w + x] = sum;
    }
}

/**
 * Deformable Convolution v1
 * 
 * In deformable convolution v1, we add 2D offsets to the sampling locations.
 * This allows the network to sample from variable positions.
 */
__global__ void deform_conv2d_v1(
    float* input,    // Input feature map
    float* offset,   // Learned offsets (2 * ksize * ksize values per output position)
    float* weight,   // Convolution weights
    float* output,   // Output feature map
    int in_h, int in_w,
    int out_h, int out_w,
    int ksize, int stride, int padding
) {
    // Compute output pixel coordinates
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Ensure the thread is within output dimensions
    if (x < out_w && y < out_h) {
        float sum = 0.0f; // Accumulate convolution result
        
        // Compute the top-left corner of the receptive field
        int in_y = y * stride - padding;
        int in_x = x * stride - padding;
        
        // Iterate over the kernel window
        for (int ky = 0; ky < ksize; ky++) {
            for (int kx = 0; kx < ksize; kx++) {
                // Compute the index for the offset values
                int offset_idx = ((y * out_w + x) * ksize * ksize + ky * ksize + kx) * 2;
                
                // Get the offsets for this kernel position
                float offset_y = offset[offset_idx];
                float offset_x = offset[offset_idx + 1];
                
                // Compute the sampling position with offsets
                float sample_y = in_y + ky + offset_y;
                float sample_x = in_x + kx + offset_x;
                
                // Use bilinear interpolation to sample from the input
                float in_val = bilinear_interpolate(input, in_h, in_w, sample_y, sample_x);
                
                // Get the weight for this kernel position
                float w_val = weight[ky * ksize + kx];
                
                // Accumulate weighted value
                sum += in_val * w_val;
            }
        }
        
        // Store the result
        output[y * out_w + x] = sum;
    }
}

/**
 * Deformable Convolution v2
 * 
 * In deformable convolution v2, we add both 2D offsets and modulation scalars.
 * The modulation scalars control the contribution of each sampling point.
 */
__global__ void deform_conv2d_v2(
    float* input,     // Input feature map
    float* offset,    // Learned offsets (2 * ksize * ksize values per output position)
    float* modulation, // Learned modulation scalars (ksize * ksize values per output position)
    float* weight,    // Convolution weights
    float* output,    // Output feature map
    int in_h, int in_w,
    int out_h, int out_w,
    int ksize, int stride, int padding
) {
    // Compute output pixel coordinates
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Ensure the thread is within output dimensions
    if (x < out_w && y < out_h) {
        float sum = 0.0f; // Accumulate convolution result
        
        // Compute the top-left corner of the receptive field
        int in_y = y * stride - padding;
        int in_x = x * stride - padding;
        
        // Iterate over the kernel window
        for (int ky = 0; ky < ksize; ky++) {
            for (int kx = 0; kx < ksize; kx++) {
                // Compute the index for the offset values
                int offset_idx = ((y * out_w + x) * ksize * ksize + ky * ksize + kx) * 2;
                
                // Get the offsets for this kernel position
                float offset_y = offset[offset_idx];
                float offset_x = offset[offset_idx + 1];
                
                // Compute the sampling position with offsets
                float sample_y = in_y + ky + offset_y;
                float sample_x = in_x + kx + offset_x;
                
                // Use bilinear interpolation to sample from the input
                float in_val = bilinear_interpolate(input, in_h, in_w, sample_y, sample_x);
                
                // Get the weight for this kernel position
                float w_val = weight[ky * ksize + kx];
                
                // Get the modulation scalar for this kernel position
                int mod_idx = (y * out_w + x) * ksize * ksize + ky * ksize + kx;
                float mod_val = modulation[mod_idx];
                
                // Apply modulation and accumulate weighted value
                sum += in_val * w_val * mod_val;
            }
        }
        
        // Store the result
        output[y * out_w + x] = sum;
    }
}

/**
 * Backward pass for deformable convolution v1
 * 
 * Computes gradients for input, weights, and offsets.
 */
__global__ void deform_conv2d_v1_backward(
    float* grad_output,  // Gradient of the loss with respect to the output
    float* input,        // Input feature map
    float* offset,       // Learned offsets
    float* weight,       // Convolution weights
    float* grad_input,   // Gradient of the loss with respect to the input
    float* grad_offset,  // Gradient of the loss with respect to the offsets
    float* grad_weight,  // Gradient of the loss with respect to the weights
    int in_h, int in_w,
    int out_h, int out_w,
    int ksize, int stride, int padding
) {
    // Compute output pixel coordinates
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Ensure the thread is within output dimensions
    if (x < out_w && y < out_h) {
        float grad = grad_output[y * out_w + x];
        
        // Compute the top-left corner of the receptive field
        int in_y = y * stride - padding;
        int in_x = x * stride - padding;
        
        // Iterate over the kernel window
        for (int ky = 0; ky < ksize; ky++) {
            for (int kx = 0; kx < ksize; kx++) {
                // Compute the index for the offset values
                int offset_idx = ((y * out_w + x) * ksize * ksize + ky * ksize + kx) * 2;
                
                // Get the offsets for this kernel position
                float offset_y = offset[offset_idx];
                float offset_x = offset[offset_idx + 1];
                
                // Compute the sampling position with offsets
                float sample_y = in_y + ky + offset_y;
                float sample_x = in_x + kx + offset_x;
                
                // Get the weight for this kernel position
                float w_val = weight[ky * ksize + kx];
                
                // Compute gradients for input and weight
                // (Simplified - in practice, would need to compute gradients through bilinear interpolation)
                
                // For input gradient, we need to distribute the gradient to the four surrounding pixels
                // based on the bilinear interpolation weights
                int x1 = floor(sample_x);
                int x2 = x1 + 1;
                int y1 = floor(sample_y);
                int y2 = y1 + 1;
                
                float wx1 = x2 - sample_x;
                float wx2 = sample_x - x1;
                float wy1 = y2 - sample_y;
                float wy2 = sample_y - y1;
                
                // Update gradients for input (using atomic operations to handle race conditions)
                if (y1 >= 0 && y1 < in_h && x1 >= 0 && x1 < in_w)
                    atomicAdd(&grad_input[y1 * in_w + x1], grad * w_val * wy1 * wx1);
                if (y1 >= 0 && y1 < in_h && x2 >= 0 && x2 < in_w)
                    atomicAdd(&grad_input[y1 * in_w + x2], grad * w_val * wy1 * wx2);
                if (y2 >= 0 && y2 < in_h && x1 >= 0 && x1 < in_w)
                    atomicAdd(&grad_input[y2 * in_w + x1], grad * w_val * wy2 * wx1);
                if (y2 >= 0 && y2 < in_h && x2 >= 0 && x2 < in_w)
                    atomicAdd(&grad_input[y2 * in_w + x2], grad * w_val * wy2 * wx2);
                
                // Update gradients for weight
                float in_val = bilinear_interpolate(input, in_h, in_w, sample_y, sample_x);
                atomicAdd(&grad_weight[ky * ksize + kx], grad * in_val);
                
                // Update gradients for offsets
                // (Simplified - in practice, would need to compute gradients of bilinear interpolation w.r.t. offsets)
                // This is a complex calculation involving partial derivatives of the bilinear interpolation
                
                // Placeholder for offset gradients
                atomicAdd(&grad_offset[offset_idx], 0.0f);     // Gradient for y offset
                atomicAdd(&grad_offset[offset_idx + 1], 0.0f); // Gradient for x offset
            }
        }
    }
}

/**
 * Backward pass for deformable convolution v2
 * 
 * Computes gradients for input, weights, offsets, and modulation scalars.
 */
__global__ void deform_conv2d_v2_backward(
    float* grad_output,   // Gradient of the loss with respect to the output
    float* input,         // Input feature map
    float* offset,        // Learned offsets
    float* modulation,    // Learned modulation scalars
    float* weight,        // Convolution weights
    float* grad_input,    // Gradient of the loss with respect to the input
    float* grad_offset,   // Gradient of the loss with respect to the offsets
    float* grad_modulation, // Gradient of the loss with respect to the modulation scalars
    float* grad_weight,   // Gradient of the loss with respect to the weights
    int in_h, int in_w,
    int out_h, int out_w,
    int ksize, int stride, int padding
) {
    // Compute output pixel coordinates
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Ensure the thread is within output dimensions
    if (x < out_w && y < out_h) {
        float grad = grad_output[y * out_w + x];
        
        // Compute the top-left corner of the receptive field
        int in_y = y * stride - padding;
        int in_x = x * stride - padding;
        
        // Iterate over the kernel window
        for (int ky = 0; ky < ksize; ky++) {
            for (int kx = 0; kx < ksize; kx++) {
                // Compute the index for the offset values
                int offset_idx = ((y * out_w + x) * ksize * ksize + ky * ksize + kx) * 2;
                
                // Get the offsets for this kernel position
                float offset_y = offset[offset_idx];
                float offset_x = offset[offset_idx + 1];
                
                // Compute the sampling position with offsets
                float sample_y = in_y + ky + offset_y;
                float sample_x = in_x + kx + offset_x;
                
                // Get the weight for this kernel position
                float w_val = weight[ky * ksize + kx];
                
                // Get the modulation scalar for this kernel position
                int mod_idx = (y * out_w + x) * ksize * ksize + ky * ksize + kx;
                float mod_val = modulation[mod_idx];
                
                // Compute gradients for input, weight, and modulation
                // (Simplified - in practice, would need to compute gradients through bilinear interpolation)
                
                // For input gradient, we need to distribute the gradient to the four surrounding pixels
                // based on the bilinear interpolation weights
                int x1 = floor(sample_x);
                int x2 = x1 + 1;
                int y1 = floor(sample_y);
                int y2 = y1 + 1;
                
                float wx1 = x2 - sample_x;
                float wx2 = sample_x - x1;
                float wy1 = y2 - sample_y;
                float wy2 = sample_y - y1;
                
                // Update gradients for input (using atomic operations to handle race conditions)
                if (y1 >= 0 && y1 < in_h && x1 >= 0 && x1 < in_w)
                    atomicAdd(&grad_input[y1 * in_w + x1], grad * w_val * mod_val * wy1 * wx1);
                if (y1 >= 0 && y1 < in_h && x2 >= 0 && x2 < in_w)
                    atomicAdd(&grad_input[y1 * in_w + x2], grad * w_val * mod_val * wy1 * wx2);
                if (y2 >= 0 && y2 < in_h && x1 >= 0 && x1 < in_w)
                    atomicAdd(&grad_input[y2 * in_w + x1], grad * w_val * mod_val * wy2 * wx1);
                if (y2 >= 0 && y2 < in_h && x2 >= 0 && x2 < in_w)
                    atomicAdd(&grad_input[y2 * in_w + x2], grad * w_val * mod_val * wy2 * wx2);
                
                // Update gradients for weight
                float in_val = bilinear_interpolate(input, in_h, in_w, sample_y, sample_x);
                atomicAdd(&grad_weight[ky * ksize + kx], grad * in_val * mod_val);
                
                // Update gradients for modulation
                atomicAdd(&grad_modulation[mod_idx], grad * in_val * w_val);
                
                // Update gradients for offsets
                // (Simplified - in practice, would need to compute gradients of bilinear interpolation w.r.t. offsets)
                // This is a complex calculation involving partial derivatives of the bilinear interpolation
                
                // Placeholder for offset gradients
                atomicAdd(&grad_offset[offset_idx], 0.0f);     // Gradient for y offset
                atomicAdd(&grad_offset[offset_idx + 1], 0.0f); // Gradient for x offset
            }
        }
    }
}

/**
 * Example of how to use the deformable convolution kernels
 */
void example_usage() {
    // Input dimensions
    int in_h = 32;
    int in_w = 32;
    
    // Kernel parameters
    int ksize = 3;
    int stride = 1;
    int padding = 1;
    
    // Output dimensions
    int out_h = (in_h + 2 * padding - ksize) / stride + 1;
    int out_w = (in_w + 2 * padding - ksize) / stride + 1;
    
    // Allocate memory for input, weights, offsets, modulation, and output
    float *d_input, *d_weight, *d_offset, *d_modulation, *d_output;
    cudaMalloc(&d_input, in_h * in_w * sizeof(float));
    cudaMalloc(&d_weight, ksize * ksize * sizeof(float));
    cudaMalloc(&d_offset, out_h * out_w * ksize * ksize * 2 * sizeof(float));
    cudaMalloc(&d_modulation, out_h * out_w * ksize * ksize * sizeof(float));
    cudaMalloc(&d_output, out_h * out_w * sizeof(float));
    
    // Define grid and block dimensions
    dim3 block(16, 16);
    dim3 grid((out_w + block.x - 1) / block.x, (out_h + block.y - 1) / block.y);
    
    // Launch standard convolution kernel
    standard_conv2d<<<grid, block>>>(
        d_input, d_weight, d_output,
        in_h, in_w, out_h, out_w,
        ksize, stride, padding
    );
    
    // Launch deformable convolution v1 kernel
    deform_conv2d_v1<<<grid, block>>>(
        d_input, d_offset, d_weight, d_output,
        in_h, in_w, out_h, out_w,
        ksize, stride, padding
    );
    
    // Launch deformable convolution v2 kernel
    deform_conv2d_v2<<<grid, block>>>(
        d_input, d_offset, d_modulation, d_weight, d_output,
        in_h, in_w, out_h, out_w,
        ksize, stride, padding
    );
    
    // Allocate memory for gradients
    float *d_grad_output, *d_grad_input, *d_grad_weight, *d_grad_offset, *d_grad_modulation;
    cudaMalloc(&d_grad_output, out_h * out_w * sizeof(float));
    cudaMalloc(&d_grad_input, in_h * in_w * sizeof(float));
    cudaMalloc(&d_grad_weight, ksize * ksize * sizeof(float));
    cudaMalloc(&d_grad_offset, out_h * out_w * ksize * ksize * 2 * sizeof(float));
    cudaMalloc(&d_grad_modulation, out_h * out_w * ksize * ksize * sizeof(float));
    
    // Launch backward pass for deformable convolution v1
    deform_conv2d_v1_backward<<<grid, block>>>(
        d_grad_output, d_input, d_offset, d_weight,
        d_grad_input, d_grad_offset, d_grad_weight,
        in_h, in_w, out_h, out_w,
        ksize, stride, padding
    );
    
    // Launch backward pass for deformable convolution v2
    deform_conv2d_v2_backward<<<grid, block>>>(
        d_grad_output, d_input, d_offset, d_modulation, d_weight,
        d_grad_input, d_grad_offset, d_grad_modulation, d_grad_weight,
        in_h, in_w, out_h, out_w,
        ksize, stride, padding
    );
    
    // Free memory
    cudaFree(d_input);
    cudaFree(d_weight);
    cudaFree(d_offset);
    cudaFree(d_modulation);
    cudaFree(d_output);
    cudaFree(d_grad_output);
    cudaFree(d_grad_input);
    cudaFree(d_grad_weight);
    cudaFree(d_grad_offset);
    cudaFree(d_grad_modulation);
}

/**
 * Main function (not used in this example)
 */
int main() {
    // This is just an example implementation
    // The actual usage would be in a larger system
    printf("This is an example implementation of deformable convolution v1 and v2.\n");
    return 0;
}
