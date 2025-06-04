#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>
#include <curand_kernel.h>

/**
 * @file deform_conv_explanation.cu
 * @brief Example implementation of deformable convolution with offset prediction
 *
 * This file contains simplified implementations of standard 2D convolution,
 * offset-generating convolution, and deformable convolution to illustrate
 * a two-stage approach where one convolution predicts offsets for another.
 */

/**
 * Optimized bilinear interpolation function for GPU execution
 *
 * This implementation is optimized for GPU execution by:
 * 1. Using direct texture-like sampling approach
 * 2. Minimizing branching with math operations instead of conditionals
 * 3. Using fast math operations where possible
 * 4. Coalescing memory accesses for better performance
 */
__device__ __forceinline__ float bilinear_interpolate(float* input, int height, int width, float y, float x) {
    // Clamp coordinates to valid range for addressing
    // This avoids the need for boundary checks later
    float x_clamped = fmaxf(0.0f, fminf(width - 1.001f, x));
    float y_clamped = fmaxf(0.0f, fminf(height - 1.001f, y));

    // Get the four nearest integer coordinates
    int x1 = __float2int_rd(x_clamped);  // floor
    int y1 = __float2int_rd(y_clamped);  // floor
    int x2 = x1 + 1;
    int y2 = y1 + 1;

    // Calculate the interpolation weights
    float wx2 = x_clamped - x1; // Weight for x2
    float wx1 = 1.0f - wx2;     // Weight for x1
    float wy2 = y_clamped - y1; // Weight for y2
    float wy1 = 1.0f - wy2;     // Weight for y1

    // Pre-compute indices for coalesced memory access
    int idx11 = y1 * width + x1;
    int idx12 = y1 * width + x2;
    int idx21 = y2 * width + x1;
    int idx22 = y2 * width + x2;

    // Get pixel values
    // Note: We've already clamped coordinates, so we know these are valid
    float v11 = input[idx11];
    float v12 = (x2 < width) ? input[idx12] : 0.0f;
    float v21 = (y2 < height) ? input[idx21] : 0.0f;
    float v22 = (x2 < width && y2 < height) ? input[idx22] : 0.0f;

    // Bilinear interpolation using fused multiply-add for better performance
    float value = wy1 * wx1 * v11 +
                 wy1 * wx2 * v12 +
                 wy2 * wx1 * v21 +
                 wy2 * wx2 * v22;

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
 * Offset-Generating Convolution with Shared Memory Optimization
 *
 * This convolution generates the offsets that will be used by the deformable convolution.
 * It outputs 2 * ksize * ksize values per spatial location, representing the x and y offsets.
 * Uses shared memory to cache input values for faster access.
 */
__global__ void offset_generating_conv2d(
    float* input,    // Input feature map
    float* weight,   // Convolution weights for generating offsets
    float* offset,   // Output offsets (2 * ksize * ksize values per output position)
    int in_h, int in_w,
    int out_h, int out_w,
    int ksize, int stride, int padding
) {
    // Compute output pixel coordinates
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Define shared memory for input tile
    // Add padding on each side to account for the kernel window
    extern __shared__ float shared_input[];

    // Calculate the dimensions of the input tile we need to load
    int tile_w = blockDim.x + (ksize - 1);
    int tile_h = blockDim.y + (ksize - 1);

    // Calculate the top-left corner of the input tile in global memory
    int tile_start_x = blockIdx.x * blockDim.x - padding;
    int tile_start_y = blockIdx.y * blockDim.y - padding;

    // Load input tile into shared memory (collaborative loading)
    for (int i = threadIdx.y; i < tile_h; i += blockDim.y) {
        for (int j = threadIdx.x; j < tile_w; j += blockDim.x) {
            int global_y = tile_start_y + i;
            int global_x = tile_start_x + j;

            // Check bounds and load from global memory or use zero padding
            if (global_y >= 0 && global_y < in_h && global_x >= 0 && global_x < in_w) {
                shared_input[i * tile_w + j] = input[global_y * in_w + global_x];
            } else {
                shared_input[i * tile_w + j] = 0.0f;
            }
        }
    }

    // Ensure all threads have loaded their part of the input
    __syncthreads();

    // Ensure the thread is within output dimensions
    if (x < out_w && y < out_h) {
        // For each kernel position, we need to generate 2 offset values (x and y)
        // Iterate over the kernel window
        for (int ky = 0; ky < ksize; ky++) {
            for (int kx = 0; kx < ksize; kx++) {
                // Compute the base index for this kernel position's offsets
                int offset_base_idx = ((y * out_w + x) * ksize * ksize + ky * ksize + kx) * 2;

                // Generate y-offset and x-offset
                float offset_y_sum = 0.0f;
                float offset_x_sum = 0.0f;

                // Compute the top-left corner of the receptive field in shared memory
                int sm_y = threadIdx.y * stride;
                int sm_x = threadIdx.x * stride;

                // Perform convolution to generate the offset values
                for (int fy = 0; fy < ksize; fy++) {
                    for (int fx = 0; fx < ksize; fx++) {
                        // Compute shared memory position
                        int sm_pos_y = sm_y + fy;
                        int sm_pos_x = sm_x + fx;

                        // Get input value from shared memory
                        float in_val = shared_input[sm_pos_y * tile_w + sm_pos_x];

                        // Get weights for y and x offsets
                        float w_y_val = weight[(ky * ksize + kx) * 2 * ksize * ksize + fy * ksize + fx];
                        float w_x_val = weight[(ky * ksize + kx) * 2 * ksize * ksize + ksize * ksize + fy * ksize + fx];

                        // Accumulate weighted values
                        offset_y_sum += in_val * w_y_val;
                        offset_x_sum += in_val * w_x_val;
                    }
                }

                // Store the offset values
                offset[offset_base_idx] = offset_y_sum;     // y-offset
                offset[offset_base_idx + 1] = offset_x_sum; // x-offset
            }
        }
    }
}

/**
 * Deformable Convolution with Shared Memory Optimization
 *
 * This convolution uses the offsets generated by the offset-generating convolution
 * to sample from the input feature map at variable positions.
 * Uses shared memory to cache weights for faster access.
 */
__global__ void deform_conv2d(
    float* input,     // Input feature map
    float* offset,    // Offsets from offset-generating convolution (2 * ksize * ksize values per output position)
    float* weight,    // Convolution weights
    float* output,    // Output feature map
    int in_h, int in_w,
    int out_h, int out_w,
    int ksize, int stride, int padding
) {
    // Compute output pixel coordinates
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Define shared memory for weights
    extern __shared__ float shared_weights[];

    // Collaborative loading of weights into shared memory
    // Each thread loads one or more weights
    for (int i = threadIdx.y * blockDim.x + threadIdx.x; i < ksize * ksize; i += blockDim.x * blockDim.y) {
        shared_weights[i] = weight[i];
    }

    // Ensure all threads have loaded their part of the weights
    __syncthreads();

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

                // Get the weight for this kernel position from shared memory
                float w_val = shared_weights[ky * ksize + kx];

                // Accumulate weighted value
                sum += in_val * w_val;
            }
        }

        // Store the result
        output[y * out_w + x] = sum;
    }
}

/**
 * Deformable Convolution with Per-Kernel Output Capture
 *
 * This version of deformable convolution captures the contribution of each kernel position
 * to the output, allowing us to analyze the effect of offsets on the final result.
 */
__global__ void deform_conv2d_with_capture(
    float* input,     // Input feature map
    float* offset,    // Offsets from offset-generating convolution
    float* weight,    // Convolution weights
    float* output,    // Output feature map
    float* kernel_outputs, // Per-kernel outputs
    int in_h, int in_w,
    int out_h, int out_w,
    int ksize, int stride, int padding
) {
    // Compute output pixel coordinates
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Define shared memory for weights
    extern __shared__ float shared_weights[];

    // Collaborative loading of weights into shared memory
    for (int i = threadIdx.y * blockDim.x + threadIdx.x; i < ksize * ksize; i += blockDim.x * blockDim.y) {
        shared_weights[i] = weight[i];
    }

    __syncthreads();

    if (x < out_w && y < out_h) {
        float sum = 0.0f;

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

                // Get the weight for this kernel position from shared memory
                float w_val = shared_weights[ky * ksize + kx];

                // Calculate the contribution of this kernel position
                float kernel_output = in_val * w_val;

                // Store the per-kernel output
                kernel_outputs[(y * out_w + x) * ksize * ksize + ky * ksize + kx] = kernel_output;

                // Accumulate weighted value
                sum += kernel_output;
            }
        }

        // Store the result
        output[y * out_w + x] = sum;
    }
}

/**
 * Example of how to use the two-stage deformable convolution approach
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

    // Allocate memory for input, weights, offsets, and output
    float *d_input, *d_offset_weight, *d_conv_weight, *d_offset, *d_output;
    cudaMalloc(&d_input, in_h * in_w * sizeof(float));

    // Weights for offset-generating convolution (needs 2 * ksize * ksize * ksize * ksize values)
    // For each output offset (2 * ksize * ksize), we need a separate ksize * ksize filter
    cudaMalloc(&d_offset_weight, 2 * ksize * ksize * ksize * ksize * sizeof(float));

    // Weights for the deformable convolution
    cudaMalloc(&d_conv_weight, ksize * ksize * sizeof(float));

    // Offsets (2 values per kernel position)
    cudaMalloc(&d_offset, out_h * out_w * ksize * ksize * 2 * sizeof(float));

    // Output of the deformable convolution
    cudaMalloc(&d_output, out_h * out_w * sizeof(float));

    // Define grid and block dimensions
    dim3 block(16, 16);
    dim3 grid((out_w + block.x - 1) / block.x, (out_h + block.y - 1) / block.y);

    // Step 1: Generate offsets using the offset-generating convolution
    offset_generating_conv2d<<<grid, block>>>(
        d_input, d_offset_weight, d_offset,
        in_h, in_w, out_h, out_w,
        ksize, stride, padding
    );

    // Step 2: Use the generated offsets in the deformable convolution
    deform_conv2d<<<grid, block>>>(
        d_input, d_offset, d_conv_weight, d_output,
        in_h, in_w, out_h, out_w,
        ksize, stride, padding
    );

    // For comparison, you can also run a standard convolution
    standard_conv2d<<<grid, block>>>(
        d_input, d_conv_weight, d_output,
        in_h, in_w, out_h, out_w,
        ksize, stride, padding
    );

    // Free memory
    cudaFree(d_input);
    cudaFree(d_offset_weight);
    cudaFree(d_conv_weight);
    cudaFree(d_offset);
    cudaFree(d_output);
}

/**
 * Main function to demonstrate the two-stage deformable convolution
 */
int main() {
    printf("Demonstrating two-stage deformable convolution\n");

    // Input dimensions
    int in_h = 28;
    int in_w = 28;

    // Kernel parameters
    int ksize = 3;
    int stride = 1;
    int padding = 1;

    // Output dimensions
    int out_h = (in_h + 2 * padding - ksize) / stride + 1;
    int out_w = (in_w + 2 * padding - ksize) / stride + 1;

    // Allocate and initialize input data
    float *d_input;
    cudaMalloc(&d_input, in_h * in_w * sizeof(float));

    // Initialize input with some values (could be an image)
    float *h_input = new float[in_h * in_w];
    for (int i = 0; i < in_h * in_w; i++) {
        h_input[i] = (float)(rand() % 100) / 100.0f;  // Random values between 0 and 1
    }
    cudaMemcpy(d_input, h_input, in_h * in_w * sizeof(float), cudaMemcpyHostToDevice);

    // Allocate and initialize weights for offset generation
    float *d_offset_weight;
    cudaMalloc(&d_offset_weight, 2 * ksize * ksize * ksize * ksize * sizeof(float));

    float *h_offset_weight = new float[2 * ksize * ksize * ksize * ksize];
    for (int i = 0; i < 2 * ksize * ksize * ksize * ksize; i++) {
        h_offset_weight[i] = (float)(rand() % 200 - 100) / 1000.0f;  // Small random values
    }
    cudaMemcpy(d_offset_weight, h_offset_weight, 2 * ksize * ksize * ksize * ksize * sizeof(float), cudaMemcpyHostToDevice);

    // Allocate and initialize weights for deformable convolution
    float *d_conv_weight;
    cudaMalloc(&d_conv_weight, ksize * ksize * sizeof(float));

    float *h_conv_weight = new float[ksize * ksize];
    for (int i = 0; i < ksize * ksize; i++) {
        h_conv_weight[i] = (float)(rand() % 200 - 100) / 100.0f;  // Random values
    }
    cudaMemcpy(d_conv_weight, h_conv_weight, ksize * ksize * sizeof(float), cudaMemcpyHostToDevice);

    // Allocate memory for offsets and output
    float *d_offset, *d_output;
    cudaMalloc(&d_offset, out_h * out_w * ksize * ksize * 2 * sizeof(float));
    cudaMalloc(&d_output, out_h * out_w * sizeof(float));

    // Define grid and block dimensions
    dim3 block(16, 16);
    dim3 grid((out_w + block.x - 1) / block.x, (out_h + block.y - 1) / block.y);

    // Calculate shared memory size for offset_generating_conv2d
    int tile_width = block.x + (ksize - 1);
    int tile_height = block.y + (ksize - 1);
    int offset_shared_mem_size = tile_width * tile_height * sizeof(float);

    printf("\n--- STARTING TWO-STAGE DEFORMABLE CONVOLUTION ---\n");
    printf("Input dimensions: %d x %d\n", in_h, in_w);
    printf("Output dimensions: %d x %d\n", out_h, out_w);
    printf("Kernel size: %d, Stride: %d, Padding: %d\n\n", ksize, stride, padding);

    // Step 1: Generate offsets using the offset-generating convolution
    printf("Step 1: Generating offsets with offset_generating_conv2d\n");
    printf("  - Using shared memory size: %d bytes\n", offset_shared_mem_size);
    printf("  - Grid dimensions: (%d, %d)\n", grid.x, grid.y);
    printf("  - Block dimensions: (%d, %d)\n", block.x, block.y);

    offset_generating_conv2d<<<grid, block, offset_shared_mem_size>>>(
        d_input, d_offset_weight, d_offset,
        in_h, in_w, out_h, out_w,
        ksize, stride, padding
    );

    // Wait for the kernel to complete
    cudaDeviceSynchronize();
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
    }

    // Calculate shared memory size for deform_conv2d (just for weights)
    int deform_shared_mem_size = ksize * ksize * sizeof(float);

    // Step 2: Use the generated offsets in the deformable convolution
    printf("\nStep 2: Applying deformable convolution with the generated offsets\n");
    printf("  - Using shared memory size: %d bytes\n", deform_shared_mem_size);
    printf("  - Grid dimensions: (%d, %d)\n", grid.x, grid.y);
    printf("  - Block dimensions: (%d, %d)\n", block.x, block.y);
    printf("  - Processing entire %d x %d input matrix\n", in_h, in_w);
    printf("  - Generating %d x %d output matrix\n", out_h, out_w);

    deform_conv2d<<<grid, block, deform_shared_mem_size>>>(
        d_input, d_offset, d_conv_weight, d_output,
        in_h, in_w, out_h, out_w,
        ksize, stride, padding
    );

    // Wait for the kernel to complete
    cudaDeviceSynchronize();
    error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
    }

    // Retrieve and display some of the results
    float *h_output = new float[out_h * out_w];
    cudaMemcpy(h_output, d_output, out_h * out_w * sizeof(float), cudaMemcpyDeviceToHost);
    printf("\nOutput data transfer complete: %d bytes\n", out_h * out_w * sizeof(float));

    // Also retrieve some offset values to display
    float *h_offset = new float[out_h * out_w * ksize * ksize * 2];
    cudaMemcpy(h_offset, d_offset, out_h * out_w * ksize * ksize * 2 * sizeof(float), cudaMemcpyDeviceToHost);
    printf("Offset data transfer complete: %d bytes\n", out_h * out_w * ksize * ksize * 2 * sizeof(float));

    // Display full output matrix dimensions
    printf("\nFull output matrix dimensions: %d x %d\n", out_h, out_w);

    // Display a preview of the output matrix
    int preview_size = 27; // Show 5x5 preview by default
    printf("\nOutput matrix preview (%dx%d values):\n", preview_size, preview_size);
    for (int i = 0; i < preview_size && i < out_h; i++) {
        for (int j = 0; j < preview_size && j < out_w; j++) {
            printf("%8.4f ", h_output[i * out_w + j]);
        }
        printf("\n");
    }

    // Display center of the output matrix if it's large enough
    if (out_h > 10 && out_w > 10) {
        int center_y = out_h / 2;
        int center_x = out_w / 2;
        printf("\nCenter of output matrix (position (%d,%d)): %8.4f\n",
               center_y, center_x, h_output[center_y * out_w + center_x]);
    }

    // Display corner values to show the full extent of the matrix
    if (out_h > 5 && out_w > 5) {
        printf("\nCorner values of output matrix:\n");
        printf("Top-left     (0,0):              %8.4f\n", h_output[0]);
        printf("Top-right    (0,%d):             %8.4f\n", out_w-1, h_output[out_w-1]);
        printf("Bottom-left  (%d,0):             %8.4f\n", out_h-1, h_output[(out_h-1) * out_w]);
        printf("Bottom-right (%d,%d):            %8.4f\n", out_h-1, out_w-1, h_output[(out_h-1) * out_w + (out_w-1)]);
    }

    // Display offset values for multiple positions
    printf("\nOffset values for selected positions:\n");

    // Top-left position (0,0)
    printf("\nPosition (0,0) - Output value: %8.4f\n", h_output[0]);
    printf("  Kernel Pos |   Y-Offset |   X-Offset\n");
    printf("------------+------------+------------\n");
    for (int ky = 0; ky < ksize; ky++) {
        for (int kx = 0; kx < ksize; kx++) {
            int offset_idx = ((0 * out_w + 0) * ksize * ksize + ky * ksize + kx) * 2;
            printf("   (%d,%d)     |  %8.3f  |  %8.3f\n",
                   ky, kx,
                   h_offset[offset_idx], h_offset[offset_idx + 1]);
        }
    }

    // Center position if matrix is large enough
    if (out_h > 10 && out_w > 10) {
        int center_y = out_h / 2;
        int center_x = out_w / 2;
        printf("\nPosition (%d,%d) - Output value: %8.4f\n",
               center_y, center_x, h_output[center_y * out_w + center_x]);
        printf("  Kernel Pos |   Y-Offset |   X-Offset\n");
        printf("------------+------------+------------\n");
        for (int ky = 0; ky < ksize; ky++) {
            for (int kx = 0; kx < ksize; kx++) {
                int offset_idx = ((center_y * out_w + center_x) * ksize * ksize + ky * ksize + kx) * 2;
                printf("   (%d,%d)     |  %8.3f  |  %8.3f\n",
                       ky, kx,
                       h_offset[offset_idx], h_offset[offset_idx + 1]);
            }
        }
    }

    // Clean up
    printf("\nCleaning up resources...\n");
    delete[] h_input;
    printf("  - Freed host input memory\n");
    delete[] h_offset_weight;
    printf("  - Freed host offset weight memory\n");
    delete[] h_conv_weight;
    printf("  - Freed host convolution weight memory\n");
    delete[] h_output;
    printf("  - Freed host output memory\n");
    delete[] h_offset;
    printf("  - Freed host offset memory\n");

    cudaFree(d_input);
    printf("  - Freed device input memory\n");
    cudaFree(d_offset_weight);
    printf("  - Freed device offset weight memory\n");
    cudaFree(d_conv_weight);
    printf("  - Freed device convolution weight memory\n");
    cudaFree(d_offset);
    printf("  - Freed device offset memory\n");
    cudaFree(d_output);
    printf("  - Freed device output memory\n");

    printf("\nTwo-stage deformable convolution completed successfully!\n");
    printf("--- EXECUTION COMPLETE ---\n");
    return 0;
}
