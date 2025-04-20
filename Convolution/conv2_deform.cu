#if __cplusplus > 202002L // C++23 or later -> C++98 is 199711L, C++11 is 201103L, C++14 is 201402L, C++17 is 201703L, C++20 is 202002L
#define CPP26
#include <assert.h>  // to use assert, defined in C++26 only
#endif
#include <iostream>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <device_launch_parameters.h>
#include "main_header.cuh"

#define BLOCK_SIZE 16 // Thread block dimensions

// -----------------------------------------------------------------
// CUDA kernel to initialize random matrices
template <typename T>
__global__ void init_matrix(T* mat, int rows, int cols, float scale = 1.0f) {
    // Calculate the global thread indices
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int index = idy * cols + idx;

    // Check if the thread is within matrix bounds
    if (idx < cols && idy < rows) {
        // Initialize the random state
        curandState state;
		curand_init(1234, index, 0, &state);  // Use index for unique seed

        // Assign a random value to the matrix element, scaled and shifted to [-0.5, 0.5]
        mat[index] = scale * (curand_uniform(&state) - 0.5f);
    }
}

// -----------------------------------------------------------------
// Simple kernel to initialize an array with random values
__global__ void init_matrix_v2(float* mat, int size, float scale, float fixed_value) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        if (fixed_value != 0.0f) {
            mat[idx] = fixed_value;
        } else {
            // A simple (deterministic) pseudo-random initializer for demonstration.
            unsigned int seed = idx;
            mat[idx] = scale * (((seed * 16807u) % 2147483647u) / float(2147483647u) - 0.5f);
        }
    }
}


// Helper function for bilinear interpolation
__device__ float bilinear_interpolate(float* input, int in_h, int in_w, float y, float x) {
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
    bool valid_x1 = (x1 >= 0 && x1 < in_w);
    bool valid_x2 = (x2 >= 0 && x2 < in_w);
    bool valid_y1 = (y1 >= 0 && y1 < in_h);
    bool valid_y2 = (y2 >= 0 && y2 < in_h);

    // Get pixel values (with zero padding for out-of-bounds)
    float v11 = (valid_y1 && valid_x1) ? input[y1 * in_w + x1] : 0.0f;
    float v12 = (valid_y1 && valid_x2) ? input[y1 * in_w + x2] : 0.0f;
    float v21 = (valid_y2 && valid_x1) ? input[y2 * in_w + x1] : 0.0f;
    float v22 = (valid_y2 && valid_x2) ? input[y2 * in_w + x2] : 0.0f;

    // Bilinear interpolation
    float value = wy1 * wx1 * v11 + wy1 * wx2 * v12 + wy2 * wx1 * v21 + wy2 * wx2 * v22;

    return value;
}

// Forward pass for deformable convolution with bilinear interpolation
/* Deformable Convolutional Networks https://arxiv.org/abs/1703.06211 have 2 type of parameters:
		- weight: the convolutional kernel weights (ksize x ksize) so 2D
		- offset: the learned offsets to the sampling positions (out_h,out_w,2*ksize*ksize) tensor
*/
__global__ void deform_conv2d_forward(float* input, float* offset, float* weight, float* output,
    int in_h, int in_w, int out_h, int out_w, int ksize, int stride) {
    // Compute output pixel coordinates
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Ensure the thread is within output dimensions
    if (x < out_w && y < out_h) {
        float sum = 0.0f; // Accumulate convolution result

        // Calculate the center of the receptive field
        int center_h = y * stride;
        int center_w = x * stride;

        // Iterate over the kernel window
        for (int i = 0; i < ksize; i++) {
            for (int j = 0; j < ksize; j++) {
                // Compute the index for the offset values
                // Each kernel position (i,j) has its own offset (dx, dy)
                int offset_idx = ((y * out_w + x) * ksize * ksize + i * ksize + j) * 2;

                // Compute the sampling positions with learned offsets
                float px = center_w + j - (ksize - 1) / 2.0f + offset[offset_idx];      // Horizontal position
                float py = center_h + i - (ksize - 1) / 2.0f + offset[offset_idx + 1];  // Vertical position

                // Use bilinear interpolation to sample from the input
                float val = bilinear_interpolate(input, in_h, in_w, py, px);

                // Apply convolution
                sum += val * weight[i * ksize + j];
            }
        }

        // Store the computed output value
        output[y * out_w + x] = sum;
    }
}


// Custom atomicAdd for float (only needed for old architectures)
__device__ float atomicAddFloat(float* address, float val) {
    return atomicAdd(address, val);
}


// Helper function to compute gradients for bilinear interpolation
__device__ void bilinear_interpolate_gradient(float* grad_input, int in_h, int in_w, float y, float x, float gradient,
                                             float* grad_x = nullptr, float* grad_y = nullptr) {
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
    bool valid_x1 = (x1 >= 0 && x1 < in_w);
    bool valid_x2 = (x2 >= 0 && x2 < in_w);
    bool valid_y1 = (y1 >= 0 && y1 < in_h);
    bool valid_y2 = (y2 >= 0 && y2 < in_h);

    // Accumulate gradients for the input
    if (valid_y1 && valid_x1) atomicAddFloat(&grad_input[y1 * in_w + x1], gradient * wy1 * wx1);
    if (valid_y1 && valid_x2) atomicAddFloat(&grad_input[y1 * in_w + x2], gradient * wy1 * wx2);
    if (valid_y2 && valid_x1) atomicAddFloat(&grad_input[y2 * in_w + x1], gradient * wy2 * wx1);
    if (valid_y2 && valid_x2) atomicAddFloat(&grad_input[y2 * in_w + x2], gradient * wy2 * wx2);

    // Compute gradients for x and y if requested
    if (grad_x != nullptr && grad_y != nullptr) {
        // Get pixel values (with zero padding for out-of-bounds)
        float v11 = (valid_y1 && valid_x1) ? grad_input[y1 * in_w + x1] : 0.0f;
        float v12 = (valid_y1 && valid_x2) ? grad_input[y1 * in_w + x2] : 0.0f;
        float v21 = (valid_y2 && valid_x1) ? grad_input[y2 * in_w + x1] : 0.0f;
        float v22 = (valid_y2 && valid_x2) ? grad_input[y2 * in_w + x2] : 0.0f;

        // Compute gradients for x and y
        *grad_x = gradient * (wy1 * (v12 - v11) + wy2 * (v22 - v21));
        *grad_y = gradient * (wx1 * (v21 - v11) + wx2 * (v22 - v12));
    }
}

// Backward pass for deformable convolution with bilinear interpolation
__global__ void deform_conv2d_backward(float* grad_output, float* grad_input, float* grad_weight, float* grad_offset,
                                      float* input, float* weight, float* offset,
                                      int in_h, int in_w, int out_h, int out_w, int ksize, int stride) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < out_w && y < out_h) {
        float grad = grad_output[y * out_w + x];

        // Calculate the center of the receptive field
        int center_h = y * stride;
        int center_w = x * stride;

        for (int i = 0; i < ksize; i++) {
            for (int j = 0; j < ksize; j++) {
                // Compute the index for the offset values
                int offset_idx = ((y * out_w + x) * ksize * ksize + i * ksize + j) * 2;

                // Compute the sampling positions with learned offsets
                float px = center_w + j - (ksize - 1) / 2.0f + offset[offset_idx];      // Horizontal position
                float py = center_h + i - (ksize - 1) / 2.0f + offset[offset_idx + 1];  // Vertical position

                // Get the interpolated value
                float val = bilinear_interpolate(input, in_h, in_w, py, px);

                // Gradient for the weight
                atomicAddFloat(&grad_weight[i * ksize + j], grad * val);

                // Gradient for the input and offset
                float grad_val = grad * weight[i * ksize + j];
                float grad_offset_x, grad_offset_y;

                // Compute gradients for input and offset
                bilinear_interpolate_gradient(grad_input, in_h, in_w, py, px, grad_val, &grad_offset_x, &grad_offset_y);

                // Accumulate gradients for offsets
                atomicAddFloat(&grad_offset[offset_idx], grad_offset_x);
                atomicAddFloat(&grad_offset[offset_idx + 1], grad_offset_y);
            }
        }
    }
}


// Something is wrong: offset shouldn't it be the same size of kernel ?
// Backward pass for deformable convolution with detailed offset calculation explanation
__global__ void deform_conv2d_backward_explained(float* grad_output, float* grad_input, float* grad_weight, float* grad_offset, float* input, float* weight, float* offset, int in_h, int in_w, int out_h, int out_w, int ksize, int stride) {
    // Compute output coordinates that this thread is responsible for
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Proceed only if within output boundaries
    if (x < out_w && y < out_h) {
        // Retrieve the gradient corresponding to the output pixel at (x, y)
        float grad = grad_output[y * out_w + x];

        // Loop over the kernel window for the convolution operation
        for (int i = 0; i < ksize; i++) {
            for (int j = 0; j < ksize; j++) {
                // Compute the index for offset values corresponding to the output pixel (x, y)
                int offset_idx = (y * out_w + x) * 2;

                // Retrieve the learned offsets:
                // offset[offset_idx] corresponds to horizontal displacement (offset_x)
                // offset[offset_idx + 1] corresponds to vertical displacement (offset_y)
                float offset_x = offset[offset_idx];
                float offset_y = offset[offset_idx + 1];

                // Calculate the sampling positions in the input using the stride, kernel displacement, and learned offsets
                // The expected position without offset would be (x*stride + j, y*stride + i)
                // The addition of offset_x and offset_y perturbs these positions based on learned deformations.
                int px = x * stride + j + static_cast<int>(offset_x);
                int py = y * stride + i + static_cast<int>(offset_y);

                // Check if the calculated position falls within the input boundaries
                if (px >= 0 && px < in_w && py >= 0 && py < in_h) {
                    // Accumulate gradients for the input pixel using the weight
                    atomicAddFloat(&grad_input[py * in_w + px], grad * weight[i * ksize + j]);
                    // Accumulate gradients for the convolution kernel weight using the input pixel
                    atomicAddFloat(&grad_weight[i * ksize + j], grad * input[py * in_w + px]);

                    // Accumulate gradients for the offset values
                    // Gradient with respect to the horizontal offset (offset_x) is influenced by the kernel's column position j
                    atomicAddFloat(&grad_offset[offset_idx], grad * input[py * in_w + px] * j);
                    // Gradient with respect to the vertical offset (offset_y) is influenced by the kernel's row position i
                    atomicAddFloat(&grad_offset[offset_idx + 1], grad * input[py * in_w + px] * i);
                }
            }
        }
    }
}

// CUDA kernel for the backward pass of deformable convolution with detailed explanations.
__global__ void deform_conv2d_backward_explained_v2(
    float* grad_output,   // Gradient of the loss with respect to the output feature map (size: out_h * out_w)
    float* grad_input,    // Gradient to be accumulated for the input image (size: in_h * in_w)
    float* grad_weight,   // Gradient to be accumulated for the convolution weights (size: ksize * ksize)
    float* grad_offset,   // Gradient to be accumulated for the offset parameters (size: out_h * out_w * 2)
    float* input,         // Original input image (size: in_h * in_w)
    float* weight,        // Convolution filter weights (size: ksize * ksize)
    float* offset,        // Learned offset parameters (each output pixel has 2 offsets: x and y)
    int in_h, int in_w,   // Input image dimensions
    int out_h, int out_w, // Output feature map dimensions
    int ksize,            // Kernel size (assumed square: ksize x ksize)
    int stride)           // Stride for convolution
{
    // Compute the current output pixel coordinates (x, y) based on the thread index.
    int x = blockIdx.x * blockDim.x + threadIdx.x; // column index in output
    int y = blockIdx.y * blockDim.y + threadIdx.y; // row index in output

    // Ensure that the thread corresponds to a valid output pixel.
    if (x < out_w && y < out_h) {
        // The gradient at the output pixel (from the loss function).
        float grad = grad_output[y * out_w + x];

        // For each position in the convolution kernel:
        for (int i = 0; i < ksize; i++) {
            for (int j = 0; j < ksize; j++) {
                // Each output pixel uses two offset values (for x and y),
                // stored in the 'offset' array consecutively.
                // Compute the base index for the offset corresponding to output pixel (x, y).
                int offset_idx = (y * out_w + x) * 2;

                // Calculate the deformed sampling positions on the input.
                // The original sampling position would be (x * stride + j, y * stride + i).
                // We add the learned offsets (one for x, one for y) to these positions.
                int px = x * stride + j + offset[offset_idx];       // Adjusted horizontal position
                int py = y * stride + i + offset[offset_idx + 1];     // Adjusted vertical position

                // Only accumulate gradients if the deformed sampling position is within bounds.
                if (px >= 0 && px < in_w && py >= 0 && py < in_h) {
                    // 1. Compute gradient w.r.t. the input pixel:
                    //    The input pixel at (py, px) contributed to the output via multiplication with the weight.
                    //    So its gradient is scaled by that weight and the gradient from the output.
                    atomicAdd(&grad_input[py * in_w + px], grad * weight[i * ksize + j]);

                    // 2. Compute gradient w.r.t. the convolution weight:
                    //    The weight multiplies the input pixel value at the deformed location.
                    //    So, we accumulate the product of the output gradient and the input pixel.
                    atomicAdd(&grad_weight[i * ksize + j], grad * input[py * in_w + px]);

                    // 3. Compute gradient w.r.t. the offset parameters:
                    //    The offset influences which input pixel was sampled.
                    //    In this simplified example, we assume the gradient for the offset is proportional to:
                    //    - The output gradient,
                    //    - The value of the input pixel,
                    //    - A factor (here, j for horizontal and i for vertical) to indicate the sensitivity.
                    //    (Note: In a true deformable convolution, bilinear interpolation is used,
                    //     and the offset gradients would be computed via a more complex chain rule.)
                    atomicAdd(&grad_offset[offset_idx], grad * input[py * in_w + px] * j);       // Horizontal offset gradient
                    atomicAdd(&grad_offset[offset_idx + 1], grad * input[py * in_w + px] * i);   // Vertical offset gradient
                }
            }
        }
    }
}



// -----------------------------------------------------------------
// Forward Pass Kernel for Deformable Convolution with Bilinear Interpolation
//
// For each output pixel at (x, y) in the output feature map:
//   1. A receptive field of size ksize x ksize is extracted from the input.
//   2. For each kernel element (i, j):
//      - The sampling position on the input is deformed by adding learned
//        offsets specific to each kernel position (i,j).
//      - Bilinear interpolation is used to sample from the input at the deformed position.
//      - The interpolated value is multiplied by the corresponding filter weight and accumulated.
//   3. The result is stored in the output feature map.
__global__ void deform_conv2d_forward_exaplained(
    float* input,    // Input image (size: in_h * in_w)
    float* offset,   // Learned offsets for each output pixel and kernel position (size: out_h * out_w * 2 * ksize * ksize)
    float* weight,   // Convolution filter weights (size: ksize * ksize)
    float* output,   // Output feature map (size: out_h * out_w)
    int in_h, int in_w,
    int out_h, int out_w,
    int ksize,       // Kernel size (assumed square)
    int stride)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;  // output column
    int y = blockIdx.y * blockDim.y + threadIdx.y;  // output row

    if (x < out_w && y < out_h) {
        float sum = 0.0f;

        // Calculate the center of the receptive field
        int center_h = y * stride;
        int center_w = x * stride;

        // Loop over kernel window
        for (int i = 0; i < ksize; i++) {
            for (int j = 0; j < ksize; j++) {
                // Each kernel position (i,j) has its own offset (dx, dy)
                int offset_idx = ((y * out_w + x) * ksize * ksize + i * ksize + j) * 2;

                // Deformed sampling position on input
                float px = center_w + j - (ksize - 1) / 2.0f + offset[offset_idx];      // adjusted x position
                float py = center_h + i - (ksize - 1) / 2.0f + offset[offset_idx + 1];  // adjusted y position

                // Use bilinear interpolation to sample from the input
                float val = bilinear_interpolate(input, in_h, in_w, py, px);

                // Apply convolution
                sum += val * weight[i * ksize + j];
            }
        }
        output[y * out_w + x] = sum;
    }
}


// Single Forwardpass for deformable convolution
int conv2d_deform_infer() {
    int in_h = 7,  // Input height
        in_w = 7,  // Input width
        out_h = 5,  // Output height
        out_w = 5,  // Output width
        ksize = 5,  // Kernel size
        stride = 1;  // Stride

#ifdef CPP26
    assert("Kernel size is correct: " && in_h - ksize >= 0 && in_w - ksize >= 0);  // to try assert in C++26
#endif
    float* input, // Input matrix
        * offset, // Offset matrix
        * weight,  // Weight matrix
        * output,  // Output matrix
        * grad_output,  // Gradient of output matrix
        * grad_input,  // Gradient of input matrix
        * grad_weight,  // Gradient of weight matrix
        * grad_offset;  // Gradient of offset matrix

	size_t size_in = in_h * in_w * sizeof(float);    // Size of input matrix
	size_t size_out = out_h * out_w * sizeof(float);  // Size of output matrix
	size_t size_k = ksize * ksize * sizeof(float);  // Size of kernel matrix

	// Allocate memory for matrices
    cudaMallocManaged(&input, size_in);
    cudaMallocManaged(&offset, size_out * 2 * ksize * ksize);  // 2 offsets (x,y) per kernel position per output pixel
    cudaMallocManaged(&weight, size_k);
    cudaMallocManaged(&output, size_out);
    cudaMallocManaged(&grad_output, size_out);
    cudaMallocManaged(&grad_input, size_in);
    cudaMallocManaged(&grad_weight, size_k);
    cudaMallocManaged(&grad_offset, size_out * 2 * ksize * ksize);  // Gradients for offsets

	dim3 block(BLOCK_SIZE, BLOCK_SIZE);  // Thread block dimensions
	dim3 grid((in_w + block.x - 1) / block.x, (in_h + block.y - 1) / block.y);   // Grid dimensions

    init_matrix << <grid, block >> > (input, in_h, in_w);
    init_matrix << <grid, block >> > (offset, out_h, out_w * 2 * ksize * ksize);  // Initialize offsets
    init_matrix << <grid, block >> > (weight, ksize, ksize);
    init_matrix << <grid, block >> > (grad_output, out_h, out_w);
    cudaDeviceSynchronize();

    deform_conv2d_forward << <grid, block >> > (input, offset, weight, output, in_h, in_w, out_h, out_w, ksize, stride);
    cudaDeviceSynchronize();

	// Print the Ksize x Ksize Offset matrix
    std::cout << "Offset matrix (" << ksize << "x" << ksize << "):" << std::endl;
    for (int i = 0; i < ksize; i++) {
        for (int j = 0; j < ksize; j++) {
            std::cout << offset[i * out_w * 2 + j * 2] << " ";
        }
        std::cout << std::endl;
    }

	// Print the Ksize x Ksize Weight matrix
    std::cout << "Weight matrix (" << ksize << "x" << ksize << "):" << std::endl;
	for (int i = 0; i < ksize; i++) {
		for (int j = 0; j < ksize; j++) {
			std::cout << weight[i * ksize + j] << " ";
		}
		std::cout << std::endl;
	}

	// Print the out_h x out_w Output matrix
	std::cout << "Output matrix (" << out_h << "x" << out_w << "):" << std::endl;
	for (int i = 0; i < out_h; i++) {
		for (int j = 0; j < out_w; j++) {
			std::cout << output[i * out_w + j] << " ";
		}
		std::cout << std::endl;
	}


	//deform_conv2d_backward << <grid, block >> > (grad_output, grad_input, grad_weight, grad_offset, input, weight, offset, in_h, in_w, out_h, out_w, ksize, stride);  // -> Momentarily disabled
    deform_conv2d_backward_explained_v2 << <grid, block >> > (grad_output, grad_input, grad_weight, grad_offset, input, weight, offset, in_h, in_w, out_h, out_w, ksize, stride);
    cudaDeviceSynchronize();

    std::cout << "First gradient value (input): " << grad_input[0] << std::endl;

    cudaFree(input);
    cudaFree(offset);
    cudaFree(weight);
    cudaFree(output);
    cudaFree(grad_output);
    cudaFree(grad_input);
    cudaFree(grad_weight);
    cudaFree(grad_offset);
    return 0;
}

// -----------------------------------------------------------------
// Kernel to update weights using gradient descent
__global__ void update_weights_kernel_v2(float* weight, float* grad_weight, int filter_size, float lr) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < filter_size) {
        weight[idx] -= lr * grad_weight[idx];
    }
}


// -----------------------------------------------------------------
// Kernel to update offset parameters using gradient descent
__global__ void update_offsets_kernel(float* offset, float* grad_offset, int offset_size, float lr) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < offset_size) {
        offset[idx] -= lr * grad_offset[idx];
    }
}


// -----------------------------------------------------------------
// Main Function: Training Loop
//
// This function runs a training loop for X epochs. In each epoch:
//   1. A forward pass is performed using the deformable convolution kernel.
//   2. The loss is computed as the mean squared error (MSE) between the output and a target of 0.
//   3. The gradient of the loss (grad_output) is computed.
//   4. The backward pass kernel computes gradients for input, weights, and offsets.
//   5. The weights and offsets are updated using gradient descent.
// The goal is to drive the output toward zero.
int conv2d_deform_training_loop() {
    // -----------------------------------------------------------------
    // Define dimensions and parameters
    // -----------------------------------------------------------------
    int in_h = 32, in_w = 32;      // Input image dimensions
    int ksize = 5;                 // Convolution kernel size (3x3)
    int stride = 1;
    int out_h = in_h - ksize + 1;  // Output dimensions computed from input and kernel size
    int out_w = in_w - ksize + 1;
    int input_size = in_h * in_w;
    int output_size = out_h * out_w;
    int filter_size = ksize * ksize;
    // offset_size already defined above

    // -----------------------------------------------------------------
    // Allocate unified memory (accessible from both CPU and GPU)
    // -----------------------------------------------------------------
    float* input, * offset, * weight, * output;
    float* grad_output, * grad_input, * grad_weight, * grad_offset;
    cudaMallocManaged(&input, input_size * sizeof(float));
    int offset_size = output_size * 2 * ksize * ksize;  // Two offsets (x,y) per kernel position per output pixel
    cudaMallocManaged(&offset, offset_size * sizeof(float));
    cudaMallocManaged(&weight, filter_size * sizeof(float));
    cudaMallocManaged(&output, output_size * sizeof(float));
    cudaMallocManaged(&grad_output, output_size * sizeof(float));
    cudaMallocManaged(&grad_input, input_size * sizeof(float));
    cudaMallocManaged(&grad_weight, filter_size * sizeof(float));
    cudaMallocManaged(&grad_offset, offset_size * sizeof(float));

    // -----------------------------------------------------------------
    // Initialize input, offset, and weight with random values.
    // -----------------------------------------------------------------
    int numThreads = 256;
    int blocks_input = (input_size + numThreads - 1) / numThreads;
    int blocks_weight = (filter_size + numThreads - 1) / numThreads;
    int blocks_offset = (offset_size + numThreads - 1) / numThreads;
	init_matrix_v2 << <blocks_input, numThreads >> > (input, input_size, 1.0f, 0.0f);  // Initialize input with random values
	init_matrix_v2 << <blocks_offset, numThreads >> > (offset, offset_size, 1.0f, 0.0f);  // Initialize offset with random values
	init_matrix_v2 << <blocks_weight, numThreads >> > (weight, filter_size, 1.0f, 1.0f);  // Initialize weight with random values
    cudaDeviceSynchronize();

    // -----------------------------------------------------------------
    // Training Loop Parameters
    // -----------------------------------------------------------------
    int numEpochs = 1000;
    float learning_rate = 0.001f;

    // Grid dimensions for the forward and backward kernels.
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((out_w + BLOCK_SIZE - 1) / BLOCK_SIZE, (out_h + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // -----------------------------------------------------------------
    // Training Loop: Iteratively perform forward and backward passes and update parameters.
    // -----------------------------------------------------------------
    for (int epoch = 0; epoch < numEpochs; epoch++) {
        // Zero the output and gradients from the previous epoch.
        for (int i = 0; i < output_size; i++) {
            output[i] = 0.0f;
            grad_output[i] = 0.0f;
        }
        for (int i = 0; i < input_size; i++) {
            grad_input[i] = 0.0f;
        }
        for (int i = 0; i < filter_size; i++) {
            grad_weight[i] = 0.0f;
        }
        for (int i = 0; i < offset_size; i++) {
            grad_offset[i] = 0.0f;
        }

        // ----------------------- Forward Pass -----------------------
        deform_conv2d_forward_exaplained << <gridDim, blockDim >> > (input, offset, weight, output, in_h, in_w, out_h, out_w, ksize, stride);
        cudaDeviceSynchronize();

        // ----------------------- Loss Computation -----------------------
        // The target output is 0, so we use the mean squared error (MSE) loss.
        float loss = 0.0f;
        for (int i = 0; i < output_size; i++) {
            loss += output[i] * output[i];
            // The gradient of MSE loss with respect to output is: 2*output / output_size.
            grad_output[i] = 2.0f * output[i] / output_size;
        }
        loss /= output_size;

        // Print loss every 10 epochs.
        if (epoch % 10 == 0) {
            std::cout << "Epoch " << epoch << " Loss: " << loss << std::endl;
        }

        // ----------------------- Backward Pass -----------------------
        // Compute gradients for input, weight, and offset using the backward kernel.
        deform_conv2d_backward_explained << <gridDim, blockDim >> > (grad_output, grad_input, grad_weight, grad_offset, input, weight, offset, in_h, in_w, out_h, out_w, ksize, stride);
        cudaDeviceSynchronize();

        // ----------------------- Update Weights & Offsets -----------------------
        int blocks_update_w = (filter_size + numThreads - 1) / numThreads;
        update_weights_kernel_v2 << <blocks_update_w, numThreads >> > (weight, grad_weight, filter_size, learning_rate);
        int blocks_update_o = (offset_size + numThreads - 1) / numThreads;
        update_offsets_kernel << <blocks_update_o, numThreads >> > (offset, grad_offset, offset_size, learning_rate);
        cudaDeviceSynchronize();
    }

    // -----------------------------------------------------------------
    // Report Final Loss and Sample Weight Value
    // -----------------------------------------------------------------
    float final_loss = 0.0f;
    for (int i = 0; i < output_size; i++) {
        final_loss += output[i] * output[i];
    }
    final_loss /= output_size;
    std::cout << "Final Loss: " << final_loss << std::endl;
    std::cout << "First filter weight after training: " << weight[0] << std::endl;

    // Print the weights of the deform convolution kernel as a 2D matrix
    std::cout << "Weights of the deform convolution kernel (" << ksize << "x" << ksize << "):" << std::endl;
    for (int i = 0; i < ksize; i++) {
        for (int j = 0; j < ksize; j++) {
            std::cout << weight[i * ksize + j] << " ";
        }
        std::cout << std::endl;
    }

    // -----------------------------------------------------------------
    // Free Unified Memory
    // -----------------------------------------------------------------
    cudaFree(input);
    cudaFree(offset);
    cudaFree(weight);
    cudaFree(output);
    cudaFree(grad_output);
    cudaFree(grad_input);
    cudaFree(grad_weight);
    cudaFree(grad_offset);

    return 0;
}