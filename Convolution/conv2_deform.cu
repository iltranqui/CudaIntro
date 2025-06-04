#if __cplusplus > 202002L // C++23 or later -> C++98 is 199711L, C++11 is 201103L, C++14 is 201402L, C++17 is 201703L, C++20 is 202002L
#define CPP26
#include <assert.h>  // to use assert, defined in C++26 only
#endif
#include <iostream>
#include <vector> // Added for host-side data verification if needed
#include <cmath>  // Added for floor
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <device_launch_parameters.h>
#include "main_header.cuh" // Assuming this is needed for your project

#define BLOCK_SIZE 16 // Thread block dimensions

// Macro for checking CUDA errors
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while (0)


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
        // Using grid/block dimensions and thread indices for a more robust seed
        unsigned long long seed = (unsigned long long)blockIdx.x * blockDim.x * gridDim.y * blockDim.y +
                                  (unsigned long long)blockIdx.y * blockDim.y * blockDim.x +
                                  (unsigned long long)threadIdx.y * blockDim.x + threadIdx.x;
        curand_init(seed, index, 0, &state); // Use unique seed and sequence per thread

        // Assign a random value to the matrix element, scaled and shifted to [-0.5*scale, 0.5*scale]
        mat[index] = scale * (curand_uniform(&state) - 0.5f);
    }
}

// -----------------------------------------------------------------
// Simple kernel to initialize an array with random values or fixed value
__global__ void init_matrix_v2(float* mat, int size, float scale, float fixed_value) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        if (fixed_value != 0.0f) { // Using a small epsilon check might be safer for floats
            mat[idx] = fixed_value;
        } else {
            // A simple (deterministic) pseudo-random initializer for demonstration.
            // Consider using curand for better randomness if needed here too.
            unsigned int seed = idx + 12345; // Simple seed
            mat[idx] = scale * (((seed * 1103515245u + 12345u)) / (float)(0xFFFFFFFFU) - 0.5f);
        }
    }
}


// Helper function for bilinear interpolation (Forward) - unchanged
__device__ float bilinear_interpolate(const float* input, int in_h, int in_w, float y, float x) {
    // Get the four nearest integer coordinates
    int x1 = floorf(x); // Use floorf for float
    int x2 = x1 + 1;
    int y1 = floorf(y); // Use floorf for float
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
    // Use const qualifier as input should not be modified here
    const float v11 = (valid_y1 && valid_x1) ? input[y1 * in_w + x1] : 0.0f;
    const float v12 = (valid_y1 && valid_x2) ? input[y1 * in_w + x2] : 0.0f;
    const float v21 = (valid_y2 && valid_x1) ? input[y2 * in_w + x1] : 0.0f;
    const float v22 = (valid_y2 && valid_x2) ? input[y2 * in_w + x2] : 0.0f;

    // Bilinear interpolation
    float value = wy1 * wx1 * v11 + wy1 * wx2 * v12 + wy2 * wx1 * v21 + wy2 * wx2 * v22;

    return value;
}

// Forward pass for deformable convolution with bilinear interpolation (unchanged, using bilinear_interpolate)
__global__ void deform_conv2d_forward_exaplained(
    const float* input,  // Input should be const
    const float* offset, // Offset should be const
    const float* weight, // Weight should be const
    float* output,       // Output is written to
    int in_h, int in_w,
    int out_h, int out_w,
    int ksize,
    int stride)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;  // output column
    int y = blockIdx.y * blockDim.y + threadIdx.y;  // output row

    if (x < out_w && y < out_h) {
        float sum = 0.0f;

        // Calculate the center of the receptive field in the *input* corresponding to the output (y, x)
        // Note: Integer division for kernel center might be preferred if ksize is always odd
        // float center_h = y * stride + (ksize - 1) / 2.0f;
        // float center_w = x * stride + (ksize - 1) / 2.0f;
        // Alternative: Calculate top-left corner and add kernel indices directly
        int base_in_y = y * stride;
        int base_in_x = x * stride;

        // Loop over kernel window
        for (int i = 0; i < ksize; i++) { // Kernel row index
            for (int j = 0; j < ksize; j++) { // Kernel column index
                // Each kernel position (i,j) for each output pixel (y,x) has its own offset (dy, dx)
                int offset_idx = ((y * out_w + x) * ksize * ksize + i * ksize + j) * 2;

                // Base sampling position on input grid relative to kernel center
                // Correct calculation: center of receptive field + relative kernel position + offset
                float p_y_base = base_in_y + i; // Base y position for kernel element (i,j)
                float p_x_base = base_in_x + j; // Base x position for kernel element (i,j)

                // Retrieve offsets (d_y, d_x)
                float offset_dx = offset[offset_idx];     // Horizontal offset
                float offset_dy = offset[offset_idx + 1]; // Vertical offset

                // Deformed sampling position on input (float coordinates)
                // Note: DCN paper adds offset to regular grid p0 + pn + Δpn
                // Here p0 is (base_in_x, base_in_y) - top left of patch
                // pn is (j, i) - relative pos in kernel grid
                // So the absolute sampling position is (base_in_x + j + offset_dx, base_in_y + i + offset_dy)
                float px = p_x_base + offset_dx; // adjusted x position
                float py = p_y_base + offset_dy; // adjusted y position

                // Use bilinear interpolation to sample from the input
                float val = bilinear_interpolate(input, in_h, in_w, py, px);

                // Apply convolution
                sum += val * weight[i * ksize + j];
            }
        }
        output[y * out_w + x] = sum;
    }
}


// Custom atomicAdd for float (only needed for old architectures, but safe to use)
__device__ float atomicAddFloat(float* address, float val) {
    // On newer architectures (Compute Capability 6.0+), atomicAdd has native float support.
    // On older architectures, this uses an atomicCAS loop.
    // The standard atomicAdd(float*, float) should work correctly on supported devices.
    return atomicAdd(address, val);
}


// *** CORRECTED Helper function to compute gradients for bilinear interpolation ***
__device__ void bilinear_interpolate_gradient(
    float* grad_input, // Gradient to accumulate into input
    const float* input, // *** ADDED: Original input needed for coordinate gradients ***
    int in_h, int in_w,
    float y, float x,   // The fractional coordinates used in forward pass
    float gradient,     // The incoming gradient (dL/dVal_interpolated)
    float* grad_x,      // Output: gradient w.r.t. x coordinate (dL/dx)
    float* grad_y)      // Output: gradient w.r.t. y coordinate (dL/dy)
{
    // Get the four nearest integer coordinates
    int x1 = floorf(x);
    int x2 = x1 + 1;
    int y1 = floorf(y);
    int y2 = y1 + 1;

    // Calculate the interpolation weights (same as forward)
    float wx1 = x2 - x;
    float wx2 = x - x1;
    float wy1 = y2 - y;
    float wy2 = y - y1;

    // Boundary check (needed for both grad_input and coordinate grads)
    bool valid_x1 = (x1 >= 0 && x1 < in_w);
    bool valid_x2 = (x2 >= 0 && x2 < in_w);
    bool valid_y1 = (y1 >= 0 && y1 < in_h);
    bool valid_y2 = (y2 >= 0 && y2 < in_h);

    // -------------------------------------------------
    // 1. Accumulate gradients for the input feature map (dL/dInput)
    // dL/dInput = dL/dVal_interpolated * dVal_interpolated/dInput
    // -------------------------------------------------
    if (valid_y1 && valid_x1) atomicAddFloat(&grad_input[y1 * in_w + x1], gradient * wy1 * wx1);
    if (valid_y1 && valid_x2) atomicAddFloat(&grad_input[y1 * in_w + x2], gradient * wy1 * wx2);
    if (valid_y2 && valid_x1) atomicAddFloat(&grad_input[y2 * in_w + x1], gradient * wy2 * wx1);
    if (valid_y2 && valid_x2) atomicAddFloat(&grad_input[y2 * in_w + x2], gradient * wy2 * wx2);

    // -------------------------------------------------
    // 2. Compute gradients for x and y coordinates (dL/dx, dL/dy)
    // dL/dx = dL/dVal_interpolated * dVal_interpolated/dx
    // dL/dy = dL/dVal_interpolated * dVal_interpolated/dy
    // where Val_interpolated = wy1*(wx1*v11 + wx2*v12) + wy2*(wx1*v21 + wx2*v22)
    // We need the *original input values* v11, v12, v21, v22 here.
    // -------------------------------------------------
    // *** CORRECTED: Read from original input, not grad_input ***
    const float v11 = (valid_y1 && valid_x1) ? input[y1 * in_w + x1] : 0.0f;
    const float v12 = (valid_y1 && valid_x2) ? input[y1 * in_w + x2] : 0.0f;
    const float v21 = (valid_y2 && valid_x1) ? input[y2 * in_w + x1] : 0.0f;
    const float v22 = (valid_y2 && valid_x2) ? input[y2 * in_w + x2] : 0.0f;

    // Compute gradients for x and y using the chain rule and derivatives of bilinear interp.
    // dVal/dx = wy1 * (v12 - v11) + wy2 * (v22 - v21)
    // dVal/dy = wx1 * (v21 - v11) + wx2 * (v22 - v12)
    *grad_x = gradient * (wy1 * (v12 - v11) + wy2 * (v22 - v21));
    *grad_y = gradient * (wx1 * (v21 - v11) + wx2 * (v22 - v12));
}

// *** CORRECTED Backward pass for deformable convolution consistent with forward pass ***
__global__ void deform_conv2d_backward(
    const float* grad_output, // Gradient from next layer (dL/dOutput)
    float* grad_input,        // Gradient to compute for input (dL/dInput)
    float* grad_weight,       // Gradient to compute for weights (dL/dWeight)
    float* grad_offset,       // Gradient to compute for offsets (dL/dOffset)
    const float* input,       // *** ADDED: Original input needed for gradients ***
    const float* weight,      // Original weights needed for grad_input and grad_offset
    const float* offset,      // Original offsets needed to find sampling points
    int in_h, int in_w,
    int out_h, int out_w,
    int ksize,
    int stride)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x; // output column
    int y = blockIdx.y * blockDim.y + threadIdx.y; // output row

    if (x < out_w && y < out_h) {
        // Gradient from the loss w.r.t the output of this layer at position (y, x)
        float grad_out_yx = grad_output[y * out_w + x];

        // Calculate the base input coordinates (top-left of receptive field)
        int base_in_y = y * stride;
        int base_in_x = x * stride;

        // Iterate over the kernel elements
        for (int i = 0; i < ksize; i++) { // Kernel row
            for (int j = 0; j < ksize; j++) { // Kernel column
                // Calculate the index for the offset values for this output pixel and kernel position
                int offset_idx = ((y * out_w + x) * ksize * ksize + i * ksize + j) * 2;

                // Recalculate the deformed sampling position (same logic as forward)
                float p_y_base = base_in_y + i;
                float p_x_base = base_in_x + j;
                float offset_dx = offset[offset_idx];
                float offset_dy = offset[offset_idx + 1];
                float px = p_x_base + offset_dx; // fractional x coord
                float py = p_y_base + offset_dy; // fractional y coord

                // Get the interpolated value from the *original* input (needed for grad_weight)
                // Note: This repeats work from forward pass, could be optimized by passing output values
                // but is correct for calculating gradients from scratch.
                float val_interpolated = bilinear_interpolate(input, in_h, in_w, py, px);

                // ---------------------------------------
                // 1. Calculate Gradient for Weights (dL/dWeight)
                // dL/dW_ij = sum_{y,x} (dL/dOutput_yx * dOutput_yx/dW_ij)
                // dOutput_yx/dW_ij = val_interpolated (at position y,x,i,j)
                // Accumulate using atomic add as multiple threads (output pixels) update the same weight
                // ---------------------------------------
                atomicAddFloat(&grad_weight[i * ksize + j], grad_out_yx * val_interpolated);

                // ---------------------------------------
                // 2. Calculate Gradient flowing back to the interpolated value stage (dL/dVal_interpolated)
                // dL/dVal_interpolated = dL/dOutput_yx * dOutput_yx/dVal_interpolated
                // dOutput_yx/dVal_interpolated = Weight_ij
                // ---------------------------------------
                float grad_val = grad_out_yx * weight[i * ksize + j];

                // ---------------------------------------
                // 3. Backpropagate through Bilinear Interpolation
                // This computes dL/dInput and (dL/dpy, dL/dpx) using the helper function
                // ---------------------------------------
                float grad_offset_x, grad_offset_y; // Gradients w.r.t. fractional coordinates px, py

                // Call the corrected gradient function
                bilinear_interpolate_gradient(grad_input, input, // Pass grad_input and original input
                                              in_h, in_w,
                                              py, px,            // Fractional coordinates
                                              grad_val,          // Incoming gradient (dL/dVal)
                                              &grad_offset_x,    // Output: dL/dpx
                                              &grad_offset_y);   // Output: dL/dpy

                // ---------------------------------------
                // 4. Calculate Gradient for Offsets (dL/dOffset)
                // We have dL/dpx and dL/dpy from the interpolation gradient.
                // Since px = p_x_base + offset_dx, then dpx/d(offset_dx) = 1
                // Since py = p_y_base + offset_dy, then dpy/d(offset_dy) = 1
                // By chain rule:
                // dL/d(offset_dx) = dL/dpx * dpx/d(offset_dx) = dL/dpx * 1 = grad_offset_x
                // dL/d(offset_dy) = dL/dpy * dpy/d(offset_dy) = dL/dpy * 1 = grad_offset_y
                // ---------------------------------------
                // Accumulate gradients for offsets using atomic add.
                // Although each output pixel (y,x) with kernel pos (i,j) has unique offsets,
                // the calculation is done per thread, so atomics are safest if redesigning launch params later.
                // If gridDim matches out_h * out_w strictly, non-atomic might work, but atomic is safer.
                atomicAddFloat(&grad_offset[offset_idx], grad_offset_x);     // Accumulate dL/d(offset_dx)
                atomicAddFloat(&grad_offset[offset_idx + 1], grad_offset_y); // Accumulate dL/d(offset_dy)
            }
        }
    }
}

// --- Kernels below this line were problematic/inconsistent and are now UNUSED by main funcs ---
// --- Keep them for reference or remove if desired ---

// /* Inconsistent backward pass (assumes offset per output pixel, nearest neighbor)
__global__ void deform_conv2d_backward_explained(float* grad_output, float* grad_input, float* grad_weight, float* grad_offset, float* input, float* weight, float* offset, int in_h, int in_w, int out_h, int out_w, int ksize, int stride) {
    // ... (original implementation - inconsistent with forward pass) ...
}
// */
// /* Inconsistent backward pass v2 (assumes offset per output pixel, nearest neighbor)
__global__ void deform_conv2d_backward_explained_v2(
    float* grad_output, float* grad_input, float* grad_weight, float* grad_offset, float* input, float* weight, float* offset,
    int in_h, int in_w, int out_h, int out_w, int ksize, int stride)
{
    // ... (original implementation - inconsistent with forward pass) ...
}
// */


// Single Forward/Backward Pass Example (Inference Style but calculates gradients)
int conv2d_deform_infer() {
    int in_h = 7,   // Input height
        in_w = 7,   // Input width
        ksize = 3,  // Kernel size (changed to 3 for valid output with size 5)
        stride = 1; // Stride
    // Calculate output size based on 'valid' convolution (no padding)
    int out_h = in_h - ksize + 1; // 7 - 3 + 1 = 5
    int out_w = in_w - ksize + 1; // 7 - 3 + 1 = 5

    // *** CORRECTED Assert: Check if input is large enough for kernel ***
    if (!(in_h >= ksize && in_w >= ksize)) {
         fprintf(stderr, "Assertion failed: Kernel size (%d) is too large for input dimensions (%d x %d)\n", ksize, in_h, in_w);
         return 1;
    }

    float* d_input, * d_offset, * d_weight, * d_output;
    float* d_grad_output, * d_grad_input, * d_grad_weight, * d_grad_offset;

    size_t size_in = (size_t)in_h * in_w * sizeof(float);
    size_t size_out = (size_t)out_h * out_w * sizeof(float);
    size_t size_k = (size_t)ksize * ksize * sizeof(float);
    // Offset size: (out_h, out_w, ksize, ksize, 2)
    size_t size_offset = (size_t)out_h * out_w * ksize * ksize * 2 * sizeof(float);

    // Allocate memory using cudaMalloc for explicitness (or cudaMallocManaged)
    CUDA_CHECK(cudaMalloc(&d_input, size_in));
    CUDA_CHECK(cudaMalloc(&d_offset, size_offset));
    CUDA_CHECK(cudaMalloc(&d_weight, size_k));
    CUDA_CHECK(cudaMalloc(&d_output, size_out));
    CUDA_CHECK(cudaMalloc(&d_grad_output, size_out));
    CUDA_CHECK(cudaMalloc(&d_grad_input, size_in));
    CUDA_CHECK(cudaMalloc(&d_grad_weight, size_k));
    CUDA_CHECK(cudaMalloc(&d_grad_offset, size_offset));

    // --- Initialization ---
    // Use appropriate grid/block dimensions for each allocation
    dim3 grid_in((in_w + BLOCK_SIZE - 1) / BLOCK_SIZE, (in_h + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    init_matrix<<<grid_in, block>>>(d_input, in_h, in_w);

    dim3 grid_k((ksize + BLOCK_SIZE - 1) / BLOCK_SIZE, (ksize + BLOCK_SIZE - 1) / BLOCK_SIZE);
    init_matrix<<<grid_k, block>>>(d_weight, ksize, ksize);

    // Offset initialization needs a grid based on its large size
    // Treat offset as a 1D array for initialization grid calculation
    int offset_total_elements = out_h * out_w * ksize * ksize * 2;
    dim3 grid_offset((offset_total_elements + block.x - 1) / block.x, 1); // Use 1D grid/block for simplicity
    dim3 block_1d(block.x, 1);
    // Initialize offsets to small random values (or zero if preferred start)
    init_matrix_v2<<<grid_offset, block_1d>>>(d_offset, offset_total_elements, 0.1f, 0.0f); // Small random offsets

    // Initialize grad_output (e.g., with ones or random for testing backward pass)
    dim3 grid_out((out_w + BLOCK_SIZE - 1) / BLOCK_SIZE, (out_h + BLOCK_SIZE - 1) / BLOCK_SIZE);
    init_matrix<<<grid_out, block>>>(d_grad_output, out_h, out_w, 1.0f); // Init with scale 1.0

    // Zero initialize other gradient buffers
    CUDA_CHECK(cudaMemset(d_grad_input, 0, size_in));
    CUDA_CHECK(cudaMemset(d_grad_weight, 0, size_k));
    CUDA_CHECK(cudaMemset(d_grad_offset, 0, size_offset));

    CUDA_CHECK(cudaDeviceSynchronize()); // Wait for initializations

    // --- Forward Pass ---
    deform_conv2d_forward_exaplained<<<grid_out, block>>>(d_input, d_offset, d_weight, d_output, in_h, in_w, out_h, out_w, ksize, stride);
    CUDA_CHECK(cudaDeviceSynchronize()); // Wait for forward pass

    // --- Backward Pass ---
    // *** CALL THE CORRECTED BACKWARD KERNEL ***
    deform_conv2d_backward<<<grid_out, block>>>(
        d_grad_output, d_grad_input, d_grad_weight, d_grad_offset,
        d_input, d_weight, d_offset, // Pass original input, weight, offset
        in_h, in_w, out_h, out_w, ksize, stride);
    CUDA_CHECK(cudaDeviceSynchronize()); // Wait for backward pass

    // --- Verification / Printing (Optional) ---
    // Copy results back to host to print/verify
    std::vector<float> h_output(out_h * out_w);
    std::vector<float> h_weight(ksize * ksize);
    std::vector<float> h_grad_weight(ksize * ksize);
    std::vector<float> h_grad_offset(offset_total_elements); // Get all offsets back

    CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, size_out, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_weight.data(), d_weight, size_k, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_grad_weight.data(), d_grad_weight, size_k, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_grad_offset.data(), d_grad_offset, size_offset, cudaMemcpyDeviceToHost));


    std::cout << "--- Inference Example ---" << std::endl;
    std::cout << "Weight matrix (" << ksize << "x" << ksize << "):" << std::endl;
    for (int i = 0; i < ksize; i++) {
        for (int j = 0; j < ksize; j++) {
            std::cout << h_weight[i * ksize + j] << " ";
        }
        std::cout << std::endl;
    }

    std::cout << "Output matrix (" << out_h << "x" << out_w << "):" << std::endl;
    for (int i = 0; i < out_h; i++) {
        for (int j = 0; j < out_w; j++) {
            std::cout << h_output[i * out_w + j] << " ";
        }
        std::cout << std::endl;
    }

    std::cout << "Gradient w.r.t. Weight matrix (" << ksize << "x" << ksize << "):" << std::endl;
     for (int i = 0; i < ksize; i++) {
        for (int j = 0; j < ksize; j++) {
            std::cout << h_grad_weight[i * ksize + j] << " ";
        }
        std::cout << std::endl;
    }

    // *** CORRECTED Offset Printing: Print offsets for the first output pixel (0,0) ***
    std::cout << "Offset gradients (dx, dy) for output pixel (0,0) and kernel pos (i,j):" << std::endl;
    // Offset layout: (out_y, out_x, kernel_i, kernel_j, 2)
    for (int i = 0; i < ksize; i++) { // kernel row i
         std::cout << "  Kernel row " << i << ": ";
        for (int j = 0; j < ksize; j++) { // kernel col j
            int offset_grad_base_idx = (0 * out_w + 0) * ksize * ksize * 2 + (i * ksize + j) * 2;
            if (offset_grad_base_idx + 1 < h_grad_offset.size()) {
                 std::cout << "(" << h_grad_offset[offset_grad_base_idx] << ", " << h_grad_offset[offset_grad_base_idx + 1] << ") ";
            } else {
                std::cout << "(idx out of bounds)";
            }
        }
        std::cout << std::endl;
    }


    // Cleanup
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_offset));
    CUDA_CHECK(cudaFree(d_weight));
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaFree(d_grad_output));
    CUDA_CHECK(cudaFree(d_grad_input));
    CUDA_CHECK(cudaFree(d_grad_weight));
    CUDA_CHECK(cudaFree(d_grad_offset));
    return 0;
}


// -----------------------------------------------------------------
// Kernel to update weights using gradient descent (unchanged)
__global__ void update_weights_kernel_v2(float* weight, const float* grad_weight, int filter_size, float lr) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < filter_size) {
        weight[idx] -= lr * grad_weight[idx];
    }
}


// -----------------------------------------------------------------
// Kernel to update offset parameters using gradient descent (unchanged)
__global__ void update_offsets_kernel(float* offset, const float* grad_offset, int offset_size, float lr) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < offset_size) {
        offset[idx] -= lr * grad_offset[idx];
    }
}


// -----------------------------------------------------------------
// Main Function: Training Loop
// -----------------------------------------------------------------
int conv2d_deform_training_loop() {
    std::cout << "\n--- Training Loop Example ---" << std::endl;
    // -----------------------------------------------------------------
    // Define dimensions and parameters
    // -----------------------------------------------------------------
    int in_h = 32, in_w = 32;      // Input image dimensions
    int ksize = 5;                 // Convolution kernel size (5x5)
    int stride = 1;
    // Calculate output size assuming 'valid' convolution
    int out_h = in_h - ksize + 1;
    int out_w = in_w - ksize + 1;
    if (out_h <= 0 || out_w <= 0) {
        fprintf(stderr, "Error: Kernel size %d too large for input %dx%d\n", ksize, in_h, in_w);
        return 1;
    }
    int input_size = in_h * in_w;
    int output_size = out_h * out_w;
    int filter_size = ksize * ksize;
    // Offset size: (out_h, out_w, ksize, ksize, 2)
    int offset_size = output_size * ksize * ksize * 2;

    // -----------------------------------------------------------------
    // Allocate unified memory (accessible from both CPU and GPU)
    // -----------------------------------------------------------------
    float* input, * offset, * weight, * output;
    float* grad_output, * grad_input, * grad_weight, * grad_offset;
    CUDA_CHECK(cudaMallocManaged(&input, input_size * sizeof(float)));
    CUDA_CHECK(cudaMallocManaged(&offset, offset_size * sizeof(float)));
    CUDA_CHECK(cudaMallocManaged(&weight, filter_size * sizeof(float)));
    CUDA_CHECK(cudaMallocManaged(&output, output_size * sizeof(float)));
    CUDA_CHECK(cudaMallocManaged(&grad_output, output_size * sizeof(float)));
    CUDA_CHECK(cudaMallocManaged(&grad_input, input_size * sizeof(float)));
    CUDA_CHECK(cudaMallocManaged(&grad_weight, filter_size * sizeof(float)));
    CUDA_CHECK(cudaMallocManaged(&grad_offset, offset_size * sizeof(float)));

    // -----------------------------------------------------------------
    // Initialize input, offset, and weight
    // -----------------------------------------------------------------
    int numThreads = 256; // For 1D initializers
    dim3 block1D(numThreads);
    int blocks_input = (input_size + numThreads - 1) / numThreads;
    int blocks_weight = (filter_size + numThreads - 1) / numThreads;
    int blocks_offset = (offset_size + numThreads - 1) / numThreads;

    init_matrix_v2<<<blocks_input, block1D>>>(input, input_size, 1.0f, 0.0f);   // Initialize input with random values
    // Initialize offsets near zero to start closer to regular convolution
    init_matrix_v2<<<blocks_offset, block1D>>>(offset, offset_size, 0.01f, 0.0f); // Small random offsets
    init_matrix_v2<<<blocks_weight, block1D>>>(weight, filter_size, 1.0f, 0.0f);  // Initialize weight with random values

    CUDA_CHECK(cudaDeviceSynchronize());

    // -----------------------------------------------------------------
    // Training Loop Parameters
    // -----------------------------------------------------------------
    int numEpochs = 100; // Reduced for quicker demo
    float learning_rate = 0.001f;

    // Grid dimensions for the 2D forward and backward kernels.
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((out_w + BLOCK_SIZE - 1) / BLOCK_SIZE, (out_h + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // -----------------------------------------------------------------
    // Training Loop
    // -----------------------------------------------------------------
    for (int epoch = 0; epoch < numEpochs; epoch++) {
        // Zero the gradients (much faster on GPU)
        CUDA_CHECK(cudaMemsetAsync(grad_input, 0, input_size * sizeof(float)));
        CUDA_CHECK(cudaMemsetAsync(grad_weight, 0, filter_size * sizeof(float)));
        CUDA_CHECK(cudaMemsetAsync(grad_offset, 0, offset_size * sizeof(float)));
        // Output and grad_output will be overwritten, zeroing output not strictly needed
        // but zeroing grad_output before loss calculation is good practice if done on GPU.
        // Here, grad_output is calculated entirely on CPU, so no need to zero on GPU beforehand.

        // ----------------------- Forward Pass -----------------------
        deform_conv2d_forward_exaplained<<<gridDim, blockDim>>>(input, offset, weight, output, in_h, in_w, out_h, out_w, ksize, stride);
        CUDA_CHECK(cudaDeviceSynchronize()); // Sync needed before CPU reads output

        // ----------------------- Loss Computation (CPU) ---------------
        // The target output is 0, so we use the mean squared error (MSE) loss.
        float loss = 0.0f;
        for (int i = 0; i < output_size; i++) {
            loss += output[i] * output[i];
        }
        loss /= output_size;

        // Compute gradient of MSE loss w.r.t output on CPU
        for (int i = 0; i < output_size; i++) {
             grad_output[i] = 2.0f * output[i] / output_size;
        }
        // Note: Sync is implicitly handled here as grad_output is Managed memory accessed after device sync.

        // Print loss periodically.
        if (epoch % 10 == 0 || epoch == numEpochs - 1) {
            std::cout << "Epoch " << epoch << " Loss: " << loss << std::endl;
        }

        // ----------------------- Backward Pass -----------------------
        // *** CALL THE CORRECTED BACKWARD KERNEL ***
        deform_conv2d_backward<<<gridDim, blockDim>>>(
            grad_output, grad_input, grad_weight, grad_offset,
            input, weight, offset, // Pass original input, weight, offset
            in_h, in_w, out_h, out_w, ksize, stride);
        CUDA_CHECK(cudaDeviceSynchronize()); // Sync needed before updates read gradients

        // ----------------------- Update Weights & Offsets -----------
        int blocks_update_w = (filter_size + numThreads - 1) / numThreads;
        update_weights_kernel_v2<<<blocks_update_w, block1D>>>(weight, grad_weight, filter_size, learning_rate);

        int blocks_update_o = (offset_size + numThreads - 1) / numThreads;
        update_offsets_kernel<<<blocks_update_o, block1D>>>(offset, grad_offset, offset_size, learning_rate);

        CUDA_CHECK(cudaDeviceSynchronize()); // Sync after updates before next iteration
    }

    // -----------------------------------------------------------------
    // Report Final Loss and Sample Weight Value
    // -----------------------------------------------------------------
    // Recalculate final loss after last update (optional)
     deform_conv2d_forward_exaplained<<<gridDim, blockDim>>>(input, offset, weight, output, in_h, in_w, out_h, out_w, ksize, stride);
     CUDA_CHECK(cudaDeviceSynchronize());
     float final_loss = 0.0f;
     for (int i = 0; i < output_size; i++) {
         final_loss += output[i] * output[i];
     }
     final_loss /= output_size;
     std::cout << "Final Loss after epoch " << numEpochs - 1 << ": " << final_loss << std::endl;
     std::cout << "First filter weight after training: " << weight[0] << std::endl;

    // Print the final weights
    std::cout << "Final Weights (" << ksize << "x" << ksize << "):" << std::endl;
    for (int i = 0; i < ksize; i++) {
        for (int j = 0; j < ksize; j++) {
            std::cout << weight[i * ksize + j] << " ";
        }
        std::cout << std::endl;
    }

    // -----------------------------------------------------------------
    // Free Unified Memory
    // -----------------------------------------------------------------
    CUDA_CHECK(cudaFree(input));
    CUDA_CHECK(cudaFree(offset));
    CUDA_CHECK(cudaFree(weight));
    CUDA_CHECK(cudaFree(output));
    CUDA_CHECK(cudaFree(grad_output));
    CUDA_CHECK(cudaFree(grad_input));
    CUDA_CHECK(cudaFree(grad_weight));
    CUDA_CHECK(cudaFree(grad_offset));

    return 0;
}


// Main function to run the examples
int main() {
    // Run the inference-style example
    conv2d_deform_infer();

    // Run the training loop example
    conv2d_deform_training_loop();

    return 0;
}