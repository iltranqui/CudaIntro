#include <iostream>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <device_launch_parameters.h>
#include <iomanip> // For std::setw, std::fixed, std::setprecision
#include <ctime>   // For time seeding

#define BLOCK_SIZE 16 // Thread block dimensions

// Macro for checking CUDA errors
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

// --- Paste all the previously defined CUDA kernels here ---
// init_matrix, init_matrix_v2, bilinear_interpolate,
// deform_conv2d_forward_exaplained, atomicAddFloat,
// bilinear_interpolate_gradient, deform_conv2d_backward,
// update_weights_kernel_v2, update_offsets_kernel
// --- (Kernels omitted for brevity, assume they are present as corrected before) ---

// CUDA kernel to initialize random matrices
template <typename T>
__global__ void init_matrix(T* mat, int rows, int cols, unsigned long long base_seed, float scale = 1.0f) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int index = idy * cols + idx;

    if (idx < cols && idy < rows) {
        curandState state;
        // Combine base_seed with thread/block info for unique states
        unsigned long long seed = base_seed + (unsigned long long)blockIdx.x * gridDim.y * blockDim.x * blockDim.y +
                                  (unsigned long long)blockIdx.y * blockDim.x * blockDim.y +
                                  (unsigned long long)threadIdx.y * blockDim.x + threadIdx.x;
        curand_init(seed, index, 0, &state);
        mat[index] = scale * (curand_uniform(&state) - 0.5f);
    }
}

// Simple kernel to initialize an array with random values or fixed value
__global__ void init_matrix_v2(float* mat, int size, unsigned long long base_seed, float scale, float fixed_value) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        if (fabsf(fixed_value) > 1e-9f) { // Use epsilon check for float comparison
            mat[idx] = fixed_value;
        } else {
             curandState state;
             unsigned long long seed = base_seed + idx;
             curand_init(seed, 0, 0, &state);
             mat[idx] = scale * (curand_uniform(&state) - 0.5f); // Using curand here too
        }
    }
}

// Helper function for bilinear interpolation (Forward)
__device__ float bilinear_interpolate(const float* input, int in_h, int in_w, float y, float x) {
    int x1 = floorf(x);
    int x2 = x1 + 1;
    int y1 = floorf(y);
    int y2 = y1 + 1;
    float wx1 = x2 - x;
    float wx2 = x - x1;
    float wy1 = y2 - y;
    float wy2 = y - y1;
    bool valid_x1 = (x1 >= 0 && x1 < in_w);
    bool valid_x2 = (x2 >= 0 && x2 < in_w);
    bool valid_y1 = (y1 >= 0 && y1 < in_h);
    bool valid_y2 = (y2 >= 0 && y2 < in_h);
    const float v11 = (valid_y1 && valid_x1) ? input[y1 * in_w + x1] : 0.0f;
    const float v12 = (valid_y1 && valid_x2) ? input[y1 * in_w + x2] : 0.0f;
    const float v21 = (valid_y2 && valid_x1) ? input[y2 * in_w + x1] : 0.0f;
    const float v22 = (valid_y2 && valid_x2) ? input[y2 * in_w + x2] : 0.0f;
    return wy1 * wx1 * v11 + wy1 * wx2 * v12 + wy2 * wx1 * v21 + wy2 * wx2 * v22;
}

// Forward pass kernel
/**
 * @brief Forward pass kernel for deformable convolution
 * @param input Input feature map (const float*)
 * @param offset Offset values for deformable sampling (const float*)
 * @param weight Convolution weights (const float*)
 * @param output Output feature map (float*)
 * @param in_h Input height
 * @param in_w Input width
 * @param out_h Output height
 * @param out_w Output width
 * @param ksize Kernel size
 * @param stride Convolution stride
 */
__global__ void deform_conv2d_forward_exaplained(
    const float* input, const float* offset, const float* weight, float* output, 
    int in_h, int in_w, int out_h, int out_w, int ksize, int stride)
{
    // Calculate output pixel coordinates
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < out_w && y < out_h) {
        float sum = 0.0f;
        int base_in_y = y * stride;
        int base_in_x = x * stride;
        
        // Iterate over the kernel
        for (int i = 0; i < ksize; i++) {
            for (int j = 0; j < ksize; j++) {
                // Calculate offset index
                int offset_idx = ((y * out_w + x) * ksize * ksize + i * ksize + j) * 2;
                
                // Calculate sampling position
                float p_y_base = base_in_y + i;
                float p_x_base = base_in_x + j;
                float offset_dx = offset[offset_idx];
                float offset_dy = offset[offset_idx + 1];
                float px = p_x_base + offset_dx;
                float py = p_y_base + offset_dy;
                
                // Sample input using bilinear interpolation
                float val = bilinear_interpolate(input, in_h, in_w, py, px);
                
                // Multiply sampled value with weight and accumulate
                sum += val * weight[i * ksize + j];
            }
        }
        
        // Store the result in the output
        output[y * out_w + x] = sum;
    }
}

// atomicAddFloat wrapper
__device__ float atomicAddFloat(float* address, float val) {
    return atomicAdd(address, val);
}

// CORRECTED Bilinear Interpolation Gradient Helper
__device__ void bilinear_interpolate_gradient(
    float* grad_input, const float* input, int in_h, int in_w,
    float y, float x, float gradient, float* grad_x, float* grad_y)
{
    int x1 = floorf(x);
    int x2 = x1 + 1;
    int y1 = floorf(y);
    int y2 = y1 + 1;
    float wx1 = x2 - x;
    float wx2 = x - x1;
    float wy1 = y2 - y;
    float wy2 = y - y1;
    bool valid_x1 = (x1 >= 0 && x1 < in_w);
    bool valid_x2 = (x2 >= 0 && x2 < in_w);
    bool valid_y1 = (y1 >= 0 && y1 < in_h);
    bool valid_y2 = (y2 >= 0 && y2 < in_h);

    // Accumulate dL/dInput
    if (valid_y1 && valid_x1) atomicAddFloat(&grad_input[y1 * in_w + x1], gradient * wy1 * wx1);
    if (valid_y1 && valid_x2) atomicAddFloat(&grad_input[y1 * in_w + x2], gradient * wy1 * wx2);
    if (valid_y2 && valid_x1) atomicAddFloat(&grad_input[y2 * in_w + x1], gradient * wy2 * wx1);
    if (valid_y2 && valid_x2) atomicAddFloat(&grad_input[y2 * in_w + x2], gradient * wy2 * wx2);

    // Compute dL/dx, dL/dy
    const float v11 = (valid_y1 && valid_x1) ? input[y1 * in_w + x1] : 0.0f;
    const float v12 = (valid_y1 && valid_x2) ? input[y1 * in_w + x2] : 0.0f;
    const float v21 = (valid_y2 && valid_x1) ? input[y2 * in_w + x1] : 0.0f;
    const float v22 = (valid_y2 && valid_x2) ? input[y2 * in_w + x2] : 0.0f;
    *grad_x = gradient * (wy1 * (v12 - v11) + wy2 * (v22 - v21));
    *grad_y = gradient * (wx1 * (v21 - v11) + wx2 * (v22 - v12));
}

// CORRECTED Backward Pass Kernel
__global__ void deform_conv2d_backward(
    const float* grad_output, float* grad_input, float* grad_weight, float* grad_offset,
    const float* input, const float* weight, const float* offset,
    int in_h, int in_w, int out_h, int out_w, int ksize, int stride)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < out_w && y < out_h) {
        float grad_out_yx = grad_output[y * out_w + x];
        int base_in_y = y * stride;
        int base_in_x = x * stride;

        for (int i = 0; i < ksize; i++) {
            for (int j = 0; j < ksize; j++) {
                int offset_idx = ((y * out_w + x) * ksize * ksize + i * ksize + j) * 2;
                float p_y_base = base_in_y + i;
                float p_x_base = base_in_x + j;
                float offset_dx = offset[offset_idx];
                float offset_dy = offset[offset_idx + 1];
                float px = p_x_base + offset_dx;
                float py = p_y_base + offset_dy;
                float val_interpolated = bilinear_interpolate(input, in_h, in_w, py, px);

                // 1. Grad Weight
                atomicAddFloat(&grad_weight[i * ksize + j], grad_out_yx * val_interpolated);

                // 2. Grad flowing back to interp value stage
                float grad_val = grad_out_yx * weight[i * ksize + j];

                // 3. Backprop through Bilinear Interp
                float grad_offset_x, grad_offset_y;
                bilinear_interpolate_gradient(grad_input, input, in_h, in_w, py, px,
                                              grad_val, &grad_offset_x, &grad_offset_y);

                // 4. Grad Offset
                atomicAddFloat(&grad_offset[offset_idx], grad_offset_x);
                atomicAddFloat(&grad_offset[offset_idx + 1], grad_offset_y);
            }
        }
    }
}

// Update Weights Kernel
__global__ void update_weights_kernel_v2(float* weight, const float* grad_weight, int filter_size, float lr) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < filter_size) {
        weight[idx] -= lr * grad_weight[idx];
    }
}

// Update Offsets Kernel
__global__ void update_offsets_kernel(float* offset, const float* grad_offset, int offset_size, float lr) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < offset_size) {
        offset[idx] -= lr * grad_offset[idx];
    }
}


// -----------------------------------------------------------------
// Helper function to print a 2D matrix (CPU side)
// -----------------------------------------------------------------
void print_matrix_cpu(const float* matrix, int rows, int cols, const std::string& title) {
    std::cout << title << " (" << rows << "x" << cols << "):" << std::endl;
    if (!matrix) {
        std::cout << "  <Null Pointer>" << std::endl;
        return;
    }
    std::cout << std::fixed << std::setprecision(4);
    for (int i = 0; i < rows; ++i) {
        std::cout << "  ";
        for (int j = 0; j < cols; ++j) {
            // Ensure index is within bounds for safety, although should be correct if called properly
            if (i * cols + j < rows * cols) {
                std::cout << std::setw(9) << matrix[i * cols + j] << " ";
            } else {
                 std::cout << std::setw(9) << " OOB " << " "; // Out Of Bounds
            }
        }
        std::cout << std::endl;
    }
    std::cout << std::resetiosflags(std::ios_base::fixed | std::ios_base::floatfield); // Reset precision format
}


// -----------------------------------------------------------------
// Helper function to print offsets or their gradients for a specific output pixel (CPU side)
// -----------------------------------------------------------------
void print_offsets_for_pixel(const float* offset_tensor, int out_h, int out_w, int ksize, int pixel_y, int pixel_x, const std::string& title) {
    std::cout << title << " for output pixel (" << pixel_y << "," << pixel_x << "):" << std::endl;
     if (!offset_tensor) {
        std::cout << "  <Null Pointer>" << std::endl;
        return;
    }
    // Check pixel bounds
    if(pixel_y < 0 || pixel_y >= out_h || pixel_x < 0 || pixel_x >= out_w) {
        std::cout << "  <Pixel index out of bounds>" << std::endl;
        return;
    }

    std::cout << std::fixed << std::setprecision(4);
    int offset_base_idx = (pixel_y * out_w + pixel_x) * ksize * ksize * 2;
    int total_offset_elements = out_h * out_w * ksize * ksize * 2; // Calculate total size for bound check

    for (int i = 0; i < ksize; ++i) { // Kernel row i
        std::cout << "  Kernel row " << i << ": ";
        for (int j = 0; j < ksize; ++j) { // Kernel col j
            int current_offset_idx = offset_base_idx + (i * ksize + j) * 2;
            // Bounds check before accessing memory
            if (current_offset_idx + 1 < total_offset_elements) {
                float dx = offset_tensor[current_offset_idx];
                float dy = offset_tensor[current_offset_idx + 1];
                std::cout << "(" << std::setw(8) << dx << "," << std::setw(8) << dy << ") ";
            } else {
                 std::cout << "( OOB ) "; // Index Out Of Bounds
            }
        }
        std::cout << std::endl;
    }
     std::cout << std::resetiosflags(std::ios_base::fixed | std::ios_base::floatfield);
}


// -----------------------------------------------------------------
// Function to run the 2-epoch demonstration
// -----------------------------------------------------------------
int run_two_epoch_demo() {
    std::cout << "\n--- Running 2-Epoch Deformable Conv Demo ---" << std::endl;

    // 1. Define Dimensions
    int in_h = 7, in_w = 7;
    int ksize = 3;
    int stride = 1;
    int out_h = in_h - ksize + 1; // 7 - 3 + 1 = 5
    int out_w = in_w - ksize + 1; // 7 - 3 + 1 = 5

    // 2. Calculate Sizes
    int input_size = in_h * in_w;
    int output_size = out_h * out_w;
    int filter_size = ksize * ksize;
    int offset_size = output_size * ksize * ksize * 2; // (5*5*3*3*2 = 450)

    // 3. Allocate Managed Memory (simplifies printing)
    float* input, * weight, * offset, * output;
    float* grad_input, * grad_weight, * grad_offset, * grad_output;

    std::cout << "Allocating memory..." << std::endl;
    CUDA_CHECK(cudaMallocManaged(&input, input_size * sizeof(float)));
    CUDA_CHECK(cudaMallocManaged(&weight, filter_size * sizeof(float)));
    CUDA_CHECK(cudaMallocManaged(&offset, offset_size * sizeof(float)));
    CUDA_CHECK(cudaMallocManaged(&output, output_size * sizeof(float)));
    CUDA_CHECK(cudaMallocManaged(&grad_input, input_size * sizeof(float)));
    CUDA_CHECK(cudaMallocManaged(&grad_weight, filter_size * sizeof(float)));
    CUDA_CHECK(cudaMallocManaged(&grad_offset, offset_size * sizeof(float)));
    CUDA_CHECK(cudaMallocManaged(&grad_output, output_size * sizeof(float)));
    std::cout << "Memory allocated." << std::endl;

    // Get current time for seeding randomness
    unsigned long long seed = time(0);

    // 4. Initialize Input (once, random)
    dim3 grid_in((in_w + BLOCK_SIZE - 1) / BLOCK_SIZE, (in_h + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 block2D(BLOCK_SIZE, BLOCK_SIZE);
    init_matrix<<<grid_in, block2D>>>(input, in_h, in_w, seed++);
    CUDA_CHECK(cudaDeviceSynchronize()); // Ensure init is done
    print_matrix_cpu(input, in_h, in_w, "Initial Input Data"); // Print input once

    // 5. Initialize Weight (once, random)
    dim3 grid_k((ksize + BLOCK_SIZE - 1) / BLOCK_SIZE, (ksize + BLOCK_SIZE - 1) / BLOCK_SIZE);
    init_matrix<<<grid_k, block2D>>>(weight, ksize, ksize, seed++);

    // 6. Initialize Offset (once, near zero)
    int numThreads1D = 256;
    dim3 block1D(numThreads1D);
    dim3 grid_offset((offset_size + numThreads1D - 1) / numThreads1D);
    init_matrix_v2<<<grid_offset, block1D>>>(offset, offset_size, seed++, 0.01f, 0.0f); // scale 0.01

    // 7. Define learning rate
    float learning_rate = 0.01f; // A small LR for demonstration

    // Ensure initializations are complete before the loop
    CUDA_CHECK(cudaDeviceSynchronize());

    // --- Epoch Loop ---
    int numEpochs = 2;
    for (int epoch = 0; epoch < numEpochs; ++epoch) {
        std::cout << "\n=============== EPOCH " << epoch << " ================" << std::endl;

        // Print current state BEFORE forward/backward
        print_matrix_cpu(weight, ksize, ksize, "Current Weights (Start of Epoch)");
        // Print offsets for the central output pixel (e.g., 2,2 for 5x5 output)
        print_offsets_for_pixel(offset, out_h, out_w, ksize, out_h / 2, out_w / 2, "Current Offsets (Start of Epoch)");

        // Zero Gradients on GPU
        CUDA_CHECK(cudaMemsetAsync(grad_input, 0, input_size * sizeof(float)));
        CUDA_CHECK(cudaMemsetAsync(grad_weight, 0, filter_size * sizeof(float)));
        CUDA_CHECK(cudaMemsetAsync(grad_offset, 0, offset_size * sizeof(float)));

        // Define dummy grad_output (e.g., all ones)
        dim3 grid_out_1d((output_size + numThreads1D - 1) / numThreads1D);
        init_matrix_v2<<<grid_out_1d, block1D>>>(grad_output, output_size, seed++, 0.0f, 1.0f); // Fixed value 1.0

        // Ensure gradient zeroing and grad_output init are done
        CUDA_CHECK(cudaDeviceSynchronize());

        // Launch Forward Kernel
        dim3 grid_out_2d((out_w + BLOCK_SIZE - 1) / BLOCK_SIZE, (out_h + BLOCK_SIZE - 1) / BLOCK_SIZE);
        deform_conv2d_forward_exaplained<<<grid_out_2d, block2D>>>(
            input, offset, weight, output,
            in_h, in_w, out_h, out_w, ksize, stride);

        // Launch Backward Kernel
        deform_conv2d_backward<<<grid_out_2d, block2D>>>(
            grad_output, grad_input, grad_weight, grad_offset,
            input, weight, offset,
            in_h, in_w, out_h, out_w, ksize, stride);

        // Synchronize to ensure kernels are finished before printing/updating
        CUDA_CHECK(cudaDeviceSynchronize());

        // --- Print Gradients and the Offsets that generated them ---
        print_matrix_cpu(grad_weight, ksize, ksize, "Weight Gradients (dL/dW)");

        // *** ADDED: Print current offsets *after* backward, *before* update ***
        // This shows the offset values that were used in the just-completed pass
        print_offsets_for_pixel(offset, out_h, out_w, ksize, out_h / 2, out_w / 2, "Current Offsets (Used for Gradients)");

        // Print Offset Gradients for center pixel (2,2)
        print_offsets_for_pixel(grad_offset, out_h, out_w, ksize, out_h / 2, out_w / 2, "Offset Gradients (dL/dOffset)");


        // Update Weights & Offsets
        std::cout << "Updating weights and offsets..." << std::endl;
        dim3 grid_update_w((filter_size + numThreads1D - 1) / numThreads1D);
        update_weights_kernel_v2<<<grid_update_w, block1D>>>(weight, grad_weight, filter_size, learning_rate);

        dim3 grid_update_o((offset_size + numThreads1D - 1) / numThreads1D);
        update_offsets_kernel<<<grid_update_o, block1D>>>(offset, grad_offset, offset_size, learning_rate);

        // Synchronize after updates
        CUDA_CHECK(cudaDeviceSynchronize());

        std::cout << "=============== END OF EPOCH " << epoch << " ================" << std::endl;

    } // End epoch loop

    // Free Memory
    std::cout << "\nFreeing memory..." << std::endl;
    CUDA_CHECK(cudaFree(input));
    CUDA_CHECK(cudaFree(weight));
    CUDA_CHECK(cudaFree(offset));
    CUDA_CHECK(cudaFree(output));
    CUDA_CHECK(cudaFree(grad_input));
    CUDA_CHECK(cudaFree(grad_weight));
    CUDA_CHECK(cudaFree(grad_offset));
    CUDA_CHECK(cudaFree(grad_output));
    std::cout << "Memory freed." << std::endl;

    return 0;
}

// Main function
int main() {
    // Set device if needed, e.g., cudaSetDevice(0);
    run_two_epoch_demo();
    return 0;
}