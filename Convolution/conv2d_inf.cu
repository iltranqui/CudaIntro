// conv2d_infer.cu with inference
#include <iostream>
#include <cstdlib>
#include <cstddef> // For size_t
#include <ctime>
#include <cuda_runtime.h>              // necessary to use cudaMalloc, cudaMemcpy, etc. and CUDA runtime APIs
#include <device_launch_parameters.h>  // necessary to use threadIdx, blockIdx, blockDim, gridDim
#include <chrono>

// Define the image dimensions and kernel dimensions
#define IMAGE_WIDTH 512
#define IMAGE_HEIGHT 512
#define KERNEL_WIDTH 3
#define KERNEL_HEIGHT 3

// The convolution kernel is stored in constant memory for efficiency. -> constant memory is stored
__constant__ float d_kernel[KERNEL_WIDTH * KERNEL_HEIGHT];

// CUDA kernel for performing 2D convolution on the input matrix.
__global__ void conv2D(const float* input, float* output, int width, int height) {
    // Compute the (x,y) coordinate of the element this thread will process.
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Compute the radius (offset) of the kernel.
    int kCenterX = KERNEL_WIDTH / 2;
    int kCenterY = KERNEL_HEIGHT / 2;

    if (x < width && y < height) {
        float sum = 0.0f;
        // Iterate over the kernel window.
        for (int m = 0; m < KERNEL_HEIGHT; m++) {
            for (int n = 0; n < KERNEL_WIDTH; n++) {
                // Compute the corresponding input pixel location.
                int row = y + m - kCenterY;
                int col = x + n - kCenterX;
                // Check for boundary conditions. Here, we use zero-padding.
                if (row >= 0 && row < height && col >= 0 && col < width) {
                    sum += input[row * width + col] * d_kernel[m * KERNEL_WIDTH + n];
                }
            }
        }
        output[y * width + x] = sum;
    }
}

int conv2d_inf() {
    // Set seed for random number generation
    std::srand(static_cast<unsigned int>(std::time(0)));

    // Calculate the total size of the image.
    const int imageSize = IMAGE_WIDTH * IMAGE_HEIGHT;
    const int imageBytes = imageSize * sizeof(float);

    // Allocate host memory for input and output matrices.
    float* h_input = new float[imageSize];
    float* h_output = new float[imageSize];

    // Fill the input matrix with random values.
    for (int i = 0; i < imageSize; i++) {
        h_input[i] = static_cast<float>(std::rand() % 256); // Values in [0, 255]
    }

    // Define a 3x3 averaging kernel (blur filter).
    float h_kernel[KERNEL_WIDTH * KERNEL_HEIGHT] = {
        1.0f / 9, 1.0f / 9, 1.0f / 9,
        1.0f / 9, 1.0f / 9, 1.0f / 9,
        1.0f / 9, 1.0f / 9, 1.0f / 9
    };

    // Copy the kernel to the device constant memory.
    cudaMemcpyToSymbol(d_kernel, h_kernel, KERNEL_WIDTH * KERNEL_HEIGHT * sizeof(float));

    // Allocate device memory for the input and output.
    float* d_input, * d_output;
    cudaMalloc((void**)&d_input, imageBytes);
    cudaMalloc((void**)&d_output, imageBytes);

    // Copy the input matrix from the host to the device.
    cudaMemcpy(d_input, h_input, imageBytes, cudaMemcpyHostToDevice);

    // Define the block and grid dimensions.
    dim3 blockDim(16, 16);
    dim3 gridDim((IMAGE_WIDTH + blockDim.x - 1) / blockDim.x, (IMAGE_HEIGHT + blockDim.y - 1) / blockDim.y);

    // Launch the convolution kernel.
    conv2D << <gridDim, blockDim >> > (d_input, d_output, IMAGE_WIDTH, IMAGE_HEIGHT);
    cudaDeviceSynchronize();

    // Copy the output matrix back to the host.
    cudaMemcpy(h_output, d_output, imageBytes, cudaMemcpyDeviceToHost);

    // Print a small portion of the output for verification.
    std::cout << "Sample output values:" << std::endl;
    for (int y = 250; y < 255; y++) {
        for (int x = 250; x < 255; x++) {
            std::cout << h_output[y * IMAGE_WIDTH + x] << "\t";
        }
        std::cout << std::endl;
    }

    // Clean up host and device memory.
    delete[] h_input;
    delete[] h_output;
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}

/*
__global__ void conv2d_kernel(const float* A, const float* B, float* C, size_t H, size_t W, size_t Kh, size_t Kw);
__global__ void conv2d_kernel_shared(const float* A, const float* B, float* C, size_t H, size_t W, size_t Kh, size_t Kw);

extern "C" void solution_2d_gpu(const float* A, const float* B, float* C, size_t H, size_t W, size_t Kh, size_t Kw) {    // Define block size (tune this for your device)
    dim3 blockDim(16, 16);
    // Compute grid size ensuring we cover the entire image
    dim3 gridDim((W + blockDim.x - 1) / blockDim.x, (H + blockDim.y - 1) / blockDim.y);
    // Launch the kernel
    conv2d_kernel << <gridDim, blockDim >> > (A, B, C, H, W, Kh, Kw);
    // Synchronize to ensure completion before returning
    cudaDeviceSynchronize();
}

extern "C" void solution_2d_gpu_shared(const float* A, const float* B, float* C, size_t H, size_t W, size_t Kh, size_t Kw) {    // Define block size (tune this for your device)
    dim3 blockDim(16, 16);
    // Compute grid size ensuring we cover the entire image
    dim3 gridDim((W + blockDim.x - 1) / blockDim.x, (H + blockDim.y - 1) / blockDim.y);
    // Launch the kernel
    int sharedMemSize = (blockDim.x + 2 * pad_w) * (blockDim.y + 2 * pad_h) * sizeof(float);
    conv2d_kernel_shared << <gridDim, blockDim >> > (A, B, C, H, W, Kh, Kw);
    // Synchronize to ensure completion before returning
    cudaDeviceSynchronize();
}

// Tensara loop with shared memory
__global__ void conv2d_kernel_shared(const float* A, const float* B, float* C, size_t H, size_t W, size_t Kh, size_t Kw) {
    // Define block size and shared memory tile size
    const int TILE_WIDTH = blockDim.x;
    const int TILE_HEIGHT = blockDim.y;

    // Compute the (i, j) coordinate of the output pixel
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    // Compute padding offsets
    int pad_h = (Kh - 1) / 2;
    int pad_w = (Kw - 1) / 2;

    // Define shared memory for the tile
	extern __shared__ float shared_A[];  // Size is [TILE_HEIGHT + 2 * pad_h][TILE_WIDTH + 2 * pad_w]

    // Compute shared memory dimensions (including padding)
    int shared_W = TILE_WIDTH + 2 * pad_w;
    int shared_H = TILE_HEIGHT + 2 * pad_h;

    // Compute thread’s position in shared memory
    int shared_i = threadIdx.y + pad_h;
    int shared_j = threadIdx.x + pad_w;

    // Compute global memory coordinates
    int global_i = i - pad_h;
    int global_j = j - pad_w;

    // Load data into shared memory (handle boundary conditions)
    if (global_i >= 0 && global_i < H && global_j >= 0 && global_j < W) {
        shared_A[shared_i * shared_W + shared_j] = A[global_i * W + global_j];
    }
    else {
        shared_A[shared_i * shared_W + shared_j] = 0.0f;  // Zero-padding
    }

    __syncthreads();  // Ensure all threads load data before computation

    // Compute convolution only for valid output pixels
    if (i < H && j < W) {
        float sum = 0.0f;
        for (int k = 0; k < Kh; ++k) {
            for (int l = 0; l < Kw; ++l) {
                int row = shared_i + k - pad_h;
                int col = shared_j + l - pad_w;
                sum += shared_A[row * shared_W + col] * B[k * Kw + l];
            }
        }
        C[i * W + j] = sum;
    }
}

// Tensara loop
__global__ void conv2d_kernel(const float* A, const float* B, float* C, size_t H, size_t W, size_t Kh, size_t Kw) {
    // Compute the (i,j) coordinate of the output pixel
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < H && j < W) {  // Ensure we are within the image bounds
        float sum = 0.0f;
        // Compute the padding offsets
        int pad_h = (Kh - 1) / 2;
        int pad_w = (Kw - 1) / 2;

        // Loop over the kernel dimensions
        for (int k = 0; k < Kh; ++k) {
            for (int l = 0; l < Kw; ++l) {
                // Calculate corresponding image indices (center the kernel)
                int row = i + k - pad_h;
                int col = j + l - pad_w;
                // Use zero padding if outside the valid region
                if (row >= 0 && row < H && col >= 0 && col < W) {
                    sum += A[row * W + col] * B[k * Kw + l];
                }
            }
        }
        // Write the result into the output image
        C[i * W + j] = sum;
    }
}

extern "C" void solution_2d_cpu(const float* A, const float* B, float* C,
    size_t H, size_t W, size_t Kh, size_t Kw) {
    // Calculate the amount of padding needed on each side
    int pad_h = (Kh - 1) / 2;
    int pad_w = (Kw - 1) / 2;

    // Loop over every output pixel position in the image
    for (size_t i = 0; i < H; ++i) {
        for (size_t j = 0; j < W; ++j) {
            float sum = 0.0f;
            // Loop over the kernel dimensions
            for (size_t k = 0; k < Kh; ++k) {
                for (size_t l = 0; l < Kw; ++l) {
                    // Calculate corresponding input image indices, centering the kernel
                    int row = i + k - pad_h;
                    int col = j + l - pad_w;
                    // Check for valid indices; if invalid, treat as zero (zero padding)
                    if (row >= 0 && row < static_cast<int>(H) &&
                        col >= 0 && col < static_cast<int>(W)) {
                        sum += A[row * W + col] * B[k * Kw + l];
                    }
                }
            }
            // Write the computed sum to the output image
            C[i * W + j] = sum;
        }
    }
}

// Forward declarations of CPU and GPU functions
extern "C" void solution_2d_cpu(const float* A, const float* B, float* C, size_t H, size_t W, size_t Kh, size_t Kw);
extern "C" void solution_2d_gpu(const float* A, const float* B, float* C, size_t H, size_t W, size_t Kh, size_t Kw);


void benchmark_convolution2d(size_t H, size_t W, size_t Kh, size_t Kw) {
    // ======================
	// Benchmarking Code 2D Convolution
	// ======================
	std::cout << "Benchmarking 2D Convolution" << std::endl;
    size_t image_size = H * W * sizeof(float);
    size_t kernel_size = Kh * Kw * sizeof(float);

    // Allocate memory for input, kernel, and output on CPU
    float* h_A = new float[H * W];
    float* h_B = new float[Kh * Kw];
    float* h_C_cpu = new float[H * W];
    float* h_C_gpu = new float[H * W];

    // Initialize input image and kernel with random values
    for (size_t i = 0; i < H * W; ++i) h_A[i] = static_cast<float>(rand()) / RAND_MAX;
    for (size_t i = 0; i < Kh * Kw; ++i) h_B[i] = static_cast<float>(rand()) / RAND_MAX;

    // Measure CPU execution time
    auto start_cpu = std::chrono::high_resolution_clock::now();
    solution_2d_cpu(h_A, h_B, h_C_cpu, H, W, Kh, Kw);
    auto end_cpu = std::chrono::high_resolution_clock::now();
    double cpu_time = std::chrono::duration<double, std::milli>(end_cpu - start_cpu).count();

    // Allocate device memory for GPU
    float* d_A, * d_B, * d_C;
    cudaMalloc((void**)&d_A, image_size);
    cudaMalloc((void**)&d_B, kernel_size);
    cudaMalloc((void**)&d_C, image_size);

    // Copy data from host to device
    cudaMemcpy(d_A, h_A, image_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, kernel_size, cudaMemcpyHostToDevice);

    // Measure GPU execution time using CUDA events
    cudaEvent_t start_gpu, stop_gpu;
    cudaEventCreate(&start_gpu);
    cudaEventCreate(&stop_gpu);

    cudaEventRecord(start_gpu);
    solution_2d_gpu(d_A, d_B, d_C, H, W, Kh, Kw);
    cudaEventRecord(stop_gpu);
    cudaEventSynchronize(stop_gpu);

    float gpu_time;
    cudaEventElapsedTime(&gpu_time, start_gpu, stop_gpu);

    // Copy result back to host
    cudaMemcpy(h_C_gpu, d_C, image_size, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Destroy CUDA events
    cudaEventDestroy(start_gpu);
    cudaEventDestroy(stop_gpu);

    // Print timing results
    std::cout << "CPU Time: " << cpu_time << " ms" << std::endl;
    std::cout << "GPU Time: " << gpu_time << " ms" << std::endl;

    // Compare results for correctness
    double error = 0.0;
    for (size_t i = 0; i < H * W; ++i) {
        error += std::abs(h_C_cpu[i] - h_C_gpu[i]);
    }
    std::cout << "Total Error: " << error << std::endl;

    // Free host memory
    delete[] h_A;
    delete[] h_B;
    delete[] h_C_cpu;
    delete[] h_C_gpu;
}
*/