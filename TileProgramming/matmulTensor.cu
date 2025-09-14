#define CCCL_IGNORE_DEPRECATED_CPP_DIALECT
#include "tensor_utils.h"
#include <cuda_device_runtime_api.h>
#include <device_launch_parameters.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h> // For bfloat16
#include <mma.h>
#include <cmath>
#include <stdio.h>

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16
#define MATRIX_SIZE 1024

// Confronto tra risultati GPU e CPU (con tolleranza, default 1e-5f)
inline bool checkResult(const float* GPU, const float* CPU, int width, float eps = 1e-5f) {
    int size = width * width;
    for (int i = 0; i < size; ++i) {
        if (std::fabs(static_cast<double>(GPU[i]) - static_cast<double>(CPU[i])) > static_cast<double>(eps)) {
            printf("Mismatch at %d: GPU=%f, CPU=%f\n", i, GPU[i], CPU[i]);
            return false;
        }
    }
    return true;
}

// Tensor Core kernel
__global__ void matrixMulTensorCores(half* A, half* B, float* C, int width) {
    using namespace nvcuda::wmma;
    fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, half, row_major> a_frag;
    fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, half, col_major> b_frag;
    fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
    fill_fragment(c_frag, 0.0f);

    int tile_row = blockIdx.y * WMMA_M;
    int tile_col = blockIdx.x * WMMA_N;

    for (int k = 0; k < width; k += WMMA_K) {
        load_matrix_sync(a_frag, A + tile_row * width + k, width);
        load_matrix_sync(b_frag, B + k * width + tile_col, width);
        mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    store_matrix_sync(C + tile_row * width + tile_col, c_frag, width, mem_row_major);
}

// bfloat16 Tensor Core kernel
__global__ void matrixMulTensorCoresBfloat16(__nv_bfloat16* A, __nv_bfloat16* B, float* C, int width) {
    using namespace nvcuda::wmma;
    fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, row_major> a_frag;
    fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, col_major> b_frag;
    fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
    fill_fragment(c_frag, 0.0f);

    int tile_row = blockIdx.y * WMMA_M;
    int tile_col = blockIdx.x * WMMA_N;

    for (int k = 0; k < width; k += WMMA_K) {
        load_matrix_sync(a_frag, A + tile_row * width + k, width);
        load_matrix_sync(b_frag, B + k * width + tile_col, width);
        mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    store_matrix_sync(C + tile_row * width + tile_col, c_frag, width, mem_row_major);
}

// Host wrappers with default launch parameters
inline void launchMatrixMulTensorCores(half* A, half* B, float* C,
    int width = MATRIX_SIZE,
    dim3 grid = dim3(MATRIX_SIZE / WMMA_M, MATRIX_SIZE / WMMA_N),
    dim3 block = dim3(WMMA_M, WMMA_N),
    cudaStream_t stream = 0)
{
    matrixMulTensorCores<<<grid, block, 0, stream>>>(A, B, C, width);
}

inline void launchMatrixMulTensorCoresBfloat16(__nv_bfloat16* A, __nv_bfloat16* B, float* C,
    int width = MATRIX_SIZE,
    dim3 grid = dim3(MATRIX_SIZE / WMMA_M, MATRIX_SIZE / WMMA_N),
    dim3 block = dim3(WMMA_M, WMMA_N),
    cudaStream_t stream = 0)
{
    matrixMulTensorCoresBfloat16<<<grid, block, 0, stream>>>(A, B, C, width);
}

#define TILE_WIDTH 32
// default grid/block helper for CUDA core kernel
inline dim3 defaultCudaGrid(int width) {
    return dim3((width + TILE_WIDTH - 1) / TILE_WIDTH, (width + TILE_WIDTH - 1) / TILE_WIDTH);
}
inline dim3 defaultCudaBlock() {
    return dim3(TILE_WIDTH, TILE_WIDTH);
}

// CUDA Core kernel with shared memory tiling
__global__ void matrixMulCudaCores(float* C, float* A, float* B, int width) {
    __shared__ float sA[TILE_WIDTH][TILE_WIDTH];
    __shared__ float sB[TILE_WIDTH][TILE_WIDTH];

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;

    for (int t = 0; t < (width + TILE_WIDTH - 1) / TILE_WIDTH; ++t) {
        if (row < width && t * TILE_WIDTH + threadIdx.x < width)
            sA[threadIdx.y][threadIdx.x] = A[row * width + t * TILE_WIDTH + threadIdx.x];
        else
            sA[threadIdx.y][threadIdx.x] = 0.0f;
        if (col < width && t * TILE_WIDTH + threadIdx.y < width)
            sB[threadIdx.y][threadIdx.x] = B[(t * TILE_WIDTH + threadIdx.y) * width + col];
        else
            sB[threadIdx.y][threadIdx.x] = 0.0f;
        __syncthreads();

        for (int k = 0; k < TILE_WIDTH; ++k)
            sum += sA[threadIdx.y][k] * sB[k][threadIdx.x];
        __syncthreads();
    }

    if (row < width && col < width)
        C[row * width + col] = sum;
}

// Host wrapper for CUDA-core kernel with defaults
inline void launchMatrixMulCudaCores(float* C, float* A, float* B,
    int width = MATRIX_SIZE,
    dim3 grid = defaultCudaGrid(MATRIX_SIZE),
    dim3 block = defaultCudaBlock(),
    cudaStream_t stream = 0)
{
    matrixMulCudaCores<<<grid, block, 0, stream>>>(C, A, B, width);
}

// --- thrust / host-side test functions follow ---

// bfloat16 Tensor Core function
int thrustTensorBfloat16() {
    thrust::host_vector<float> h_A(MATRIX_SIZE * MATRIX_SIZE);
    thrust::host_vector<float> h_B(MATRIX_SIZE * MATRIX_SIZE);
    thrust::host_vector<__nv_bfloat16> h_A_bf16(MATRIX_SIZE * MATRIX_SIZE);
    thrust::host_vector<__nv_bfloat16> h_B_bf16(MATRIX_SIZE * MATRIX_SIZE);
    thrust::host_vector<float> h_C(MATRIX_SIZE * MATRIX_SIZE);
    thrust::host_vector<float> h_C_cpu(MATRIX_SIZE * MATRIX_SIZE);

    srand(1234);
    randomInit(h_A.data(), h_A_bf16.data(), MATRIX_SIZE * MATRIX_SIZE);
    randomInit(h_B.data(), h_B_bf16.data(), MATRIX_SIZE * MATRIX_SIZE);

    thrust::device_vector<__nv_bfloat16> d_A = h_A_bf16;
    thrust::device_vector<__nv_bfloat16> d_B = h_B_bf16;
    thrust::device_vector<float> d_C(MATRIX_SIZE * MATRIX_SIZE);

    dim3 threadsPerBlock(WMMA_M, WMMA_N);
    dim3 numBlocks(MATRIX_SIZE / WMMA_M, MATRIX_SIZE / WMMA_N);
    // use wrapper with defaults (equivalent to previous explicit launch)
    launchMatrixMulTensorCoresBfloat16(d_A.data().get(), d_B.data().get(), d_C.data().get(), MATRIX_SIZE);

    h_C = d_C;
    matrixMulCPU(h_C_cpu.data(), h_A.data(), h_B.data(), MATRIX_SIZE);
    if (checkResult(h_C.data(), h_C_cpu.data(), MATRIX_SIZE, 2e-3f))
        printf("bfloat16 Tensor Cores: Matrix multiplication successful!\n");
    else
        printf("bfloat16 Tensor Cores: Results do not match!\n");

    return 0;
}


int thrustensor() {
    // Thrust for input initialization
    thrust::host_vector<float> h_A(MATRIX_SIZE * MATRIX_SIZE);
    thrust::host_vector<float> h_B(MATRIX_SIZE * MATRIX_SIZE);
    thrust::host_vector<half> h_A_half(MATRIX_SIZE * MATRIX_SIZE);
    thrust::host_vector<half> h_B_half(MATRIX_SIZE * MATRIX_SIZE);
    thrust::host_vector<float> h_C(MATRIX_SIZE * MATRIX_SIZE);
    thrust::host_vector<float> h_C_cpu(MATRIX_SIZE * MATRIX_SIZE);

    srand(1234); // Reproducible
    randomInit(h_A.data(), h_A_half.data(), MATRIX_SIZE * MATRIX_SIZE);
    randomInit(h_B.data(), h_B_half.data(), MATRIX_SIZE * MATRIX_SIZE);

    thrust::device_vector<half> d_A = h_A_half;
    thrust::device_vector<half> d_B = h_B_half;
    thrust::device_vector<float> d_C(MATRIX_SIZE * MATRIX_SIZE);

    // use wrapper with defaults
    launchMatrixMulTensorCores(d_A.data().get(), d_B.data().get(), d_C.data().get(), MATRIX_SIZE);

    h_C = d_C;
    matrixMulCPU(h_C_cpu.data(), h_A.data(), h_B.data(), MATRIX_SIZE);
    if (checkResult(h_C.data(), h_C_cpu.data(), MATRIX_SIZE))
        printf("Tensor Cores: Matrix multiplication successful!\n");
    else
        printf("Tensor Cores: Results do not match!\n");

    return 0;
}

int thrustcudacore() {
    // Thrust for input initialization
    thrust::host_vector<float> h_A(MATRIX_SIZE * MATRIX_SIZE);
    thrust::host_vector<float> h_B(MATRIX_SIZE * MATRIX_SIZE);
    thrust::host_vector<float> h_C(MATRIX_SIZE * MATRIX_SIZE);
    thrust::host_vector<float> h_C_cpu(MATRIX_SIZE * MATRIX_SIZE);
    for (int i = 0; i < MATRIX_SIZE * MATRIX_SIZE; ++i) {
        h_A[i] = rand() / (float)RAND_MAX;
        h_B[i] = rand() / (float)RAND_MAX;
    }

    thrust::device_vector<float> d_A = h_A;
    thrust::device_vector<float> d_B = h_B;
    thrust::device_vector<float> d_C(MATRIX_SIZE * MATRIX_SIZE);

    // use wrapper with defaults
    launchMatrixMulCudaCores(d_C.data().get(), d_A.data().get(), d_B.data().get(), MATRIX_SIZE);

    h_C = d_C;
    matrixMulCPU(h_C_cpu.data(), h_A.data(), h_B.data(), MATRIX_SIZE);
    if (checkResult(h_C.data(), h_C_cpu.data(), MATRIX_SIZE))
        printf("CUDA Cores: Matrix multiplication successful!\n");
    else
        printf("CUDA Cores: Results do not match!\n");

    return 0;
}