#include "gemm_common.h"

#include <cuda_runtime.h>

#include <chrono>
#include <cstdint>

__global__ void fill_random_int8(int8_t* data, int64_t count, uint64_t seed,
                                 int lo, int hi) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= count) {
    return;
  }
  uint64_t x = static_cast<uint64_t>(idx) ^ seed;
  x = x * 2862933555777941757ULL + 3037000493ULL;
  int r = static_cast<int>((x >> 11) & 0xFFFF);
  int val = lo + (r % (hi - lo + 1));
  data[idx] = static_cast<int8_t>(val);
}

GemmResult run_gemm_int8(cublasHandle_t handle) {
  constexpr int64_t M = 4096;
  constexpr int64_t N = 4096;
  constexpr int64_t K = 4096;

  constexpr int kMin = -127;
  constexpr int kMax = 127;

  int64_t a_elems = M * K;
  int64_t b_elems = K * N;
  int64_t c_elems = M * N;

  int8_t* a = nullptr;
  int8_t* b = nullptr;
  int32_t* c = nullptr;

  CHECK_CUDA(cudaMalloc(&a, a_elems * sizeof(int8_t)));
  CHECK_CUDA(cudaMalloc(&b, b_elems * sizeof(int8_t)));
  CHECK_CUDA(cudaMalloc(&c, c_elems * sizeof(int32_t)));

  constexpr int threads = 256;
  int blocks_a = static_cast<int>((a_elems + threads - 1) / threads);
  int blocks_b = static_cast<int>((b_elems + threads - 1) / threads);
  fill_random_int8<<<blocks_a, threads>>>(a, a_elems, 0x4234U, kMin, kMax);
  fill_random_int8<<<blocks_b, threads>>>(b, b_elems, 0x6676U, kMin, kMax);
  CHECK_CUDA(cudaGetLastError());

  int32_t alpha = 1;
  int32_t beta = 0;

  CHECK_CUBLAS(cublasGemmEx(
      handle, CUBLAS_OP_N, CUBLAS_OP_N, static_cast<int>(M),
      static_cast<int>(N), static_cast<int>(K), &alpha, a, CUDA_R_8I,
      static_cast<int>(M), b, CUDA_R_8I, static_cast<int>(K), &beta, c,
      CUDA_R_32I, static_cast<int>(M), CUBLAS_COMPUTE_32I,
      CUBLAS_GEMM_DEFAULT));
  CHECK_CUDA(cudaDeviceSynchronize());

  auto start = std::chrono::steady_clock::now();
  auto now = start;
  uint64_t iters = 0;

  while (std::chrono::duration<double>(now - start).count() < 10.0) {
    CHECK_CUBLAS(cublasGemmEx(
        handle, CUBLAS_OP_N, CUBLAS_OP_N, static_cast<int>(M),
        static_cast<int>(N), static_cast<int>(K), &alpha, a, CUDA_R_8I,
        static_cast<int>(M), b, CUDA_R_8I, static_cast<int>(K), &beta, c,
        CUDA_R_32I, static_cast<int>(M), CUBLAS_COMPUTE_32I,
        CUBLAS_GEMM_DEFAULT));
    ++iters;
    if ((iters & 0xFF) == 0) {
      CHECK_CUDA(cudaDeviceSynchronize());
      now = std::chrono::steady_clock::now();
    }
  }
  CHECK_CUDA(cudaDeviceSynchronize());
  auto end = std::chrono::steady_clock::now();
  double secs = std::chrono::duration<double>(end - start).count();

  double flops_per_iter =
      2.0 * static_cast<double>(M) * static_cast<double>(N) *
      static_cast<double>(K);
  double tflops =
      (flops_per_iter * static_cast<double>(iters)) / (secs * 1.0e12);

  CHECK_CUDA(cudaFree(c));
  CHECK_CUDA(cudaFree(a));
  CHECK_CUDA(cudaFree(b));

  return {"INT8", iters, secs, tflops};
}
