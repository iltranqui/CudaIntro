#include "gemm_common.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <chrono>

__global__ void fill_random_fp16(__half* data, int64_t count, uint64_t seed,
                                 float lo, float hi) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= count) {
    return;
  }
  uint64_t x = static_cast<uint64_t>(idx) ^ seed;
  x = x * 2862933555777941757ULL + 3037000493ULL;
  float r = static_cast<float>((x >> 11) & 0xFFFFFF) / 16777215.0f;
  float val = lo + r * (hi - lo);
  data[idx] = __float2half(val);
}

GemmResult run_gemm_fp16(cublasHandle_t handle) {
  constexpr int64_t M = 4096;
  constexpr int64_t N = 4096;
  constexpr int64_t K = 4096;

  constexpr float kMin = -1.0f;
  constexpr float kMax = 1.0f;

  int64_t a_elems = M * K;
  int64_t b_elems = K * N;
  int64_t c_elems = M * N;

  __half* a = nullptr;
  __half* b = nullptr;
  __half* c = nullptr;

  CHECK_CUDA(cudaMalloc(&a, a_elems * sizeof(__half)));
  CHECK_CUDA(cudaMalloc(&b, b_elems * sizeof(__half)));
  CHECK_CUDA(cudaMalloc(&c, c_elems * sizeof(__half)));

  constexpr int threads = 256;
  int blocks_a = static_cast<int>((a_elems + threads - 1) / threads);
  int blocks_b = static_cast<int>((b_elems + threads - 1) / threads);
  fill_random_fp16<<<blocks_a, threads>>>(a, a_elems, 0x2234U, kMin, kMax);
  fill_random_fp16<<<blocks_b, threads>>>(b, b_elems, 0x8876U, kMin, kMax);
  CHECK_CUDA(cudaGetLastError());

  float alpha = 1.0f;
  float beta = 0.0f;

  CHECK_CUBLAS(cublasGemmEx(
      handle, CUBLAS_OP_N, CUBLAS_OP_N, static_cast<int>(M),
      static_cast<int>(N), static_cast<int>(K), &alpha, a, CUDA_R_16F,
      static_cast<int>(M), b, CUDA_R_16F, static_cast<int>(K), &beta, c,
      CUDA_R_16F, static_cast<int>(M), CUBLAS_COMPUTE_32F,
      CUBLAS_GEMM_DEFAULT_TENSOR_OP));
  CHECK_CUDA(cudaDeviceSynchronize());

  auto start = std::chrono::steady_clock::now();
  auto now = start;
  uint64_t iters = 0;

  while (std::chrono::duration<double>(now - start).count() < 10.0) {
    CHECK_CUBLAS(cublasGemmEx(
        handle, CUBLAS_OP_N, CUBLAS_OP_N, static_cast<int>(M),
        static_cast<int>(N), static_cast<int>(K), &alpha, a, CUDA_R_16F,
        static_cast<int>(M), b, CUDA_R_16F, static_cast<int>(K), &beta, c,
        CUDA_R_16F, static_cast<int>(M), CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP));
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

  return {"FP16", iters, secs, tflops};
}
