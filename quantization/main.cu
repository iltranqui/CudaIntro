#include "gemm_common.h"

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <iostream>

int main() {
  cublasHandle_t handle = nullptr;
  CHECK_CUBLAS(cublasCreate(&handle));
  CHECK_CUBLAS(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));

  std::cout << "Running BF16..." << std::endl;
  GemmResult bf16 = run_gemm_bf16(handle);
  std::cout << "Running FP16..." << std::endl;
  GemmResult fp16 = run_gemm_fp16(handle);
  std::cout << "Running FP32..." << std::endl;
  GemmResult fp32 = run_gemm_fp32(handle);
  std::cout << "Running INT8..." << std::endl;
  GemmResult int8 = run_gemm_int8(handle);

  const GemmResult results[] = {bf16, fp16, fp32, int8};
  for (const auto& result : results) {
    std::cout << "Ran " << result.iters << " " << result.name
              << " GEMMs in " << result.secs << " s (" << result.tflops
              << " TFLOP/s)\n";
  }

  CHECK_CUBLAS(cublasDestroy(handle));
  return 0;
}
