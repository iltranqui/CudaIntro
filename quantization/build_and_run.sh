#!/usr/bin/env bash
set -euo pipefail

CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
CUDA_INCLUDE="${CUDA_INCLUDE:-${CUDA_HOME}/include}"
CUDA_LIB="${CUDA_LIB:-${CUDA_HOME}/lib64}"

echo "Building gemm_suite..."
nvcc -std=c++17 -O3 -arch=sm_89 \
  -I"${CUDA_INCLUDE}" -L"${CUDA_LIB}" \
  -lcublas -lcudart \
  main.cu GEMM_bf16.cu GEMM_fp16.cu GEMM_fp32.cu GEMM_int8.cu -o gemm_suite

echo "Running gemm_suite..."
./gemm_suite
