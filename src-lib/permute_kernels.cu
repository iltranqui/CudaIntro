#include "darknet_internal.hpp"
#include "permute_kernels.hpp"

__global__ void nchw_to_nhwc_kernel(float *input, float *output, int channels, int spatial)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= channels * spatial) return;

    int b = blockIdx.y;
    int c = index / spatial;
    int s = index % spatial;

    output[(b * spatial + s) * channels + c] = input[(b * channels + c) * spatial + s];
}

__global__ void nhwc_to_nchw_kernel(float *input, float *output, int channels, int spatial)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= channels * spatial) return;

    int b = blockIdx.y;
    int c = index / spatial;
    int s = index % spatial;

    output[(b * channels + c) * spatial + s] = input[(b * spatial + s) * channels + c];
}

void nchw_to_nhwc_gpu(float *input, float *output, int batch, int channels, int height, int width)
{
    int spatial = height * width;
    int size = channels * spatial;
    int threads = 512;
    int blocks = (size + threads - 1) / threads;
    dim3 grid(blocks, batch);
    nchw_to_nhwc_kernel<<<grid, threads, 0, get_cuda_stream()>>>(input, output, channels, spatial);
    CHECK_CUDA(cudaPeekAtLastError());
}

void nhwc_to_nchw_gpu(float *input, float *output, int batch, int channels, int height, int width)
{
    int spatial = height * width;
    int size = channels * spatial;
    int threads = 512;
    int blocks = (size + threads - 1) / threads;
    dim3 grid(blocks, batch);
    nhwc_to_nchw_kernel<<<grid, threads, 0, get_cuda_stream()>>>(input, output, channels, spatial);
    CHECK_CUDA(cudaPeekAtLastError());
}
