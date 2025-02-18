#ifndef CUDA_KERNELS_CUH
#define CUDA_KERNELS_CUH

void forward_cuda(float* input, float* weights, float* biases, float* output, int in_size, int out_size);
void backward_cuda(float* grad_output, float* weights, float* biases, int in_size, int out_size, float learning_rate);

#endif // CUDA_KERNELS_CUH
