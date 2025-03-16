#include "device_launch_parameters.h"

int conv_1d_host();

int conv_1d_padded(); // 1D convolution with zero padding
int conv_1d_padded_shared(); // int convoloutiin with zero padding and shared memory

int conv2d_deform_infer();
int conv2d_deform_backpass();
int conv2d_inf();
int conv2d_backpass();
int conv2d_deform_training_loop();

int launchConvolution2DBasicKernel(int width = 5, int height = 5, int r = 1,
    float* h_N = nullptr, float* h_F = nullptr, float* h_P = nullptr,
    dim3 blockDim = dim3(16, 16), dim3 gridDim = dim3(0, 0));


