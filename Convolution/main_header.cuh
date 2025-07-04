#include "device_launch_parameters.h"

// *************
// **  CONV 1D **
// ************
//#define CONV1D_BASIC_KERNEL
//#define CONV1D_PADDED_SHARED_KERNEL
//#define CONV1D_CUDNN_KERNEL

// 1 Dimensional Convolution
int conv_1d_host();

#ifdef CONV1D_BASIC_KERNEL
int conv_1d_padded(); // 1D convolution with zero padding
#endif

#ifdef CONV1D_PADDED_SHARED_KERNEL
int conv_1d_padded_shared(); // int convoloutiin with zero padding and shared memory
#endif

#ifdef CONV1D_CUDNN_KERNEL
int conv_1d_cudnn(); // 1D convolution using cuDNN
#endif

// *************
// **  CONV 2D **
// *************

//#define CONV2D_BASIC_KERNEL
//#define CONV2D_DEFORM_KERNEL
//#define CONV2D_DEFORM_TRAINING_LOOP
//#define CONV2D_BACKPASS_KERNEL

int conv2d_1d_host(); // 1D convolution on the host (CPU)

#ifdef CONV2D_BASIC_KERNEL
int launchConvolution2DBasicKernel(int width = 5, int height = 5, int r = 1,
    float* h_N = nullptr, float* h_F = nullptr, float* h_P = nullptr,
    dim3 blockDim = dim3(16, 16), dim3 gridDim = dim3(0, 0));
#endif

#ifdef CONV2D_DEFORM_KERNEL
int conv2d_deform_infer();  //int conv2d_deform_backpass();
#endif


int conv2d_inf();

#ifdef CONV2D_BACKPASS_KERNEL
int conv2d_backpass();
#endif

#ifdef CONV2D_DEFORM_TRAINING_LOOP
int conv2d_deform_training_loop();
#endif

//void benchmark_convolution2d(size_t H = 1024, size_t W = 1024, size_t Kh = 3, size_t Kw = 3);

// CUDNN Loop
int cudnn_loop();

// 1 Dimensional Max Pooling
int max_pooling_1d();
int average_pooling_1d();

// Declaration of the external CUDA max pooling function
extern "C" void solution_max_pooling_1d(const float* input, int kernel_size, int stride, int padding, int dilation, float* output, size_t H);
