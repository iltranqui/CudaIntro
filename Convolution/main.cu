#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <stdio.h>
#include "main_header.cuh"



int main() {
	// AI conv deform - Deformable Convolution with Bilinear Interpolation
#ifdef CONV2D_DEFORM_KERNEL
	conv2d_deform_infer();
#endif
#ifdef CONV2D_DEFORM_TRAINING_LOOP
	conv2d_deform_training_loop();
#endif

	// Other convolution implementations
	// conv 2d pmpp
	//launchConvolution2DBasicKernel();
	// conv 2d backpass
	//conv2d_backpass();
	// conv_1d_host: CPU vs GPU
	//conv_1d_host();
	// conv_1d_padded: CPU vs GPU
	//conv_1d_padded();
	//conv_1d_padded_shared();
	//conv_1d_cudnn();
	// benchmarkconv 2d
	//benchmark_convolution2d();

	// cudnn loop
	//cudnn_loop();

	// max pooling 1d
	//max_pooling_1d();
	//average_pooling_1d();

	return 0;
}
