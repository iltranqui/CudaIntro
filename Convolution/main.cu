#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <stdio.h>
#include "main_header.cuh"

int main() {

	// AI conv deform
	//conv2d_deform_infer();
	//conv2d_deform_training_loop();
	// conv 2d pmpp
	//launchConvolution2DBasicKernel();
	// conv 2d backpass
	//conv2d_backpass();
	// conv_1d_host: CPU vs GPU
	//conv_1d_host();
	// conv_1d_padded: CPU vs GPU
	conv_1d_padded();
	conv_1d_padded_shared();
	return 0;
}