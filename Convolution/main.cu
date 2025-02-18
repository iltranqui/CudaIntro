#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <stdio.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "main_header.cuh"

int main() {

	// AI conv deform
	conv2d_deform();
	// conv 2d pmpp
	launchConvolution2DBasicKernel();
	return 0;
}