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

	conv2d_deform();
	return 0;
}