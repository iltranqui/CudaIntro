#pragma once

#include "darknet_internal.hpp"

#ifdef DARKNET_GPU
void nchw_to_nhwc_gpu(float *input, float *output, int batch, int channels, int height, int width);
void nhwc_to_nchw_gpu(float *input, float *output, int batch, int channels, int height, int width);
#endif
