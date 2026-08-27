// Reference: https://github.com/gmayday1997/darknet.CG/tree/master/src/blas_kernels.cu
#include "darknet_internal.hpp"
#include "channel_slice_layer.hpp"

#ifdef DARKNET_GPU
__global__ void channel_slice_kernel(int count, float *output, float *input, int spatial_size, int input_slice_axis, int output_slice_axis, int begin_slice_axis, int forward)
{
	const int index = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
	if (index >= count) return;

	const int feature_map_size = spatial_size * output_slice_axis;
	const int slice_num = index / feature_map_size;
	const int slice_index = index % feature_map_size;
	const int input_index = slice_index + (slice_num * input_slice_axis + begin_slice_axis) * spatial_size;

	if (forward)
	{
		output[index] = input[input_index];
	}
	else
	{
		output[input_index] = input[index];
	}
}

void channel_slice_ongpu(int count, float *output, float *input, int batch_size, int spatial_size, int input_slice_axis, int output_slice_axis, int begin_slice_axis, int forward)
{
	TAT(TATPARMS);
	(void) batch_size;

	channel_slice_kernel<<<cuda_gridsize(count), BLOCK, 0, get_cuda_stream()>>>(
		count,
		output,
		input,
		spatial_size,
		input_slice_axis,
		output_slice_axis,
		begin_slice_axis,
		forward);
	CHECK_CUDA(cudaPeekAtLastError());
}
#endif
