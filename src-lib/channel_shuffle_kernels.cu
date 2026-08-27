// Reference: https://github.com/gmayday1997/darknet.CG/tree/master/src/blas_kernels.cu
#include "darknet_internal.hpp"
#include "channel_shuffle_layer.hpp"

#ifdef DARKNET_GPU
__global__ void channel_shuffle_kernel(int count, float *output, float *input, int group_row, int group_column, int feature_map_size, int spatial_size)
{
	const int index = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
	if (index >= count) return;

	const int batch_index = index / feature_map_size;
	const int feature_offset = index % feature_map_size;
	const int input_row = feature_offset / (group_column * spatial_size);
	const int input_column = (feature_offset / spatial_size) % group_column;
	const int spatial_offset = feature_offset % spatial_size;
	const int output_index = batch_index * feature_map_size + (input_column * group_row + input_row) * spatial_size + spatial_offset;

	output[output_index] = input[index];
}

void channel_shuffle_ongpu(int count, float *output, float *input, int group_row, int group_column, int feature_map_size, int spatial_size)
{
	TAT(TATPARMS);

	channel_shuffle_kernel<<<cuda_gridsize(count), BLOCK, 0, get_cuda_stream()>>>(
		count,
		output,
		input,
		group_row,
		group_column,
		feature_map_size,
		spatial_size);
	CHECK_CUDA(cudaPeekAtLastError());
}
#endif
