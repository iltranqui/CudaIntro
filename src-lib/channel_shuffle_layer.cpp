// Reference: https://github.com/gmayday1997/darknet.CG/tree/master/src/channel_shuffle.c
#include "darknet_internal.hpp"


namespace
{
	static auto & cfg_and_state = Darknet::CfgAndState::get();

	void channel_shuffle_op(float *output, float *input, int group_row, int group_column, int len)
	{
		for (int i = 0; i < group_row; ++i)
		{
			for (int j = 0; j < group_column; ++j)
			{
				float *p_i = input + (i * group_column + j) * len;
				float *p_o = output + (j * group_row + i) * len;
				copy_cpu(len, p_i, 1, p_o, 1);
			}
		}
	}
}


Darknet::Layer make_channel_shuffle_layer(int batch, int w, int h, int c, int groups)
{
	TAT(TATPARMS);

	if (groups < 1 || c < 1 || c % groups != 0)
	{
		darknet_fatal_error(DARKNET_LOC, "[channel_shuffle] requires channels=%d to be divisible by groups=%d", c, groups);
	}

	Darknet::Layer l = { (Darknet::ELayerType)0 };
	l.type = Darknet::ELayerType::CHANNEL_SHUFFLE;
	l.batch = batch;
	l.w = w;
	l.h = h;
	l.c = c;
	l.out_w = w;
	l.out_h = h;
	l.out_c = c;
	l.groups = groups;
	l.outputs = l.out_w * l.out_h * l.out_c;
	l.inputs = l.w * l.h * l.c;

	const int output_size = l.outputs * batch;
	l.delta = (float*)xcalloc(output_size, sizeof(float));
	l.output = (float*)xcalloc(output_size, sizeof(float));

	l.forward = forward_channel_shuffle_layer;
	l.backward = backward_channel_shuffle_layer;
#ifdef DARKNET_GPU
	l.forward_gpu = forward_channel_shuffle_layer_gpu;
	l.backward_gpu = backward_channel_shuffle_layer_gpu;
	l.delta_gpu = cuda_make_array(l.delta, output_size);
	l.output_gpu = cuda_make_array(l.output, output_size);
#endif

	*cfg_and_state.output
		<< "channel_shuffle            "
		<< w << " x " << h << " x " << c
		<< " -> " << l.out_w << " x " << l.out_h << " x " << l.out_c
		<< "  groups=" << groups
		<< std::endl;

	return l;
}


void resize_channel_shuffle_layer(Darknet::Layer *l, int h, int w)
{
	TAT(TATPARMS);

	l->h = h;
	l->w = w;
	l->out_h = h;
	l->out_w = w;
	l->outputs = l->out_h * l->out_w * l->out_c;
	l->inputs = l->h * l->w * l->c;
	l->delta = (float*)xrealloc(l->delta, l->outputs * l->batch * sizeof(float));
	l->output = (float*)xrealloc(l->output, l->outputs * l->batch * sizeof(float));

#ifdef DARKNET_GPU
	const int output_size = l->outputs * l->batch;
	cuda_free(l->output_gpu);
	cuda_free(l->delta_gpu);
	l->output_gpu = cuda_make_array(l->output, output_size);
	l->delta_gpu = cuda_make_array(l->delta, output_size);
#endif
}


void forward_channel_shuffle_layer(Darknet::Layer & l, Darknet::NetworkState state)
{
	TAT(TATPARMS);

	const int spatial_size = l.w * l.h;
	const int feature_map_size = spatial_size * l.c;
	const int group_row = l.groups;
	const int group_column = l.c / group_row;

	for (int batch_index = 0; batch_index < l.batch; ++batch_index)
	{
		channel_shuffle_op(
			l.output + batch_index * feature_map_size,
			state.input + batch_index * feature_map_size,
			group_row,
			group_column,
			spatial_size);
	}
}


void backward_channel_shuffle_layer(Darknet::Layer & l, Darknet::NetworkState state)
{
	TAT(TATPARMS);

	const int spatial_size = l.w * l.h;
	const int feature_map_size = spatial_size * l.c;
	const int group_column = l.groups;
	const int group_row = l.c / group_column;

	for (int batch_index = 0; batch_index < l.batch; ++batch_index)
	{
		channel_shuffle_op(
			state.delta + batch_index * feature_map_size,
			l.delta + batch_index * feature_map_size,
			group_row,
			group_column,
			spatial_size);
	}
}


#ifdef DARKNET_GPU
void forward_channel_shuffle_layer_gpu(Darknet::Layer & l, Darknet::NetworkState state)
{
	TAT(TATPARMS);

	const int spatial_size = l.w * l.h;
	const int feature_map_size = spatial_size * l.c;
	const int group_row = l.groups;
	const int group_column = l.c / group_row;
	const int count = l.batch * feature_map_size;

	channel_shuffle_ongpu(count, l.output_gpu, state.input, group_row, group_column, feature_map_size, spatial_size);
}


void backward_channel_shuffle_layer_gpu(Darknet::Layer & l, Darknet::NetworkState state)
{
	TAT(TATPARMS);

	const int spatial_size = l.w * l.h;
	const int feature_map_size = spatial_size * l.c;
	const int group_column = l.groups;
	const int group_row = l.c / group_column;
	const int count = l.batch * feature_map_size;

	channel_shuffle_ongpu(count, state.delta, l.delta_gpu, group_row, group_column, feature_map_size, spatial_size);
}
#endif
