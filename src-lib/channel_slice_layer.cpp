// Reference: https://github.com/gmayday1997/darknet.CG/tree/master/src/channel_slice.c
#include "darknet_internal.hpp"


namespace
{
	static auto & cfg_and_state = Darknet::CfgAndState::get();
}


Darknet::Layer make_channel_slice_layer(int batch, int w, int h, int c, int begin_slice_point, int end_slice_point, int axis, int n, int *input_layers, int *input_sizes)
{
	TAT(TATPARMS);

	if (axis != 1)
	{
		darknet_fatal_error(DARKNET_LOC, "[channel_slice] only axis=1 is currently supported (got axis=%d)", axis);
	}
	if (begin_slice_point < 0 || end_slice_point > c || end_slice_point <= begin_slice_point)
	{
		darknet_fatal_error(DARKNET_LOC, "[channel_slice] invalid slice range start=%d end=%d for channels=%d", begin_slice_point, end_slice_point, c);
	}

	Darknet::Layer l = { (Darknet::ELayerType)0 };
	l.type = Darknet::ELayerType::CHANNEL_SLICE;
	l.batch = batch;
	l.w = w;
	l.h = h;
	l.c = c;
	l.out_w = w;
	l.out_h = h;
	l.n = n;
	l.out_c = end_slice_point - begin_slice_point;
	l.axis = axis;
	l.begin_slice_point = begin_slice_point;
	l.end_slice_point = end_slice_point;
	l.outputs = l.out_w * l.out_h * l.out_c;
	l.inputs = l.w * l.h * l.c;
	l.input_layers = input_layers;
	l.input_sizes = input_sizes;

	const int output_size = l.outputs * batch;
	l.delta = (float*)xcalloc(output_size, sizeof(float));
	l.output = (float*)xcalloc(output_size, sizeof(float));

	l.forward = forward_channel_slice_layer;
	l.backward = backward_channel_slice_layer;
#ifdef DARKNET_GPU
	l.forward_gpu = forward_channel_slice_layer_gpu;
	l.backward_gpu = backward_channel_slice_layer_gpu;
	l.output_gpu = cuda_make_array(l.output, output_size);
	l.delta_gpu = cuda_make_array(l.delta, output_size);
#endif

	*cfg_and_state.output
		<< "channel_slice              "
		<< w << " x " << h << " x " << c
		<< " -> " << l.out_w << " x " << l.out_h << " x " << l.out_c
		<< "  start=" << begin_slice_point
		<< " end=" << end_slice_point
		<< std::endl;

	return l;
}


void resize_channel_slice_layer(Darknet::Layer *l, Darknet::Network *net)
{
	TAT(TATPARMS);

	Darknet::Layer & from = net->layers[l->input_layers[0]];
	l->h = from.out_h;
	l->w = from.out_w;
	l->c = from.out_c;
	l->out_h = from.out_h;
	l->out_w = from.out_w;
	l->outputs = l->out_h * l->out_w * l->out_c;
	l->inputs = l->h * l->w * l->c;
	l->input_sizes[0] = from.outputs;
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


void forward_channel_slice_layer(Darknet::Layer & l, Darknet::NetworkState state)
{
	TAT(TATPARMS);

	const int input_slice_axis = l.c;
	const int output_slice_axis = l.end_slice_point - l.begin_slice_point;
	const int spatial_size = l.h * l.w;

	for (int i = 0; i < l.n; ++i)
	{
		const int index = l.input_layers[i];
		float *input = state.net.layers[index].output;
		for (int batch_index = 0; batch_index < l.batch; ++batch_index)
		{
			const int input_offset = (batch_index * input_slice_axis + l.begin_slice_point) * spatial_size;
			const int output_offset = batch_index * output_slice_axis * spatial_size;
			copy_cpu(output_slice_axis * spatial_size, input + input_offset, 1, l.output + output_offset, 1);
		}
	}
}


void backward_channel_slice_layer(Darknet::Layer & l, Darknet::NetworkState state)
{
	TAT(TATPARMS);

	const int input_slice_axis = l.c;
	const int output_slice_axis = l.end_slice_point - l.begin_slice_point;
	const int spatial_size = l.h * l.w;

	for (int i = 0; i < l.n; ++i)
	{
		const int index = l.input_layers[i];
		float *delta = state.net.layers[index].delta;
		for (int batch_index = 0; batch_index < l.batch; ++batch_index)
		{
			const int input_offset = batch_index * output_slice_axis * spatial_size;
			const int output_offset = (batch_index * input_slice_axis + l.begin_slice_point) * spatial_size;
			copy_cpu(output_slice_axis * spatial_size, l.delta + input_offset, 1, delta + output_offset, 1);
		}
	}
}


#ifdef DARKNET_GPU
void forward_channel_slice_layer_gpu(Darknet::Layer & l, Darknet::NetworkState state)
{
	TAT(TATPARMS);

	const int input_slice_axis = l.c;
	const int output_slice_axis = l.end_slice_point - l.begin_slice_point;
	const int spatial_size = l.h * l.w;
	const int count = spatial_size * l.batch * output_slice_axis;

	for (int i = 0; i < l.n; ++i)
	{
		const int index = l.input_layers[i];
		float *input = state.net.layers[index].output_gpu;
		channel_slice_ongpu(count, l.output_gpu, input, l.batch, spatial_size, input_slice_axis, output_slice_axis, l.begin_slice_point, 1);
	}
}


void backward_channel_slice_layer_gpu(Darknet::Layer & l, Darknet::NetworkState state)
{
	TAT(TATPARMS);

	const int input_slice_axis = l.c;
	const int output_slice_axis = l.end_slice_point - l.begin_slice_point;
	const int spatial_size = l.h * l.w;
	const int count = spatial_size * l.batch * output_slice_axis;

	for (int i = 0; i < l.n; ++i)
	{
		const int index = l.input_layers[i];
		float *delta = state.net.layers[index].delta_gpu;
		channel_slice_ongpu(count, delta, l.delta_gpu, l.batch, spatial_size, input_slice_axis, output_slice_axis, l.begin_slice_point, 0);
	}
}
#endif
