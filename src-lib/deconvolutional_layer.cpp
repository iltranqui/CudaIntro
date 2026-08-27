// Reference: https://github.com/gmayday1997/darknet.CG/tree/master/src/deconvolutional_layer.c
#include "darknet_internal.hpp"
#include "deconvolutional_layer.hpp"
#include "activations.hpp"
#include "blas.hpp"
#include "col2im.hpp"
#include "gemm.hpp"
#include "im2col.hpp"


namespace
{
	static auto & cfg_and_state = Darknet::CfgAndState::get();
}


int deconvolutional_out_height(const Darknet::Layer & l)
{
	TAT(TATPARMS);
	return l.stride * (l.h - 1) + l.size;
}


int deconvolutional_out_width(const Darknet::Layer & l)
{
	TAT(TATPARMS);
	return l.stride * (l.w - 1) + l.size;
}


Darknet::Image get_deconvolutional_image(const Darknet::Layer & l)
{
	TAT(TATPARMS);
	return Darknet::float_to_image(deconvolutional_out_width(l), deconvolutional_out_height(l), l.n, l.output);
}


Darknet::Image get_deconvolutional_delta(const Darknet::Layer & l)
{
	TAT(TATPARMS);
	return Darknet::float_to_image(deconvolutional_out_width(l), deconvolutional_out_height(l), l.n, l.delta);
}


Darknet::Layer make_deconvolutional_layer(int batch, int h, int w, int c, int n, int size, int stride, ACTIVATION activation)
{
	TAT(TATPARMS);

	if (batch < 1 || h < 1 || w < 1 || c < 1 || n < 1 || size < 1 || stride < 1)
	{
		darknet_fatal_error(DARKNET_LOC, "[deconvolutional] invalid dimensions batch=%d h=%d w=%d c=%d filters=%d size=%d stride=%d", batch, h, w, c, n, size, stride);
	}

	Darknet::Layer l = { (Darknet::ELayerType)0 };
	l.type = Darknet::ELayerType::DECONVOLUTIONAL;
	l.batch = batch;
	l.h = h;
	l.w = w;
	l.c = c;
	l.n = n;
	l.size = size;
	l.stride = l.stride_x = l.stride_y = stride;
	l.groups = 1;
	l.pad = 0;
	l.dilation = 1;
	l.activation = activation;
	l.learning_rate_scale = 1.0f;

	l.nweights = c * n * size * size;
	l.weights = (float*)xcalloc(l.nweights, sizeof(float));
	l.weight_updates = (float*)xcalloc(l.nweights, sizeof(float));
	l.biases = (float*)xcalloc(n, sizeof(float));
	l.bias_updates = (float*)xcalloc(n, sizeof(float));

	const float scale = 1.0f / std::sqrt(static_cast<float>(size * size * c));
	for (int i = 0; i < l.nweights; ++i)
	{
		l.weights[i] = scale * rand_normal();
	}
	for (int i = 0; i < n; ++i)
	{
		l.biases[i] = scale;
	}

	l.out_h = deconvolutional_out_height(l);
	l.out_w = deconvolutional_out_width(l);
	l.out_c = n;
	l.outputs = l.out_w * l.out_h * l.out_c;
	l.inputs = l.w * l.h * l.c;

	const int col_image_size = l.h * l.w * l.size * l.size * l.n;
	l.col_image = (float*)xcalloc(col_image_size, sizeof(float));
	l.output = (float*)xcalloc(l.batch * l.outputs, sizeof(float));
	l.delta = (float*)xcalloc(l.batch * l.outputs, sizeof(float));

	l.forward = forward_deconvolutional_layer;
	l.backward = backward_deconvolutional_layer;
	l.update = update_deconvolutional_layer;

#ifdef DARKNET_GPU
	l.forward_gpu = forward_deconvolutional_layer_gpu;
	l.backward_gpu = backward_deconvolutional_layer_gpu;
	l.update_gpu = update_deconvolutional_layer_gpu;
	l.weights_gpu = cuda_make_array(l.weights, l.nweights);
	l.weight_updates_gpu = cuda_make_array(l.weight_updates, l.nweights);
	l.biases_gpu = cuda_make_array(l.biases, n);
	l.bias_updates_gpu = cuda_make_array(l.bias_updates, n);
	l.col_image_gpu = cuda_make_array(l.col_image, col_image_size);
	l.output_gpu = cuda_make_array(l.output, l.batch * l.outputs);
	l.delta_gpu = cuda_make_array(l.delta, l.batch * l.outputs);
#endif

	*cfg_and_state.output
		<< "deconvolutional            "
		<< h << " x " << w << " x " << c
		<< " -> " << l.out_h << " x " << l.out_w << " x " << l.out_c
		<< "  filters=" << n
		<< " size=" << size
		<< " stride=" << stride
		<< std::endl;

	return l;
}


void resize_deconvolutional_layer(Darknet::Layer *l, int h, int w)
{
	TAT(TATPARMS);

	l->h = h;
	l->w = w;
	l->out_h = deconvolutional_out_height(*l);
	l->out_w = deconvolutional_out_width(*l);
	l->out_c = l->n;
	l->outputs = l->out_h * l->out_w * l->out_c;
	l->inputs = l->h * l->w * l->c;

	const int col_image_size = l->h * l->w * l->size * l->size * l->n;
	l->col_image = (float*)xrealloc(l->col_image, col_image_size * sizeof(float));
	l->output = (float*)xrealloc(l->output, l->batch * l->outputs * sizeof(float));
	l->delta = (float*)xrealloc(l->delta, l->batch * l->outputs * sizeof(float));

#ifdef DARKNET_GPU
	cuda_free(l->col_image_gpu);
	cuda_free(l->output_gpu);
	cuda_free(l->delta_gpu);
	l->col_image_gpu = cuda_make_array(l->col_image, col_image_size);
	l->output_gpu = cuda_make_array(l->output, l->batch * l->outputs);
	l->delta_gpu = cuda_make_array(l->delta, l->batch * l->outputs);
#endif
}


void forward_deconvolutional_layer(Darknet::Layer & l, Darknet::NetworkState state)
{
	TAT(TATPARMS);

	const int out_h = deconvolutional_out_height(l);
	const int out_w = deconvolutional_out_width(l);
	const int spatial = out_h * out_w;

	const int m = l.size * l.size * l.n;
	const int n = l.h * l.w;
	const int k = l.c;

	fill_cpu(l.outputs * l.batch, 0.0f, l.output, 1);

	for (int i = 0; i < l.batch; ++i)
	{
		float *a = l.weights;
		float *b = state.input + i * l.inputs;
		float *c = l.col_image;

		gemm_cpu(1, 0, m, n, k, 1.0f, a, m, b, n, 0.0f, c, n);
		col2im_cpu(c, l.n, out_h, out_w, l.size, l.stride, 0, l.output + i * l.n * spatial);
	}

	add_bias(l.output, l.biases, l.batch, l.n, spatial);
	activate_array(l.output, l.batch * l.n * spatial, l.activation);
}


void backward_deconvolutional_layer(Darknet::Layer & l, Darknet::NetworkState state)
{
	TAT(TATPARMS);

	const float alpha = 1.0f / l.batch;
	const int out_h = deconvolutional_out_height(l);
	const int out_w = deconvolutional_out_width(l);
	const int spatial = out_h * out_w;

	gradient_array(l.output, spatial * l.n * l.batch, l.activation, l.delta);
	backward_bias(l.bias_updates, l.delta, l.batch, l.n, spatial);

	for (int i = 0; i < l.batch; ++i)
	{
		{
			const int m = l.c;
			const int n = l.size * l.size * l.n;
			const int k = l.h * l.w;

			float *a = state.input + i * l.inputs;
			float *b = l.col_image;
			float *c = l.weight_updates;

			im2col_cpu(l.delta + i * l.outputs, l.n, out_h, out_w, l.size, l.stride, 0, b);
			gemm_cpu(0, 1, m, n, k, alpha, a, k, b, k, 1.0f, c, n);
		}

		if (state.delta)
		{
			const int m = l.c;
			const int n = l.h * l.w;
			const int k = l.size * l.size * l.n;

			float *a = l.weights;
			float *b = l.col_image;
			float *c = state.delta + i * l.inputs;

			gemm_cpu(0, 0, m, n, k, 1.0f, a, k, b, n, 1.0f, c, n);
		}
	}
}


void update_deconvolutional_layer(Darknet::Layer & l, int batch, float learning_rate_init, float momentum, float decay)
{
	TAT(TATPARMS);

	const float learning_rate = learning_rate_init * l.learning_rate_scale;

	axpy_cpu(l.n, learning_rate / batch, l.bias_updates, 1, l.biases, 1);
	scal_cpu(l.n, momentum, l.bias_updates, 1);

	axpy_cpu(l.nweights, -decay * batch, l.weights, 1, l.weight_updates, 1);
	axpy_cpu(l.nweights, learning_rate / batch, l.weight_updates, 1, l.weights, 1);
	scal_cpu(l.nweights, momentum, l.weight_updates, 1);
}
