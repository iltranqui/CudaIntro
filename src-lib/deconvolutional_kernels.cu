// Reference: https://github.com/gmayday1997/darknet.CG/tree/master/src/deconvolutional_kernels.cu
#include "darknet_internal.hpp"
#include "deconvolutional_layer.hpp"
#include "activations.hpp"
#include "blas.hpp"
#include "col2im.hpp"
#include "gemm.hpp"
#include "im2col.hpp"


#ifdef DARKNET_GPU
void forward_deconvolutional_layer_gpu(Darknet::Layer & l, Darknet::NetworkState state)
{
	TAT(TATPARMS);

	const int out_h = deconvolutional_out_height(l);
	const int out_w = deconvolutional_out_width(l);
	const int spatial = out_h * out_w;

	const int m = l.size * l.size * l.n;
	const int n = l.h * l.w;
	const int k = l.c;

	fill_ongpu(l.outputs * l.batch, 0.0f, l.output_gpu, 1);

	for (int i = 0; i < l.batch; ++i)
	{
		float *a = l.weights_gpu;
		float *b = state.input + i * l.inputs;
		float *c = l.col_image_gpu;

		gemm_ongpu(1, 0, m, n, k, 1.0f, a, m, b, n, 0.0f, c, n);
		col2im_ongpu(c, l.n, out_h, out_w, l.size, l.stride, 0, l.output_gpu + i * l.n * spatial);
	}

	add_bias_gpu(l.output_gpu, l.biases_gpu, l.batch, l.n, spatial);
	activate_array_ongpu(l.output_gpu, l.batch * l.n * spatial, l.activation);
}


void backward_deconvolutional_layer_gpu(Darknet::Layer & l, Darknet::NetworkState state)
{
	TAT(TATPARMS);

	const float alpha = 1.0f / l.batch;
	const int out_h = deconvolutional_out_height(l);
	const int out_w = deconvolutional_out_width(l);
	const int spatial = out_h * out_w;

	gradient_array_ongpu(l.output_gpu, spatial * l.n * l.batch, l.activation, l.delta_gpu);
	backward_bias_gpu(l.bias_updates_gpu, l.delta_gpu, l.batch, l.n, spatial);

	for (int i = 0; i < l.batch; ++i)
	{
		{
			const int m = l.c;
			const int n = l.size * l.size * l.n;
			const int k = l.h * l.w;

			float *a = state.input + i * l.inputs;
			float *b = l.col_image_gpu;
			float *c = l.weight_updates_gpu;

			im2col_ongpu(l.delta_gpu + i * l.outputs, l.n, out_h, out_w, l.size, l.stride, 0, b);
			gemm_ongpu(0, 1, m, n, k, alpha, a, k, b, k, 1.0f, c, n);
		}

		if (state.delta)
		{
			const int m = l.c;
			const int n = l.h * l.w;
			const int k = l.size * l.size * l.n;

			float *a = l.weights_gpu;
			float *b = l.col_image_gpu;
			float *c = state.delta + i * l.inputs;

			gemm_ongpu(0, 0, m, n, k, 1.0f, a, k, b, n, 1.0f, c, n);
		}
	}
}


void pull_deconvolutional_layer(Darknet::Layer & l)
{
	TAT(TATPARMS);

	cuda_pull_array(l.weights_gpu, l.weights, l.nweights);
	cuda_pull_array(l.biases_gpu, l.biases, l.n);
	cuda_pull_array(l.weight_updates_gpu, l.weight_updates, l.nweights);
	cuda_pull_array(l.bias_updates_gpu, l.bias_updates, l.n);
	CHECK_CUDA(cudaPeekAtLastError());
}


void push_deconvolutional_layer(Darknet::Layer & l)
{
	TAT(TATPARMS);

	cuda_push_array(l.weights_gpu, l.weights, l.nweights);
	cuda_push_array(l.biases_gpu, l.biases, l.n);
	cuda_push_array(l.weight_updates_gpu, l.weight_updates, l.nweights);
	cuda_push_array(l.bias_updates_gpu, l.bias_updates, l.n);
	CHECK_CUDA(cudaPeekAtLastError());
}


void update_deconvolutional_layer_gpu(Darknet::Layer & l, int batch, float learning_rate_init, float momentum, float decay, float loss_scale)
{
	TAT(TATPARMS);

	const float learning_rate = learning_rate_init * l.learning_rate_scale;

	if (loss_scale != 1.0f)
	{
		scal_ongpu(l.nweights, 1.0f / loss_scale, l.weight_updates_gpu, 1);
		scal_ongpu(l.n, 1.0f / loss_scale, l.bias_updates_gpu, 1);
	}

	axpy_ongpu(l.n, learning_rate / batch, l.bias_updates_gpu, 1, l.biases_gpu, 1);
	scal_ongpu(l.n, momentum, l.bias_updates_gpu, 1);

	axpy_ongpu(l.nweights, -decay * batch, l.weights_gpu, 1, l.weight_updates_gpu, 1);
	axpy_ongpu(l.nweights, learning_rate / batch, l.weight_updates_gpu, 1, l.weights_gpu, 1);
	scal_ongpu(l.nweights, momentum, l.weight_updates_gpu, 1);
}
#endif
