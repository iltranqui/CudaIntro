#include "tucker_attention_layer.hpp"
#include "activations.hpp"
#include "gemm.hpp"
#ifdef DARKNET_HAS_FP8
#include "fp8_gemm.hpp"
#include "fp8_kernels.hpp"
#include "fp8_layer_release.hpp"
#endif
#ifdef DARKNET_HAS_FP4
#include "fp4_gemm.hpp"
#endif

#include <algorithm>
#include <cmath>

namespace
{
	auto & cfg_and_state = Darknet::CfgAndState::get();

	int pad_to(int x, int m)
	{
		return ((x + m - 1) / m) * m;
	}

	struct TuckerOffsets
	{
		size_t q_basis;
		size_t k_basis;
		size_t v_basis;
		size_t q_core;
		size_t k_core;
		size_t v_core;
		size_t o_core;
		size_t o_basis;
		size_t total;
	};

	TuckerOffsets offsets(const Darknet::Layer &l)
	{
		TuckerOffsets o = {};
		const size_t C = l.c;
		const size_t N = l.n;
		const size_t H = l.tucker_heads;
		const size_t D = l.tucker_head_dim;
		const size_t Rq = l.tucker_rank_q;
		const size_t Rk = l.tucker_rank_k;
		const size_t Rv = l.tucker_rank_v;
		const size_t Ro = l.tucker_rank_o;
		o.q_basis = 0;
		o.k_basis = o.q_basis + C * Rq;
		o.v_basis = o.k_basis + C * Rk;
		o.q_core = o.v_basis + C * Rv;
		o.k_core = o.q_core + H * Rq * D;
		o.v_core = o.k_core + H * Rk * D;
		o.o_core = o.v_core + H * Rv * D;
		o.o_basis = o.o_core + H * D * Ro;
		o.total = o.o_basis + Ro * N;
		return o;
	}

#ifdef DARKNET_GPU
	size_t tucker_window_count(const Darknet::Layer &l)
	{
		const int win_h = pad_to(l.h, l.tucker_window_size) / l.tucker_window_size;
		const int win_w = pad_to(l.w, l.tucker_window_size) / l.tucker_window_size;
		return static_cast<size_t>(l.batch) * win_h * win_w;
	}

#ifdef DARKNET_HAS_FP8
	/// Opt-in, additive FP8 buffers for the Q@K^T and attn@V GEMMs only (see
	/// forward_tucker_attention_layer_gpu's DARKNET_TUCKER_USE_CUBLAS_HALF branch).
	/// Allocated unconditionally when FP8 support is built in; l.fp8_tucker_attention
	/// defaults to 0 (off) so this has no effect until enabled.
	///
	/// T (window_size^2) is not generally a multiple of 16, which cuBLASLt FP8 requires
	/// for the padded dimension -- key_pad is T rounded up to the next multiple of 16.
	/// Scores GEMM: A=K (key_pad, D), B=Q (T, D).  Context GEMM: A=V^T (D, key_pad),
	/// B=attn (T, key_pad).  See the field comments in darknet_layers.hpp.
	void allocate_tucker_fp8_workspace(Darknet::Layer &l, const int windows, const int heads, const int T, const int D)
	{
		const int key_pad = Darknet::fp8_round_up_to_16(T);
		const int batch = heads * windows;
		const size_t qk_bytes = static_cast<size_t>(batch) * T * D;
		const size_t k_vt_bytes = static_cast<size_t>(batch) * key_pad * D;
		const size_t attn_bytes = static_cast<size_t>(batch) * T * key_pad;

		CHECK_CUDA(cudaMalloc(reinterpret_cast<void **>(&l.fp8_tucker_q_gpu), qk_bytes));
		CHECK_CUDA(cudaMalloc(reinterpret_cast<void **>(&l.fp8_tucker_k_gpu), k_vt_bytes));
		CHECK_CUDA(cudaMalloc(reinterpret_cast<void **>(&l.fp8_tucker_attn_gpu), attn_bytes));
		CHECK_CUDA(cudaMalloc(reinterpret_cast<void **>(&l.fp8_tucker_v_t_gpu), k_vt_bytes));

		l.fp8_tucker_q_amax_gpu     = cuda_make_array(nullptr, 1);
		l.fp8_tucker_q_scale_gpu    = cuda_make_array(const_cast<float *>(&l.fp8_tucker_q_scale_host), 1);
		l.fp8_tucker_k_amax_gpu     = cuda_make_array(nullptr, 1);
		l.fp8_tucker_k_scale_gpu    = cuda_make_array(const_cast<float *>(&l.fp8_tucker_k_scale_host), 1);
		l.fp8_tucker_attn_amax_gpu  = cuda_make_array(nullptr, 1);
		l.fp8_tucker_attn_scale_gpu = cuda_make_array(const_cast<float *>(&l.fp8_tucker_attn_scale_host), 1);
		l.fp8_tucker_v_amax_gpu     = cuda_make_array(nullptr, 1);
		l.fp8_tucker_v_scale_gpu    = cuda_make_array(const_cast<float *>(&l.fp8_tucker_v_scale_host), 1);

		l.fp8_tucker_scores_out_gpu  = cuda_make_array(nullptr, static_cast<size_t>(batch) * key_pad * T);
		l.fp8_tucker_context_out_gpu = cuda_make_array(nullptr, static_cast<size_t>(batch) * D * T);

		Darknet::Fp8GemmSpec scores_spec;
		scores_spec.output_rows = key_pad;
		scores_spec.output_cols = T;
		scores_spec.reduction = D;
		scores_spec.reduction_pad = Darknet::fp8_round_up_to_16(D);
		scores_spec.batch = batch;
		scores_spec.batch_a = true;
		l.fp8_tucker_scores_gemm_plan = Darknet::fp8_gemm_plan_create_ex(scores_spec, l.fp8_tucker_k_scale_gpu, l.fp8_tucker_q_scale_gpu);

		Darknet::Fp8GemmSpec context_spec;
		context_spec.output_rows = D;
		context_spec.output_cols = T;
		context_spec.reduction = key_pad;
		context_spec.reduction_pad = key_pad;
		context_spec.batch = batch;
		context_spec.batch_a = true;
		l.fp8_tucker_context_gemm_plan = Darknet::fp8_gemm_plan_create_ex(context_spec, l.fp8_tucker_v_scale_gpu, l.fp8_tucker_attn_scale_gpu);

		const size_t lt_workspace_bytes = Darknet::fp8_gemm_workspace_bytes();
		if (lt_workspace_bytes > 0)
		{
			CHECK_CUDA(cudaMalloc(&l.fp8_tucker_lt_workspace_gpu, lt_workspace_bytes));
		}
	}

	void free_tucker_fp8_workspace(Darknet::Layer &l)
	{
		Darknet::fp8_release_device_ptr(l.fp8_tucker_q_gpu);
		Darknet::fp8_release_device_ptr(l.fp8_tucker_k_gpu);
		Darknet::fp8_release_device_ptr(l.fp8_tucker_attn_gpu);
		Darknet::fp8_release_device_ptr(l.fp8_tucker_v_t_gpu);
		Darknet::fp8_release_cuda_alloc(l.fp8_tucker_q_amax_gpu);
		Darknet::fp8_release_cuda_alloc(l.fp8_tucker_q_scale_gpu);
		Darknet::fp8_release_cuda_alloc(l.fp8_tucker_k_amax_gpu);
		Darknet::fp8_release_cuda_alloc(l.fp8_tucker_k_scale_gpu);
		Darknet::fp8_release_cuda_alloc(l.fp8_tucker_attn_amax_gpu);
		Darknet::fp8_release_cuda_alloc(l.fp8_tucker_attn_scale_gpu);
		Darknet::fp8_release_cuda_alloc(l.fp8_tucker_v_amax_gpu);
		Darknet::fp8_release_cuda_alloc(l.fp8_tucker_v_scale_gpu);
		Darknet::fp8_release_cuda_alloc(l.fp8_tucker_scores_out_gpu);
		Darknet::fp8_release_cuda_alloc(l.fp8_tucker_context_out_gpu);
		Darknet::fp8_release_device_ptr(l.fp8_tucker_lt_workspace_gpu);
		Darknet::fp8_release_plan(l.fp8_tucker_scores_gemm_plan, Darknet::fp8_gemm_plan_destroy);
		Darknet::fp8_release_plan(l.fp8_tucker_context_gemm_plan, Darknet::fp8_gemm_plan_destroy);
	}
#endif

#ifdef DARKNET_HAS_FP4
	/// Opt-in, additive, diagnostic-only FP4 for the same two GEMMs as the FP8 path
	/// (see darknet_layers.hpp's fp4_tucker_* field comments).  Unlike FP8,
	/// fp4_gemm_execute() quantizes its FP32 inputs internally per call (no external
	/// scale pointer, no persistent quantized buffer), and only two plan *shapes*
	/// exist across all six GEMM roles (forward + the four attention backward GEMMs),
	/// so only two Fp4GemmPlans are created regardless of training/inference.
	void allocate_tucker_fp4_workspace(Darknet::Layer &l, const int windows, const int heads, const int T, const int D)
	{
		const int key_pad = ((T + 15) / 16) * 16;
		const int batch = heads * windows;
		const int pad_dim = std::max({T, D, key_pad});

		l.fp4_tucker_a_gpu   = cuda_make_array(nullptr, static_cast<size_t>(batch) * pad_dim * pad_dim);
		l.fp4_tucker_b_gpu   = cuda_make_array(nullptr, static_cast<size_t>(batch) * pad_dim * pad_dim);
		l.fp4_tucker_out_gpu = cuda_make_array(nullptr, static_cast<size_t>(batch) * pad_dim * pad_dim);

		// Shared across forward Q@K^T and backward dContext@V^T (dAttn): both are (T, T, D).
		l.fp4_tucker_scores_gemm_plan = Darknet::fp4_gemm_plan_create({1, T, T, D, false});
		// Shared across forward attn@V, backward dV/dQ/dK: all are (T, D, key_pad).
		l.fp4_tucker_context_gemm_plan = Darknet::fp4_gemm_plan_create({1, T, D, key_pad, false});

		if (l.fp4_tucker_scores_gemm_plan)
		{
			const size_t bytes = Darknet::fp4_gemm_workspace_bytes(static_cast<Darknet::Fp4GemmPlan *>(l.fp4_tucker_scores_gemm_plan));
			if (bytes > 0) CHECK_CUDA(cudaMalloc(&l.fp4_tucker_scores_lt_workspace, bytes));
		}
		if (l.fp4_tucker_context_gemm_plan)
		{
			const size_t bytes = Darknet::fp4_gemm_workspace_bytes(static_cast<Darknet::Fp4GemmPlan *>(l.fp4_tucker_context_gemm_plan));
			if (bytes > 0) CHECK_CUDA(cudaMalloc(&l.fp4_tucker_context_lt_workspace, bytes));
		}
	}

	void free_tucker_fp4_workspace(Darknet::Layer &l)
	{
		if (l.fp4_tucker_a_gpu)   { cuda_free(l.fp4_tucker_a_gpu);   l.fp4_tucker_a_gpu = nullptr; }
		if (l.fp4_tucker_b_gpu)   { cuda_free(l.fp4_tucker_b_gpu);   l.fp4_tucker_b_gpu = nullptr; }
		if (l.fp4_tucker_out_gpu) { cuda_free(l.fp4_tucker_out_gpu); l.fp4_tucker_out_gpu = nullptr; }
		if (l.fp4_tucker_scores_lt_workspace)  { CHECK_CUDA(cudaFree(l.fp4_tucker_scores_lt_workspace));  l.fp4_tucker_scores_lt_workspace = nullptr; }
		if (l.fp4_tucker_context_lt_workspace) { CHECK_CUDA(cudaFree(l.fp4_tucker_context_lt_workspace)); l.fp4_tucker_context_lt_workspace = nullptr; }
		if (l.fp4_tucker_scores_gemm_plan)
		{
			Darknet::fp4_gemm_plan_destroy(static_cast<Darknet::Fp4GemmPlan *>(l.fp4_tucker_scores_gemm_plan));
			l.fp4_tucker_scores_gemm_plan = nullptr;
		}
		if (l.fp4_tucker_context_gemm_plan)
		{
			Darknet::fp4_gemm_plan_destroy(static_cast<Darknet::Fp4GemmPlan *>(l.fp4_tucker_context_gemm_plan));
			l.fp4_tucker_context_gemm_plan = nullptr;
		}
	}
#endif

	void allocate_tucker_gpu_workspace(Darknet::Layer &l)
	{
		const size_t windows = tucker_window_count(l);
		const size_t T = static_cast<size_t>(l.tucker_window_size) * l.tucker_window_size;
		const size_t HD = static_cast<size_t>(l.tucker_heads) * l.tucker_head_dim;
		const size_t head_count = windows * T * HD;
		const size_t score_count = windows * l.tucker_heads * T * T;
#ifdef DARKNET_HAS_FP8
		allocate_tucker_fp8_workspace(l, static_cast<int>(windows), l.tucker_heads, static_cast<int>(T), l.tucker_head_dim);
#endif
#ifdef DARKNET_HAS_FP4
		allocate_tucker_fp4_workspace(l, static_cast<int>(windows), l.tucker_heads, static_cast<int>(T), l.tucker_head_dim);
#endif

		// Float scratch: windowed tokens + o_mix + d_o_mix.
		l.tucker_windowed_input_gpu = cuda_make_array(nullptr, windows * T * (l.c + 2 * l.tucker_rank_o));
		l.tucker_q_latent_gpu = cuda_make_array(nullptr, 2 * windows * T * l.tucker_rank_q);
		l.tucker_k_latent_gpu = cuda_make_array(nullptr, 2 * windows * T * l.tucker_rank_k);
		l.tucker_v_latent_gpu = cuda_make_array(nullptr, 2 * windows * T * l.tucker_rank_v);

#if defined(CUDNN_HALF) && !defined(DARKNET_GPU_ROCM)
		// These Layer members are typed as float* in Darknet, but are used as raw half buffers here.
		// tucker_windowed_input_gpu has enough bytes for FP16 tokens, o_mix, output/delta, and d_o_mix.
		l.tucker_q_gpu = cuda_make_array(nullptr, (2 * head_count + 1) / 2 + 1);
		l.tucker_k_gpu = cuda_make_array(nullptr, (2 * head_count + 1) / 2 + 1);
		l.tucker_v_gpu = cuda_make_array(nullptr, (2 * head_count + 1) / 2 + 1);
		l.tucker_scores_gpu = cuda_make_array(nullptr, (3 * score_count + 1) / 2 + 1);
		l.tucker_context_gpu = cuda_make_array(nullptr, (2 * head_count + 1) / 2 + 1);
#else
		l.tucker_q_gpu = cuda_make_array(nullptr, 2 * head_count);
		l.tucker_k_gpu = cuda_make_array(nullptr, 2 * head_count);
		l.tucker_v_gpu = cuda_make_array(nullptr, 2 * head_count);
#ifdef CUDNN
		l.tucker_scores_gpu = cuda_make_array(nullptr, 3 * score_count);
#else
		l.tucker_scores_gpu = cuda_make_array(nullptr, score_count);
#endif
		l.tucker_context_gpu = cuda_make_array(nullptr, 2 * head_count);
#endif
	}

	void free_tucker_gpu_workspace(Darknet::Layer &l)
	{
#ifdef DARKNET_HAS_FP8
		free_tucker_fp8_workspace(l);
#endif
#ifdef DARKNET_HAS_FP4
		free_tucker_fp4_workspace(l);
#endif
		cuda_free(l.tucker_windowed_input_gpu);
		cuda_free(l.tucker_q_latent_gpu);
		cuda_free(l.tucker_k_latent_gpu);
		cuda_free(l.tucker_v_latent_gpu);
		cuda_free(l.tucker_q_gpu);
		cuda_free(l.tucker_k_gpu);
		cuda_free(l.tucker_v_gpu);
		cuda_free(l.tucker_scores_gpu);
		cuda_free(l.tucker_context_gpu);
		l.tucker_windowed_input_gpu = nullptr;
		l.tucker_q_latent_gpu = nullptr;
		l.tucker_k_latent_gpu = nullptr;
		l.tucker_v_latent_gpu = nullptr;
		l.tucker_q_gpu = nullptr;
		l.tucker_k_gpu = nullptr;
		l.tucker_v_gpu = nullptr;
		l.tucker_scores_gpu = nullptr;
		l.tucker_context_gpu = nullptr;
	}

#ifdef CUDNN
	void create_tucker_attention_cudnn_tensors(Darknet::Layer *l)
	{
		if (l->srcTensorDesc == nullptr) CHECK_CUDNN(cudnnCreateTensorDescriptor(&l->srcTensorDesc));
		if (l->dstTensorDesc == nullptr) CHECK_CUDNN(cudnnCreateTensorDescriptor(&l->dstTensorDesc));
#ifdef CUDNN_HALF
		if (l->srcTensorDesc16 == nullptr) CHECK_CUDNN(cudnnCreateTensorDescriptor(&l->srcTensorDesc16));
		if (l->dstTensorDesc16 == nullptr) CHECK_CUDNN(cudnnCreateTensorDescriptor(&l->dstTensorDesc16));
		if (l->dsrcTensorDesc16 == nullptr) CHECK_CUDNN(cudnnCreateTensorDescriptor(&l->dsrcTensorDesc16));
		if (l->ddstTensorDesc16 == nullptr) CHECK_CUDNN(cudnnCreateTensorDescriptor(&l->ddstTensorDesc16));
#endif
	}

	void cudnn_tucker_attention_setup(Darknet::Layer *l)
	{
		const int M = l->tucker_window_size;
		const int T = M * M;
		const int win_h = (l->h + M - 1) / M;
		const int win_w = (l->w + M - 1) / M;
		const int rows = l->batch * win_h * win_w * l->tucker_heads * T;
		CHECK_CUDNN(cudnnSetTensor4dDescriptor(l->srcTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, rows, T, 1, 1));
		CHECK_CUDNN(cudnnSetTensor4dDescriptor(l->dstTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, rows, T, 1, 1));
#ifdef CUDNN_HALF
		CHECK_CUDNN(cudnnSetTensor4dDescriptor(l->srcTensorDesc16, CUDNN_TENSOR_NCHW, CUDNN_DATA_HALF, rows, T, 1, 1));
		CHECK_CUDNN(cudnnSetTensor4dDescriptor(l->dstTensorDesc16, CUDNN_TENSOR_NCHW, CUDNN_DATA_HALF, rows, T, 1, 1));
		CHECK_CUDNN(cudnnSetTensor4dDescriptor(l->dsrcTensorDesc16, CUDNN_TENSOR_NCHW, CUDNN_DATA_HALF, rows, T, 1, 1));
		CHECK_CUDNN(cudnnSetTensor4dDescriptor(l->ddstTensorDesc16, CUDNN_TENSOR_NCHW, CUDNN_DATA_HALF, rows, T, 1, 1));
#endif
	}
#endif
#endif

	void project_latent(const float *tokens, const float *basis, float *latent, int T, int C, int R)
	{
		for (int t = 0; t < T; ++t)
		{
			for (int r = 0; r < R; ++r)
			{
				float sum = 0.0f;
				for (int c = 0; c < C; ++c)
				{
					sum += tokens[t * C + c] * basis[c * R + r];
				}
				latent[t * R + r] = sum;
			}
		}
	}

	void expand_heads(const float *latent, const float *core, float *out, int T, int H, int R, int D)
	{
		for (int t = 0; t < T; ++t)
		{
			for (int h = 0; h < H; ++h)
			{
				for (int d = 0; d < D; ++d)
				{
					float sum = 0.0f;
					for (int r = 0; r < R; ++r)
					{
						sum += latent[t * R + r] * core[(h * R + r) * D + d];
					}
					out[(t * H + h) * D + d] = sum;
				}
			}
		}
	}

	void softmax_in_place(float *x, int n)
	{
		float m = x[0];
		for (int i = 1; i < n; ++i) m = std::max(m, x[i]);
		float s = 0.0f;
		for (int i = 0; i < n; ++i)
		{
			x[i] = std::exp(x[i] - m);
			s += x[i];
		}
		const float inv = s > 0.0f ? 1.0f / s : 0.0f;
		for (int i = 0; i < n; ++i) x[i] *= inv;
	}
}

Darknet::Layer make_tucker_attention_layer(int batch, int h, int w, int c, int n,
	int size, int heads, int rank_q, int rank_k, int rank_v, int rank_o,
	ACTIVATION activation, int index, int train)
{
	TAT(TATPARMS);

	if (size < 1) darknet_fatal_error(DARKNET_LOC, "tucker_attention: size must be >= 1, got %d", size);
	if (heads < 1) darknet_fatal_error(DARKNET_LOC, "tucker_attention: heads must be >= 1, got %d", heads);
	if (c % heads != 0) darknet_fatal_error(DARKNET_LOC, "tucker_attention: channels (%d) must be divisible by heads (%d)", c, heads);
	if (n != c) darknet_fatal_error(DARKNET_LOC, "tucker_attention: first implementation requires filters (%d) == input channels (%d)", n, c);
	if (rank_q < 1 || rank_k < 1 || rank_v < 1 || rank_o < 1)
	{
		darknet_fatal_error(DARKNET_LOC, "tucker_attention: all ranks must be >= 1");
	}

	Darknet::Layer l = {};
	l.type = Darknet::ELayerType::TUCKER_ATTENTION;
	l.batch = batch;
	l.h = h;
	l.w = w;
	l.c = c;
	l.n = n;
	l.out_h = h;
	l.out_w = w;
	l.out_c = n;
	l.inputs = h * w * c;
	l.outputs = h * w * n;
	l.index = index;
	l.train = train;
	l.activation = activation;
	l.size = size;
	l.tucker_heads = heads;
	l.tucker_head_dim = c / heads;
	l.tucker_rank_q = rank_q;
	l.tucker_rank_k = rank_k;
	l.tucker_rank_v = rank_v;
	l.tucker_rank_o = rank_o;
	l.tucker_window_size = size;
	l.tucker_pad_h = pad_to(h, size);
	l.tucker_pad_w = pad_to(w, size);

	const TuckerOffsets off = offsets(l);
	l.nweights = static_cast<int>(off.total);
	l.nbiases = n;
	l.weights = (float*)xcalloc(l.nweights, sizeof(float));
	l.weight_updates = (float*)xcalloc(l.nweights, sizeof(float));
	l.biases = (float*)xcalloc(n, sizeof(float));
	l.bias_updates = (float*)xcalloc(n, sizeof(float));

	const float scale = std::sqrt(6.0f / static_cast<float>(c + n));
	rand_uniform_many_weight_init(l.weights, l.nweights, -scale, scale);

	const int T = size * size;
	l.output = (float*)xcalloc(batch * l.outputs, sizeof(float));
	l.delta = (float*)xcalloc(batch * l.outputs, sizeof(float));
	l.tucker_windowed_input = (float*)xcalloc(3 * T * c + 2 * rank_o, sizeof(float));
	l.tucker_q_latent = (float*)xcalloc(2 * T * rank_q, sizeof(float));
	l.tucker_k_latent = (float*)xcalloc(2 * T * rank_k, sizeof(float));
	l.tucker_v_latent = (float*)xcalloc(2 * T * rank_v, sizeof(float));
	l.tucker_q = (float*)xcalloc(2 * T * heads * l.tucker_head_dim, sizeof(float));
	l.tucker_k = (float*)xcalloc(2 * T * heads * l.tucker_head_dim, sizeof(float));
	l.tucker_v = (float*)xcalloc(2 * T * heads * l.tucker_head_dim, sizeof(float));
	l.tucker_scores = (float*)xcalloc(3 * heads * T * T, sizeof(float));
	l.tucker_context = (float*)xcalloc(2 * T * heads * l.tucker_head_dim, sizeof(float));

	l.forward = forward_tucker_attention_layer;
	l.backward = backward_tucker_attention_layer;
	l.update = update_tucker_attention_layer;

#ifdef DARKNET_GPU
	l.output_gpu = cuda_make_array(l.output, static_cast<size_t>(batch) * l.outputs);
	l.delta_gpu = cuda_make_array(l.delta, static_cast<size_t>(batch) * l.outputs);
	l.weights_gpu = cuda_make_array(l.weights, l.nweights);
	l.weight_updates_gpu = cuda_make_array(l.weight_updates, l.nweights);
#if defined(CUDNN) && defined(CUDNN_HALF) && !defined(DARKNET_GPU_ROCM)
	l.weights_gpu16 = cuda_make_array(nullptr, l.nweights / 2 + 1);
	cuda_convert_f32_to_cudnn_16bit(l.weights_gpu, l.nweights, l.weights_gpu16, DARKNET_CUDNN_16BIT_HALF);
#endif
	l.biases_gpu = cuda_make_array(l.biases, n);
	l.bias_updates_gpu = cuda_make_array(l.bias_updates, n);
	allocate_tucker_gpu_workspace(l);
#ifdef CUDNN
	create_tucker_attention_cudnn_tensors(&l);
	cudnn_tucker_attention_setup(&l);
#endif
	l.tucker_gpu_input_cpu = (float*)xcalloc(static_cast<size_t>(batch) * l.inputs, sizeof(float));
	l.forward_gpu = forward_tucker_attention_layer_gpu;
	l.backward_gpu = backward_tucker_attention_layer_gpu;
	l.update_gpu = update_tucker_attention_layer_gpu;
#endif

	l.bflops = static_cast<float>(batch) * h * w * (rank_q + rank_k + rank_v + rank_o) * c / 1000000000.0f;
	*cfg_and_state.output << "tucker_attention " << h << " x " << w << " x " << c
		<< " -> " << n << " (heads=" << heads
		<< ", ranks=" << rank_q << "/" << rank_k << "/" << rank_v << "/" << rank_o
		<< ", window=" << size << ")" << std::endl;
	return l;
}

void resize_tucker_attention_layer(Darknet::Layer *l, int w, int h)
{
	TAT(TATPARMS);

	l->w = w;
	l->h = h;
	l->out_w = w;
	l->out_h = h;
	l->inputs = h * w * l->c;
	l->outputs = h * w * l->n;
	l->tucker_pad_h = pad_to(h, l->tucker_window_size);
	l->tucker_pad_w = pad_to(w, l->tucker_window_size);
	l->output = (float*)xrealloc(l->output, static_cast<size_t>(l->batch) * l->outputs * sizeof(float));
	if (l->train) l->delta = (float*)xrealloc(l->delta, static_cast<size_t>(l->batch) * l->outputs * sizeof(float));
#ifdef DARKNET_GPU
	cuda_free(l->output_gpu);
	l->output_gpu = cuda_make_array(l->output, static_cast<size_t>(l->batch) * l->outputs);
	if (l->train)
	{
		cuda_free(l->delta_gpu);
		l->delta_gpu = cuda_make_array(l->delta, static_cast<size_t>(l->batch) * l->outputs);
	}
	free_tucker_gpu_workspace(*l);
	allocate_tucker_gpu_workspace(*l);
#ifdef CUDNN
	cudnn_tucker_attention_setup(l);
#endif
	l->tucker_gpu_input_cpu = (float*)xrealloc(l->tucker_gpu_input_cpu, static_cast<size_t>(l->batch) * l->inputs * sizeof(float));
#endif
}

void forward_tucker_attention_layer(Darknet::Layer & l, Darknet::NetworkState state)
{
	TAT(TATPARMS);

	const TuckerOffsets off = offsets(l);
	const float *q_basis = l.weights + off.q_basis;
	const float *k_basis = l.weights + off.k_basis;
	const float *v_basis = l.weights + off.v_basis;
	const float *q_core = l.weights + off.q_core;
	const float *k_core = l.weights + off.k_core;
	const float *v_core = l.weights + off.v_core;
	const float *o_core = l.weights + off.o_core;
	const float *o_basis = l.weights + off.o_basis;

	const int B = l.batch;
	const int H = l.h;
	const int W = l.w;
	const int C = l.c;
	const int N = l.n;
	const int heads = l.tucker_heads;
	const int D = l.tucker_head_dim;
	const int M = l.tucker_window_size;
	const int T = M * M;
	const float inv_sqrt_d = 1.0f / std::sqrt(static_cast<float>(D));

	std::fill(l.output, l.output + static_cast<size_t>(B) * l.outputs, 0.0f);

	for (int b = 0; b < B; ++b)
	{
		const float *input_b = state.input + static_cast<size_t>(b) * l.inputs;
		float *output_b = l.output + static_cast<size_t>(b) * l.outputs;
		for (int wy = 0; wy < H; wy += M)
		{
			for (int wx = 0; wx < W; wx += M)
			{
				std::fill(l.tucker_windowed_input, l.tucker_windowed_input + T * C, 0.0f);
				for (int yy = 0; yy < M; ++yy)
				{
					for (int xx = 0; xx < M; ++xx)
					{
						const int y = wy + yy;
						const int x = wx + xx;
						const int t = yy * M + xx;
						if (y >= H || x >= W) continue;
						for (int c = 0; c < C; ++c)
						{
							l.tucker_windowed_input[t * C + c] = input_b[(c * H + y) * W + x];
						}
					}
				}

				project_latent(l.tucker_windowed_input, q_basis, l.tucker_q_latent, T, C, l.tucker_rank_q);
				project_latent(l.tucker_windowed_input, k_basis, l.tucker_k_latent, T, C, l.tucker_rank_k);
				project_latent(l.tucker_windowed_input, v_basis, l.tucker_v_latent, T, C, l.tucker_rank_v);
				expand_heads(l.tucker_q_latent, q_core, l.tucker_q, T, heads, l.tucker_rank_q, D);
				expand_heads(l.tucker_k_latent, k_core, l.tucker_k, T, heads, l.tucker_rank_k, D);
				expand_heads(l.tucker_v_latent, v_core, l.tucker_v, T, heads, l.tucker_rank_v, D);

				for (int hidx = 0; hidx < heads; ++hidx)
				{
					float *scores = l.tucker_scores + hidx * T * T;
					for (int tq = 0; tq < T; ++tq)
					{
						for (int tk = 0; tk < T; ++tk)
						{
							float s = 0.0f;
							for (int d = 0; d < D; ++d)
							{
								s += l.tucker_q[(tq * heads + hidx) * D + d] * l.tucker_k[(tk * heads + hidx) * D + d];
							}
							scores[tq * T + tk] = s * inv_sqrt_d;
						}
						softmax_in_place(scores + tq * T, T);
					}
				}

				std::fill(l.tucker_context, l.tucker_context + T * heads * D, 0.0f);
				for (int hidx = 0; hidx < heads; ++hidx)
				{
					const float *scores = l.tucker_scores + hidx * T * T;
					for (int tq = 0; tq < T; ++tq)
					{
						for (int tk = 0; tk < T; ++tk)
						{
							const float a = scores[tq * T + tk];
							for (int d = 0; d < D; ++d)
							{
								l.tucker_context[(tq * heads + hidx) * D + d] += a * l.tucker_v[(tk * heads + hidx) * D + d];
							}
						}
					}
				}

				for (int yy = 0; yy < M; ++yy)
				{
					for (int xx = 0; xx < M; ++xx)
					{
						const int y = wy + yy;
						const int x = wx + xx;
						if (y >= H || x >= W) continue;
						const int t = yy * M + xx;
						for (int n = 0; n < N; ++n)
						{
							float sum = l.biases[n];
							for (int r = 0; r < l.tucker_rank_o; ++r)
							{
								float head_mix = 0.0f;
								for (int hidx = 0; hidx < heads; ++hidx)
								{
									for (int d = 0; d < D; ++d)
									{
										head_mix += l.tucker_context[(t * heads + hidx) * D + d] * o_core[(hidx * D + d) * l.tucker_rank_o + r];
									}
								}
								sum += head_mix * o_basis[r * N + n];
							}
							sum += input_b[(n * H + y) * W + x];
							output_b[(n * H + y) * W + x] = sum;
						}
					}
				}
			}
		}
	}
	activate_array_cpu_custom(l.output, B * l.outputs, l.activation);
}

void backward_tucker_attention_layer(Darknet::Layer & l, Darknet::NetworkState state)
{
	TAT(TATPARMS);

	gradient_array(l.output, l.batch * l.outputs, l.activation, l.delta);

	const TuckerOffsets off = offsets(l);
	const float *q_basis = l.weights + off.q_basis;
	const float *k_basis = l.weights + off.k_basis;
	const float *v_basis = l.weights + off.v_basis;
	const float *q_core = l.weights + off.q_core;
	const float *k_core = l.weights + off.k_core;
	const float *v_core = l.weights + off.v_core;
	const float *o_core = l.weights + off.o_core;
	const float *o_basis = l.weights + off.o_basis;
	float *dq_basis = l.weight_updates + off.q_basis;
	float *dk_basis = l.weight_updates + off.k_basis;
	float *dv_basis = l.weight_updates + off.v_basis;
	float *dq_core = l.weight_updates + off.q_core;
	float *dk_core = l.weight_updates + off.k_core;
	float *dv_core = l.weight_updates + off.v_core;
	float *do_core = l.weight_updates + off.o_core;
	float *do_basis = l.weight_updates + off.o_basis;

	const int B = l.batch;
	const int H = l.h;
	const int W = l.w;
	const int C = l.c;
	const int N = l.n;
	const int heads = l.tucker_heads;
	const int D = l.tucker_head_dim;
	const int M = l.tucker_window_size;
	const int T = M * M;
	const float inv_sqrt_d = 1.0f / std::sqrt(static_cast<float>(D));

	float *d_out = l.tucker_windowed_input + T * C;
	float *d_tokens = d_out + T * N;
	float *head_mix = d_tokens + T * C;
	float *d_head_mix = head_mix + l.tucker_rank_o;
	float *d_context = l.tucker_context + T * heads * D;
	float *d_scores = l.tucker_scores + heads * T * T;
	float *d_attn = d_scores + heads * T * T;
	float *d_q = l.tucker_q + T * heads * D;
	float *d_k = l.tucker_k + T * heads * D;
	float *d_v = l.tucker_v + T * heads * D;
	float *d_q_latent = l.tucker_q_latent + T * l.tucker_rank_q;
	float *d_k_latent = l.tucker_k_latent + T * l.tucker_rank_k;
	float *d_v_latent = l.tucker_v_latent + T * l.tucker_rank_v;

	for (int b = 0; b < B; ++b)
	{
		const float *input_b = state.input + static_cast<size_t>(b) * l.inputs;
		const float *delta_b = l.delta + static_cast<size_t>(b) * l.outputs;
		float *state_delta_b = state.delta ? state.delta + static_cast<size_t>(b) * l.inputs : nullptr;

		for (int wy = 0; wy < H; wy += M)
		{
			for (int wx = 0; wx < W; wx += M)
			{
				std::fill(l.tucker_windowed_input, l.tucker_windowed_input + T * C, 0.0f);
				std::fill(d_out, d_out + T * N, 0.0f);
				std::fill(d_tokens, d_tokens + T * C, 0.0f);
				std::fill(d_context, d_context + T * heads * D, 0.0f);
				std::fill(d_scores, d_scores + heads * T * T, 0.0f);
				std::fill(d_attn, d_attn + heads * T * T, 0.0f);
				std::fill(d_q, d_q + T * heads * D, 0.0f);
				std::fill(d_k, d_k + T * heads * D, 0.0f);
				std::fill(d_v, d_v + T * heads * D, 0.0f);
				std::fill(d_q_latent, d_q_latent + T * l.tucker_rank_q, 0.0f);
				std::fill(d_k_latent, d_k_latent + T * l.tucker_rank_k, 0.0f);
				std::fill(d_v_latent, d_v_latent + T * l.tucker_rank_v, 0.0f);

				for (int yy = 0; yy < M; ++yy)
				{
					for (int xx = 0; xx < M; ++xx)
					{
						const int y = wy + yy;
						const int x = wx + xx;
						const int t = yy * M + xx;
						if (y >= H || x >= W) continue;
						for (int c = 0; c < C; ++c)
						{
							l.tucker_windowed_input[t * C + c] = input_b[(c * H + y) * W + x];
						}
						for (int n = 0; n < N; ++n)
						{
							d_out[t * N + n] = delta_b[(n * H + y) * W + x];
						}
					}
				}

				project_latent(l.tucker_windowed_input, q_basis, l.tucker_q_latent, T, C, l.tucker_rank_q);
				project_latent(l.tucker_windowed_input, k_basis, l.tucker_k_latent, T, C, l.tucker_rank_k);
				project_latent(l.tucker_windowed_input, v_basis, l.tucker_v_latent, T, C, l.tucker_rank_v);
				expand_heads(l.tucker_q_latent, q_core, l.tucker_q, T, heads, l.tucker_rank_q, D);
				expand_heads(l.tucker_k_latent, k_core, l.tucker_k, T, heads, l.tucker_rank_k, D);
				expand_heads(l.tucker_v_latent, v_core, l.tucker_v, T, heads, l.tucker_rank_v, D);

				for (int hidx = 0; hidx < heads; ++hidx)
				{
					float *scores = l.tucker_scores + hidx * T * T;
					for (int tq = 0; tq < T; ++tq)
					{
						for (int tk = 0; tk < T; ++tk)
						{
							float s = 0.0f;
							for (int d = 0; d < D; ++d)
							{
								s += l.tucker_q[(tq * heads + hidx) * D + d] * l.tucker_k[(tk * heads + hidx) * D + d];
							}
							scores[tq * T + tk] = s * inv_sqrt_d;
						}
						softmax_in_place(scores + tq * T, T);
					}
				}

				std::fill(l.tucker_context, l.tucker_context + T * heads * D, 0.0f);
				for (int hidx = 0; hidx < heads; ++hidx)
				{
					const float *scores = l.tucker_scores + hidx * T * T;
					for (int tq = 0; tq < T; ++tq)
					{
						for (int tk = 0; tk < T; ++tk)
						{
							const float a = scores[tq * T + tk];
							for (int d = 0; d < D; ++d)
							{
								l.tucker_context[(tq * heads + hidx) * D + d] += a * l.tucker_v[(tk * heads + hidx) * D + d];
							}
						}
					}
				}

				for (int t = 0; t < T; ++t)
				{
					for (int n = 0; n < N; ++n)
					{
						const float dy = d_out[t * N + n];
						if (dy == 0.0f) continue;
						l.bias_updates[n] += dy;
						if (n < C) d_tokens[t * C + n] += dy; // residual path

						std::fill(head_mix, head_mix + l.tucker_rank_o, 0.0f);
						for (int r = 0; r < l.tucker_rank_o; ++r)
						{
							for (int hidx = 0; hidx < heads; ++hidx)
							{
								for (int d = 0; d < D; ++d)
								{
									head_mix[r] += l.tucker_context[(t * heads + hidx) * D + d] * o_core[(hidx * D + d) * l.tucker_rank_o + r];
								}
							}
							do_basis[r * N + n] += head_mix[r] * dy;
							d_head_mix[r] = o_basis[r * N + n] * dy;
						}

						for (int r = 0; r < l.tucker_rank_o; ++r)
						{
							for (int hidx = 0; hidx < heads; ++hidx)
							{
								for (int d = 0; d < D; ++d)
								{
									const int hd = hidx * D + d;
									do_core[hd * l.tucker_rank_o + r] += l.tucker_context[(t * heads + hidx) * D + d] * d_head_mix[r];
									d_context[(t * heads + hidx) * D + d] += o_core[hd * l.tucker_rank_o + r] * d_head_mix[r];
								}
							}
						}
					}
				}

				for (int hidx = 0; hidx < heads; ++hidx)
				{
					const float *attn = l.tucker_scores + hidx * T * T;
					float *d_attn_h = d_attn + hidx * T * T;
					float *d_scores_h = d_scores + hidx * T * T;
					for (int tq = 0; tq < T; ++tq)
					{
						for (int tk = 0; tk < T; ++tk)
						{
							for (int d = 0; d < D; ++d)
							{
								d_attn_h[tq * T + tk] += d_context[(tq * heads + hidx) * D + d] * l.tucker_v[(tk * heads + hidx) * D + d];
								d_v[(tk * heads + hidx) * D + d] += attn[tq * T + tk] * d_context[(tq * heads + hidx) * D + d];
							}
						}
						float row_dot = 0.0f;
						for (int tk = 0; tk < T; ++tk) row_dot += d_attn_h[tq * T + tk] * attn[tq * T + tk];
						for (int tk = 0; tk < T; ++tk) d_scores_h[tq * T + tk] = attn[tq * T + tk] * (d_attn_h[tq * T + tk] - row_dot);
					}

					for (int tq = 0; tq < T; ++tq)
					{
						for (int tk = 0; tk < T; ++tk)
						{
							const float ds = d_scores_h[tq * T + tk] * inv_sqrt_d;
							for (int d = 0; d < D; ++d)
							{
								d_q[(tq * heads + hidx) * D + d] += ds * l.tucker_k[(tk * heads + hidx) * D + d];
								d_k[(tk * heads + hidx) * D + d] += ds * l.tucker_q[(tq * heads + hidx) * D + d];
							}
						}
					}
				}

				for (int t = 0; t < T; ++t)
				{
					for (int hidx = 0; hidx < heads; ++hidx)
					{
						for (int d = 0; d < D; ++d)
						{
							const float dq = d_q[(t * heads + hidx) * D + d];
							for (int r = 0; r < l.tucker_rank_q; ++r)
							{
								dq_core[(hidx * l.tucker_rank_q + r) * D + d] += l.tucker_q_latent[t * l.tucker_rank_q + r] * dq;
								d_q_latent[t * l.tucker_rank_q + r] += q_core[(hidx * l.tucker_rank_q + r) * D + d] * dq;
							}

							const float dk = d_k[(t * heads + hidx) * D + d];
							for (int r = 0; r < l.tucker_rank_k; ++r)
							{
								dk_core[(hidx * l.tucker_rank_k + r) * D + d] += l.tucker_k_latent[t * l.tucker_rank_k + r] * dk;
								d_k_latent[t * l.tucker_rank_k + r] += k_core[(hidx * l.tucker_rank_k + r) * D + d] * dk;
							}

							const float dv = d_v[(t * heads + hidx) * D + d];
							for (int r = 0; r < l.tucker_rank_v; ++r)
							{
								dv_core[(hidx * l.tucker_rank_v + r) * D + d] += l.tucker_v_latent[t * l.tucker_rank_v + r] * dv;
								d_v_latent[t * l.tucker_rank_v + r] += v_core[(hidx * l.tucker_rank_v + r) * D + d] * dv;
							}
						}
					}

					for (int c = 0; c < C; ++c)
					{
						for (int r = 0; r < l.tucker_rank_q; ++r)
						{
							dq_basis[c * l.tucker_rank_q + r] += l.tucker_windowed_input[t * C + c] * d_q_latent[t * l.tucker_rank_q + r];
							d_tokens[t * C + c] += q_basis[c * l.tucker_rank_q + r] * d_q_latent[t * l.tucker_rank_q + r];
						}
						for (int r = 0; r < l.tucker_rank_k; ++r)
						{
							dk_basis[c * l.tucker_rank_k + r] += l.tucker_windowed_input[t * C + c] * d_k_latent[t * l.tucker_rank_k + r];
							d_tokens[t * C + c] += k_basis[c * l.tucker_rank_k + r] * d_k_latent[t * l.tucker_rank_k + r];
						}
						for (int r = 0; r < l.tucker_rank_v; ++r)
						{
							dv_basis[c * l.tucker_rank_v + r] += l.tucker_windowed_input[t * C + c] * d_v_latent[t * l.tucker_rank_v + r];
							d_tokens[t * C + c] += v_basis[c * l.tucker_rank_v + r] * d_v_latent[t * l.tucker_rank_v + r];
						}
					}
				}

				if (state_delta_b)
				{
					for (int yy = 0; yy < M; ++yy)
					{
						for (int xx = 0; xx < M; ++xx)
						{
							const int y = wy + yy;
							const int x = wx + xx;
							const int t = yy * M + xx;
							if (y >= H || x >= W) continue;
							for (int c = 0; c < C; ++c)
							{
								state_delta_b[(c * H + y) * W + x] += d_tokens[t * C + c];
							}
						}
					}
				}
			}
		}
	}
}

void update_tucker_attention_layer(Darknet::Layer & l, int batch, float learning_rate, float momentum, float decay)
{
	TAT(TATPARMS);

	axpy_cpu(l.nweights, -decay * batch, l.weights, 1, l.weight_updates, 1);
	axpy_cpu(l.nweights, learning_rate / batch, l.weight_updates, 1, l.weights, 1);
	scal_cpu(l.nweights, momentum, l.weight_updates, 1);

	axpy_cpu(l.nbiases, learning_rate / batch, l.bias_updates, 1, l.biases, 1);
	scal_cpu(l.nbiases, momentum, l.bias_updates, 1);
}
