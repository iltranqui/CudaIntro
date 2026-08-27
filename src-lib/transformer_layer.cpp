#include "transformer_layer.hpp"
#include "blas.hpp"
#include "gemm.hpp"
#include "activations.hpp"
#include "utils.hpp"
#include "dark_cuda.hpp"

#include <cmath>
#include <cstring>
#include <algorithm>
#include <cstdio>

namespace
{
	static auto & cfg_and_state = Darknet::CfgAndState::get();

	static void check_nan_cpu(const char *step_name, const float *arr, int n, int layer_idx)
	{
		if (arr == nullptr || n <= 0)
		{
			return;
		}

		for (int i = 0; i < n; ++i)
		{
			const float value = arr[i];
			if (std::isnan(value) || std::isinf(value))
			{
				const std::string layer_label = Darknet::layer_type_diagnostic_label(Darknet::ELayerType::TRANSFORMER);
				std::printf("[%s layer] NaN/Inf detected at layer %d, step: %s (idx=%d, value=%g)\n",
					layer_label.c_str(), layer_idx, step_name, i, static_cast<double>(value));
				return;
			}
		}
	}
}

// ─── mHC residual mixer helpers ─────────────────────────────────────────────
// Conservative local adaptation of "mHC: Manifold-Constrained Hyper-Connections".
// The paper constrains residual mappings to the Birkhoff polytope with
// Sinkhorn-Knopp.  For the two streams available at each existing residual merge
// (skip stream + branch stream), the Sinkhorn projection has this closed form:
//     [[p, 1-p], [1-p, p]], where p = sigmoid((a00+a11-a01-a10)/2).
// This keeps the merge non-negative and exactly doubly stochastic without
// changing Darknet tensor shapes.
static constexpr int MHC_RESIDUAL_SITES = 2;
static constexpr int MHC_PARAMS_PER_SITE = 6;   // 4 residual logits + 2 post-scale logits
static constexpr int MHC_PARAM_COUNT = MHC_RESIDUAL_SITES * MHC_PARAMS_PER_SITE;
static constexpr float MHC_IDENTITY_LOGIT = 4.0f;
static constexpr float MHC_SKIP_SCALE_LOGIT = 0.0f;
static constexpr float MHC_BRANCH_SCALE_LOGIT = -8.0f;
static constexpr float MHC_PARAM_CLAMP = 8.0f;

static inline float mhc_sigmoid(float x)
{
	if (!std::isfinite(x)) return 0.5f;
	if (x >= 0.0f)
	{
		const float z = expf(-x);
		return 1.0f / (1.0f + z);
	}
	const float z = expf(x);
	return z / (1.0f + z);
}

static inline float mhc_post_scale(float raw)
{
	return 2.0f * mhc_sigmoid(raw);
}

static void init_mhc_residual_params(float *params)
{
	if (!params) return;
	for (int s = 0; s < MHC_RESIDUAL_SITES; ++s)
	{
		const int o = s * MHC_PARAMS_PER_SITE;
		params[o + 0] = MHC_IDENTITY_LOGIT;
		params[o + 1] = -MHC_IDENTITY_LOGIT;
		params[o + 2] = -MHC_IDENTITY_LOGIT;
		params[o + 3] = MHC_IDENTITY_LOGIT;
		params[o + 4] = MHC_SKIP_SCALE_LOGIT;
		params[o + 5] = MHC_BRANCH_SCALE_LOGIT;
	}
}

static void sanitize_and_constrain_mhc_params_cpu(float *params)
{
	if (!params) return;
	for (int i = 0; i < MHC_PARAM_COUNT; ++i)
	{
		if (!std::isfinite(params[i]))
		{
			params[i] = 0.0f;
		}
		else
		{
			params[i] = std::max(-MHC_PARAM_CLAMP, std::min(MHC_PARAM_CLAMP, params[i]));
		}
	}
}

static inline void mhc_residual_coefficients(const float *params, int site,
	float &skip_coeff, float &branch_coeff, float &p, float &post_skip, float &post_branch)
{
	const int o = site * MHC_PARAMS_PER_SITE;
	const float z = 0.5f * (params[o + 0] + params[o + 3] - params[o + 1] - params[o + 2]);
	p = mhc_sigmoid(z);
	post_skip = mhc_post_scale(params[o + 4]);
	post_branch = mhc_post_scale(params[o + 5]);
	skip_coeff = post_skip * p + post_branch * (1.0f - p);
	branch_coeff = post_skip * (1.0f - p) + post_branch * p;
}

static void mhc_residual_forward_cpu(const float *skip, const float *branch, float *out,
	size_t count, const float *params, int site)
{
	if (!params)
	{
		for (size_t i = 0; i < count; ++i) out[i] = skip[i] + branch[i];
		return;
	}
	float a, b, p, post_a, post_b;
	mhc_residual_coefficients(params, site, a, b, p, post_a, post_b);
	for (size_t i = 0; i < count; ++i)
	{
		out[i] = a * skip[i] + b * branch[i];
	}
}

static void mhc_residual_backward_cpu(const float *skip, const float *branch, const float *dout,
	float *d_skip, float *d_branch, size_t count, const float *params, float *updates, int site)
{
	if (!params)
	{
		for (size_t i = 0; i < count; ++i)
		{
			d_skip[i] = dout[i];
			d_branch[i] = dout[i];
		}
		return;
	}

	float skip_coeff, branch_coeff, p, post_skip, post_branch;
	mhc_residual_coefficients(params, site, skip_coeff, branch_coeff, p, post_skip, post_branch);

	float grad_post_skip = 0.0f;
	float grad_post_branch = 0.0f;
	float grad_p = 0.0f;
	for (size_t i = 0; i < count; ++i)
	{
		const float g = dout[i];
		d_skip[i] = skip_coeff * g;
		d_branch[i] = branch_coeff * g;
		grad_post_skip += g * (p * skip[i] + (1.0f - p) * branch[i]);
		grad_post_branch += g * ((1.0f - p) * skip[i] + p * branch[i]);
		grad_p += g * (post_skip - post_branch) * (skip[i] - branch[i]);
	}

	if (updates)
	{
		const int o = site * MHC_PARAMS_PER_SITE;
		const float dz = 0.5f * grad_p * p * (1.0f - p);
		updates[o + 0] += dz;
		updates[o + 1] -= dz;
		updates[o + 2] -= dz;
		updates[o + 3] += dz;
		updates[o + 4] += grad_post_skip * post_skip * (1.0f - 0.5f * post_skip);
		updates[o + 5] += grad_post_branch * post_branch * (1.0f - 0.5f * post_branch);
	}
}

// ─── helpers ──────────────────────────────────────────────────────────────────

static inline int ceil_div(int a, int b)
{
	return (a + b - 1) / b;
}

static inline int pad_to(int x, int window_size)
{
	return ceil_div(x, window_size) * window_size;
}

/// LayerNorm forward: normalize over the last dim (C) for each token.
/// x: [total_tokens, C], out: [total_tokens, C]
/// mean, var: [total_tokens] — saved for backward
/// xhat: [total_tokens, C] — saved normalized values for backward
/// gamma, beta: [C]
static void layernorm_forward(const float *x, float *out, float *mean, float *var, float *xhat,
	const float *gamma, const float *beta, int total_tokens, int C)
{
	const float eps = 1e-5f;
	for (int i = 0; i < total_tokens; i++)
	{
		const float *xi = x + i * C;
		float *oi = out + i * C;
		float *xh = xhat + i * C;

		float m = 0.0f;
		for (int j = 0; j < C; j++) m += xi[j];
		m /= C;
		mean[i] = m;

		float v = 0.0f;
		for (int j = 0; j < C; j++)
		{
			float d = xi[j] - m;
			v += d * d;
		}
		v /= C;
		var[i] = v;

		float inv_std = 1.0f / sqrtf(v + eps);
		for (int j = 0; j < C; j++)
		{
			xh[j] = (xi[j] - m) * inv_std;
			oi[j] = xh[j] * gamma[j] + beta[j];
		}
	}
}

/// LayerNorm backward: compute dx, dgamma, dbeta
/// dout: [total_tokens, C] — incoming gradient
/// xhat: [total_tokens, C] — saved normalized values
/// var: [total_tokens] — saved variance
/// gamma: [C]
/// dx: [total_tokens, C] — output gradient w.r.t. input
/// dgamma, dbeta: [C] — accumulated gradient w.r.t. params
static void layernorm_backward(const float *dout, const float *xhat, const float *var,
	const float *gamma, float *dx, float *dgamma, float *dbeta, int total_tokens, int C)
{
	const float eps = 1e-5f;

	for (int i = 0; i < total_tokens; i++)
	{
		const float *doi = dout + i * C;
		const float *xhi = xhat + i * C;
		float *dxi = dx + i * C;
		float inv_std = 1.0f / sqrtf(var[i] + eps);

		// accumulate dgamma, dbeta
		for (int j = 0; j < C; j++)
		{
			dgamma[j] += doi[j] * xhi[j];
			dbeta[j] += doi[j];
		}

		// compute dx
		float sum_dxhat = 0.0f;
		float dot_dxhat_xhat = 0.0f;
		for (int j = 0; j < C; j++)
		{
			const float dxhat = doi[j] * gamma[j];
			sum_dxhat += dxhat;
			dot_dxhat_xhat += dxhat * xhi[j];
		}
		for (int j = 0; j < C; j++)
		{
			const float dxhat = doi[j] * gamma[j];
			dxi[j] = inv_std * (dxhat - (sum_dxhat + xhi[j] * dot_dxhat_xhat) / C);
		}
	}
}

/// Window partition: NCHW → [B*nW, T, C]
/// Input is [B, C, Hp, Wp], output is [B * nH * nW, T, C] where T = ws*ws
static void window_partition(const float *input, float *output,
	int B, int C, int Hp, int Wp, int ws)
{
	const int nH = Hp / ws;
	const int nW = Wp / ws;
	const int T = ws * ws;

	for (int b = 0; b < B; b++)
	{
		for (int wh = 0; wh < nH; wh++)
		{
			for (int ww = 0; ww < nW; ww++)
			{
				const int win_idx = b * nH * nW + wh * nW + ww;
				for (int i = 0; i < ws; i++)
				{
					for (int j = 0; j < ws; j++)
					{
						const int token = i * ws + j;
						const int y = wh * ws + i;
						const int x = ww * ws + j;
						for (int c = 0; c < C; c++)
						{
							// input: NCHW = [b, c, y, x]
							const int in_idx = ((b * C + c) * Hp + y) * Wp + x;
							// output: [win_idx, token, c]
							const int out_idx = (win_idx * T + token) * C + c;
							output[out_idx] = input[in_idx];
						}
					}
				}
			}
		}
	}
}

/// Window unpartition: [B*nW, T, C] → NCHW [B, C, Hp, Wp]
static void window_unpartition(const float *input, float *output,
	int B, int C, int Hp, int Wp, int ws)
{
	const int nH = Hp / ws;
	const int nW = Wp / ws;
	const int T = ws * ws;

	for (int b = 0; b < B; b++)
	{
		for (int wh = 0; wh < nH; wh++)
		{
			for (int ww = 0; ww < nW; ww++)
			{
				const int win_idx = b * nH * nW + wh * nW + ww;
				for (int i = 0; i < ws; i++)
				{
					for (int j = 0; j < ws; j++)
					{
						const int token = i * ws + j;
						const int y = wh * ws + i;
						const int x = ww * ws + j;
						for (int c = 0; c < C; c++)
						{
							const int in_idx = (win_idx * T + token) * C + c;
							const int out_idx = ((b * C + c) * Hp + y) * Wp + x;
							output[out_idx] = input[in_idx];
						}
					}
				}
			}
		}
	}
}

/// Cyclic shift: shift NCHW tensor by (dy, dx) with wrap-around
static void cyclic_shift(const float *input, float *output,
	int B, int C, int H, int W, int dy, int dx)
{
	for (int b = 0; b < B; b++)
	{
		for (int c = 0; c < C; c++)
		{
			for (int y = 0; y < H; y++)
			{
				for (int x = 0; x < W; x++)
				{
					int sy = ((y - dy) % H + H) % H;
					int sx = ((x - dx) % W + W) % W;
					int out_idx = ((b * C + c) * H + y) * W + x;
					int in_idx = ((b * C + c) * H + sy) * W + sx;
					output[out_idx] = input[in_idx];
				}
			}
		}
	}
}

/// Softmax over last dimension: x[i, :] of length n
static void softmax_row(float *x, int n)
{
	const float clip = 80.0f;
	for (int i = 0; i < n; i++)
	{
		float &v = x[i];
		if (std::isnan(v))
		{
			v = 0.0f;
		}
		else if (std::isinf(v))
		{
			v = (v > 0.0f) ? clip : -clip;
		}
		else
		{
			v = std::max(-clip, std::min(clip, v));
		}
	}

	float max_val = x[0];
	for (int i = 1; i < n; i++) max_val = std::max(max_val, x[i]);
	float sum = 0.0f;
	for (int i = 0; i < n; i++)
	{
		x[i] = expf(x[i] - max_val);
		sum += x[i];
	}
	if (!std::isfinite(sum) || sum <= 0.0f)
	{
		const float uniform = 1.0f / n;
		for (int i = 0; i < n; i++) x[i] = uniform;
		return;
	}
	float inv_sum = 1.0f / (sum + 1e-9f);
	for (int i = 0; i < n; i++) x[i] *= inv_sum;
}

/// Build attention mask for shifted windows AND padding.
/// mask: [nW_total, T, T] where nW_total = nH*nW, T = ws*ws
/// Tokens from different shift regions get -100.0f.
/// Padded tokens (outside original H×W) get -100.0f in both directions.
static void build_attention_mask(float *mask, int H, int W, int Hp, int Wp, int ws, int shift_size)
{
	const int nH = Hp / ws;
	const int nW = Wp / ws;
	const int T = ws * ws;
	const bool has_padding = (Hp != H || Wp != W);

	if (shift_size == 0 && !has_padding) return; // mask already zeroed

	// Create region map for shifted grid
	// Each pixel gets a region ID; padded pixels get a special ID (-1)
	std::vector<int> region_map(Hp * Wp, 0);

	if (shift_size > 0)
	{
		int region_id = 0;
		const int h_slices[] = {0, Hp - shift_size, Hp};
		const int w_slices[] = {0, Wp - shift_size, Wp};

		for (int hi = 0; hi < 2; hi++)
		{
			for (int wi = 0; wi < 2; wi++)
			{
				for (int y = h_slices[hi]; y < h_slices[hi + 1]; y++)
				{
					for (int x = w_slices[wi]; x < w_slices[wi + 1]; x++)
					{
						region_map[y * Wp + x] = region_id;
					}
				}
				region_id++;
			}
		}
	}

	// Mark padded positions with special region ID -1
	if (has_padding)
	{
		for (int y = 0; y < Hp; y++)
		{
			for (int x = 0; x < Wp; x++)
			{
				if (y >= H || x >= W)
				{
					region_map[y * Wp + x] = -1;
				}
			}
		}
	}

	// For each window, compare region IDs
	for (int wh = 0; wh < nH; wh++)
	{
		for (int ww = 0; ww < nW; ww++)
		{
			const int win_idx = wh * nW + ww;
			for (int ti = 0; ti < T; ti++)
			{
				int yi = wh * ws + ti / ws;
				int xi = ww * ws + ti % ws;
				int ri = region_map[yi * Wp + xi];
				for (int tj = 0; tj < T; tj++)
				{
					int yj = wh * ws + tj / ws;
					int xj = ww * ws + tj % ws;
					int rj = region_map[yj * Wp + xj];
					// Mask if: different regions, or either token is padding
					if (ri == -1 || rj == -1 || ri != rj)
					{
						mask[(win_idx * T + ti) * T + tj] = -100.0f;
					}
				}
			}
		}
	}
}


static void layernorm_affine_from_xhat(const float *xhat, float *out,
	const float *gamma, const float *beta, int total_tokens, int C)
{
	for (int i = 0; i < total_tokens; ++i)
	{
		const float *xhi = xhat + i * C;
		float *oi = out + i * C;
		for (int j = 0; j < C; ++j)
		{
			oi[j] = xhi[j] * gamma[j] + beta[j];
		}
	}
}

static void add_bias_rows(float *buf, const float *bias, int M, int N)
{
	for (int i = 0; i < M; ++i)
	{
		float *row = buf + i * N;
		for (int j = 0; j < N; ++j)
		{
			row[j] += bias[j];
		}
	}
}

template <typename T>
static inline T *workspace_ptr(T *base, size_t offset)
{
	return base + offset;
}

// ─── make ─────────────────────────────────────────────────────────────────────

Darknet::Layer make_transformer_layer(int batch, int h, int w, int c, int n,
	int size, int heads, int shift, int ffn_ratio, ACTIVATION activation, int index, int train)
{
	TAT(TATPARMS);

	Darknet::Layer l = {};
	l.type = Darknet::ELayerType::TRANSFORMER;

	// Validate
	if (c % heads != 0)
	{
		darknet_fatal_error(DARKNET_LOC, "transformer: input channels (%d) must be divisible by heads (%d)", c, heads);
	}
	if (size < 1)
	{
		darknet_fatal_error(DARKNET_LOC, "transformer: window size must be >= 1, got %d", size);
	}

	l.batch = batch;
	l.h = h;
	l.w = w;
	l.c = c;
	l.n = n;	// output channels (filters)
	l.out_h = h;
	l.out_w = w;
	l.out_c = n;
	l.outputs = n * h * w;
	l.inputs = c * h * w;
	l.index = index;
	l.activation = activation;
	l.size = size;
	l.train = train;

	// Transformer-specific config
	l.tf_heads = heads;
	l.tf_head_dim = c / heads;
	l.tf_ffn_ratio = ffn_ratio;
	l.tf_shift = shift;
	l.tf_window_size = size;

	const int Hp = pad_to(h, size);
	const int Wp = pad_to(w, size);
	l.tf_pad_h = Hp;
	l.tf_pad_w = Wp;

	const int T = size * size;
	const int nW = (Hp / size) * (Wp / size);
	const int total_windows = batch * nW;
	const int ffn_hidden = n * ffn_ratio;
	const TransformerWorkspaceLayout workspace = make_transformer_workspace_layout(batch, c, n, Hp, Wp, size, heads, ffn_ratio);

	// --- Allocate weights ---

	// QKV projection: [3*C, C] stored in l.weights, biases in l.biases
	l.nweights = 3 * c * c;
	l.weights = (float*)xcalloc(l.nweights, sizeof(float));
	l.weight_updates = (float*)xcalloc(l.nweights, sizeof(float));
	l.biases = (float*)xcalloc(3 * c, sizeof(float));
	l.bias_updates = (float*)xcalloc(3 * c, sizeof(float));

	// Output projection: [N, C]
	l.tf_wo = (float*)xcalloc(n * c, sizeof(float));
	l.tf_wo_updates = (float*)xcalloc(n * c, sizeof(float));
	l.tf_wo_bias = (float*)xcalloc(n, sizeof(float));
	l.tf_wo_bias_updates = (float*)xcalloc(n, sizeof(float));

	// LayerNorm 1: [C]
	l.tf_ln1_gamma = (float*)xcalloc(c, sizeof(float));
	l.tf_ln1_gamma_updates = (float*)xcalloc(c, sizeof(float));
	l.tf_ln1_beta = (float*)xcalloc(c, sizeof(float));
	l.tf_ln1_beta_updates = (float*)xcalloc(c, sizeof(float));

	// LayerNorm 2: [N]
	l.tf_ln2_gamma = (float*)xcalloc(n, sizeof(float));
	l.tf_ln2_gamma_updates = (float*)xcalloc(n, sizeof(float));
	l.tf_ln2_beta = (float*)xcalloc(n, sizeof(float));
	l.tf_ln2_beta_updates = (float*)xcalloc(n, sizeof(float));

	// FFN: up [ffn_hidden, N], down [N, ffn_hidden]
	l.tf_ffn_w1 = (float*)xcalloc(ffn_hidden * n, sizeof(float));
	l.tf_ffn_w1_updates = (float*)xcalloc(ffn_hidden * n, sizeof(float));
	l.tf_ffn_b1 = (float*)xcalloc(ffn_hidden, sizeof(float));
	l.tf_ffn_b1_updates = (float*)xcalloc(ffn_hidden, sizeof(float));
	l.tf_ffn_w2 = (float*)xcalloc(n * ffn_hidden, sizeof(float));
	l.tf_ffn_w2_updates = (float*)xcalloc(n * ffn_hidden, sizeof(float));
	l.tf_ffn_b2 = (float*)xcalloc(n, sizeof(float));
	l.tf_ffn_b2_updates = (float*)xcalloc(n, sizeof(float));

	// Residual projection when C != N: [N, C] linear (no bias)
	if (c != n)
	{
		l.tf_res_proj = (float*)xcalloc(n * c, sizeof(float));
		l.tf_res_proj_updates = (float*)xcalloc(n * c, sizeof(float));
	}

	// Relative position bias: [heads, (2*size-1)*(2*size-1)]
	const int bias_table_len = (2 * size - 1) * (2 * size - 1);
	l.tf_rel_pos_bias = (float*)xcalloc(heads * bias_table_len, sizeof(float));
	l.tf_rel_pos_bias_updates = (float*)xcalloc(heads * bias_table_len, sizeof(float));

	// Relative position index: [T, T]
	l.tf_rel_pos_index = (int*)xcalloc(T * T, sizeof(int));
	for (int i = 0; i < T; i++)
	{
		int yi = i / size, xi = i % size;
		for (int j = 0; j < T; j++)
		{
			int yj = j / size, xj = j % size;
			int dy = yi - yj + size - 1;
			int dx = xi - xj + size - 1;
			l.tf_rel_pos_index[i * T + j] = dy * (2 * size - 1) + dx;
		}
	}

	// --- Initialize weights ---
	// Xavier uniform for QKV and output projection
	float xavier_qkv = sqrtf(6.0f / (float)(c + 3 * c));
	rand_uniform_many_weight_init(l.weights, l.nweights, -xavier_qkv, xavier_qkv);

	float xavier_wo = sqrtf(6.0f / (float)(c + n));
	rand_uniform_many_weight_init(l.tf_wo, n * c, -xavier_wo, xavier_wo);

	// He uniform for FFN
	float he_ffn1 = sqrtf(6.0f / (float)n);
	rand_uniform_many_weight_init(l.tf_ffn_w1, ffn_hidden * n, -he_ffn1, he_ffn1);

	float he_ffn2 = sqrtf(6.0f / (float)ffn_hidden);
	rand_uniform_many_weight_init(l.tf_ffn_w2, n * ffn_hidden, -he_ffn2, he_ffn2);

	// Residual projection: Xavier uniform
	if (c != n)
	{
		float xavier_res = sqrtf(6.0f / (float)(c + n));
		rand_uniform_many_weight_init(l.tf_res_proj, n * c, -xavier_res, xavier_res);
	}

	// LN gamma = 1.0, beta = 0.0
	for (int i = 0; i < c; i++) l.tf_ln1_gamma[i] = 1.0f;
	for (int i = 0; i < n; i++) l.tf_ln2_gamma[i] = 1.0f;
	// beta already zero from xcalloc

	// Relative position bias: small random
	rand_uniform_many_weight_init(l.tf_rel_pos_bias, heads * bias_table_len, -0.02f, 0.02f);

	// mHC residual mixer parameters.  Reuses generic scale storage because the
	// transformer layer does not use batch-norm scales.
	l.scales = (float*)xcalloc(MHC_PARAM_COUNT, sizeof(float));
	l.scale_updates = (float*)xcalloc(MHC_PARAM_COUNT, sizeof(float));
	init_mhc_residual_params(l.scales);

	// --- Allocate runtime buffers ---
	l.output = (float*)xcalloc(batch * l.outputs, sizeof(float));
	l.delta = (float*)xcalloc(batch * l.outputs, sizeof(float));

	l.tf_qkv_out = (float*)xcalloc(total_windows * T * 3 * c, sizeof(float));
	l.tf_attn_scores = (float*)xcalloc(total_windows * heads * T * T, sizeof(float));
	l.tf_attn_out = (float*)xcalloc(total_windows * T * c, sizeof(float));
	l.tf_ffn_hidden = (float*)xcalloc(total_windows * T * ffn_hidden, sizeof(float));
	l.activation_input = (float*)xcalloc(total_windows * T * ffn_hidden, sizeof(float));
	l.tf_ln1_mean = (float*)xcalloc(total_windows * T, sizeof(float));
	l.tf_ln1_var = (float*)xcalloc(total_windows * T, sizeof(float));
	l.tf_ln2_mean = (float*)xcalloc(total_windows * T, sizeof(float));
	l.tf_ln2_var = (float*)xcalloc(total_windows * T, sizeof(float));
	l.tf_ln1_xhat = (float*)xcalloc(total_windows * T * c, sizeof(float));
	l.tf_ln2_xhat = (float*)xcalloc(total_windows * T * n, sizeof(float));
	l.tf_pre_res2 = (float*)xcalloc(total_windows * T * n, sizeof(float));
	l.tf_windowed_input = (float*)xcalloc(total_windows * T * c, sizeof(float));
	l.x = (float*)xcalloc(total_windows * T * n, sizeof(float));       // mHC residual1 branch cache
	l.x_norm = (float*)xcalloc(total_windows * T * n, sizeof(float));  // mHC residual2 branch/cache
	l.tf_workspace = (float*)xcalloc(workspace.total, sizeof(float));
	l.tf_workspace_size = workspace.total;

	// Attention mask for shifted windows and padding
	l.tf_attn_mask = (float*)xcalloc(nW * T * T, sizeof(float));
	const bool needs_mask = shift || (Hp != h) || (Wp != w);
	if (needs_mask)
	{
		build_attention_mask(l.tf_attn_mask, h, w, Hp, Wp, size, shift ? size / 2 : 0);
	}

	// Function pointers
	l.forward = forward_transformer_layer;
	l.backward = backward_transformer_layer;
	l.update = update_transformer_layer;

#ifdef DARKNET_GPU
	l.forward_gpu = forward_transformer_layer_gpu;
	l.backward_gpu = backward_transformer_layer_gpu;
	l.update_gpu = update_transformer_layer_gpu;

	// GPU weight arrays
	l.weights_gpu = cuda_make_array(l.weights, l.nweights);
	l.weight_updates_gpu = cuda_make_array(l.weight_updates, l.nweights);
	l.biases_gpu = cuda_make_array(l.biases, 3 * c);
	l.bias_updates_gpu = cuda_make_array(l.bias_updates, 3 * c);

	l.tf_wo_gpu = cuda_make_array(l.tf_wo, n * c);
	l.tf_wo_updates_gpu = cuda_make_array(l.tf_wo_updates, n * c);
	l.tf_wo_bias_gpu = cuda_make_array(l.tf_wo_bias, n);
	l.tf_wo_bias_updates_gpu = cuda_make_array(l.tf_wo_bias_updates, n);

	if (c != n)
	{
		l.tf_res_proj_gpu = cuda_make_array(l.tf_res_proj, n * c);
		l.tf_res_proj_updates_gpu = cuda_make_array(l.tf_res_proj_updates, n * c);
	}

	l.tf_ln1_gamma_gpu = cuda_make_array(l.tf_ln1_gamma, c);
	l.tf_ln1_gamma_updates_gpu = cuda_make_array(l.tf_ln1_gamma_updates, c);
	l.tf_ln1_beta_gpu = cuda_make_array(l.tf_ln1_beta, c);
	l.tf_ln1_beta_updates_gpu = cuda_make_array(l.tf_ln1_beta_updates, c);

	l.tf_ln2_gamma_gpu = cuda_make_array(l.tf_ln2_gamma, n);
	l.tf_ln2_gamma_updates_gpu = cuda_make_array(l.tf_ln2_gamma_updates, n);
	l.tf_ln2_beta_gpu = cuda_make_array(l.tf_ln2_beta, n);
	l.tf_ln2_beta_updates_gpu = cuda_make_array(l.tf_ln2_beta_updates, n);

	l.tf_ffn_w1_gpu = cuda_make_array(l.tf_ffn_w1, ffn_hidden * n);
	l.tf_ffn_w1_updates_gpu = cuda_make_array(l.tf_ffn_w1_updates, ffn_hidden * n);
	l.tf_ffn_b1_gpu = cuda_make_array(l.tf_ffn_b1, ffn_hidden);
	l.tf_ffn_b1_updates_gpu = cuda_make_array(l.tf_ffn_b1_updates, ffn_hidden);

	l.tf_ffn_w2_gpu = cuda_make_array(l.tf_ffn_w2, n * ffn_hidden);
	l.tf_ffn_w2_updates_gpu = cuda_make_array(l.tf_ffn_w2_updates, n * ffn_hidden);
	l.tf_ffn_b2_gpu = cuda_make_array(l.tf_ffn_b2, n);
	l.tf_ffn_b2_updates_gpu = cuda_make_array(l.tf_ffn_b2_updates, n);

	l.tf_rel_pos_bias_gpu = cuda_make_array(l.tf_rel_pos_bias, heads * bias_table_len);
	l.tf_rel_pos_bias_updates_gpu = cuda_make_array(l.tf_rel_pos_bias_updates, heads * bias_table_len);
	l.tf_rel_pos_index_gpu = cuda_make_int_array_new_api(l.tf_rel_pos_index, T * T);
	l.scales_gpu = cuda_make_array(l.scales, MHC_PARAM_COUNT);
	l.scale_updates_gpu = cuda_make_array(l.scale_updates, MHC_PARAM_COUNT);

	// GPU runtime buffers
	l.output_gpu = cuda_make_array(l.output, batch * l.outputs);
	l.delta_gpu = cuda_make_array(l.delta, batch * l.outputs);

	l.tf_qkv_out_gpu = cuda_make_array(nullptr, total_windows * T * 3 * c);
	l.tf_attn_scores_gpu = cuda_make_array(nullptr, total_windows * heads * T * T);
	l.tf_attn_out_gpu = cuda_make_array(nullptr, total_windows * T * c);
	l.tf_ffn_hidden_gpu = cuda_make_array(nullptr, total_windows * T * ffn_hidden);
	l.activation_input_gpu = cuda_make_array(nullptr, total_windows * T * ffn_hidden);
	l.tf_ln1_mean_gpu = cuda_make_array(nullptr, total_windows * T);
	l.tf_ln1_var_gpu = cuda_make_array(nullptr, total_windows * T);
	l.tf_ln2_mean_gpu = cuda_make_array(nullptr, total_windows * T);
	l.tf_ln2_var_gpu = cuda_make_array(nullptr, total_windows * T);
	l.tf_ln1_xhat_gpu = cuda_make_array(nullptr, total_windows * T * c);
	l.tf_ln2_xhat_gpu = cuda_make_array(nullptr, total_windows * T * n);
	l.tf_pre_res2_gpu = cuda_make_array(nullptr, total_windows * T * n);
	l.tf_windowed_input_gpu = cuda_make_array(nullptr, total_windows * T * c);
	l.x_gpu = cuda_make_array(nullptr, total_windows * T * n);
	l.x_norm_gpu = cuda_make_array(nullptr, total_windows * T * n);
	l.tf_attn_mask_gpu = cuda_make_array(l.tf_attn_mask, nW * T * T);
	l.tf_gpu_workspace = cuda_make_array(nullptr, workspace.total);
	l.tf_gpu_workspace_size = workspace.total;
#endif

	*cfg_and_state.output
		<< "transformer   " << index
		<< "  " << w << " x " << h << " x " << c
		<< "  ->  " << w << " x " << h << " x " << n
		<< "  heads=" << heads << " win=" << size << " shift=" << shift
		<< "  ffn=" << ffn_ratio << "x"
		<< std::endl;

	return l;
}

// ─── resize ───────────────────────────────────────────────────────────────────

void resize_transformer_layer(Darknet::Layer * l, int w, int h)
{
	TAT(TATPARMS);

	l->h = h;
	l->w = w;
	l->out_h = h;
	l->out_w = w;
	l->outputs = l->n * h * w;
	l->inputs = l->c * h * w;

	const int ws = l->tf_window_size;
	const int Hp = pad_to(h, ws);
	const int Wp = pad_to(w, ws);
	l->tf_pad_h = Hp;
	l->tf_pad_w = Wp;

	const int T = ws * ws;
	const int nW = (Hp / ws) * (Wp / ws);
	const int total_windows = l->batch * nW;
	const int ffn_hidden = l->n * l->tf_ffn_ratio;
	const TransformerWorkspaceLayout workspace = make_transformer_workspace_layout(l->batch, l->c, l->n, Hp, Wp, ws, l->tf_heads, l->tf_ffn_ratio);

	l->output = (float*)xrealloc(l->output, l->batch * l->outputs * sizeof(float));
	l->delta = (float*)xrealloc(l->delta, l->batch * l->outputs * sizeof(float));

	l->tf_qkv_out = (float*)xrealloc(l->tf_qkv_out, total_windows * T * 3 * l->c * sizeof(float));
	l->tf_attn_scores = (float*)xrealloc(l->tf_attn_scores, total_windows * l->tf_heads * T * T * sizeof(float));
	l->tf_attn_out = (float*)xrealloc(l->tf_attn_out, total_windows * T * l->c * sizeof(float));
	l->tf_ffn_hidden = (float*)xrealloc(l->tf_ffn_hidden, total_windows * T * ffn_hidden * sizeof(float));
	l->activation_input = (float*)xrealloc(l->activation_input, total_windows * T * ffn_hidden * sizeof(float));
	l->tf_ln1_mean = (float*)xrealloc(l->tf_ln1_mean, total_windows * T * sizeof(float));
	l->tf_ln1_var = (float*)xrealloc(l->tf_ln1_var, total_windows * T * sizeof(float));
	l->tf_ln2_mean = (float*)xrealloc(l->tf_ln2_mean, total_windows * T * sizeof(float));
	l->tf_ln2_var = (float*)xrealloc(l->tf_ln2_var, total_windows * T * sizeof(float));
	l->tf_ln1_xhat = (float*)xrealloc(l->tf_ln1_xhat, total_windows * T * l->c * sizeof(float));
	l->tf_ln2_xhat = (float*)xrealloc(l->tf_ln2_xhat, total_windows * T * l->n * sizeof(float));
	l->tf_pre_res2 = (float*)xrealloc(l->tf_pre_res2, total_windows * T * l->n * sizeof(float));
	l->tf_windowed_input = (float*)xrealloc(l->tf_windowed_input, total_windows * T * l->c * sizeof(float));
	l->x = (float*)xrealloc(l->x, total_windows * T * l->n * sizeof(float));
	l->x_norm = (float*)xrealloc(l->x_norm, total_windows * T * l->n * sizeof(float));
	l->tf_attn_mask = (float*)xrealloc(l->tf_attn_mask, nW * T * T * sizeof(float));
	l->tf_workspace = (float*)xrealloc(l->tf_workspace, workspace.total * sizeof(float));
	l->tf_workspace_size = workspace.total;

	// Rebuild attention mask (handles both shift and padding)
	memset(l->tf_attn_mask, 0, nW * T * T * sizeof(float));
	const bool needs_mask = l->tf_shift || (Hp != h) || (Wp != w);
	if (needs_mask)
	{
		build_attention_mask(l->tf_attn_mask, h, w, Hp, Wp, ws, l->tf_shift ? ws / 2 : 0);
	}

#ifdef DARKNET_GPU
	cuda_free(l->output_gpu);
	cuda_free(l->delta_gpu);
	l->output_gpu = cuda_make_array(l->output, l->batch * l->outputs);
	l->delta_gpu = cuda_make_array(l->delta, l->batch * l->outputs);

	cuda_free(l->tf_qkv_out_gpu);
	cuda_free(l->tf_attn_scores_gpu);
	cuda_free(l->tf_attn_out_gpu);
	cuda_free(l->tf_ffn_hidden_gpu);
	cuda_free(l->activation_input_gpu);
	cuda_free(l->tf_ln1_mean_gpu);
	cuda_free(l->tf_ln1_var_gpu);
	cuda_free(l->tf_ln2_mean_gpu);
	cuda_free(l->tf_ln2_var_gpu);
	cuda_free(l->tf_ln1_xhat_gpu);
	cuda_free(l->tf_ln2_xhat_gpu);
	cuda_free(l->tf_pre_res2_gpu);
	cuda_free(l->tf_windowed_input_gpu);
	cuda_free(l->x_gpu);
	cuda_free(l->x_norm_gpu);
	cuda_free(l->tf_attn_mask_gpu);
	cuda_free(l->tf_gpu_workspace);

	l->tf_qkv_out_gpu = cuda_make_array(nullptr, total_windows * T * 3 * l->c);
	l->tf_attn_scores_gpu = cuda_make_array(nullptr, total_windows * l->tf_heads * T * T);
	l->tf_attn_out_gpu = cuda_make_array(nullptr, total_windows * T * l->c);
	l->tf_ffn_hidden_gpu = cuda_make_array(nullptr, total_windows * T * ffn_hidden);
	l->activation_input_gpu = cuda_make_array(nullptr, total_windows * T * ffn_hidden);
	l->tf_ln1_mean_gpu = cuda_make_array(nullptr, total_windows * T);
	l->tf_ln1_var_gpu = cuda_make_array(nullptr, total_windows * T);
	l->tf_ln2_mean_gpu = cuda_make_array(nullptr, total_windows * T);
	l->tf_ln2_var_gpu = cuda_make_array(nullptr, total_windows * T);
	l->tf_ln1_xhat_gpu = cuda_make_array(nullptr, total_windows * T * l->c);
	l->tf_ln2_xhat_gpu = cuda_make_array(nullptr, total_windows * T * l->n);
	l->tf_pre_res2_gpu = cuda_make_array(nullptr, total_windows * T * l->n);
	l->tf_windowed_input_gpu = cuda_make_array(nullptr, total_windows * T * l->c);
	l->x_gpu = cuda_make_array(nullptr, total_windows * T * l->n);
	l->x_norm_gpu = cuda_make_array(nullptr, total_windows * T * l->n);
	l->tf_attn_mask_gpu = cuda_make_array(l->tf_attn_mask, nW * T * T);
	l->tf_gpu_workspace = cuda_make_array(nullptr, workspace.total);
	l->tf_gpu_workspace_size = workspace.total;
#endif
}

// ─── forward ──────────────────────────────────────────────────────────────────

void forward_transformer_layer(Darknet::Layer & l, Darknet::NetworkState state)
{
	TAT(TATPARMS);

	const int B = l.batch;
	const int C = l.c;
	const int N = l.n;
	const int H = l.h;
	const int W = l.w;
	const int ws = l.tf_window_size;
	const int Hp = l.tf_pad_h;
	const int Wp = l.tf_pad_w;
	const int T = ws * ws;
	const int nH = Hp / ws;
	const int nW_spatial = nH * (Wp / ws);
	const int total_windows = B * nW_spatial;
	const int heads = l.tf_heads;
	const int d = l.tf_head_dim;
	const int shift_size = l.tf_shift ? ws / 2 : 0;
	const int ffn_hidden = N * l.tf_ffn_ratio;
	const int M_ffn = total_windows * T;
	const float scale = 1.0f / sqrtf((float)d);
	const TransformerWorkspaceLayout layout = make_transformer_workspace_layout(B, C, N, Hp, Wp, ws, heads, l.tf_ffn_ratio);

	assert(l.tf_workspace != nullptr);
	assert(l.tf_workspace_size >= layout.total);

	float *spatial0 = workspace_ptr(l.tf_workspace, layout.spatial0);
	float *spatial1 = workspace_ptr(l.tf_workspace, layout.spatial1);
	float *ln1_out = workspace_ptr(l.tf_workspace, layout.token_c0);
	float *proj_out = workspace_ptr(l.tf_workspace, layout.token_n0);
	float *res1_out = workspace_ptr(l.tf_workspace, layout.token_n1);
	float *Q = workspace_ptr(l.tf_workspace, layout.head0);
	float *K = workspace_ptr(l.tf_workspace, layout.head1);
	float *V = workspace_ptr(l.tf_workspace, layout.head2);
	float *attn_result = workspace_ptr(l.tf_workspace, layout.head3);
	const size_t padded_size = static_cast<size_t>(B) * C * Hp * Wp;
	const size_t windowed_c = static_cast<size_t>(total_windows) * T * C;
	const size_t windowed_n = static_cast<size_t>(total_windows) * T * N;

	float *padded = spatial0;
	if (Hp == H && Wp == W)
	{
		memcpy(padded, state.input, padded_size * sizeof(float));
	}
	else
	{
		memset(padded, 0, padded_size * sizeof(float));
		for (int b = 0; b < B; ++b)
			for (int c_idx = 0; c_idx < C; ++c_idx)
				for (int y = 0; y < H; ++y)
					memcpy(padded + ((b * C + c_idx) * Hp + y) * Wp,
						   state.input + ((b * C + c_idx) * H + y) * W,
						   static_cast<size_t>(W) * sizeof(float));
	}

	float *shifted = padded;
	if (shift_size > 0)
	{
		shifted = spatial1;
		cyclic_shift(padded, shifted, B, C, Hp, Wp, -shift_size, -shift_size);
	}
	check_nan_cpu("forward: after pad/shift", shifted, static_cast<int>(padded_size), l.index);

	window_partition(shifted, l.tf_windowed_input, B, C, Hp, Wp, ws);
	check_nan_cpu("forward: after window partition", l.tf_windowed_input, static_cast<int>(windowed_c), l.index);

	layernorm_forward(l.tf_windowed_input, ln1_out,
		l.tf_ln1_mean, l.tf_ln1_var, l.tf_ln1_xhat,
		l.tf_ln1_gamma, l.tf_ln1_beta, total_windows * T, C);
	check_nan_cpu("forward: after layernorm1", ln1_out, static_cast<int>(windowed_c), l.index);

	const int M_qkv = total_windows * T;
	gemm_cpu(0, 1, M_qkv, 3 * C, C, 1.0f,
		ln1_out, C,
		l.weights, C,
		0.0f,
		l.tf_qkv_out, 3 * C);
	add_bias_rows(l.tf_qkv_out, l.biases, M_qkv, 3 * C);
	check_nan_cpu("forward: after qkv projection", l.tf_qkv_out, total_windows * T * 3 * C, l.index);

	memset(l.tf_attn_out, 0, windowed_c * sizeof(float));
	memset(l.tf_attn_scores, 0, static_cast<size_t>(total_windows) * heads * T * T * sizeof(float));

	for (int win = 0; win < total_windows; ++win)
	{
		const int win_in_batch = win % nW_spatial;
		const float *qkv_win = l.tf_qkv_out + static_cast<size_t>(win) * T * 3 * C;
		float *attn_out_win = l.tf_attn_out + static_cast<size_t>(win) * T * C;

		for (int h_idx = 0; h_idx < heads; ++h_idx)
		{
			for (int t = 0; t < T; ++t)
			{
				const float *token_qkv = qkv_win + static_cast<size_t>(t) * 3 * C;
				memcpy(Q + static_cast<size_t>(t) * d, token_qkv + h_idx * d, static_cast<size_t>(d) * sizeof(float));
				memcpy(K + static_cast<size_t>(t) * d, token_qkv + C + h_idx * d, static_cast<size_t>(d) * sizeof(float));
				memcpy(V + static_cast<size_t>(t) * d, token_qkv + 2 * C + h_idx * d, static_cast<size_t>(d) * sizeof(float));
			}
			check_nan_cpu("forward: after q split", Q, T * d, l.index);
			check_nan_cpu("forward: after k split", K, T * d, l.index);

			constrain_cpu(T * d, 256.0f, Q);
			constrain_cpu(T * d, 256.0f, K);

			float *scores = l.tf_attn_scores + static_cast<size_t>(win * heads + h_idx) * T * T;
			gemm_cpu(0, 1, T, T, d, scale,
				Q, d,
				K, d,
				0.0f,
				scores, T);
			check_nan_cpu("forward: raw attention scores", scores, T * T, l.index);

			for (int i = 0; i < T; ++i)
				for (int j = 0; j < T; ++j)
					scores[i * T + j] += l.tf_rel_pos_bias[h_idx * (2 * ws - 1) * (2 * ws - 1) + l.tf_rel_pos_index[i * T + j]];

			const float *mask = l.tf_attn_mask + static_cast<size_t>(win_in_batch) * T * T;
			for (int i = 0; i < T * T; ++i)
				scores[i] += mask[i];

			constrain_cpu(T * T, 20.0f, scores);
			for (int t = 0; t < T; ++t)
				softmax_row(scores + t * T, T);

			gemm_cpu(0, 0, T, d, T, 1.0f,
				scores, T,
				V, d,
				0.0f,
				attn_result, d);

			for (int t = 0; t < T; ++t)
			{
				memcpy(attn_out_win + static_cast<size_t>(t) * C + h_idx * d,
					attn_result + static_cast<size_t>(t) * d,
					static_cast<size_t>(d) * sizeof(float));
			}
		}
	}
	check_nan_cpu("forward: after attention scores", l.tf_attn_scores, total_windows * heads * T * T, l.index);
	check_nan_cpu("forward: after attention output", l.tf_attn_out, static_cast<int>(windowed_c), l.index);

	gemm_cpu(0, 1, total_windows * T, N, C, 1.0f,
		l.tf_attn_out, C,
		l.tf_wo, C,
		0.0f,
		proj_out, N);
	add_bias_rows(proj_out, l.tf_wo_bias, total_windows * T, N);
	check_nan_cpu("forward: after output projection", proj_out, static_cast<int>(windowed_n), l.index);

	memcpy(l.x, proj_out, windowed_n * sizeof(float));
	if (C == N)
	{
		mhc_residual_forward_cpu(l.tf_windowed_input, l.x, res1_out, windowed_n, l.scales, 0);
	}
	else
	{
		gemm_cpu(0, 1, total_windows * T, N, C, 1.0f,
			l.tf_windowed_input, C,
			l.tf_res_proj, C,
			0.0f,
			res1_out, N);
		mhc_residual_forward_cpu(res1_out, l.x, res1_out, windowed_n, l.scales, 0);
	}
	check_nan_cpu("forward: after residual1", res1_out, static_cast<int>(windowed_n), l.index);

	memcpy(l.tf_pre_res2, res1_out, windowed_n * sizeof(float));

	float *ln2_out = proj_out;
	layernorm_forward(res1_out, ln2_out,
		l.tf_ln2_mean, l.tf_ln2_var, l.tf_ln2_xhat,
		l.tf_ln2_gamma, l.tf_ln2_beta, total_windows * T, N);
	check_nan_cpu("forward: after layernorm2", ln2_out, static_cast<int>(windowed_n), l.index);

	gemm_cpu(0, 1, M_ffn, ffn_hidden, N, 1.0f,
		ln2_out, N,
		l.tf_ffn_w1, N,
		0.0f,
		l.tf_ffn_hidden, ffn_hidden);
	add_bias_rows(l.tf_ffn_hidden, l.tf_ffn_b1, M_ffn, ffn_hidden);
	memcpy(l.activation_input, l.tf_ffn_hidden, static_cast<size_t>(M_ffn) * ffn_hidden * sizeof(float));
	activate_array(l.tf_ffn_hidden, M_ffn * ffn_hidden, l.activation);
	check_nan_cpu("forward: after ffn hidden", l.tf_ffn_hidden, M_ffn * ffn_hidden, l.index);

	float *ffn_out = proj_out;
	gemm_cpu(0, 1, M_ffn, N, ffn_hidden, 1.0f,
		l.tf_ffn_hidden, ffn_hidden,
		l.tf_ffn_w2, ffn_hidden,
		0.0f,
		ffn_out, N);
	add_bias_rows(ffn_out, l.tf_ffn_b2, M_ffn, N);
	memcpy(l.x_norm, ffn_out, windowed_n * sizeof(float));
	mhc_residual_forward_cpu(res1_out, l.x_norm, ffn_out, windowed_n, l.scales, 1);
	check_nan_cpu("forward: after ffn output + residual", ffn_out, static_cast<int>(windowed_n), l.index);

	if (shift_size == 0 && Hp == H && Wp == W)
	{
		window_unpartition(ffn_out, l.output, B, N, Hp, Wp, ws);
	}
	else
	{
		float *unpartitioned = spatial0;
		window_unpartition(ffn_out, unpartitioned, B, N, Hp, Wp, ws);

		float *unshifted = unpartitioned;
		if (shift_size > 0)
		{
			unshifted = spatial1;
			cyclic_shift(unpartitioned, unshifted, B, N, Hp, Wp, shift_size, shift_size);
		}

		if (Hp == H && Wp == W)
		{
			memcpy(l.output, unshifted, static_cast<size_t>(B) * N * H * W * sizeof(float));
		}
		else
		{
			for (int b = 0; b < B; ++b)
				for (int c_idx = 0; c_idx < N; ++c_idx)
					for (int y = 0; y < H; ++y)
						memcpy(l.output + ((b * N + c_idx) * H + y) * W,
							   unshifted + ((b * N + c_idx) * Hp + y) * Wp,
							   static_cast<size_t>(W) * sizeof(float));
		}
	}
	constrain_cpu(B * N * H * W, 100.0f, l.output);
	check_nan_cpu("forward: final output", l.output, B * N * H * W, l.index);
}

// ─── backward ─────────────────────────────────────────────────────────────────

void backward_transformer_layer(Darknet::Layer & l, Darknet::NetworkState state)
{
	TAT(TATPARMS);

	const int B = l.batch;
	const int C = l.c;
	const int N = l.n;
	const int H = l.h;
	const int W = l.w;
	const int ws = l.tf_window_size;
	const int Hp = l.tf_pad_h;
	const int Wp = l.tf_pad_w;
	const int T = ws * ws;
	const int nW_spatial = (Hp / ws) * (Wp / ws);
	const int total_windows = B * nW_spatial;
	const int heads = l.tf_heads;
	const int d = l.tf_head_dim;
	const int shift_size = l.tf_shift ? ws / 2 : 0;
	const int ffn_hidden = N * l.tf_ffn_ratio;
	const int M_ffn = total_windows * T;
	const float scale = 1.0f / sqrtf((float)d);
	const TransformerWorkspaceLayout layout = make_transformer_workspace_layout(B, C, N, Hp, Wp, ws, heads, l.tf_ffn_ratio);

	assert(l.tf_workspace != nullptr);
	assert(l.tf_workspace_size >= layout.total);

	float *spatial0 = workspace_ptr(l.tf_workspace, layout.spatial0);
	float *spatial1 = workspace_ptr(l.tf_workspace, layout.spatial1);
	float *d_windows = workspace_ptr(l.tf_workspace, layout.token_n0);
	float *ln2_out = workspace_ptr(l.tf_workspace, layout.token_n1);
	float *d_pre_ln2 = workspace_ptr(l.tf_workspace, layout.token_n2);
	float *d_windowed_input = workspace_ptr(l.tf_workspace, layout.token_c0);
	float *d_attn_out = workspace_ptr(l.tf_workspace, layout.token_c1);
	float *d_ln1_out = workspace_ptr(l.tf_workspace, layout.token_c2);
	float *d_ffn_hidden = workspace_ptr(l.tf_workspace, layout.ffn);
	float *d_qkv_buf = workspace_ptr(l.tf_workspace, layout.token_3c);
	float *Q = workspace_ptr(l.tf_workspace, layout.head0);
	float *K_mat = workspace_ptr(l.tf_workspace, layout.head1);
	float *V = workspace_ptr(l.tf_workspace, layout.head2);
	float *scratch_a = workspace_ptr(l.tf_workspace, layout.head3);
	float *scratch_b = workspace_ptr(l.tf_workspace, layout.head4);
	float *d_scores = workspace_ptr(l.tf_workspace, layout.scores);
	const size_t padded_n = static_cast<size_t>(B) * N * Hp * Wp;
	const size_t padded_c = static_cast<size_t>(B) * C * Hp * Wp;
	const size_t windowed_n = static_cast<size_t>(total_windows) * T * N;
	const size_t windowed_c = static_cast<size_t>(total_windows) * T * C;

	constrain_cpu(B * N * H * W, 1.0f, l.delta);

	float *dout_padded = spatial0;
	if (Hp == H && Wp == W)
	{
		memcpy(dout_padded, l.delta, padded_n * sizeof(float));
	}
	else
	{
		memset(dout_padded, 0, padded_n * sizeof(float));
		for (int b = 0; b < B; ++b)
			for (int c_idx = 0; c_idx < N; ++c_idx)
				for (int y = 0; y < H; ++y)
					memcpy(dout_padded + ((b * N + c_idx) * Hp + y) * Wp,
						   l.delta + ((b * N + c_idx) * H + y) * W,
						   static_cast<size_t>(W) * sizeof(float));
	}
	check_nan_cpu("backward: initial delta", dout_padded, static_cast<int>(padded_n), l.index);

	float *dout_shifted = dout_padded;
	if (shift_size > 0)
	{
		dout_shifted = spatial1;
		cyclic_shift(dout_padded, dout_shifted, B, N, Hp, Wp, -shift_size, -shift_size);
	}

	window_partition(dout_shifted, d_windows, B, N, Hp, Wp, ws);
	check_nan_cpu("backward: after window partition", d_windows, static_cast<int>(windowed_n), l.index);

	float *d_ffn_out = ln2_out;
	mhc_residual_backward_cpu(l.tf_pre_res2, l.x_norm, d_windows,
		d_windows, d_ffn_out, windowed_n, l.scales, l.scale_updates, 1);

	gemm_cpu(0, 0, M_ffn, ffn_hidden, N, 1.0f,
		d_ffn_out, N,
		l.tf_ffn_w2, ffn_hidden,
		0.0f,
		d_ffn_hidden, ffn_hidden);
	gemm_cpu(1, 0, N, ffn_hidden, M_ffn, 1.0f,
		d_ffn_out, N,
		l.tf_ffn_hidden, ffn_hidden,
		1.0f,
		l.tf_ffn_w2_updates, ffn_hidden);
	for (int i = 0; i < M_ffn; ++i)
		for (int j = 0; j < N; ++j)
			l.tf_ffn_b2_updates[j] += d_ffn_out[i * N + j];

	gradient_array(l.activation_input, M_ffn * ffn_hidden, l.activation, d_ffn_hidden);
	check_nan_cpu("backward: after ffn hidden grad", d_ffn_hidden, M_ffn * ffn_hidden, l.index);

	layernorm_affine_from_xhat(l.tf_ln2_xhat, ln2_out, l.tf_ln2_gamma, l.tf_ln2_beta, total_windows * T, N);
	gemm_cpu(1, 0, ffn_hidden, N, M_ffn, 1.0f,
		d_ffn_hidden, ffn_hidden,
		ln2_out, N,
		1.0f,
		l.tf_ffn_w1_updates, N);
	for (int i = 0; i < M_ffn; ++i)
		for (int j = 0; j < ffn_hidden; ++j)
			l.tf_ffn_b1_updates[j] += d_ffn_hidden[i * ffn_hidden + j];

	float *d_ln2_out = ln2_out;
	gemm_cpu(0, 0, M_ffn, N, ffn_hidden, 1.0f,
		d_ffn_hidden, ffn_hidden,
		l.tf_ffn_w1, N,
		0.0f,
		d_ln2_out, N);
	check_nan_cpu("backward: before layernorm2 backward", d_ln2_out, static_cast<int>(windowed_n), l.index);

	layernorm_backward(d_ln2_out, l.tf_ln2_xhat, l.tf_ln2_var,
		l.tf_ln2_gamma, d_pre_ln2, l.tf_ln2_gamma_updates, l.tf_ln2_beta_updates,
		total_windows * T, N);
	for (size_t i = 0; i < windowed_n; ++i)
		d_pre_ln2[i] += d_windows[i];
	check_nan_cpu("backward: after layernorm2 backward + residual", d_pre_ln2, static_cast<int>(windowed_n), l.index);

	float *d_proj_out = ln2_out;
	memset(d_windowed_input, 0, windowed_c * sizeof(float));
	if (C == N)
	{
		mhc_residual_backward_cpu(l.tf_windowed_input, l.x, d_pre_ln2,
			d_windowed_input, d_proj_out, windowed_n, l.scales, l.scale_updates, 0);
	}
	else
	{
		gemm_cpu(0, 1, total_windows * T, N, C, 1.0f,
			l.tf_windowed_input, C,
			l.tf_res_proj, C,
			0.0f,
			l.x_norm, N);
		mhc_residual_backward_cpu(l.x_norm, l.x, d_pre_ln2,
			d_pre_ln2, d_proj_out, windowed_n, l.scales, l.scale_updates, 0);
		gemm_cpu(0, 0, total_windows * T, C, N, 1.0f,
			d_pre_ln2, N,
			l.tf_res_proj, C,
			0.0f,
			d_windowed_input, C);
		gemm_cpu(1, 0, N, C, total_windows * T, 1.0f,
			d_pre_ln2, N,
			l.tf_windowed_input, C,
			1.0f,
			l.tf_res_proj_updates, C);
	}

	gemm_cpu(0, 0, total_windows * T, C, N, 1.0f,
		d_proj_out, N,
		l.tf_wo, C,
		0.0f,
		d_attn_out, C);
	check_nan_cpu("backward: after output projection backward", d_attn_out, static_cast<int>(windowed_c), l.index);
	gemm_cpu(1, 0, N, C, total_windows * T, 1.0f,
		d_proj_out, N,
		l.tf_attn_out, C,
		1.0f,
		l.tf_wo_updates, C);
	for (int i = 0; i < total_windows * T; ++i)
		for (int j = 0; j < N; ++j)
			l.tf_wo_bias_updates[j] += d_proj_out[i * N + j];

	memset(d_ln1_out, 0, windowed_c * sizeof(float));
	for (int win = 0; win < total_windows; ++win)
	{
		const float *qkv_win = l.tf_qkv_out + static_cast<size_t>(win) * T * 3 * C;
		const float *d_attn_out_win = d_attn_out + static_cast<size_t>(win) * T * C;
		float *d_ln1_win = d_ln1_out + static_cast<size_t>(win) * T * C;
		memset(d_qkv_buf, 0, static_cast<size_t>(T) * 3 * C * sizeof(float));

		for (int h_idx = 0; h_idx < heads; ++h_idx)
		{
			for (int t = 0; t < T; ++t)
			{
				const float *token_qkv = qkv_win + static_cast<size_t>(t) * 3 * C;
				memcpy(Q + static_cast<size_t>(t) * d, token_qkv + h_idx * d, static_cast<size_t>(d) * sizeof(float));
				memcpy(K_mat + static_cast<size_t>(t) * d, token_qkv + C + h_idx * d, static_cast<size_t>(d) * sizeof(float));
				memcpy(V + static_cast<size_t>(t) * d, token_qkv + 2 * C + h_idx * d, static_cast<size_t>(d) * sizeof(float));
				memcpy(scratch_a + static_cast<size_t>(t) * d, d_attn_out_win + static_cast<size_t>(t) * C + h_idx * d, static_cast<size_t>(d) * sizeof(float));
			}

			constrain_cpu(T * d, 256.0f, Q);
			constrain_cpu(T * d, 256.0f, K_mat);
			float *scores = l.tf_attn_scores + static_cast<size_t>(win * heads + h_idx) * T * T;

			float *dV = scratch_b;
			gemm_cpu(1, 0, T, d, T, 1.0f,
				scores, T,
				scratch_a, d,
				0.0f,
				dV, d);

			gemm_cpu(0, 1, T, T, d, 1.0f,
				scratch_a, d,
				V, d,
				0.0f,
				d_scores, T);

			for (int t = 0; t < T; ++t)
			{
				float dot = 0.0f;
				for (int j = 0; j < T; ++j)
					dot += d_scores[t * T + j] * scores[t * T + j];
				for (int j = 0; j < T; ++j)
					d_scores[t * T + j] = scores[t * T + j] * (d_scores[t * T + j] - dot);
			}

			for (int i = 0; i < T; ++i)
				for (int j = 0; j < T; ++j)
					l.tf_rel_pos_bias_updates[h_idx * (2 * ws - 1) * (2 * ws - 1) + l.tf_rel_pos_index[i * T + j]] += d_scores[i * T + j];

			float *dQ = scratch_a;
			float *dK = K_mat;
			gemm_cpu(0, 0, T, d, T, scale,
				d_scores, T,
				K_mat, d,
				0.0f,
				dQ, d);
			gemm_cpu(1, 0, T, d, T, scale,
				d_scores, T,
				Q, d,
				0.0f,
				dK, d);

			for (int t = 0; t < T; ++t)
			{
				float *token_dqkv = d_qkv_buf + static_cast<size_t>(t) * 3 * C;
				memcpy(token_dqkv + h_idx * d, dQ + static_cast<size_t>(t) * d, static_cast<size_t>(d) * sizeof(float));
				memcpy(token_dqkv + C + h_idx * d, dK + static_cast<size_t>(t) * d, static_cast<size_t>(d) * sizeof(float));
				memcpy(token_dqkv + 2 * C + h_idx * d, dV + static_cast<size_t>(t) * d, static_cast<size_t>(d) * sizeof(float));
			}
		}

		for (int t = 0; t < T; ++t)
			for (int j = 0; j < 3 * C; ++j)
				l.bias_updates[j] += d_qkv_buf[t * 3 * C + j];

		float *ln1_out_win = K_mat;
		layernorm_affine_from_xhat(l.tf_ln1_xhat + static_cast<size_t>(win) * T * C,
			ln1_out_win, l.tf_ln1_gamma, l.tf_ln1_beta, T, C);
		gemm_cpu(1, 0, 3 * C, C, T, 1.0f,
			d_qkv_buf, 3 * C,
			ln1_out_win, C,
			1.0f,
			l.weight_updates, C);
		gemm_cpu(0, 0, T, C, 3 * C, 1.0f,
			d_qkv_buf, 3 * C,
			l.weights, C,
			0.0f,
			d_ln1_win, C);
	}
	constrain_cpu(total_windows * T * C, 100.0f, d_ln1_out);
	check_nan_cpu("backward: after attention backward", d_ln1_out, static_cast<int>(windowed_c), l.index);

	float *d_pre_ln1 = d_ln1_out;
	layernorm_backward(d_ln1_out, l.tf_ln1_xhat, l.tf_ln1_var,
		l.tf_ln1_gamma, d_pre_ln1, l.tf_ln1_gamma_updates, l.tf_ln1_beta_updates,
		total_windows * T, C);
	for (size_t i = 0; i < windowed_c; ++i)
		d_pre_ln1[i] += d_windowed_input[i];
	check_nan_cpu("backward: after layernorm1 backward", d_pre_ln1, static_cast<int>(windowed_c), l.index);

	float *d_shifted = spatial0;
	memset(d_shifted, 0, padded_c * sizeof(float));
	window_unpartition(d_pre_ln1, d_shifted, B, C, Hp, Wp, ws);

	float *d_padded = d_shifted;
	if (shift_size > 0)
	{
		d_padded = spatial1;
		cyclic_shift(d_shifted, d_padded, B, C, Hp, Wp, shift_size, shift_size);
	}

	if (state.delta)
	{
		constrain_cpu(B * C * Hp * Wp, 1.0f, d_padded);
		if (Hp == H && Wp == W)
		{
			for (int i = 0; i < B * C * H * W; ++i)
				state.delta[i] += d_padded[i];
		}
		else
		{
			for (int b = 0; b < B; ++b)
				for (int c_idx = 0; c_idx < C; ++c_idx)
					for (int y = 0; y < H; ++y)
						for (int x = 0; x < W; ++x)
							state.delta[((b * C + c_idx) * H + y) * W + x] += d_padded[((b * C + c_idx) * Hp + y) * Wp + x];
		}
		check_nan_cpu("backward: propagated state.delta", state.delta, B * C * H * W, l.index);
	}
}

// ─── update ───────────────────────────────────────────────────────────────────

void update_transformer_layer(Darknet::Layer & l, int batch, float learning_rate_init, float momentum, float decay)
{
	TAT(TATPARMS);

	const float lr = learning_rate_init * l.learning_rate_scale;
	const int C = l.c;
	const int N = l.n;
	const int ffn_hidden = N * l.tf_ffn_ratio;
	const int heads = l.tf_heads;
	const int ws = l.tf_window_size;
	const int bias_table_len = (2 * ws - 1) * (2 * ws - 1);

	// ── Gradient norm clipping (L2) ──
	{
		const float max_grad_norm = 5.0f;

		auto l2_norm = [](float *buf, int n) -> double
		{
			double sum = 0.0;
			for (int i = 0; i < n; i++)
			{
				const float value = buf[i];
				if (!std::isfinite(value))
				{
					buf[i] = 0.0f;
					continue;
				}
				sum += static_cast<double>(value) * static_cast<double>(value);
			}
			return sum;
		};

		double global_norm_sq = 0.0;
		global_norm_sq += l2_norm(l.weight_updates, l.nweights);
		global_norm_sq += l2_norm(l.bias_updates, 3 * C);
		global_norm_sq += l2_norm(l.tf_wo_updates, N * C);
		global_norm_sq += l2_norm(l.tf_wo_bias_updates, N);
		global_norm_sq += l2_norm(l.tf_ln1_gamma_updates, C);
		global_norm_sq += l2_norm(l.tf_ln1_beta_updates, C);
		global_norm_sq += l2_norm(l.tf_ln2_gamma_updates, N);
		global_norm_sq += l2_norm(l.tf_ln2_beta_updates, N);
		global_norm_sq += l2_norm(l.tf_ffn_w1_updates, ffn_hidden * N);
		global_norm_sq += l2_norm(l.tf_ffn_b1_updates, ffn_hidden);
		global_norm_sq += l2_norm(l.tf_ffn_w2_updates, N * ffn_hidden);
		global_norm_sq += l2_norm(l.tf_ffn_b2_updates, N);
		global_norm_sq += l2_norm(l.tf_rel_pos_bias_updates, heads * bias_table_len);
		global_norm_sq += l2_norm(l.scale_updates, MHC_PARAM_COUNT);
		if (C != N && l.tf_res_proj_updates)
			global_norm_sq += l2_norm(l.tf_res_proj_updates, N * C);

		const double global_norm = std::sqrt(global_norm_sq);
		if (!std::isfinite(global_norm) || global_norm > max_grad_norm)
		{
			const float clip_coef = std::isfinite(global_norm) ?
				static_cast<float>(max_grad_norm / global_norm) : 0.0f;
			scal_cpu(l.nweights, clip_coef, l.weight_updates, 1);
			scal_cpu(3 * C, clip_coef, l.bias_updates, 1);
			scal_cpu(N * C, clip_coef, l.tf_wo_updates, 1);
			scal_cpu(N, clip_coef, l.tf_wo_bias_updates, 1);
			scal_cpu(C, clip_coef, l.tf_ln1_gamma_updates, 1);
			scal_cpu(C, clip_coef, l.tf_ln1_beta_updates, 1);
			scal_cpu(N, clip_coef, l.tf_ln2_gamma_updates, 1);
			scal_cpu(N, clip_coef, l.tf_ln2_beta_updates, 1);
			scal_cpu(ffn_hidden * N, clip_coef, l.tf_ffn_w1_updates, 1);
			scal_cpu(ffn_hidden, clip_coef, l.tf_ffn_b1_updates, 1);
			scal_cpu(N * ffn_hidden, clip_coef, l.tf_ffn_w2_updates, 1);
			scal_cpu(N, clip_coef, l.tf_ffn_b2_updates, 1);
			scal_cpu(heads * bias_table_len, clip_coef, l.tf_rel_pos_bias_updates, 1);
			scal_cpu(MHC_PARAM_COUNT, clip_coef, l.scale_updates, 1);
			if (C != N && l.tf_res_proj_updates)
				scal_cpu(N * C, clip_coef, l.tf_res_proj_updates, 1);
		}
	}

	// QKV weights
	axpy_cpu(l.nweights, -decay * batch, l.weights, 1, l.weight_updates, 1);
	axpy_cpu(l.nweights, lr / batch, l.weight_updates, 1, l.weights, 1);
	scal_cpu(l.nweights, momentum, l.weight_updates, 1);

	// QKV biases
	axpy_cpu(3 * C, lr / batch, l.bias_updates, 1, l.biases, 1);
	scal_cpu(3 * C, momentum, l.bias_updates, 1);

	// Output projection
	{
		const int count = N * C;
		axpy_cpu(count, -decay * batch, l.tf_wo, 1, l.tf_wo_updates, 1);
		axpy_cpu(count, lr / batch, l.tf_wo_updates, 1, l.tf_wo, 1);
		scal_cpu(count, momentum, l.tf_wo_updates, 1);
	}
	axpy_cpu(N, lr / batch, l.tf_wo_bias_updates, 1, l.tf_wo_bias, 1);
	scal_cpu(N, momentum, l.tf_wo_bias_updates, 1);

	// Residual projection (when C != N)
	if (C != N && l.tf_res_proj)
	{
		const int count = N * C;
		axpy_cpu(count, -decay * batch, l.tf_res_proj, 1, l.tf_res_proj_updates, 1);
		axpy_cpu(count, lr / batch, l.tf_res_proj_updates, 1, l.tf_res_proj, 1);
		scal_cpu(count, momentum, l.tf_res_proj_updates, 1);
	}

	// mHC residual mixer
	axpy_cpu(MHC_PARAM_COUNT, lr / batch, l.scale_updates, 1, l.scales, 1);
	sanitize_and_constrain_mhc_params_cpu(l.scales);
	scal_cpu(MHC_PARAM_COUNT, momentum, l.scale_updates, 1);

	// LayerNorm 1
	axpy_cpu(C, lr / batch, l.tf_ln1_gamma_updates, 1, l.tf_ln1_gamma, 1);
	scal_cpu(C, momentum, l.tf_ln1_gamma_updates, 1);
	axpy_cpu(C, lr / batch, l.tf_ln1_beta_updates, 1, l.tf_ln1_beta, 1);
	scal_cpu(C, momentum, l.tf_ln1_beta_updates, 1);

	// LayerNorm 2
	axpy_cpu(N, lr / batch, l.tf_ln2_gamma_updates, 1, l.tf_ln2_gamma, 1);
	scal_cpu(N, momentum, l.tf_ln2_gamma_updates, 1);
	axpy_cpu(N, lr / batch, l.tf_ln2_beta_updates, 1, l.tf_ln2_beta, 1);
	scal_cpu(N, momentum, l.tf_ln2_beta_updates, 1);

	// FFN W1
	{
		const int count = ffn_hidden * N;
		axpy_cpu(count, -decay * batch, l.tf_ffn_w1, 1, l.tf_ffn_w1_updates, 1);
		axpy_cpu(count, lr / batch, l.tf_ffn_w1_updates, 1, l.tf_ffn_w1, 1);
		scal_cpu(count, momentum, l.tf_ffn_w1_updates, 1);
	}
	axpy_cpu(ffn_hidden, lr / batch, l.tf_ffn_b1_updates, 1, l.tf_ffn_b1, 1);
	scal_cpu(ffn_hidden, momentum, l.tf_ffn_b1_updates, 1);

	// FFN W2
	{
		const int count = N * ffn_hidden;
		axpy_cpu(count, -decay * batch, l.tf_ffn_w2, 1, l.tf_ffn_w2_updates, 1);
		axpy_cpu(count, lr / batch, l.tf_ffn_w2_updates, 1, l.tf_ffn_w2, 1);
		scal_cpu(count, momentum, l.tf_ffn_w2_updates, 1);
	}
	axpy_cpu(N, lr / batch, l.tf_ffn_b2_updates, 1, l.tf_ffn_b2, 1);
	scal_cpu(N, momentum, l.tf_ffn_b2_updates, 1);

	// Relative position bias
	{
		const int count = heads * bias_table_len;
		axpy_cpu(count, lr / batch, l.tf_rel_pos_bias_updates, 1, l.tf_rel_pos_bias, 1);
		scal_cpu(count, momentum, l.tf_rel_pos_bias_updates, 1);
	}
}
