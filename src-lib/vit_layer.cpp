#include "darknet_internal.hpp"
#include "vit_layer.hpp"
#include "gemm.hpp"
#include "utils.hpp"
#include "blas.hpp"
#include "activations.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

namespace
{
	static auto & cfg_and_state = Darknet::CfgAndState::get();

	static constexpr float VIT_ATTENTION_QK_CLAMP = 128.0f;
	static constexpr float VIT_ATTENTION_SCORE_CLAMP = 10.0f;
	static constexpr float VIT_FEATURE_CLAMP = 10.0f;
	static constexpr float VIT_GRAD_CLAMP = 20.0f;
	static constexpr float VIT_MHC_PARAM_CLAMP = 8.0f;
	static constexpr int VIT_POS_EMBED_LEARNED = 0;
	static constexpr int VIT_POS_EMBED_SINUSOIDAL = 1;
	static constexpr int VIT_POS_INIT_RANDOM = 0;
	static constexpr int VIT_POS_INIT_ZERO = 1;

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
				const std::string layer_label = Darknet::layer_type_diagnostic_label(Darknet::ELayerType::VIT);
				std::printf("[%s layer] NaN/Inf detected at layer %d, step: %s (idx=%d, value=%g)\n",
					layer_label.c_str(), layer_idx, step_name, i, static_cast<double>(value));
				return;
			}
		}
	}

	static void sanitize_and_constrain_cpu(float *arr, int n, float limit)
	{
		if (arr == nullptr || n <= 0)
		{
			return;
		}

		for (int i = 0; i < n; ++i)
		{
			float value = arr[i];
			if (!std::isfinite(value))
			{
				value = 0.0f;
			}
			else
			{
				value = std::max(-limit, std::min(limit, value));
			}
			arr[i] = value;
		}
	}

	static inline float sanitize_and_constrain_value(float value, float limit)
	{
		if (!std::isfinite(value))
		{
			return 0.0f;
		}
		return std::max(-limit, std::min(limit, value));
	}

	static inline int vit_patch_dim(const Darknet::Layer & l)
	{
		return l.vit_patch_size * l.vit_patch_size * l.c;
	}

	static void vit_validate_shape(int h, int w, int patch_size, int patch_stride, int patch_pad,
		int filters, int heads, int mlp_dim)
	{
		if (patch_size < 1)
		{
			darknet_fatal_error(DARKNET_LOC, "vit: patch_size must be >= 1, got %d", patch_size);
		}
		if (patch_stride < 1)
		{
			darknet_fatal_error(DARKNET_LOC, "vit: patch_stride must be >= 1, got %d", patch_stride);
		}
		if (patch_pad < 0)
		{
			darknet_fatal_error(DARKNET_LOC, "vit: patch_pad must be >= 0, got %d", patch_pad);
		}
		const int eff_h = h + 2 * patch_pad - patch_size;
		const int eff_w = w + 2 * patch_pad - patch_size;
		if (eff_h < 0 || eff_w < 0)
		{
			darknet_fatal_error(DARKNET_LOC,
				"vit: feature map %dx%d too small for patch_size=%d patch_pad=%d (eff %dx%d < 0)",
				w, h, patch_size, patch_pad, eff_w, eff_h);
		}
		if (eff_h % patch_stride != 0 || eff_w % patch_stride != 0)
		{
			darknet_fatal_error(DARKNET_LOC,
				"vit: (h+2*pad-P)=%d and (w+2*pad-P)=%d must both be divisible by patch_stride=%d "
				"(feature map %dx%d, patch_size=%d, patch_pad=%d)",
				eff_h, eff_w, patch_stride, w, h, patch_size, patch_pad);
		}
		if (heads < 1)
		{
			darknet_fatal_error(DARKNET_LOC, "vit: heads must be >= 1, got %d", heads);
		}
		if (filters % heads != 0)
		{
			darknet_fatal_error(DARKNET_LOC,
				"vit: filters (%d) must be divisible by heads (%d)", filters, heads);
		}
		if (mlp_dim < 1)
		{
			darknet_fatal_error(DARKNET_LOC, "vit: mlp_dim must be >= 1, got %d", mlp_dim);
		}
	}

	static void vit_fill_2d_sinusoidal_pos_embed(float *pos_embed, int H, int W, int C)
	{
		if (pos_embed == nullptr || H <= 0 || W <= 0 || C <= 0)
		{
			return;
		}

		const int quarter = std::max(1, C / 4);
		for (int y = 0; y < H; ++y)
		{
			for (int x = 0; x < W; ++x)
			{
				float *row = pos_embed + (y * W + x) * C;
				for (int c = 0; c < C; ++c)
				{
					const int band = c / 4;
					const float div = std::exp(-std::log(10000.0f) * static_cast<float>(band) / static_cast<float>(quarter));
					switch (c % 4)
					{
						case 0: row[c] = std::sin(static_cast<float>(y) * div); break;
						case 1: row[c] = std::cos(static_cast<float>(y) * div); break;
						case 2: row[c] = std::sin(static_cast<float>(x) * div); break;
						default: row[c] = std::cos(static_cast<float>(x) * div); break;
					}
				}
			}
		}
	}

	static bool vit_init_dropin_patch_embed(Darknet::Layer & l)
	{
		if (l.vit_patch_embed == nullptr || l.n != l.c)
		{
			return false;
		}

		const int P = l.vit_patch_size;
		const int S = l.vit_patch_stride;
		const int pad = l.vit_patch_pad;
		const int C = l.c;
		const int K = vit_patch_dim(l);
		std::fill(l.vit_patch_embed, l.vit_patch_embed + l.n * K, 0.0f);

		// Same-resolution overlapping mode (stride=1, pad=(P-1)/2, P odd):
		// true identity = only center pixel contributes weight 1. Avoids box-blur
		// that would destroy high-frequency / small-object features when several
		// such ViTs are stacked.
		const bool same_res_overlap = (S == 1) && (P >= 1) && (P % 2 == 1) && (pad == (P - 1) / 2);
		if (same_res_overlap)
		{
			const int center = (P / 2) * P + (P / 2);
			for (int oc = 0; oc < C; ++oc)
			{
				l.vit_patch_embed[oc * K + center * C + oc] = 1.0f;
			}
			return true;
		}

		// Classic non-overlap: per-channel average over the P*P window (box-blur,
		// reduces to identity when P==1).
		const float scale = 1.0f / static_cast<float>(P * P);
		for (int oc = 0; oc < C; ++oc)
		{
			for (int dy = 0; dy < P; ++dy)
			{
				for (int dx = 0; dx < P; ++dx)
				{
					const int patch_offset = (dy * P + dx) * C + oc;
					l.vit_patch_embed[oc * K + patch_offset] = scale;
				}
			}
		}
		return true;
	}

	static void vit_patchify_cpu(const Darknet::Layer & l, const float *input, float *patches)
	{
		const int B = l.batch;
		const int C = l.c;
		const int H = l.h;
		const int W = l.w;
		const int P = l.vit_patch_size;
		const int S = l.vit_patch_stride;
		const int pad = l.vit_patch_pad;
		const int Hp = l.out_h;
		const int Wp = l.out_w;
		const int T = Hp * Wp;
		const int K = vit_patch_dim(l);

		for (int b = 0; b < B; ++b)
		{
			for (int py = 0; py < Hp; ++py)
			{
				for (int px = 0; px < Wp; ++px)
				{
					const int t = py * Wp + px;
					for (int dy = 0; dy < P; ++dy)
					{
						for (int dx = 0; dx < P; ++dx)
						{
							const int y = py * S + dy - pad;
							const int x = px * S + dx - pad;
							const int patch_offset = (dy * P + dx) * C;
							const bool in_bounds = (y >= 0 && y < H && x >= 0 && x < W);
							if (in_bounds)
							{
								const int spatial = y * W + x;
								for (int c = 0; c < C; ++c)
								{
									patches[(b * T + t) * K + patch_offset + c] =
										input[(b * C + c) * H * W + spatial];
								}
							}
							else
							{
								for (int c = 0; c < C; ++c)
								{
									patches[(b * T + t) * K + patch_offset + c] = 0.0f;
								}
							}
						}
					}
				}
			}
		}
	}

	static void vit_patch_embed_backward_cpu(Darknet::Layer & l, Darknet::NetworkState state, const float *d_embed)
	{
		if (d_embed == nullptr)
		{
			return;
		}

	const int B = l.batch;
	const int N = l.n;
	const int T = l.out_h * l.out_w;
	const int K = vit_patch_dim(l);
	float *d_embed_mut = const_cast<float*>(d_embed);

	if (state.input != nullptr)
	{
		std::vector<float> patches(static_cast<size_t>(B) * T * K);
		vit_patchify_cpu(l, state.input, patches.data());
		gemm_cpu(1, 0, N, K, B * T, 1.0f,
			d_embed_mut, N,
			patches.data(), K,
			1.0f,
			l.vit_patch_embed_updates, K);
		}

		for (int i = 0; i < B * T; ++i)
		{
			for (int n = 0; n < N; ++n)
			{
				l.vit_patch_bias_updates[n] += d_embed[i * N + n];
			}
		}

		if (state.delta != nullptr)
		{
			std::vector<float> d_patches(static_cast<size_t>(B) * T * K);
			gemm_cpu(0, 0, B * T, K, N, 1.0f,
				d_embed_mut, N,
				l.vit_patch_embed, K,
				0.0f,
				d_patches.data(), K);

			const int C = l.c;
			const int H = l.h;
			const int W = l.w;
			const int P = l.vit_patch_size;
			const int S = l.vit_patch_stride;
			const int pad = l.vit_patch_pad;
			const int Hp = l.out_h;
			const int Wp = l.out_w;
			for (int b = 0; b < B; ++b)
			{
				for (int py = 0; py < Hp; ++py)
				{
					for (int px = 0; px < Wp; ++px)
					{
						const int t = py * Wp + px;
						for (int dy = 0; dy < P; ++dy)
						{
							for (int dx = 0; dx < P; ++dx)
							{
								const int y = py * S + dy - pad;
								const int x = px * S + dx - pad;
								if (y < 0 || y >= H || x < 0 || x >= W) continue;
								const int patch_offset = (dy * P + dx) * C;
								const int spatial = y * W + x;
								for (int c = 0; c < C; ++c)
								{
									state.delta[(b * C + c) * H * W + spatial] +=
										d_patches[(b * T + t) * K + patch_offset + c];
								}
							}
						}
					}
				}
			}
			check_nan_cpu("backward: propagated patch delta", state.delta, B * C * H * W, l.index);
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
		const float s = sanitize_and_constrain_value(skip[i], VIT_FEATURE_CLAMP);
		const float br = sanitize_and_constrain_value(branch[i], VIT_FEATURE_CLAMP);
		out[i] = sanitize_and_constrain_value(a * s + b * br, VIT_FEATURE_CLAMP);
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
		const float g = sanitize_and_constrain_value(dout[i], VIT_GRAD_CLAMP);
		const float s = sanitize_and_constrain_value(skip[i], VIT_FEATURE_CLAMP);
		const float br = sanitize_and_constrain_value(branch[i], VIT_FEATURE_CLAMP);
		d_skip[i] = sanitize_and_constrain_value(skip_coeff * g, VIT_GRAD_CLAMP);
		d_branch[i] = sanitize_and_constrain_value(branch_coeff * g, VIT_GRAD_CLAMP);
		grad_post_skip += g * (p * s + (1.0f - p) * br);
		grad_post_branch += g * ((1.0f - p) * s + p * br);
		grad_p += g * (post_skip - post_branch) * (s - br);
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

static inline void softmax_row(float *x, int n)
{
	// Guard against NaN/Inf *before* exp() is called.
	// exp() on large positive values overflows to inf; on large negative values
	// underflows to 0.  Both poison the gradient.  A narrow logit clamp keeps
	// attention finite without changing the row-wise softmax contract.
	const float clip = VIT_ATTENTION_SCORE_CLAMP;
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

	// Numerically stable softmax: subtract the row maximum before exp().
	// Mathematically equivalent to the naive formula because the constant
	// cancels: exp(x_i - max) / Σ exp(x_j - max) == exp(x_i) / Σ exp(x_j).
	// The benefit is that the largest exp() argument is now 0, so exp() never
	// overflows and precision is maximised for the dominant probabilities.
	float max_val = x[0];
	for (int i = 1; i < n; i++) max_val = std::max(max_val, x[i]);
	float sum = 0.0f;
	for (int i = 0; i < n; i++)
	{
		x[i] = std::exp(x[i] - max_val);
		sum += x[i];
	}
	// Degenerate case: all inputs were clipped to the same value → all exp()
	// results are 1 → sum = n (shouldn't reach here), but also handles cases
	// where every input was -clip so exp underflows.  Uniform is the safest
	// uninformative prior — at least it doesn't produce NaN.
	if (!std::isfinite(sum) || sum <= 0.0f)
	{
		const float uniform = 1.0f / n;
		for (int i = 0; i < n; i++) x[i] = uniform;
		return;
	}
	// +1e-9 is a second safety net: prevents a hard divide-by-zero if the
	// float sum rounds to exactly zero despite the guard above.
	float inv_sum = 1.0f / (sum + 1e-9f);
	for (int i = 0; i < n; i++) x[i] *= inv_sum;
}

// LayerNorm forward: for each token i, normalises across the C channel dimension,
// then applies a learned affine rescale (gamma, beta).
//
// Unlike BatchNorm (which normalises across the batch dimension), LayerNorm
// works independently for each token.  That makes it suitable for variable-
// sequence-length transformers where batch statistics are unreliable.
//
// We save mean[i], var[i], and the normalised xhat[i*C..] because the backward
// pass needs all three to compute the correct gradient without re-doing the
// forward pass.  Skipping the save would cost an extra forward pass per backward.
static void layernorm_forward(const float *x, float *out, float *mean, float *var, float *xhat,
	const float *gamma, const float *beta, int total_tokens, int C)
{
	const float eps = 1e-5f;   // prevents divide-by-zero when all channels are identical
	for (int i = 0; i < total_tokens; i++)
	{
		const float *xi = x + i * C;
		float *oi = out + i * C;
		float *xh = xhat + i * C;

		// Step 1: compute per-token mean across C channels
		float m = 0.0f;
		for (int j = 0; j < C; j++) m += xi[j];
		m /= C;
		mean[i] = m;   // saved for backward

		// Step 2: compute variance (mean-subtracted squared deviations)
		float v = 0.0f;
		for (int j = 0; j < C; j++)
		{
			float d = xi[j] - m;
			v += d * d;
		}
		v /= C;
		var[i] = v;   // saved for backward (used to reconstruct inv_std)

		// Step 3: normalise and apply affine transform
		// xhat is saved because the backward pass needs (xi - m) / std without
		// recomputing mean and variance a second time.
		float inv_std = 1.0f / std::sqrt(v + eps);
		for (int j = 0; j < C; j++)
		{
			xh[j] = (xi[j] - m) * inv_std;   // unit-variance, zero-mean
			oi[j] = xh[j] * gamma[j] + beta[j];  // learned rescale
		}
	}
}

// LayerNorm backward: given dL/d(out), compute dL/dx, dL/dgamma, dL/dbeta.
//
// The formula is non-trivial because the mean and variance in the forward pass
// both depend on x, so we must account for their gradient contributions.
//
// Full derivation (per token i):
//   Let xhat_j = (x_j - mean) / std,  out_j = gamma_j * xhat_j + beta_j
//
//   dL/dgamma_j = Σ_tokens  dout_j * xhat_j       (sum over all tokens)
//   dL/dbeta_j  = Σ_tokens  dout_j
//
//   For dL/dx_j we chain through xhat, mean, and var simultaneously:
//
//     dL/dxhat_j = dout_j * gamma_j
//
//     dL/dx_j = (1/std) * [ dL/dxhat_j
//                          - (1/C) * Σ_k dL/dxhat_k          ← mean gradient
//                          - (xhat_j/C) * Σ_k dL/dxhat_k * xhat_k ]  ← var gradient
//
//   Which collapses to the single-pass formula used below:
//     dxi[j] = inv_std * (dxhat - (sum_dxhat + xhi[j] * dot_dxhat_xhat) / C)
//
// Note: dgamma/dbeta use += because they accumulate gradients across all tokens
// (all tokens share the same gamma/beta parameters).
static void layernorm_backward(const float *dout, const float *xhat, const float *var,
	const float *gamma, float *dx, float *dgamma, float *dbeta, int total_tokens, int C)
{
	const float eps = 1e-5f;

	for (int i = 0; i < total_tokens; i++)
	{
		const float *doi = dout + i * C;
		const float *xhi = xhat + i * C;
		float *dxi = dx + i * C;
		float inv_std = 1.0f / std::sqrt(var[i] + eps);

		// Affine param grads: accumulated over all tokens (same params shared)
		for (int j = 0; j < C; j++)
		{
			dgamma[j] += doi[j] * xhi[j];
			dbeta[j] += doi[j];
		}

		// Pre-compute the two scalar reductions needed by the dx formula
		float sum_dxhat = 0.0f;        // Σ_k dL/dxhat_k  (mean-gradient term)
		float dot_dxhat_xhat = 0.0f;   // Σ_k dL/dxhat_k * xhat_k  (var-gradient term)
		for (int j = 0; j < C; j++)
		{
			const float dxhat = doi[j] * gamma[j];
			sum_dxhat += dxhat;
			dot_dxhat_xhat += dxhat * xhi[j];
		}
		// Apply the full LN gradient — the /C terms correct for the mean and
		// variance being computed *from* x rather than being constants.
		for (int j = 0; j < C; j++)
		{
			const float dxhat = doi[j] * gamma[j];
			dxi[j] = inv_std * (dxhat - (sum_dxhat + xhi[j] * dot_dxhat_xhat) / C);
		}
	}
}

// backward_vit_attention_tail — computes gradients through the QKV projection
// and LN1 sublayer, and optionally fuses the first residual skip gradient.
//
// Why a separate function?  The GPU and CPU backward paths share this logic,
// so it was factored out to avoid duplication.
//
// What it does (in order):
//   1. For each attention head: backprop through V mixing, softmax, and QK^T
//      to obtain dQ, dK, dV.
//   2. Accumulate QKV bias gradients (bias_updates) by summing dQKV over tokens.
//   3. Re-run LN1 forward on vit_pre_res1 to obtain ln1_b — needed for the
//      weight grad computation (we did NOT cache ln1_out during forward to save
//      memory, so we pay with a re-computation here).
//   4. Backprop through the QKV weight matrix to get d_ln1_out and update
//      weight_updates.
//   5. Run layernorm_backward to obtain d_pre_ln1.
//   6. If d_skip is non-null, add it to d_pre_ln1 so both residual branches
//      are properly accumulated.
//   7. Accumulate positional embedding gradients.
//   8. Backprop through patch embedding and scatter gradients back to NCHW.
void backward_vit_attention_tail(Darknet::Layer & l, Darknet::NetworkState state, const float *d_attn_out, const float *d_skip)
{
	const int B = l.batch;
	const int N = l.n;
	const int T = l.out_h * l.out_w;
	const int heads = l.vit_heads;
	const int d = l.vit_head_dim;
	const float scale = 1.0f / std::sqrt((float)d);

	std::vector<float> d_ln1_out(B * T * N, 0.0f);
	std::vector<float> d_qkv_buf(T * 3 * N);

	for (int b = 0; b < B; b++)
	{
		float *qkv_b = l.vit_qkv_out + b * T * 3 * N;
		const float *d_attn_out_b = d_attn_out + b * T * N;

		std::fill(d_qkv_buf.begin(), d_qkv_buf.end(), 0.0f);

		for (int h_idx = 0; h_idx < heads; h_idx++)
		{
			std::vector<float> Q(T * d), K_mat(T * d), V(T * d);
			for (int t = 0; t < T; t++)
			{
				for (int dd = 0; dd < d; dd++)
				{
					Q[t * d + dd] = qkv_b[t * 3 * N + h_idx * d + dd];
					K_mat[t * d + dd] = qkv_b[t * 3 * N + N + h_idx * d + dd];
					V[t * d + dd] = qkv_b[t * 3 * N + 2 * N + h_idx * d + dd];
				}
			}
			sanitize_and_constrain_cpu(Q.data(), T * d, VIT_ATTENTION_QK_CLAMP);
			sanitize_and_constrain_cpu(K_mat.data(), T * d, VIT_ATTENTION_QK_CLAMP);

			std::vector<float> d_head(T * d);
			for (int t = 0; t < T; t++)
			{
				for (int dd = 0; dd < d; dd++)
				{
					d_head[t * d + dd] = d_attn_out_b[t * N + h_idx * d + dd];
				}
			}

			float *scores = l.vit_attn_scores + (b * heads + h_idx) * T * T;

			// ── Attention backward (per head) ─────────────────────────────────
			// Forward was:  ctx = softmax(Q @ K^T / sqrt(d)) @ V
			//
			// dV = scores^T @ d_head
			//   Each value token contributed to all queries proportional to the
			//   attention weight; the transpose routes credit back accordingly.
			std::vector<float> dV(T * d);
			gemm_cpu(1, 0, T, d, T, 1.0f,
				scores, T,
				d_head.data(), d,
				0.0f,
				dV.data(), d);

			// d_scores = d_head @ V^T
			//   Gradient w.r.t. the post-softmax attention weights.
			std::vector<float> d_scores(T * T);
			gemm_cpu(0, 1, T, T, d, 1.0f,
				d_head.data(), d,
				V.data(), d,
				0.0f,
				d_scores.data(), T);

			// Softmax backward (row-wise Jacobian):
			//   For row p (post-softmax) and upstream grad g:
			//     d_pre[j] = p[j] * (g[j] - dot(g, p))
			//   The dot term subtracts the component of g that is "aligned" with
			//   the probability vector, enforcing the constraint that softmax
			//   outputs sum to 1.
			std::vector<float> d_pre_softmax(T * T);
			for (int t = 0; t < T; t++)
			{
				float dot = 0.0f;
				for (int j = 0; j < T; j++) dot += d_scores[t * T + j] * scores[t * T + j];
				for (int j = 0; j < T; j++) d_pre_softmax[t * T + j] = scores[t * T + j] * (d_scores[t * T + j] - dot);
			}

			// dQ = d_pre_softmax @ K * scale
			//   Forward: pre_softmax = Q @ K^T * scale → dQ = d_pre @ K * scale
			std::vector<float> dQ(T * d);
			gemm_cpu(0, 0, T, d, T, scale,
				d_pre_softmax.data(), T,
				K_mat.data(), d,
				0.0f,
				dQ.data(), d);

			// dK = d_pre_softmax^T @ Q * scale
			//   K appeared transposed in the forward (Q @ K^T), so the gradient
			//   requires the transpose of d_pre_softmax.
			std::vector<float> dK(T * d);
			gemm_cpu(1, 0, T, d, T, scale,
				d_pre_softmax.data(), T,
				Q.data(), d,
				0.0f,
				dK.data(), d);

			for (int t = 0; t < T; t++)
			{
				for (int dd = 0; dd < d; dd++)
				{
					d_qkv_buf[t * 3 * N + h_idx * d + dd] += dQ[t * d + dd];
					d_qkv_buf[t * 3 * N + N + h_idx * d + dd] += dK[t * d + dd];
					d_qkv_buf[t * 3 * N + 2 * N + h_idx * d + dd] += dV[t * d + dd];
				}
			}
		}
		sanitize_and_constrain_cpu(d_qkv_buf.data(), T * 3 * N, VIT_GRAD_CLAMP);

		for (int t = 0; t < T; t++)
		{
			for (int j = 0; j < 3 * N; j++)
			{
				l.bias_updates[j] += d_qkv_buf[t * 3 * N + j];
			}
		}

		// Re-run LN1 forward to recover ln1_b (the LN1 output for this batch
		// element).  We need it to compute the weight gradient:
		//   dW_qkv = d_qkv^T @ ln1_b
		// We did NOT cache ln1_out during forward (to save memory), so we pay
		// for it with one extra forward pass per backward call.
		std::vector<float> ln1_b(T * N);
		std::vector<float> ln1_xhat_tmp(T * N);
		std::vector<float> ln1_mean_tmp(T);
		std::vector<float> ln1_var_tmp(T);
		layernorm_forward(l.vit_pre_res1 + b * T * N, ln1_b.data(),
			ln1_mean_tmp.data(), ln1_var_tmp.data(), ln1_xhat_tmp.data(),
			l.vit_ln1_gamma, l.vit_ln1_beta, T, N);

		gemm_cpu(1, 0, 3 * N, N, T, 1.0f,
			d_qkv_buf.data(), 3 * N,
			ln1_b.data(), N,
			1.0f,
			l.weight_updates, N);

		float *d_ln1_b = d_ln1_out.data() + b * T * N;
		gemm_cpu(0, 0, T, N, 3 * N, 1.0f,
			d_qkv_buf.data(), 3 * N,
			l.weights, N,
			0.0f,
			d_ln1_b, N);
	}
	sanitize_and_constrain_cpu(d_ln1_out.data(), B * T * N, VIT_GRAD_CLAMP);
	check_nan_cpu("backward: after attention backward", d_ln1_out.data(), B * T * N, l.index);

	std::vector<float> d_pre_ln1(B * T * N);
	layernorm_backward(d_ln1_out.data(), l.vit_ln1_xhat, l.vit_ln1_var,
		l.vit_ln1_gamma, d_pre_ln1.data(), l.vit_ln1_gamma_updates, l.vit_ln1_beta_updates,
		B * T, N);

	if (d_skip != nullptr)
	{
		for (int i = 0; i < B * T * N; i++)
		{
			d_pre_ln1[i] += d_skip[i];
		}
	}
	sanitize_and_constrain_cpu(d_pre_ln1.data(), B * T * N, VIT_GRAD_CLAMP);
	check_nan_cpu("backward: after layernorm1 backward", d_pre_ln1.data(), B * T * N, l.index);

	if (l.vit_pos_embed_type == VIT_POS_EMBED_LEARNED)
	{
		for (int b = 0; b < B; b++)
		{
			for (int t = 0; t < T; t++)
			{
				for (int n = 0; n < N; n++)
				{
					l.vit_pos_embed_updates[t * N + n] += d_pre_ln1[b * T * N + t * N + n];
				}
			}
		}
	}

	vit_patch_embed_backward_cpu(l, state, d_pre_ln1.data());
}

// ─── make ─────────────────────────────────────────────────────────────────────

Darknet::Layer make_vit_layer(int batch, int h, int w, int c, int n,
	int patch_size, int patch_stride, int patch_pad,
	int heads, int ffn_ratio, int mlp_dim, int pos_embed_type, int pos_init_type,
	ACTIVATION activation, int index, int train)
{
	TAT(TATPARMS);

	vit_validate_shape(h, w, patch_size, patch_stride, patch_pad, n, heads, mlp_dim);
	if (pos_embed_type != VIT_POS_EMBED_LEARNED && pos_embed_type != VIT_POS_EMBED_SINUSOIDAL)
	{
		darknet_fatal_error(DARKNET_LOC, "vit: unsupported pos_embed_type=%d", pos_embed_type);
	}
	if (pos_init_type != VIT_POS_INIT_RANDOM && pos_init_type != VIT_POS_INIT_ZERO)
	{
		darknet_fatal_error(DARKNET_LOC, "vit: unsupported pos_init_type=%d", pos_init_type);
	}

	Darknet::Layer l = { (Darknet::ELayerType)0 };
	l.type = Darknet::ELayerType::VIT;
	l.batch = batch;
	l.h = h;
	l.w = w;
	l.c = c;
	l.n = n;
	l.out_h = (h + 2 * patch_pad - patch_size) / patch_stride + 1;
	l.out_w = (w + 2 * patch_pad - patch_size) / patch_stride + 1;
	l.out_c = n;
	l.outputs = l.out_h * l.out_w * l.out_c;
	l.inputs = l.w * l.h * l.c;
	l.index = index;
	l.train = train;

	l.vit_patch_size = patch_size;
	l.vit_patch_stride = patch_stride;
	l.vit_patch_pad = patch_pad;
	l.vit_heads = heads;
	l.vit_head_dim = n / heads;
	l.vit_ffn_ratio = ffn_ratio;
	l.vit_mlp_dim = mlp_dim;
	l.vit_pos_embed_type = pos_embed_type;
	l.vit_pos_init_type = pos_init_type;
	l.activation = activation;

	l.forward = forward_vit_layer;
	l.backward = backward_vit_layer;
	l.update = update_vit_layer;

#ifdef DARKNET_GPU
	l.forward_gpu = forward_vit_layer_gpu;
	l.backward_gpu = backward_vit_layer_gpu;
	l.update_gpu = update_vit_layer_gpu;
#endif

	const int T = l.out_h * l.out_w;
	const int patch_dim = vit_patch_dim(l);
	
	// Patch embedding weights: flatten each P x P x C patch and project to filters.
	l.vit_patch_embed = (float*)xcalloc(n * patch_dim, sizeof(float));
	l.vit_patch_embed_updates = (float*)xcalloc(n * patch_dim, sizeof(float));
	l.vit_patch_bias = (float*)xcalloc(n, sizeof(float));
	l.vit_patch_bias_updates = (float*)xcalloc(n, sizeof(float));

	// QKV weights
	l.nweights = 3 * n * n;
	l.nbiases = 3 * n;
	l.weights = (float*)xcalloc(l.nweights, sizeof(float));
	l.weight_updates = (float*)xcalloc(l.nweights, sizeof(float));
	l.biases = (float*)xcalloc(l.nbiases, sizeof(float));
	l.bias_updates = (float*)xcalloc(l.nbiases, sizeof(float));

	// Output projection
	l.vit_wo = (float*)xcalloc(n * n, sizeof(float));
	l.vit_wo_updates = (float*)xcalloc(n * n, sizeof(float));
	l.vit_wo_bias = (float*)xcalloc(n, sizeof(float));
	l.vit_wo_bias_updates = (float*)xcalloc(n, sizeof(float));

	// LayerNorm 1
	l.vit_ln1_gamma = (float*)xcalloc(n, sizeof(float));
	l.vit_ln1_gamma_updates = (float*)xcalloc(n, sizeof(float));
	l.vit_ln1_beta = (float*)xcalloc(n, sizeof(float));
	l.vit_ln1_beta_updates = (float*)xcalloc(n, sizeof(float));

	// LayerNorm 2
	l.vit_ln2_gamma = (float*)xcalloc(n, sizeof(float));
	l.vit_ln2_gamma_updates = (float*)xcalloc(n, sizeof(float));
	l.vit_ln2_beta = (float*)xcalloc(n, sizeof(float));
	l.vit_ln2_beta_updates = (float*)xcalloc(n, sizeof(float));

	// FFN
	l.vit_ffn_w1 = (float*)xcalloc(n * mlp_dim, sizeof(float));
	l.vit_ffn_w1_updates = (float*)xcalloc(n * mlp_dim, sizeof(float));
	l.vit_ffn_b1 = (float*)xcalloc(mlp_dim, sizeof(float));
	l.vit_ffn_b1_updates = (float*)xcalloc(mlp_dim, sizeof(float));

	l.vit_ffn_w2 = (float*)xcalloc(n * mlp_dim, sizeof(float));
	l.vit_ffn_w2_updates = (float*)xcalloc(n * mlp_dim, sizeof(float));
	l.vit_ffn_b2 = (float*)xcalloc(n, sizeof(float));
	l.vit_ffn_b2_updates = (float*)xcalloc(n, sizeof(float));

	// Absolute Positional Embedding
	l.vit_pos_embed = (float*)xcalloc(T * n, sizeof(float));
	l.vit_pos_embed_updates = (float*)xcalloc(T * n, sizeof(float));

	// mHC residual mixer parameters.  Reuses generic scale storage because ViT
	// does not use batch-norm scales.
	l.scales = (float*)xcalloc(MHC_PARAM_COUNT, sizeof(float));
	l.scale_updates = (float*)xcalloc(MHC_PARAM_COUNT, sizeof(float));
	init_mhc_residual_params(l.scales);

	// ── Weight initialisation ────────────────────────────────────────────────

	// Patch projection maps flattened P*P*C patches into the ViT embedding width N.
	// If N == C, initialize as a drop-in per-channel average over each patch
	// (identity when P == 1).  This avoids destroying pretrained detector
	// features when a ViT block replaces a maxpool/conv in an existing cfg.
	if (!vit_init_dropin_patch_embed(l))
	{
		float scale_patch = std::sqrt(2.0f / (patch_dim + n));
		for (int i = 0; i < n * patch_dim; ++i) l.vit_patch_embed[i] = scale_patch * rand_uniform_weight_init(-1, 1);
	}

	// Xavier / Glorot uniform init for QKV: scale = sqrt(2 / (fan_in + fan_out)).
	// fan_in=N, fan_out=3N because the weight matrix maps N → 3N (Q, K, V stacked).
	// This keeps the variance of activations and gradients roughly equal at init.
	float scale_qkv = std::sqrt(2.0f / (n + 3 * n));
	for(int i = 0; i < l.nweights; ++i) l.weights[i] = scale_qkv * rand_uniform_weight_init(-1, 1);

	// Output projection Wo maps N → N; same Xavier formula.
	float scale_o = std::sqrt(2.0f / (n + n));
	for(int i = 0; i < n * n; ++i) l.vit_wo[i] = scale_o * rand_uniform_weight_init(-1, 1);

	// LayerNorm starts as identity: gamma=1, beta=0 (already zeroed by xcalloc).
	// This means the network begins training without any scaling distortion.
	for(int i = 0; i < n; ++i) l.vit_ln1_gamma[i] = 1.0f;
	for(int i = 0; i < n; ++i) l.vit_ln2_gamma[i] = 1.0f;

	// FFN up-projection: fan_in=N → Xavier scale = sqrt(2/N)
	float scale_w1 = std::sqrt(2.0f / n);
	for(int i = 0; i < n * mlp_dim; ++i) l.vit_ffn_w1[i] = scale_w1 * rand_uniform_weight_init(-1, 1);

	// FFN down-projection: fan_in = mlp_dim
	float scale_w2 = std::sqrt(2.0f / mlp_dim);
	for(int i = 0; i < n * mlp_dim; ++i) l.vit_ffn_w2[i] = scale_w2 * rand_uniform_weight_init(-1, 1);

	if (l.vit_pos_embed_type == VIT_POS_EMBED_SINUSOIDAL)
	{
		vit_fill_2d_sinusoidal_pos_embed(l.vit_pos_embed, l.out_h, l.out_w, n);
	}
	else
	{
		if (l.vit_pos_init_type == VIT_POS_INIT_ZERO)
		{
			std::fill(l.vit_pos_embed, l.vit_pos_embed + T * n, 0.0f);
		}
		else
		{
			// Positional embeddings: small random init (ViT paper uses 0.02 std).
			// Absolute PE encodes the spatial address of each token; starting small
			// lets the network learn the right scale rather than forcing one.
			for(int i = 0; i < T * n; ++i) l.vit_pos_embed[i] = 0.02f * rand_uniform_weight_init(-1, 1);
		}
	}

	// Runtime buffers
	l.vit_qkv_out = (float*)xcalloc(batch * T * 3 * n, sizeof(float));
	l.vit_attn_scores = (float*)xcalloc(batch * heads * T * T, sizeof(float));
	l.vit_attn_out = (float*)xcalloc(batch * T * n, sizeof(float));
	l.vit_ffn_hidden = (float*)xcalloc(batch * T * mlp_dim, sizeof(float));
	l.activation_input = (float*)xcalloc(batch * T * mlp_dim, sizeof(float));
	l.vit_ln1_mean = (float*)xcalloc(batch * T, sizeof(float));
	l.vit_ln1_var = (float*)xcalloc(batch * T, sizeof(float));
	l.vit_ln2_mean = (float*)xcalloc(batch * T, sizeof(float));
	l.vit_ln2_var = (float*)xcalloc(batch * T, sizeof(float));
	l.vit_ln1_xhat = (float*)xcalloc(batch * T * n, sizeof(float));
	l.vit_ln2_xhat = (float*)xcalloc(batch * T * n, sizeof(float));
	
	l.vit_pre_res1 = (float*)xcalloc(batch * T * n, sizeof(float));
	l.vit_pre_res2 = (float*)xcalloc(batch * T * n, sizeof(float));
	l.x = (float*)xcalloc(batch * T * n, sizeof(float));       // mHC residual1 branch cache
	l.x_norm = (float*)xcalloc(batch * T * n, sizeof(float));  // mHC residual2 branch cache

	l.output = (float*)xcalloc(batch * l.outputs, sizeof(float));
	l.delta = (float*)xcalloc(batch * l.outputs, sizeof(float));

#ifdef DARKNET_GPU
	l.vit_patch_embed_gpu = cuda_make_array(l.vit_patch_embed, n * patch_dim);
	l.vit_patch_embed_updates_gpu = cuda_make_array(l.vit_patch_embed_updates, n * patch_dim);
	l.vit_patch_bias_gpu = cuda_make_array(l.vit_patch_bias, n);
	l.vit_patch_bias_updates_gpu = cuda_make_array(l.vit_patch_bias_updates, n);

	l.weights_gpu = cuda_make_array(l.weights, l.nweights);
	l.weight_updates_gpu = cuda_make_array(l.weight_updates, l.nweights);
	l.biases_gpu = cuda_make_array(l.biases, l.nbiases);
	l.bias_updates_gpu = cuda_make_array(l.bias_updates, l.nbiases);

	l.vit_wo_gpu = cuda_make_array(l.vit_wo, n * n);
	l.vit_wo_updates_gpu = cuda_make_array(l.vit_wo_updates, n * n);
	l.vit_wo_bias_gpu = cuda_make_array(l.vit_wo_bias, n);
	l.vit_wo_bias_updates_gpu = cuda_make_array(l.vit_wo_bias_updates, n);

	l.vit_ln1_gamma_gpu = cuda_make_array(l.vit_ln1_gamma, n);
	l.vit_ln1_gamma_updates_gpu = cuda_make_array(l.vit_ln1_gamma_updates, n);
	l.vit_ln1_beta_gpu = cuda_make_array(l.vit_ln1_beta, n);
	l.vit_ln1_beta_updates_gpu = cuda_make_array(l.vit_ln1_beta_updates, n);

	l.vit_ln2_gamma_gpu = cuda_make_array(l.vit_ln2_gamma, n);
	l.vit_ln2_gamma_updates_gpu = cuda_make_array(l.vit_ln2_gamma_updates, n);
	l.vit_ln2_beta_gpu = cuda_make_array(l.vit_ln2_beta, n);
	l.vit_ln2_beta_updates_gpu = cuda_make_array(l.vit_ln2_beta_updates, n);

	l.vit_ffn_w1_gpu = cuda_make_array(l.vit_ffn_w1, n * mlp_dim);
	l.vit_ffn_w1_updates_gpu = cuda_make_array(l.vit_ffn_w1_updates, n * mlp_dim);
	l.vit_ffn_b1_gpu = cuda_make_array(l.vit_ffn_b1, mlp_dim);
	l.vit_ffn_b1_updates_gpu = cuda_make_array(l.vit_ffn_b1_updates, mlp_dim);

	l.vit_ffn_w2_gpu = cuda_make_array(l.vit_ffn_w2, n * mlp_dim);
	l.vit_ffn_w2_updates_gpu = cuda_make_array(l.vit_ffn_w2_updates, n * mlp_dim);
	l.vit_ffn_b2_gpu = cuda_make_array(l.vit_ffn_b2, n);
	l.vit_ffn_b2_updates_gpu = cuda_make_array(l.vit_ffn_b2_updates, n);

	l.vit_pos_embed_gpu = cuda_make_array(l.vit_pos_embed, T * n);
	l.vit_pos_embed_updates_gpu = cuda_make_array(l.vit_pos_embed_updates, T * n);
	l.scales_gpu = cuda_make_array(l.scales, MHC_PARAM_COUNT);
	l.scale_updates_gpu = cuda_make_array(l.scale_updates, MHC_PARAM_COUNT);

	l.vit_qkv_out_gpu = cuda_make_array(l.vit_qkv_out, batch * T * 3 * n);
	l.vit_attn_scores_gpu = cuda_make_array(l.vit_attn_scores, batch * heads * T * T);
	l.vit_attn_out_gpu = cuda_make_array(l.vit_attn_out, batch * T * n);
	l.vit_ffn_hidden_gpu = cuda_make_array(l.vit_ffn_hidden, batch * T * mlp_dim);
	l.activation_input_gpu = cuda_make_array(nullptr, batch * T * mlp_dim);
	l.vit_ln1_mean_gpu = cuda_make_array(l.vit_ln1_mean, batch * T);
	l.vit_ln1_var_gpu = cuda_make_array(l.vit_ln1_var, batch * T);
	l.vit_ln2_mean_gpu = cuda_make_array(l.vit_ln2_mean, batch * T);
	l.vit_ln2_var_gpu = cuda_make_array(l.vit_ln2_var, batch * T);
	l.vit_ln1_xhat_gpu = cuda_make_array(l.vit_ln1_xhat, batch * T * n);
	l.vit_ln2_xhat_gpu = cuda_make_array(l.vit_ln2_xhat, batch * T * n);

	l.vit_pre_res1_gpu = cuda_make_array(l.vit_pre_res1, batch * T * n);
	l.vit_pre_res2_gpu = cuda_make_array(l.vit_pre_res2, batch * T * n);
	l.x_gpu = cuda_make_array(nullptr, batch * T * n);
	l.x_norm_gpu = cuda_make_array(nullptr, batch * T * n);
	l.vit_patch_tokens_gpu = cuda_make_array(nullptr, batch * T * patch_dim);
	l.vit_patch_delta_gpu = cuda_make_array(nullptr, batch * T * patch_dim);
	l.vit_tmp_token_c1_gpu = cuda_make_array(nullptr, batch * T * n);
	l.vit_tmp_token_c2_gpu = cuda_make_array(nullptr, batch * T * n);
	l.vit_tmp_token_n1_gpu = cuda_make_array(nullptr, batch * T * n);
	l.vit_tmp_token_n2_gpu = cuda_make_array(nullptr, batch * T * n);
	l.vit_tmp_token_n3_gpu = cuda_make_array(nullptr, batch * T * n);
	l.vit_tmp_ffn_hidden_gpu = cuda_make_array(nullptr, batch * T * mlp_dim);
	l.vit_tmp_head1_gpu = cuda_make_array(nullptr, batch * heads * T * l.vit_head_dim);
	l.vit_tmp_head2_gpu = cuda_make_array(nullptr, batch * heads * T * l.vit_head_dim);
	l.vit_tmp_head3_gpu = cuda_make_array(nullptr, batch * heads * T * l.vit_head_dim);
	l.vit_tmp_head4_gpu = cuda_make_array(nullptr, batch * heads * T * l.vit_head_dim);
	l.vit_tmp_head5_gpu = cuda_make_array(nullptr, batch * heads * T * l.vit_head_dim);
	l.vit_tmp_scores_gpu = cuda_make_array(nullptr, batch * heads * T * T);

	l.output_gpu = cuda_make_array(l.output, batch * l.outputs);
	l.delta_gpu = cuda_make_array(l.delta, batch * l.outputs);
#endif

	if (cfg_and_state.output)
	{
		*cfg_and_state.output << "vit             "
			<< l.c << " x " << l.w << " x " << l.h << " -> "
			<< l.out_c << " x " << l.out_w << " x " << l.out_h
			<< " (patch=" << patch_size << " stride=" << patch_stride << " pad=" << patch_pad
			<< ", heads=" << heads << ", mlp_dim=" << mlp_dim
			<< ", pos_embed=" << (pos_embed_type == VIT_POS_EMBED_SINUSOIDAL ? "sinusoidal" : "learned")
			<< ", pos_init=" << (pos_init_type == VIT_POS_INIT_ZERO ? "zero" : "random") << ")"
			<< std::endl;
	}

	return l;
}

// ─── resize ───────────────────────────────────────────────────────────────────

void resize_vit_layer(Darknet::Layer * l, int w, int h)
{
	vit_validate_shape(h, w, l->vit_patch_size, l->vit_patch_stride, l->vit_patch_pad,
		l->n, l->vit_heads, l->vit_mlp_dim);

	int old_H = l->out_h;
	int old_W = l->out_w;
	const int new_out_h = (h + 2 * l->vit_patch_pad - l->vit_patch_size) / l->vit_patch_stride + 1;
	const int new_out_w = (w + 2 * l->vit_patch_pad - l->vit_patch_size) / l->vit_patch_stride + 1;
	int T = new_out_h * new_out_w;
	const int N = l->n;
	const int patch_dim = vit_patch_dim(*l);
	l->w = w;
	l->h = h;
	l->out_w = new_out_w;
	l->out_h = new_out_h;
	l->inputs = w * h * l->c;
	l->outputs = l->out_w * l->out_h * l->out_c;

	if (old_H != l->out_h || old_W != l->out_w)
	{
		l->vit_pos_embed = (float*)xrealloc(l->vit_pos_embed, T * N * sizeof(float));
		l->vit_pos_embed_updates = (float*)xrealloc(l->vit_pos_embed_updates, T * N * sizeof(float));
		if (l->vit_pos_embed_type == VIT_POS_EMBED_SINUSOIDAL)
		{
			vit_fill_2d_sinusoidal_pos_embed(l->vit_pos_embed, l->out_h, l->out_w, N);
		}
		else
		{
			// CPU path: zero-init (no CPU bilinear yet); GPU path below overwrites with interpolated values.
			memset(l->vit_pos_embed, 0, T * N * sizeof(float));
		}
		memset(l->vit_pos_embed_updates, 0, T * N * sizeof(float));
		
		l->vit_qkv_out = (float*)xrealloc(l->vit_qkv_out, l->batch * T * 3 * N * sizeof(float));
		l->vit_attn_scores = (float*)xrealloc(l->vit_attn_scores, l->batch * l->vit_heads * T * T * sizeof(float));
		l->vit_attn_out = (float*)xrealloc(l->vit_attn_out, l->batch * T * N * sizeof(float));
		l->vit_ffn_hidden = (float*)xrealloc(l->vit_ffn_hidden, l->batch * T * l->vit_mlp_dim * sizeof(float));
		l->activation_input = (float*)xrealloc(l->activation_input, l->batch * T * l->vit_mlp_dim * sizeof(float));
		l->vit_ln1_mean = (float*)xrealloc(l->vit_ln1_mean, l->batch * T * sizeof(float));
		l->vit_ln1_var = (float*)xrealloc(l->vit_ln1_var, l->batch * T * sizeof(float));
		l->vit_ln2_mean = (float*)xrealloc(l->vit_ln2_mean, l->batch * T * sizeof(float));
		l->vit_ln2_var = (float*)xrealloc(l->vit_ln2_var, l->batch * T * sizeof(float));
		l->vit_ln1_xhat = (float*)xrealloc(l->vit_ln1_xhat, l->batch * T * N * sizeof(float));
		l->vit_ln2_xhat = (float*)xrealloc(l->vit_ln2_xhat, l->batch * T * l->out_c * sizeof(float));

		l->vit_pre_res1 = (float*)xrealloc(l->vit_pre_res1, l->batch * T * N * sizeof(float));
		l->vit_pre_res2 = (float*)xrealloc(l->vit_pre_res2, l->batch * T * l->out_c * sizeof(float));
		l->x = (float*)xrealloc(l->x, l->batch * T * l->out_c * sizeof(float));
		l->x_norm = (float*)xrealloc(l->x_norm, l->batch * T * l->out_c * sizeof(float));
		
		l->output = (float*)xrealloc(l->output, l->batch * l->outputs * sizeof(float));
		l->delta = (float*)xrealloc(l->delta, l->batch * l->outputs * sizeof(float));

#ifdef DARKNET_GPU
		// Interpolate learned pos embed from [old_H, old_W, C] to [new_H, new_W, C].
		// This preserves training continuity instead of re-initializing randomly.
		{
			float *old_pos_gpu = l->vit_pos_embed_gpu;
			float *new_pos_gpu = nullptr;
			if (l->vit_pos_embed_type == VIT_POS_EMBED_SINUSOIDAL)
			{
				new_pos_gpu = cuda_make_array(l->vit_pos_embed, T * N);
			}
			else
			{
				new_pos_gpu = cuda_make_array(nullptr, T * N);
				resize_vit_pos_embed_gpu(old_pos_gpu, new_pos_gpu, old_H, old_W, l->out_h, l->out_w, N);
				// Sync interpolated embed back to CPU for checkpointing / CPU-path consistency
				cuda_pull_array(new_pos_gpu, l->vit_pos_embed, T * N);
			}
			cuda_free(old_pos_gpu);
			l->vit_pos_embed_gpu = new_pos_gpu;
		}
		cuda_free(l->vit_pos_embed_updates_gpu);
		l->vit_pos_embed_updates_gpu = cuda_make_array(nullptr, T * N);  // reset gradient accum
		
		cuda_free(l->vit_qkv_out_gpu);
		cuda_free(l->vit_attn_scores_gpu);
		cuda_free(l->vit_attn_out_gpu);
		cuda_free(l->vit_ffn_hidden_gpu);
		cuda_free(l->activation_input_gpu);
		cuda_free(l->vit_ln1_mean_gpu);
		cuda_free(l->vit_ln1_var_gpu);
		cuda_free(l->vit_ln2_mean_gpu);
		cuda_free(l->vit_ln2_var_gpu);
		cuda_free(l->vit_ln1_xhat_gpu);
		cuda_free(l->vit_ln2_xhat_gpu);

		l->vit_qkv_out_gpu = cuda_make_array(l->vit_qkv_out, l->batch * T * 3 * N);
		l->vit_attn_scores_gpu = cuda_make_array(l->vit_attn_scores, l->batch * l->vit_heads * T * T);
		l->vit_attn_out_gpu = cuda_make_array(l->vit_attn_out, l->batch * T * N);
		l->vit_ffn_hidden_gpu = cuda_make_array(l->vit_ffn_hidden, l->batch * T * l->vit_mlp_dim);
		l->activation_input_gpu = cuda_make_array(nullptr, l->batch * T * l->vit_mlp_dim);
		l->vit_ln1_mean_gpu = cuda_make_array(l->vit_ln1_mean, l->batch * T);
		l->vit_ln1_var_gpu = cuda_make_array(l->vit_ln1_var, l->batch * T);
		l->vit_ln2_mean_gpu = cuda_make_array(l->vit_ln2_mean, l->batch * T);
		l->vit_ln2_var_gpu = cuda_make_array(l->vit_ln2_var, l->batch * T);
		l->vit_ln1_xhat_gpu = cuda_make_array(l->vit_ln1_xhat, l->batch * T * N);
		l->vit_ln2_xhat_gpu = cuda_make_array(l->vit_ln2_xhat, l->batch * T * l->out_c);

		cuda_free(l->vit_pre_res1_gpu);
		l->vit_pre_res1_gpu = cuda_make_array(l->vit_pre_res1, l->batch * T * N);
		cuda_free(l->vit_pre_res2_gpu);
		l->vit_pre_res2_gpu = cuda_make_array(l->vit_pre_res2, l->batch * T * l->out_c);
		cuda_free(l->x_gpu);
		l->x_gpu = cuda_make_array(nullptr, l->batch * T * l->out_c);
		cuda_free(l->x_norm_gpu);
		l->x_norm_gpu = cuda_make_array(nullptr, l->batch * T * l->out_c);
		cuda_free(l->vit_patch_tokens_gpu);
		l->vit_patch_tokens_gpu = cuda_make_array(nullptr, l->batch * T * patch_dim);
		cuda_free(l->vit_patch_delta_gpu);
		l->vit_patch_delta_gpu = cuda_make_array(nullptr, l->batch * T * patch_dim);
		cuda_free(l->vit_tmp_token_c1_gpu);
		l->vit_tmp_token_c1_gpu = cuda_make_array(nullptr, l->batch * T * N);
		cuda_free(l->vit_tmp_token_c2_gpu);
		l->vit_tmp_token_c2_gpu = cuda_make_array(nullptr, l->batch * T * N);
		cuda_free(l->vit_tmp_token_n1_gpu);
		l->vit_tmp_token_n1_gpu = cuda_make_array(nullptr, l->batch * T * l->out_c);
		cuda_free(l->vit_tmp_token_n2_gpu);
		l->vit_tmp_token_n2_gpu = cuda_make_array(nullptr, l->batch * T * l->out_c);
		cuda_free(l->vit_tmp_token_n3_gpu);
		l->vit_tmp_token_n3_gpu = cuda_make_array(nullptr, l->batch * T * l->out_c);
		cuda_free(l->vit_tmp_ffn_hidden_gpu);
		l->vit_tmp_ffn_hidden_gpu = cuda_make_array(nullptr, l->batch * T * l->vit_mlp_dim);
		const int head_scratch = l->batch * l->vit_heads * T * l->vit_head_dim;
		cuda_free(l->vit_tmp_head1_gpu);
		l->vit_tmp_head1_gpu = cuda_make_array(nullptr, head_scratch);
		cuda_free(l->vit_tmp_head2_gpu);
		l->vit_tmp_head2_gpu = cuda_make_array(nullptr, head_scratch);
		cuda_free(l->vit_tmp_head3_gpu);
		l->vit_tmp_head3_gpu = cuda_make_array(nullptr, head_scratch);
		cuda_free(l->vit_tmp_head4_gpu);
		l->vit_tmp_head4_gpu = cuda_make_array(nullptr, head_scratch);
		cuda_free(l->vit_tmp_head5_gpu);
		l->vit_tmp_head5_gpu = cuda_make_array(nullptr, head_scratch);
		cuda_free(l->vit_tmp_scores_gpu);
		l->vit_tmp_scores_gpu = cuda_make_array(nullptr, l->batch * l->vit_heads * T * T);
		
		cuda_free(l->output_gpu);
		cuda_free(l->delta_gpu);
		l->output_gpu = cuda_make_array(l->output, l->batch * l->outputs);
		l->delta_gpu = cuda_make_array(l->delta, l->batch * l->outputs);
#endif
	}
}

// ─── forward ──────────────────────────────────────────────────────────────────

void forward_vit_layer(Darknet::Layer & l, Darknet::NetworkState state)
{
	TAT(TATPARMS);

	const int B = l.batch;
	const int N = l.n;
	const int T = l.out_h * l.out_w;
	const int patch_dim = vit_patch_dim(l);
	const int heads = l.vit_heads;
	const int d = l.vit_head_dim;
	const int ffn_hidden = l.vit_mlp_dim;
	const float scale = 1.0f / std::sqrt((float)d);

	std::vector<float> patches(static_cast<size_t>(B) * T * patch_dim);
	std::vector<float> X(B * T * N);

	// 1. Patch embedding: [B, C, H, W] -> [B, T, P*P*C] -> [B, T, N].
	vit_patchify_cpu(l, state.input, patches.data());
	gemm_cpu(0, 1, B * T, N, patch_dim, 1.0f,
		patches.data(), patch_dim,
		l.vit_patch_embed, patch_dim,
		0.0f,
		X.data(), N);

	for (int i = 0; i < B * T; ++i)
	{
		for (int n = 0; n < N; ++n)
		{
			X[i * N + n] += l.vit_patch_bias[n];
		}
	}

	// 2. Positional Embedding: token_i += pos_embed[i]
	//
	// Patch flattening destroys explicit 2D spatial structure.  Additive
	// absolute PE re-injects patch-grid location identity and is shared across
	// batch items.
	for (int b = 0; b < B; b++)
	{
		for (int t = 0; t < T; t++)
		{
			for (int n = 0; n < N; n++)
			{
				X[b * T * N + t * N + n] += l.vit_pos_embed[t * N + n];
			}
		}
	}
	check_nan_cpu("forward: patch tokens + pos_embed", X.data(), B * T * N, l.index);

	std::copy(X.begin(), X.end(), l.vit_pre_res1);

	// 3. LayerNorm 1
	std::vector<float> ln1_out(B * T * N);
	layernorm_forward(X.data(), ln1_out.data(),
		l.vit_ln1_mean, l.vit_ln1_var, l.vit_ln1_xhat,
		l.vit_ln1_gamma, l.vit_ln1_beta, B * T, N);
	check_nan_cpu("forward: after layernorm1", ln1_out.data(), B * T * N, l.index);

	// 4. QKV Projection: [B*T, N] @ W_qkv^T -> [B*T, 3N]
	//
	// Q, K, V are produced in a single GEMM by stacking [Wq; Wk; Wv] into one
	// weight matrix of shape [3N, N].  The output layout is interleaved.
	gemm_cpu(0, 1, B * T, 3 * N, N, 1.0f,
		ln1_out.data(), N,
		l.weights, N,
		0.0f,
		l.vit_qkv_out, 3 * N);

	for (int i = 0; i < B * T; i++)
	{
		for (int j = 0; j < 3 * N; j++)
		{
			l.vit_qkv_out[i * 3 * N + j] += l.biases[j];
		}
	}
	check_nan_cpu("forward: after qkv projection", l.vit_qkv_out, B * T * 3 * N, l.index);

	// 5. & 6. Multi-Head Attention
	std::fill(l.vit_attn_out, l.vit_attn_out + B * T * N, 0.0f);
	std::fill(l.vit_attn_scores, l.vit_attn_scores + B * heads * T * T, 0.0f);

	for (int b = 0; b < B; b++)
	{
		float *qkv_b = l.vit_qkv_out + b * T * 3 * N;
		float *attn_out_b = l.vit_attn_out + b * T * N;

		for (int h_idx = 0; h_idx < heads; h_idx++)
		{
			std::vector<float> Q(T * d), K_mat(T * d), V(T * d);
			for (int t = 0; t < T; t++)
			{
				for (int dd = 0; dd < d; dd++)
				{
					Q[t * d + dd] = qkv_b[t * 3 * N + h_idx * d + dd];
					K_mat[t * d + dd] = qkv_b[t * 3 * N + N + h_idx * d + dd];
					V[t * d + dd] = qkv_b[t * 3 * N + 2 * N + h_idx * d + dd];
				}
			}
			sanitize_and_constrain_cpu(Q.data(), T * d, VIT_ATTENTION_QK_CLAMP);
			sanitize_and_constrain_cpu(K_mat.data(), T * d, VIT_ATTENTION_QK_CLAMP);

			float *scores = l.vit_attn_scores + (b * heads + h_idx) * T * T;

			// Scaled dot-product attention: scores[t_q, t_k] = Q[t_q] · K[t_k] / sqrt(d)
			// Dividing by sqrt(d) prevents saturation: without it, growing d makes
			// dot products large, pushing softmax into regions with near-zero gradients.
			gemm_cpu(0, 1, T, T, d, scale,
				Q.data(), d,
				K_mat.data(), d,
				0.0f,
				scores, T);
			sanitize_and_constrain_cpu(scores, T * T, VIT_ATTENTION_SCORE_CLAMP);

			// Softmax over keys (row-wise): converts raw scores into routing probabilities.
			// After softmax, scores[t_q, :] sums to 1 and acts as a soft dictionary lookup.
			for (int t = 0; t < T; t++)
			{
				softmax_row(scores + t * T, T);
			}

			// Context = scores @ V: each output token is a weighted sum of all value tokens.
			// This is the information-routing step — high attention weight = more contribution.
			std::vector<float> attn_result(T * d);
			gemm_cpu(0, 0, T, d, T, 1.0f,
				scores, T,
				V.data(), d,
				0.0f,
				attn_result.data(), d);

			for (int t = 0; t < T; t++)
			{
				for (int dd = 0; dd < d; dd++)
				{
					attn_out_b[t * N + h_idx * d + dd] = attn_result[t * d + dd];
				}
			}
		}
	}
	check_nan_cpu("forward: after attention scores", l.vit_attn_scores, B * heads * T * T, l.index);
	check_nan_cpu("forward: after attention output", l.vit_attn_out, B * T * N, l.index);

	// 7. Output Projection
	std::vector<float> proj_out(B * T * N);
	gemm_cpu(0, 1, B * T, N, N, 1.0f,
		l.vit_attn_out, N,
		l.vit_wo, N,
		0.0f,
		proj_out.data(), N);

	for (int i = 0; i < B * T; i++)
	{
		for (int j = 0; j < N; j++)
		{
			proj_out[i * N + j] += l.vit_wo_bias[j];
		}
	}
	check_nan_cpu("forward: after output projection", proj_out.data(), B * T * N, l.index);

	// 8. Residual 1 with mHC constrained mixing
	sanitize_and_constrain_cpu(proj_out.data(), B * T * N, VIT_FEATURE_CLAMP);
	std::copy(proj_out.begin(), proj_out.end(), l.x);
	std::vector<float> res1_out(B * T * N);
	mhc_residual_forward_cpu(X.data(), l.x, res1_out.data(), B * T * N, l.scales, 0);

	sanitize_and_constrain_cpu(res1_out.data(), B * T * N, VIT_FEATURE_CLAMP);
	std::copy(res1_out.begin(), res1_out.end(), l.vit_pre_res2);
	check_nan_cpu("forward: after residual1", res1_out.data(), B * T * N, l.index);

	// 9. LayerNorm 2
	std::vector<float> ln2_out(B * T * N);
	layernorm_forward(res1_out.data(), ln2_out.data(),
		l.vit_ln2_mean, l.vit_ln2_var, l.vit_ln2_xhat,
		l.vit_ln2_gamma, l.vit_ln2_beta, B * T, N);
	check_nan_cpu("forward: after layernorm2", ln2_out.data(), B * T * N, l.index);

	// 10. FFN Up
	gemm_cpu(0, 1, B * T, ffn_hidden, N, 1.0f,
		ln2_out.data(), N,
		l.vit_ffn_w1, N,
		0.0f,
		l.vit_ffn_hidden, ffn_hidden);

	for (int i = 0; i < B * T; i++)
	{
		for (int j = 0; j < ffn_hidden; j++)
		{
			l.vit_ffn_hidden[i * ffn_hidden + j] += l.vit_ffn_b1[j];
		}
	}

	sanitize_and_constrain_cpu(l.vit_ffn_hidden, B * T * ffn_hidden, VIT_FEATURE_CLAMP);
	memcpy(l.activation_input, l.vit_ffn_hidden, B * T * ffn_hidden * sizeof(float));

	activate_array(l.vit_ffn_hidden, B * T * ffn_hidden, l.activation);
	sanitize_and_constrain_cpu(l.vit_ffn_hidden, B * T * ffn_hidden, VIT_FEATURE_CLAMP);
	check_nan_cpu("forward: after ffn hidden", l.vit_ffn_hidden, B * T * ffn_hidden, l.index);

	// FFN Down
	std::vector<float> ffn_out(B * T * N);
	gemm_cpu(0, 1, B * T, N, ffn_hidden, 1.0f,
		l.vit_ffn_hidden, ffn_hidden,
		l.vit_ffn_w2, ffn_hidden,
		0.0f,
		ffn_out.data(), N);

	for (int i = 0; i < B * T; i++)
	{
		for (int j = 0; j < N; j++)
		{
			ffn_out[i * N + j] += l.vit_ffn_b2[j];
		}
	}

	// 11. Residual 2 with mHC constrained mixing
	sanitize_and_constrain_cpu(ffn_out.data(), B * T * N, VIT_FEATURE_CLAMP);
	std::copy(ffn_out.begin(), ffn_out.end(), l.x_norm);
	mhc_residual_forward_cpu(res1_out.data(), l.x_norm, ffn_out.data(), B * T * N, l.scales, 1);
	sanitize_and_constrain_cpu(ffn_out.data(), B * T * N, VIT_FEATURE_CLAMP);
	check_nan_cpu("forward: after ffn output + residual", ffn_out.data(), B * T * N, l.index);

	// 12. Reshape to Spatial [B, N, H, W]
	for (int b = 0; b < B; b++)
	{
		for (int t = 0; t < T; t++)
		{
			for (int n_c = 0; n_c < N; n_c++)
			{
				l.output[b * N * T + n_c * T + t] = ffn_out[b * T * N + t * N + n_c];
			}
		}
	}
	sanitize_and_constrain_cpu(l.output, B * N * T, VIT_FEATURE_CLAMP);
	check_nan_cpu("forward: final output", l.output, B * N * T, l.index);
}

// ─── backward ─────────────────────────────────────────────────────────────────

void backward_vit_layer(Darknet::Layer & l, Darknet::NetworkState state)
{
	TAT(TATPARMS);

	const int B = l.batch;
	const int N = l.n;
	const int T = l.out_h * l.out_w;
	const int ffn_hidden = l.vit_mlp_dim;

	// Convert gradient from NCHW → token layout [B, T, N] (mirror of forward step 12).
	// l.delta is [B, N, H, W] in NCHW; dout_tokens must be [B, T, N] in NTC.
	std::vector<float> dout_tokens(B * T * N);
	for (int b = 0; b < B; b++)
	{
		for (int t = 0; t < T; t++)
		{
			for (int n_c = 0; n_c < N; n_c++)
			{
				dout_tokens[b * T * N + t * N + n_c] = l.delta[b * N * T + n_c * T + t];
			}
		}
	}
	check_nan_cpu("backward: initial delta", dout_tokens.data(), B * T * N, l.index);

	// Residual 2 mHC backward splits the gradient into skip and FFN branches.
	std::vector<float> d_res1(B * T * N);
	std::vector<float> d_ffn_out(B * T * N);
	mhc_residual_backward_cpu(l.vit_pre_res2, l.x_norm, dout_tokens.data(),
		d_res1.data(), d_ffn_out.data(), B * T * N, l.scales, l.scale_updates, 1);
	sanitize_and_constrain_cpu(d_res1.data(), B * T * N, VIT_GRAD_CLAMP);
	sanitize_and_constrain_cpu(d_ffn_out.data(), B * T * N, VIT_GRAD_CLAMP);

	// FFN Down Backward
	std::vector<float> d_ffn_hidden(B * T * ffn_hidden);
	gemm_cpu(0, 0, B * T, ffn_hidden, N, 1.0f,
		d_ffn_out.data(), N,
		l.vit_ffn_w2, ffn_hidden,
		0.0f,
		d_ffn_hidden.data(), ffn_hidden);

	gemm_cpu(1, 0, N, ffn_hidden, B * T, 1.0f,
		d_ffn_out.data(), N,
		l.vit_ffn_hidden, ffn_hidden,
		1.0f,
		l.vit_ffn_w2_updates, ffn_hidden);

	for (int i = 0; i < B * T; i++)
	{
		for (int j = 0; j < N; j++)
		{
			l.vit_ffn_b2_updates[j] += d_ffn_out[i * N + j];
		}
	}

	sanitize_and_constrain_cpu(d_ffn_hidden.data(), B * T * ffn_hidden, VIT_GRAD_CLAMP);
	gradient_array(l.activation_input, B * T * ffn_hidden, l.activation, d_ffn_hidden.data());
	sanitize_and_constrain_cpu(d_ffn_hidden.data(), B * T * ffn_hidden, VIT_GRAD_CLAMP);
	check_nan_cpu("backward: after ffn hidden grad", d_ffn_hidden.data(), B * T * ffn_hidden, l.index);

	// Recompute LN2 output — needed for FFN up-weight gradient:
	//   dW_ffn1 = d_ffn_hidden^T @ ln2_out
	// Like LN1 in backward_vit_attention_tail, we did not cache ln2_out during
	// forward to avoid the memory cost, so we re-run the forward here.
	std::vector<float> ln2_out(B * T * N);
	std::vector<float> ln2_xhat_tmp(B * T * N);
	std::vector<float> ln2_mean_tmp(B * T);
	std::vector<float> ln2_var_tmp(B * T);
	layernorm_forward(l.vit_pre_res2, ln2_out.data(),
		ln2_mean_tmp.data(), ln2_var_tmp.data(), ln2_xhat_tmp.data(),
		l.vit_ln2_gamma, l.vit_ln2_beta, B * T, N);

	gemm_cpu(1, 0, ffn_hidden, N, B * T, 1.0f,
		d_ffn_hidden.data(), ffn_hidden,
		ln2_out.data(), N,
		1.0f,
		l.vit_ffn_w1_updates, N);

	for (int i = 0; i < B * T; i++)
	{
		for (int j = 0; j < ffn_hidden; j++)
		{
			l.vit_ffn_b1_updates[j] += d_ffn_hidden[i * ffn_hidden + j];
		}
	}

	std::vector<float> d_ln2_out(B * T * N);
	gemm_cpu(0, 0, B * T, N, ffn_hidden, 1.0f,
		d_ffn_hidden.data(), ffn_hidden,
		l.vit_ffn_w1, N,
		0.0f,
		d_ln2_out.data(), N);
	sanitize_and_constrain_cpu(d_ln2_out.data(), B * T * N, VIT_GRAD_CLAMP);
	check_nan_cpu("backward: before layernorm2 backward", d_ln2_out.data(), B * T * N, l.index);

	std::vector<float> d_pre_ln2(B * T * N);
	layernorm_backward(d_ln2_out.data(), l.vit_ln2_xhat, l.vit_ln2_var,
		l.vit_ln2_gamma, d_pre_ln2.data(), l.vit_ln2_gamma_updates, l.vit_ln2_beta_updates,
		B * T, N);

	for (int i = 0; i < B * T * N; i++)
	{
		d_pre_ln2[i] += d_res1[i];
	}
	sanitize_and_constrain_cpu(d_pre_ln2.data(), B * T * N, VIT_GRAD_CLAMP);
	check_nan_cpu("backward: after layernorm2 backward + residual", d_pre_ln2.data(), B * T * N, l.index);

	std::vector<float> d_proj_out(B * T * N);
	std::vector<float> d_skip1(B * T * N);
	mhc_residual_backward_cpu(l.vit_pre_res1, l.x, d_pre_ln2.data(),
		d_skip1.data(), d_proj_out.data(), B * T * N, l.scales, l.scale_updates, 0);
	sanitize_and_constrain_cpu(d_skip1.data(), B * T * N, VIT_GRAD_CLAMP);
	sanitize_and_constrain_cpu(d_proj_out.data(), B * T * N, VIT_GRAD_CLAMP);

	std::vector<float> d_attn_out(B * T * N);
	gemm_cpu(0, 0, B * T, N, N, 1.0f,
		d_proj_out.data(), N,
		l.vit_wo, N,
		0.0f,
		d_attn_out.data(), N);
	sanitize_and_constrain_cpu(d_attn_out.data(), B * T * N, VIT_GRAD_CLAMP);
	check_nan_cpu("backward: after output projection backward", d_attn_out.data(), B * T * N, l.index);

	gemm_cpu(1, 0, N, N, B * T, 1.0f,
		d_proj_out.data(), N,
		l.vit_attn_out, N,
		1.0f,
		l.vit_wo_updates, N);

	for (int i = 0; i < B * T; i++)
	{
		for (int j = 0; j < N; j++)
		{
			l.vit_wo_bias_updates[j] += d_proj_out[i * N + j];
		}
	}

	backward_vit_attention_tail(l, state, d_attn_out.data(), d_skip1.data());
}

// ─── update ───────────────────────────────────────────────────────────────────

void update_vit_layer(Darknet::Layer & l, int batch, float learning_rate_init, float momentum, float decay)
{
	TAT(TATPARMS);

	const float lr = learning_rate_init * l.learning_rate_scale;
	const int N = l.n;
	const int ffn_hidden = l.vit_mlp_dim;
	const int T = l.out_h * l.out_w;
	const int patch_dim = vit_patch_dim(l);

	// Match the transformer layer's global gradient clipping. The mHC residual
	// mixer adds trainable residual parameters, so ViT needs the same guard.
	{
		const float max_grad_norm = 5.0f;

		auto sanitize_and_l2_norm = [](float *buf, int n) -> double
		{
			double sum = 0.0;
			for (int i = 0; i < n; ++i)
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
		global_norm_sq += sanitize_and_l2_norm(l.vit_patch_embed_updates, N * patch_dim);
		global_norm_sq += sanitize_and_l2_norm(l.vit_patch_bias_updates, N);
		global_norm_sq += sanitize_and_l2_norm(l.weight_updates, l.nweights);
		global_norm_sq += sanitize_and_l2_norm(l.bias_updates, 3 * N);
		global_norm_sq += sanitize_and_l2_norm(l.vit_wo_updates, N * N);
		global_norm_sq += sanitize_and_l2_norm(l.vit_wo_bias_updates, N);
		global_norm_sq += sanitize_and_l2_norm(l.vit_ln1_gamma_updates, N);
		global_norm_sq += sanitize_and_l2_norm(l.vit_ln1_beta_updates, N);
		global_norm_sq += sanitize_and_l2_norm(l.vit_ln2_gamma_updates, N);
		global_norm_sq += sanitize_and_l2_norm(l.vit_ln2_beta_updates, N);
		global_norm_sq += sanitize_and_l2_norm(l.vit_ffn_w1_updates, ffn_hidden * N);
		global_norm_sq += sanitize_and_l2_norm(l.vit_ffn_b1_updates, ffn_hidden);
		global_norm_sq += sanitize_and_l2_norm(l.vit_ffn_w2_updates, N * ffn_hidden);
		global_norm_sq += sanitize_and_l2_norm(l.vit_ffn_b2_updates, N);
		if (l.vit_pos_embed_type == VIT_POS_EMBED_LEARNED)
		{
			global_norm_sq += sanitize_and_l2_norm(l.vit_pos_embed_updates, T * N);
		}
		global_norm_sq += sanitize_and_l2_norm(l.scale_updates, MHC_PARAM_COUNT);

		const double global_norm = std::sqrt(global_norm_sq);
		if (!std::isfinite(global_norm) || global_norm > max_grad_norm)
		{
			const float clip_coef = std::isfinite(global_norm) ?
				static_cast<float>(max_grad_norm / global_norm) : 0.0f;
			scal_cpu(N * patch_dim, clip_coef, l.vit_patch_embed_updates, 1);
			scal_cpu(N, clip_coef, l.vit_patch_bias_updates, 1);
			scal_cpu(l.nweights, clip_coef, l.weight_updates, 1);
			scal_cpu(3 * N, clip_coef, l.bias_updates, 1);
			scal_cpu(N * N, clip_coef, l.vit_wo_updates, 1);
			scal_cpu(N, clip_coef, l.vit_wo_bias_updates, 1);
			scal_cpu(N, clip_coef, l.vit_ln1_gamma_updates, 1);
			scal_cpu(N, clip_coef, l.vit_ln1_beta_updates, 1);
			scal_cpu(N, clip_coef, l.vit_ln2_gamma_updates, 1);
			scal_cpu(N, clip_coef, l.vit_ln2_beta_updates, 1);
			scal_cpu(ffn_hidden * N, clip_coef, l.vit_ffn_w1_updates, 1);
			scal_cpu(ffn_hidden, clip_coef, l.vit_ffn_b1_updates, 1);
			scal_cpu(N * ffn_hidden, clip_coef, l.vit_ffn_w2_updates, 1);
			scal_cpu(N, clip_coef, l.vit_ffn_b2_updates, 1);
			if (l.vit_pos_embed_type == VIT_POS_EMBED_LEARNED)
			{
				scal_cpu(T * N, clip_coef, l.vit_pos_embed_updates, 1);
			}
			scal_cpu(MHC_PARAM_COUNT, clip_coef, l.scale_updates, 1);
		}
	}

	// Patch embedding
	{
		const int count = N * patch_dim;
		axpy_cpu(count, -decay * batch, l.vit_patch_embed, 1, l.vit_patch_embed_updates, 1);
		axpy_cpu(count, lr / batch, l.vit_patch_embed_updates, 1, l.vit_patch_embed, 1);
		scal_cpu(count, momentum, l.vit_patch_embed_updates, 1);
	}
	axpy_cpu(N, lr / batch, l.vit_patch_bias_updates, 1, l.vit_patch_bias, 1);
	scal_cpu(N, momentum, l.vit_patch_bias_updates, 1);

	// QKV weights
	axpy_cpu(l.nweights, -decay * batch, l.weights, 1, l.weight_updates, 1);
	axpy_cpu(l.nweights, lr / batch, l.weight_updates, 1, l.weights, 1);
	scal_cpu(l.nweights, momentum, l.weight_updates, 1);

	// QKV biases
	axpy_cpu(3 * N, lr / batch, l.bias_updates, 1, l.biases, 1);
	scal_cpu(3 * N, momentum, l.bias_updates, 1);

	// Output projection
	{
		const int count = N * N;
		axpy_cpu(count, -decay * batch, l.vit_wo, 1, l.vit_wo_updates, 1);
		axpy_cpu(count, lr / batch, l.vit_wo_updates, 1, l.vit_wo, 1);
		scal_cpu(count, momentum, l.vit_wo_updates, 1);
	}
	axpy_cpu(N, lr / batch, l.vit_wo_bias_updates, 1, l.vit_wo_bias, 1);
	scal_cpu(N, momentum, l.vit_wo_bias_updates, 1);

	// mHC residual mixer — slow parameter. Uses 0.1x LR so the branch coefficient
	// cannot jump at LR-schedule steps and tip the network into NaN/collapse once
	// attention starts contributing meaningfully.
	axpy_cpu(MHC_PARAM_COUNT, 0.1f * lr / batch, l.scale_updates, 1, l.scales, 1);
	sanitize_and_constrain_cpu(l.scales, MHC_PARAM_COUNT, VIT_MHC_PARAM_CLAMP);
	scal_cpu(MHC_PARAM_COUNT, momentum, l.scale_updates, 1);

	// LayerNorm 1
	axpy_cpu(N, lr / batch, l.vit_ln1_gamma_updates, 1, l.vit_ln1_gamma, 1);
	scal_cpu(N, momentum, l.vit_ln1_gamma_updates, 1);
	axpy_cpu(N, lr / batch, l.vit_ln1_beta_updates, 1, l.vit_ln1_beta, 1);
	scal_cpu(N, momentum, l.vit_ln1_beta_updates, 1);

	// LayerNorm 2
	axpy_cpu(N, lr / batch, l.vit_ln2_gamma_updates, 1, l.vit_ln2_gamma, 1);
	scal_cpu(N, momentum, l.vit_ln2_gamma_updates, 1);
	axpy_cpu(N, lr / batch, l.vit_ln2_beta_updates, 1, l.vit_ln2_beta, 1);
	scal_cpu(N, momentum, l.vit_ln2_beta_updates, 1);

	// FFN W1
	{
		const int count = ffn_hidden * N;
		axpy_cpu(count, -decay * batch, l.vit_ffn_w1, 1, l.vit_ffn_w1_updates, 1);
		axpy_cpu(count, lr / batch, l.vit_ffn_w1_updates, 1, l.vit_ffn_w1, 1);
		scal_cpu(count, momentum, l.vit_ffn_w1_updates, 1);
	}
	axpy_cpu(ffn_hidden, lr / batch, l.vit_ffn_b1_updates, 1, l.vit_ffn_b1, 1);
	scal_cpu(ffn_hidden, momentum, l.vit_ffn_b1_updates, 1);

	// FFN W2
	{
		const int count = N * ffn_hidden;
		axpy_cpu(count, -decay * batch, l.vit_ffn_w2, 1, l.vit_ffn_w2_updates, 1);
		axpy_cpu(count, lr / batch, l.vit_ffn_w2_updates, 1, l.vit_ffn_w2, 1);
		scal_cpu(count, momentum, l.vit_ffn_w2_updates, 1);
	}
	axpy_cpu(N, lr / batch, l.vit_ffn_b2_updates, 1, l.vit_ffn_b2, 1);
	scal_cpu(N, momentum, l.vit_ffn_b2_updates, 1);

	// Learned positional embeddings. SimpleViT-style sinusoidal embeddings are fixed.
	if (l.vit_pos_embed_type == VIT_POS_EMBED_LEARNED)
	{
		const int count = T * N;
		axpy_cpu(count, -decay * batch, l.vit_pos_embed, 1, l.vit_pos_embed_updates, 1);
		axpy_cpu(count, lr / batch, l.vit_pos_embed_updates, 1, l.vit_pos_embed, 1);
		scal_cpu(count, momentum, l.vit_pos_embed_updates, 1);
	}
}
