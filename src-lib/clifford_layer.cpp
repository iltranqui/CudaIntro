#include "darknet_internal.hpp"
#include "clifford_layer.hpp"
#include "activations.hpp"
#include "blas.hpp"
#include "convolutional_layer.hpp"
#include "gemm.hpp"
#include "utils.hpp"

#include <cmath>
#include <cstring>
#include <vector>

namespace
{
	static auto & cfg_and_state = Darknet::CfgAndState::get();

	constexpr float k_clifford_ln_eps = 1e-5f;

	inline bool clifford_has_local(const Darknet::Layer & l)
	{
		return l.cli_gffn_mode != 1;
	}

	inline bool clifford_has_global(const Darknet::Layer & l)
	{
		return l.cli_gffn_mode != 0;
	}

	inline bool clifford_has_higher_local(const Darknet::Layer & l)
	{
		return clifford_has_local(l) && l.cli_higher_mode == 1 && l.cli_interaction_mode != 0;
	}

	inline size_t clifford_tensor_count(const Darknet::Layer & l)
	{
		return static_cast<size_t>(l.batch) * l.c * l.h * l.w;
	}

	inline size_t clifford_token_count(const Darknet::Layer & l)
	{
		return static_cast<size_t>(l.batch) * l.h * l.w;
	}

	inline int clifford_index(int b, int c, int s, int C, int HW)
	{
		return ((b * C + c) * HW + s);
	}

	inline int clifford_normalize_shift(int shift, int C)
	{
		shift %= C;
		if (shift < 0)
		{
			shift += C;
		}
		return shift;
	}

	inline int clifford_wrap_channel(int channel, int shift, int C)
	{
		channel += shift;
		return (channel >= C) ? (channel - C) : channel;
	}

	inline int clifford_inner_num_shifts(const Darknet::Layer & l)
	{
		return (l.cli_shifts_inner && l.cli_num_shifts_inner > 0) ? l.cli_num_shifts_inner : l.cli_num_shifts;
	}

	inline const int * clifford_inner_shifts(const Darknet::Layer & l)
	{
		return (l.cli_shifts_inner && l.cli_num_shifts_inner > 0) ? l.cli_shifts_inner : l.cli_shifts;
	}

	inline int clifford_wedge_base_channel(int shift_idx, int num_inner_shifts, int C)
	{
		const int block = (shift_idx < num_inner_shifts) ? (2 * shift_idx) : (num_inner_shifts + shift_idx);
		return block * C;
	}

	inline int clifford_inner_base_channel(int shift_idx, int num_wedge_shifts, int C)
	{
		const int block = (shift_idx < num_wedge_shifts) ? (2 * shift_idx + 1) : (num_wedge_shifts + shift_idx);
		return block * C;
	}

	inline int clifford_raw_channels(int cli_mode, int num_wedge_shifts, int num_inner_shifts, int C)
	{
		if (cli_mode == 0)
		{
			return num_inner_shifts * C;
		}
		if (cli_mode == 1)
		{
			return num_wedge_shifts * C;
		}
		return (num_wedge_shifts + num_inner_shifts) * C;
	}

	inline int clifford_raw_channels(const Darknet::Layer & l)
	{
		return clifford_raw_channels(l.cli_interaction_mode, l.cli_num_shifts, clifford_inner_num_shifts(l), l.c);
	}

	static int clifford_count_unique_shifts(const int *shifts, int num_shifts, int C)
	{
		std::vector<unsigned char> seen(static_cast<size_t>(C), 0);
		int unique = 0;
		for (int i = 0; i < num_shifts; ++i)
		{
			const int shift = clifford_normalize_shift(shifts[i], C);
			if (seen[shift] == 0)
			{
				seen[shift] = 1;
				++unique;
			}
		}
		return unique;
	}

	static int clifford_count_matching_shifts(const int *shifts, int num_shifts, int match)
	{
		int count = 0;
		for (int i = 0; i < num_shifts; ++i)
		{
			if (shifts[i] == match)
			{
				++count;
			}
		}
		return count;
	}

	static bool clifford_shift_schedules_match(const int *lhs, int lhs_count, const int *rhs, int rhs_count, int C)
	{
		if (lhs_count != rhs_count)
		{
			return false;
		}
		for (int i = 0; i < lhs_count; ++i)
		{
			if (clifford_normalize_shift(lhs[i], C) != clifford_normalize_shift(rhs[i], C))
			{
				return false;
			}
		}
		return true;
	}

	inline float * alloc_float_buffer(size_t count)
	{
		return (count == 0) ? nullptr : (float*)xcalloc(count, sizeof(float));
	}

	inline float * resize_float_buffer(float *ptr, size_t count)
	{
		if (count == 0)
		{
			if (ptr)
			{
				free(ptr);
			}
			return nullptr;
		}

		return (float*)xrealloc(ptr, count * sizeof(float));
	}

#ifdef DARKNET_GPU
	inline float * alloc_gpu_buffer(const float *cpu_ptr, size_t count)
	{
		return (count == 0) ? nullptr : cuda_make_array(const_cast<float*>(cpu_ptr), count);
	}

	inline int * alloc_gpu_int_buffer(const int *cpu_ptr, size_t count)
	{
		return (count == 0) ? nullptr : cuda_make_int_array_new_api(const_cast<int*>(cpu_ptr), count);
	}

	inline float * resize_gpu_float_buffer(float *ptr, const float *cpu_ptr, size_t count)
	{
		if (ptr)
		{
			cuda_free(ptr);
		}
		return alloc_gpu_buffer(cpu_ptr, count);
	}

	static void allocate_clifford_runtime_buffers_gpu(Darknet::Layer & l)
	{
		const size_t common_count = clifford_tensor_count(l);
		const size_t token_count = clifford_token_count(l);
		const size_t raw_count = clifford_has_local(l) ? common_count * l.cli_proj_in_dim / l.c : 0;
		const size_t raw_count_g = clifford_has_global(l) ? common_count * l.cli_proj_in_dim / l.c : 0;
		const size_t global_count = clifford_has_global(l) ? static_cast<size_t>(l.batch) * l.c : 0;

		l.cli_ln_out_gpu = alloc_gpu_buffer(l.cli_ln_out, common_count);
		l.cli_ln_mean_gpu = alloc_gpu_buffer(l.cli_ln_mean, token_count);
		l.cli_ln_var_gpu = alloc_gpu_buffer(l.cli_ln_var, token_count);
		l.cli_ln_xhat_gpu = alloc_gpu_buffer(l.cli_ln_xhat, common_count);
		l.cli_z_det_gpu = alloc_gpu_buffer(l.cli_z_det, common_count);
		l.cli_z_ctx_gpu = alloc_gpu_buffer(l.cli_z_ctx, common_count);
		l.cli_z_ctx_pre_diff_gpu = alloc_gpu_buffer(l.cli_z_ctx_pre_diff, common_count);
		l.cli_g_raw_gpu = alloc_gpu_buffer(l.cli_g_raw, raw_count);
		l.cli_g_feat_gpu = alloc_gpu_buffer(l.cli_g_feat, clifford_has_local(l) ? common_count : 0);
		l.cli_gate_alpha_gpu = alloc_gpu_buffer(l.cli_gate_alpha, clifford_has_local(l) ? common_count : 0);
		l.cli_gate_pre_sigmoid_gpu = alloc_gpu_buffer(l.cli_gate_pre_sigmoid, clifford_has_local(l) ? common_count : 0);
		l.cli_vb_feat_gpu = alloc_gpu_buffer(l.cli_vb_feat, clifford_has_higher_local(l) ? common_count : 0);
		l.cli_hmix_gpu = alloc_gpu_buffer(l.cli_hmix, common_count);
		l.cli_drop_mask_gpu = alloc_gpu_buffer(l.cli_drop_mask, l.batch);

		l.cli_global_ctx_gpu = alloc_gpu_buffer(l.cli_global_ctx, global_count);
		l.cli_g_raw_g_gpu = alloc_gpu_buffer(l.cli_g_raw_g, raw_count_g);
		l.cli_g_feat_g_gpu = alloc_gpu_buffer(l.cli_g_feat_g, clifford_has_global(l) ? common_count : 0);
		l.cli_gate_alpha_g_gpu = alloc_gpu_buffer(l.cli_gate_alpha_g, clifford_has_global(l) ? common_count : 0);
		l.cli_gate_pre_sigmoid_g_gpu = alloc_gpu_buffer(l.cli_gate_pre_sigmoid_g, clifford_has_global(l) ? common_count : 0);
	}

	static void resize_clifford_runtime_buffers_gpu(Darknet::Layer *l)
	{
		const size_t common_count = clifford_tensor_count(*l);
		const size_t token_count = clifford_token_count(*l);
		const size_t raw_count = clifford_has_local(*l) ? common_count * l->cli_proj_in_dim / l->c : 0;
		const size_t raw_count_g = clifford_has_global(*l) ? common_count * l->cli_proj_in_dim / l->c : 0;
		const size_t global_count = clifford_has_global(*l) ? static_cast<size_t>(l->batch) * l->c : 0;

		l->cli_ln_out_gpu = resize_gpu_float_buffer(l->cli_ln_out_gpu, l->cli_ln_out, common_count);
		l->cli_ln_mean_gpu = resize_gpu_float_buffer(l->cli_ln_mean_gpu, l->cli_ln_mean, token_count);
		l->cli_ln_var_gpu = resize_gpu_float_buffer(l->cli_ln_var_gpu, l->cli_ln_var, token_count);
		l->cli_ln_xhat_gpu = resize_gpu_float_buffer(l->cli_ln_xhat_gpu, l->cli_ln_xhat, common_count);
		l->cli_z_det_gpu = resize_gpu_float_buffer(l->cli_z_det_gpu, l->cli_z_det, common_count);
		l->cli_z_ctx_gpu = resize_gpu_float_buffer(l->cli_z_ctx_gpu, l->cli_z_ctx, common_count);
		l->cli_z_ctx_pre_diff_gpu = resize_gpu_float_buffer(l->cli_z_ctx_pre_diff_gpu, l->cli_z_ctx_pre_diff, common_count);
		l->cli_g_raw_gpu = resize_gpu_float_buffer(l->cli_g_raw_gpu, l->cli_g_raw, raw_count);
		l->cli_g_feat_gpu = resize_gpu_float_buffer(l->cli_g_feat_gpu, l->cli_g_feat, clifford_has_local(*l) ? common_count : 0);
		l->cli_gate_alpha_gpu = resize_gpu_float_buffer(l->cli_gate_alpha_gpu, l->cli_gate_alpha, clifford_has_local(*l) ? common_count : 0);
		l->cli_gate_pre_sigmoid_gpu = resize_gpu_float_buffer(l->cli_gate_pre_sigmoid_gpu, l->cli_gate_pre_sigmoid, clifford_has_local(*l) ? common_count : 0);
		l->cli_vb_feat_gpu = resize_gpu_float_buffer(l->cli_vb_feat_gpu, l->cli_vb_feat, clifford_has_higher_local(*l) ? common_count : 0);
		l->cli_hmix_gpu = resize_gpu_float_buffer(l->cli_hmix_gpu, l->cli_hmix, common_count);
		l->cli_drop_mask_gpu = resize_gpu_float_buffer(l->cli_drop_mask_gpu, l->cli_drop_mask, l->batch);

		l->cli_global_ctx_gpu = resize_gpu_float_buffer(l->cli_global_ctx_gpu, l->cli_global_ctx, global_count);
		l->cli_g_raw_g_gpu = resize_gpu_float_buffer(l->cli_g_raw_g_gpu, l->cli_g_raw_g, raw_count_g);
		l->cli_g_feat_g_gpu = resize_gpu_float_buffer(l->cli_g_feat_g_gpu, l->cli_g_feat_g, clifford_has_global(*l) ? common_count : 0);
		l->cli_gate_alpha_g_gpu = resize_gpu_float_buffer(l->cli_gate_alpha_g_gpu, l->cli_gate_alpha_g, clifford_has_global(*l) ? common_count : 0);
		l->cli_gate_pre_sigmoid_g_gpu = resize_gpu_float_buffer(l->cli_gate_pre_sigmoid_g_gpu, l->cli_gate_pre_sigmoid_g, clifford_has_global(*l) ? common_count : 0);
	}
#endif

	static void layernorm_nchw_forward(const float *input, float *output, float *mean, float *var, float *xhat,
		const float *gamma, const float *beta, int B, int C, int HW)
	{
		for (int b = 0; b < B; ++b)
		{
			for (int s = 0; s < HW; ++s)
			{
				const int token_idx = b * HW + s;
				float token_mean = 0.0f;
				for (int c = 0; c < C; ++c)
				{
					token_mean += input[clifford_index(b, c, s, C, HW)];
				}
				token_mean /= static_cast<float>(C);
				mean[token_idx] = token_mean;

				float token_var = 0.0f;
				for (int c = 0; c < C; ++c)
				{
					const float diff = input[clifford_index(b, c, s, C, HW)] - token_mean;
					token_var += diff * diff;
				}
				token_var /= static_cast<float>(C);
				var[token_idx] = token_var;

				const float inv_std = 1.0f / std::sqrt(token_var + k_clifford_ln_eps);
				for (int c = 0; c < C; ++c)
				{
					const int idx = clifford_index(b, c, s, C, HW);
					const float normalized = (input[idx] - token_mean) * inv_std;
					xhat[idx] = normalized;
					output[idx] = normalized * gamma[c] + beta[c];
				}
			}
		}
	}

	static void layernorm_nchw_backward(const float *dout, const float *xhat, const float *var,
		const float *gamma, float *dx, float *dgamma, float *dbeta, int B, int C, int HW)
	{
		for (int b = 0; b < B; ++b)
		{
			for (int s = 0; s < HW; ++s)
			{
				const int token_idx = b * HW + s;
				const float inv_std = 1.0f / std::sqrt(var[token_idx] + k_clifford_ln_eps);

				float sum_dxhat = 0.0f;
				float dot_dxhat_xhat = 0.0f;
				for (int c = 0; c < C; ++c)
				{
					const int idx = clifford_index(b, c, s, C, HW);
					dgamma[c] += dout[idx] * xhat[idx];
					dbeta[c] += dout[idx];

					const float dxhat = dout[idx] * gamma[c];
					sum_dxhat += dxhat;
					dot_dxhat_xhat += dxhat * xhat[idx];
				}

				if (dx == nullptr)
				{
					continue;
				}

				for (int c = 0; c < C; ++c)
				{
					const int idx = clifford_index(b, c, s, C, HW);
					const float dxhat = dout[idx] * gamma[c];
					dx[idx] = inv_std * (dxhat - (sum_dxhat + xhat[idx] * dot_dxhat_xhat) / static_cast<float>(C));
				}
			}
		}
	}

	static void add_bias_nchw(float *output, const float *bias, int B, int C, int HW)
	{
		if (bias == nullptr)
		{
			return;
		}

		for (int b = 0; b < B; ++b)
		{
			for (int c = 0; c < C; ++c)
			{
				float *dst = output + (b * C + c) * HW;
				const float bias_val = bias[c];
				for (int s = 0; s < HW; ++s)
				{
					dst[s] += bias_val;
				}
			}
		}
	}

	static void linear_1x1_forward(const float *input, float *output, const float *weights, const float *bias,
		int B, int C_in, int C_out, int HW)
	{
		for (int b = 0; b < B; ++b)
		{
			gemm_cpu(0, 0, C_out, HW, C_in, 1.0f,
				const_cast<float*>(weights), C_in,
				const_cast<float*>(input) + b * C_in * HW, HW,
				0.0f,
				output + b * C_out * HW, HW);
		}

		add_bias_nchw(output, bias, B, C_out, HW);
	}

	static void linear_1x1_backward(const float *input, const float *dout, const float *weights,
		float *weight_updates, float *bias_updates, float *dinput,
		int B, int C_in, int C_out, int HW)
	{
		if (bias_updates)
		{
			for (int b = 0; b < B; ++b)
			{
				for (int c = 0; c < C_out; ++c)
				{
					const float *src = dout + (b * C_out + c) * HW;
					float sum = 0.0f;
					for (int s = 0; s < HW; ++s)
					{
						sum += src[s];
					}
					bias_updates[c] += sum;
				}
			}
		}

		for (int b = 0; b < B; ++b)
		{
			gemm_cpu(0, 1, C_out, C_in, HW, 1.0f,
				const_cast<float*>(dout) + b * C_out * HW, HW,
				const_cast<float*>(input) + b * C_in * HW, HW,
				1.0f,
				weight_updates, C_in);

			if (dinput)
			{
				gemm_cpu(1, 0, C_in, HW, C_out, 1.0f,
					const_cast<float*>(weights), C_in,
					const_cast<float*>(dout) + b * C_out * HW, HW,
					1.0f,
					dinput + b * C_in * HW, HW);
			}
		}
	}

	static void gate_linear_forward(const float *xln, const float *gfeat, float *out,
		const float *weights, const float *bias, int B, int C, int HW)
	{
		for (int b = 0; b < B; ++b)
		{
			float *dst = out + b * C * HW;
			gemm_cpu(0, 0, C, HW, C, 1.0f,
				const_cast<float*>(weights), 2 * C,
				const_cast<float*>(xln) + b * C * HW, HW,
				0.0f,
				dst, HW);
			gemm_cpu(0, 0, C, HW, C, 1.0f,
				const_cast<float*>(weights) + C, 2 * C,
				const_cast<float*>(gfeat) + b * C * HW, HW,
				1.0f,
				dst, HW);
		}

		add_bias_nchw(out, bias, B, C, HW);
	}

	static void gate_linear_backward(const float *xln, const float *gfeat, const float *dout,
		const float *weights, float *weight_updates, float *bias_updates,
		float *d_xln, float *d_gfeat, int B, int C, int HW)
	{
		if (bias_updates)
		{
			for (int b = 0; b < B; ++b)
			{
				for (int c = 0; c < C; ++c)
				{
					const float *src = dout + (b * C + c) * HW;
					float sum = 0.0f;
					for (int s = 0; s < HW; ++s)
					{
						sum += src[s];
					}
					bias_updates[c] += sum;
				}
			}
		}

		for (int b = 0; b < B; ++b)
		{
			const float *dout_b = dout + b * C * HW;
			const float *xln_b = xln + b * C * HW;
			const float *gfeat_b = gfeat + b * C * HW;

			gemm_cpu(0, 1, C, C, HW, 1.0f,
				const_cast<float*>(dout_b), HW,
				const_cast<float*>(xln_b), HW,
				1.0f,
				weight_updates, 2 * C);
			gemm_cpu(0, 1, C, C, HW, 1.0f,
				const_cast<float*>(dout_b), HW,
				const_cast<float*>(gfeat_b), HW,
				1.0f,
				weight_updates + C, 2 * C);

			if (d_xln)
			{
				gemm_cpu(1, 0, C, HW, C, 1.0f,
					const_cast<float*>(weights), 2 * C,
					const_cast<float*>(dout_b), HW,
					1.0f,
					d_xln + b * C * HW, HW);
			}

			if (d_gfeat)
			{
				gemm_cpu(1, 0, C, HW, C, 1.0f,
					const_cast<float*>(weights) + C, 2 * C,
					const_cast<float*>(dout_b), HW,
					1.0f,
					d_gfeat + b * C * HW, HW);
			}
		}
	}

	static void clifford_rolling_forward_local(
		const float *z_det, const float *z_ctx, float *g_raw,
		int B, int C, int HW,
		const int *wedge_shifts, int num_wedge_shifts,
		const int *inner_shifts, int num_inner_shifts,
		int cli_mode)
	{
		const int raw_C = clifford_raw_channels(cli_mode, num_wedge_shifts, num_inner_shifts, C);
		for (int b = 0; b < B; ++b)
		{
			const size_t state_base = static_cast<size_t>(b) * C * HW;
			const size_t raw_base = static_cast<size_t>(b) * raw_C * HW;
			if (cli_mode != 0)
			{
				for (int shift_idx = 0; shift_idx < num_wedge_shifts; ++shift_idx)
				{
					const int shift = wedge_shifts[shift_idx];
					const int base_channel = (cli_mode == 1) ? (shift_idx * C) : clifford_wedge_base_channel(shift_idx, num_inner_shifts, C);
					for (int c = 0; c < C; ++c)
					{
						const int c_roll = clifford_wrap_channel(c, shift, C);
						const float *det_cur = z_det + state_base + static_cast<size_t>(c) * HW;
						const float *ctx_cur = z_ctx + state_base + static_cast<size_t>(c) * HW;
						const float *det_roll = z_det + state_base + static_cast<size_t>(c_roll) * HW;
						const float *ctx_roll = z_ctx + state_base + static_cast<size_t>(c_roll) * HW;
						float *out = g_raw + raw_base + static_cast<size_t>(base_channel + c) * HW;
						for (int s = 0; s < HW; ++s)
						{
							const float prod = det_cur[s] * ctx_roll[s];
							out[s] = prod - ctx_cur[s] * det_roll[s];
						}
					}
				}
			}

			if (cli_mode != 1)
			{
				for (int shift_idx = 0; shift_idx < num_inner_shifts; ++shift_idx)
				{
					const int shift = inner_shifts[shift_idx];
					const int base_channel = (cli_mode == 0) ? (shift_idx * C) : clifford_inner_base_channel(shift_idx, num_wedge_shifts, C);
					for (int c = 0; c < C; ++c)
					{
						const int c_roll = clifford_wrap_channel(c, shift, C);
						const float *det_cur = z_det + state_base + static_cast<size_t>(c) * HW;
						const float *ctx_roll = z_ctx + state_base + static_cast<size_t>(c_roll) * HW;
						float *out = g_raw + raw_base + static_cast<size_t>(base_channel + c) * HW;
						for (int s = 0; s < HW; ++s)
						{
							const float prod = det_cur[s] * ctx_roll[s];
							out[s] = prod * logistic_activate(prod);
						}
					}
				}
			}
		}
	}

	static void clifford_rolling_backward_local(
		const float *z_det, const float *z_ctx, const float *d_g_raw,
		float *d_z_det, float *d_z_ctx,
		int B, int C, int HW,
		const int *wedge_shifts, int num_wedge_shifts,
		const int *inner_shifts, int num_inner_shifts,
		int cli_mode)
	{
		const int raw_C = clifford_raw_channels(cli_mode, num_wedge_shifts, num_inner_shifts, C);
		for (int b = 0; b < B; ++b)
		{
			const size_t state_base = static_cast<size_t>(b) * C * HW;
			const size_t raw_base = static_cast<size_t>(b) * raw_C * HW;
			if (cli_mode != 0)
			{
				for (int shift_idx = 0; shift_idx < num_wedge_shifts; ++shift_idx)
				{
					const int shift = wedge_shifts[shift_idx];
					const int base_channel = (cli_mode == 1) ? (shift_idx * C) : clifford_wedge_base_channel(shift_idx, num_inner_shifts, C);
					for (int c = 0; c < C; ++c)
					{
						const int c_roll = clifford_wrap_channel(c, shift, C);
						const float *det_cur = z_det + state_base + static_cast<size_t>(c) * HW;
						const float *ctx_cur = z_ctx + state_base + static_cast<size_t>(c) * HW;
						const float *det_roll = z_det + state_base + static_cast<size_t>(c_roll) * HW;
						const float *ctx_roll = z_ctx + state_base + static_cast<size_t>(c_roll) * HW;
						float *d_det_cur = d_z_det + state_base + static_cast<size_t>(c) * HW;
						float *d_ctx_cur = d_z_ctx + state_base + static_cast<size_t>(c) * HW;
						float *d_det_roll = d_z_det + state_base + static_cast<size_t>(c_roll) * HW;
						float *d_ctx_roll = d_z_ctx + state_base + static_cast<size_t>(c_roll) * HW;
						const float *d_wedge = d_g_raw + raw_base + static_cast<size_t>(base_channel + c) * HW;
						for (int s = 0; s < HW; ++s)
						{
							const float grad = d_wedge[s];
							if (grad != 0.0f)
							{
								d_det_cur[s] += grad * ctx_roll[s];
								d_ctx_roll[s] += grad * det_cur[s];
								d_ctx_cur[s] -= grad * det_roll[s];
								d_det_roll[s] -= grad * ctx_cur[s];
							}
						}
					}
				}
			}

			if (cli_mode != 1)
			{
				for (int shift_idx = 0; shift_idx < num_inner_shifts; ++shift_idx)
				{
					const int shift = inner_shifts[shift_idx];
					const int base_channel = (cli_mode == 0) ? (shift_idx * C) : clifford_inner_base_channel(shift_idx, num_wedge_shifts, C);
					for (int c = 0; c < C; ++c)
					{
						const int c_roll = clifford_wrap_channel(c, shift, C);
						const float *det_cur = z_det + state_base + static_cast<size_t>(c) * HW;
						const float *ctx_roll = z_ctx + state_base + static_cast<size_t>(c_roll) * HW;
						float *d_det_cur = d_z_det + state_base + static_cast<size_t>(c) * HW;
						float *d_ctx_roll = d_z_ctx + state_base + static_cast<size_t>(c_roll) * HW;
						const float *d_inner = d_g_raw + raw_base + static_cast<size_t>(base_channel + c) * HW;
						for (int s = 0; s < HW; ++s)
						{
							const float prod = det_cur[s] * ctx_roll[s];
							const float sig = logistic_activate(prod);
							const float d_prod = d_inner[s] * (sig + prod * sig * (1.0f - sig));
							d_det_cur[s] += d_prod * ctx_roll[s];
							d_ctx_roll[s] += d_prod * det_cur[s];
						}
					}
				}
			}
		}
	}

	static void clifford_rolling_forward_global(
		const float *state_stream, const float *global_ctx, float *g_raw,
		int B, int C, int HW,
		const int *wedge_shifts, int num_wedge_shifts,
		const int *inner_shifts, int num_inner_shifts,
		int cli_mode)
	{
		const int raw_C = clifford_raw_channels(cli_mode, num_wedge_shifts, num_inner_shifts, C);
		for (int b = 0; b < B; ++b)
		{
			const float *g = global_ctx + b * C;
			const size_t state_base = static_cast<size_t>(b) * C * HW;
			const size_t raw_base = static_cast<size_t>(b) * raw_C * HW;
			if (cli_mode != 0)
			{
				for (int shift_idx = 0; shift_idx < num_wedge_shifts; ++shift_idx)
				{
					const int shift = wedge_shifts[shift_idx];
					const int base_channel = (cli_mode == 1) ? (shift_idx * C) : clifford_wedge_base_channel(shift_idx, num_inner_shifts, C);
					for (int c = 0; c < C; ++c)
					{
						const int c_roll = clifford_wrap_channel(c, shift, C);
						const float g_cur = g[c];
						const float g_roll = g[c_roll];
						const float *state_cur = state_stream + state_base + static_cast<size_t>(c) * HW;
						const float *state_roll = state_stream + state_base + static_cast<size_t>(c_roll) * HW;
						float *out = g_raw + raw_base + static_cast<size_t>(base_channel + c) * HW;
						for (int s = 0; s < HW; ++s)
						{
							const float prod = state_cur[s] * g_roll;
							out[s] = prod - g_cur * state_roll[s];
						}
					}
				}
			}

			if (cli_mode != 1)
			{
				for (int shift_idx = 0; shift_idx < num_inner_shifts; ++shift_idx)
				{
					const int shift = inner_shifts[shift_idx];
					const int base_channel = (cli_mode == 0) ? (shift_idx * C) : clifford_inner_base_channel(shift_idx, num_wedge_shifts, C);
					for (int c = 0; c < C; ++c)
					{
						const int c_roll = clifford_wrap_channel(c, shift, C);
						const float g_roll = g[c_roll];
						const float *state_cur = state_stream + state_base + static_cast<size_t>(c) * HW;
						float *out = g_raw + raw_base + static_cast<size_t>(base_channel + c) * HW;
						for (int s = 0; s < HW; ++s)
						{
							const float prod = state_cur[s] * g_roll;
							out[s] = prod * logistic_activate(prod);
						}
					}
				}
			}
		}
	}

	static void clifford_rolling_backward_global(
		const float *state_stream, const float *global_ctx, const float *d_g_raw,
		float *d_state_stream, float *d_global_ctx,
		int B, int C, int HW,
		const int *wedge_shifts, int num_wedge_shifts,
		const int *inner_shifts, int num_inner_shifts,
		int cli_mode)
	{
		const int raw_C = clifford_raw_channels(cli_mode, num_wedge_shifts, num_inner_shifts, C);
		for (int b = 0; b < B; ++b)
		{
			const float *g = global_ctx + b * C;
			float *d_g = d_global_ctx + b * C;
			const size_t state_base = static_cast<size_t>(b) * C * HW;
			const size_t raw_base = static_cast<size_t>(b) * raw_C * HW;
			if (cli_mode != 0)
			{
				for (int shift_idx = 0; shift_idx < num_wedge_shifts; ++shift_idx)
				{
					const int shift = wedge_shifts[shift_idx];
					const int base_channel = (cli_mode == 1) ? (shift_idx * C) : clifford_wedge_base_channel(shift_idx, num_inner_shifts, C);
					for (int c = 0; c < C; ++c)
					{
						const int c_roll = clifford_wrap_channel(c, shift, C);
						const float g_cur = g[c];
						const float g_roll = g[c_roll];
						const float *state_cur = state_stream + state_base + static_cast<size_t>(c) * HW;
						const float *state_roll = state_stream + state_base + static_cast<size_t>(c_roll) * HW;
						float *d_state_cur = d_state_stream + state_base + static_cast<size_t>(c) * HW;
						float *d_state_roll = d_state_stream + state_base + static_cast<size_t>(c_roll) * HW;
						const float *d_wedge = d_g_raw + raw_base + static_cast<size_t>(base_channel + c) * HW;
						for (int s = 0; s < HW; ++s)
						{
							const float grad = d_wedge[s];
							if (grad != 0.0f)
							{
								d_state_cur[s] += grad * g_roll;
								d_g[c_roll] += grad * state_cur[s];
								d_g[c] -= grad * state_roll[s];
								d_state_roll[s] -= grad * g_cur;
							}
						}
					}
				}
			}

			if (cli_mode != 1)
			{
				for (int shift_idx = 0; shift_idx < num_inner_shifts; ++shift_idx)
				{
					const int shift = inner_shifts[shift_idx];
					const int base_channel = (cli_mode == 0) ? (shift_idx * C) : clifford_inner_base_channel(shift_idx, num_wedge_shifts, C);
					for (int c = 0; c < C; ++c)
					{
						const int c_roll = clifford_wrap_channel(c, shift, C);
						const float g_roll = g[c_roll];
						const float *state_cur = state_stream + state_base + static_cast<size_t>(c) * HW;
						float *d_state_cur = d_state_stream + state_base + static_cast<size_t>(c) * HW;
						const float *d_inner = d_g_raw + raw_base + static_cast<size_t>(base_channel + c) * HW;
						for (int s = 0; s < HW; ++s)
						{
							const float prod = state_cur[s] * g_roll;
							const float sig = logistic_activate(prod);
							const float d_prod = d_inner[s] * (sig + prod * sig * (1.0f - sig));
							d_state_cur[s] += d_prod * g_roll;
							d_g[c_roll] += d_prod * state_cur[s];
						}
					}
				}
			}
		}
	}

	static void clifford_vb_forward_local(
		const float *z_det, const float *bivector_feat, float *vb_feat,
		int B, int C, int HW,
		const int *shifts, int num_shifts)
	{
		if (num_shifts < 1)
		{
			return;
		}

		const float scale = 1.0f / static_cast<float>(num_shifts);
		memset(vb_feat, 0, sizeof(float) * static_cast<size_t>(B) * C * HW);
		for (int b = 0; b < B; ++b)
		{
			const size_t state_base = static_cast<size_t>(b) * C * HW;
			for (int shift_idx = 0; shift_idx < num_shifts; ++shift_idx)
			{
				const int shift = shifts[shift_idx];
				for (int c = 0; c < C; ++c)
				{
					const int c_roll = clifford_wrap_channel(c, shift, C);
					const float *det_cur = z_det + state_base + static_cast<size_t>(c) * HW;
					const float *biv_cur = bivector_feat + state_base + static_cast<size_t>(c) * HW;
					const float *det_roll = z_det + state_base + static_cast<size_t>(c_roll) * HW;
					const float *biv_roll = bivector_feat + state_base + static_cast<size_t>(c_roll) * HW;
					float *out = vb_feat + state_base + static_cast<size_t>(c) * HW;
					for (int s = 0; s < HW; ++s)
					{
						out[s] += scale * (det_cur[s] * biv_roll[s] - biv_cur[s] * det_roll[s]);
					}
				}
			}
		}
	}

	static void clifford_vb_backward_local(
		const float *z_det, const float *bivector_feat, const float *d_vb_feat,
		float *d_z_det, float *d_bivector_feat,
		int B, int C, int HW,
		const int *shifts, int num_shifts)
	{
		if (num_shifts < 1)
		{
			return;
		}

		const float scale = 1.0f / static_cast<float>(num_shifts);
		for (int b = 0; b < B; ++b)
		{
			const size_t state_base = static_cast<size_t>(b) * C * HW;
			for (int shift_idx = 0; shift_idx < num_shifts; ++shift_idx)
			{
				const int shift = shifts[shift_idx];
				for (int c = 0; c < C; ++c)
				{
					const int c_roll = clifford_wrap_channel(c, shift, C);
					const float *det_cur = z_det + state_base + static_cast<size_t>(c) * HW;
					const float *biv_cur = bivector_feat + state_base + static_cast<size_t>(c) * HW;
					const float *det_roll = z_det + state_base + static_cast<size_t>(c_roll) * HW;
					const float *biv_roll = bivector_feat + state_base + static_cast<size_t>(c_roll) * HW;
					float *d_det_cur = d_z_det + state_base + static_cast<size_t>(c) * HW;
					float *d_biv_cur = d_bivector_feat + state_base + static_cast<size_t>(c) * HW;
					float *d_det_roll = d_z_det + state_base + static_cast<size_t>(c_roll) * HW;
					float *d_biv_roll = d_bivector_feat + state_base + static_cast<size_t>(c_roll) * HW;
					const float *grad_cur = d_vb_feat + state_base + static_cast<size_t>(c) * HW;
					for (int s = 0; s < HW; ++s)
					{
						const float grad = scale * grad_cur[s];
						if (grad != 0.0f)
						{
							d_det_cur[s] += grad * biv_roll[s];
							d_biv_roll[s] += grad * det_cur[s];
							d_biv_cur[s] -= grad * det_roll[s];
							d_det_roll[s] -= grad * biv_cur[s];
						}
					}
				}
			}
		}
	}

	static void global_avg_pool_forward(const float *input, float *output, int B, int C, int HW)
	{
		const float inv_hw = 1.0f / static_cast<float>(HW);
		for (int b = 0; b < B; ++b)
		{
			for (int c = 0; c < C; ++c)
			{
				const float *src = input + (b * C + c) * HW;
				float sum = 0.0f;
				for (int s = 0; s < HW; ++s)
				{
					sum += src[s];
				}
				output[b * C + c] = sum * inv_hw;
			}
		}
	}

	static void global_avg_pool_backward(const float *d_out, float *d_in, int B, int C, int HW)
	{
		const float inv_hw = 1.0f / static_cast<float>(HW);
		for (int b = 0; b < B; ++b)
		{
			for (int c = 0; c < C; ++c)
			{
				float *dst = d_in + (b * C + c) * HW;
				const float grad = d_out[b * C + c] * inv_hw;
				for (int s = 0; s < HW; ++s)
				{
					dst[s] += grad;
				}
			}
		}
	}

	static void forward_clifford_dwconv_stack(Darknet::Layer & l, const float *input, float *output, Darknet::NetworkState state)
	{
		const size_t total = clifford_tensor_count(l);
		const float *current_input = input;
		Darknet::NetworkState sub_state = {0};
		sub_state.workspace = state.workspace;
		sub_state.train = state.train;
		sub_state.index = state.index;
		sub_state.net = state.net;

		for (int i = 0; i < l.cli_num_dwconv; ++i)
		{
			sub_state.input = const_cast<float*>(current_input);
			forward_convolutional_layer(l.cli_dwconv[i], sub_state);
			current_input = l.cli_dwconv[i].output;
		}

		copy_cpu(static_cast<int>(total), const_cast<float*>(current_input), 1, output, 1);
	}

	static void backward_clifford_dwconv_stack(Darknet::Layer & l, const float *input, const float *d_output,
		float *d_input, Darknet::NetworkState state)
	{
		const size_t total = clifford_tensor_count(l);
		copy_cpu(static_cast<int>(total), const_cast<float*>(d_output), 1, l.cli_dwconv[l.cli_num_dwconv - 1].delta, 1);

		Darknet::NetworkState sub_state = {0};
		sub_state.workspace = state.workspace;
		sub_state.train = state.train;
		sub_state.index = state.index;
		sub_state.net = state.net;

		for (int i = l.cli_num_dwconv - 1; i >= 0; --i)
		{
			sub_state.input = (i == 0) ? const_cast<float*>(input) : l.cli_dwconv[i - 1].output;
			sub_state.delta = (i == 0) ? d_input : l.cli_dwconv[i - 1].delta;
			if (sub_state.delta != nullptr)
			{
				fill_cpu(static_cast<int>(total), 0.0f, sub_state.delta, 1);
			}
			backward_convolutional_layer(l.cli_dwconv[i], sub_state);
		}
	}

	static void allocate_clifford_runtime_buffers(Darknet::Layer & l)
	{
		const size_t common_count = clifford_tensor_count(l);
		const size_t token_count = clifford_token_count(l);
		const size_t raw_count = clifford_has_local(l) ? common_count * l.cli_proj_in_dim / l.c : 0;
		const size_t raw_count_g = clifford_has_global(l) ? common_count * l.cli_proj_in_dim / l.c : 0;
		const size_t global_count = clifford_has_global(l) ? static_cast<size_t>(l.batch) * l.c : 0;

		l.cli_ln_out = alloc_float_buffer(common_count);
		l.cli_ln_mean = alloc_float_buffer(token_count);
		l.cli_ln_var = alloc_float_buffer(token_count);
		l.cli_ln_xhat = alloc_float_buffer(common_count);
		l.cli_z_det = alloc_float_buffer(common_count);
		l.cli_z_ctx = alloc_float_buffer(common_count);
		l.cli_z_ctx_pre_diff = alloc_float_buffer(common_count);
		l.cli_g_raw = alloc_float_buffer(raw_count);
		l.cli_g_feat = alloc_float_buffer(clifford_has_local(l) ? common_count : 0);
		l.cli_gate_alpha = alloc_float_buffer(clifford_has_local(l) ? common_count : 0);
		l.cli_gate_pre_sigmoid = alloc_float_buffer(clifford_has_local(l) ? common_count : 0);
		l.cli_vb_feat = alloc_float_buffer(clifford_has_higher_local(l) ? common_count : 0);
		l.cli_hmix = alloc_float_buffer(common_count);
		l.cli_drop_mask = alloc_float_buffer(l.batch);

		l.cli_global_ctx = alloc_float_buffer(global_count);
		l.cli_g_raw_g = alloc_float_buffer(raw_count_g);
		l.cli_g_feat_g = alloc_float_buffer(clifford_has_global(l) ? common_count : 0);
		l.cli_gate_alpha_g = alloc_float_buffer(clifford_has_global(l) ? common_count : 0);
		l.cli_gate_pre_sigmoid_g = alloc_float_buffer(clifford_has_global(l) ? common_count : 0);
	}

	static void resize_clifford_runtime_buffers(Darknet::Layer *l)
	{
		const size_t common_count = clifford_tensor_count(*l);
		const size_t token_count = clifford_token_count(*l);
		const size_t raw_count = clifford_has_local(*l) ? common_count * l->cli_proj_in_dim / l->c : 0;
		const size_t raw_count_g = clifford_has_global(*l) ? common_count * l->cli_proj_in_dim / l->c : 0;
		const size_t global_count = clifford_has_global(*l) ? static_cast<size_t>(l->batch) * l->c : 0;

		l->cli_ln_out = resize_float_buffer(l->cli_ln_out, common_count);
		l->cli_ln_mean = resize_float_buffer(l->cli_ln_mean, token_count);
		l->cli_ln_var = resize_float_buffer(l->cli_ln_var, token_count);
		l->cli_ln_xhat = resize_float_buffer(l->cli_ln_xhat, common_count);
		l->cli_z_det = resize_float_buffer(l->cli_z_det, common_count);
		l->cli_z_ctx = resize_float_buffer(l->cli_z_ctx, common_count);
		l->cli_z_ctx_pre_diff = resize_float_buffer(l->cli_z_ctx_pre_diff, common_count);
		l->cli_g_raw = resize_float_buffer(l->cli_g_raw, raw_count);
		l->cli_g_feat = resize_float_buffer(l->cli_g_feat, clifford_has_local(*l) ? common_count : 0);
		l->cli_gate_alpha = resize_float_buffer(l->cli_gate_alpha, clifford_has_local(*l) ? common_count : 0);
		l->cli_gate_pre_sigmoid = resize_float_buffer(l->cli_gate_pre_sigmoid, clifford_has_local(*l) ? common_count : 0);
		l->cli_vb_feat = resize_float_buffer(l->cli_vb_feat, clifford_has_higher_local(*l) ? common_count : 0);
		l->cli_hmix = resize_float_buffer(l->cli_hmix, common_count);
		l->cli_drop_mask = resize_float_buffer(l->cli_drop_mask, l->batch);

		l->cli_global_ctx = resize_float_buffer(l->cli_global_ctx, global_count);
		l->cli_g_raw_g = resize_float_buffer(l->cli_g_raw_g, raw_count_g);
		l->cli_g_feat_g = resize_float_buffer(l->cli_g_feat_g, clifford_has_global(*l) ? common_count : 0);
		l->cli_gate_alpha_g = resize_float_buffer(l->cli_gate_alpha_g, clifford_has_global(*l) ? common_count : 0);
		l->cli_gate_pre_sigmoid_g = resize_float_buffer(l->cli_gate_pre_sigmoid_g, clifford_has_global(*l) ? common_count : 0);
	}

	static void ensure_clifford_dwconv_cpu_training_buffers(Darknet::Layer & dwconv)
	{
		if (!dwconv.train)
		{
			return;
		}

		const int steps = (dwconv.steps > 0) ? dwconv.steps : 1;
		const size_t output_count = static_cast<size_t>(dwconv.batch) * steps * dwconv.outputs;
		if (dwconv.delta == nullptr)
		{
			dwconv.delta = alloc_float_buffer(output_count);
		}
		if (dwconv.batch_normalize)
		{
			if (dwconv.x == nullptr)
			{
				dwconv.x = alloc_float_buffer(output_count);
			}
			if (dwconv.x_norm == nullptr)
			{
				dwconv.x_norm = alloc_float_buffer(output_count);
			}
		}
	}

	static size_t clifford_dwconv_cpu_workspace_size(const Darknet::Layer & dwconv)
	{
		const int groups = (dwconv.groups > 0) ? dwconv.groups : 1;
		const int channels_per_group = dwconv.c / groups;
		return static_cast<size_t>(dwconv.out_h) * dwconv.out_w *
			dwconv.size * dwconv.size * channels_per_group * sizeof(float);
	}

	static void update_clifford_dwconv_workspace_size(Darknet::Layer & l, const Darknet::Layer & dwconv)
	{
		if (dwconv.workspace_size > l.workspace_size)
		{
			l.workspace_size = dwconv.workspace_size;
		}
		const size_t cpu_workspace_size = clifford_dwconv_cpu_workspace_size(dwconv);
		if (cpu_workspace_size > l.workspace_size)
		{
			l.workspace_size = cpu_workspace_size;
		}
	}
}

Darknet::Layer make_clifford_layer(int batch, int h, int w, int c, int n,
	const int *shifts, int num_shifts,
	const int *inner_shifts, int num_inner_shifts,
	int ctx_mode, int cli_mode, int gffn_mode, int higher_mode,
	int dwconv_size, int num_dwconv,
	ACTIVATION activation, float drop_path, float layerscale_init,
	int index, int train)
{
	TAT(TATPARMS);

	if (n != c)
	{
		darknet_fatal_error(DARKNET_LOC, "clifford layer currently requires filters (%d) to match input channels (%d)", n, c);
	}
	if (num_shifts < 1 || shifts == nullptr)
	{
		darknet_fatal_error(DARKNET_LOC, "clifford layer requires at least one shift");
	}
	if (num_inner_shifts < 0 || (num_inner_shifts > 0 && inner_shifts == nullptr))
	{
		darknet_fatal_error(DARKNET_LOC, "invalid clifford inner shift schedule");
	}
	if (ctx_mode < 0 || ctx_mode > 1)
	{
		darknet_fatal_error(DARKNET_LOC, "invalid clifford ctx_mode=%d", ctx_mode);
	}
	if (cli_mode < 0 || cli_mode > 2)
	{
		darknet_fatal_error(DARKNET_LOC, "invalid clifford cli_mode=%d", cli_mode);
	}
	if (gffn_mode < 0 || gffn_mode > 2)
	{
		darknet_fatal_error(DARKNET_LOC, "invalid clifford gffn_mode=%d", gffn_mode);
	}
	if (higher_mode < 0 || higher_mode > 1)
	{
		darknet_fatal_error(DARKNET_LOC, "invalid clifford higher_mode=%d", higher_mode);
	}
	if (activation != SWISH)
	{
		darknet_fatal_error(DARKNET_LOC, "clifford layer currently requires activation=swish");
	}
	if (dwconv_size < 1 || (dwconv_size % 2) == 0)
	{
		darknet_fatal_error(DARKNET_LOC, "clifford layer requires an odd dwconv_size >= 1");
	}
	if (drop_path < 0.0f || drop_path >= 1.0f)
	{
		darknet_fatal_error(DARKNET_LOC, "clifford layer requires drop_path in [0, 1)");
	}

	Darknet::Layer l = { (Darknet::ELayerType)0 };
	l.type = Darknet::ELayerType::CLIFFORD;
	l.batch = batch;
	l.h = h;
	l.w = w;
	l.c = c;
	l.n = n;
	l.out_h = h;
	l.out_w = w;
	l.out_c = c;
	l.inputs = h * w * c;
	l.outputs = l.inputs;
	l.index = index;
	l.train = train;
	l.activation = activation;
	l.learning_rate_scale = 1.0f;
	l.size = dwconv_size;
	l.stride = 1;
	l.stride_x = 1;
	l.stride_y = 1;
	l.dilation = 1;
	l.pad = dwconv_size / 2;
	l.cli_num_shifts = num_shifts;
	const bool inner_matches_wedge =
		(num_inner_shifts > 0) &&
		clifford_shift_schedules_match(shifts, num_shifts, inner_shifts, num_inner_shifts, c);
	l.cli_num_shifts_inner = inner_matches_wedge ? 0 : num_inner_shifts;
	l.cli_ctx_mode = ctx_mode;
	l.cli_interaction_mode = cli_mode;
	l.cli_gffn_mode = gffn_mode;
	l.cli_higher_mode = higher_mode;
	l.cli_drop_path = drop_path;
	l.cli_layerscale_init = layerscale_init;
	l.cli_num_dwconv = std::max(1, num_dwconv);

	l.cli_shifts = (int*)xcalloc(l.cli_num_shifts, sizeof(int));
	for (int i = 0; i < l.cli_num_shifts; ++i)
	{
		l.cli_shifts[i] = clifford_normalize_shift(shifts[i], c);
	}
	if (l.cli_num_shifts_inner > 0)
	{
		l.cli_shifts_inner = (int*)xcalloc(l.cli_num_shifts_inner, sizeof(int));
		for (int i = 0; i < l.cli_num_shifts_inner; ++i)
		{
			l.cli_shifts_inner[i] = clifford_normalize_shift(inner_shifts[i], c);
		}
	}
	l.cli_proj_in_dim = clifford_raw_channels(l);

	const int inner_shift_count = clifford_inner_num_shifts(l);
	const int unique_wedge_shift_count = clifford_count_unique_shifts(l.cli_shifts, l.cli_num_shifts, c);
	const int unique_inner_shift_count = clifford_count_unique_shifts(clifford_inner_shifts(l), inner_shift_count, c);
	const int zero_wedge_shift_count = clifford_count_matching_shifts(l.cli_shifts, l.cli_num_shifts, 0);
	const int zero_inner_shift_count = clifford_count_matching_shifts(clifford_inner_shifts(l), inner_shift_count, 0);

	l.output = (float*)xcalloc(static_cast<size_t>(batch) * l.outputs, sizeof(float));
	l.delta = (float*)xcalloc(static_cast<size_t>(batch) * l.outputs, sizeof(float));

	l.cli_w_det = (float*)xcalloc(static_cast<size_t>(c) * c, sizeof(float));
	l.cli_w_det_updates = (float*)xcalloc(static_cast<size_t>(c) * c, sizeof(float));
	l.cli_b_det = (float*)xcalloc(c, sizeof(float));
	l.cli_b_det_updates = (float*)xcalloc(c, sizeof(float));

	l.cli_w_proj = (float*)xcalloc(static_cast<size_t>(c) * l.cli_proj_in_dim, sizeof(float));
	l.cli_w_proj_updates = (float*)xcalloc(static_cast<size_t>(c) * l.cli_proj_in_dim, sizeof(float));
	l.cli_b_proj = (float*)xcalloc(c, sizeof(float));
	l.cli_b_proj_updates = (float*)xcalloc(c, sizeof(float));

	l.cli_w_gate = (float*)xcalloc(static_cast<size_t>(c) * 2 * c, sizeof(float));
	l.cli_w_gate_updates = (float*)xcalloc(static_cast<size_t>(c) * 2 * c, sizeof(float));
	l.cli_b_gate = (float*)xcalloc(c, sizeof(float));
	l.cli_b_gate_updates = (float*)xcalloc(c, sizeof(float));

	l.cli_ln_gamma = (float*)xcalloc(c, sizeof(float));
	l.cli_ln_gamma_updates = (float*)xcalloc(c, sizeof(float));
	l.cli_ln_beta = (float*)xcalloc(c, sizeof(float));
	l.cli_ln_beta_updates = (float*)xcalloc(c, sizeof(float));
	l.cli_layer_scale = (float*)xcalloc(c, sizeof(float));
	l.cli_layer_scale_updates = (float*)xcalloc(c, sizeof(float));

	if (clifford_has_global(l))
	{
		l.cli_w_proj_g = (float*)xcalloc(static_cast<size_t>(c) * l.cli_proj_in_dim, sizeof(float));
		l.cli_w_proj_g_updates = (float*)xcalloc(static_cast<size_t>(c) * l.cli_proj_in_dim, sizeof(float));
		l.cli_b_proj_g = (float*)xcalloc(c, sizeof(float));
		l.cli_b_proj_g_updates = (float*)xcalloc(c, sizeof(float));
		l.cli_w_gate_g = (float*)xcalloc(static_cast<size_t>(c) * 2 * c, sizeof(float));
		l.cli_w_gate_g_updates = (float*)xcalloc(static_cast<size_t>(c) * 2 * c, sizeof(float));
		l.cli_b_gate_g = (float*)xcalloc(c, sizeof(float));
		l.cli_b_gate_g_updates = (float*)xcalloc(c, sizeof(float));
	}

	const float det_scale = std::sqrt(2.0f / static_cast<float>(c + c));
	rand_uniform_many_weight_init(l.cli_w_det, static_cast<size_t>(c) * c, -1.0f, 1.0f, det_scale);

	const float proj_scale = std::sqrt(2.0f / static_cast<float>(l.cli_proj_in_dim + c));
	rand_uniform_many_weight_init(l.cli_w_proj, static_cast<size_t>(c) * l.cli_proj_in_dim, -1.0f, 1.0f, proj_scale);

	const float gate_scale = std::sqrt(2.0f / static_cast<float>(3 * c));
	rand_uniform_many_weight_init(l.cli_w_gate, static_cast<size_t>(c) * 2 * c, -1.0f, 1.0f, gate_scale);

	if (clifford_has_global(l))
	{
		rand_uniform_many_weight_init(l.cli_w_proj_g, static_cast<size_t>(c) * l.cli_proj_in_dim, -1.0f, 1.0f, proj_scale);
		rand_uniform_many_weight_init(l.cli_w_gate_g, static_cast<size_t>(c) * 2 * c, -1.0f, 1.0f, gate_scale);
	}

	for (int i = 0; i < c; ++i)
	{
		l.cli_ln_gamma[i] = 1.0f;
		l.cli_layer_scale[i] = layerscale_init;
	}

	l.cli_dwconv = (Darknet::Layer*)xcalloc(l.cli_num_dwconv, sizeof(Darknet::Layer));
	for (int i = 0; i < l.cli_num_dwconv; ++i)
	{
		l.cli_dwconv[i] = make_convolutional_layer(
			batch, 1, h, w, c, c, c,
			dwconv_size, 1, 1, 1, dwconv_size / 2,
			LINEAR, 1,
			0, 0, 0, 0, index, 0, nullptr, 0, 0, train);
		ensure_clifford_dwconv_cpu_training_buffers(l.cli_dwconv[i]);
		update_clifford_dwconv_workspace_size(l, l.cli_dwconv[i]);
	}

	allocate_clifford_runtime_buffers(l);

	l.forward = forward_clifford_layer;
	l.backward = backward_clifford_layer;
	l.update = update_clifford_layer;

#ifdef DARKNET_GPU
	l.forward_gpu = forward_clifford_layer_gpu;
	l.backward_gpu = backward_clifford_layer_gpu;
	l.update_gpu = update_clifford_layer_gpu;

	if (cfg_and_state.gpu_index >= 0)
		{
			const size_t output_count = static_cast<size_t>(batch) * l.outputs;
			l.output_gpu = cuda_make_array(l.output, output_count);
			l.delta_gpu = cuda_make_array(l.delta, output_count);
			l.cli_shifts_gpu = alloc_gpu_int_buffer(l.cli_shifts, l.cli_num_shifts);
			l.cli_shifts_inner_gpu = alloc_gpu_int_buffer(l.cli_shifts_inner, l.cli_num_shifts_inner);

		l.cli_w_det_gpu = alloc_gpu_buffer(l.cli_w_det, static_cast<size_t>(c) * c);
		l.cli_w_det_updates_gpu = alloc_gpu_buffer(l.cli_w_det_updates, static_cast<size_t>(c) * c);
		l.cli_b_det_gpu = alloc_gpu_buffer(l.cli_b_det, c);
		l.cli_b_det_updates_gpu = alloc_gpu_buffer(l.cli_b_det_updates, c);
		l.cli_w_proj_gpu = alloc_gpu_buffer(l.cli_w_proj, static_cast<size_t>(c) * l.cli_proj_in_dim);
		l.cli_w_proj_updates_gpu = alloc_gpu_buffer(l.cli_w_proj_updates, static_cast<size_t>(c) * l.cli_proj_in_dim);
		l.cli_b_proj_gpu = alloc_gpu_buffer(l.cli_b_proj, c);
		l.cli_b_proj_updates_gpu = alloc_gpu_buffer(l.cli_b_proj_updates, c);
		l.cli_w_gate_gpu = alloc_gpu_buffer(l.cli_w_gate, static_cast<size_t>(c) * 2 * c);
		l.cli_w_gate_updates_gpu = alloc_gpu_buffer(l.cli_w_gate_updates, static_cast<size_t>(c) * 2 * c);
		l.cli_b_gate_gpu = alloc_gpu_buffer(l.cli_b_gate, c);
		l.cli_b_gate_updates_gpu = alloc_gpu_buffer(l.cli_b_gate_updates, c);
		l.cli_ln_gamma_gpu = alloc_gpu_buffer(l.cli_ln_gamma, c);
		l.cli_ln_gamma_updates_gpu = alloc_gpu_buffer(l.cli_ln_gamma_updates, c);
		l.cli_ln_beta_gpu = alloc_gpu_buffer(l.cli_ln_beta, c);
		l.cli_ln_beta_updates_gpu = alloc_gpu_buffer(l.cli_ln_beta_updates, c);
		l.cli_layer_scale_gpu = alloc_gpu_buffer(l.cli_layer_scale, c);
		l.cli_layer_scale_updates_gpu = alloc_gpu_buffer(l.cli_layer_scale_updates, c);

		if (clifford_has_global(l))
		{
			l.cli_w_proj_g_gpu = alloc_gpu_buffer(l.cli_w_proj_g, static_cast<size_t>(c) * l.cli_proj_in_dim);
			l.cli_w_proj_g_updates_gpu = alloc_gpu_buffer(l.cli_w_proj_g_updates, static_cast<size_t>(c) * l.cli_proj_in_dim);
			l.cli_b_proj_g_gpu = alloc_gpu_buffer(l.cli_b_proj_g, c);
			l.cli_b_proj_g_updates_gpu = alloc_gpu_buffer(l.cli_b_proj_g_updates, c);
			l.cli_w_gate_g_gpu = alloc_gpu_buffer(l.cli_w_gate_g, static_cast<size_t>(c) * 2 * c);
			l.cli_w_gate_g_updates_gpu = alloc_gpu_buffer(l.cli_w_gate_g_updates, static_cast<size_t>(c) * 2 * c);
			l.cli_b_gate_g_gpu = alloc_gpu_buffer(l.cli_b_gate_g, c);
			l.cli_b_gate_g_updates_gpu = alloc_gpu_buffer(l.cli_b_gate_g_updates, c);
		}

		allocate_clifford_runtime_buffers_gpu(l);
	}
#endif

	*cfg_and_state.output << "clifford " << c
		<< " channels, wedge_shifts=" << l.cli_num_shifts
		<< ", inner_shifts=" << inner_shift_count;
	if (l.cli_shifts_inner == nullptr)
	{
		*cfg_and_state.output << "(shared)";
	}
		*cfg_and_state.output
			<< ", unique_wedge_shifts=" << unique_wedge_shift_count
			<< ", unique_inner_shifts=" << unique_inner_shift_count
			<< ", cli_mode=" << l.cli_interaction_mode
			<< ", gffn_mode=" << l.cli_gffn_mode
			<< ", higher_mode=" << l.cli_higher_mode
			<< ", dwconv=" << dwconv_size << "x" << dwconv_size
			<< ", drop_path=" << drop_path;
	if (zero_wedge_shift_count > 0)
	{
		*cfg_and_state.output << ", zero_wedge_shift=" << zero_wedge_shift_count;
	}
	if (zero_inner_shift_count > 0)
	{
		*cfg_and_state.output << ", zero_inner_shift=" << zero_inner_shift_count;
	}
	*cfg_and_state.output
		<< std::endl;

	return l;
}

void forward_clifford_layer(Darknet::Layer & l, Darknet::NetworkState state)
{
	TAT(TATPARMS);

	const int B = l.batch;
	const int C = l.c;
	const int HW = l.h * l.w;
	const size_t total = clifford_tensor_count(l);

	// Step 1: normalize each spatial token across channels in NCHW layout.
	layernorm_nchw_forward(state.input, l.cli_ln_out, l.cli_ln_mean, l.cli_ln_var, l.cli_ln_xhat,
		l.cli_ln_gamma, l.cli_ln_beta, B, C, HW);

		// Step 2a: deterministic stream is only needed by the local branch.
		if (clifford_has_local(l))
		{
			linear_1x1_forward(l.cli_ln_out, l.cli_z_det, l.cli_w_det, l.cli_b_det, B, C, C, HW);
		}

	if (clifford_has_local(l))
	{
		// Step 2b: contextual stream comes from stacked DWConv(+BN), then SiLU and optional diff mode.
		forward_clifford_dwconv_stack(l, l.cli_ln_out, l.cli_z_ctx_pre_diff, state);
		for (size_t i = 0; i < total; ++i)
		{
			const float pre = l.cli_z_ctx_pre_diff[i];
			const float sig = logistic_activate(pre);
			l.cli_z_ctx[i] = pre * sig;
		}
		if (l.cli_ctx_mode == 1)
		{
			for (size_t i = 0; i < total; ++i)
			{
				l.cli_z_ctx[i] -= l.cli_z_det[i];
			}
		}

		// Step 3: compute shifted geometric products, concatenate them, then project back to C channels.
			clifford_rolling_forward_local(l.cli_z_det, l.cli_z_ctx, l.cli_g_raw,
				B, C, HW,
				l.cli_shifts, l.cli_num_shifts,
				clifford_inner_shifts(l), clifford_inner_num_shifts(l),
				l.cli_interaction_mode);
			linear_1x1_forward(l.cli_g_raw, l.cli_g_feat, l.cli_w_proj, l.cli_b_proj, B, l.cli_proj_in_dim, C, HW);
			if (clifford_has_higher_local(l))
			{
				clifford_vb_forward_local(l.cli_z_det, l.cli_g_feat, l.cli_vb_feat, B, C, HW, l.cli_shifts, l.cli_num_shifts);
			}
		}

	if (clifford_has_global(l))
	{
			// Optional global branch: pool once per channel, then interact directly with the normalized state.
			global_avg_pool_forward(l.cli_ln_out, l.cli_global_ctx, B, C, HW);
			clifford_rolling_forward_global(l.cli_ln_out, l.cli_global_ctx, l.cli_g_raw_g,
				B, C, HW,
				l.cli_shifts, l.cli_num_shifts,
				clifford_inner_shifts(l), clifford_inner_num_shifts(l),
				l.cli_interaction_mode);
			linear_1x1_forward(l.cli_g_raw_g, l.cli_g_feat_g, l.cli_w_proj_g, l.cli_b_proj_g, B, l.cli_proj_in_dim, C, HW);
		}

	// Step 4: gated geometric residual starts from SiLU(X_ln) and adds gated local/global features.
	for (size_t i = 0; i < total; ++i)
	{
		const float x = l.cli_ln_out[i];
		l.cli_hmix[i] = x * logistic_activate(x);
	}

		if (clifford_has_local(l))
		{
			gate_linear_forward(l.cli_ln_out, l.cli_g_feat, l.cli_gate_pre_sigmoid, l.cli_w_gate, l.cli_b_gate, B, C, HW);
			for (size_t i = 0; i < total; ++i)
			{
				l.cli_gate_alpha[i] = logistic_activate(l.cli_gate_pre_sigmoid[i]);
				l.cli_hmix[i] += l.cli_gate_alpha[i] * l.cli_g_feat[i];
			}
			if (clifford_has_higher_local(l))
			{
				for (size_t i = 0; i < total; ++i)
				{
					l.cli_hmix[i] += l.cli_vb_feat[i];
				}
			}
		}

	if (clifford_has_global(l))
	{
		gate_linear_forward(l.cli_ln_out, l.cli_g_feat_g, l.cli_gate_pre_sigmoid_g, l.cli_w_gate_g, l.cli_b_gate_g, B, C, HW);
		for (size_t i = 0; i < total; ++i)
		{
			l.cli_gate_alpha_g[i] = logistic_activate(l.cli_gate_pre_sigmoid_g[i]);
			l.cli_hmix[i] += l.cli_gate_alpha_g[i] * l.cli_g_feat_g[i];
		}
	}

	if (state.train && l.cli_drop_path > 0.0f)
	{
		const float keep_prob = 1.0f - l.cli_drop_path;
		for (int b = 0; b < B; ++b)
		{
			l.cli_drop_mask[b] = (rand_uniform(0.0f, 1.0f) < keep_prob) ? (1.0f / keep_prob) : 0.0f;
		}
	}
	else
	{
		for (int b = 0; b < B; ++b)
		{
			l.cli_drop_mask[b] = 1.0f;
		}
	}

	// Step 5: apply LayerScale and DropPath, then add the result back to the residual stream.
	copy_cpu(static_cast<int>(total), state.input, 1, l.output, 1);
	for (int b = 0; b < B; ++b)
	{
		const float mask = l.cli_drop_mask[b];
		for (int c = 0; c < C; ++c)
		{
			const float scale = l.cli_layer_scale[c] * mask;
			float *dst = l.output + (b * C + c) * HW;
			const float *src = l.cli_hmix + (b * C + c) * HW;
			for (int s = 0; s < HW; ++s)
			{
				dst[s] += src[s] * scale;
			}
		}
	}
}

void backward_clifford_layer(Darknet::Layer & l, Darknet::NetworkState state)
{
	TAT(TATPARMS);

	const int B = l.batch;
	const int C = l.c;
	const int HW = l.h * l.w;
	const size_t total = clifford_tensor_count(l);

	// Step 5 backward: residual gradient first flows through DropPath and LayerScale.
	std::vector<float> d_hmix(total, 0.0f);
	copy_cpu(static_cast<int>(total), l.delta, 1, d_hmix.data(), 1);

	for (int b = 0; b < B; ++b)
	{
		const float mask = l.cli_drop_mask[b];
		for (int c = 0; c < C; ++c)
		{
			const float gamma = l.cli_layer_scale[c];
			for (int s = 0; s < HW; ++s)
			{
				const int idx = clifford_index(b, c, s, C, HW);
				l.cli_layer_scale_updates[c] += d_hmix[idx] * mask * l.cli_hmix[idx];
				d_hmix[idx] *= mask * gamma;
			}
		}
	}

	// Step 4 backward: base SiLU(X_ln) contributes even when geometric branches are disabled.
	std::vector<float> d_xln(total, 0.0f);
	for (size_t i = 0; i < total; ++i)
	{
		const float x = l.cli_ln_out[i];
		const float sig = logistic_activate(x);
		d_xln[i] += d_hmix[i] * (sig + x * sig * (1.0f - sig));
	}

	std::vector<float> d_z_det(total, 0.0f);

	if (clifford_has_local(l))
	{
		// Local GGR gate backward: split gradients between gate logits and projected geometric features.
		std::vector<float> d_gfeat(total, 0.0f);
		std::vector<float> d_gate_pre(total, 0.0f);

		for (size_t i = 0; i < total; ++i)
		{
			const float alpha = l.cli_gate_alpha[i];
			d_gfeat[i] += d_hmix[i] * alpha;
			d_gate_pre[i] = (d_hmix[i] * l.cli_g_feat[i]) * alpha * (1.0f - alpha);
		}
		if (clifford_has_higher_local(l))
		{
			clifford_vb_backward_local(l.cli_z_det, l.cli_g_feat, d_hmix.data(),
				d_z_det.data(), d_gfeat.data(), B, C, HW, l.cli_shifts, l.cli_num_shifts);
		}

		gate_linear_backward(l.cli_ln_out, l.cli_g_feat, d_gate_pre.data(),
			l.cli_w_gate, l.cli_w_gate_updates, l.cli_b_gate_updates,
			d_xln.data(), d_gfeat.data(), B, C, HW);

		std::vector<float> d_graw(static_cast<size_t>(B) * l.cli_proj_in_dim * HW, 0.0f);
		linear_1x1_backward(l.cli_g_raw, d_gfeat.data(), l.cli_w_proj,
			l.cli_w_proj_updates, l.cli_b_proj_updates, d_graw.data(),
			B, l.cli_proj_in_dim, C, HW);

		// Step 3 backward: unroll shifted inner/wedge products back into deterministic/context streams.
		std::vector<float> d_z_ctx(total, 0.0f);
		clifford_rolling_backward_local(l.cli_z_det, l.cli_z_ctx, d_graw.data(),
			d_z_det.data(), d_z_ctx.data(), B, C, HW,
			l.cli_shifts, l.cli_num_shifts,
			clifford_inner_shifts(l), clifford_inner_num_shifts(l),
			l.cli_interaction_mode);

		if (l.cli_ctx_mode == 1)
		{
			for (size_t i = 0; i < total; ++i)
			{
				d_z_det[i] -= d_z_ctx[i];
			}
		}

		for (size_t i = 0; i < total; ++i)
		{
			const float pre = l.cli_z_ctx_pre_diff[i];
			const float sig = logistic_activate(pre);
			d_z_ctx[i] *= (sig + pre * sig * (1.0f - sig));
		}

		// Step 2 backward: send context-stream gradients through the stacked DWConv path.
		std::vector<float> d_ctx_input(total, 0.0f);
		backward_clifford_dwconv_stack(l, l.cli_ln_out, d_z_ctx.data(), d_ctx_input.data(), state);
		axpy_cpu(static_cast<int>(total), 1.0f, d_ctx_input.data(), 1, d_xln.data(), 1);
	}

	if (clifford_has_global(l))
	{
		// Global branch backward mirrors the local branch, then scatters GAP gradients across HW.
		std::vector<float> d_gfeat_g(total, 0.0f);
		std::vector<float> d_gate_pre_g(total, 0.0f);

		for (size_t i = 0; i < total; ++i)
		{
			const float alpha = l.cli_gate_alpha_g[i];
			d_gfeat_g[i] += d_hmix[i] * alpha;
			d_gate_pre_g[i] = (d_hmix[i] * l.cli_g_feat_g[i]) * alpha * (1.0f - alpha);
		}

		gate_linear_backward(l.cli_ln_out, l.cli_g_feat_g, d_gate_pre_g.data(),
			l.cli_w_gate_g, l.cli_w_gate_g_updates, l.cli_b_gate_g_updates,
			d_xln.data(), d_gfeat_g.data(), B, C, HW);

		std::vector<float> d_graw_g(static_cast<size_t>(B) * l.cli_proj_in_dim * HW, 0.0f);
		linear_1x1_backward(l.cli_g_raw_g, d_gfeat_g.data(), l.cli_w_proj_g,
			l.cli_w_proj_g_updates, l.cli_b_proj_g_updates, d_graw_g.data(),
			B, l.cli_proj_in_dim, C, HW);

		std::vector<float> d_global_ctx(static_cast<size_t>(B) * C, 0.0f);
			clifford_rolling_backward_global(l.cli_ln_out, l.cli_global_ctx, d_graw_g.data(),
				d_xln.data(), d_global_ctx.data(), B, C, HW,
				l.cli_shifts, l.cli_num_shifts,
				clifford_inner_shifts(l), clifford_inner_num_shifts(l),
				l.cli_interaction_mode);
			global_avg_pool_backward(d_global_ctx.data(), d_xln.data(), B, C, HW);
		}

		if (clifford_has_local(l))
		{
			linear_1x1_backward(l.cli_ln_out, d_z_det.data(), l.cli_w_det,
				l.cli_w_det_updates, l.cli_b_det_updates, d_xln.data(), B, C, C, HW);
		}

	// Step 1 backward: collapse all normalized-input contributions through LayerNorm.
	std::vector<float> d_input(total, 0.0f);
	layernorm_nchw_backward(d_xln.data(), l.cli_ln_xhat, l.cli_ln_var,
		l.cli_ln_gamma, state.delta ? d_input.data() : nullptr,
		l.cli_ln_gamma_updates, l.cli_ln_beta_updates, B, C, HW);

	if (state.delta)
	{
		axpy_cpu(static_cast<int>(total), 1.0f, l.delta, 1, state.delta, 1);
		axpy_cpu(static_cast<int>(total), 1.0f, d_input.data(), 1, state.delta, 1);
	}
}

void update_clifford_layer(Darknet::Layer & l, int batch, float learning_rate_init, float momentum, float decay)
{
	TAT(TATPARMS);

	const float learning_rate = learning_rate_init * l.learning_rate_scale;
	const int C = l.c;

	const int det_count = C * C;
	axpy_cpu(det_count, -decay * batch, l.cli_w_det, 1, l.cli_w_det_updates, 1);
	axpy_cpu(det_count, learning_rate / batch, l.cli_w_det_updates, 1, l.cli_w_det, 1);
	scal_cpu(det_count, momentum, l.cli_w_det_updates, 1);
	axpy_cpu(C, learning_rate / batch, l.cli_b_det_updates, 1, l.cli_b_det, 1);
	scal_cpu(C, momentum, l.cli_b_det_updates, 1);

	const int proj_count = C * l.cli_proj_in_dim;
	axpy_cpu(proj_count, -decay * batch, l.cli_w_proj, 1, l.cli_w_proj_updates, 1);
	axpy_cpu(proj_count, learning_rate / batch, l.cli_w_proj_updates, 1, l.cli_w_proj, 1);
	scal_cpu(proj_count, momentum, l.cli_w_proj_updates, 1);
	axpy_cpu(C, learning_rate / batch, l.cli_b_proj_updates, 1, l.cli_b_proj, 1);
	scal_cpu(C, momentum, l.cli_b_proj_updates, 1);

	const int gate_count = C * 2 * C;
	axpy_cpu(gate_count, -decay * batch, l.cli_w_gate, 1, l.cli_w_gate_updates, 1);
	axpy_cpu(gate_count, learning_rate / batch, l.cli_w_gate_updates, 1, l.cli_w_gate, 1);
	scal_cpu(gate_count, momentum, l.cli_w_gate_updates, 1);
	axpy_cpu(C, learning_rate / batch, l.cli_b_gate_updates, 1, l.cli_b_gate, 1);
	scal_cpu(C, momentum, l.cli_b_gate_updates, 1);

	axpy_cpu(C, learning_rate / batch, l.cli_ln_gamma_updates, 1, l.cli_ln_gamma, 1);
	scal_cpu(C, momentum, l.cli_ln_gamma_updates, 1);
	axpy_cpu(C, learning_rate / batch, l.cli_ln_beta_updates, 1, l.cli_ln_beta, 1);
	scal_cpu(C, momentum, l.cli_ln_beta_updates, 1);
	axpy_cpu(C, learning_rate / batch, l.cli_layer_scale_updates, 1, l.cli_layer_scale, 1);
	scal_cpu(C, momentum, l.cli_layer_scale_updates, 1);

	if (clifford_has_global(l))
	{
		axpy_cpu(proj_count, -decay * batch, l.cli_w_proj_g, 1, l.cli_w_proj_g_updates, 1);
		axpy_cpu(proj_count, learning_rate / batch, l.cli_w_proj_g_updates, 1, l.cli_w_proj_g, 1);
		scal_cpu(proj_count, momentum, l.cli_w_proj_g_updates, 1);
		axpy_cpu(C, learning_rate / batch, l.cli_b_proj_g_updates, 1, l.cli_b_proj_g, 1);
		scal_cpu(C, momentum, l.cli_b_proj_g_updates, 1);

		axpy_cpu(gate_count, -decay * batch, l.cli_w_gate_g, 1, l.cli_w_gate_g_updates, 1);
		axpy_cpu(gate_count, learning_rate / batch, l.cli_w_gate_g_updates, 1, l.cli_w_gate_g, 1);
		scal_cpu(gate_count, momentum, l.cli_w_gate_g_updates, 1);
		axpy_cpu(C, learning_rate / batch, l.cli_b_gate_g_updates, 1, l.cli_b_gate_g, 1);
		scal_cpu(C, momentum, l.cli_b_gate_g_updates, 1);
	}

	for (int i = 0; i < l.cli_num_dwconv; ++i)
	{
		update_convolutional_layer(l.cli_dwconv[i], batch, learning_rate_init, momentum, decay);
	}
}

void resize_clifford_layer(Darknet::Layer * l, int w, int h)
{
	TAT(TATPARMS);

	l->w = w;
	l->h = h;
	l->out_w = w;
	l->out_h = h;
	l->inputs = w * h * l->c;
	l->outputs = l->inputs;

	l->output = resize_float_buffer(l->output, static_cast<size_t>(l->batch) * l->outputs);
	l->delta = resize_float_buffer(l->delta, static_cast<size_t>(l->batch) * l->outputs);
	resize_clifford_runtime_buffers(l);

	l->workspace_size = 0;
	for (int i = 0; i < l->cli_num_dwconv; ++i)
	{
		resize_convolutional_layer(&l->cli_dwconv[i], w, h);
		ensure_clifford_dwconv_cpu_training_buffers(l->cli_dwconv[i]);
		update_clifford_dwconv_workspace_size(*l, l->cli_dwconv[i]);
	}

#ifdef DARKNET_GPU
	if (cfg_and_state.gpu_index >= 0)
	{
		cuda_free(l->output_gpu);
		cuda_free(l->delta_gpu);
		l->output_gpu = cuda_make_array(l->output, static_cast<size_t>(l->batch) * l->outputs);
		l->delta_gpu = cuda_make_array(l->delta, static_cast<size_t>(l->batch) * l->outputs);
		resize_clifford_runtime_buffers_gpu(l);
	}
#endif
}
