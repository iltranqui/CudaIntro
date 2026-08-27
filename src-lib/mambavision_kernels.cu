#include "darknet_internal.hpp"

#ifdef DARKNET_GPU

#include "mambavision_layer.hpp"
#include "connected_layer.hpp"
#include "convolutional_layer.hpp"
#include "blas.hpp"
#include "dark_cuda.hpp"

#include <cmath>

#ifdef CUDNN_HALF
#include <cuda_fp16.h>
#endif

namespace
{
	constexpr int MV_BLOCK = BLOCK;
	constexpr int MV_WARP = 32;

	__device__ size_t mv_grid_index()
	{
		return (static_cast<size_t>(blockIdx.y) * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
	}

	__device__ size_t mv_grid_stride()
	{
		return static_cast<size_t>(blockDim.x) * gridDim.x * gridDim.y;
	}

	__device__ float mv_sigmoid(float x)
	{
		return 1.0f / (1.0f + expf(-x));
	}

	__device__ float mv_softplus(float x)
	{
		return x > 20.0f ? x : log1pf(expf(x));
	}

	__device__ __forceinline__ float mv_warp_sum(float v)
	{
		for (int offset = MV_WARP / 2; offset > 0; offset >>= 1)
		{
			v += __shfl_xor_sync(0xFFFFFFFF, v, offset);
		}
		return v;
	}

	__device__ size_t bdt_idx(int b, int d, int t, int D, int T)
	{
		return (static_cast<size_t>(b) * D + d) * T + t;
	}

	__device__ size_t state_idx(int b, int d, int t, int s, int D, int T, int S)
	{
		return (((static_cast<size_t>(b) * D + d) * T + t) * S + s);
	}

#ifdef CUDNN_HALF
	__device__ __forceinline__ float mv_h2f(__half v)
	{
		return __half2float(v);
	}

	__device__ __forceinline__ __half mv_f2h(float v)
	{
		return __float2half_rn(v);
	}

	static bool mambavision_use_cudnn_half(const Darknet::NetworkState &state)
	{
		const int iteration_num = get_current_iteration(state.net);
		const bool training_ready = !state.train || ((iteration_num > 3 * state.net.burn_in) && state.net.loss_scale != 1.0f);
		return state.net.cudnn_half && !state.net.cudnn_bf16 && training_ready;
	}
#endif

	__global__ void spatial_to_tokens_kernel(const float *src, float *dst, int B, int C, int T)
	{
		const size_t total = static_cast<size_t>(B) * C * T;
		for (size_t i = mv_grid_index(); i < total; i += mv_grid_stride())
		{
			const int t = i % T;
			const int c = (i / T) % C;
			const int b = i / (T * C);
			dst[(static_cast<size_t>(b) * T + t) * C + c] = src[(static_cast<size_t>(b) * C + c) * T + t];
		}
	}

	__global__ void tokens_to_spatial_kernel(const float *src, float *dst, int B, int C, int T)
	{
		const size_t total = static_cast<size_t>(B) * C * T;
		for (size_t i = mv_grid_index(); i < total; i += mv_grid_stride())
		{
			const int t = i % T;
			const int c = (i / T) % C;
			const int b = i / (T * C);
			dst[(static_cast<size_t>(b) * C + c) * T + t] = src[(static_cast<size_t>(b) * T + t) * C + c];
		}
	}

	__global__ void tokens_to_spatial_add_kernel(const float *src, float *dst, int B, int C, int T)
	{
		const size_t total = static_cast<size_t>(B) * C * T;
		for (size_t i = mv_grid_index(); i < total; i += mv_grid_stride())
		{
			const int t = i % T;
			const int c = (i / T) % C;
			const int b = i / (T * C);
			atomicAdd(dst + (static_cast<size_t>(b) * C + c) * T + t, src[(static_cast<size_t>(b) * T + t) * C + c]);
		}
	}

	__global__ void layernorm_forward_kernel(const float *x, float *out, float *mean, float *var, float *xhat,
		const float *gamma, const float *beta, int M, int C)
	{
		const size_t row = mv_grid_index() / MV_WARP;
		if (row >= static_cast<size_t>(M)) return;
		const int lane = threadIdx.x & (MV_WARP - 1);
		const float *xi = x + row * C;
		float *oi = out + row * C;
		float *xh = xhat + row * C;

		float local_sum = 0.0f;
		for (int j = lane; j < C; j += MV_WARP)
		{
			local_sum += xi[j];
		}
		const float m = mv_warp_sum(local_sum) / C;
		if (lane == 0) mean[row] = m;

		float local_var = 0.0f;
		for (int j = lane; j < C; j += MV_WARP)
		{
			const float d = xi[j] - m;
			local_var += d * d;
		}
		const float v = mv_warp_sum(local_var) / C;
		if (lane == 0) var[row] = v;

		const float inv_std = rsqrtf(v + 1e-5f);
		for (int j = lane; j < C; j += MV_WARP)
		{
			xh[j] = (xi[j] - m) * inv_std;
			oi[j] = xh[j] * gamma[j] + beta[j];
		}
	}

	__global__ void layernorm_backward_kernel(const float *dout, const float *xhat, const float *var,
		const float *gamma, float *dx, float *dgamma, float *dbeta, int M, int C)
	{
		const size_t row = mv_grid_index() / MV_WARP;
		if (row >= static_cast<size_t>(M)) return;
		const int lane = threadIdx.x & (MV_WARP - 1);
		const float *doi = dout + row * C;
		const float *xhi = xhat + row * C;
		float *dxi = dx + row * C;
		const float inv_std = rsqrtf(var[row] + 1e-5f);

		float local_sum_dxhat = 0.0f;
		float local_dot_dxhat_xhat = 0.0f;
		for (int j = lane; j < C; j += MV_WARP)
		{
			atomicAdd(dgamma + j, doi[j] * xhi[j]);
			atomicAdd(dbeta + j, doi[j]);
			const float dxhat = doi[j] * gamma[j];
			local_sum_dxhat += dxhat;
			local_dot_dxhat_xhat += dxhat * xhi[j];
		}
		const float sum_dxhat = mv_warp_sum(local_sum_dxhat);
		const float dot_dxhat_xhat = mv_warp_sum(local_dot_dxhat_xhat);

		for (int j = lane; j < C; j += MV_WARP)
		{
			const float dxhat = doi[j] * gamma[j];
			dxi[j] = inv_std * (dxhat - (sum_dxhat + xhi[j] * dot_dxhat_xhat) / C);
		}
	}

	__global__ void branch_to_bdt_kernel(const float *tokens, float *bdt, int branch_offset, int B, int T, int N, int D)
	{
		const size_t total = static_cast<size_t>(B) * D * T;
		for (size_t i = mv_grid_index(); i < total; i += mv_grid_stride())
		{
			const int t = i % T;
			const int d = (i / T) % D;
			const int b = i / (T * D);
			bdt[i] = tokens[(static_cast<size_t>(b) * T + t) * N + branch_offset + d];
		}
	}

	__global__ void bdt_to_tokens_kernel(const float *bdt, float *tokens, int B, int T, int D)
	{
		const size_t total = static_cast<size_t>(B) * D * T;
		for (size_t i = mv_grid_index(); i < total; i += mv_grid_stride())
		{
			const int t = i % T;
			const int d = (i / T) % D;
			const int b = i / (T * D);
			tokens[(static_cast<size_t>(b) * T + t) * D + d] = bdt[i];
		}
	}

	__global__ void bdt_add_to_branch_kernel(const float *bdt, float *tokens, int branch_offset, int B, int T, int N, int D)
	{
		const size_t total = static_cast<size_t>(B) * D * T;
		for (size_t i = mv_grid_index(); i < total; i += mv_grid_stride())
		{
			const int t = i % T;
			const int d = (i / T) % D;
			const int b = i / (T * D);
			atomicAdd(tokens + (static_cast<size_t>(b) * T + t) * N + branch_offset + d, bdt[i]);
		}
	}

	__global__ void slice_dt_raw_kernel(const float *xproj, float *dt_raw, int M, int P, int R)
	{
		const size_t total = static_cast<size_t>(M) * R;
		for (size_t i = mv_grid_index(); i < total; i += mv_grid_stride())
		{
			const int r = i % R;
			const int m = i / R;
			dt_raw[i] = xproj[static_cast<size_t>(m) * P + r];
		}
	}

	__global__ void dt_softplus_to_bdt_kernel(const float *dt_pre_tokens, float *dt_pre_bdt, float *dt_bdt, int B, int T, int D)
	{
		const size_t total = static_cast<size_t>(B) * D * T;
		for (size_t i = mv_grid_index(); i < total; i += mv_grid_stride())
		{
			const int t = i % T;
			const int d = (i / T) % D;
			const int b = i / (T * D);
			const float v = dt_pre_tokens[(static_cast<size_t>(b) * T + t) * D + d];
			dt_pre_bdt[i] = v;
			dt_bdt[i] = mv_softplus(v);
		}
	}

#ifdef CUDNN_HALF
	__global__ void dt_softplus_to_bdt_half_kernel(const float *dt_pre_tokens, __half *dt_pre_bdt, __half *dt_bdt, int B, int T, int D)
	{
		const size_t total = static_cast<size_t>(B) * D * T;
		for (size_t i = mv_grid_index(); i < total; i += mv_grid_stride())
		{
			const int t = i % T;
			const int d = (i / T) % D;
			const int b = i / (T * D);
			const float v = dt_pre_tokens[(static_cast<size_t>(b) * T + t) * D + d];
			dt_pre_bdt[i] = mv_f2h(v);
			dt_bdt[i] = mv_f2h(mv_softplus(v));
		}
	}
#endif

	__global__ void scan_forward_kernel(float *scan_state, float *scan_out, const float *xconv, const float *xproj,
		const float *dt, const float *A_log, const float *D_param, int B, int T, int D, int R, int S)
	{
		const size_t bd = mv_grid_index();
		if (bd >= B * D) return;
		const int d = bd % D;
		const int b = bd / D;
		for (int s = 0; s < S; ++s)
		{
			float hprev = 0.0f;
			for (int t = 0; t < T; ++t)
			{
				const int P = R + 2 * S;
				const float u = xconv[bdt_idx(b, d, t, D, T)];
				const float dtv = dt[bdt_idx(b, d, t, D, T)];
				const float *row = xproj + (static_cast<size_t>(b) * T + t) * P;
				const float Bp = row[R + s];
				const float A = -expf(A_log[d * S + s]);
				const float abar = expf(fminf(20.0f, fmaxf(-60.0f, dtv * A)));
				const float h = abar * hprev + dtv * Bp * u;
				scan_state[state_idx(b, d, t, s, D, T, S)] = h;
				hprev = h;
			}
		}
		for (int t = 0; t < T; ++t)
		{
			const int P = R + 2 * S;
			const float *row = xproj + (static_cast<size_t>(b) * T + t) * P;
			const float *Cp = row + R + S;
			float y = D_param[d] * xconv[bdt_idx(b, d, t, D, T)];
			for (int s = 0; s < S; ++s)
			{
				y += scan_state[state_idx(b, d, t, s, D, T, S)] * Cp[s];
			}
			scan_out[bdt_idx(b, d, t, D, T)] = y;
		}
	}

#ifdef CUDNN_HALF
	__global__ void scan_forward_half_kernel(__half *scan_state, __half *scan_out, const float *xconv, const float *xproj,
		const __half *dt, const float *A_log, const float *D_param, int B, int T, int D, int R, int S)
	{
		const size_t bd = mv_grid_index();
		if (bd >= static_cast<size_t>(B) * D) return;
		const int d = bd % D;
		const int b = bd / D;
		const int P = R + 2 * S;
		for (int s = 0; s < S; ++s)
		{
			float hprev = 0.0f;
			const float A = -expf(A_log[d * S + s]);
			for (int t = 0; t < T; ++t)
			{
				const float u = xconv[bdt_idx(b, d, t, D, T)];
				const float dtv = mv_h2f(dt[bdt_idx(b, d, t, D, T)]);
				const float *row = xproj + (static_cast<size_t>(b) * T + t) * P;
				const float Bp = row[R + s];
				const float abar = expf(fminf(20.0f, fmaxf(-60.0f, dtv * A)));
				const float h = abar * hprev + dtv * Bp * u;
				scan_state[state_idx(b, d, t, s, D, T, S)] = mv_f2h(h);
				hprev = h;
			}
		}
		for (int t = 0; t < T; ++t)
		{
			const float *row = xproj + (static_cast<size_t>(b) * T + t) * P;
			const float *Cp = row + R + S;
			float y = D_param[d] * xconv[bdt_idx(b, d, t, D, T)];
			for (int s = 0; s < S; ++s)
			{
				y += mv_h2f(scan_state[state_idx(b, d, t, s, D, T, S)]) * Cp[s];
			}
			scan_out[bdt_idx(b, d, t, D, T)] = mv_f2h(y);
		}
	}
#endif

	__global__ void scan_backward_kernel(const float *dscan, float *dxconv, float *dxproj, float *ddt,
		const float *scan_state, const float *xconv, const float *xproj, const float *dt,
		const float *A_log, float *A_log_updates, const float *D_param, float *D_updates,
		int B, int T, int D, int R, int S)
	{
		const size_t bd = mv_grid_index();
		if (bd >= B * D) return;
		const int d = bd % D;
		const int b = bd / D;
		const int P = R + 2 * S;
		for (int s = 0; s < S; ++s)
		{
			float dh_next = 0.0f;
			for (int t = T - 1; t >= 0; --t)
			{
				const float u = xconv[bdt_idx(b, d, t, D, T)];
				const float dtv = dt[bdt_idx(b, d, t, D, T)];
				const float dy = dscan[bdt_idx(b, d, t, D, T)];
				const float *row = xproj + (static_cast<size_t>(b) * T + t) * P;
				float *drow = dxproj + (static_cast<size_t>(b) * T + t) * P;
				const float Bp = row[R + s];
				const float Cp = row[R + S + s];
				const float h = scan_state[state_idx(b, d, t, s, D, T, S)];
				const float hprev = t == 0 ? 0.0f : scan_state[state_idx(b, d, t - 1, s, D, T, S)];
				const float A = -expf(A_log[d * S + s]);
				const float abar = expf(fminf(20.0f, fmaxf(-60.0f, dtv * A)));
				atomicAdd(drow + R + S + s, dy * h);
				const float dh = dy * Cp + dh_next;
				const float d_abar = dh * hprev;
				const float d_exp_arg = d_abar * abar;
				atomicAdd(ddt + bdt_idx(b, d, t, D, T), dh * Bp * u + d_exp_arg * A);
				atomicAdd(drow + R + s, dh * dtv * u);
				atomicAdd(dxconv + bdt_idx(b, d, t, D, T), dh * dtv * Bp);
				atomicAdd(A_log_updates + d * S + s, d_exp_arg * dtv * A);
				dh_next = dh * abar;
			}
		}
		for (int t = 0; t < T; ++t)
		{
			const float dy = dscan[bdt_idx(b, d, t, D, T)];
			const float u = xconv[bdt_idx(b, d, t, D, T)];
			atomicAdd(D_updates + d, dy * u);
			atomicAdd(dxconv + bdt_idx(b, d, t, D, T), dy * D_param[d]);
		}
	}

#ifdef CUDNN_HALF
	__global__ void scan_backward_half_kernel(const float *dscan, float *dxconv, float *dxproj, float *ddt,
		const __half *scan_state, const float *xconv, const float *xproj, const __half *dt,
		const float *A_log, float *A_log_updates, const float *D_param, float *D_updates,
		int B, int T, int D, int R, int S)
	{
		const size_t bd = mv_grid_index();
		if (bd >= static_cast<size_t>(B) * D) return;
		const int d = bd % D;
		const int b = bd / D;
		const int P = R + 2 * S;
		for (int s = 0; s < S; ++s)
		{
			float dh_next = 0.0f;
			const float A = -expf(A_log[d * S + s]);
			for (int t = T - 1; t >= 0; --t)
			{
				const float u = xconv[bdt_idx(b, d, t, D, T)];
				const float dtv = mv_h2f(dt[bdt_idx(b, d, t, D, T)]);
				const float dy = dscan[bdt_idx(b, d, t, D, T)];
				const float *row = xproj + (static_cast<size_t>(b) * T + t) * P;
				float *drow = dxproj + (static_cast<size_t>(b) * T + t) * P;
				const float Bp = row[R + s];
				const float Cp = row[R + S + s];
				const float h = mv_h2f(scan_state[state_idx(b, d, t, s, D, T, S)]);
				const float hprev = t == 0 ? 0.0f : mv_h2f(scan_state[state_idx(b, d, t - 1, s, D, T, S)]);
				const float abar = expf(fminf(20.0f, fmaxf(-60.0f, dtv * A)));
				atomicAdd(drow + R + S + s, dy * h);
				const float dh = dy * Cp + dh_next;
				const float d_abar = dh * hprev;
				const float d_exp_arg = d_abar * abar;
				atomicAdd(ddt + bdt_idx(b, d, t, D, T), dh * Bp * u + d_exp_arg * A);
				atomicAdd(drow + R + s, dh * dtv * u);
				atomicAdd(dxconv + bdt_idx(b, d, t, D, T), dh * dtv * Bp);
				atomicAdd(A_log_updates + d * S + s, d_exp_arg * dtv * A);
				dh_next = dh * abar;
			}
		}
		for (int t = 0; t < T; ++t)
		{
			const float dy = dscan[bdt_idx(b, d, t, D, T)];
			const float u = xconv[bdt_idx(b, d, t, D, T)];
			atomicAdd(D_updates + d, dy * u);
			atomicAdd(dxconv + bdt_idx(b, d, t, D, T), dy * D_param[d]);
		}
	}
#endif

	__global__ void cat_scan_z_kernel(const float *scan, const float *z, float *cat, int B, int T, int N, int D)
	{
		const size_t total = static_cast<size_t>(B) * D * T;
		for (size_t i = mv_grid_index(); i < total; i += mv_grid_stride())
		{
			const int t = i % T;
			const int d = (i / T) % D;
			const int b = i / (T * D);
			float *row = cat + (static_cast<size_t>(b) * T + t) * N;
			row[d] = scan[i];
			row[D + d] = z[i];
		}
	}

#ifdef CUDNN_HALF
	__global__ void cat_scan_z_half_kernel(const __half *scan, const float *z, float *cat, int B, int T, int N, int D)
	{
		const size_t total = static_cast<size_t>(B) * D * T;
		for (size_t i = mv_grid_index(); i < total; i += mv_grid_stride())
		{
			const int t = i % T;
			const int d = (i / T) % D;
			const int b = i / (T * D);
			float *row = cat + (static_cast<size_t>(b) * T + t) * N;
			row[d] = mv_h2f(scan[i]);
			row[D + d] = z[i];
		}
	}
#endif

	__global__ void split_cat_grad_kernel(const float *dcat, float *dscan, float *dz, int B, int T, int N, int D)
	{
		const size_t total = static_cast<size_t>(B) * D * T;
		for (size_t i = mv_grid_index(); i < total; i += mv_grid_stride())
		{
			const int t = i % T;
			const int d = (i / T) % D;
			const int b = i / (T * D);
			const float *row = dcat + (static_cast<size_t>(b) * T + t) * N;
			dscan[i] = row[d];
			dz[i] = row[D + d];
		}
	}

	__global__ void add_kernel(const float *a, const float *b, float *out, size_t total)
	{
		for (size_t i = mv_grid_index(); i < total; i += mv_grid_stride())
		{
			out[i] = a[i] + b[i];
		}
	}

	__global__ void add_inplace_kernel(float *dst, const float *src, size_t total)
	{
		for (size_t i = mv_grid_index(); i < total; i += mv_grid_stride())
		{
			dst[i] += src[i];
		}
	}

	__global__ void ddt_to_dt_delta_kernel(const float *ddt, const float *dt_pre, float *dt_delta_tokens, int B, int T, int D)
	{
		const size_t total = static_cast<size_t>(B) * D * T;
		for (size_t i = mv_grid_index(); i < total; i += mv_grid_stride())
		{
			const int t = i % T;
			const int d = (i / T) % D;
			const int b = i / (T * D);
			dt_delta_tokens[(static_cast<size_t>(b) * T + t) * D + d] = ddt[i] * mv_sigmoid(dt_pre[i]);
		}
	}

#ifdef CUDNN_HALF
	__global__ void ddt_to_dt_delta_half_kernel(const float *ddt, const __half *dt_pre, float *dt_delta_tokens, int B, int T, int D)
	{
		const size_t total = static_cast<size_t>(B) * D * T;
		for (size_t i = mv_grid_index(); i < total; i += mv_grid_stride())
		{
			const int t = i % T;
			const int d = (i / T) % D;
			const int b = i / (T * D);
			dt_delta_tokens[(static_cast<size_t>(b) * T + t) * D + d] = ddt[i] * mv_sigmoid(mv_h2f(dt_pre[i]));
		}
	}
#endif

	__global__ void add_dt_raw_grad_kernel(const float *dt_raw_delta, float *dxproj, int M, int P, int R)
	{
		const size_t total = static_cast<size_t>(M) * R;
		for (size_t i = mv_grid_index(); i < total; i += mv_grid_stride())
		{
			const int r = i % R;
			const int m = i / R;
			dxproj[static_cast<size_t>(m) * P + r] += dt_raw_delta[i];
		}
	}

}

void mambavision_forward_gpu_impl(Darknet::Layer & l, Darknet::NetworkState state)
{
	const int B = l.batch;
	const int C = l.c;
	const int N = l.n;
	const int T = l.h * l.w;
	const int M = B * T;
	const int D = N / 2;
	const int R = l.mv_dt_rank;
	const int S = l.mv_d_state;
	const int P = R + 2 * S;

	spatial_to_tokens_kernel<<<cuda_gridsize(static_cast<size_t>(B) * C * T), MV_BLOCK, 0, get_cuda_stream()>>>(state.input, l.mv_tokens_gpu, B, C, T);
	layernorm_forward_kernel<<<cuda_gridsize(static_cast<size_t>(M) * MV_WARP), MV_BLOCK, 0, get_cuda_stream()>>>(l.mv_tokens_gpu, l.mv_ln1_out_gpu, l.mv_ln1_mean_gpu, l.mv_ln1_var_gpu,
		l.mv_ln1_xhat_gpu, l.mv_ln1_gamma_gpu, l.mv_ln1_beta_gpu, M, C);

	Darknet::NetworkState s = state;
	s.input = l.mv_ln1_out_gpu;
	forward_connected_layer_gpu(*l.mv_in_proj_layer, s);

	branch_to_bdt_kernel<<<cuda_gridsize(static_cast<size_t>(B) * D * T), MV_BLOCK, 0, get_cuda_stream()>>>(l.mv_in_proj_layer->output_gpu, l.mv_tmp_bdt_gpu, 0, B, T, N, D);
	s.input = l.mv_tmp_bdt_gpu;
	forward_convolutional_layer_gpu(*l.mv_conv_x_layer, s);

	branch_to_bdt_kernel<<<cuda_gridsize(static_cast<size_t>(B) * D * T), MV_BLOCK, 0, get_cuda_stream()>>>(l.mv_in_proj_layer->output_gpu, l.mv_tmp_bdt2_gpu, D, B, T, N, D);
	s.input = l.mv_tmp_bdt2_gpu;
	forward_convolutional_layer_gpu(*l.mv_conv_z_layer, s);

	bdt_to_tokens_kernel<<<cuda_gridsize(static_cast<size_t>(B) * D * T), MV_BLOCK, 0, get_cuda_stream()>>>(l.mv_conv_x_layer->output_gpu, l.mv_tmp_token_n_gpu, B, T, D);
	s.input = l.mv_tmp_token_n_gpu;
	forward_connected_layer_gpu(*l.mv_x_proj_layer, s);

	slice_dt_raw_kernel<<<cuda_gridsize(static_cast<size_t>(M) * R), MV_BLOCK, 0, get_cuda_stream()>>>(l.mv_x_proj_layer->output_gpu, l.mv_tmp_token_p_gpu, M, P, R);
	s.input = l.mv_tmp_token_p_gpu;
	forward_connected_layer_gpu(*l.mv_dt_proj_layer, s);

#ifdef CUDNN_HALF
	if (mambavision_use_cudnn_half(state))
	{
		__half *dt_pre16 = reinterpret_cast<__half *>(l.mv_dt_pre_gpu);
		__half *dt16 = reinterpret_cast<__half *>(l.mv_dt_gpu);
		__half *scan_state16 = reinterpret_cast<__half *>(l.mv_scan_state_gpu);
		__half *scan_out16 = reinterpret_cast<__half *>(l.mv_scan_out_gpu);

		dt_softplus_to_bdt_half_kernel<<<cuda_gridsize(static_cast<size_t>(B) * D * T), MV_BLOCK, 0, get_cuda_stream()>>>(
			l.mv_dt_proj_layer->output_gpu, dt_pre16, dt16, B, T, D);
		scan_forward_half_kernel<<<cuda_gridsize(B * D), MV_BLOCK, 0, get_cuda_stream()>>>(
			scan_state16, scan_out16, l.mv_conv_x_layer->output_gpu,
			l.mv_x_proj_layer->output_gpu, dt16, l.mv_A_log_gpu, l.mv_D_gpu, B, T, D, R, S);
		cat_scan_z_half_kernel<<<cuda_gridsize(static_cast<size_t>(B) * D * T), MV_BLOCK, 0, get_cuda_stream()>>>(
			scan_out16, l.mv_conv_z_layer->output_gpu, l.mv_mixer_cat_gpu, B, T, N, D);
	}
	else
#endif
	{
		dt_softplus_to_bdt_kernel<<<cuda_gridsize(static_cast<size_t>(B) * D * T), MV_BLOCK, 0, get_cuda_stream()>>>(
			l.mv_dt_proj_layer->output_gpu, l.mv_dt_pre_gpu, l.mv_dt_gpu, B, T, D);
		scan_forward_kernel<<<cuda_gridsize(B * D), MV_BLOCK, 0, get_cuda_stream()>>>(
			l.mv_scan_state_gpu, l.mv_scan_out_gpu, l.mv_conv_x_layer->output_gpu,
			l.mv_x_proj_layer->output_gpu, l.mv_dt_gpu, l.mv_A_log_gpu, l.mv_D_gpu, B, T, D, R, S);
		cat_scan_z_kernel<<<cuda_gridsize(static_cast<size_t>(B) * D * T), MV_BLOCK, 0, get_cuda_stream()>>>(
			l.mv_scan_out_gpu, l.mv_conv_z_layer->output_gpu, l.mv_mixer_cat_gpu, B, T, N, D);
	}
	s.input = l.mv_mixer_cat_gpu;
	forward_connected_layer_gpu(*l.mv_out_proj_layer, s);

	if (C == N)
	{
		add_kernel<<<cuda_gridsize(static_cast<size_t>(M) * N), MV_BLOCK, 0, get_cuda_stream()>>>(l.mv_out_proj_layer->output_gpu, l.mv_tokens_gpu, l.mv_pre_res2_gpu, M * N);
	}
	else
	{
		s.input = l.mv_tokens_gpu;
		forward_connected_layer_gpu(*l.mv_res_proj_layer, s);
		add_kernel<<<cuda_gridsize(static_cast<size_t>(M) * N), MV_BLOCK, 0, get_cuda_stream()>>>(l.mv_out_proj_layer->output_gpu, l.mv_res_proj_layer->output_gpu, l.mv_pre_res2_gpu, M * N);
	}

	layernorm_forward_kernel<<<cuda_gridsize(static_cast<size_t>(M) * MV_WARP), MV_BLOCK, 0, get_cuda_stream()>>>(l.mv_pre_res2_gpu, l.mv_ln2_out_gpu, l.mv_ln2_mean_gpu, l.mv_ln2_var_gpu,
		l.mv_ln2_xhat_gpu, l.mv_ln2_gamma_gpu, l.mv_ln2_beta_gpu, M, N);
	s.input = l.mv_ln2_out_gpu;
	forward_connected_layer_gpu(*l.mv_ffn1_layer, s);
	s.input = l.mv_ffn1_layer->output_gpu;
	forward_connected_layer_gpu(*l.mv_ffn2_layer, s);
	add_kernel<<<cuda_gridsize(static_cast<size_t>(M) * N), MV_BLOCK, 0, get_cuda_stream()>>>(l.mv_ffn2_layer->output_gpu, l.mv_pre_res2_gpu, l.mv_tmp_token_n_gpu, M * N);
	tokens_to_spatial_kernel<<<cuda_gridsize(static_cast<size_t>(B) * N * T), MV_BLOCK, 0, get_cuda_stream()>>>(l.mv_tmp_token_n_gpu, l.output_gpu, B, N, T);
	CHECK_CUDA(cudaPeekAtLastError());
}

void mambavision_backward_gpu_impl(Darknet::Layer & l, Darknet::NetworkState state)
{
	const int B = l.batch;
	const int C = l.c;
	const int N = l.n;
	const int T = l.h * l.w;
	const int M = B * T;
	const int D = N / 2;
	const int R = l.mv_dt_rank;
	const int S = l.mv_d_state;
	const int P = R + 2 * S;
	const int ffn_hidden = N * l.mv_ffn_ratio;

	float *dout_tokens = l.mv_tmp_token_n_gpu;
	float *d_pre_res2 = l.mv_tmp_ffn_gpu; // sized M*ffn_hidden, always >= M*N
	float *d_ln2_out = l.mv_ffn2_layer->delta_gpu;
	float *d_tokens = l.mv_tmp_token_c_gpu;
	float *d_cat = l.mv_tmp_token_n_gpu;
	float *d_scan = l.mv_tmp_bdt_gpu;
	float *d_xconv = l.mv_tmp_bdt2_gpu;
	float *d_zconv = l.mv_conv_z_layer->delta_gpu;
	float *d_xproj = l.mv_tmp_token_p_gpu;
	float *ddt = l.mv_conv_x_layer->delta_gpu;

	spatial_to_tokens_kernel<<<cuda_gridsize(static_cast<size_t>(B) * N * T), MV_BLOCK, 0, get_cuda_stream()>>>(l.delta_gpu, dout_tokens, B, N, T);
	simple_copy_ongpu(M * N, dout_tokens, l.mv_ffn2_layer->delta_gpu);
	fill_ongpu(M * ffn_hidden, 0.0f, l.mv_ffn1_layer->delta_gpu, 1);
	Darknet::NetworkState s = state;
	s.input = l.mv_ffn1_layer->output_gpu;
	s.delta = l.mv_ffn1_layer->delta_gpu;
	backward_connected_layer_gpu(*l.mv_ffn2_layer, s);

	fill_ongpu(M * N, 0.0f, d_ln2_out, 1);
	s.input = l.mv_ln2_out_gpu;
	s.delta = d_ln2_out;
	backward_connected_layer_gpu(*l.mv_ffn1_layer, s);

	simple_copy_ongpu(M * N, dout_tokens, d_pre_res2);
	layernorm_backward_kernel<<<cuda_gridsize(static_cast<size_t>(M) * MV_WARP), MV_BLOCK, 0, get_cuda_stream()>>>(d_ln2_out, l.mv_ln2_xhat_gpu, l.mv_ln2_var_gpu, l.mv_ln2_gamma_gpu,
		l.mv_pre_res2_gpu, l.mv_ln2_gamma_updates_gpu, l.mv_ln2_beta_updates_gpu, M, N);
	add_inplace_kernel<<<cuda_gridsize(static_cast<size_t>(M) * N), MV_BLOCK, 0, get_cuda_stream()>>>(d_pre_res2, l.mv_pre_res2_gpu, M * N);

	fill_ongpu(M * C, 0.0f, d_tokens, 1);
	if (C == N)
	{
		add_inplace_kernel<<<cuda_gridsize(static_cast<size_t>(M) * C), MV_BLOCK, 0, get_cuda_stream()>>>(d_tokens, d_pre_res2, M * C);
	}
	else
	{
		simple_copy_ongpu(M * N, d_pre_res2, l.mv_res_proj_layer->delta_gpu);
		s.input = l.mv_tokens_gpu;
		s.delta = d_tokens;
		backward_connected_layer_gpu(*l.mv_res_proj_layer, s);
	}

	simple_copy_ongpu(M * N, d_pre_res2, l.mv_out_proj_layer->delta_gpu);
	fill_ongpu(M * N, 0.0f, d_cat, 1);
	s.input = l.mv_mixer_cat_gpu;
	s.delta = d_cat;
	backward_connected_layer_gpu(*l.mv_out_proj_layer, s);

	split_cat_grad_kernel<<<cuda_gridsize(static_cast<size_t>(B) * D * T), MV_BLOCK, 0, get_cuda_stream()>>>(d_cat, d_scan, d_zconv, B, T, N, D);
	fill_ongpu(B * D * T, 0.0f, d_xconv, 1);
	fill_ongpu(M * P, 0.0f, d_xproj, 1);
	fill_ongpu(B * D * T, 0.0f, ddt, 1);
#ifdef CUDNN_HALF
	if (mambavision_use_cudnn_half(state))
	{
		const __half *dt_pre16 = reinterpret_cast<const __half *>(l.mv_dt_pre_gpu);
		const __half *dt16 = reinterpret_cast<const __half *>(l.mv_dt_gpu);
		const __half *scan_state16 = reinterpret_cast<const __half *>(l.mv_scan_state_gpu);

		scan_backward_half_kernel<<<cuda_gridsize(B * D), MV_BLOCK, 0, get_cuda_stream()>>>(
			d_scan, d_xconv, d_xproj, ddt, scan_state16,
			l.mv_conv_x_layer->output_gpu, l.mv_x_proj_layer->output_gpu, dt16, l.mv_A_log_gpu, l.mv_A_log_updates_gpu,
			l.mv_D_gpu, l.mv_D_updates_gpu, B, T, D, R, S);
		ddt_to_dt_delta_half_kernel<<<cuda_gridsize(static_cast<size_t>(B) * D * T), MV_BLOCK, 0, get_cuda_stream()>>>(
			ddt, dt_pre16, l.mv_dt_proj_layer->delta_gpu, B, T, D);
	}
	else
#endif
	{
		scan_backward_kernel<<<cuda_gridsize(B * D), MV_BLOCK, 0, get_cuda_stream()>>>(
			d_scan, d_xconv, d_xproj, ddt, l.mv_scan_state_gpu,
			l.mv_conv_x_layer->output_gpu, l.mv_x_proj_layer->output_gpu, l.mv_dt_gpu, l.mv_A_log_gpu, l.mv_A_log_updates_gpu,
			l.mv_D_gpu, l.mv_D_updates_gpu, B, T, D, R, S);
		ddt_to_dt_delta_kernel<<<cuda_gridsize(static_cast<size_t>(B) * D * T), MV_BLOCK, 0, get_cuda_stream()>>>(
			ddt, l.mv_dt_pre_gpu, l.mv_dt_proj_layer->delta_gpu, B, T, D);
	}
	slice_dt_raw_kernel<<<cuda_gridsize(static_cast<size_t>(M) * R), MV_BLOCK, 0, get_cuda_stream()>>>(l.mv_x_proj_layer->output_gpu, l.mv_tmp_token_n_gpu, M, P, R);
	fill_ongpu(M * R, 0.0f, l.mv_ln2_out_gpu, 1);
	s.input = l.mv_tmp_token_n_gpu;
	s.delta = l.mv_ln2_out_gpu;
	backward_connected_layer_gpu(*l.mv_dt_proj_layer, s);
	add_dt_raw_grad_kernel<<<cuda_gridsize(static_cast<size_t>(M) * R), MV_BLOCK, 0, get_cuda_stream()>>>(l.mv_ln2_out_gpu, d_xproj, M, P, R);

	simple_copy_ongpu(M * P, d_xproj, l.mv_x_proj_layer->delta_gpu);
	bdt_to_tokens_kernel<<<cuda_gridsize(static_cast<size_t>(B) * D * T), MV_BLOCK, 0, get_cuda_stream()>>>(l.mv_conv_x_layer->output_gpu, l.mv_tmp_token_n_gpu, B, T, D);
	fill_ongpu(M * D, 0.0f, l.mv_ln2_out_gpu, 1);
	s.input = l.mv_tmp_token_n_gpu;
	s.delta = l.mv_ln2_out_gpu;
	backward_connected_layer_gpu(*l.mv_x_proj_layer, s);
	branch_to_bdt_kernel<<<cuda_gridsize(static_cast<size_t>(B) * D * T), MV_BLOCK, 0, get_cuda_stream()>>>(l.mv_ln2_out_gpu, ddt, 0, B, T, D, D);
	add_inplace_kernel<<<cuda_gridsize(static_cast<size_t>(B) * D * T), MV_BLOCK, 0, get_cuda_stream()>>>(d_xconv, ddt, B * D * T);

	fill_ongpu(M * N, 0.0f, l.mv_in_proj_layer->delta_gpu, 1);
	simple_copy_ongpu(B * D * T, d_zconv, l.mv_conv_z_layer->delta_gpu);
	branch_to_bdt_kernel<<<cuda_gridsize(static_cast<size_t>(B) * D * T), MV_BLOCK, 0, get_cuda_stream()>>>(l.mv_in_proj_layer->output_gpu, l.mv_tmp_bdt_gpu, D, B, T, N, D);
	fill_ongpu(B * D * T, 0.0f, l.mv_tmp_bdt2_gpu, 1);
	s.input = l.mv_tmp_bdt_gpu;
	s.delta = l.mv_tmp_bdt2_gpu;
	backward_convolutional_layer_gpu(*l.mv_conv_z_layer, s);
	bdt_add_to_branch_kernel<<<cuda_gridsize(static_cast<size_t>(B) * D * T), MV_BLOCK, 0, get_cuda_stream()>>>(l.mv_tmp_bdt2_gpu, l.mv_in_proj_layer->delta_gpu, D, B, T, N, D);

	simple_copy_ongpu(B * D * T, d_xconv, l.mv_conv_x_layer->delta_gpu);
	branch_to_bdt_kernel<<<cuda_gridsize(static_cast<size_t>(B) * D * T), MV_BLOCK, 0, get_cuda_stream()>>>(l.mv_in_proj_layer->output_gpu, l.mv_tmp_bdt_gpu, 0, B, T, N, D);
	fill_ongpu(B * D * T, 0.0f, l.mv_tmp_bdt2_gpu, 1);
	s.input = l.mv_tmp_bdt_gpu;
	s.delta = l.mv_tmp_bdt2_gpu;
	backward_convolutional_layer_gpu(*l.mv_conv_x_layer, s);
	bdt_add_to_branch_kernel<<<cuda_gridsize(static_cast<size_t>(B) * D * T), MV_BLOCK, 0, get_cuda_stream()>>>(l.mv_tmp_bdt2_gpu, l.mv_in_proj_layer->delta_gpu, 0, B, T, N, D);

	fill_ongpu(M * C, 0.0f, l.mv_ln2_out_gpu, 1);
	s.input = l.mv_ln1_out_gpu;
	s.delta = l.mv_ln2_out_gpu;
	backward_connected_layer_gpu(*l.mv_in_proj_layer, s);
	layernorm_backward_kernel<<<cuda_gridsize(static_cast<size_t>(M) * MV_WARP), MV_BLOCK, 0, get_cuda_stream()>>>(l.mv_ln2_out_gpu, l.mv_ln1_xhat_gpu, l.mv_ln1_var_gpu, l.mv_ln1_gamma_gpu,
		l.mv_tokens_gpu, l.mv_ln1_gamma_updates_gpu, l.mv_ln1_beta_updates_gpu, M, C);
	add_inplace_kernel<<<cuda_gridsize(static_cast<size_t>(M) * C), MV_BLOCK, 0, get_cuda_stream()>>>(d_tokens, l.mv_tokens_gpu, M * C);
	if (state.delta)
	{
		tokens_to_spatial_add_kernel<<<cuda_gridsize(static_cast<size_t>(B) * C * T), MV_BLOCK, 0, get_cuda_stream()>>>(d_tokens, state.delta, B, C, T);
	}
	CHECK_CUDA(cudaPeekAtLastError());
}

#endif
