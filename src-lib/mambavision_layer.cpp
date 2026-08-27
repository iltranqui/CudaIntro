#include "mambavision_layer.hpp"
#include "activations.hpp"
#include "blas.hpp"
#include "connected_layer.hpp"
#include "convolutional_layer.hpp"
#include "gemm.hpp"
#include "utils.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <ostream>
#include <vector>

#ifdef DARKNET_GPU
void mambavision_forward_gpu_impl(Darknet::Layer & l, Darknet::NetworkState state);
void mambavision_backward_gpu_impl(Darknet::Layer & l, Darknet::NetworkState state);
#endif

namespace
{
	static auto & cfg_and_state = Darknet::CfgAndState::get();

	static inline float sigmoid_stable(float x)
	{
		if (x >= 0.0f)
		{
			const float z = std::exp(-x);
			return 1.0f / (1.0f + z);
		}
		const float z = std::exp(x);
		return z / (1.0f + z);
	}

	static inline float silu(float x)
	{
		return x * sigmoid_stable(x);
	}

	static inline float silu_grad_from_pre(float x)
	{
		const float s = sigmoid_stable(x);
		return s * (1.0f + x * (1.0f - s));
	}

	static inline float softplus(float x)
	{
		return softplus_activate(x, 20.0f);
	}

	static inline size_t bdt_idx(int b, int d, int t, int D, int T)
	{
		return (static_cast<size_t>(b) * D + d) * T + t;
	}

	static inline size_t state_idx(int b, int d, int t, int s, int D, int T, int S)
	{
		return (((static_cast<size_t>(b) * D + d) * T + t) * S + s);
	}

	static void add_bias_rows(float *buf, const float *bias, int M, int N)
	{
		for (int i = 0; i < M; ++i)
		{
			float *row = buf + static_cast<size_t>(i) * N;
			for (int j = 0; j < N; ++j) row[j] += bias[j];
		}
	}

	static void layernorm_forward(const float *x, float *out, float *mean, float *var, float *xhat,
		const float *gamma, const float *beta, int total_tokens, int C)
	{
		const float eps = 1e-5f;
		for (int i = 0; i < total_tokens; ++i)
		{
			const float *xi = x + static_cast<size_t>(i) * C;
			float *oi = out + static_cast<size_t>(i) * C;
			float *xh = xhat + static_cast<size_t>(i) * C;

			float m = 0.0f;
			for (int j = 0; j < C; ++j) m += xi[j];
			m /= C;
			mean[i] = m;

			float v = 0.0f;
			for (int j = 0; j < C; ++j)
			{
				const float d = xi[j] - m;
				v += d * d;
			}
			v /= C;
			var[i] = v;

			const float inv_std = 1.0f / std::sqrt(v + eps);
			for (int j = 0; j < C; ++j)
			{
				xh[j] = (xi[j] - m) * inv_std;
				oi[j] = xh[j] * gamma[j] + beta[j];
			}
		}
	}

	static void layernorm_backward(const float *dout, const float *xhat, const float *var,
		const float *gamma, float *dx, float *dgamma, float *dbeta, int total_tokens, int C)
	{
		const float eps = 1e-5f;
		for (int i = 0; i < total_tokens; ++i)
		{
			const float *doi = dout + static_cast<size_t>(i) * C;
			const float *xhi = xhat + static_cast<size_t>(i) * C;
			float *dxi = dx + static_cast<size_t>(i) * C;
			const float inv_std = 1.0f / std::sqrt(var[i] + eps);

			for (int j = 0; j < C; ++j)
			{
				dgamma[j] += doi[j] * xhi[j];
				dbeta[j] += doi[j];
			}

			float sum_dxhat = 0.0f;
			float dot_dxhat_xhat = 0.0f;
			for (int j = 0; j < C; ++j)
			{
				const float dxhat = doi[j] * gamma[j];
				sum_dxhat += dxhat;
				dot_dxhat_xhat += dxhat * xhi[j];
			}
			for (int j = 0; j < C; ++j)
			{
				const float dxhat = doi[j] * gamma[j];
				dxi[j] = inv_std * (dxhat - (sum_dxhat + xhi[j] * dot_dxhat_xhat) / C);
			}
		}
	}

	static void depthwise_conv1d_from_proj(const float *proj, int branch_offset, float *pre, float *out,
		const float *weights, const float *biases, int B, int T, int N, int D, int K)
	{
		const int pad = K / 2;
		for (int b = 0; b < B; ++b)
		{
			for (int d = 0; d < D; ++d)
			{
				for (int t = 0; t < T; ++t)
				{
					float sum = biases[d];
					for (int k = 0; k < K; ++k)
					{
						const int src_t = t + k - pad;
						if (src_t < 0 || src_t >= T) continue;
						sum += weights[d * K + k] * proj[(static_cast<size_t>(b) * T + src_t) * N + branch_offset + d];
					}
					const size_t idx = bdt_idx(b, d, t, D, T);
					pre[idx] = sum;
					out[idx] = silu(sum);
				}
			}
		}
	}

	static void depthwise_conv1d_backward_to_proj(const float *proj, int branch_offset,
		const float *pre, const float *dout, float *dproj,
		const float *weights, float *weight_updates, float *bias_updates,
		int B, int T, int N, int D, int K)
	{
		const int pad = K / 2;
		for (int b = 0; b < B; ++b)
		{
			for (int d = 0; d < D; ++d)
			{
				for (int t = 0; t < T; ++t)
				{
					const size_t out_idx = bdt_idx(b, d, t, D, T);
					const float dpre = dout[out_idx] * silu_grad_from_pre(pre[out_idx]);
					bias_updates[d] += dpre;
					for (int k = 0; k < K; ++k)
					{
						const int src_t = t + k - pad;
						if (src_t < 0 || src_t >= T) continue;
						const size_t in_idx = (static_cast<size_t>(b) * T + src_t) * N + branch_offset + d;
						weight_updates[d * K + k] += dpre * proj[in_idx];
						dproj[in_idx] += dpre * weights[d * K + k];
					}
				}
			}
		}
	}

	static void x_proj_forward(const float *x, float *out, const float *w,
		int B, int T, int D, int P)
	{
		for (int b = 0; b < B; ++b)
			for (int t = 0; t < T; ++t)
				for (int p = 0; p < P; ++p)
				{
					float sum = 0.0f;
					for (int d = 0; d < D; ++d)
						sum += x[bdt_idx(b, d, t, D, T)] * w[p * D + d];
					out[(static_cast<size_t>(b) * T + t) * P + p] = sum;
				}
	}

	static void x_proj_backward(const float *x, const float *dout, float *dx,
		const float *w, float *w_updates, int B, int T, int D, int P)
	{
		for (int b = 0; b < B; ++b)
			for (int t = 0; t < T; ++t)
				for (int p = 0; p < P; ++p)
				{
					const float grad = dout[(static_cast<size_t>(b) * T + t) * P + p];
					for (int d = 0; d < D; ++d)
					{
						w_updates[p * D + d] += grad * x[bdt_idx(b, d, t, D, T)];
						dx[bdt_idx(b, d, t, D, T)] += grad * w[p * D + d];
					}
				}
	}

	static void dt_proj_forward(Darknet::Layer & l, int B, int T, int D, int R, int S)
	{
		for (int b = 0; b < B; ++b)
			for (int t = 0; t < T; ++t)
			{
				const float *dt_raw = l.mv_x_proj_out + (static_cast<size_t>(b) * T + t) * (R + 2 * S);
				for (int d = 0; d < D; ++d)
				{
					float sum = l.mv_dt_bias[d];
					for (int r = 0; r < R; ++r) sum += l.mv_dt_proj[d * R + r] * dt_raw[r];
					const size_t idx = bdt_idx(b, d, t, D, T);
					l.mv_dt_pre[idx] = sum;
					l.mv_dt[idx] = softplus(sum);
				}
			}
	}

	static void dt_proj_backward(Darknet::Layer & l, const float *ddt, float *dxproj, int B, int T, int D, int R, int S)
	{
		const int P = R + 2 * S;
		for (int b = 0; b < B; ++b)
			for (int t = 0; t < T; ++t)
			{
				const float *dt_raw = l.mv_x_proj_out + (static_cast<size_t>(b) * T + t) * P;
				float *d_dt_raw = dxproj + (static_cast<size_t>(b) * T + t) * P;
				for (int d = 0; d < D; ++d)
				{
					const size_t idx = bdt_idx(b, d, t, D, T);
					const float dpre = ddt[idx] * sigmoid_stable(l.mv_dt_pre[idx]);
					l.mv_dt_bias_updates[d] += dpre;
					for (int r = 0; r < R; ++r)
					{
						l.mv_dt_proj_updates[d * R + r] += dpre * dt_raw[r];
						d_dt_raw[r] += dpre * l.mv_dt_proj[d * R + r];
					}
				}
			}
	}

	static void scan_forward(Darknet::Layer & l, int B, int T, int D, int R, int S)
	{
		const int P = R + 2 * S;
		for (int b = 0; b < B; ++b)
		{
			for (int d = 0; d < D; ++d)
			{
				std::vector<float> hprev(S, 0.0f);
				for (int t = 0; t < T; ++t)
				{
					const float u = l.mv_x_conv[bdt_idx(b, d, t, D, T)];
					const float dt = l.mv_dt[bdt_idx(b, d, t, D, T)];
					const float *xproj = l.mv_x_proj_out + (static_cast<size_t>(b) * T + t) * P;
					const float *Bparam = xproj + R;
					const float *Cparam = xproj + R + S;
					float y = l.mv_D[d] * u;

					for (int s = 0; s < S; ++s)
					{
						const float A = -std::exp(l.mv_A_log[d * S + s]);
						const float abar = std::exp(std::max(-60.0f, std::min(20.0f, dt * A)));
						const float h = abar * hprev[s] + dt * Bparam[s] * u;
						l.mv_scan_state[state_idx(b, d, t, s, D, T, S)] = h;
						hprev[s] = h;
						y += h * Cparam[s];
					}
					l.mv_scan_out[bdt_idx(b, d, t, D, T)] = y;
				}
			}
		}
	}

	static void scan_backward(Darknet::Layer & l, const float *dscan_out, float *dxconv, float *dxproj, float *ddt,
		int B, int T, int D, int R, int S)
	{
		const int P = R + 2 * S;
		for (int b = 0; b < B; ++b)
		{
			for (int d = 0; d < D; ++d)
			{
				std::vector<float> dh_next(S, 0.0f);
				for (int t = T - 1; t >= 0; --t)
				{
					const float u = l.mv_x_conv[bdt_idx(b, d, t, D, T)];
					const float dt = l.mv_dt[bdt_idx(b, d, t, D, T)];
					const float dy = dscan_out[bdt_idx(b, d, t, D, T)];
					const float *xproj = l.mv_x_proj_out + (static_cast<size_t>(b) * T + t) * P;
					float *dxproj_row = dxproj + (static_cast<size_t>(b) * T + t) * P;
					const float *Bparam = xproj + R;
					const float *Cparam = xproj + R + S;
					float *dBparam = dxproj_row + R;
					float *dCparam = dxproj_row + R + S;

					l.mv_D_updates[d] += dy * u;
					dxconv[bdt_idx(b, d, t, D, T)] += dy * l.mv_D[d];

					std::vector<float> dh_prev(S, 0.0f);
					for (int s = 0; s < S; ++s)
					{
						const float h = l.mv_scan_state[state_idx(b, d, t, s, D, T, S)];
						const float hprev = (t == 0) ? 0.0f : l.mv_scan_state[state_idx(b, d, t - 1, s, D, T, S)];
						const float A = -std::exp(l.mv_A_log[d * S + s]);
						const float abar = std::exp(std::max(-60.0f, std::min(20.0f, dt * A)));

						dCparam[s] += dy * h;
						float dh = dy * Cparam[s] + dh_next[s];

						const float d_abar = dh * hprev;
						dh_prev[s] += dh * abar;
						ddt[bdt_idx(b, d, t, D, T)] += dh * Bparam[s] * u;
						dBparam[s] += dh * dt * u;
						dxconv[bdt_idx(b, d, t, D, T)] += dh * dt * Bparam[s];

						const float d_exp_arg = d_abar * abar;
						const float dA = d_exp_arg * dt;
						ddt[bdt_idx(b, d, t, D, T)] += d_exp_arg * A;
						l.mv_A_log_updates[d * S + s] += dA * A;
					}
					dh_next.swap(dh_prev);
				}
			}
		}
	}

	static void update_param(float *w, float *dw, int count, int batch, float lr, float momentum, float decay, bool decay_weights)
	{
		if (!w || !dw || count <= 0) return;
		if (decay_weights) axpy_cpu(count, -decay * batch, w, 1, dw, 1);
		axpy_cpu(count, lr / batch, dw, 1, w, 1);
		scal_cpu(count, momentum, dw, 1);
	}

	class NullStreamBuffer final : public std::streambuf
	{
		public:
			int overflow(int c) override { return c; }
	};

	static NullStreamBuffer null_stream_buffer;
	static std::ostream null_stream(&null_stream_buffer);

	class ScopedOutputSilencer final
	{
		public:
			ScopedOutputSilencer() : previous(cfg_and_state.output)
			{
				cfg_and_state.output = &null_stream;
			}

			~ScopedOutputSilencer()
			{
				cfg_and_state.output = previous;
			}

		private:
			std::ostream *previous;
	};

	static void copy_connected_params_to_layer(Darknet::Layer *dst, const float *w, const float *b, int w_count, int b_count)
	{
		if (!dst) return;
		std::memcpy(dst->weights, w, static_cast<size_t>(w_count) * sizeof(float));
		std::memset(dst->weight_updates, 0, static_cast<size_t>(w_count) * sizeof(float));
		if (b)
		{
			std::memcpy(dst->biases, b, static_cast<size_t>(b_count) * sizeof(float));
		}
		else
		{
			std::memset(dst->biases, 0, static_cast<size_t>(b_count) * sizeof(float));
		}
		std::memset(dst->bias_updates, 0, static_cast<size_t>(b_count) * sizeof(float));
	}

	static void copy_connected_params_from_layer(Darknet::Layer *src, float *w, float *dw, float *b, float *db, int w_count, int b_count)
	{
		if (!src) return;
		if (w) std::memcpy(w, src->weights, static_cast<size_t>(w_count) * sizeof(float));
		if (dw) std::memcpy(dw, src->weight_updates, static_cast<size_t>(w_count) * sizeof(float));
		if (b) std::memcpy(b, src->biases, static_cast<size_t>(b_count) * sizeof(float));
		if (db) std::memcpy(db, src->bias_updates, static_cast<size_t>(b_count) * sizeof(float));
	}

	static void copy_depthwise_1d_to_conv(Darknet::Layer *dst, const float *w, const float *b, int D, int K)
	{
		if (!dst) return;
		std::memset(dst->weights, 0, static_cast<size_t>(dst->nweights) * sizeof(float));
		if (dst->weight_updates) std::memset(dst->weight_updates, 0, static_cast<size_t>(dst->nweights) * sizeof(float));
		const int center = K / 2;
		for (int d = 0; d < D; ++d)
		{
			for (int k = 0; k < K; ++k)
			{
				dst->weights[d * K * K + center * K + k] = w[d * K + k];
			}
		}
		std::memcpy(dst->biases, b, static_cast<size_t>(D) * sizeof(float));
		if (dst->bias_updates) std::memset(dst->bias_updates, 0, static_cast<size_t>(D) * sizeof(float));
	}

	static void copy_depthwise_1d_from_conv(Darknet::Layer *src, float *w, float *dw, float *b, float *db, int D, int K)
	{
		if (!src) return;
		const int center = K / 2;
		for (int d = 0; d < D; ++d)
		{
			for (int k = 0; k < K; ++k)
			{
				w[d * K + k] = src->weights[d * K * K + center * K + k];
				if (dw && src->weight_updates) dw[d * K + k] = src->weight_updates[d * K * K + center * K + k];
			}
		}
		std::memcpy(b, src->biases, static_cast<size_t>(D) * sizeof(float));
		if (db && src->bias_updates) std::memcpy(db, src->bias_updates, static_cast<size_t>(D) * sizeof(float));
	}

	static void free_mambavision_sublayer(Darknet::Layer *&sub)
	{
		if (!sub) return;
		free_layer(*sub);
		free(sub);
		sub = nullptr;
	}

	static void recreate_mambavision_sublayers(Darknet::Layer &l)
	{
		const int T = l.h * l.w;
		const int M = l.batch * T;
		const int D = l.n / 2;
		const int P = l.mv_dt_rank + 2 * l.mv_d_state;
		const int ffn_hidden = l.n * l.mv_ffn_ratio;
		const ScopedOutputSilencer quiet;

		free_mambavision_sublayer(l.mv_in_proj_layer);
		free_mambavision_sublayer(l.mv_conv_x_layer);
		free_mambavision_sublayer(l.mv_conv_z_layer);
		free_mambavision_sublayer(l.mv_x_proj_layer);
		free_mambavision_sublayer(l.mv_dt_proj_layer);
		free_mambavision_sublayer(l.mv_out_proj_layer);
		free_mambavision_sublayer(l.mv_res_proj_layer);
		free_mambavision_sublayer(l.mv_ffn1_layer);
		free_mambavision_sublayer(l.mv_ffn2_layer);

		l.mv_in_proj_layer = (Darknet::Layer*)xcalloc(1, sizeof(Darknet::Layer));
		*(l.mv_in_proj_layer) = make_connected_layer(M, 1, l.c, l.n, LINEAR, 0);
		copy_connected_params_to_layer(l.mv_in_proj_layer, l.weights, l.biases, l.n * l.c, l.n);

		l.mv_conv_x_layer = (Darknet::Layer*)xcalloc(1, sizeof(Darknet::Layer));
		*(l.mv_conv_x_layer) = make_convolutional_layer(l.batch, 1, 1, T, D, D, D, l.mv_conv_size, 1, 1, 1, 1,
			SWISH, 0, 0, 0, 0, 0, l.index, 0, nullptr, 0, 0, l.train);
		copy_depthwise_1d_to_conv(l.mv_conv_x_layer, l.mv_conv_x, l.mv_conv_x_bias, D, l.mv_conv_size);

		l.mv_conv_z_layer = (Darknet::Layer*)xcalloc(1, sizeof(Darknet::Layer));
		*(l.mv_conv_z_layer) = make_convolutional_layer(l.batch, 1, 1, T, D, D, D, l.mv_conv_size, 1, 1, 1, 1,
			SWISH, 0, 0, 0, 0, 0, l.index, 0, nullptr, 0, 0, l.train);
		copy_depthwise_1d_to_conv(l.mv_conv_z_layer, l.mv_conv_z, l.mv_conv_z_bias, D, l.mv_conv_size);

		l.mv_x_proj_layer = (Darknet::Layer*)xcalloc(1, sizeof(Darknet::Layer));
		*(l.mv_x_proj_layer) = make_connected_layer(M, 1, D, P, LINEAR, 0);
		copy_connected_params_to_layer(l.mv_x_proj_layer, l.mv_x_proj, nullptr, P * D, P);

		l.mv_dt_proj_layer = (Darknet::Layer*)xcalloc(1, sizeof(Darknet::Layer));
		*(l.mv_dt_proj_layer) = make_connected_layer(M, 1, l.mv_dt_rank, D, LINEAR, 0);
		copy_connected_params_to_layer(l.mv_dt_proj_layer, l.mv_dt_proj, l.mv_dt_bias, D * l.mv_dt_rank, D);

		l.mv_out_proj_layer = (Darknet::Layer*)xcalloc(1, sizeof(Darknet::Layer));
		*(l.mv_out_proj_layer) = make_connected_layer(M, 1, l.n, l.n, LINEAR, 0);
		copy_connected_params_to_layer(l.mv_out_proj_layer, l.mv_out_proj, l.mv_out_bias, l.n * l.n, l.n);

		if (l.c != l.n)
		{
			l.mv_res_proj_layer = (Darknet::Layer*)xcalloc(1, sizeof(Darknet::Layer));
			*(l.mv_res_proj_layer) = make_connected_layer(M, 1, l.c, l.n, LINEAR, 0);
			copy_connected_params_to_layer(l.mv_res_proj_layer, l.mv_res_proj, nullptr, l.n * l.c, l.n);
		}

		l.mv_ffn1_layer = (Darknet::Layer*)xcalloc(1, sizeof(Darknet::Layer));
		*(l.mv_ffn1_layer) = make_connected_layer(M, 1, l.n, ffn_hidden, l.activation, 0);
		copy_connected_params_to_layer(l.mv_ffn1_layer, l.mv_ffn_w1, l.mv_ffn_b1, ffn_hidden * l.n, ffn_hidden);

		l.mv_ffn2_layer = (Darknet::Layer*)xcalloc(1, sizeof(Darknet::Layer));
		*(l.mv_ffn2_layer) = make_connected_layer(M, 1, ffn_hidden, l.n, LINEAR, 0);
		copy_connected_params_to_layer(l.mv_ffn2_layer, l.mv_ffn_w2, l.mv_ffn_b2, l.n * ffn_hidden, l.n);
	}

#ifdef DARKNET_GPU
	static void free_gpu_workspace(Darknet::Layer &l)
	{
		cuda_free(l.mv_tokens_gpu);
		cuda_free(l.mv_ln1_out_gpu);
		cuda_free(l.mv_ln1_mean_gpu);
		cuda_free(l.mv_ln1_var_gpu);
		cuda_free(l.mv_ln1_xhat_gpu);
		cuda_free(l.mv_dt_pre_gpu);
		cuda_free(l.mv_dt_gpu);
		cuda_free(l.mv_scan_state_gpu);
		cuda_free(l.mv_scan_out_gpu);
		cuda_free(l.mv_mixer_cat_gpu);
		cuda_free(l.mv_pre_res2_gpu);
		cuda_free(l.mv_ln2_out_gpu);
		cuda_free(l.mv_ln2_mean_gpu);
		cuda_free(l.mv_ln2_var_gpu);
		cuda_free(l.mv_ln2_xhat_gpu);
		cuda_free(l.mv_tmp_token_c_gpu);
		cuda_free(l.mv_tmp_token_n_gpu);
		cuda_free(l.mv_tmp_token_p_gpu);
		cuda_free(l.mv_tmp_bdt_gpu);
		cuda_free(l.mv_tmp_bdt2_gpu);
		cuda_free(l.mv_tmp_ffn_gpu);
	}

	static void allocate_gpu_workspace(Darknet::Layer &l)
	{
		const int T = l.h * l.w;
		const int M = l.batch * T;
		const int D = l.n / 2;
		const int P = l.mv_dt_rank + 2 * l.mv_d_state;
		const int ffn_hidden = l.n * l.mv_ffn_ratio;

		l.mv_tokens_gpu = cuda_make_array(nullptr, M * l.c);
		l.mv_ln1_out_gpu = cuda_make_array(nullptr, M * l.c);
		l.mv_ln1_mean_gpu = cuda_make_array(nullptr, M);
		l.mv_ln1_var_gpu = cuda_make_array(nullptr, M);
		l.mv_ln1_xhat_gpu = cuda_make_array(nullptr, M * l.c);
		l.mv_dt_pre_gpu = cuda_make_array(nullptr, l.batch * D * T);
		l.mv_dt_gpu = cuda_make_array(nullptr, l.batch * D * T);
		l.mv_scan_state_gpu = cuda_make_array(nullptr, static_cast<size_t>(l.batch) * D * T * l.mv_d_state);
		l.mv_scan_out_gpu = cuda_make_array(nullptr, l.batch * D * T);
		l.mv_mixer_cat_gpu = cuda_make_array(nullptr, M * l.n);
		l.mv_pre_res2_gpu = cuda_make_array(nullptr, M * l.n);
		l.mv_ln2_out_gpu = cuda_make_array(nullptr, M * l.n);
		l.mv_ln2_mean_gpu = cuda_make_array(nullptr, M);
		l.mv_ln2_var_gpu = cuda_make_array(nullptr, M);
		l.mv_ln2_xhat_gpu = cuda_make_array(nullptr, M * l.n);
		l.mv_tmp_token_c_gpu = cuda_make_array(nullptr, M * l.c);
		l.mv_tmp_token_n_gpu = cuda_make_array(nullptr, M * l.n);
		l.mv_tmp_token_p_gpu = cuda_make_array(nullptr, M * P);
		l.mv_tmp_bdt_gpu = cuda_make_array(nullptr, l.batch * D * T);
		l.mv_tmp_bdt2_gpu = cuda_make_array(nullptr, l.batch * D * T);
		l.mv_tmp_ffn_gpu = cuda_make_array(nullptr, M * ffn_hidden);
	}
#endif
}

Darknet::Layer make_mambavision_layer(int batch, int h, int w, int c, int n,
	int d_state, int conv_size, int dt_rank, int ffn_ratio, ACTIVATION activation, int index, int train)
{
	TAT(TATPARMS);

	if (n < 2 || (n % 2) != 0)
	{
		darknet_fatal_error(DARKNET_LOC, "mambavision: filters must be an even value >= 2, got %d", n);
	}
	if (d_state < 1)
	{
		darknet_fatal_error(DARKNET_LOC, "mambavision: state must be >= 1, got %d", d_state);
	}
	if (conv_size < 1 || (conv_size % 2) == 0)
	{
		darknet_fatal_error(DARKNET_LOC, "mambavision: conv_size must be a positive odd value, got %d", conv_size);
	}
	if (ffn_ratio < 1)
	{
		darknet_fatal_error(DARKNET_LOC, "mambavision: ffn_ratio must be >= 1, got %d", ffn_ratio);
	}

	Darknet::Layer l = {};
	l.type = Darknet::ELayerType::MAMBAVISION;
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
	l.learning_rate_scale = 1.0f;

	l.mv_d_state = d_state;
	l.mv_conv_size = conv_size;
	l.mv_dt_rank = dt_rank > 0 ? dt_rank : (n + 15) / 16;
	l.mv_ffn_ratio = ffn_ratio;

	const int T = h * w;
	const int M = batch * T;
	const int D = n / 2;
	const int R = l.mv_dt_rank;
	const int S = d_state;
	const int P = R + 2 * S;
	const int ffn_hidden = n * ffn_ratio;

	l.nweights = n * c;
	l.nbiases = n;
	l.weights = (float*)xcalloc(l.nweights, sizeof(float));
	l.weight_updates = (float*)xcalloc(l.nweights, sizeof(float));
	l.biases = (float*)xcalloc(l.nbiases, sizeof(float));
	l.bias_updates = (float*)xcalloc(l.nbiases, sizeof(float));

	l.mv_conv_x = (float*)xcalloc(D * conv_size, sizeof(float));
	l.mv_conv_x_updates = (float*)xcalloc(D * conv_size, sizeof(float));
	l.mv_conv_x_bias = (float*)xcalloc(D, sizeof(float));
	l.mv_conv_x_bias_updates = (float*)xcalloc(D, sizeof(float));
	l.mv_conv_z = (float*)xcalloc(D * conv_size, sizeof(float));
	l.mv_conv_z_updates = (float*)xcalloc(D * conv_size, sizeof(float));
	l.mv_conv_z_bias = (float*)xcalloc(D, sizeof(float));
	l.mv_conv_z_bias_updates = (float*)xcalloc(D, sizeof(float));
	l.mv_x_proj = (float*)xcalloc(P * D, sizeof(float));
	l.mv_x_proj_updates = (float*)xcalloc(P * D, sizeof(float));
	l.mv_dt_proj = (float*)xcalloc(D * R, sizeof(float));
	l.mv_dt_proj_updates = (float*)xcalloc(D * R, sizeof(float));
	l.mv_dt_bias = (float*)xcalloc(D, sizeof(float));
	l.mv_dt_bias_updates = (float*)xcalloc(D, sizeof(float));
	l.mv_A_log = (float*)xcalloc(D * S, sizeof(float));
	l.mv_A_log_updates = (float*)xcalloc(D * S, sizeof(float));
	l.mv_D = (float*)xcalloc(D, sizeof(float));
	l.mv_D_updates = (float*)xcalloc(D, sizeof(float));
	l.mv_out_proj = (float*)xcalloc(n * n, sizeof(float));
	l.mv_out_proj_updates = (float*)xcalloc(n * n, sizeof(float));
	l.mv_out_bias = (float*)xcalloc(n, sizeof(float));
	l.mv_out_bias_updates = (float*)xcalloc(n, sizeof(float));
	if (c != n)
	{
		l.mv_res_proj = (float*)xcalloc(n * c, sizeof(float));
		l.mv_res_proj_updates = (float*)xcalloc(n * c, sizeof(float));
	}
	l.mv_ln1_gamma = (float*)xcalloc(c, sizeof(float));
	l.mv_ln1_gamma_updates = (float*)xcalloc(c, sizeof(float));
	l.mv_ln1_beta = (float*)xcalloc(c, sizeof(float));
	l.mv_ln1_beta_updates = (float*)xcalloc(c, sizeof(float));
	l.mv_ln2_gamma = (float*)xcalloc(n, sizeof(float));
	l.mv_ln2_gamma_updates = (float*)xcalloc(n, sizeof(float));
	l.mv_ln2_beta = (float*)xcalloc(n, sizeof(float));
	l.mv_ln2_beta_updates = (float*)xcalloc(n, sizeof(float));
	l.mv_ffn_w1 = (float*)xcalloc(ffn_hidden * n, sizeof(float));
	l.mv_ffn_w1_updates = (float*)xcalloc(ffn_hidden * n, sizeof(float));
	l.mv_ffn_b1 = (float*)xcalloc(ffn_hidden, sizeof(float));
	l.mv_ffn_b1_updates = (float*)xcalloc(ffn_hidden, sizeof(float));
	l.mv_ffn_w2 = (float*)xcalloc(n * ffn_hidden, sizeof(float));
	l.mv_ffn_w2_updates = (float*)xcalloc(n * ffn_hidden, sizeof(float));
	l.mv_ffn_b2 = (float*)xcalloc(n, sizeof(float));
	l.mv_ffn_b2_updates = (float*)xcalloc(n, sizeof(float));

	const float xavier_in = std::sqrt(6.0f / (float)(c + n));
	rand_uniform_many_weight_init(l.weights, l.nweights, -xavier_in, xavier_in);
	const float conv_scale = std::sqrt(2.0f / (float)conv_size);
	rand_uniform_many_weight_init(l.mv_conv_x, D * conv_size, -conv_scale, conv_scale);
	rand_uniform_many_weight_init(l.mv_conv_z, D * conv_size, -conv_scale, conv_scale);
	const float xproj_scale = std::sqrt(6.0f / (float)(D + P));
	rand_uniform_many_weight_init(l.mv_x_proj, P * D, -xproj_scale, xproj_scale);
	const float dt_scale = std::pow((float)R, -0.5f);
	rand_uniform_many_weight_init(l.mv_dt_proj, D * R, -dt_scale, dt_scale);
	for (int d = 0; d < D; ++d)
	{
		const float dt = std::exp(rand_uniform_weight_init(std::log(0.001f), std::log(0.1f)));
		l.mv_dt_bias[d] = dt + std::log(-std::expm1(-dt));
		l.mv_D[d] = 1.0f;
	}
	for (int d = 0; d < D; ++d)
		for (int s = 0; s < S; ++s)
			l.mv_A_log[d * S + s] = std::log((float)(s + 1));
	const float out_scale = std::sqrt(6.0f / (float)(n + n));
	rand_uniform_many_weight_init(l.mv_out_proj, n * n, -out_scale, out_scale);
	if (c != n)
	{
		const float res_scale = std::sqrt(6.0f / (float)(c + n));
		rand_uniform_many_weight_init(l.mv_res_proj, n * c, -res_scale, res_scale);
	}
	for (int i = 0; i < c; ++i) l.mv_ln1_gamma[i] = 1.0f;
	for (int i = 0; i < n; ++i) l.mv_ln2_gamma[i] = 1.0f;
	const float ffn1_scale = std::sqrt(6.0f / (float)n);
	const float ffn2_scale = std::sqrt(6.0f / (float)ffn_hidden);
	rand_uniform_many_weight_init(l.mv_ffn_w1, ffn_hidden * n, -ffn1_scale, ffn1_scale);
	rand_uniform_many_weight_init(l.mv_ffn_w2, n * ffn_hidden, -ffn2_scale, ffn2_scale);
	recreate_mambavision_sublayers(l);

	l.mv_tokens = (float*)xcalloc(M * c, sizeof(float));
	l.mv_ln1_out = (float*)xcalloc(M * c, sizeof(float));
	l.mv_ln1_mean = (float*)xcalloc(M, sizeof(float));
	l.mv_ln1_var = (float*)xcalloc(M, sizeof(float));
	l.mv_ln1_xhat = (float*)xcalloc(M * c, sizeof(float));
	l.mv_in_proj_out = (float*)xcalloc(M * n, sizeof(float));
	l.mv_x_conv_pre = (float*)xcalloc(batch * D * T, sizeof(float));
	l.mv_x_conv = (float*)xcalloc(batch * D * T, sizeof(float));
	l.mv_z_conv_pre = (float*)xcalloc(batch * D * T, sizeof(float));
	l.mv_z_conv = (float*)xcalloc(batch * D * T, sizeof(float));
	l.mv_x_proj_out = (float*)xcalloc(M * P, sizeof(float));
	l.mv_dt_pre = (float*)xcalloc(batch * D * T, sizeof(float));
	l.mv_dt = (float*)xcalloc(batch * D * T, sizeof(float));
	l.mv_scan_state = (float*)xcalloc(static_cast<size_t>(batch) * D * T * S, sizeof(float));
	l.mv_scan_out = (float*)xcalloc(batch * D * T, sizeof(float));
	l.mv_mixer_cat = (float*)xcalloc(M * n, sizeof(float));
	l.mv_mixer_out = (float*)xcalloc(M * n, sizeof(float));
	l.mv_pre_res2 = (float*)xcalloc(M * n, sizeof(float));
	l.mv_ln2_out = (float*)xcalloc(M * n, sizeof(float));
	l.mv_ln2_mean = (float*)xcalloc(M, sizeof(float));
	l.mv_ln2_var = (float*)xcalloc(M, sizeof(float));
	l.mv_ln2_xhat = (float*)xcalloc(M * n, sizeof(float));
	l.mv_ffn_hidden = (float*)xcalloc(M * ffn_hidden, sizeof(float));
	l.activation_input = (float*)xcalloc(M * ffn_hidden, sizeof(float));
	l.output = (float*)xcalloc(batch * l.outputs, sizeof(float));
	l.delta = (float*)xcalloc(batch * l.outputs, sizeof(float));

	l.forward = forward_mambavision_layer;
	l.backward = backward_mambavision_layer;
	l.update = update_mambavision_layer;

#ifdef DARKNET_GPU
	l.forward_gpu = forward_mambavision_layer_gpu;
	l.backward_gpu = backward_mambavision_layer_gpu;
	l.update_gpu = update_mambavision_layer_gpu;
	l.mv_gpu_input_cpu = (float*)xcalloc(batch * l.inputs, sizeof(float));
	l.output_gpu = cuda_make_array(l.output, batch * l.outputs);
	l.delta_gpu = cuda_make_array(l.delta, batch * l.outputs);
	l.mv_ln1_gamma_gpu = cuda_make_array(l.mv_ln1_gamma, c);
	l.mv_ln1_gamma_updates_gpu = cuda_make_array(l.mv_ln1_gamma_updates, c);
	l.mv_ln1_beta_gpu = cuda_make_array(l.mv_ln1_beta, c);
	l.mv_ln1_beta_updates_gpu = cuda_make_array(l.mv_ln1_beta_updates, c);
	l.mv_ln2_gamma_gpu = cuda_make_array(l.mv_ln2_gamma, n);
	l.mv_ln2_gamma_updates_gpu = cuda_make_array(l.mv_ln2_gamma_updates, n);
	l.mv_ln2_beta_gpu = cuda_make_array(l.mv_ln2_beta, n);
	l.mv_ln2_beta_updates_gpu = cuda_make_array(l.mv_ln2_beta_updates, n);
	l.mv_A_log_gpu = cuda_make_array(l.mv_A_log, D * S);
	l.mv_A_log_updates_gpu = cuda_make_array(l.mv_A_log_updates, D * S);
	l.mv_D_gpu = cuda_make_array(l.mv_D, D);
	l.mv_D_updates_gpu = cuda_make_array(l.mv_D_updates, D);
	allocate_gpu_workspace(l);
#endif

	return l;
}

void resize_mambavision_layer(Darknet::Layer * l, int w, int h)
{
	TAT(TATPARMS);

	l->h = h;
	l->w = w;
	l->out_h = h;
	l->out_w = w;
	l->inputs = h * w * l->c;
	l->outputs = h * w * l->n;

	const int T = h * w;
	const int M = l->batch * T;
	const int D = l->n / 2;
	const int P = l->mv_dt_rank + 2 * l->mv_d_state;
	const int ffn_hidden = l->n * l->mv_ffn_ratio;

	l->mv_tokens = (float*)xrealloc(l->mv_tokens, M * l->c * sizeof(float));
	l->mv_ln1_out = (float*)xrealloc(l->mv_ln1_out, M * l->c * sizeof(float));
	l->mv_ln1_mean = (float*)xrealloc(l->mv_ln1_mean, M * sizeof(float));
	l->mv_ln1_var = (float*)xrealloc(l->mv_ln1_var, M * sizeof(float));
	l->mv_ln1_xhat = (float*)xrealloc(l->mv_ln1_xhat, M * l->c * sizeof(float));
	l->mv_in_proj_out = (float*)xrealloc(l->mv_in_proj_out, M * l->n * sizeof(float));
	l->mv_x_conv_pre = (float*)xrealloc(l->mv_x_conv_pre, l->batch * D * T * sizeof(float));
	l->mv_x_conv = (float*)xrealloc(l->mv_x_conv, l->batch * D * T * sizeof(float));
	l->mv_z_conv_pre = (float*)xrealloc(l->mv_z_conv_pre, l->batch * D * T * sizeof(float));
	l->mv_z_conv = (float*)xrealloc(l->mv_z_conv, l->batch * D * T * sizeof(float));
	l->mv_x_proj_out = (float*)xrealloc(l->mv_x_proj_out, M * P * sizeof(float));
	l->mv_dt_pre = (float*)xrealloc(l->mv_dt_pre, l->batch * D * T * sizeof(float));
	l->mv_dt = (float*)xrealloc(l->mv_dt, l->batch * D * T * sizeof(float));
	l->mv_scan_state = (float*)xrealloc(l->mv_scan_state, static_cast<size_t>(l->batch) * D * T * l->mv_d_state * sizeof(float));
	l->mv_scan_out = (float*)xrealloc(l->mv_scan_out, l->batch * D * T * sizeof(float));
	l->mv_mixer_cat = (float*)xrealloc(l->mv_mixer_cat, M * l->n * sizeof(float));
	l->mv_mixer_out = (float*)xrealloc(l->mv_mixer_out, M * l->n * sizeof(float));
	l->mv_pre_res2 = (float*)xrealloc(l->mv_pre_res2, M * l->n * sizeof(float));
	l->mv_ln2_out = (float*)xrealloc(l->mv_ln2_out, M * l->n * sizeof(float));
	l->mv_ln2_mean = (float*)xrealloc(l->mv_ln2_mean, M * sizeof(float));
	l->mv_ln2_var = (float*)xrealloc(l->mv_ln2_var, M * sizeof(float));
	l->mv_ln2_xhat = (float*)xrealloc(l->mv_ln2_xhat, M * l->n * sizeof(float));
	l->mv_ffn_hidden = (float*)xrealloc(l->mv_ffn_hidden, M * ffn_hidden * sizeof(float));
	l->activation_input = (float*)xrealloc(l->activation_input, M * ffn_hidden * sizeof(float));
	l->output = (float*)xrealloc(l->output, l->batch * l->outputs * sizeof(float));
	l->delta = (float*)xrealloc(l->delta, l->batch * l->outputs * sizeof(float));

	recreate_mambavision_sublayers(*l);

#ifdef DARKNET_GPU
	l->mv_gpu_input_cpu = (float*)xrealloc(l->mv_gpu_input_cpu, l->batch * l->inputs * sizeof(float));
	cuda_free(l->output_gpu);
	cuda_free(l->delta_gpu);
	l->output_gpu = cuda_make_array(l->output, l->batch * l->outputs);
	l->delta_gpu = cuda_make_array(l->delta, l->batch * l->outputs);
	free_gpu_workspace(*l);
	allocate_gpu_workspace(*l);
	push_mambavision_layer(*l);
#endif
}

void forward_mambavision_layer(Darknet::Layer & l, Darknet::NetworkState state)
{
	TAT(TATPARMS);

	const int B = l.batch;
	const int C = l.c;
	const int N = l.n;
	const int H = l.h;
	const int W = l.w;
	const int T = H * W;
	const int M = B * T;
	const int D = N / 2;
	const int R = l.mv_dt_rank;
	const int S = l.mv_d_state;
	const int P = R + 2 * S;
	const int ffn_hidden = N * l.mv_ffn_ratio;

	for (int b = 0; b < B; ++b)
		for (int c = 0; c < C; ++c)
			for (int t = 0; t < T; ++t)
				l.mv_tokens[(static_cast<size_t>(b) * T + t) * C + c] = state.input[(static_cast<size_t>(b) * C + c) * T + t];

	layernorm_forward(l.mv_tokens, l.mv_ln1_out, l.mv_ln1_mean, l.mv_ln1_var, l.mv_ln1_xhat,
		l.mv_ln1_gamma, l.mv_ln1_beta, M, C);

	gemm_cpu(0, 1, M, N, C, 1.0f, l.mv_ln1_out, C, l.weights, C, 0.0f, l.mv_in_proj_out, N);
	add_bias_rows(l.mv_in_proj_out, l.biases, M, N);

	depthwise_conv1d_from_proj(l.mv_in_proj_out, 0, l.mv_x_conv_pre, l.mv_x_conv,
		l.mv_conv_x, l.mv_conv_x_bias, B, T, N, D, l.mv_conv_size);
	depthwise_conv1d_from_proj(l.mv_in_proj_out, D, l.mv_z_conv_pre, l.mv_z_conv,
		l.mv_conv_z, l.mv_conv_z_bias, B, T, N, D, l.mv_conv_size);

	x_proj_forward(l.mv_x_conv, l.mv_x_proj_out, l.mv_x_proj, B, T, D, P);
	dt_proj_forward(l, B, T, D, R, S);
	scan_forward(l, B, T, D, R, S);

	for (int b = 0; b < B; ++b)
		for (int t = 0; t < T; ++t)
		{
			float *row = l.mv_mixer_cat + (static_cast<size_t>(b) * T + t) * N;
			for (int d = 0; d < D; ++d)
			{
				row[d] = l.mv_scan_out[bdt_idx(b, d, t, D, T)];
				row[D + d] = l.mv_z_conv[bdt_idx(b, d, t, D, T)];
			}
		}

	gemm_cpu(0, 1, M, N, N, 1.0f, l.mv_mixer_cat, N, l.mv_out_proj, N, 0.0f, l.mv_mixer_out, N);
	add_bias_rows(l.mv_mixer_out, l.mv_out_bias, M, N);

	if (C == N)
	{
		for (int i = 0; i < M * N; ++i) l.mv_pre_res2[i] = l.mv_mixer_out[i] + l.mv_tokens[i];
	}
	else
	{
		gemm_cpu(0, 1, M, N, C, 1.0f, l.mv_tokens, C, l.mv_res_proj, C, 0.0f, l.mv_pre_res2, N);
		for (int i = 0; i < M * N; ++i) l.mv_pre_res2[i] += l.mv_mixer_out[i];
	}

	layernorm_forward(l.mv_pre_res2, l.mv_ln2_out, l.mv_ln2_mean, l.mv_ln2_var, l.mv_ln2_xhat,
		l.mv_ln2_gamma, l.mv_ln2_beta, M, N);
	gemm_cpu(0, 1, M, ffn_hidden, N, 1.0f, l.mv_ln2_out, N, l.mv_ffn_w1, N, 0.0f, l.mv_ffn_hidden, ffn_hidden);
	add_bias_rows(l.mv_ffn_hidden, l.mv_ffn_b1, M, ffn_hidden);
	std::memcpy(l.activation_input, l.mv_ffn_hidden, static_cast<size_t>(M) * ffn_hidden * sizeof(float));
	activate_array(l.mv_ffn_hidden, M * ffn_hidden, l.activation);
	gemm_cpu(0, 1, M, N, ffn_hidden, 1.0f, l.mv_ffn_hidden, ffn_hidden, l.mv_ffn_w2, ffn_hidden, 0.0f, l.mv_mixer_out, N);
	add_bias_rows(l.mv_mixer_out, l.mv_ffn_b2, M, N);

	for (int b = 0; b < B; ++b)
		for (int t = 0; t < T; ++t)
			for (int n = 0; n < N; ++n)
			{
				const float value = l.mv_mixer_out[(static_cast<size_t>(b) * T + t) * N + n] + l.mv_pre_res2[(static_cast<size_t>(b) * T + t) * N + n];
				l.output[(static_cast<size_t>(b) * N + n) * T + t] = value;
			}
}

void backward_mambavision_layer(Darknet::Layer & l, Darknet::NetworkState state)
{
	TAT(TATPARMS);

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

	std::vector<float> d_out(M * N, 0.0f);
	for (int b = 0; b < B; ++b)
		for (int n = 0; n < N; ++n)
			for (int t = 0; t < T; ++t)
				d_out[(static_cast<size_t>(b) * T + t) * N + n] = l.delta[(static_cast<size_t>(b) * N + n) * T + t];

	std::vector<float> d_pre_res2 = d_out;
	std::vector<float> d_ffn_hidden(M * ffn_hidden, 0.0f);
	std::vector<float> d_ln2_out(M * N, 0.0f);

	gemm_cpu(1, 0, N, ffn_hidden, M, 1.0f, d_out.data(), N, l.mv_ffn_hidden, ffn_hidden, 1.0f, l.mv_ffn_w2_updates, ffn_hidden);
	for (int i = 0; i < M; ++i)
		for (int j = 0; j < N; ++j)
			l.mv_ffn_b2_updates[j] += d_out[static_cast<size_t>(i) * N + j];
	gemm_cpu(0, 0, M, ffn_hidden, N, 1.0f, d_out.data(), N, l.mv_ffn_w2, ffn_hidden, 0.0f, d_ffn_hidden.data(), ffn_hidden);
	gradient_array(l.activation_input, M * ffn_hidden, l.activation, d_ffn_hidden.data());
	gemm_cpu(1, 0, ffn_hidden, N, M, 1.0f, d_ffn_hidden.data(), ffn_hidden, l.mv_ln2_out, N, 1.0f, l.mv_ffn_w1_updates, N);
	for (int i = 0; i < M; ++i)
		for (int j = 0; j < ffn_hidden; ++j)
			l.mv_ffn_b1_updates[j] += d_ffn_hidden[static_cast<size_t>(i) * ffn_hidden + j];
	gemm_cpu(0, 0, M, N, ffn_hidden, 1.0f, d_ffn_hidden.data(), ffn_hidden, l.mv_ffn_w1, N, 0.0f, d_ln2_out.data(), N);

	std::vector<float> d_ln2_in(M * N, 0.0f);
	layernorm_backward(d_ln2_out.data(), l.mv_ln2_xhat, l.mv_ln2_var,
		l.mv_ln2_gamma, d_ln2_in.data(), l.mv_ln2_gamma_updates, l.mv_ln2_beta_updates, M, N);
	for (int i = 0; i < M * N; ++i) d_pre_res2[i] += d_ln2_in[i];

	std::vector<float> d_tokens(M * C, 0.0f);
	if (C == N)
	{
		for (int i = 0; i < M * C; ++i) d_tokens[i] += d_pre_res2[i];
	}
	else
	{
		gemm_cpu(1, 0, N, C, M, 1.0f, d_pre_res2.data(), N, l.mv_tokens, C, 1.0f, l.mv_res_proj_updates, C);
		gemm_cpu(0, 0, M, C, N, 1.0f, d_pre_res2.data(), N, l.mv_res_proj, C, 0.0f, d_tokens.data(), C);
	}

	std::vector<float> d_cat(M * N, 0.0f);
	gemm_cpu(1, 0, N, N, M, 1.0f, d_pre_res2.data(), N, l.mv_mixer_cat, N, 1.0f, l.mv_out_proj_updates, N);
	for (int i = 0; i < M; ++i)
		for (int j = 0; j < N; ++j)
			l.mv_out_bias_updates[j] += d_pre_res2[static_cast<size_t>(i) * N + j];
	gemm_cpu(0, 0, M, N, N, 1.0f, d_pre_res2.data(), N, l.mv_out_proj, N, 0.0f, d_cat.data(), N);

	std::vector<float> d_scan(B * D * T, 0.0f);
	std::vector<float> d_xconv(B * D * T, 0.0f);
	std::vector<float> d_zconv(B * D * T, 0.0f);
	for (int b = 0; b < B; ++b)
		for (int t = 0; t < T; ++t)
		{
			const float *row = d_cat.data() + (static_cast<size_t>(b) * T + t) * N;
			for (int d = 0; d < D; ++d)
			{
				d_scan[bdt_idx(b, d, t, D, T)] += row[d];
				d_zconv[bdt_idx(b, d, t, D, T)] += row[D + d];
			}
		}

	std::vector<float> d_xproj(M * P, 0.0f);
	std::vector<float> ddt(B * D * T, 0.0f);
	scan_backward(l, d_scan.data(), d_xconv.data(), d_xproj.data(), ddt.data(), B, T, D, R, S);
	dt_proj_backward(l, ddt.data(), d_xproj.data(), B, T, D, R, S);
	x_proj_backward(l.mv_x_conv, d_xproj.data(), d_xconv.data(), l.mv_x_proj, l.mv_x_proj_updates, B, T, D, P);

	std::vector<float> d_in_proj(M * N, 0.0f);
	depthwise_conv1d_backward_to_proj(l.mv_in_proj_out, 0, l.mv_x_conv_pre, d_xconv.data(), d_in_proj.data(),
		l.mv_conv_x, l.mv_conv_x_updates, l.mv_conv_x_bias_updates, B, T, N, D, l.mv_conv_size);
	depthwise_conv1d_backward_to_proj(l.mv_in_proj_out, D, l.mv_z_conv_pre, d_zconv.data(), d_in_proj.data(),
		l.mv_conv_z, l.mv_conv_z_updates, l.mv_conv_z_bias_updates, B, T, N, D, l.mv_conv_size);

	std::vector<float> d_ln1_out(M * C, 0.0f);
	gemm_cpu(1, 0, N, C, M, 1.0f, d_in_proj.data(), N, l.mv_ln1_out, C, 1.0f, l.weight_updates, C);
	for (int i = 0; i < M; ++i)
		for (int j = 0; j < N; ++j)
			l.bias_updates[j] += d_in_proj[static_cast<size_t>(i) * N + j];
	gemm_cpu(0, 0, M, C, N, 1.0f, d_in_proj.data(), N, l.weights, C, 0.0f, d_ln1_out.data(), C);

	std::vector<float> d_ln1_in(M * C, 0.0f);
	layernorm_backward(d_ln1_out.data(), l.mv_ln1_xhat, l.mv_ln1_var,
		l.mv_ln1_gamma, d_ln1_in.data(), l.mv_ln1_gamma_updates, l.mv_ln1_beta_updates, M, C);
	for (int i = 0; i < M * C; ++i) d_tokens[i] += d_ln1_in[i];

	if (state.delta)
	{
		for (int b = 0; b < B; ++b)
			for (int c = 0; c < C; ++c)
				for (int t = 0; t < T; ++t)
					state.delta[(static_cast<size_t>(b) * C + c) * T + t] += d_tokens[(static_cast<size_t>(b) * T + t) * C + c];
	}
}

void update_mambavision_layer(Darknet::Layer & l, int batch, float learning_rate_init, float momentum, float decay)
{
	TAT(TATPARMS);

	const float lr = learning_rate_init * l.learning_rate_scale;
	const int C = l.c;
	const int N = l.n;
	const int D = N / 2;
	const int R = l.mv_dt_rank;
	const int S = l.mv_d_state;
	const int P = R + 2 * S;
	const int ffn_hidden = N * l.mv_ffn_ratio;

	update_param(l.weights, l.weight_updates, l.nweights, batch, lr, momentum, decay, true);
	update_param(l.biases, l.bias_updates, l.nbiases, batch, lr, momentum, decay, false);
	update_param(l.mv_conv_x, l.mv_conv_x_updates, D * l.mv_conv_size, batch, lr, momentum, decay, true);
	update_param(l.mv_conv_x_bias, l.mv_conv_x_bias_updates, D, batch, lr, momentum, decay, false);
	update_param(l.mv_conv_z, l.mv_conv_z_updates, D * l.mv_conv_size, batch, lr, momentum, decay, true);
	update_param(l.mv_conv_z_bias, l.mv_conv_z_bias_updates, D, batch, lr, momentum, decay, false);
	update_param(l.mv_x_proj, l.mv_x_proj_updates, P * D, batch, lr, momentum, decay, true);
	update_param(l.mv_dt_proj, l.mv_dt_proj_updates, D * R, batch, lr, momentum, decay, true);
	update_param(l.mv_dt_bias, l.mv_dt_bias_updates, D, batch, lr, momentum, decay, false);
	update_param(l.mv_A_log, l.mv_A_log_updates, D * S, batch, lr, momentum, 0.0f, false);
	update_param(l.mv_D, l.mv_D_updates, D, batch, lr, momentum, 0.0f, false);
	update_param(l.mv_out_proj, l.mv_out_proj_updates, N * N, batch, lr, momentum, decay, true);
	update_param(l.mv_out_bias, l.mv_out_bias_updates, N, batch, lr, momentum, decay, false);
	if (C != N) update_param(l.mv_res_proj, l.mv_res_proj_updates, N * C, batch, lr, momentum, decay, true);
	update_param(l.mv_ln1_gamma, l.mv_ln1_gamma_updates, C, batch, lr, momentum, decay, false);
	update_param(l.mv_ln1_beta, l.mv_ln1_beta_updates, C, batch, lr, momentum, decay, false);
	update_param(l.mv_ln2_gamma, l.mv_ln2_gamma_updates, N, batch, lr, momentum, decay, false);
	update_param(l.mv_ln2_beta, l.mv_ln2_beta_updates, N, batch, lr, momentum, decay, false);
	update_param(l.mv_ffn_w1, l.mv_ffn_w1_updates, ffn_hidden * N, batch, lr, momentum, decay, true);
	update_param(l.mv_ffn_b1, l.mv_ffn_b1_updates, ffn_hidden, batch, lr, momentum, decay, false);
	update_param(l.mv_ffn_w2, l.mv_ffn_w2_updates, N * ffn_hidden, batch, lr, momentum, decay, true);
	update_param(l.mv_ffn_b2, l.mv_ffn_b2_updates, N, batch, lr, momentum, decay, false);
}

#ifdef DARKNET_GPU
void forward_mambavision_layer_gpu(Darknet::Layer & l, Darknet::NetworkState state)
{
	mambavision_forward_gpu_impl(l, state);
}

void backward_mambavision_layer_gpu(Darknet::Layer & l, Darknet::NetworkState state)
{
	mambavision_backward_gpu_impl(l, state);
}

void update_mambavision_layer_gpu(Darknet::Layer & l, int batch, float learning_rate, float momentum, float decay, float loss_scale)
{
	const float lr = learning_rate * l.learning_rate_scale;
	const int D = l.n / 2;
	const int P = l.mv_dt_rank + 2 * l.mv_d_state;

	update_connected_layer_gpu(*l.mv_in_proj_layer, batch, learning_rate, momentum, decay, loss_scale);
	update_convolutional_layer_gpu(*l.mv_conv_x_layer, batch, learning_rate, momentum, decay, loss_scale);
	update_convolutional_layer_gpu(*l.mv_conv_z_layer, batch, learning_rate, momentum, decay, loss_scale);
	update_connected_layer_gpu(*l.mv_x_proj_layer, batch, learning_rate, momentum, decay, loss_scale);
	update_connected_layer_gpu(*l.mv_dt_proj_layer, batch, learning_rate, momentum, decay, loss_scale);
	update_connected_layer_gpu(*l.mv_out_proj_layer, batch, learning_rate, momentum, decay, loss_scale);
	if (l.mv_res_proj_layer) update_connected_layer_gpu(*l.mv_res_proj_layer, batch, learning_rate, momentum, decay, loss_scale);
	update_connected_layer_gpu(*l.mv_ffn1_layer, batch, learning_rate, momentum, decay, loss_scale);
	update_connected_layer_gpu(*l.mv_ffn2_layer, batch, learning_rate, momentum, decay, loss_scale);

	fill_ongpu(P, 0.0f, l.mv_x_proj_layer->biases_gpu, 1);
	fill_ongpu(P, 0.0f, l.mv_x_proj_layer->bias_updates_gpu, 1);
	if (l.mv_res_proj_layer)
	{
		fill_ongpu(l.n, 0.0f, l.mv_res_proj_layer->biases_gpu, 1);
		fill_ongpu(l.n, 0.0f, l.mv_res_proj_layer->bias_updates_gpu, 1);
	}

	if (loss_scale != 1.0f)
	{
		scal_ongpu(l.c, 1.0f / loss_scale, l.mv_ln1_gamma_updates_gpu, 1);
		scal_ongpu(l.c, 1.0f / loss_scale, l.mv_ln1_beta_updates_gpu, 1);
		scal_ongpu(l.n, 1.0f / loss_scale, l.mv_ln2_gamma_updates_gpu, 1);
		scal_ongpu(l.n, 1.0f / loss_scale, l.mv_ln2_beta_updates_gpu, 1);
		scal_ongpu(D * l.mv_d_state, 1.0f / loss_scale, l.mv_A_log_updates_gpu, 1);
		scal_ongpu(D, 1.0f / loss_scale, l.mv_D_updates_gpu, 1);
	}

	axpy_ongpu(l.c, lr / batch, l.mv_ln1_gamma_updates_gpu, 1, l.mv_ln1_gamma_gpu, 1);
	scal_ongpu(l.c, momentum, l.mv_ln1_gamma_updates_gpu, 1);
	axpy_ongpu(l.c, lr / batch, l.mv_ln1_beta_updates_gpu, 1, l.mv_ln1_beta_gpu, 1);
	scal_ongpu(l.c, momentum, l.mv_ln1_beta_updates_gpu, 1);
	axpy_ongpu(l.n, lr / batch, l.mv_ln2_gamma_updates_gpu, 1, l.mv_ln2_gamma_gpu, 1);
	scal_ongpu(l.n, momentum, l.mv_ln2_gamma_updates_gpu, 1);
	axpy_ongpu(l.n, lr / batch, l.mv_ln2_beta_updates_gpu, 1, l.mv_ln2_beta_gpu, 1);
	scal_ongpu(l.n, momentum, l.mv_ln2_beta_updates_gpu, 1);
	axpy_ongpu(D * l.mv_d_state, lr / batch, l.mv_A_log_updates_gpu, 1, l.mv_A_log_gpu, 1);
	scal_ongpu(D * l.mv_d_state, momentum, l.mv_A_log_updates_gpu, 1);
	axpy_ongpu(D, lr / batch, l.mv_D_updates_gpu, 1, l.mv_D_gpu, 1);
	scal_ongpu(D, momentum, l.mv_D_updates_gpu, 1);
}

void push_mambavision_layer(Darknet::Layer & l)
{
	const int D = l.n / 2;
	const int P = l.mv_dt_rank + 2 * l.mv_d_state;
	const int ffn_hidden = l.n * l.mv_ffn_ratio;

	copy_connected_params_to_layer(l.mv_in_proj_layer, l.weights, l.biases, l.n * l.c, l.n);
	copy_depthwise_1d_to_conv(l.mv_conv_x_layer, l.mv_conv_x, l.mv_conv_x_bias, D, l.mv_conv_size);
	copy_depthwise_1d_to_conv(l.mv_conv_z_layer, l.mv_conv_z, l.mv_conv_z_bias, D, l.mv_conv_size);
	copy_connected_params_to_layer(l.mv_x_proj_layer, l.mv_x_proj, nullptr, P * D, P);
	copy_connected_params_to_layer(l.mv_dt_proj_layer, l.mv_dt_proj, l.mv_dt_bias, D * l.mv_dt_rank, D);
	copy_connected_params_to_layer(l.mv_out_proj_layer, l.mv_out_proj, l.mv_out_bias, l.n * l.n, l.n);
	if (l.mv_res_proj_layer) copy_connected_params_to_layer(l.mv_res_proj_layer, l.mv_res_proj, nullptr, l.n * l.c, l.n);
	copy_connected_params_to_layer(l.mv_ffn1_layer, l.mv_ffn_w1, l.mv_ffn_b1, ffn_hidden * l.n, ffn_hidden);
	copy_connected_params_to_layer(l.mv_ffn2_layer, l.mv_ffn_w2, l.mv_ffn_b2, l.n * ffn_hidden, l.n);

	push_connected_layer(*l.mv_in_proj_layer);
	push_convolutional_layer(*l.mv_conv_x_layer);
	push_convolutional_layer(*l.mv_conv_z_layer);
	push_connected_layer(*l.mv_x_proj_layer);
	push_connected_layer(*l.mv_dt_proj_layer);
	push_connected_layer(*l.mv_out_proj_layer);
	if (l.mv_res_proj_layer) push_connected_layer(*l.mv_res_proj_layer);
	push_connected_layer(*l.mv_ffn1_layer);
	push_connected_layer(*l.mv_ffn2_layer);
	cuda_push_array(l.mv_ln1_gamma_gpu, l.mv_ln1_gamma, l.c);
	cuda_push_array(l.mv_ln1_gamma_updates_gpu, l.mv_ln1_gamma_updates, l.c);
	cuda_push_array(l.mv_ln1_beta_gpu, l.mv_ln1_beta, l.c);
	cuda_push_array(l.mv_ln1_beta_updates_gpu, l.mv_ln1_beta_updates, l.c);
	cuda_push_array(l.mv_ln2_gamma_gpu, l.mv_ln2_gamma, l.n);
	cuda_push_array(l.mv_ln2_gamma_updates_gpu, l.mv_ln2_gamma_updates, l.n);
	cuda_push_array(l.mv_ln2_beta_gpu, l.mv_ln2_beta, l.n);
	cuda_push_array(l.mv_ln2_beta_updates_gpu, l.mv_ln2_beta_updates, l.n);
	cuda_push_array(l.mv_A_log_gpu, l.mv_A_log, D * l.mv_d_state);
	cuda_push_array(l.mv_A_log_updates_gpu, l.mv_A_log_updates, D * l.mv_d_state);
	cuda_push_array(l.mv_D_gpu, l.mv_D, D);
	cuda_push_array(l.mv_D_updates_gpu, l.mv_D_updates, D);
}

void pull_mambavision_layer(Darknet::Layer & l)
{
	const int D = l.n / 2;
	const int P = l.mv_dt_rank + 2 * l.mv_d_state;
	const int ffn_hidden = l.n * l.mv_ffn_ratio;

	pull_connected_layer(*l.mv_in_proj_layer);
	pull_convolutional_layer(*l.mv_conv_x_layer);
	pull_convolutional_layer(*l.mv_conv_z_layer);
	pull_connected_layer(*l.mv_x_proj_layer);
	pull_connected_layer(*l.mv_dt_proj_layer);
	pull_connected_layer(*l.mv_out_proj_layer);
	if (l.mv_res_proj_layer) pull_connected_layer(*l.mv_res_proj_layer);
	pull_connected_layer(*l.mv_ffn1_layer);
	pull_connected_layer(*l.mv_ffn2_layer);

	copy_connected_params_from_layer(l.mv_in_proj_layer, l.weights, l.weight_updates, l.biases, l.bias_updates, l.n * l.c, l.n);
	copy_depthwise_1d_from_conv(l.mv_conv_x_layer, l.mv_conv_x, l.mv_conv_x_updates, l.mv_conv_x_bias, l.mv_conv_x_bias_updates, D, l.mv_conv_size);
	copy_depthwise_1d_from_conv(l.mv_conv_z_layer, l.mv_conv_z, l.mv_conv_z_updates, l.mv_conv_z_bias, l.mv_conv_z_bias_updates, D, l.mv_conv_size);
	copy_connected_params_from_layer(l.mv_x_proj_layer, l.mv_x_proj, l.mv_x_proj_updates, nullptr, nullptr, P * D, P);
	copy_connected_params_from_layer(l.mv_dt_proj_layer, l.mv_dt_proj, l.mv_dt_proj_updates, l.mv_dt_bias, l.mv_dt_bias_updates, D * l.mv_dt_rank, D);
	copy_connected_params_from_layer(l.mv_out_proj_layer, l.mv_out_proj, l.mv_out_proj_updates, l.mv_out_bias, l.mv_out_bias_updates, l.n * l.n, l.n);
	if (l.mv_res_proj_layer) copy_connected_params_from_layer(l.mv_res_proj_layer, l.mv_res_proj, l.mv_res_proj_updates, nullptr, nullptr, l.n * l.c, l.n);
	copy_connected_params_from_layer(l.mv_ffn1_layer, l.mv_ffn_w1, l.mv_ffn_w1_updates, l.mv_ffn_b1, l.mv_ffn_b1_updates, ffn_hidden * l.n, ffn_hidden);
	copy_connected_params_from_layer(l.mv_ffn2_layer, l.mv_ffn_w2, l.mv_ffn_w2_updates, l.mv_ffn_b2, l.mv_ffn_b2_updates, l.n * ffn_hidden, l.n);
	cuda_pull_array(l.mv_ln1_gamma_gpu, l.mv_ln1_gamma, l.c);
	cuda_pull_array(l.mv_ln1_gamma_updates_gpu, l.mv_ln1_gamma_updates, l.c);
	cuda_pull_array(l.mv_ln1_beta_gpu, l.mv_ln1_beta, l.c);
	cuda_pull_array(l.mv_ln1_beta_updates_gpu, l.mv_ln1_beta_updates, l.c);
	cuda_pull_array(l.mv_ln2_gamma_gpu, l.mv_ln2_gamma, l.n);
	cuda_pull_array(l.mv_ln2_gamma_updates_gpu, l.mv_ln2_gamma_updates, l.n);
	cuda_pull_array(l.mv_ln2_beta_gpu, l.mv_ln2_beta, l.n);
	cuda_pull_array(l.mv_ln2_beta_updates_gpu, l.mv_ln2_beta_updates, l.n);
	cuda_pull_array(l.mv_A_log_gpu, l.mv_A_log, D * l.mv_d_state);
	cuda_pull_array(l.mv_A_log_updates_gpu, l.mv_A_log_updates, D * l.mv_d_state);
	cuda_pull_array(l.mv_D_gpu, l.mv_D, D);
	cuda_pull_array(l.mv_D_updates_gpu, l.mv_D_updates, D);
}
#endif
