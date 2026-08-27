#include "wmhf_layer.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>
#include <vector>

// Some Darknet trees expose these through convolutional_layer.hpp; this fork's
// layer sources include darknet_internal.hpp directly, so keeping local
// declarations makes this file easy to drop into src/.
Darknet::Layer make_convolutional_layer(int batch, int steps, int h, int w, int c, int n, int groups, int size, int stride_x, int stride_y, int dilation, int padding, ACTIVATION activation, int batch_normalize, int binary, int xnor, int adam, int use_bin_output, int index, int antialiasing, Darknet::Layer *share_layer, int assisted_excitation, int deform, int train);
void resize_convolutional_layer(Darknet::Layer * l, int w, int h);
void forward_convolutional_layer(Darknet::Layer & l, Darknet::NetworkState state);
void backward_convolutional_layer(Darknet::Layer & l, Darknet::NetworkState state);
void update_convolutional_layer(Darknet::Layer & l, int batch, float learning_rate, float momentum, float decay);
size_t get_convolutional_workspace_size(const Darknet::Layer & l);
int convolutional_out_height(const Darknet::Layer & l);
int convolutional_out_width(const Darknet::Layer & l);
#ifdef DARKNET_GPU
void forward_convolutional_layer_gpu(Darknet::Layer & l, Darknet::NetworkState state);
void backward_convolutional_layer_gpu(Darknet::Layer & l, Darknet::NetworkState state);
void update_convolutional_layer_gpu(Darknet::Layer & l, int batch, float learning_rate, float momentum, float decay, float loss_scale);
#endif

namespace
{
	static auto & cfg_and_state = Darknet::CfgAndState::get();

	enum WMHFSubLayer
	{
		WMHF_PRE = 0,
		WMHF_LOCAL_DW3,
		WMHF_LOCAL_DW5,
		WMHF_LOCAL_DW7,
		WMHF_LOCAL_MIX,
		WMHF_HF_GATE,
		WMHF_FUSE,
		WMHF_SUB_COUNT
	};

	struct WMHFCounts
	{
		int filters = 0;
		int id = 0;
		int local = 0;
		int global = 0;
	};

	inline int spatial(const int h, const int w)
	{
		return h * w;
	}

	inline int offset4(const int b, const int c, const int s, const int channels, const int area)
	{
		return (b * channels + c) * area + s;
	}

	inline int offset4xy(const int b, const int c, const int y, const int x, const int channels, const int h, const int w)
	{
		(void)h;
		return (b * channels + c) * (h * w) + y * w + x;
	}

	inline float stable_sigmoid(const float x)
	{
		if (x >= 0.0f)
		{
			const float z = std::exp(-x);
			return 1.0f / (1.0f + z);
		}
		const float z = std::exp(x);
		return z / (1.0f + z);
	}

	inline float sign_no_zero(const float x)
	{
		return (x > 0.0f) ? 1.0f : ((x < 0.0f) ? -1.0f : 0.0f);
	}

	inline WMHFCounts get_counts(const Darknet::Layer & l)
	{
		WMHFCounts c;
		c.filters = l.out_c;
		c.id = l.groups;
		c.local = l.group_id;
		c.global = l.out_c - c.id - c.local;
		return c;
	}

	inline WMHFCounts choose_counts(const int filters, const float identity_ratio, const float local_ratio)
	{
		if (filters < 4)
		{
			darknet_fatal_error(DARKNET_LOC, "[wmhf] filters must be >= 4, got %d", filters);
		}

		WMHFCounts c;
		c.filters = filters;
		c.id = std::max(0, static_cast<int>(std::round(filters * identity_ratio)));
		c.local = std::max(1, static_cast<int>(std::round(filters * local_ratio)));
		if (c.id + c.local > filters - 1)
		{
			c.local = std::max(1, filters - c.id - 1);
		}
		c.global = filters - c.id - c.local;
		if (c.global < 1)
		{
			c.global = 1;
			if (c.local > 1)
			{
				--c.local;
			}
			else if (c.id > 0)
			{
				--c.id;
			}
		}
		if (c.id + c.local + c.global != filters || c.local < 1 || c.global < 1)
		{
			darknet_fatal_error(DARKNET_LOC, "[wmhf] invalid split filters=%d id=%d local=%d global=%d", filters, c.id, c.local, c.global);
		}
		return c;
	}

	inline Darknet::Layer & sub(Darknet::Layer & l, const int idx)
	{
		return l.input_layer[idx];
	}

	inline const Darknet::Layer & sub(const Darknet::Layer & l, const int idx)
	{
		return l.input_layer[idx];
	}

	inline int scan_param_count(const int global_channels)
	{
		return 4 * global_channels;
	}

	inline void init_scan_weights(float * weights, const int channels)
	{
		for (int c = 0; c < channels; ++c)
		{
			weights[0 * channels + c] = 0.0f; // raw A -> 0.49 after 0.98*sigmoid
			weights[1 * channels + c] = 1.0f; // B
			weights[2 * channels + c] = 1.0f; // C
			weights[3 * channels + c] = 0.0f; // D
		}
	}

	void resize_convolutional_layer_cpu_only(Darknet::Layer * l, const int w, const int h)
	{
		const int total_batch = l->batch * std::max(1, l->steps);

		l->w = w;
		l->h = h;
		l->out_w = convolutional_out_width(*l);
		l->out_h = convolutional_out_height(*l);
		l->outputs = l->out_h * l->out_w * l->out_c;
		l->inputs = l->w * l->h * l->c;

		l->output = (float*)xrealloc(l->output, static_cast<size_t>(total_batch) * l->outputs * sizeof(float));
		if (l->train)
		{
			l->delta = (float*)xrealloc(l->delta, static_cast<size_t>(total_batch) * l->outputs * sizeof(float));
			if (l->batch_normalize)
			{
				l->x = (float*)xrealloc(l->x, static_cast<size_t>(total_batch) * l->outputs * sizeof(float));
				l->x_norm = (float*)xrealloc(l->x_norm, static_cast<size_t>(total_batch) * l->outputs * sizeof(float));
			}
		}
		if (l->activation == SWISH || l->activation == MISH || l->activation == HARD_MISH || l->activation == EML)
		{
			l->activation_input = (float*)xrealloc(l->activation_input, static_cast<size_t>(total_batch) * l->outputs * sizeof(float));
		}
		l->workspace_size = get_convolutional_workspace_size(*l);
	}

	inline void resize_wmhf_sub_layer(Darknet::Layer * l, const int w, const int h)
	{
#ifdef DARKNET_GPU
		if (cfg_and_state.gpu_index >= 0)
		{
			resize_convolutional_layer(l, w, h);
			return;
		}
#endif
		resize_convolutional_layer_cpu_only(l, w, h);
	}

	inline void extract_channels_cpu(const float * input, float * output, const int batch, const int in_c, const int out_c, const int begin_c, const int area)
	{
		for (int b = 0; b < batch; ++b)
		{
			for (int c = 0; c < out_c; ++c)
			{
				const float * src = input + offset4(b, begin_c + c, 0, in_c, area);
				float * dst = output + offset4(b, c, 0, out_c, area);
				std::memcpy(dst, src, area * sizeof(float));
			}
		}
	}

	inline void insert_channels_cpu(const float * input, float * output, const int batch, const int out_c, const int in_c, const int begin_c, const int area, const float scale = 1.0f)
	{
		for (int b = 0; b < batch; ++b)
		{
			for (int c = 0; c < in_c; ++c)
			{
				const float * src = input + offset4(b, c, 0, in_c, area);
				float * dst = output + offset4(b, begin_c + c, 0, out_c, area);
				for (int i = 0; i < area; ++i)
				{
					dst[i] += scale * src[i];
				}
			}
		}
	}

	inline void local_concat_cpu(const float * a, const float * b, const float * c, float * out, const int batch, const int channels, const int area)
	{
		const int branch_size = channels * area;
		for (int n = 0; n < batch; ++n)
		{
			std::memcpy(out + n * 3 * branch_size + 0 * branch_size, a + n * branch_size, branch_size * sizeof(float));
			std::memcpy(out + n * 3 * branch_size + 1 * branch_size, b + n * branch_size, branch_size * sizeof(float));
			std::memcpy(out + n * 3 * branch_size + 2 * branch_size, c + n * branch_size, branch_size * sizeof(float));
		}
	}

	inline void local_concat_backward_cpu(const float * cat_delta, float * da, float * db, float * dc, const int batch, const int channels, const int area)
	{
		const int branch_size = channels * area;
		for (int n = 0; n < batch; ++n)
		{
			std::memcpy(da + n * branch_size, cat_delta + n * 3 * branch_size + 0 * branch_size, branch_size * sizeof(float));
			std::memcpy(db + n * branch_size, cat_delta + n * 3 * branch_size + 1 * branch_size, branch_size * sizeof(float));
			std::memcpy(dc + n * branch_size, cat_delta + n * 3 * branch_size + 2 * branch_size, branch_size * sizeof(float));
		}
	}

	inline void fuse_concat_cpu(const float * projected, const float * local, const float * global, float * out, const int batch, const int id_c, const int local_c, const int global_c, const int area)
	{
		const int filters = id_c + local_c + global_c;
		for (int b = 0; b < batch; ++b)
		{
			if (id_c > 0)
			{
				std::memcpy(out + offset4(b, 0, 0, filters, area), projected + offset4(b, 0, 0, filters, area), id_c * area * sizeof(float));
			}
			std::memcpy(out + offset4(b, id_c, 0, filters, area), local + offset4(b, 0, 0, local_c, area), local_c * area * sizeof(float));
			std::memcpy(out + offset4(b, id_c + local_c, 0, filters, area), global + offset4(b, 0, 0, global_c, area), global_c * area * sizeof(float));
		}
	}

	inline void fuse_concat_backward_cpu(const float * cat_delta, float * projected_delta, float * local_delta, float * global_delta, const int batch, const int id_c, const int local_c, const int global_c, const int area)
	{
		const int filters = id_c + local_c + global_c;
		for (int b = 0; b < batch; ++b)
		{
			if (id_c > 0)
			{
				insert_channels_cpu(cat_delta + b * filters * area, projected_delta + b * filters * area, 1, filters, id_c, 0, area, 1.0f);
			}
			std::memcpy(local_delta + offset4(b, 0, 0, local_c, area), cat_delta + offset4(b, id_c, 0, filters, area), local_c * area * sizeof(float));
			std::memcpy(global_delta + offset4(b, 0, 0, global_c, area), cat_delta + offset4(b, id_c + local_c, 0, filters, area), global_c * area * sizeof(float));
		}
	}

	void dwt_cpu(const float * input, float * ll, float * lh, float * hl, float * hh, const int batch, const int channels, const int h, const int w, const int h2, const int w2)
	{
		const int area = h * w;
		const int area2 = h2 * w2;
		for (int b = 0; b < batch; ++b)
		{
			for (int c = 0; c < channels; ++c)
			{
				for (int y = 0; y < h2; ++y)
				{
					const int y0 = std::min(2 * y, h - 1);
					const int y1 = std::min(y0 + 1, h - 1);
					for (int x = 0; x < w2; ++x)
					{
						const int x0 = std::min(2 * x, w - 1);
						const int x1 = std::min(x0 + 1, w - 1);
						const float v00 = input[offset4(b, c, y0 * w + x0, channels, area)];
						const float v01 = input[offset4(b, c, y0 * w + x1, channels, area)];
						const float v10 = input[offset4(b, c, y1 * w + x0, channels, area)];
						const float v11 = input[offset4(b, c, y1 * w + x1, channels, area)];
						const int o = offset4(b, c, y * w2 + x, channels, area2);
						ll[o] = 0.5f * (v00 + v01 + v10 + v11);
						lh[o] = 0.5f * (v00 - v01 + v10 - v11);
						hl[o] = 0.5f * (v00 + v01 - v10 - v11);
						hh[o] = 0.5f * (v00 - v01 - v10 + v11);
					}
				}
			}
		}
	}

	void dwt_backward_cpu(const float * dll, const float * dlh, const float * dhl, const float * dhh, float * input_delta, const int batch, const int channels, const int h, const int w, const int h2, const int w2)
	{
		const int area = h * w;
		const int area2 = h2 * w2;
		for (int b = 0; b < batch; ++b)
		{
			for (int c = 0; c < channels; ++c)
			{
				for (int y = 0; y < h2; ++y)
				{
					const int y0 = std::min(2 * y, h - 1);
					const int y1 = std::min(y0 + 1, h - 1);
					for (int x = 0; x < w2; ++x)
					{
						const int x0 = std::min(2 * x, w - 1);
						const int x1 = std::min(x0 + 1, w - 1);
						const int o = offset4(b, c, y * w2 + x, channels, area2);
						const float a = 0.5f * dll[o];
						const float b1 = 0.5f * dlh[o];
						const float c1 = 0.5f * dhl[o];
						const float d = 0.5f * dhh[o];
						input_delta[offset4(b, c, y0 * w + x0, channels, area)] += a + b1 + c1 + d;
						input_delta[offset4(b, c, y0 * w + x1, channels, area)] += a - b1 + c1 - d;
						input_delta[offset4(b, c, y1 * w + x0, channels, area)] += a + b1 - c1 - d;
						input_delta[offset4(b, c, y1 * w + x1, channels, area)] += a - b1 - c1 + d;
					}
				}
			}
		}
	}

	void iwt_cpu(const float * ll, const float * lh, const float * hl, const float * hh, float * output, const int batch, const int channels, const int h, const int w, const int h2, const int w2)
	{
		const int area = h * w;
		const int area2 = h2 * w2;
		std::fill(output, output + batch * channels * area, 0.0f);
		for (int b = 0; b < batch; ++b)
		{
			for (int c = 0; c < channels; ++c)
			{
				for (int y = 0; y < h2; ++y)
				{
					const int y0 = std::min(2 * y, h - 1);
					const int y1 = std::min(y0 + 1, h - 1);
					for (int x = 0; x < w2; ++x)
					{
						const int x0 = std::min(2 * x, w - 1);
						const int x1 = std::min(x0 + 1, w - 1);
						const int o = offset4(b, c, y * w2 + x, channels, area2);
						const float L = ll[o];
						const float H1 = lh[o];
						const float H2 = hl[o];
						const float H3 = hh[o];
						output[offset4(b, c, y0 * w + x0, channels, area)] += 0.5f * (L + H1 + H2 + H3);
						output[offset4(b, c, y0 * w + x1, channels, area)] += 0.5f * (L - H1 + H2 - H3);
						output[offset4(b, c, y1 * w + x0, channels, area)] += 0.5f * (L + H1 - H2 - H3);
						output[offset4(b, c, y1 * w + x1, channels, area)] += 0.5f * (L - H1 - H2 + H3);
					}
				}
			}
		}
	}

	void iwt_backward_cpu(const float * output_delta, float * dll, float * dlh, float * dhl, float * dhh, const int batch, const int channels, const int h, const int w, const int h2, const int w2)
	{
		const int area = h * w;
		const int area2 = h2 * w2;
		std::fill(dll, dll + batch * channels * area2, 0.0f);
		std::fill(dlh, dlh + batch * channels * area2, 0.0f);
		std::fill(dhl, dhl + batch * channels * area2, 0.0f);
		std::fill(dhh, dhh + batch * channels * area2, 0.0f);
		for (int b = 0; b < batch; ++b)
		{
			for (int c = 0; c < channels; ++c)
			{
				for (int y = 0; y < h2; ++y)
				{
					const int y0 = std::min(2 * y, h - 1);
					const int y1 = std::min(y0 + 1, h - 1);
					for (int x = 0; x < w2; ++x)
					{
						const int x0 = std::min(2 * x, w - 1);
						const int x1 = std::min(x0 + 1, w - 1);
						const int o = offset4(b, c, y * w2 + x, channels, area2);
						const float d00 = output_delta[offset4(b, c, y0 * w + x0, channels, area)];
						const float d01 = output_delta[offset4(b, c, y0 * w + x1, channels, area)];
						const float d10 = output_delta[offset4(b, c, y1 * w + x0, channels, area)];
						const float d11 = output_delta[offset4(b, c, y1 * w + x1, channels, area)];
						dll[o] += 0.5f * (d00 + d01 + d10 + d11);
						dlh[o] += 0.5f * (d00 - d01 + d10 - d11);
						dhl[o] += 0.5f * (d00 + d01 - d10 - d11);
						dhh[o] += 0.5f * (d00 - d01 - d10 + d11);
					}
				}
			}
		}
	}

	void scan_sequence_forward_cpu(
		const float * input,
		float * output,
		const float * weights,
		const int batch,
		const int channels,
		const int h,
		const int w,
		const int b,
		const int c,
		const std::vector<int> & idx)
	{
		const int area = h * w;
		const float raw_a = weights[0 * channels + c];
		const float A = 0.98f * stable_sigmoid(raw_a);
		const float B = weights[1 * channels + c];
		const float C = weights[2 * channels + c];
		const float D = weights[3 * channels + c];
		float s = 0.0f;
		for (int pos : idx)
		{
			const int o = offset4(b, c, pos, channels, area);
			const float x = input[o];
			s = A * s + B * x;
			output[o] += 0.25f * (C * s + D * x);
		}
	}

	void scan4_forward_cpu(const float * input, const float * weights, float * output, const int batch, const int channels, const int h, const int w)
	{
		const int area = h * w;
		std::fill(output, output + batch * channels * area, 0.0f);
		std::vector<int> idx;
		idx.reserve(std::max(h, w));
		for (int b = 0; b < batch; ++b)
		{
			for (int c = 0; c < channels; ++c)
			{
				for (int y = 0; y < h; ++y)
				{
					idx.clear();
					for (int x = 0; x < w; ++x) idx.push_back(y * w + x);
					scan_sequence_forward_cpu(input, output, weights, batch, channels, h, w, b, c, idx);
					idx.clear();
					for (int x = w - 1; x >= 0; --x) idx.push_back(y * w + x);
					scan_sequence_forward_cpu(input, output, weights, batch, channels, h, w, b, c, idx);
				}
				for (int x = 0; x < w; ++x)
				{
					idx.clear();
					for (int y = 0; y < h; ++y) idx.push_back(y * w + x);
					scan_sequence_forward_cpu(input, output, weights, batch, channels, h, w, b, c, idx);
					idx.clear();
					for (int y = h - 1; y >= 0; --y) idx.push_back(y * w + x);
					scan_sequence_forward_cpu(input, output, weights, batch, channels, h, w, b, c, idx);
				}
			}
		}
	}

	void scan_sequence_backward_cpu(
		const float * input,
		const float * output_delta,
		const float * weights,
		float * weight_updates,
		float * input_delta,
		const int batch,
		const int channels,
		const int h,
		const int w,
		const int b,
		const int c,
		const std::vector<int> & idx)
	{
		const int area = h * w;
		const int L = static_cast<int>(idx.size());
		if (L == 0) return;
		const float raw_a = weights[0 * channels + c];
		const float sig = stable_sigmoid(raw_a);
		const float A = 0.98f * sig;
		const float B = weights[1 * channels + c];
		const float C = weights[2 * channels + c];
		const float D = weights[3 * channels + c];
		std::vector<float> s(L, 0.0f);
		std::vector<float> xval(L, 0.0f);
		float state = 0.0f;
		for (int t = 0; t < L; ++t)
		{
			const int o = offset4(b, c, idx[t], channels, area);
			const float x = input[o];
			xval[t] = x;
			state = A * state + B * x;
			s[t] = state;
		}

		float dA = 0.0f;
		float dB = 0.0f;
		float dC = 0.0f;
		float dD = 0.0f;
		float ds_next = 0.0f;
		for (int t = L - 1; t >= 0; --t)
		{
			const int o = offset4(b, c, idx[t], channels, area);
			const float dy = 0.25f * output_delta[o];
			const float x = xval[t];
			const float s_t = s[t];
			const float s_prev = (t > 0) ? s[t - 1] : 0.0f;
			dC += dy * s_t;
			dD += dy * x;
			float ds = ds_next + dy * C;
			dA += ds * s_prev;
			dB += ds * x;
			input_delta[o] += dy * D + ds * B;
			ds_next = ds * A;
		}
		const float draw = dA * 0.98f * sig * (1.0f - sig);
		weight_updates[0 * channels + c] += draw;
		weight_updates[1 * channels + c] += dB;
		weight_updates[2 * channels + c] += dC;
		weight_updates[3 * channels + c] += dD;
	}

	void scan4_backward_cpu(const float * input, const float * output_delta, const float * weights, float * weight_updates, float * input_delta, const int batch, const int channels, const int h, const int w)
	{
		const int area = h * w;
		std::fill(input_delta, input_delta + batch * channels * area, 0.0f);
		std::vector<int> idx;
		idx.reserve(std::max(h, w));
		for (int b = 0; b < batch; ++b)
		{
			for (int c = 0; c < channels; ++c)
			{
				for (int y = 0; y < h; ++y)
				{
					idx.clear();
					for (int x = 0; x < w; ++x) idx.push_back(y * w + x);
					scan_sequence_backward_cpu(input, output_delta, weights, weight_updates, input_delta, batch, channels, h, w, b, c, idx);
					idx.clear();
					for (int x = w - 1; x >= 0; --x) idx.push_back(y * w + x);
					scan_sequence_backward_cpu(input, output_delta, weights, weight_updates, input_delta, batch, channels, h, w, b, c, idx);
				}
				for (int x = 0; x < w; ++x)
				{
					idx.clear();
					for (int y = 0; y < h; ++y) idx.push_back(y * w + x);
					scan_sequence_backward_cpu(input, output_delta, weights, weight_updates, input_delta, batch, channels, h, w, b, c, idx);
					idx.clear();
					for (int y = h - 1; y >= 0; --y) idx.push_back(y * w + x);
					scan_sequence_backward_cpu(input, output_delta, weights, weight_updates, input_delta, batch, channels, h, w, b, c, idx);
				}
			}
		}
	}

	void hf_energy_upsample_cpu(const float * lh, const float * hl, const float * hh, float * e_up, const int batch, const int channels, const int h, const int w, const int h2, const int w2)
	{
		const int area = h * w;
		const int area2 = h2 * w2;
		for (int b = 0; b < batch; ++b)
		{
			for (int c = 0; c < channels; ++c)
			{
				for (int y = 0; y < h; ++y)
				{
					const int yy = std::min(y / 2, h2 - 1);
					for (int x = 0; x < w; ++x)
					{
						const int xx = std::min(x / 2, w2 - 1);
						const int s2 = yy * w2 + xx;
						const int o2 = offset4(b, c, s2, channels, area2);
						e_up[offset4(b, c, y * w + x, channels, area)] = std::fabs(lh[o2]) + std::fabs(hl[o2]) + std::fabs(hh[o2]);
					}
				}
			}
		}
	}

	void hf_energy_upsample_backward_cpu(const float * e_up_delta, const float * lh, const float * hl, const float * hh, float * dlh, float * dhl, float * dhh, const int batch, const int channels, const int h, const int w, const int h2, const int w2)
	{
		const int area = h * w;
		const int area2 = h2 * w2;
		for (int b = 0; b < batch; ++b)
		{
			for (int c = 0; c < channels; ++c)
			{
				for (int y = 0; y < h; ++y)
				{
					const int yy = std::min(y / 2, h2 - 1);
					for (int x = 0; x < w; ++x)
					{
						const int xx = std::min(x / 2, w2 - 1);
						const int o2 = offset4(b, c, yy * w2 + xx, channels, area2);
						const float d = e_up_delta[offset4(b, c, y * w + x, channels, area)];
						dlh[o2] += d * sign_no_zero(lh[o2]);
						dhl[o2] += d * sign_no_zero(hl[o2]);
						dhh[o2] += d * sign_no_zero(hh[o2]);
					}
				}
			}
		}
	}

	struct WMHFForwardBuffersCPU
	{
		float * x_local = nullptr;
		float * x_global = nullptr;
		float * local_cat = nullptr;
		float * ll = nullptr;
		float * lh = nullptr;
		float * hl = nullptr;
		float * hh = nullptr;
		float * ll_scan = nullptr;
		float * y_global = nullptr;
		float * e_up = nullptr;
		float * fuse_in = nullptr;
	};

	void free_forward_buffers(WMHFForwardBuffersCPU & p)
	{
		free(p.x_local);
		free(p.x_global);
		free(p.local_cat);
		free(p.ll);
		free(p.lh);
		free(p.hl);
		free(p.hh);
		free(p.ll_scan);
		free(p.y_global);
		free(p.e_up);
		free(p.fuse_in);
		p = WMHFForwardBuffersCPU{};
	}

	WMHFForwardBuffersCPU build_forward_buffers_cpu(Darknet::Layer & l, Darknet::NetworkState state, const bool run_convs)
	{
		const WMHFCounts c = get_counts(l);
		const int area = l.h * l.w;
		const int h2 = (l.h + 1) / 2;
		const int w2 = (l.w + 1) / 2;
		const int area2 = h2 * w2;

		WMHFForwardBuffersCPU p;
		p.x_local = (float*)xcalloc(l.batch * c.local * area, sizeof(float));
		p.x_global = (float*)xcalloc(l.batch * c.global * area, sizeof(float));
		p.local_cat = (float*)xcalloc(l.batch * 3 * c.local * area, sizeof(float));
		p.ll = (float*)xcalloc(l.batch * c.global * area2, sizeof(float));
		p.lh = (float*)xcalloc(l.batch * c.global * area2, sizeof(float));
		p.hl = (float*)xcalloc(l.batch * c.global * area2, sizeof(float));
		p.hh = (float*)xcalloc(l.batch * c.global * area2, sizeof(float));
		p.ll_scan = (float*)xcalloc(l.batch * c.global * area2, sizeof(float));
		p.y_global = (float*)xcalloc(l.batch * c.global * area, sizeof(float));
		p.e_up = (float*)xcalloc(l.batch * c.global * area, sizeof(float));
		p.fuse_in = (float*)xcalloc(l.batch * c.filters * area, sizeof(float));

		if (run_convs)
		{
			Darknet::NetworkState s = state;
			s.input = state.input;
			forward_convolutional_layer(sub(l, WMHF_PRE), s);
		}

		const float * projected = sub(l, WMHF_PRE).output;
		extract_channels_cpu(projected, p.x_local, l.batch, c.filters, c.local, c.id, area);
		extract_channels_cpu(projected, p.x_global, l.batch, c.filters, c.global, c.id + c.local, area);

		if (run_convs)
		{
			Darknet::NetworkState s = state;
			s.input = p.x_local;
			forward_convolutional_layer(sub(l, WMHF_LOCAL_DW3), s);
			forward_convolutional_layer(sub(l, WMHF_LOCAL_DW5), s);
			forward_convolutional_layer(sub(l, WMHF_LOCAL_DW7), s);
		}
		local_concat_cpu(sub(l, WMHF_LOCAL_DW3).output, sub(l, WMHF_LOCAL_DW5).output, sub(l, WMHF_LOCAL_DW7).output, p.local_cat, l.batch, c.local, area);
		if (run_convs)
		{
			Darknet::NetworkState s = state;
			s.input = p.local_cat;
			forward_convolutional_layer(sub(l, WMHF_LOCAL_MIX), s);
		}

		dwt_cpu(p.x_global, p.ll, p.lh, p.hl, p.hh, l.batch, c.global, l.h, l.w, h2, w2);
		scan4_forward_cpu(p.ll, l.weights, p.ll_scan, l.batch, c.global, h2, w2);
		iwt_cpu(p.ll_scan, p.lh, p.hl, p.hh, p.y_global, l.batch, c.global, l.h, l.w, h2, w2);
		hf_energy_upsample_cpu(p.lh, p.hl, p.hh, p.e_up, l.batch, c.global, l.h, l.w, h2, w2);

		if (run_convs)
		{
			Darknet::NetworkState s = state;
			s.input = p.e_up;
			forward_convolutional_layer(sub(l, WMHF_HF_GATE), s);
		}
		fuse_concat_cpu(projected, sub(l, WMHF_LOCAL_MIX).output, p.y_global, p.fuse_in, l.batch, c.id, c.local, c.global, area);
		if (run_convs)
		{
			Darknet::NetworkState s = state;
			s.input = p.fuse_in;
			forward_convolutional_layer(sub(l, WMHF_FUSE), s);
		}
		return p;
	}

	void zero_internal_deltas_cpu(Darknet::Layer & l)
	{
		for (int i = 0; i < WMHF_SUB_COUNT; ++i)
		{
			Darknet::Layer & s = sub(l, i);
			if (s.delta)
			{
				fill_cpu(s.outputs * s.batch, 0.0f, s.delta, 1);
			}
		}
	}

#ifdef DARKNET_GPU
	struct WMHFForwardBuffersGPU
	{
		float * x_local = nullptr;
		float * x_global = nullptr;
		float * local_cat = nullptr;
		float * ll = nullptr;
		float * lh = nullptr;
		float * hl = nullptr;
		float * hh = nullptr;
		float * ll_scan = nullptr;
		float * y_global = nullptr;
		float * e_up = nullptr;
		float * fuse_in = nullptr;
	};

	void free_forward_buffers_gpu(WMHFForwardBuffersGPU & p)
	{
		cuda_free(p.x_local);
		cuda_free(p.x_global);
		cuda_free(p.local_cat);
		cuda_free(p.ll);
		cuda_free(p.lh);
		cuda_free(p.hl);
		cuda_free(p.hh);
		cuda_free(p.ll_scan);
		cuda_free(p.y_global);
		cuda_free(p.e_up);
		cuda_free(p.fuse_in);
		p = WMHFForwardBuffersGPU{};
	}

	WMHFForwardBuffersGPU alloc_forward_buffers_gpu(const Darknet::Layer & l)
	{
		const WMHFCounts c = get_counts(l);
		const int area = l.h * l.w;
		const int h2 = (l.h + 1) / 2;
		const int w2 = (l.w + 1) / 2;
		const int area2 = h2 * w2;
		WMHFForwardBuffersGPU p;
		p.x_local = cuda_make_array(nullptr, l.batch * c.local * area);
		p.x_global = cuda_make_array(nullptr, l.batch * c.global * area);
		p.local_cat = cuda_make_array(nullptr, l.batch * 3 * c.local * area);
		p.ll = cuda_make_array(nullptr, l.batch * c.global * area2);
		p.lh = cuda_make_array(nullptr, l.batch * c.global * area2);
		p.hl = cuda_make_array(nullptr, l.batch * c.global * area2);
		p.hh = cuda_make_array(nullptr, l.batch * c.global * area2);
		p.ll_scan = cuda_make_array(nullptr, l.batch * c.global * area2);
		p.y_global = cuda_make_array(nullptr, l.batch * c.global * area);
		p.e_up = cuda_make_array(nullptr, l.batch * c.global * area);
		p.fuse_in = cuda_make_array(nullptr, l.batch * c.filters * area);
		return p;
	}

	WMHFForwardBuffersGPU build_forward_buffers_gpu(Darknet::Layer & l, Darknet::NetworkState state, const bool run_convs)
	{
		const WMHFCounts c = get_counts(l);
		const int area = l.h * l.w;
		const int h2 = (l.h + 1) / 2;
		const int w2 = (l.w + 1) / 2;
		const int area2 = h2 * w2;
		WMHFForwardBuffersGPU p = alloc_forward_buffers_gpu(l);

		if (run_convs)
		{
			Darknet::NetworkState s = state;
			s.input = state.input;
			forward_convolutional_layer_gpu(sub(l, WMHF_PRE), s);
		}

		wmhf_extract_channels_ongpu(l.batch * c.local * area, sub(l, WMHF_PRE).output_gpu, p.x_local, l.batch, c.filters, c.local, c.id, area);
		wmhf_extract_channels_ongpu(l.batch * c.global * area, sub(l, WMHF_PRE).output_gpu, p.x_global, l.batch, c.filters, c.global, c.id + c.local, area);

		if (run_convs)
		{
			Darknet::NetworkState s = state;
			s.input = p.x_local;
			forward_convolutional_layer_gpu(sub(l, WMHF_LOCAL_DW3), s);
			forward_convolutional_layer_gpu(sub(l, WMHF_LOCAL_DW5), s);
			forward_convolutional_layer_gpu(sub(l, WMHF_LOCAL_DW7), s);
		}
		wmhf_local_concat_ongpu(l.batch * 3 * c.local * area, sub(l, WMHF_LOCAL_DW3).output_gpu, sub(l, WMHF_LOCAL_DW5).output_gpu, sub(l, WMHF_LOCAL_DW7).output_gpu, p.local_cat, c.local, area);

		if (run_convs)
		{
			Darknet::NetworkState s = state;
			s.input = p.local_cat;
			forward_convolutional_layer_gpu(sub(l, WMHF_LOCAL_MIX), s);
		}

		wmhf_dwt_ongpu(l.batch * c.global * area2, p.x_global, p.ll, p.lh, p.hl, p.hh, l.batch, c.global, l.h, l.w, h2, w2);
		wmhf_scan4_forward_ongpu(l.batch * c.global * (h2 + w2), p.ll, l.weights_gpu, p.ll_scan, l.batch, c.global, h2, w2);
		wmhf_iwt_ongpu(l.batch * c.global * area, p.ll_scan, p.lh, p.hl, p.hh, p.y_global, l.batch, c.global, l.h, l.w, h2, w2);
		wmhf_hf_energy_upsample_ongpu(l.batch * c.global * area, p.lh, p.hl, p.hh, p.e_up, l.batch, c.global, l.h, l.w, h2, w2);

		if (run_convs)
		{
			Darknet::NetworkState s = state;
			s.input = p.e_up;
			forward_convolutional_layer_gpu(sub(l, WMHF_HF_GATE), s);
		}
		wmhf_fuse_concat_ongpu(l.batch * c.filters * area, sub(l, WMHF_PRE).output_gpu, sub(l, WMHF_LOCAL_MIX).output_gpu, p.y_global, p.fuse_in, c.id, c.local, c.global, area);

		if (run_convs)
		{
			Darknet::NetworkState s = state;
			s.input = p.fuse_in;
			forward_convolutional_layer_gpu(sub(l, WMHF_FUSE), s);
		}
		return p;
	}
#endif
}

Darknet::Layer Darknet::make_wmhf_layer(
	int batch,
	int h,
	int w,
	int c,
	int filters,
	float identity_ratio,
	float local_ratio,
	float freq_scale,
	int shortcut,
	ACTIVATION activation,
	int batch_normalize,
	int adam,
	int index,
	int train)
{
	TAT(TATPARMS);

	const WMHFCounts split = choose_counts(filters, identity_ratio, local_ratio);

	Darknet::Layer l = { (Darknet::ELayerType)0 };
	l.type = Darknet::ELayerType::WMHF;
	l.batch = batch;
	l.train = train;
	l.index = index;
	l.h = h;
	l.w = w;
	l.c = c;
	l.out_h = h;
	l.out_w = w;
	l.out_c = filters;
	l.inputs = h * w * c;
	l.outputs = h * w * filters;
	l.activation = activation;
	l.shortcut = shortcut;
	l.groups = split.id;       // parent field reused: identity channels
	l.group_id = split.local;  // parent field reused: local channels
	l.scale = freq_scale;      // high-frequency residual scale
	l.batch_normalize = batch_normalize;
	l.learning_rate_scale = 1.0f;

	l.input_layer = (Darknet::Layer*)xcalloc(WMHF_SUB_COUNT, sizeof(Darknet::Layer));

	// Internal convolutions.  All keep HxW.  The gate uses LOGISTIC so its output is [0, 1].
	sub(l, WMHF_PRE) = make_convolutional_layer(batch, 1, h, w, c, filters, 1, 1, 1, 1, 1, 0, LINEAR, batch_normalize, 0, 0, adam, 0, index, 0, nullptr, 0, 0, train);
	sub(l, WMHF_LOCAL_DW3) = make_convolutional_layer(batch, 1, h, w, split.local, split.local, split.local, 3, 1, 1, 1, 1, LINEAR, 0, 0, 0, adam, 0, index, 0, nullptr, 0, 0, train);
	sub(l, WMHF_LOCAL_DW5) = make_convolutional_layer(batch, 1, h, w, split.local, split.local, split.local, 5, 1, 1, 1, 2, LINEAR, 0, 0, 0, adam, 0, index, 0, nullptr, 0, 0, train);
	sub(l, WMHF_LOCAL_DW7) = make_convolutional_layer(batch, 1, h, w, split.local, split.local, split.local, 7, 1, 1, 1, 3, LINEAR, 0, 0, 0, adam, 0, index, 0, nullptr, 0, 0, train);
	sub(l, WMHF_LOCAL_MIX) = make_convolutional_layer(batch, 1, h, w, 3 * split.local, split.local, 1, 1, 1, 1, 1, 0, LINEAR, batch_normalize, 0, 0, adam, 0, index, 0, nullptr, 0, 0, train);
	sub(l, WMHF_HF_GATE) = make_convolutional_layer(batch, 1, h, w, split.global, filters, 1, 1, 1, 1, 1, 0, LOGISTIC, 0, 0, 0, adam, 0, index, 0, nullptr, 0, 0, train);
	sub(l, WMHF_FUSE) = make_convolutional_layer(batch, 1, h, w, filters, filters, 1, 1, 1, 1, 1, 0, LINEAR, batch_normalize, 0, 0, adam, 0, index, 0, nullptr, 0, 0, train);

	l.nweights = scan_param_count(split.global);
	l.weights = (float*)xcalloc(l.nweights, sizeof(float));
	l.weight_updates = (float*)xcalloc(l.nweights, sizeof(float));
	init_scan_weights(l.weights, split.global);

	l.output = (float*)xcalloc(batch * l.outputs, sizeof(float));
	l.delta = (float*)xcalloc(batch * l.outputs, sizeof(float));
	l.forward = Darknet::forward_wmhf_layer;
	l.backward = Darknet::backward_wmhf_layer;
	l.update = Darknet::update_wmhf_layer;

	l.workspace_size = 0;
	l.bflops = 0.0f;
	for (int i = 0; i < WMHF_SUB_COUNT; ++i)
	{
		l.workspace_size = std::max(l.workspace_size, sub(l, i).workspace_size);
		l.bflops += sub(l, i).bflops;
	}
	// Haar + scan + gate multiply bookkeeping.  Approximate but useful in layer table.
	l.bflops += static_cast<float>(batch) * h * w * filters * 8.0f / 1000000000.0f;

#ifdef DARKNET_GPU
	l.forward_gpu = Darknet::forward_wmhf_layer_gpu;
	l.backward_gpu = Darknet::backward_wmhf_layer_gpu;
	l.update_gpu = Darknet::update_wmhf_layer_gpu;
	if (cfg_and_state.gpu_index >= 0)
	{
		l.output_gpu = cuda_make_array(l.output, batch * l.outputs);
		l.delta_gpu = cuda_make_array(l.delta, batch * l.outputs);
		l.weights_gpu = cuda_make_array(l.weights, l.nweights);
		l.weight_updates_gpu = cuda_make_array(l.weight_updates, l.nweights);
	}
#endif

	if (cfg_and_state.is_verbose)
	{
		*cfg_and_state.output
			<< "wmhf                     "
			<< w << " x " << h << " x " << c
			<< " -> " << l.out_w << " x " << l.out_h << " x " << l.out_c
			<< "  split=" << split.id << "/" << split.local << "/" << split.global
			<< "  freq=" << freq_scale
			<< "  shortcut=" << shortcut
			<< std::endl;
	}

	return l;
}

void Darknet::resize_wmhf_layer(Darknet::Layer * l, int w, int h)
{
	TAT(TATPARMS);
	if (l == nullptr) return;

	l->w = w;
	l->h = h;
	l->out_w = w;
	l->out_h = h;
	l->inputs = w * h * l->c;
	l->outputs = w * h * l->out_c;

	for (int i = 0; i < WMHF_SUB_COUNT; ++i)
	{
		sub(*l, i).batch = l->batch;
	}

	resize_wmhf_sub_layer(&sub(*l, WMHF_PRE), w, h);
	resize_wmhf_sub_layer(&sub(*l, WMHF_LOCAL_DW3), w, h);
	resize_wmhf_sub_layer(&sub(*l, WMHF_LOCAL_DW5), w, h);
	resize_wmhf_sub_layer(&sub(*l, WMHF_LOCAL_DW7), w, h);
	resize_wmhf_sub_layer(&sub(*l, WMHF_LOCAL_MIX), w, h);
	resize_wmhf_sub_layer(&sub(*l, WMHF_HF_GATE), w, h);
	resize_wmhf_sub_layer(&sub(*l, WMHF_FUSE), w, h);

	l->output = (float*)xrealloc(l->output, l->batch * l->outputs * sizeof(float));
	l->delta = (float*)xrealloc(l->delta, l->batch * l->outputs * sizeof(float));

	l->workspace_size = 0;
	for (int i = 0; i < WMHF_SUB_COUNT; ++i)
	{
		l->workspace_size = std::max(l->workspace_size, sub(*l, i).workspace_size);
	}

#ifdef DARKNET_GPU
	if (cfg_and_state.gpu_index >= 0)
	{
		cuda_free(l->output_gpu);
		cuda_free(l->delta_gpu);
		l->output_gpu = cuda_make_array(l->output, l->batch * l->outputs);
		l->delta_gpu = cuda_make_array(l->delta, l->batch * l->outputs);
	}
#endif
}

void Darknet::forward_wmhf_layer(Darknet::Layer & l, Darknet::NetworkState state)
{
	TAT(TATPARMS);
	WMHFForwardBuffersCPU p = build_forward_buffers_cpu(l, state, true);
	const int total = l.batch * l.outputs;
	const int use_shortcut = (l.shortcut && l.c == l.out_c && l.inputs == l.outputs) ? 1 : 0;
	const float freq_scale = l.scale;
	const float * fuse = sub(l, WMHF_FUSE).output;
	const float * gate = sub(l, WMHF_HF_GATE).output;
	const float * projected = sub(l, WMHF_PRE).output;

	for (int i = 0; i < total; ++i)
	{
		float out = fuse[i] + freq_scale * gate[i] * projected[i];
		if (use_shortcut)
		{
			out += state.input[i];
		}
		l.output[i] = out;
	}
	activate_array(l.output, total, l.activation);
	free_forward_buffers(p);
}

void Darknet::backward_wmhf_layer(Darknet::Layer & l, Darknet::NetworkState state)
{
	TAT(TATPARMS);
	const WMHFCounts c = get_counts(l);
	const int area = l.h * l.w;
	const int h2 = (l.h + 1) / 2;
	const int w2 = (l.w + 1) / 2;
	const int area2 = h2 * w2;
	const int total = l.batch * l.outputs;
	const int use_shortcut = (l.shortcut && l.c == l.out_c && l.inputs == l.outputs && state.delta) ? 1 : 0;

	zero_internal_deltas_cpu(l);
	gradient_array(l.output, total, l.activation, l.delta);

	WMHFForwardBuffersCPU p = build_forward_buffers_cpu(l, state, false);

	float * pre_delta = (float*)xcalloc(l.batch * c.filters * area, sizeof(float));
	float * fuse_delta = (float*)xcalloc(l.batch * c.filters * area, sizeof(float));
	float * fuse_in_delta = (float*)xcalloc(l.batch * c.filters * area, sizeof(float));
	float * gate_delta = (float*)xcalloc(l.batch * c.filters * area, sizeof(float));
	float * local_delta = (float*)xcalloc(l.batch * c.local * area, sizeof(float));
	float * local_cat_delta = (float*)xcalloc(l.batch * 3 * c.local * area, sizeof(float));
	float * local_dx = (float*)xcalloc(l.batch * c.local * area, sizeof(float));
	float * global_delta = (float*)xcalloc(l.batch * c.global * area, sizeof(float));
	float * global_input_delta = (float*)xcalloc(l.batch * c.global * area, sizeof(float));
	float * e_up_delta = (float*)xcalloc(l.batch * c.global * area, sizeof(float));
	float * dll = (float*)xcalloc(l.batch * c.global * area2, sizeof(float));
	float * dlh = (float*)xcalloc(l.batch * c.global * area2, sizeof(float));
	float * dhl = (float*)xcalloc(l.batch * c.global * area2, sizeof(float));
	float * dhh = (float*)xcalloc(l.batch * c.global * area2, sizeof(float));
	float * dll_from_scan = (float*)xcalloc(l.batch * c.global * area2, sizeof(float));

	for (int i = 0; i < total; ++i)
	{
		const float d = l.delta[i];
		fuse_delta[i] = d;
		gate_delta[i] = l.scale * sub(l, WMHF_PRE).output[i] * d;
		pre_delta[i] += l.scale * sub(l, WMHF_HF_GATE).output[i] * d;
		if (use_shortcut)
		{
			state.delta[i] += d;
		}
	}

	// Final fusion conv.
	copy_cpu(total, fuse_delta, 1, sub(l, WMHF_FUSE).delta, 1);
	{
		Darknet::NetworkState s = state;
		s.input = p.fuse_in;
		s.delta = fuse_in_delta;
		backward_convolutional_layer(sub(l, WMHF_FUSE), s);
	}
	fuse_concat_backward_cpu(fuse_in_delta, pre_delta, local_delta, global_delta, l.batch, c.id, c.local, c.global, area);

	// High-frequency gate conv.
	copy_cpu(total, gate_delta, 1, sub(l, WMHF_HF_GATE).delta, 1);
	{
		Darknet::NetworkState s = state;
		s.input = p.e_up;
		s.delta = e_up_delta;
		backward_convolutional_layer(sub(l, WMHF_HF_GATE), s);
	}

	// Local branch.
	copy_cpu(l.batch * c.local * area, local_delta, 1, sub(l, WMHF_LOCAL_MIX).delta, 1);
	{
		Darknet::NetworkState s = state;
		s.input = p.local_cat;
		s.delta = local_cat_delta;
		backward_convolutional_layer(sub(l, WMHF_LOCAL_MIX), s);
	}
	local_concat_backward_cpu(local_cat_delta, sub(l, WMHF_LOCAL_DW3).delta, sub(l, WMHF_LOCAL_DW5).delta, sub(l, WMHF_LOCAL_DW7).delta, l.batch, c.local, area);
	{
		Darknet::NetworkState s = state;
		s.input = p.x_local;
		s.delta = local_dx;
		backward_convolutional_layer(sub(l, WMHF_LOCAL_DW3), s);
		backward_convolutional_layer(sub(l, WMHF_LOCAL_DW5), s);
		backward_convolutional_layer(sub(l, WMHF_LOCAL_DW7), s);
	}
	insert_channels_cpu(local_dx, pre_delta, l.batch, c.filters, c.local, c.id, area, 1.0f);

	// Global wavelet and high-frequency branch.
	iwt_backward_cpu(global_delta, dll, dlh, dhl, dhh, l.batch, c.global, l.h, l.w, h2, w2);
	hf_energy_upsample_backward_cpu(e_up_delta, p.lh, p.hl, p.hh, dlh, dhl, dhh, l.batch, c.global, l.h, l.w, h2, w2);
	scan4_backward_cpu(p.ll, dll, l.weights, l.weight_updates, dll_from_scan, l.batch, c.global, h2, w2);
	dwt_backward_cpu(dll_from_scan, dlh, dhl, dhh, global_input_delta, l.batch, c.global, l.h, l.w, h2, w2);
	insert_channels_cpu(global_input_delta, pre_delta, l.batch, c.filters, c.global, c.id + c.local, area, 1.0f);

	// Projection conv to previous layer.
	copy_cpu(l.batch * c.filters * area, pre_delta, 1, sub(l, WMHF_PRE).delta, 1);
	{
		Darknet::NetworkState s = state;
		s.input = state.input;
		s.delta = state.delta;
		backward_convolutional_layer(sub(l, WMHF_PRE), s);
	}

	free(pre_delta);
	free(fuse_delta);
	free(fuse_in_delta);
	free(gate_delta);
	free(local_delta);
	free(local_cat_delta);
	free(local_dx);
	free(global_delta);
	free(global_input_delta);
	free(e_up_delta);
	free(dll);
	free(dlh);
	free(dhl);
	free(dhh);
	free(dll_from_scan);
	free_forward_buffers(p);
}

void Darknet::update_wmhf_layer(Darknet::Layer & l, int batch, float learning_rate, float momentum, float decay)
{
	TAT(TATPARMS);
	for (int i = 0; i < WMHF_SUB_COUNT; ++i)
	{
		update_convolutional_layer(sub(l, i), batch, learning_rate, momentum, decay);
	}
	const float lr = learning_rate * l.learning_rate_scale;
	axpy_cpu(l.nweights, -decay * batch, l.weights, 1, l.weight_updates, 1);
	axpy_cpu(l.nweights, lr / batch, l.weight_updates, 1, l.weights, 1);
	scal_cpu(l.nweights, momentum, l.weight_updates, 1);
}

void Darknet::free_wmhf_layer(Darknet::Layer & l)
{
	// Hook this from free_layer_custom()/free_layer() after adding the new enum.
	// The parent Layer object is owned by the network; this only releases resources
	// allocated by make_wmhf_layer().
	if (l.input_layer)
	{
		for (int i = 0; i < WMHF_SUB_COUNT; ++i)
		{
			free_layer(l.input_layer[i]);
		}
		free(l.input_layer);
		l.input_layer = nullptr;
	}
	free(l.output); l.output = nullptr;
	free(l.delta); l.delta = nullptr;
	free(l.weights); l.weights = nullptr;
	free(l.weight_updates); l.weight_updates = nullptr;
#ifdef DARKNET_GPU
	cuda_free(l.output_gpu); l.output_gpu = nullptr;
	cuda_free(l.delta_gpu); l.delta_gpu = nullptr;
	cuda_free(l.weights_gpu); l.weights_gpu = nullptr;
	cuda_free(l.weight_updates_gpu); l.weight_updates_gpu = nullptr;
#endif
}

#ifdef DARKNET_GPU
void Darknet::forward_wmhf_layer_gpu(Darknet::Layer & l, Darknet::NetworkState state)
{
	TAT(TATPARMS);
	WMHFForwardBuffersGPU p = build_forward_buffers_gpu(l, state, true);
	const int total = l.batch * l.outputs;
	const int use_shortcut = (l.shortcut && l.c == l.out_c && l.inputs == l.outputs) ? 1 : 0;
	wmhf_apply_gate_forward_ongpu(total, sub(l, WMHF_FUSE).output_gpu, sub(l, WMHF_HF_GATE).output_gpu, sub(l, WMHF_PRE).output_gpu, state.input, l.output_gpu, l.scale, use_shortcut);
	activate_array_ongpu(l.output_gpu, total, l.activation);
	free_forward_buffers_gpu(p);
}

void Darknet::backward_wmhf_layer_gpu(Darknet::Layer & l, Darknet::NetworkState state)
{
	TAT(TATPARMS);
	const WMHFCounts c = get_counts(l);
	const int area = l.h * l.w;
	const int h2 = (l.h + 1) / 2;
	const int w2 = (l.w + 1) / 2;
	const int area2 = h2 * w2;
	const int total = l.batch * l.outputs;
	const int use_shortcut = (l.shortcut && l.c == l.out_c && l.inputs == l.outputs && state.delta) ? 1 : 0;

	gradient_array_ongpu(l.output_gpu, total, l.activation, l.delta_gpu);
	WMHFForwardBuffersGPU p = build_forward_buffers_gpu(l, state, false);

	float * pre_delta = cuda_make_array(nullptr, l.batch * c.filters * area);
	float * fuse_delta = cuda_make_array(nullptr, l.batch * c.filters * area);
	float * fuse_in_delta = cuda_make_array(nullptr, l.batch * c.filters * area);
	float * gate_delta = cuda_make_array(nullptr, l.batch * c.filters * area);
	float * local_delta = cuda_make_array(nullptr, l.batch * c.local * area);
	float * local_cat_delta = cuda_make_array(nullptr, l.batch * 3 * c.local * area);
	float * local_dx = cuda_make_array(nullptr, l.batch * c.local * area);
	float * global_delta = cuda_make_array(nullptr, l.batch * c.global * area);
	float * global_input_delta = cuda_make_array(nullptr, l.batch * c.global * area);
	float * e_up_delta = cuda_make_array(nullptr, l.batch * c.global * area);
	float * dll = cuda_make_array(nullptr, l.batch * c.global * area2);
	float * dlh = cuda_make_array(nullptr, l.batch * c.global * area2);
	float * dhl = cuda_make_array(nullptr, l.batch * c.global * area2);
	float * dhh = cuda_make_array(nullptr, l.batch * c.global * area2);
	float * dll_from_scan = cuda_make_array(nullptr, l.batch * c.global * area2);

	fill_ongpu(l.batch * c.filters * area, 0.0f, pre_delta, 1);
	fill_ongpu(l.batch * c.filters * area, 0.0f, fuse_in_delta, 1);
	fill_ongpu(l.batch * c.local * area, 0.0f, local_dx, 1);
	fill_ongpu(l.batch * c.global * area, 0.0f, global_input_delta, 1);
	fill_ongpu(l.batch * c.global * area, 0.0f, e_up_delta, 1);
	fill_ongpu(l.batch * c.global * area2, 0.0f, dll, 1);
	fill_ongpu(l.batch * c.global * area2, 0.0f, dlh, 1);
	fill_ongpu(l.batch * c.global * area2, 0.0f, dhl, 1);
	fill_ongpu(l.batch * c.global * area2, 0.0f, dhh, 1);
	fill_ongpu(l.batch * c.global * area2, 0.0f, dll_from_scan, 1);

	wmhf_apply_gate_backward_ongpu(total, l.delta_gpu, sub(l, WMHF_HF_GATE).output_gpu, sub(l, WMHF_PRE).output_gpu, fuse_delta, gate_delta, pre_delta, state.delta, l.scale, use_shortcut);

	simple_copy_ongpu(total, fuse_delta, sub(l, WMHF_FUSE).delta_gpu);
	{
		Darknet::NetworkState s = state;
		s.input = p.fuse_in;
		s.delta = fuse_in_delta;
		backward_convolutional_layer_gpu(sub(l, WMHF_FUSE), s);
	}
	wmhf_fuse_concat_backward_ongpu(total, fuse_in_delta, pre_delta, local_delta, global_delta, c.id, c.local, c.global, area);

	simple_copy_ongpu(total, gate_delta, sub(l, WMHF_HF_GATE).delta_gpu);
	{
		Darknet::NetworkState s = state;
		s.input = p.e_up;
		s.delta = e_up_delta;
		backward_convolutional_layer_gpu(sub(l, WMHF_HF_GATE), s);
	}

	simple_copy_ongpu(l.batch * c.local * area, local_delta, sub(l, WMHF_LOCAL_MIX).delta_gpu);
	{
		Darknet::NetworkState s = state;
		s.input = p.local_cat;
		s.delta = local_cat_delta;
		backward_convolutional_layer_gpu(sub(l, WMHF_LOCAL_MIX), s);
	}
	wmhf_local_concat_backward_ongpu(l.batch * 3 * c.local * area, local_cat_delta, sub(l, WMHF_LOCAL_DW3).delta_gpu, sub(l, WMHF_LOCAL_DW5).delta_gpu, sub(l, WMHF_LOCAL_DW7).delta_gpu, c.local, area);
	{
		Darknet::NetworkState s = state;
		s.input = p.x_local;
		s.delta = local_dx;
		backward_convolutional_layer_gpu(sub(l, WMHF_LOCAL_DW3), s);
		backward_convolutional_layer_gpu(sub(l, WMHF_LOCAL_DW5), s);
		backward_convolutional_layer_gpu(sub(l, WMHF_LOCAL_DW7), s);
	}
	wmhf_insert_channels_ongpu(l.batch * c.local * area, local_dx, pre_delta, l.batch, c.filters, c.local, c.id, area, 1.0f);

	wmhf_iwt_backward_ongpu(l.batch * c.global * area, global_delta, dll, dlh, dhl, dhh, l.batch, c.global, l.h, l.w, h2, w2);
	wmhf_hf_energy_upsample_backward_ongpu(l.batch * c.global * area, e_up_delta, p.lh, p.hl, p.hh, dlh, dhl, dhh, l.batch, c.global, l.h, l.w, h2, w2);
	wmhf_scan4_backward_ongpu(l.batch * c.global * (h2 + w2), p.ll, dll, l.weights_gpu, l.weight_updates_gpu, dll_from_scan, l.batch, c.global, h2, w2);
	wmhf_dwt_backward_ongpu(l.batch * c.global * area2, dll_from_scan, dlh, dhl, dhh, global_input_delta, l.batch, c.global, l.h, l.w, h2, w2);
	wmhf_insert_channels_ongpu(l.batch * c.global * area, global_input_delta, pre_delta, l.batch, c.filters, c.global, c.id + c.local, area, 1.0f);

	simple_copy_ongpu(l.batch * c.filters * area, pre_delta, sub(l, WMHF_PRE).delta_gpu);
	{
		Darknet::NetworkState s = state;
		s.input = state.input;
		s.delta = state.delta;
		backward_convolutional_layer_gpu(sub(l, WMHF_PRE), s);
	}

	cuda_free(pre_delta);
	cuda_free(fuse_delta);
	cuda_free(fuse_in_delta);
	cuda_free(gate_delta);
	cuda_free(local_delta);
	cuda_free(local_cat_delta);
	cuda_free(local_dx);
	cuda_free(global_delta);
	cuda_free(global_input_delta);
	cuda_free(e_up_delta);
	cuda_free(dll);
	cuda_free(dlh);
	cuda_free(dhl);
	cuda_free(dhh);
	cuda_free(dll_from_scan);
	free_forward_buffers_gpu(p);
}

void Darknet::update_wmhf_layer_gpu(Darknet::Layer & l, int batch, float learning_rate, float momentum, float decay, float loss_scale)
{
	TAT(TATPARMS);
	for (int i = 0; i < WMHF_SUB_COUNT; ++i)
	{
		update_convolutional_layer_gpu(sub(l, i), batch, learning_rate, momentum, decay, loss_scale);
	}
	const float lr = learning_rate * l.learning_rate_scale / loss_scale;
	axpy_ongpu(l.nweights, -decay * batch * loss_scale, l.weights_gpu, 1, l.weight_updates_gpu, 1);
	axpy_ongpu(l.nweights, lr / batch, l.weight_updates_gpu, 1, l.weights_gpu, 1);
	scal_ongpu(l.nweights, momentum, l.weight_updates_gpu, 1);
}
#endif
