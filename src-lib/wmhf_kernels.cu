#include "darknet_internal.hpp"
#include "wmhf_layer.hpp"

#ifdef DARKNET_GPU

#ifndef WMHF_SCAN_MAX
#define WMHF_SCAN_MAX 1024
#endif

__device__ __forceinline__ float wmhf_sigmoid_device(float x)
{
	if (x >= 0.0f)
	{
		float z = expf(-x);
		return 1.0f / (1.0f + z);
	}
	float z = expf(x);
	return z / (1.0f + z);
}

__device__ __forceinline__ float wmhf_sign_device(float x)
{
	return (x > 0.0f) ? 1.0f : ((x < 0.0f) ? -1.0f : 0.0f);
}

__device__ __forceinline__ int wmhf_offset(int b, int c, int s, int channels, int spatial)
{
	return (b * channels + c) * spatial + s;
}

__global__ void wmhf_extract_channels_kernel(int count, const float * __restrict__ input, float * __restrict__ output, int in_c, int out_c, int begin_c, int spatial)
{
	const int index = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
	if (index >= count) return;
	const int s = index % spatial;
	const int t = index / spatial;
	const int oc = t % out_c;
	const int b = t / out_c;
	output[index] = input[wmhf_offset(b, begin_c + oc, s, in_c, spatial)];
}

void wmhf_extract_channels_ongpu(int count, const float * input, float * output, int batch, int in_c, int out_c, int begin_c, int spatial)
{
	TAT(TATPARMS);
	(void)batch;
	wmhf_extract_channels_kernel<<<cuda_gridsize(count), BLOCK, 0, get_cuda_stream()>>>(count, input, output, in_c, out_c, begin_c, spatial);
	CHECK_CUDA(cudaPeekAtLastError());
}

__global__ void wmhf_insert_channels_kernel(int count, const float * __restrict__ input, float * __restrict__ output, int out_c, int in_c, int begin_c, int spatial, float scale)
{
	const int index = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
	if (index >= count) return;
	const int s = index % spatial;
	const int t = index / spatial;
	const int ic = t % in_c;
	const int b = t / in_c;
	atomicAdd(output + wmhf_offset(b, begin_c + ic, s, out_c, spatial), scale * input[index]);
}

void wmhf_insert_channels_ongpu(int count, const float * input, float * output, int batch, int out_c, int in_c, int begin_c, int spatial, float scale)
{
	TAT(TATPARMS);
	(void)batch;
	wmhf_insert_channels_kernel<<<cuda_gridsize(count), BLOCK, 0, get_cuda_stream()>>>(count, input, output, out_c, in_c, begin_c, spatial, scale);
	CHECK_CUDA(cudaPeekAtLastError());
}

__global__ void wmhf_local_concat_kernel(int count, const float * __restrict__ a, const float * __restrict__ b, const float * __restrict__ c, float * __restrict__ out, int channels, int spatial)
{
	const int index = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
	if (index >= count) return;
	const int branch_size = channels * spatial;
	const int local = index % branch_size;
	const int branch = (index / branch_size) % 3;
	const int batch = index / (3 * branch_size);
	const int src_index = batch * branch_size + local;
	out[index] = (branch == 0) ? a[src_index] : ((branch == 1) ? b[src_index] : c[src_index]);
}

void wmhf_local_concat_ongpu(int count, const float * a, const float * b, const float * c, float * out, int channels, int spatial)
{
	TAT(TATPARMS);
	wmhf_local_concat_kernel<<<cuda_gridsize(count), BLOCK, 0, get_cuda_stream()>>>(count, a, b, c, out, channels, spatial);
	CHECK_CUDA(cudaPeekAtLastError());
}

__global__ void wmhf_local_concat_backward_kernel(int count, const float * __restrict__ cat_delta, float * __restrict__ da, float * __restrict__ db, float * __restrict__ dc, int channels, int spatial)
{
	const int index = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
	if (index >= count) return;
	const int branch_size = channels * spatial;
	const int local = index % branch_size;
	const int branch = (index / branch_size) % 3;
	const int batch = index / (3 * branch_size);
	const int dst = batch * branch_size + local;
	if (branch == 0) da[dst] = cat_delta[index];
	else if (branch == 1) db[dst] = cat_delta[index];
	else dc[dst] = cat_delta[index];
}

void wmhf_local_concat_backward_ongpu(int count, const float * cat_delta, float * da, float * db, float * dc, int channels, int spatial)
{
	TAT(TATPARMS);
	wmhf_local_concat_backward_kernel<<<cuda_gridsize(count), BLOCK, 0, get_cuda_stream()>>>(count, cat_delta, da, db, dc, channels, spatial);
	CHECK_CUDA(cudaPeekAtLastError());
}

__global__ void wmhf_fuse_concat_kernel(int count, const float * __restrict__ projected, const float * __restrict__ local, const float * __restrict__ global, float * __restrict__ out, int id_c, int local_c, int global_c, int spatial)
{
	const int index = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
	if (index >= count) return;
	const int filters = id_c + local_c + global_c;
	const int s = index % spatial;
	const int t = index / spatial;
	const int c = t % filters;
	const int b = t / filters;
	if (c < id_c)
	{
		out[index] = projected[wmhf_offset(b, c, s, filters, spatial)];
	}
	else if (c < id_c + local_c)
	{
		out[index] = local[wmhf_offset(b, c - id_c, s, local_c, spatial)];
	}
	else
	{
		out[index] = global[wmhf_offset(b, c - id_c - local_c, s, global_c, spatial)];
	}
}

void wmhf_fuse_concat_ongpu(int count, const float * projected, const float * local, const float * global, float * out, int id_c, int local_c, int global_c, int spatial)
{
	TAT(TATPARMS);
	wmhf_fuse_concat_kernel<<<cuda_gridsize(count), BLOCK, 0, get_cuda_stream()>>>(count, projected, local, global, out, id_c, local_c, global_c, spatial);
	CHECK_CUDA(cudaPeekAtLastError());
}

__global__ void wmhf_fuse_concat_backward_kernel(int count, const float * __restrict__ cat_delta, float * __restrict__ projected_delta, float * __restrict__ local_delta, float * __restrict__ global_delta, int id_c, int local_c, int global_c, int spatial)
{
	const int index = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
	if (index >= count) return;
	const int filters = id_c + local_c + global_c;
	const int s = index % spatial;
	const int t = index / spatial;
	const int c = t % filters;
	const int b = t / filters;
	const float d = cat_delta[index];
	if (c < id_c)
	{
		atomicAdd(projected_delta + wmhf_offset(b, c, s, filters, spatial), d);
	}
	else if (c < id_c + local_c)
	{
		local_delta[wmhf_offset(b, c - id_c, s, local_c, spatial)] = d;
	}
	else
	{
		global_delta[wmhf_offset(b, c - id_c - local_c, s, global_c, spatial)] = d;
	}
}

void wmhf_fuse_concat_backward_ongpu(int count, const float * cat_delta, float * projected_delta, float * local_delta, float * global_delta, int id_c, int local_c, int global_c, int spatial)
{
	TAT(TATPARMS);
	wmhf_fuse_concat_backward_kernel<<<cuda_gridsize(count), BLOCK, 0, get_cuda_stream()>>>(count, cat_delta, projected_delta, local_delta, global_delta, id_c, local_c, global_c, spatial);
	CHECK_CUDA(cudaPeekAtLastError());
}

__global__ void wmhf_dwt_kernel(int count, const float * __restrict__ input, float * __restrict__ ll, float * __restrict__ lh, float * __restrict__ hl, float * __restrict__ hh, int channels, int h, int w, int h2, int w2)
{
	const int index = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
	if (index >= count) return;
	const int area = h * w;
	const int area2 = h2 * w2;
	const int s2 = index % area2;
	const int t = index / area2;
	const int c = t % channels;
	const int b = t / channels;
	const int y = s2 / w2;
	const int x = s2 % w2;
	const int y0 = min(2 * y, h - 1);
	const int y1 = min(y0 + 1, h - 1);
	const int x0 = min(2 * x, w - 1);
	const int x1 = min(x0 + 1, w - 1);
	const float v00 = input[wmhf_offset(b, c, y0 * w + x0, channels, area)];
	const float v01 = input[wmhf_offset(b, c, y0 * w + x1, channels, area)];
	const float v10 = input[wmhf_offset(b, c, y1 * w + x0, channels, area)];
	const float v11 = input[wmhf_offset(b, c, y1 * w + x1, channels, area)];
	ll[index] = 0.5f * (v00 + v01 + v10 + v11);
	lh[index] = 0.5f * (v00 - v01 + v10 - v11);
	hl[index] = 0.5f * (v00 + v01 - v10 - v11);
	hh[index] = 0.5f * (v00 - v01 - v10 + v11);
}

void wmhf_dwt_ongpu(int count, const float * input, float * ll, float * lh, float * hl, float * hh, int batch, int channels, int h, int w, int h2, int w2)
{
	TAT(TATPARMS);
	(void)batch;
	wmhf_dwt_kernel<<<cuda_gridsize(count), BLOCK, 0, get_cuda_stream()>>>(count, input, ll, lh, hl, hh, channels, h, w, h2, w2);
	CHECK_CUDA(cudaPeekAtLastError());
}

__global__ void wmhf_dwt_backward_kernel(int count, const float * __restrict__ dll, const float * __restrict__ dlh, const float * __restrict__ dhl, const float * __restrict__ dhh, float * __restrict__ input_delta, int channels, int h, int w, int h2, int w2)
{
	const int index = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
	if (index >= count) return;
	const int area = h * w;
	const int area2 = h2 * w2;
	const int s2 = index % area2;
	const int t = index / area2;
	const int c = t % channels;
	const int b = t / channels;
	const int y = s2 / w2;
	const int x = s2 % w2;
	const int y0 = min(2 * y, h - 1);
	const int y1 = min(y0 + 1, h - 1);
	const int x0 = min(2 * x, w - 1);
	const int x1 = min(x0 + 1, w - 1);
	const float a = 0.5f * dll[index];
	const float b1 = 0.5f * dlh[index];
	const float c1 = 0.5f * dhl[index];
	const float d = 0.5f * dhh[index];
	atomicAdd(input_delta + wmhf_offset(b, c, y0 * w + x0, channels, area), a + b1 + c1 + d);
	atomicAdd(input_delta + wmhf_offset(b, c, y0 * w + x1, channels, area), a - b1 + c1 - d);
	atomicAdd(input_delta + wmhf_offset(b, c, y1 * w + x0, channels, area), a + b1 - c1 - d);
	atomicAdd(input_delta + wmhf_offset(b, c, y1 * w + x1, channels, area), a - b1 - c1 + d);
}

void wmhf_dwt_backward_ongpu(int count, const float * dll, const float * dlh, const float * dhl, const float * dhh, float * input_delta, int batch, int channels, int h, int w, int h2, int w2)
{
	TAT(TATPARMS);
	(void)batch;
	wmhf_dwt_backward_kernel<<<cuda_gridsize(count), BLOCK, 0, get_cuda_stream()>>>(count, dll, dlh, dhl, dhh, input_delta, channels, h, w, h2, w2);
	CHECK_CUDA(cudaPeekAtLastError());
}

__global__ void wmhf_iwt_kernel(int count, const float * __restrict__ ll, const float * __restrict__ lh, const float * __restrict__ hl, const float * __restrict__ hh, float * __restrict__ output, int channels, int h, int w, int h2, int w2)
{
	const int index = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
	if (index >= count) return;
	const int area = h * w;
	const int area2 = h2 * w2;
	const int s = index % area;
	const int t = index / area;
	const int c = t % channels;
	const int b = t / channels;
	const int y = s / w;
	const int x = s % w;
	const int y2 = min(y / 2, h2 - 1);
	const int x2 = min(x / 2, w2 - 1);
	const int o = wmhf_offset(b, c, y2 * w2 + x2, channels, area2);
	const float L = ll[o];
	const float H1 = lh[o];
	const float H2 = hl[o];
	const float H3 = hh[o];
	const int py = y & 1;
	const int px = x & 1;
	float val;
	if (py == 0 && px == 0) val = L + H1 + H2 + H3;
	else if (py == 0 && px == 1) val = L - H1 + H2 - H3;
	else if (py == 1 && px == 0) val = L + H1 - H2 - H3;
	else val = L - H1 - H2 + H3;
	output[index] = 0.5f * val;
}

void wmhf_iwt_ongpu(int count, const float * ll, const float * lh, const float * hl, const float * hh, float * output, int batch, int channels, int h, int w, int h2, int w2)
{
	TAT(TATPARMS);
	(void)batch;
	wmhf_iwt_kernel<<<cuda_gridsize(count), BLOCK, 0, get_cuda_stream()>>>(count, ll, lh, hl, hh, output, channels, h, w, h2, w2);
	CHECK_CUDA(cudaPeekAtLastError());
}

__global__ void wmhf_iwt_backward_kernel(int count, const float * __restrict__ output_delta, float * __restrict__ dll, float * __restrict__ dlh, float * __restrict__ dhl, float * __restrict__ dhh, int channels, int h, int w, int h2, int w2)
{
	const int index = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
	if (index >= count) return;
	const int area = h * w;
	const int area2 = h2 * w2;
	const int s = index % area;
	const int t = index / area;
	const int c = t % channels;
	const int b = t / channels;
	const int y = s / w;
	const int x = s % w;
	const int y2 = min(y / 2, h2 - 1);
	const int x2 = min(x / 2, w2 - 1);
	const int o = wmhf_offset(b, c, y2 * w2 + x2, channels, area2);
	const int py = y & 1;
	const int px = x & 1;
	const float d = 0.5f * output_delta[index];
	float s_lh, s_hl, s_hh;
	if (py == 0 && px == 0) { s_lh = 1.0f; s_hl = 1.0f; s_hh = 1.0f; }
	else if (py == 0 && px == 1) { s_lh = -1.0f; s_hl = 1.0f; s_hh = -1.0f; }
	else if (py == 1 && px == 0) { s_lh = 1.0f; s_hl = -1.0f; s_hh = -1.0f; }
	else { s_lh = -1.0f; s_hl = -1.0f; s_hh = 1.0f; }
	atomicAdd(dll + o, d);
	atomicAdd(dlh + o, d * s_lh);
	atomicAdd(dhl + o, d * s_hl);
	atomicAdd(dhh + o, d * s_hh);
}

void wmhf_iwt_backward_ongpu(int count, const float * output_delta, float * dll, float * dlh, float * dhl, float * dhh, int batch, int channels, int h, int w, int h2, int w2)
{
	TAT(TATPARMS);
	(void)batch;
	wmhf_iwt_backward_kernel<<<cuda_gridsize(count), BLOCK, 0, get_cuda_stream()>>>(count, output_delta, dll, dlh, dhl, dhh, channels, h, w, h2, w2);
	CHECK_CUDA(cudaPeekAtLastError());
}

__device__ __forceinline__ int wmhf_scan_pos(int dir, int line, int t, int h, int w)
{
	if (dir == 0) return line * w + t;              // left -> right, line=row
	if (dir == 1) return line * w + (w - 1 - t);    // right -> left
	if (dir == 2) return t * w + line;              // top -> bottom, line=col
	return (h - 1 - t) * w + line;                  // bottom -> top
}

__global__ void wmhf_scan4_forward_kernel(int sequences, const float * __restrict__ input, const float * __restrict__ weights, float * __restrict__ output, int channels, int h, int w)
{
	const int seq = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
	if (seq >= sequences) return;
	const int per_bc = 2 * h + 2 * w;
	const int bc = seq / per_bc;
	const int rem = seq % per_bc;
	const int c = bc % channels;
	const int b = bc / channels;
	int dir;
	int line;
	int L;
	if (rem < h) { dir = 0; line = rem; L = w; }
	else if (rem < 2 * h) { dir = 1; line = rem - h; L = w; }
	else if (rem < 2 * h + w) { dir = 2; line = rem - 2 * h; L = h; }
	else { dir = 3; line = rem - 2 * h - w; L = h; }

	const int area = h * w;
	const float raw_a = weights[0 * channels + c];
	const float A = 0.98f * wmhf_sigmoid_device(raw_a);
	const float B = weights[1 * channels + c];
	const float C = weights[2 * channels + c];
	const float D = weights[3 * channels + c];
	float s = 0.0f;
	for (int t = 0; t < L; ++t)
	{
		const int pos = wmhf_scan_pos(dir, line, t, h, w);
		const int o = wmhf_offset(b, c, pos, channels, area);
		const float x = input[o];
		s = A * s + B * x;
		atomicAdd(output + o, 0.25f * (C * s + D * x));
	}
}

void wmhf_scan4_forward_ongpu(int sequences, const float * input, const float * weights, float * output, int batch, int channels, int h, int w)
{
	TAT(TATPARMS);
	const int seq = batch * channels * (2 * h + 2 * w);
	fill_ongpu(batch * channels * h * w, 0.0f, output, 1);
	wmhf_scan4_forward_kernel<<<cuda_gridsize(seq), BLOCK, 0, get_cuda_stream()>>>(seq, input, weights, output, channels, h, w);
	(void)sequences;
	CHECK_CUDA(cudaPeekAtLastError());
}

__global__ void wmhf_scan4_backward_kernel(int sequences, const float * __restrict__ input, const float * __restrict__ output_delta, const float * __restrict__ weights, float * __restrict__ weight_updates, float * __restrict__ input_delta, int channels, int h, int w)
{
	const int seq = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
	if (seq >= sequences) return;
	const int per_bc = 2 * h + 2 * w;
	const int bc = seq / per_bc;
	const int rem = seq % per_bc;
	const int c = bc % channels;
	const int b = bc / channels;
	int dir;
	int line;
	int L;
	if (rem < h) { dir = 0; line = rem; L = w; }
	else if (rem < 2 * h) { dir = 1; line = rem - h; L = w; }
	else if (rem < 2 * h + w) { dir = 2; line = rem - 2 * h; L = h; }
	else { dir = 3; line = rem - 2 * h - w; L = h; }

	const int area = h * w;
	const float raw_a = weights[0 * channels + c];
	const float sig = wmhf_sigmoid_device(raw_a);
	const float A = 0.98f * sig;
	const float B = weights[1 * channels + c];
	const float C = weights[2 * channels + c];
	const float D = weights[3 * channels + c];

	if (L > WMHF_SCAN_MAX)
	{
		for (int t = 0; t < L; ++t)
		{
			const int pos = wmhf_scan_pos(dir, line, t, h, w);
			const int o = wmhf_offset(b, c, pos, channels, area);
			const float dy = 0.25f * output_delta[o];
			atomicAdd(input_delta + o, dy * (D + B * C));
			atomicAdd(weight_updates + 3 * channels + c, dy * input[o]);
		}
		return;
	}

	float s_arr[WMHF_SCAN_MAX];
	float x_arr[WMHF_SCAN_MAX];
	float s = 0.0f;
	for (int t = 0; t < L; ++t)
	{
		const int pos = wmhf_scan_pos(dir, line, t, h, w);
		const int o = wmhf_offset(b, c, pos, channels, area);
		const float x = input[o];
		x_arr[t] = x;
		s = A * s + B * x;
		s_arr[t] = s;
	}

	float dA = 0.0f;
	float dB = 0.0f;
	float dC = 0.0f;
	float dD = 0.0f;
	float ds_next = 0.0f;
	for (int t = L - 1; t >= 0; --t)
	{
		const int pos = wmhf_scan_pos(dir, line, t, h, w);
		const int o = wmhf_offset(b, c, pos, channels, area);
		const float dy = 0.25f * output_delta[o];
		const float x = x_arr[t];
		const float s_t = s_arr[t];
		const float s_prev = (t > 0) ? s_arr[t - 1] : 0.0f;
		dC += dy * s_t;
		dD += dy * x;
		const float ds = ds_next + dy * C;
		dA += ds * s_prev;
		dB += ds * x;
		atomicAdd(input_delta + o, dy * D + ds * B);
		ds_next = ds * A;
	}
	atomicAdd(weight_updates + 0 * channels + c, dA * 0.98f * sig * (1.0f - sig));
	atomicAdd(weight_updates + 1 * channels + c, dB);
	atomicAdd(weight_updates + 2 * channels + c, dC);
	atomicAdd(weight_updates + 3 * channels + c, dD);
}

void wmhf_scan4_backward_ongpu(int sequences, const float * input, const float * output_delta, const float * weights, float * weight_updates, float * input_delta, int batch, int channels, int h, int w)
{
	TAT(TATPARMS);
	const int seq = batch * channels * (2 * h + 2 * w);
	fill_ongpu(batch * channels * h * w, 0.0f, input_delta, 1);
	wmhf_scan4_backward_kernel<<<cuda_gridsize(seq), BLOCK, 0, get_cuda_stream()>>>(seq, input, output_delta, weights, weight_updates, input_delta, channels, h, w);
	(void)sequences;
	CHECK_CUDA(cudaPeekAtLastError());
}

__global__ void wmhf_hf_energy_upsample_kernel(int count, const float * __restrict__ lh, const float * __restrict__ hl, const float * __restrict__ hh, float * __restrict__ e_up, int channels, int h, int w, int h2, int w2)
{
	const int index = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
	if (index >= count) return;
	const int area = h * w;
	const int area2 = h2 * w2;
	const int s = index % area;
	const int t = index / area;
	const int c = t % channels;
	const int b = t / channels;
	const int y = s / w;
	const int x = s % w;
	const int yy = min(y / 2, h2 - 1);
	const int xx = min(x / 2, w2 - 1);
	const int o2 = wmhf_offset(b, c, yy * w2 + xx, channels, area2);
	e_up[index] = fabsf(lh[o2]) + fabsf(hl[o2]) + fabsf(hh[o2]);
}

void wmhf_hf_energy_upsample_ongpu(int count, const float * lh, const float * hl, const float * hh, float * e_up, int batch, int channels, int h, int w, int h2, int w2)
{
	TAT(TATPARMS);
	(void)batch;
	wmhf_hf_energy_upsample_kernel<<<cuda_gridsize(count), BLOCK, 0, get_cuda_stream()>>>(count, lh, hl, hh, e_up, channels, h, w, h2, w2);
	CHECK_CUDA(cudaPeekAtLastError());
}

__global__ void wmhf_hf_energy_upsample_backward_kernel(int count, const float * __restrict__ e_up_delta, const float * __restrict__ lh, const float * __restrict__ hl, const float * __restrict__ hh, float * __restrict__ dlh, float * __restrict__ dhl, float * __restrict__ dhh, int channels, int h, int w, int h2, int w2)
{
	const int index = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
	if (index >= count) return;
	const int area = h * w;
	const int area2 = h2 * w2;
	const int s = index % area;
	const int t = index / area;
	const int c = t % channels;
	const int b = t / channels;
	const int y = s / w;
	const int x = s % w;
	const int yy = min(y / 2, h2 - 1);
	const int xx = min(x / 2, w2 - 1);
	const int o2 = wmhf_offset(b, c, yy * w2 + xx, channels, area2);
	const float d = e_up_delta[index];
	atomicAdd(dlh + o2, d * wmhf_sign_device(lh[o2]));
	atomicAdd(dhl + o2, d * wmhf_sign_device(hl[o2]));
	atomicAdd(dhh + o2, d * wmhf_sign_device(hh[o2]));
}

void wmhf_hf_energy_upsample_backward_ongpu(int count, const float * e_up_delta, const float * lh, const float * hl, const float * hh, float * dlh, float * dhl, float * dhh, int batch, int channels, int h, int w, int h2, int w2)
{
	TAT(TATPARMS);
	(void)batch;
	wmhf_hf_energy_upsample_backward_kernel<<<cuda_gridsize(count), BLOCK, 0, get_cuda_stream()>>>(count, e_up_delta, lh, hl, hh, dlh, dhl, dhh, channels, h, w, h2, w2);
	CHECK_CUDA(cudaPeekAtLastError());
}

__global__ void wmhf_apply_gate_forward_kernel(int count, const float * __restrict__ fuse, const float * __restrict__ gate, const float * __restrict__ projected, const float * __restrict__ shortcut, float * __restrict__ out, float freq_scale, int use_shortcut)
{
	const int index = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
	if (index >= count) return;
	float y = fuse[index] + freq_scale * gate[index] * projected[index];
	if (use_shortcut && shortcut) y += shortcut[index];
	out[index] = y;
}

void wmhf_apply_gate_forward_ongpu(int count, const float * fuse, const float * gate, const float * projected, const float * shortcut, float * out, float freq_scale, int use_shortcut)
{
	TAT(TATPARMS);
	wmhf_apply_gate_forward_kernel<<<cuda_gridsize(count), BLOCK, 0, get_cuda_stream()>>>(count, fuse, gate, projected, shortcut, out, freq_scale, use_shortcut);
	CHECK_CUDA(cudaPeekAtLastError());
}

__global__ void wmhf_apply_gate_backward_kernel(int count, const float * __restrict__ delta, const float * __restrict__ gate, const float * __restrict__ projected, float * __restrict__ fuse_delta, float * __restrict__ gate_delta, float * __restrict__ projected_delta, float * __restrict__ shortcut_delta, float freq_scale, int use_shortcut)
{
	const int index = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
	if (index >= count) return;
	const float d = delta[index];
	fuse_delta[index] = d;
	gate_delta[index] = freq_scale * projected[index] * d;
	atomicAdd(projected_delta + index, freq_scale * gate[index] * d);
	if (use_shortcut && shortcut_delta)
	{
		atomicAdd(shortcut_delta + index, d);
	}
}

void wmhf_apply_gate_backward_ongpu(int count, const float * delta, const float * gate, const float * projected, float * fuse_delta, float * gate_delta, float * projected_delta, float * shortcut_delta, float freq_scale, int use_shortcut)
{
	TAT(TATPARMS);
	wmhf_apply_gate_backward_kernel<<<cuda_gridsize(count), BLOCK, 0, get_cuda_stream()>>>(count, delta, gate, projected, fuse_delta, gate_delta, projected_delta, shortcut_delta, freq_scale, use_shortcut);
	CHECK_CUDA(cudaPeekAtLastError());
}

#endif // DARKNET_GPU
