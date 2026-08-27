#include "darknet_internal.hpp"
#include "detr_decoder_layer.hpp"
#include "blas.hpp"
#include "gemm.hpp"
#include "dark_cuda.hpp"

/**
 * @file detr_decoder_kernels.cu
 * @brief GPU implementation of the DETR-style decoder head (v1).
 *
 * The heavy linear algebra runs as a sequence of @ref gemm_gpu calls in
 * column-per-query / column-per-token matrix layouts; only the small
 * non-linear glue (positional encoding, residual adds, ReLU, softmax,
 * scatter/gather) is bespoke CUDA.  The Hungarian-matched set loss is computed
 * on the host (yolo convention) via @ref detr_decoder_loss.
 *
 * Layouts (per image):
 *   X, KeyIn, Kproj, Vproj              : [D x N]   (column j = memory token j)
 *   Qsa, Ksa, Vsa, CtxSa, EF            : [D x Q]   (column q = object query q; self-attention among queries)
 *   Qproj, Ctx, Attn, FFN               : [D x Q]   (column q = object query q; cross-attention over memory)
 *   H1, HR                              : [F x Q]
 *   SaScores (self-attn, reused/head)   : [Q x Q]
 *   Aw (cross-attn scores)              : [Q x N]
 *   Logits                              : [C x Q],  BoxPre : [4 x Q]
 * All matrices are row-major to match darknet's gemm convention. Multi-head self-attention
 * splits the D rows of Qsa/Ksa/Vsa into `heads` contiguous row-blocks of dh=D/heads rows each
 * (a row-major [D,Q] sub-block of dh consecutive rows is itself a valid [dh,Q] matrix with the
 * same leading dimension Q, so per-head GEMMs just operate on pointer-offset sub-blocks).
 */

namespace
{
	struct DetrParams
	{
		int D, Q, F, C;
		size_t off_E, off_Wsq, off_Wsk, off_Wsv, off_Wso, off_Wq, off_Wk, off_Wv, off_Wo;
		size_t off_W1, off_b1, off_W2, off_b2;
		size_t off_Wc, off_bc, off_Wb, off_bb;
		size_t off_ref;	// per-query reference points (Q x 4): learnable additive box-logit bias
		size_t total;
	};

	DetrParams detr_params(const Darknet::Layer & l)
	{
		DetrParams p;
		p.D = l.c; p.Q = l.detr_queries; p.F = l.detr_ffn; p.C = l.classes;
		size_t o = 0;
		p.off_E   = o; o += (size_t)p.Q * p.D;
		p.off_Wsq = o; o += (size_t)p.D * p.D;
		p.off_Wsk = o; o += (size_t)p.D * p.D;
		p.off_Wsv = o; o += (size_t)p.D * p.D;
		p.off_Wso = o; o += (size_t)p.D * p.D;
		p.off_Wq  = o; o += (size_t)p.D * p.D;
		p.off_Wk  = o; o += (size_t)p.D * p.D;
		p.off_Wv  = o; o += (size_t)p.D * p.D;
		p.off_Wo  = o; o += (size_t)p.D * p.D;
		p.off_W1  = o; o += (size_t)p.D * p.F;
		p.off_b1  = o; o += (size_t)p.F;
		p.off_W2  = o; o += (size_t)p.F * p.D;
		p.off_b2  = o; o += (size_t)p.D;
		p.off_Wc  = o; o += (size_t)p.C * p.D;
		p.off_bc  = o; o += (size_t)p.C;
		p.off_Wb  = o; o += (size_t)4 * p.D;
		p.off_bb  = o; o += (size_t)4;
		p.off_ref = o; o += (size_t)p.Q * 4;	// per-query reference points (cx,cy,w,h pre-sigmoid bias)
		p.total = o;
		return p;
	}

	// All per-image GPU scratch buffers carved from one arena (l.detr_workspace_gpu).
	struct DetrBuf
	{
		float *KeyIn, *Kproj, *Vproj, *Qproj, *Aw, *Ctx, *Attn, *H1, *HR, *FFN, *Logits, *BoxPre;
		float *dKproj, *dVproj, *dA, *dScores, *dCtx, *dAttn, *dFFN, *dHR, *dH1, *dQproj, *dLogits, *dBoxPre;
		float *Qsa, *Ksa, *Vsa, *SaScores, *CtxSa, *EF;
		float *dQsa, *dKsa, *dVsa, *dSaScores, *dCtxSa, *dEF;
	};

	size_t detr_scratch_floats(int D, int Q, int F, int C, int N)
	{
		size_t t = 0;
		t += (size_t)D * N; // KeyIn
		t += (size_t)D * N; // Kproj
		t += (size_t)D * N; // Vproj
		t += (size_t)D * Q; // Qproj
		t += (size_t)Q * N; // Aw
		t += (size_t)D * Q; // Ctx
		t += (size_t)D * Q; // Attn
		t += (size_t)F * Q; // H1
		t += (size_t)F * Q; // HR
		t += (size_t)D * Q; // FFN
		t += (size_t)C * Q; // Logits
		t += (size_t)4 * Q; // BoxPre
		t += (size_t)D * N; // dKproj
		t += (size_t)D * N; // dVproj
		t += (size_t)Q * N; // dA
		t += (size_t)Q * N; // dScores
		t += (size_t)D * Q; // dCtx
		t += (size_t)D * Q; // dAttn
		t += (size_t)D * Q; // dFFN
		t += (size_t)F * Q; // dHR
		t += (size_t)F * Q; // dH1
		t += (size_t)D * Q; // dQproj
		t += (size_t)C * Q; // dLogits
		t += (size_t)4 * Q; // dBoxPre
		t += (size_t)D * Q; // Qsa
		t += (size_t)D * Q; // Ksa
		t += (size_t)D * Q; // Vsa
		t += (size_t)Q * Q; // SaScores
		t += (size_t)D * Q; // CtxSa
		t += (size_t)D * Q; // EF
		t += (size_t)D * Q; // dQsa
		t += (size_t)D * Q; // dKsa
		t += (size_t)D * Q; // dVsa
		t += (size_t)Q * Q; // dSaScores
		t += (size_t)D * Q; // dCtxSa
		t += (size_t)D * Q; // dEF
		return t;
	}

	DetrBuf detr_buffers(const Darknet::Layer & l, int N)
	{
		const DetrParams p = detr_params(l);
		const int D = p.D, Q = p.Q, F = p.F, C = p.C;
		float * base = l.detr_workspace_gpu;
		DetrBuf b;
		size_t o = 0;
		auto take = [&](size_t n) { float * ptr = base + o; o += n; return ptr; };
		b.KeyIn  = take((size_t)D * N);
		b.Kproj  = take((size_t)D * N);
		b.Vproj  = take((size_t)D * N);
		b.Qproj  = take((size_t)D * Q);
		b.Aw     = take((size_t)Q * N);
		b.Ctx    = take((size_t)D * Q);
		b.Attn   = take((size_t)D * Q);
		b.H1     = take((size_t)F * Q);
		b.HR     = take((size_t)F * Q);
		b.FFN    = take((size_t)D * Q);
		b.Logits = take((size_t)C * Q);
		b.BoxPre = take((size_t)4 * Q);
		b.dKproj = take((size_t)D * N);
		b.dVproj = take((size_t)D * N);
		b.dA     = take((size_t)Q * N);
		b.dScores= take((size_t)Q * N);
		b.dCtx   = take((size_t)D * Q);
		b.dAttn  = take((size_t)D * Q);
		b.dFFN   = take((size_t)D * Q);
		b.dHR    = take((size_t)F * Q);
		b.dH1    = take((size_t)F * Q);
		b.dQproj = take((size_t)D * Q);
		b.dLogits= take((size_t)C * Q);
		b.dBoxPre= take((size_t)4 * Q);
		b.Qsa      = take((size_t)D * Q);
		b.Ksa      = take((size_t)D * Q);
		b.Vsa      = take((size_t)D * Q);
		b.SaScores = take((size_t)Q * Q);
		b.CtxSa    = take((size_t)D * Q);
		b.EF       = take((size_t)D * Q);
		b.dQsa      = take((size_t)D * Q);
		b.dKsa      = take((size_t)D * Q);
		b.dVsa      = take((size_t)D * Q);
		b.dSaScores = take((size_t)Q * Q);
		b.dCtxSa    = take((size_t)D * Q);
		b.dEF       = take((size_t)D * Q);
		return b;
	}
}

// ---------------------------------------------------------------------------
// Elementwise kernels
// ---------------------------------------------------------------------------

__device__ __forceinline__ float detr_posenc_d(int j, int d, int D)
{
	const float div = powf(10000.0f, (float)(2 * (d / 2)) / (float)D);
	const float v = (float)j / div;
	return (d % 2 == 0) ? __sinf(v) : __cosf(v);
}

// KeyIn[d,j] = X[d,j] + posenc(j,d) ; layout [D x N], idx = d*N + j
__global__ void detr_keyin_kernel(const float * X, float * KeyIn, int D, int N)
{
	const int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= D * N) return;
	const int d = idx / N;
	const int j = idx % N;
	KeyIn[idx] = X[idx] + detr_posenc_d(j, d, D);
}

// Attn[D x Q] += E^T   (E is [Q x D], E[q*D+d]) ; idx = d*Q + q
__global__ void detr_add_ET_kernel(float * Attn, const float * E, int D, int Q)
{
	const int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= D * Q) return;
	const int d = idx / Q;
	const int q = idx % Q;
	Attn[idx] += E[(size_t)q * D + d];
}

// dE[Q x D] += dAttn^T  (dAttn is [D x Q]) ; unique target per (d,q), no atomics
__global__ void detr_accum_ET_kernel(float * dE, const float * dAttn, int D, int Q)
{
	const int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= D * Q) return;
	const int d = idx / Q;
	const int q = idx % Q;
	dE[(size_t)q * D + d] += dAttn[idx];
}

// H1[f,q] += b1[f] ; HR = relu(H1) ; layout [F x Q], idx = f*Q + q
__global__ void detr_bias_relu_kernel(float * H1, const float * b1, float * HR, int F, int Q)
{
	const int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= F * Q) return;
	const int f = idx / Q;
	const float v = H1[idx] + b1[f];
	H1[idx] = v;
	HR[idx] = v > 0.0f ? v : 0.0f;
}

// FFN[d,q] += b2[d] + Attn[d,q] ; layout [D x Q]
__global__ void detr_bias_resid_kernel(float * FFN, const float * b2, const float * Attn, int D, int Q)
{
	const int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= D * Q) return;
	const int d = idx / Q;
	FFN[idx] += b2[d] + Attn[idx];
}

// M[rows x cols] += bias[row]
__global__ void detr_add_bias_rows_kernel(float * M, const float * bias, int rows, int cols)
{
	const int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= rows * cols) return;
	const int r = idx / cols;
	M[idx] += bias[r];
}

// dH1 = dHR * (H1pre > 0)
__global__ void detr_relu_back_kernel(float * dH1, const float * dHR, const float * H1pre, int n)
{
	const int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= n) return;
	dH1[idx] = (H1pre[idx] > 0.0f) ? dHR[idx] : 0.0f;
}

// Row softmax over N for each of Q rows (in place). One thread per row.
__global__ void detr_softmax_rows_kernel(float * A, int Q, int N)
{
	const int q = blockIdx.x * blockDim.x + threadIdx.x;
	if (q >= Q) return;
	float * row = A + (size_t)q * N;
	float mx = -1e30f;
	for (int j = 0; j < N; ++j) mx = fmaxf(mx, row[j]);
	float sum = 0.0f;
	for (int j = 0; j < N; ++j) { row[j] = __expf(row[j] - mx); sum += row[j]; }
	const float inv = 1.0f / (sum + 1e-9f);
	for (int j = 0; j < N; ++j) row[j] *= inv;
}

// softmax backward: dScores[q,j] = A[q,j]*(dA[q,j] - sum_k A[q,k] dA[q,k]). One thread per row.
__global__ void detr_softmax_back_kernel(const float * A, const float * dA, float * dScores, int Q, int N)
{
	const int q = blockIdx.x * blockDim.x + threadIdx.x;
	if (q >= Q) return;
	const float * a = A + (size_t)q * N;
	const float * da = dA + (size_t)q * N;
	float * ds = dScores + (size_t)q * N;
	float dot = 0.0f;
	for (int j = 0; j < N; ++j) dot += a[j] * da[j];
	for (int j = 0; j < N; ++j) ds[j] = a[j] * (da[j] - dot);
}

// Scatter column-per-query Logits/BoxPre into the per-query interleaved output.
// out[q*(C+4) + k]   = Logits[k,q]            (raw logit)
// out[q*(C+4) + C+i] = sigmoid(BoxPre[i,q])   (box in [0,1])
__global__ void detr_scatter_out_kernel(float * out, const float * Logits, const float * BoxPre, int Q, int C)
{
	const int stride = C + 4;
	const int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= Q * stride) return;
	const int q = idx / stride;
	const int r = idx % stride;
	if (r < C) out[idx] = Logits[(size_t)r * Q + q];
	else
	{
		const int i = r - C;
		const float v = BoxPre[(size_t)i * Q + q];
		out[idx] = 1.0f / (1.0f + __expf(-v));
	}
}

// Gather per-query interleaved delta back into column-per-query grad matrices.
__global__ void detr_gather_delta_kernel(const float * delta, float * dLogits, float * dBoxPre, int Q, int C)
{
	const int stride = C + 4;
	const int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= Q * stride) return;
	const int q = idx / stride;
	const int r = idx % stride;
	if (r < C) dLogits[(size_t)r * Q + q] = delta[idx];
	else       dBoxPre[(size_t)(r - C) * Q + q] = delta[idx];
}

// Add per-query reference points (spatial prior) into the box pre-activation.
// BoxPre is [4 x Q] (i*Q+q); ref is [Q x 4] (q*4+i). box = sigmoid(boxpre + bb + ref_q).
__global__ void detr_add_ref_kernel(float * BoxPre, const float * ref, int Q)
{
	const int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= 4 * Q) return;
	const int i = idx / Q;
	const int q = idx % Q;
	BoxPre[idx] += ref[(size_t)q * 4 + i];
}

// Accumulate the box pre-sigmoid gradient into the reference-point gradient (ref enters the
// pre-sigmoid additively, so d_ref = d_boxpre). Each (i,q) maps to a distinct dref element.
__global__ void detr_accum_ref_kernel(float * dref, const float * dBoxPre, int Q)
{
	const int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= 4 * Q) return;
	const int i = idx / Q;
	const int q = idx % Q;
	dref[(size_t)q * 4 + i] += dBoxPre[idx];
}

// ---------------------------------------------------------------------------
// Setup / resize
// ---------------------------------------------------------------------------

void detr_decoder_setup_gpu(Darknet::Layer & l)
{
	const DetrParams p = detr_params(l);
	const int N = l.h * l.w;

	l.weights_gpu        = cuda_make_array(l.weights,        l.nweights);
	l.weight_updates_gpu = cuda_make_array(l.weight_updates, l.nweights);
	l.output_gpu         = cuda_make_array(l.output, (size_t)l.batch * l.outputs);
	l.delta_gpu          = cuda_make_array(l.delta,  (size_t)l.batch * l.outputs);

	const size_t scratch = detr_scratch_floats(p.D, p.Q, p.F, p.C, N);
	l.detr_workspace_gpu = cuda_make_array(nullptr, scratch);
}

void detr_decoder_resize_gpu(Darknet::Layer & l)
{
	const DetrParams p = detr_params(l);
	const int N = l.h * l.w;

	if (l.detr_workspace_gpu) cuda_free(l.detr_workspace_gpu);
	const size_t scratch = detr_scratch_floats(p.D, p.Q, p.F, p.C, N);
	l.detr_workspace_gpu = cuda_make_array(nullptr, scratch);
}

// ---------------------------------------------------------------------------
// Forward / backward / update
// ---------------------------------------------------------------------------

// Self-attention among the Q query embeddings, run before cross-attention. Splits the D rows of
// Qsa/Ksa/Vsa into `heads` contiguous dh-row blocks; per-head sub-block GEMMs share the [D,Q]
// buffers' leading dimension Q, so a head's row-offset pointer is a valid standalone [dh,Q] matrix.
static void detr_self_attention_forward(const Darknet::Layer & l, const DetrParams & p, const DetrBuf & b, float * E)
{
	const int D = p.D, Q = p.Q;
	const int heads = l.detr_heads > 0 ? l.detr_heads : 1;
	const int dh = D / heads;
	const float invsqrt_dh = 1.0f / sqrtf((float)dh);

	float * W = l.weights_gpu;
	float * Wsq = W + p.off_Wsq;
	float * Wsk = W + p.off_Wsk;
	float * Wsv = W + p.off_Wsv;
	float * Wso = W + p.off_Wso;

	gemm_gpu(0, 1, D, Q, D, 1, Wsq, D, E, D, 0, b.Qsa, Q);
	gemm_gpu(0, 1, D, Q, D, 1, Wsk, D, E, D, 0, b.Ksa, Q);
	gemm_gpu(0, 1, D, Q, D, 1, Wsv, D, E, D, 0, b.Vsa, Q);

	for (int hd = 0; hd < heads; ++hd)
	{
		const int off = hd * dh;
		gemm_gpu(1, 0, Q, Q, dh, invsqrt_dh, b.Qsa + (size_t)off * Q, Q, b.Ksa + (size_t)off * Q, Q, 0, b.SaScores, Q);
		detr_softmax_rows_kernel<<<get_number_of_blocks(Q, BLOCK), BLOCK>>>(b.SaScores, Q, Q);
		gemm_gpu(0, 1, dh, Q, Q, 1, b.Vsa + (size_t)off * Q, Q, b.SaScores, Q, 0, b.CtxSa + (size_t)off * Q, Q);
	}

	gemm_gpu(0, 0, D, Q, D, 1, Wso, D, b.CtxSa, Q, 0, b.EF, Q);
	// Caller adds the E^T residual into b.EF (detr_add_ET_kernel) to finish qf_sa = E + Wso*CtxSa.
}

static void detr_forward_one_image(const Darknet::Layer & l, const DetrParams & p, const DetrBuf & b,
		const float * X, float * out_image, int N)
{
	const int D = p.D, Q = p.Q, F = p.F, C = p.C;
	const float invsqrtD = 1.0f / sqrtf((float)D);

	float * W = l.weights_gpu;
	float * E  = W + p.off_E;
	float * Wq = W + p.off_Wq;
	float * Wk = W + p.off_Wk;
	float * Wv = W + p.off_Wv;
	float * Wo = W + p.off_Wo;
	float * W1 = W + p.off_W1;
	float * b1 = W + p.off_b1;
	float * W2 = W + p.off_W2;
	float * b2 = W + p.off_b2;
	float * Wc = W + p.off_Wc;
	float * bc = W + p.off_bc;
	float * Wb = W + p.off_Wb;
	float * bb = W + p.off_bb;
	float * ref = W + p.off_ref;

	detr_keyin_kernel<<<get_number_of_blocks(D * N, BLOCK), BLOCK>>>(X, b.KeyIn, D, N);

	detr_self_attention_forward(l, p, b, E);
	detr_add_ET_kernel<<<get_number_of_blocks(D * Q, BLOCK), BLOCK>>>(b.EF, E, D, Q);	// EF += E^T (residual)

	gemm_gpu(0, 0, D, N, D, 1, Wv, D, (float*)X, N, 0, b.Vproj, N);
	gemm_gpu(0, 0, D, N, D, 1, Wk, D, b.KeyIn, N, 0, b.Kproj, N);
	gemm_gpu(0, 0, D, Q, D, 1, Wq, D, b.EF, Q, 0, b.Qproj, Q);
	gemm_gpu(1, 0, Q, N, D, invsqrtD, b.Qproj, Q, b.Kproj, N, 0, b.Aw, N);
	detr_softmax_rows_kernel<<<get_number_of_blocks(Q, BLOCK), BLOCK>>>(b.Aw, Q, N);
	gemm_gpu(0, 1, D, Q, N, 1, b.Vproj, N, b.Aw, N, 0, b.Ctx, Q);
	gemm_gpu(0, 0, D, Q, D, 1, Wo, D, b.Ctx, Q, 0, b.Attn, Q);
	axpy_ongpu(D * Q, 1.0f, b.EF, 1, b.Attn, 1);	// residual: Attn += EF (both already [D,Q], no transpose needed)
	gemm_gpu(0, 0, F, Q, D, 1, W1, D, b.Attn, Q, 0, b.H1, Q);
	detr_bias_relu_kernel<<<get_number_of_blocks(F * Q, BLOCK), BLOCK>>>(b.H1, b1, b.HR, F, Q);
	gemm_gpu(0, 0, D, Q, F, 1, W2, F, b.HR, Q, 0, b.FFN, Q);
	detr_bias_resid_kernel<<<get_number_of_blocks(D * Q, BLOCK), BLOCK>>>(b.FFN, b2, b.Attn, D, Q);
	gemm_gpu(0, 0, C, Q, D, 1, Wc, D, b.FFN, Q, 0, b.Logits, Q);
	detr_add_bias_rows_kernel<<<get_number_of_blocks(C * Q, BLOCK), BLOCK>>>(b.Logits, bc, C, Q);
	gemm_gpu(0, 0, 4, Q, D, 1, Wb, D, b.FFN, Q, 0, b.BoxPre, Q);
	detr_add_bias_rows_kernel<<<get_number_of_blocks(4 * Q, BLOCK), BLOCK>>>(b.BoxPre, bb, 4, Q);
	detr_add_ref_kernel<<<get_number_of_blocks(4 * Q, BLOCK), BLOCK>>>(b.BoxPre, ref, Q);	// per-query spatial prior

	detr_scatter_out_kernel<<<get_number_of_blocks(Q * (C + 4), BLOCK), BLOCK>>>(out_image, b.Logits, b.BoxPre, Q, C);
}

void forward_detr_decoder_layer_gpu(Darknet::Layer & l, Darknet::NetworkState state)
{
	const DetrParams p = detr_params(l);
	const int N = l.h * l.w;
	const DetrBuf b = detr_buffers(l, N);

	for (int img = 0; img < l.batch; ++img)
	{
		const float * X = state.input + (size_t)img * l.inputs;
		float * out_image = l.output_gpu + (size_t)img * l.outputs;
		detr_forward_one_image(l, p, b, X, out_image, N);
	}
	CHECK_CUDA(cudaPeekAtLastError());

	// Loss + matcher on the host (predictions -> negative-gradient deltas).
	// state.truth is a DEVICE pointer during GPU training (yolo convention) -> pull it.
	if (state.train && state.truth)
	{
		cuda_pull_array(l.output_gpu, l.output, (size_t)l.batch * l.outputs);

		const size_t num_truth = (size_t)l.batch * l.truths;
		float * truth_cpu = (float*)xcalloc(num_truth, sizeof(float));
		cuda_pull_array(state.truth, truth_cpu, num_truth);

		const float cost = detr_decoder_loss(l, truth_cpu);		// fills l.delta
		if (l.cost) *l.cost = cost;
		cuda_push_array(l.delta_gpu, l.delta, (size_t)l.batch * l.outputs);

		free(truth_cpu);
	}
	else if (l.cost)
	{
		*l.cost = 0.0f;
	}
}

// Backward through self-attention: qf_sa[q] = E[q] + Wso*CtxSa[q]. Consumes b.dEF (gradient
// w.r.t. EF, already accumulated by the caller from both the Attn-residual and Qproj paths) and
// b.Qsa/b.Ksa/b.Vsa/b.CtxSa (forward values, recomputed by detr_self_attention_forward just before
// this runs). Accumulates into dE and dWsq/dWsk/dWsv/dWso (all beta=1, since dE/dWq etc already
// hold contributions from the cross-attention paths computed earlier in the caller).
static void detr_self_attention_backward(const Darknet::Layer & l, const DetrParams & p, const DetrBuf & b,
		float * E, float * dE)
{
	const int D = p.D, Q = p.Q;
	const int heads = l.detr_heads > 0 ? l.detr_heads : 1;
	const int dh = D / heads;
	const float invsqrt_dh = 1.0f / sqrtf((float)dh);

	float * W = l.weights_gpu;
	float * Wsq = W + p.off_Wsq;
	float * Wsk = W + p.off_Wsk;
	float * Wsv = W + p.off_Wsv;
	float * Wso = W + p.off_Wso;

	float * G = l.weight_updates_gpu;
	float * dWsq = G + p.off_Wsq;
	float * dWsk = G + p.off_Wsk;
	float * dWsv = G + p.off_Wsv;
	float * dWso = G + p.off_Wso;

	// qf_sa = E^T + Wso*CtxSa
	detr_accum_ET_kernel<<<get_number_of_blocks(D * Q, BLOCK), BLOCK>>>(dE, b.dEF, D, Q);
	gemm_gpu(0, 1, D, D, Q, 1, b.dEF, Q, b.CtxSa, Q, 1, dWso, D);
	gemm_gpu(1, 0, D, Q, D, 1, Wso, D, b.dEF, Q, 0, b.dCtxSa, Q);

	for (int hd = 0; hd < heads; ++hd)
	{
		const int off = hd * dh;
		// Recompute this head's forward softmax'd scores fresh (cheap, avoids caching all heads).
		gemm_gpu(1, 0, Q, Q, dh, invsqrt_dh, b.Qsa + (size_t)off * Q, Q, b.Ksa + (size_t)off * Q, Q, 0, b.SaScores, Q);
		detr_softmax_rows_kernel<<<get_number_of_blocks(Q, BLOCK), BLOCK>>>(b.SaScores, Q, Q);

		// CtxSa_head = Vsa_head * SaScores^T
		gemm_gpu(1, 0, Q, Q, dh, 1, b.dCtxSa + (size_t)off * Q, Q, b.Vsa + (size_t)off * Q, Q, 0, b.dSaScores, Q);
		gemm_gpu(0, 0, dh, Q, Q, 1, b.dCtxSa + (size_t)off * Q, Q, b.SaScores, Q, 0, b.dVsa + (size_t)off * Q, Q);

		detr_softmax_back_kernel<<<get_number_of_blocks(Q, BLOCK), BLOCK>>>(b.SaScores, b.dSaScores, b.dSaScores, Q, Q);

		// SaScores = invsqrt_dh * Qsa_head^T * Ksa_head
		gemm_gpu(0, 1, dh, Q, Q, invsqrt_dh, b.Ksa + (size_t)off * Q, Q, b.dSaScores, Q, 0, b.dQsa + (size_t)off * Q, Q);
		gemm_gpu(0, 0, dh, Q, Q, invsqrt_dh, b.Qsa + (size_t)off * Q, Q, b.dSaScores, Q, 0, b.dKsa + (size_t)off * Q, Q);
	}

	// Qsa/Ksa/Vsa = Wsq/Wsk/Wsv * E^T
	gemm_gpu(0, 0, D, D, Q, 1, b.dQsa, Q, E, D, 1, dWsq, D);
	gemm_gpu(1, 0, Q, D, D, 1, b.dQsa, Q, Wsq, D, 1, dE, D);
	gemm_gpu(0, 0, D, D, Q, 1, b.dKsa, Q, E, D, 1, dWsk, D);
	gemm_gpu(1, 0, Q, D, D, 1, b.dKsa, Q, Wsk, D, 1, dE, D);
	gemm_gpu(0, 0, D, D, Q, 1, b.dVsa, Q, E, D, 1, dWsv, D);
	gemm_gpu(1, 0, Q, D, D, 1, b.dVsa, Q, Wsv, D, 1, dE, D);
}

void backward_detr_decoder_layer_gpu(Darknet::Layer & l, Darknet::NetworkState state)
{
	const DetrParams p = detr_params(l);
	const int D = p.D, Q = p.Q, F = p.F, C = p.C;
	const int N = l.h * l.w;
	const float invsqrtD = 1.0f / sqrtf((float)D);
	const DetrBuf b = detr_buffers(l, N);

	float * W = l.weights_gpu;
	float * E  = W + p.off_E;
	float * Wq = W + p.off_Wq;
	float * Wk = W + p.off_Wk;
	float * Wv = W + p.off_Wv;
	float * Wo = W + p.off_Wo;
	float * W1 = W + p.off_W1;
	float * b1 = W + p.off_b1;
	float * W2 = W + p.off_W2;
	float * b2 = W + p.off_b2;
	float * Wc = W + p.off_Wc;
	float * Wb = W + p.off_Wb;

	float * G = l.weight_updates_gpu;
	float * dE  = G + p.off_E;
	float * dWq = G + p.off_Wq;
	float * dWk = G + p.off_Wk;
	float * dWv = G + p.off_Wv;
	float * dWo = G + p.off_Wo;
	float * dW1 = G + p.off_W1;
	float * db1 = G + p.off_b1;
	float * dW2 = G + p.off_W2;
	float * db2 = G + p.off_b2;
	float * dWc = G + p.off_Wc;
	float * dbc = G + p.off_bc;
	float * dWb = G + p.off_Wb;
	float * dbb = G + p.off_bb;
	float * dref = G + p.off_ref;

	for (int img = 0; img < l.batch; ++img)
	{
		const float * X = state.input + (size_t)img * l.inputs;
		const float * delta = l.delta_gpu + (size_t)img * l.outputs;

		// ---- recompute the forward intermediates we need (self-attn EF, Kproj, Vproj, Qproj, Aw, Ctx, Attn, H1, HR, FFN) ----
		detr_keyin_kernel<<<get_number_of_blocks(D * N, BLOCK), BLOCK>>>(X, b.KeyIn, D, N);
		detr_self_attention_forward(l, p, b, E);
		detr_add_ET_kernel<<<get_number_of_blocks(D * Q, BLOCK), BLOCK>>>(b.EF, E, D, Q);	// EF += E^T (residual)

		gemm_gpu(0, 0, D, N, D, 1, Wv, D, (float*)X, N, 0, b.Vproj, N);
		gemm_gpu(0, 0, D, N, D, 1, Wk, D, b.KeyIn, N, 0, b.Kproj, N);
		gemm_gpu(0, 0, D, Q, D, 1, Wq, D, b.EF, Q, 0, b.Qproj, Q);
		gemm_gpu(1, 0, Q, N, D, invsqrtD, b.Qproj, Q, b.Kproj, N, 0, b.Aw, N);
		detr_softmax_rows_kernel<<<get_number_of_blocks(Q, BLOCK), BLOCK>>>(b.Aw, Q, N);
		gemm_gpu(0, 1, D, Q, N, 1, b.Vproj, N, b.Aw, N, 0, b.Ctx, Q);
		gemm_gpu(0, 0, D, Q, D, 1, Wo, D, b.Ctx, Q, 0, b.Attn, Q);
		axpy_ongpu(D * Q, 1.0f, b.EF, 1, b.Attn, 1);	// residual: Attn += EF
		gemm_gpu(0, 0, F, Q, D, 1, W1, D, b.Attn, Q, 0, b.H1, Q);
		detr_bias_relu_kernel<<<get_number_of_blocks(F * Q, BLOCK), BLOCK>>>(b.H1, b1, b.HR, F, Q);
		gemm_gpu(0, 0, D, Q, F, 1, W2, F, b.HR, Q, 0, b.FFN, Q);
		detr_bias_resid_kernel<<<get_number_of_blocks(D * Q, BLOCK), BLOCK>>>(b.FFN, b2, b.Attn, D, Q);	// FFN is reused for head grads

		// ---- backward ----
		detr_gather_delta_kernel<<<get_number_of_blocks(Q * (C + 4), BLOCK), BLOCK>>>(delta, b.dLogits, b.dBoxPre, Q, C);

		// class / box heads (FFN -> logits, boxpre)
		gemm_gpu(0, 1, C, D, Q, 1, b.dLogits, Q, b.FFN, Q, 1, dWc, D);
		backward_bias_gpu(dbc, b.dLogits, 1, C, Q);
		gemm_gpu(1, 0, D, Q, C, 1, Wc, D, b.dLogits, Q, 0, b.dFFN, Q);
		gemm_gpu(0, 1, 4, D, Q, 1, b.dBoxPre, Q, b.FFN, Q, 1, dWb, D);
		backward_bias_gpu(dbb, b.dBoxPre, 1, 4, Q);
		detr_accum_ref_kernel<<<get_number_of_blocks(4 * Q, BLOCK), BLOCK>>>(dref, b.dBoxPre, Q);	// per-query spatial prior
		gemm_gpu(1, 0, D, Q, 4, 1, Wb, D, b.dBoxPre, Q, 1, b.dFFN, Q);

		// FFN = Attn + W2*HR + b2
		copy_ongpu(D * Q, b.dFFN, 1, b.dAttn, 1);					// residual to attn
		gemm_gpu(0, 1, D, F, Q, 1, b.dFFN, Q, b.HR, Q, 1, dW2, F);
		backward_bias_gpu(db2, b.dFFN, 1, D, Q);
		gemm_gpu(1, 0, F, Q, D, 1, W2, F, b.dFFN, Q, 0, b.dHR, Q);

		// HR = relu(H1)
		detr_relu_back_kernel<<<get_number_of_blocks(F * Q, BLOCK), BLOCK>>>(b.dH1, b.dHR, b.H1, F * Q);

		// H1 = W1*Attn + b1
		gemm_gpu(0, 1, F, D, Q, 1, b.dH1, Q, b.Attn, Q, 1, dW1, D);
		backward_bias_gpu(db1, b.dH1, 1, F, Q);
		gemm_gpu(1, 0, D, Q, F, 1, W1, D, b.dH1, Q, 1, b.dAttn, Q);

		// Attn = EF + Wo*Ctx  (EF already [D,Q], same layout as Attn -- no transpose needed)
		copy_ongpu(D * Q, b.dAttn, 1, b.dEF, 1);
		gemm_gpu(0, 1, D, D, Q, 1, b.dAttn, Q, b.Ctx, Q, 1, dWo, D);
		gemm_gpu(1, 0, D, Q, D, 1, Wo, D, b.dAttn, Q, 0, b.dCtx, Q);

		// Ctx = Vproj * A^T
		gemm_gpu(1, 0, Q, N, D, 1, b.dCtx, Q, b.Vproj, N, 0, b.dA, N);
		gemm_gpu(0, 0, D, N, Q, 1, b.dCtx, Q, b.Aw, N, 0, b.dVproj, N);

		// softmax backward
		detr_softmax_back_kernel<<<get_number_of_blocks(Q, BLOCK), BLOCK>>>(b.Aw, b.dA, b.dScores, Q, N);

		// Scores = invsqrtD * Qproj^T * Kproj
		gemm_gpu(0, 1, D, Q, N, invsqrtD, b.Kproj, N, b.dScores, N, 0, b.dQproj, Q);
		gemm_gpu(0, 0, D, N, Q, invsqrtD, b.Qproj, Q, b.dScores, N, 0, b.dKproj, N);

		// Qproj = Wq * EF  (EF already [D,Q] -- mirrors the Vproj=Wv*X pattern, not the old E^T one)
		gemm_gpu(0, 1, D, D, Q, 1, b.dQproj, Q, b.EF, Q, 1, dWq, D);
		gemm_gpu(1, 0, D, Q, D, 1, Wq, D, b.dQproj, Q, 1, b.dEF, Q);

		// Kproj = Wk*KeyIn ; Vproj = Wv*X
		gemm_gpu(0, 1, D, D, N, 1, b.dKproj, N, b.KeyIn, N, 1, dWk, D);
		gemm_gpu(0, 1, D, D, N, 1, b.dVproj, N, (float*)X, N, 1, dWv, D);

		// Backbone gradient: dX += Wk^T*dKproj (key path) + Wv^T*dVproj (value path).
		// keyin_j = m_j + pos_j with pos_j constant, so the key-path term passes straight
		// through to dX unchanged, exactly mirroring the CPU backward_detr_decoder_layer fix.
		if (state.delta)
		{
			float * dX = state.delta + (size_t)img * l.inputs;
			gemm_gpu(1, 0, D, N, D, 1, Wk, D, b.dKproj, N, 1, dX, N);
			gemm_gpu(1, 0, D, N, D, 1, Wv, D, b.dVproj, N, 1, dX, N);
		}

		// ---- self-attention backward: propagate dEF (accumulated above) back to dE and dWsq/dWsk/dWsv/dWso ----
		detr_self_attention_backward(l, p, b, E, dE);
	}
	CHECK_CUDA(cudaPeekAtLastError());
}

void update_detr_decoder_layer_gpu(Darknet::Layer & l, int batch, float learning_rate, float momentum, float decay, float loss_scale)
{
	(void)loss_scale;
	const int n = l.nweights;
	const float rate = learning_rate / (batch > 0 ? batch : 1);

	axpy_ongpu(n, -decay * batch, l.weights_gpu, 1, l.weight_updates_gpu, 1);	// weight decay
	axpy_ongpu(n, rate, l.weight_updates_gpu, 1, l.weights_gpu, 1);				// descent (updates hold -dL/dw)
	scal_ongpu(n, momentum, l.weight_updates_gpu, 1);
}
