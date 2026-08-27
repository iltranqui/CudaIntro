#include "darknet_internal.hpp"
#include "detr_decoder_layer.hpp"

#include <cmath>
#include <vector>
#include <array>
#include <algorithm>

/**
 * @file detr_decoder_layer.cpp
 * @brief DETR-style query-based detection decoder head (v1). See detr_decoder_layer.hpp.
 *
 * Gradient sign convention follows darknet: `l.delta` and `l.weight_updates` hold the
 * NEGATIVE gradient (so the SGD step `w += rate * update` performs descent), exactly as
 * the YOLO layers store `delta = truth - pred`.
 */

namespace
{
	static auto & cfg_and_state = Darknet::CfgAndState::get();

	inline float detr_sigmoid(float x)
	{
		return 1.0f / (1.0f + std::exp(-x));
	}

	// Contiguous parameter layout inside l.weights / l.weight_updates.
	struct DetrParams
	{
		int D;	///< model dim (= input channels)
		int Q;	///< number of object queries
		int F;	///< FFN hidden dim
		int C;	///< number of classes
		size_t off_E, off_Wsq, off_Wsk, off_Wsv, off_Wso, off_Wq, off_Wk, off_Wv, off_Wo;
		size_t off_W1, off_b1, off_W2, off_b2;
		size_t off_Wc, off_bc, off_Wb, off_bb;
		size_t off_ref;	///< per-query reference points (Q x 4): learnable additive box-logit bias (spatial prior)
		size_t total;
	};

	DetrParams detr_params(const Darknet::Layer & l)
	{
		DetrParams p;
		p.D = l.c;
		p.Q = l.detr_queries;
		p.F = l.detr_ffn;
		p.C = l.classes;

		size_t o = 0;
		p.off_E   = o; o += (size_t)p.Q * p.D;	// query embeddings (also query positional code)
		p.off_Wsq = o; o += (size_t)p.D * p.D;	// self-attention query projection (queries attend to queries)
		p.off_Wsk = o; o += (size_t)p.D * p.D;	// self-attention key projection
		p.off_Wsv = o; o += (size_t)p.D * p.D;	// self-attention value projection
		p.off_Wso = o; o += (size_t)p.D * p.D;	// self-attention output projection
		p.off_Wq  = o; o += (size_t)p.D * p.D;	// cross-attention query projection
		p.off_Wk  = o; o += (size_t)p.D * p.D;	// cross-attention key projection
		p.off_Wv  = o; o += (size_t)p.D * p.D;	// cross-attention value projection
		p.off_Wo  = o; o += (size_t)p.D * p.D;	// cross-attention output projection
		p.off_W1  = o; o += (size_t)p.D * p.F;	// FFN in
		p.off_b1  = o; o += (size_t)p.F;
		p.off_W2  = o; o += (size_t)p.F * p.D;	// FFN out
		p.off_b2  = o; o += (size_t)p.D;
		p.off_Wc  = o; o += (size_t)p.C * p.D;	// class head
		p.off_bc  = o; o += (size_t)p.C;
		p.off_Wb  = o; o += (size_t)4 * p.D;		// box head
		p.off_bb  = o; o += (size_t)4;
		p.off_ref = o; o += (size_t)p.Q * 4;		// per-query reference points (cx,cy,w,h pre-sigmoid bias)
		p.total = o;
		return p;
	}

	// Sinusoidal positional code for memory token index j over dimension D.
	void detr_posenc(int j, int D, float * dst)
	{
		for (int d = 0; d < D; ++d)
		{
			const float div = std::pow(10000.0f, (float)(2 * (d / 2)) / (float)D);
			const float v = (float)j / div;
			dst[d] = (d % 2 == 0) ? std::sin(v) : std::cos(v);
		}
	}

	// y[out] = W[out,in] * x[in]  (row-major W)
	void detr_matvec(const float * W, const float * x, float * y, int out, int in)
	{
		for (int r = 0; r < out; ++r)
		{
			const float * Wr = W + (size_t)r * in;
			float acc = 0.0f;
			for (int c = 0; c < in; ++c) acc += Wr[c] * x[c];
			y[r] = acc;
		}
	}

	// dW[out,in] += outer(g[out], x[in]) ; accumulates into update buffer
	void detr_accum_outer(float * dW, const float * g, const float * x, int out, int in)
	{
		for (int r = 0; r < out; ++r)
		{
			float * dWr = dW + (size_t)r * in;
			const float gr = g[r];
			for (int c = 0; c < in; ++c) dWr[c] += gr * x[c];
		}
	}

	// dx[in] += W[out,in]^T * g[out]
	void detr_accum_matTvec(const float * W, const float * g, float * dx, int out, int in)
	{
		for (int r = 0; r < out; ++r)
		{
			const float * Wr = W + (size_t)r * in;
			const float gr = g[r];
			for (int c = 0; c < in; ++c) dx[c] += Wr[c] * gr;
		}
	}

	// Gather memory token j (vector across channels) from a [C,H,W] feature map.
	inline void detr_gather_token(const float * input, int j, int N, int D, float * m)
	{
		for (int c = 0; c < D; ++c) m[c] = input[(size_t)c * N + j];
	}

	float detr_l1_box(const float * a, const float * b)
	{
		return std::fabs(a[0]-b[0]) + std::fabs(a[1]-b[1]) + std::fabs(a[2]-b[2]) + std::fabs(a[3]-b[3]);
	}

	// Classic O(n^3) Hungarian / Kuhn-Munkres shortest-augmenting-path assignment.
	// `cost` is G*Q row-major (row=gt index, col=query index); handles the rectangular
	// Q >= G case natively (no square-padding needed). Fills q_match[Q] with the matched
	// gt index for each query, or -1 if that query is unmatched. G == 0 leaves q_match all -1.
	void hungarian_assignment(const std::vector<float> & cost, int G, int Q, std::vector<int> & q_match)
	{
		q_match.assign(Q, -1);
		if (G == 0) return;

		const float INF = 1e30f;
		std::vector<float> u(G + 1, 0.0f), v(Q + 1, 0.0f);
		std::vector<int> p(Q + 1, 0), way(Q + 1, 0);

		for (int i = 1; i <= G; ++i)
		{
			p[0] = i;
			int j0 = 0;
			std::vector<float> minv(Q + 1, INF);
			std::vector<char> used(Q + 1, 0);
			do
			{
				used[j0] = 1;
				const int i0 = p[j0];
				int j1 = -1;
				float delta = INF;
				for (int j = 1; j <= Q; ++j)
				{
					if (used[j]) continue;
					const float cur = cost[(size_t)(i0 - 1) * Q + (j - 1)] - u[i0] - v[j];
					if (cur < minv[j]) { minv[j] = cur; way[j] = j0; }
					if (minv[j] < delta) { delta = minv[j]; j1 = j; }
				}
				for (int j = 0; j <= Q; ++j)
				{
					if (used[j]) { u[p[j]] += delta; v[j] -= delta; }
					else minv[j] -= delta;
				}
				j0 = j1;
			} while (p[j0] != 0);

			do
			{
				const int j1 = way[j0];
				p[j0] = p[j1];
				j0 = j1;
			} while (j0);
		}

		for (int j = 1; j <= Q; ++j)
		{
			if (p[j] > 0) q_match[j - 1] = p[j] - 1;
		}
	}

	int detr_best_class(const Darknet::Layer & l, const float * query, const float thresh, float & best_score)
	{
		int best_class = -1;
		best_score = thresh;
		for (int c = 0; c < l.classes; ++c)
		{
			const float score = detr_sigmoid(query[c]);
			if (score > best_score)
			{
				best_score = score;
				best_class = c;
			}
		}
		return best_class;
	}

	void correct_detr_decoder_boxes(Darknet::Detection * dets, const int n, const int w, const int h, const int netw, const int neth, const int relative, const int letter)
	{
		int new_w = netw;
		int new_h = neth;
		if (letter)
		{
			if ((static_cast<float>(netw) / static_cast<float>(w)) < (static_cast<float>(neth) / static_cast<float>(h)))
			{
				new_h = (h * netw) / w;
			}
			else
			{
				new_w = (w * neth) / h;
			}
		}

		const float deltaw = static_cast<float>(netw - new_w);
		const float deltah = static_cast<float>(neth - new_h);
		const float ratiow = static_cast<float>(new_w) / static_cast<float>(netw);
		const float ratioh = static_cast<float>(new_h) / static_cast<float>(neth);

		for (int i = 0; i < n; ++i)
		{
			Darknet::Box b = dets[i].bbox;
			b.x = (b.x - deltaw * 0.5f / static_cast<float>(netw)) / ratiow;
			b.y = (b.y - deltah * 0.5f / static_cast<float>(neth)) / ratioh;
			b.w *= 1.0f / ratiow;
			b.h *= 1.0f / ratioh;
			if (!relative)
			{
				b.x *= w;
				b.w *= w;
				b.y *= h;
				b.h *= h;
			}
			dets[i].bbox = b;
		}
	}
}

Darknet::Layer make_detr_decoder_layer(int batch, int h, int w, int c,
		int queries, int classes, int heads, int ffn, int max_boxes,
		float cls_weight, float l1_weight, float giou_weight, float noobj_weight,
		int index, int train)
{
	TAT(TATPARMS);

	if (queries < 1)	queries = 1;
	if (classes < 1)	classes = 1;
	if (heads   < 1)	heads   = 1;
	if (ffn     < 1)	ffn     = std::max(1, c);
	if (max_boxes < 1)	max_boxes = 90;

	if (c % heads != 0)
	{
		darknet_fatal_error(DARKNET_LOC, "detr_decoder: model dim (%d) must be divisible by heads (%d)", c, heads);
	}

	Darknet::Layer l = { (Darknet::ELayerType)0 };
	l.type = Darknet::ELayerType::DETR_DECODER;
	l.train = train;
	l.batch = batch;
	l.h = h;
	l.w = w;
	l.c = c;
	l.index = index;

	l.classes = classes;
	l.coords = 4;
	l.detr_queries = queries;
	l.detr_heads = heads;
	l.detr_ffn = ffn;
	l.detr_cls_weight = cls_weight;
	l.detr_l1_weight = l1_weight;
	l.detr_giou_weight = giou_weight;
	l.detr_noobj_weight = std::max(0.0f, noobj_weight);

	l.max_boxes = max_boxes;
	l.truth_size = 4 + 1;					// x,y,w,h,class  (matches yolo-style loader)
	l.truths = max_boxes * l.truth_size;

	// Output: one (class-logits + 4-box) vector per query.
	l.out_h = 1;
	l.out_w = queries;
	l.out_c = classes + 4;
	l.outputs = queries * (classes + 4);
	l.inputs = h * w * c;

	const DetrParams p = detr_params(l);
	l.nweights = (int)p.total;

	l.weights = (float*)xcalloc(p.total, sizeof(float));
	l.weight_updates = (float*)xcalloc(p.total, sizeof(float));

	// Xavier-ish init for the projection / head matrices; biases and box bias stay 0.
	const float sD = std::sqrt(1.0f / std::max(1, c));
	const float sF = std::sqrt(1.0f / std::max(1, ffn));
	auto init_range = [&](size_t off, size_t n, float s)
	{
		for (size_t i = 0; i < n; ++i) l.weights[off + i] = rand_uniform(-s, s);
	};
	init_range(p.off_E,   (size_t)p.Q * p.D, sD);
	init_range(p.off_Wsq, (size_t)p.D * p.D, sD);
	init_range(p.off_Wsk, (size_t)p.D * p.D, sD);
	init_range(p.off_Wsv, (size_t)p.D * p.D, sD);
	init_range(p.off_Wso, (size_t)p.D * p.D, sD);
	init_range(p.off_Wq,  (size_t)p.D * p.D, sD);
	init_range(p.off_Wk,  (size_t)p.D * p.D, sD);
	init_range(p.off_Wv,  (size_t)p.D * p.D, sD);
	init_range(p.off_Wo,  (size_t)p.D * p.D, sD);
	init_range(p.off_W1, (size_t)p.D * p.F, sD);
	init_range(p.off_W2, (size_t)p.F * p.D, sF);
	init_range(p.off_Wc, (size_t)p.C * p.D, sD);
	init_range(p.off_Wb, (size_t)4   * p.D, sD);

	// Reference points (DAB/RF-DETR spatial prior): stored as the pre-sigmoid additive bias
	// a = inverse_sigmoid(ref), so box = sigmoid(boxpre + bb + a). cx,cy are spread over a grid
	// so queries start at distinct locations (the YOLO-like prior); w,h default to a small box.
	{
		auto inv_sig = [](float x) -> float { x = std::clamp(x, 1e-4f, 1.0f - 1e-4f); return std::log(x / (1.0f - x)); };
		const int cols = std::max(1, (int)std::ceil(std::sqrt((float)p.Q)));
		const int rows = std::max(1, (p.Q + cols - 1) / cols);
		for (int q = 0; q < p.Q; ++q)
		{
			const int col = q % cols;
			const int row = (q / cols) % rows;
			float * refq = l.weights + p.off_ref + (size_t)q * 4;
			refq[0] = inv_sig((static_cast<float>(col) + 0.5f) / static_cast<float>(cols));	// cx
			refq[1] = inv_sig((static_cast<float>(row) + 0.5f) / static_cast<float>(rows));	// cy
			refq[2] = inv_sig(0.15f);														// w
			refq[3] = inv_sig(0.15f);														// h
		}
	}

	l.output = (float*)xcalloc((size_t)batch * l.outputs, sizeof(float));
	l.delta  = (float*)xcalloc((size_t)batch * l.outputs, sizeof(float));
	l.cost   = (float*)xcalloc(1, sizeof(float));

	l.forward = forward_detr_decoder_layer;
	l.backward = backward_detr_decoder_layer;
	l.update = update_detr_decoder_layer;

#ifdef DARKNET_GPU
	l.forward_gpu = forward_detr_decoder_layer_gpu;
	l.backward_gpu = backward_detr_decoder_layer_gpu;
	l.update_gpu = update_detr_decoder_layer_gpu;

	if (cfg_and_state.gpu_index >= 0)
	{
		detr_decoder_setup_gpu(l);		// allocates params/scratch on the device, pushes initial weights
	}
#endif

	*cfg_and_state.output
		<< "detr_decoder  queries=" << queries
		<< " classes=" << classes
		<< " dim=" << c
		<< " tokens=" << (h * w)
		<< " ffn=" << ffn
		<< " noobj=" << l.detr_noobj_weight
		<< " params=" << l.nweights << std::endl;

	return l;
}

void forward_detr_decoder_layer(Darknet::Layer & l, Darknet::NetworkState state)
{
	TAT(TATPARMS);

	const DetrParams p = detr_params(l);
	const int D = p.D, Q = p.Q, F = p.F, C = p.C;
	const int N = l.h * l.w;
	const int stride = C + 4;
	const int heads = std::max(1, l.detr_heads);
	const int dh = D / heads;
	const float invsqrtD = 1.0f / std::sqrt((float)D);
	const float invsqrt_dh = 1.0f / std::sqrt((float)dh);

	const float * W = l.weights;
	const float * E   = W + p.off_E;
	const float * Wsq = W + p.off_Wsq;
	const float * Wsk = W + p.off_Wsk;
	const float * Wsv = W + p.off_Wsv;
	const float * Wso = W + p.off_Wso;
	const float * Wq = W + p.off_Wq;
	const float * Wk = W + p.off_Wk;
	const float * Wv = W + p.off_Wv;
	const float * Wo = W + p.off_Wo;
	const float * W1 = W + p.off_W1;
	const float * b1 = W + p.off_b1;
	const float * W2 = W + p.off_W2;
	const float * b2 = W + p.off_b2;
	const float * Wc = W + p.off_Wc;
	const float * bc = W + p.off_bc;
	const float * Wb = W + p.off_Wb;
	const float * bb = W + p.off_bb;
	const float * ref = W + p.off_ref;		// per-query reference points (spatial prior)

	std::vector<float> m(D), keyin(D), pos(D);
	std::vector<float> Kproj((size_t)N * D), Vproj((size_t)N * D);
	std::vector<float> qf(D), qproj(D), scores(N), attn(D), ctx(D), h1(F), hr(F), ffn(D);
	std::vector<float> logits(C), boxpre(4), box(4);

	// Self-attention scratch (queries attend to queries, independent of the memory tokens).
	std::vector<float> Qsa((size_t)Q * D), Ksa((size_t)Q * D), Vsa((size_t)Q * D);
	std::vector<float> sa_ctx((size_t)Q * D), sa_out(D), qf_sa((size_t)Q * D);
	std::vector<float> sa_scores(Q);

	for (int b = 0; b < l.batch; ++b)
	{
		const float * input = state.input + (size_t)b * l.inputs;
		float * out = l.output + (size_t)b * l.outputs;
		float * delta = l.delta + (size_t)b * l.outputs;
		std::fill(delta, delta + l.outputs, 0.0f);

		// Precompute key/value projections for every memory token.
		for (int j = 0; j < N; ++j)
		{
			detr_gather_token(input, j, N, D, m.data());
			detr_posenc(j, D, pos.data());
			for (int d = 0; d < D; ++d) keyin[d] = m[d] + pos[d];
			detr_matvec(Wk, keyin.data(), Kproj.data() + (size_t)j * D, D, D);
			detr_matvec(Wv, m.data(),     Vproj.data() + (size_t)j * D, D, D);
		}

		// ---- self-attention among the Q query embeddings ----
		for (int q = 0; q < Q; ++q)
		{
			const float * Eq = E + (size_t)q * D;
			detr_matvec(Wsq, Eq, Qsa.data() + (size_t)q * D, D, D);
			detr_matvec(Wsk, Eq, Ksa.data() + (size_t)q * D, D, D);
			detr_matvec(Wsv, Eq, Vsa.data() + (size_t)q * D, D, D);
		}
		for (int hd = 0; hd < heads; ++hd)
		{
			const int off = hd * dh;
			for (int q = 0; q < Q; ++q)
			{
				const float * Qh = Qsa.data() + (size_t)q * D + off;
				float smax = -1e30f;
				for (int j = 0; j < Q; ++j)
				{
					const float * Kh = Ksa.data() + (size_t)j * D + off;
					float s = 0.0f;
					for (int d = 0; d < dh; ++d) s += Qh[d] * Kh[d];
					s *= invsqrt_dh;
					sa_scores[j] = s;
					if (s > smax) smax = s;
				}
				float ssum = 0.0f;
				for (int j = 0; j < Q; ++j) { sa_scores[j] = std::exp(sa_scores[j] - smax); ssum += sa_scores[j]; }
				const float inv = 1.0f / (ssum + 1e-9f);
				for (int j = 0; j < Q; ++j) sa_scores[j] *= inv;

				float * ctxq = sa_ctx.data() + (size_t)q * D + off;
				for (int d = 0; d < dh; ++d) ctxq[d] = 0.0f;
				for (int j = 0; j < Q; ++j)
				{
					const float a = sa_scores[j];
					const float * Vh = Vsa.data() + (size_t)j * D + off;
					for (int d = 0; d < dh; ++d) ctxq[d] += a * Vh[d];
				}
			}
		}
		for (int q = 0; q < Q; ++q)
		{
			detr_matvec(Wso, sa_ctx.data() + (size_t)q * D, sa_out.data(), D, D);
			const float * Eq = E + (size_t)q * D;
			float * qfq = qf_sa.data() + (size_t)q * D;
			for (int d = 0; d < D; ++d) qfq[d] = Eq[d] + sa_out[d];	// residual
		}

		for (int q = 0; q < Q; ++q)
		{
			for (int d = 0; d < D; ++d) qf[d] = qf_sa[(size_t)q * D + d];
			detr_matvec(Wq, qf.data(), qproj.data(), D, D);

			float smax = -1e30f;
			for (int j = 0; j < N; ++j)
			{
				const float * Kj = Kproj.data() + (size_t)j * D;
				float s = 0.0f;
				for (int d = 0; d < D; ++d) s += qproj[d] * Kj[d];
				s *= invsqrtD;
				scores[j] = s;
				if (s > smax) smax = s;
			}
			float ssum = 0.0f;
			for (int j = 0; j < N; ++j) { scores[j] = std::exp(scores[j] - smax); ssum += scores[j]; }
			const float inv = 1.0f / (ssum + 1e-9f);
			for (int j = 0; j < N; ++j) scores[j] *= inv;

			std::fill(ctx.begin(), ctx.end(), 0.0f);
			for (int j = 0; j < N; ++j)
			{
				const float a = scores[j];
				const float * Vj = Vproj.data() + (size_t)j * D;
				for (int d = 0; d < D; ++d) ctx[d] += a * Vj[d];
			}

			detr_matvec(Wo, ctx.data(), attn.data(), D, D);
			for (int d = 0; d < D; ++d) attn[d] += qf[d];			// residual

			detr_matvec(W1, attn.data(), h1.data(), F, D);
			for (int f = 0; f < F; ++f) { h1[f] += b1[f]; hr[f] = h1[f] > 0.0f ? h1[f] : 0.0f; }

			detr_matvec(W2, hr.data(), ffn.data(), D, F);
			for (int d = 0; d < D; ++d) ffn[d] += b2[d] + attn[d];	// residual

			detr_matvec(Wc, ffn.data(), logits.data(), C, D);
			for (int k = 0; k < C; ++k) logits[k] += bc[k];
			detr_matvec(Wb, ffn.data(), boxpre.data(), 4, D);
			const float * refq = ref + (size_t)q * 4;	// per-query reference (spatial prior)
			for (int i = 0; i < 4; ++i) { boxpre[i] += bb[i] + refq[i]; box[i] = detr_sigmoid(boxpre[i]); }

			for (int k = 0; k < C; ++k) out[q * stride + k] = logits[k];
			for (int i = 0; i < 4; ++i) out[q * stride + C + i] = box[i];
		}
	}

	if (state.train && state.truth)
	{
		const float cost = detr_decoder_loss(l, state.truth);
		if (l.cost) *l.cost = cost;
	}
	else if (l.cost)
	{
		*l.cost = 0.0f;
	}
}

float detr_decoder_loss(Darknet::Layer & l, const float * truth)
{
	TAT(TATPARMS);

	const int Q = l.detr_queries;
	const int C = l.classes;
	const int stride = C + 4;
	const float noobj_weight = std::max(0.0f, l.detr_noobj_weight);
	// Sigmoid focal loss (Deformable-DETR / RF-DETR classification). gamma modulates by
	// difficulty so the ~Q*C easy background negatives per image stop swamping the sparse
	// positives; alpha is the standard foreground/background balance.
	const float focal_gamma = 2.0f;
	const float focal_alpha = 0.25f;
	float total_cost = 0.0f;

	for (int b = 0; b < l.batch; ++b)
	{
		const float * out = l.output + (size_t)b * l.outputs;	// logits (raw) + box (sigmoid'd)
		float * delta = l.delta + (size_t)b * l.outputs;
		std::fill(delta, delta + l.outputs, 0.0f);

		// ---- gather ground-truth boxes for this image ----
		const float * tr = truth + (size_t)b * l.truths;
		std::vector<int> gt_class;
		std::vector<std::array<float,4>> gt_box;
		for (int t = 0; t < l.max_boxes; ++t)
		{
			const float * tb = tr + t * l.truth_size;
			if (tb[2] < 1e-6f && tb[3] < 1e-6f) continue;		// empty slot
			gt_box.push_back({tb[0], tb[1], tb[2], tb[3]});
			gt_class.push_back((int)tb[4]);
		}

		// ---- optimal one-to-one matching (Hungarian / Kuhn-Munkres) ----
		// Same cost formula as before (L1 box + (1-GIoU) + (1-p_class)), but now solved
		// as a globally optimal bipartite assignment instead of a greedy scan, which could
		// let an early GT box grab the only good match for a later one.
		std::vector<int> q_match;						// query -> gt (or -1)
		if (!gt_box.empty())
		{
			std::vector<float> cost_matrix((size_t)gt_box.size() * Q);
			for (size_t g = 0; g < gt_box.size(); ++g)
			{
				for (int q = 0; q < Q; ++q)
				{
					const float * pb = out + q * stride + C;
					const float l1 = detr_l1_box(pb, gt_box[g].data());
					const float pc = std::clamp(detr_sigmoid(out[q * stride + gt_class[g]]), 1e-6f, 1.0f - 1e-6f);
					const Darknet::Box pb_box{pb[0], pb[1], pb[2], pb[3]};
					const Darknet::Box gtb{gt_box[g][0], gt_box[g][1], gt_box[g][2], gt_box[g][3]};
					// Focal classification cost (Deformable-DETR): pos_cost - neg_cost at the gt class,
					// consistent with the focal training loss below so matching and loss agree.
					const float pos_cost = focal_alpha * std::pow(1.0f - pc, focal_gamma) * (-std::log(pc));
					const float neg_cost = (1.0f - focal_alpha) * std::pow(pc, focal_gamma) * (-std::log(1.0f - pc));
					const float cls_cost = pos_cost - neg_cost;
					cost_matrix[g * (size_t)Q + q] = l1 * l.detr_l1_weight + (1.0f - box_giou(pb_box, gtb)) * l.detr_giou_weight + cls_cost * l.detr_cls_weight;
				}
			}
			hungarian_assignment(cost_matrix, (int)gt_box.size(), Q, q_match);
		}
		else
		{
			q_match.assign(Q, -1);
		}

		const float box_norm = 1.0f / std::max<size_t>(1, gt_box.size());
		// Normalize classification by the number of GT boxes only (DETR convention). The old
		// extra 1/C factor additionally starved the sparse positive gradient by C x.
		const float cls_norm = 1.0f / static_cast<float>(std::max<size_t>(1, gt_box.size()));

		// ---- per-query classification (all) + box (matched) loss/deltas ----
		for (int q = 0; q < Q; ++q)
		{
			const int g = q_match[q];
			const float query_cls_weight = (g >= 0) ? 1.0f : noobj_weight;
			for (int k = 0; k < C; ++k)
			{
				const float pk = std::clamp(detr_sigmoid(out[q * stride + k]), 1e-6f, 1.0f - 1e-6f);
				const float target = (g >= 0 && gt_class[g] == k) ? 1.0f : 0.0f;
				const float scale = l.detr_cls_weight * query_cls_weight * cls_norm;
				// Sigmoid focal loss. delta holds -dL/dlogit; the (1-p)^gamma / p^gamma modulation
				// crushes easy negatives so the matched-query positive is no longer swamped.
				if (target > 0.5f)
				{
					const float focal = std::pow(1.0f - pk, focal_gamma);
					delta[q * stride + k] = focal_alpha * focal * (1.0f - pk) * scale;			// -dL/dlogit (>0, push up)
					total_cost += focal_alpha * focal * (-std::log(pk)) * scale;
				}
				else
				{
					const float focal = std::pow(pk, focal_gamma);
					delta[q * stride + k] = -(1.0f - focal_alpha) * focal * pk * scale;			// -dL/dlogit (<0, push down)
					total_cost += (1.0f - focal_alpha) * focal * (-std::log(1.0f - pk)) * scale;
				}
			}
			if (g >= 0)
			{
				for (int i = 0; i < 4; ++i)
				{
					const float pb = out[q * stride + C + i];
					const float gb = gt_box[g][i];
					const float diff = pb - gb;
					const float sgn = (diff > 0.0f) ? 1.0f : ((diff < 0.0f) ? -1.0f : 0.0f);
					// L1 through sigmoid': -dL/dboxpre = -sign(diff) * pb*(1-pb)
					delta[q * stride + C + i] = -sgn * pb * (1.0f - pb) * l.detr_l1_weight * box_norm;
					total_cost += std::fabs(diff) * l.detr_l1_weight * box_norm;
				}

				// ---- GIoU term (paper-consistent: DETR/RF-DETR combine L1 + GIoU) ----
				const Darknet::Box pred{
					out[q * stride + C + 0], out[q * stride + C + 1],
					out[q * stride + C + 2], out[q * stride + C + 3]};
				const Darknet::Box truth_box{gt_box[g][0], gt_box[g][1], gt_box[g][2], gt_box[g][3]};
				const dxrep giou_grad = dx_box_iou(pred, truth_box, GIOU);
				const float d_giou[4] = {giou_grad.dt, giou_grad.db, giou_grad.dl, giou_grad.dr};	// dx,dy,dw,dh
				for (int i = 0; i < 4; ++i)
				{
					const float pb = out[q * stride + C + i];
					delta[q * stride + C + i] += d_giou[i] * pb * (1.0f - pb) * l.detr_giou_weight * box_norm;
				}
				total_cost += (1.0f - box_giou(pred, truth_box)) * l.detr_giou_weight * box_norm;
			}
		}
	}

	return total_cost;
}

int detr_decoder_num_detections_batch(const Darknet::Layer & l, const float thresh, const int batch)
{
	TAT(TATPARMS);

	const int stride = l.classes + 4;
	const float * out = l.output + (size_t)batch * l.outputs;
	int count = 0;
	for (int q = 0; q < l.detr_queries; ++q)
	{
		float best_score = 0.0f;
		const int best_class = detr_best_class(l, out + (size_t)q * stride, thresh, best_score);
		if (best_class >= 0)
		{
			++count;
		}
	}
	return count;
}

int detr_decoder_num_detections(const Darknet::Layer & l, const float thresh)
{
	return detr_decoder_num_detections_batch(l, thresh, 0);
}

int get_detr_decoder_detections_batch(
	const Darknet::Layer & l,
	const int w,
	const int h,
	const int netw,
	const int neth,
	const float thresh,
	int * map,
	const int relative,
	Darknet::Detection * dets,
	const int letter,
	const int batch)
{
	TAT(TATPARMS);
	(void)map;

	const int stride = l.classes + 4;
	const float * out = l.output + (size_t)batch * l.outputs;
	int count = 0;
	for (int q = 0; q < l.detr_queries; ++q)
	{
		const float * query = out + (size_t)q * stride;
		float best_score = 0.0f;
		const int best_class = detr_best_class(l, query, thresh, best_score);
		if (best_class < 0)
		{
			continue;
		}

		Darknet::Detection & det = dets[count];
		det.bbox.x = query[l.classes + 0];
		det.bbox.y = query[l.classes + 1];
		det.bbox.w = query[l.classes + 2];
		det.bbox.h = query[l.classes + 3];
		det.objectness = best_score;
		det.classes = l.classes;
		det.best_class_idx = best_class;

		for (int c = 0; c < l.classes; ++c)
		{
			const float score = detr_sigmoid(query[c]);
			det.prob[c] = (score > thresh) ? score : 0.0f;
		}
		++count;
	}

	correct_detr_decoder_boxes(dets, count, w, h, netw, neth, relative, letter);
	return count;
}

int get_detr_decoder_detections(
	const Darknet::Layer & l,
	const int w,
	const int h,
	const int netw,
	const int neth,
	const float thresh,
	int * map,
	const int relative,
	Darknet::Detection * dets,
	const int letter)
{
	return get_detr_decoder_detections_batch(l, w, h, netw, neth, thresh, map, relative, dets, letter, 0);
}

void backward_detr_decoder_layer(Darknet::Layer & l, Darknet::NetworkState state)
{
	TAT(TATPARMS);

	const DetrParams p = detr_params(l);
	const int D = p.D, Q = p.Q, F = p.F, C = p.C;
	const int N = l.h * l.w;
	const int stride = C + 4;
	const int heads = std::max(1, l.detr_heads);
	const int dh = D / heads;
	const float invsqrtD = 1.0f / std::sqrt((float)D);
	const float invsqrt_dh = 1.0f / std::sqrt((float)dh);

	const float * W = l.weights;
	const float * E   = W + p.off_E;
	const float * Wsq = W + p.off_Wsq;
	const float * Wsk = W + p.off_Wsk;
	const float * Wsv = W + p.off_Wsv;
	const float * Wso = W + p.off_Wso;
	const float * Wq = W + p.off_Wq;
	const float * Wk = W + p.off_Wk;
	const float * Wv = W + p.off_Wv;
	const float * Wo = W + p.off_Wo;
	const float * W1 = W + p.off_W1;
	const float * b1 = W + p.off_b1;
	const float * W2 = W + p.off_W2;
	const float * b2 = W + p.off_b2;
	const float * Wc = W + p.off_Wc;
	const float * Wb = W + p.off_Wb;

	float * G = l.weight_updates;
	float * dE   = G + p.off_E;
	float * dWsq = G + p.off_Wsq;
	float * dWsk = G + p.off_Wsk;
	float * dWsv = G + p.off_Wsv;
	float * dWso = G + p.off_Wso;
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
	float * dref = G + p.off_ref;		// per-query reference-point gradient

	std::vector<float> m(D), keyin(D), pos(D);
	std::vector<float> Kproj((size_t)N * D), Vproj((size_t)N * D), keyin_all((size_t)N * D);
	std::vector<float> qf(D), qproj(D), scores(N), attn(D), ctx(D), h1(F), hr(F), ffn(D);
	// gradients
	std::vector<float> d_ffn(D), d_attn(D), d_ctx(D), d_qf(D), d_qproj(D), d_hr(F), d_h1(F);
	std::vector<float> d_scores(N), d_a(N);
	// Backbone-gradient scratch (reused across tokens/queries to avoid per-iteration allocation).
	std::vector<float> g_val(D), d_m(D), g_key(D), d_keyin(D);

	// Self-attention forward-recompute + gradient scratch (mirrors the cross-attention style above).
	std::vector<float> Qsa((size_t)Q * D), Ksa((size_t)Q * D), Vsa((size_t)Q * D);
	std::vector<float> sa_ctx((size_t)Q * D), sa_out(D), qf_sa((size_t)Q * D);
	std::vector<float> d_qf_sa((size_t)Q * D);					// accumulated across the per-query loop below
	std::vector<float> d_sa_ctx((size_t)Q * D);
	std::vector<float> d_Qsa((size_t)Q * D), d_Ksa((size_t)Q * D), d_Vsa((size_t)Q * D);
	std::vector<float> d_sa_scores(Q), d_sa_a(Q);

	for (int b = 0; b < l.batch; ++b)
	{
		const float * input = state.input + (size_t)b * l.inputs;
		const float * delta = l.delta + (size_t)b * l.outputs;
		float * delta_b = state.delta ? state.delta + (size_t)b * l.inputs : nullptr;

		for (int j = 0; j < N; ++j)
		{
			detr_gather_token(input, j, N, D, m.data());
			detr_posenc(j, D, pos.data());
			float * ki = keyin_all.data() + (size_t)j * D;
			for (int d = 0; d < D; ++d) ki[d] = m[d] + pos[d];
			detr_matvec(Wk, ki,        Kproj.data() + (size_t)j * D, D, D);
			detr_matvec(Wv, m.data(),  Vproj.data() + (size_t)j * D, D, D);
		}

		// ---- recompute self-attention forward (same math as forward_detr_decoder_layer) ----
		for (int q = 0; q < Q; ++q)
		{
			const float * Eq = E + (size_t)q * D;
			detr_matvec(Wsq, Eq, Qsa.data() + (size_t)q * D, D, D);
			detr_matvec(Wsk, Eq, Ksa.data() + (size_t)q * D, D, D);
			detr_matvec(Wsv, Eq, Vsa.data() + (size_t)q * D, D, D);
		}
		// Cache every head's softmax'd attention row (needed again during self-attention backward below).
		std::vector<float> sa_scores_all((size_t)heads * Q * Q);
		for (int hd = 0; hd < heads; ++hd)
		{
			const int off = hd * dh;
			for (int q = 0; q < Q; ++q)
			{
				float * srow = sa_scores_all.data() + ((size_t)hd * Q + q) * Q;
				const float * Qh = Qsa.data() + (size_t)q * D + off;
				float smax = -1e30f;
				for (int j = 0; j < Q; ++j)
				{
					const float * Kh = Ksa.data() + (size_t)j * D + off;
					float s = 0.0f;
					for (int d = 0; d < dh; ++d) s += Qh[d] * Kh[d];
					s *= invsqrt_dh;
					srow[j] = s;
					if (s > smax) smax = s;
				}
				float ssum = 0.0f;
				for (int j = 0; j < Q; ++j) { srow[j] = std::exp(srow[j] - smax); ssum += srow[j]; }
				const float inv = 1.0f / (ssum + 1e-9f);
				for (int j = 0; j < Q; ++j) srow[j] *= inv;

				float * ctxq = sa_ctx.data() + (size_t)q * D + off;
				for (int d = 0; d < dh; ++d) ctxq[d] = 0.0f;
				for (int j = 0; j < Q; ++j)
				{
					const float a = srow[j];
					const float * Vh = Vsa.data() + (size_t)j * D + off;
					for (int d = 0; d < dh; ++d) ctxq[d] += a * Vh[d];
				}
			}
		}
		for (int q = 0; q < Q; ++q)
		{
			detr_matvec(Wso, sa_ctx.data() + (size_t)q * D, sa_out.data(), D, D);
			const float * Eq = E + (size_t)q * D;
			float * qfq = qf_sa.data() + (size_t)q * D;
			for (int d = 0; d < D; ++d) qfq[d] = Eq[d] + sa_out[d];
		}
		std::fill(d_qf_sa.begin(), d_qf_sa.end(), 0.0f);

		for (int q = 0; q < Q; ++q)
		{
			// ---- recompute forward intermediates for this query ----
			for (int d = 0; d < D; ++d) qf[d] = qf_sa[(size_t)q * D + d];
			detr_matvec(Wq, qf.data(), qproj.data(), D, D);

			float smax = -1e30f;
			for (int j = 0; j < N; ++j)
			{
				const float * Kj = Kproj.data() + (size_t)j * D;
				float s = 0.0f;
				for (int d = 0; d < D; ++d) s += qproj[d] * Kj[d];
				s *= invsqrtD;
				scores[j] = s;
				if (s > smax) smax = s;
			}
			float ssum = 0.0f;
			for (int j = 0; j < N; ++j) { scores[j] = std::exp(scores[j] - smax); ssum += scores[j]; }
			const float invs = 1.0f / (ssum + 1e-9f);
			for (int j = 0; j < N; ++j) scores[j] *= invs;

			std::fill(ctx.begin(), ctx.end(), 0.0f);
			for (int j = 0; j < N; ++j)
			{
				const float a = scores[j];
				const float * Vj = Vproj.data() + (size_t)j * D;
				for (int d = 0; d < D; ++d) ctx[d] += a * Vj[d];
			}
			detr_matvec(Wo, ctx.data(), attn.data(), D, D);
			for (int d = 0; d < D; ++d) attn[d] += qf[d];
			detr_matvec(W1, attn.data(), h1.data(), F, D);
			for (int f = 0; f < F; ++f) { h1[f] += b1[f]; hr[f] = h1[f] > 0.0f ? h1[f] : 0.0f; }
			detr_matvec(W2, hr.data(), ffn.data(), D, F);
			for (int d = 0; d < D; ++d) ffn[d] += b2[d] + attn[d];	// match forward exactly

			// ---- backward ----
			const float * d_logits = delta + q * stride;			// class part (negative grad)
			const float * d_boxpre = delta + q * stride + C;		// box pre-sigmoid part

			std::fill(d_ffn.begin(), d_ffn.end(), 0.0f);
			// class head: ffn -> logits
			detr_accum_outer(dWc, d_logits, ffn.data(), C, D);
			for (int k = 0; k < C; ++k) dbc[k] += d_logits[k];
			detr_accum_matTvec(Wc, d_logits, d_ffn.data(), C, D);
			// box head: ffn -> boxpre
			detr_accum_outer(dWb, d_boxpre, ffn.data(), 4, D);
			for (int i = 0; i < 4; ++i) dbb[i] += d_boxpre[i];
			// Reference point enters the pre-sigmoid additively (box = sigmoid(boxpre+bb+ref_q)),
			// so its gradient is exactly the box pre-sigmoid delta, accumulated per query.
			float * dref_q = dref + (size_t)q * 4;
			for (int i = 0; i < 4; ++i) dref_q[i] += d_boxpre[i];
			detr_accum_matTvec(Wb, d_boxpre, d_ffn.data(), 4, D);

			// ffn = attn + W2*hr + b2
			std::fill(d_attn.begin(), d_attn.end(), 0.0f);
			std::fill(d_hr.begin(), d_hr.end(), 0.0f);
			detr_accum_outer(dW2, d_ffn.data(), hr.data(), D, F);
			for (int d = 0; d < D; ++d) db2[d] += d_ffn[d];
			detr_accum_matTvec(W2, d_ffn.data(), d_hr.data(), D, F);
			for (int d = 0; d < D; ++d) d_attn[d] += d_ffn[d];		// residual

			// hr = relu(h1)
			for (int f = 0; f < F; ++f) d_h1[f] = (h1[f] > 0.0f) ? d_hr[f] : 0.0f;
			// h1 = W1*attn + b1
			detr_accum_outer(dW1, d_h1.data(), attn.data(), F, D);
			for (int f = 0; f < F; ++f) db1[f] += d_h1[f];
			detr_accum_matTvec(W1, d_h1.data(), d_attn.data(), F, D);

			// attn = qf + Wo*ctx
			std::fill(d_qf.begin(), d_qf.end(), 0.0f);
			std::fill(d_ctx.begin(), d_ctx.end(), 0.0f);
			for (int d = 0; d < D; ++d) d_qf[d] += d_attn[d];		// residual
			detr_accum_outer(dWo, d_attn.data(), ctx.data(), D, D);
			detr_accum_matTvec(Wo, d_attn.data(), d_ctx.data(), D, D);

			// ctx = sum_j a_j * Vproj_j ; Vproj_j = Wv * m_j
			for (int j = 0; j < N; ++j)
			{
				const float * Vj = Vproj.data() + (size_t)j * D;
				float da = 0.0f;
				for (int d = 0; d < D; ++d) da += d_ctx[d] * Vj[d];
				d_a[j] = da;
				// dWv += outer(a_j * d_ctx, m_j)
				const float a = scores[j];
				detr_gather_token(input, j, N, D, m.data());
				for (int r = 0; r < D; ++r)
				{
					const float gr = a * d_ctx[r];
					g_val[r] = gr;
					float * dWvr = dWv + (size_t)r * D;
					for (int cc = 0; cc < D; ++cc) dWvr[cc] += gr * m[cc];
				}
				// Backbone gradient through the value path: d_m = Wv^T * g_val, scattered into state.delta.
				if (delta_b)
				{
					std::fill(d_m.begin(), d_m.end(), 0.0f);
					detr_accum_matTvec(Wv, g_val.data(), d_m.data(), D, D);
					for (int c = 0; c < D; ++c) delta_b[(size_t)c * N + j] += d_m[c];
				}
			}

			// softmax backward: d_scores = a * (d_a - sum_k a_k d_a_k)
			float dot = 0.0f;
			for (int j = 0; j < N; ++j) dot += scores[j] * d_a[j];
			for (int j = 0; j < N; ++j) d_scores[j] = scores[j] * (d_a[j] - dot);

			// scores_j = (qproj . Kproj_j) * invsqrtD ; Kproj_j = Wk * keyin_j
			std::fill(d_qproj.begin(), d_qproj.end(), 0.0f);
			for (int j = 0; j < N; ++j)
			{
				const float ds = d_scores[j] * invsqrtD;
				const float * Kj = Kproj.data() + (size_t)j * D;
				const float * ki = keyin_all.data() + (size_t)j * D;
				for (int d = 0; d < D; ++d) d_qproj[d] += ds * Kj[d];
				// dKproj_j = ds * qproj ; dWk += outer(dKproj_j, keyin_j)
				for (int r = 0; r < D; ++r)
				{
					const float gr = ds * qproj[r];
					g_key[r] = gr;
					float * dWkr = dWk + (size_t)r * D;
					for (int cc = 0; cc < D; ++cc) dWkr[cc] += gr * ki[cc];
				}
				// Backbone gradient through the key path: d_keyin = Wk^T * g_key, scattered into state.delta.
				// keyin_j = m_j + pos_j and pos_j is a constant (no learnable/input-derived term), so the
				// gradient passes straight through to m_j unchanged.
				if (delta_b)
				{
					std::fill(d_keyin.begin(), d_keyin.end(), 0.0f);
					detr_accum_matTvec(Wk, g_key.data(), d_keyin.data(), D, D);
					for (int c = 0; c < D; ++c) delta_b[(size_t)c * N + j] += d_keyin[c];
				}
			}

			// qproj = Wq * qf  (qf = qf_sa[q], the post-self-attention query embedding)
			detr_accum_outer(dWq, d_qproj.data(), qf.data(), D, D);
			detr_accum_matTvec(Wq, d_qproj.data(), d_qf.data(), D, D);

			// qf = qf_sa[q] -- accumulate into d_qf_sa; the self-attention backward pass below
			// carries this the rest of the way back to dE and dWsq/dWsk/dWsv/dWso.
			float * d_qf_sa_q = d_qf_sa.data() + (size_t)q * D;
			for (int d = 0; d < D; ++d) d_qf_sa_q[d] += d_qf[d];
		}

		// ---- self-attention backward: qf_sa[q] = E[q] + Wso * sa_ctx[q] ----
		for (int q = 0; q < Q; ++q)
		{
			const float * d_qfq = d_qf_sa.data() + (size_t)q * D;	// = d(Wso*sa_ctx) for this query (residual, no elementwise op)
			float * dEq = dE + (size_t)q * D;
			for (int d = 0; d < D; ++d) dEq[d] += d_qfq[d];		// residual: E[q] path

			detr_accum_outer(dWso, d_qfq, sa_ctx.data() + (size_t)q * D, D, D);
			float * d_sa_ctx_q = d_sa_ctx.data() + (size_t)q * D;
			std::fill(d_sa_ctx_q, d_sa_ctx_q + D, 0.0f);
			detr_accum_matTvec(Wso, d_qfq, d_sa_ctx_q, D, D);
		}

		std::fill(d_Qsa.begin(), d_Qsa.end(), 0.0f);
		std::fill(d_Ksa.begin(), d_Ksa.end(), 0.0f);
		std::fill(d_Vsa.begin(), d_Vsa.end(), 0.0f);

		for (int hd = 0; hd < heads; ++hd)
		{
			const int off = hd * dh;
			for (int q = 0; q < Q; ++q)
			{
				const float * srow = sa_scores_all.data() + ((size_t)hd * Q + q) * Q;
				const float * d_ctxq = d_sa_ctx.data() + (size_t)q * D + off;

				// sa_ctx[q,head] = sum_j srow[j] * Vsa[j,head]
				for (int j = 0; j < Q; ++j)
				{
					const float * Vh = Vsa.data() + (size_t)j * D + off;
					float da = 0.0f;
					for (int d = 0; d < dh; ++d) da += d_ctxq[d] * Vh[d];
					d_sa_a[j] = da;

					float * d_Vh = d_Vsa.data() + (size_t)j * D + off;
					const float a = srow[j];
					for (int d = 0; d < dh; ++d) d_Vh[d] += a * d_ctxq[d];
				}

				// softmax backward
				float dot = 0.0f;
				for (int j = 0; j < Q; ++j) dot += srow[j] * d_sa_a[j];
				for (int j = 0; j < Q; ++j) d_sa_scores[j] = srow[j] * (d_sa_a[j] - dot);

				// srow[j] = invsqrt_dh * Qsa[q,head] . Ksa[j,head]
				float * d_Qh = d_Qsa.data() + (size_t)q * D + off;
				const float * Qh = Qsa.data() + (size_t)q * D + off;
				for (int j = 0; j < Q; ++j)
				{
					const float ds = d_sa_scores[j] * invsqrt_dh;
					const float * Kh = Ksa.data() + (size_t)j * D + off;
					float * d_Kh = d_Ksa.data() + (size_t)j * D + off;
					for (int d = 0; d < dh; ++d)
					{
						d_Qh[d] += ds * Kh[d];
						d_Kh[d] += ds * Qh[d];
					}
				}
			}
		}

		for (int q = 0; q < Q; ++q)
		{
			const float * Eq = E + (size_t)q * D;
			float * dEq = dE + (size_t)q * D;

			detr_accum_outer(dWsq, d_Qsa.data() + (size_t)q * D, Eq, D, D);
			detr_accum_matTvec(Wsq, d_Qsa.data() + (size_t)q * D, dEq, D, D);

			detr_accum_outer(dWsk, d_Ksa.data() + (size_t)q * D, Eq, D, D);
			detr_accum_matTvec(Wsk, d_Ksa.data() + (size_t)q * D, dEq, D, D);

			detr_accum_outer(dWsv, d_Vsa.data() + (size_t)q * D, Eq, D, D);
			detr_accum_matTvec(Wsv, d_Vsa.data() + (size_t)q * D, dEq, D, D);
		}
	}
	// Backbone gradient is accumulated into state.delta (via delta_b) in the value-path and
	// key-path loops above, exactly analogous to backward_yolo_layer's axpy_cpu(...,l.delta,1,state.delta,1)
	// and tucker_attention_layer.cpp's state_delta_b scatter. Self-attention among queries never
	// touches the backbone: it is purely a function of E and the self-attention weights.
}

void update_detr_decoder_layer(Darknet::Layer & l, int batch, float learning_rate, float momentum, float decay)
{
	TAT(TATPARMS);

	const int n = l.nweights;
	const float rate = learning_rate / std::max(1, batch);
	for (int i = 0; i < n; ++i)
	{
		l.weight_updates[i] += -decay * batch * l.weights[i];	// weight decay
		l.weights[i] += rate * l.weight_updates[i];				// gradient descent (updates hold -dL/dw)
		l.weight_updates[i] *= momentum;
	}
}

void resize_detr_decoder_layer(Darknet::Layer * l, int w, int h)
{
	TAT(TATPARMS);

	// Only the number of memory tokens (h*w) changes; the parameter set and the
	// per-query output size are independent of input resolution.
	l->w = w;
	l->h = h;
	l->inputs = h * w * l->c;

#ifdef DARKNET_GPU
	if (cfg_and_state.gpu_index >= 0)
	{
		detr_decoder_resize_gpu(*l);		// token-count (h*w) changed -> resize GPU scratch
	}
#endif
}

void save_detr_decoder_weights(const Darknet::Layer & l, FILE * fp)
{
	TAT(TATPARMS);

#ifdef DARKNET_GPU
	// During GPU training the device copy is authoritative — pull it before writing.
	if (cfg_and_state.gpu_index >= 0 && l.weights_gpu)
	{
		cuda_pull_array(l.weights_gpu, const_cast<float*>(l.weights), l.nweights);
	}
#endif

	fwrite(l.weights, sizeof(float), l.nweights, fp);
}

size_t load_detr_decoder_weights(Darknet::Layer & l, FILE * fp)
{
	TAT(TATPARMS);

	const size_t n = fread(l.weights, sizeof(float), l.nweights, fp);

#ifdef DARKNET_GPU
	if (cfg_and_state.gpu_index >= 0 && l.weights_gpu)
	{
		cuda_push_array(l.weights_gpu, l.weights, l.nweights);
	}
#endif

	return n * sizeof(float);
}

// The GPU forward/backward/update and the setup/resize helpers live in
// detr_decoder_kernels.cu (real CUDA via gemm_gpu + elementwise kernels).
