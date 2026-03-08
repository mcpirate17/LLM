/*
 * binding_stubs.cpp — CPU reference implementations for ops declared in
 * kernels.h and referenced by bindings.cpp.
 *
 * These are full implementations (not stubs). They compose existing kernels
 * where possible and implement algorithms directly otherwise.
 */
#include "kernels_common.h"
#include <algorithm>
#include <cstdlib>

/* ══════════════════════════════════════════════════════════════════════
 * Masking / Structural
 * ══════════════════════════════════════════════════════════════════════ */

void aria_causal_mask_f32(const float *x, float *y,
                           int64_t batch, int64_t seq, int64_t dim) {
    /* Apply causal mask: for position i, zero out positions j > i.
     * Treats x as [batch, seq, seq] attention scores (dim == seq). */
    int64_t total = batch * seq * dim;
    if (x != y) memcpy(y, x, total * sizeof(float));
#ifdef ARIA_HAS_OPENMP
    #pragma omp parallel for collapse(2) if(batch * seq > 64) schedule(static)
#endif
    for (int64_t b = 0; b < batch; b++) {
        for (int64_t i = 0; i < seq; i++) {
            float *row = y + b * seq * dim + i * dim;
            for (int64_t j = i + 1; j < dim; j++) {
                row[j] = -1e9f;
            }
        }
    }
}

void aria_sliding_window_mask_f32(const float *x, float *y,
                                    int64_t batch, int64_t seq, int64_t dim,
                                    int64_t window_size) {
    /* Exponential decay mask: y[b,i,j] = x[b,i,j] * exp(-|i-j|/window) */
    int64_t ws = window_size > 0 ? window_size : 1;
    float inv_w = 1.0f / (float)ws;
#ifdef ARIA_HAS_OPENMP
    #pragma omp parallel for collapse(2) if(batch * seq > 64) schedule(static)
#endif
    for (int64_t b = 0; b < batch; b++) {
        for (int64_t i = 0; i < seq; i++) {
            const float *xr = x + b * seq * dim + i * dim;
            float *yr = y + b * seq * dim + i * dim;
            for (int64_t j = 0; j < dim; j++) {
                int64_t dist = i > j ? i - j : j - i;
                yr[j] = xr[j] * expf(-(float)dist * inv_w);
            }
        }
    }
}

void aria_sort_seq_f32(const float *x, float *y, int64_t *indices,
                        int64_t batch, int64_t seq, int64_t dim) {
    /* Sort along sequence dim by L2 norm of each token. Stable insertion sort. */
    float *norms = (float *)malloc(seq * sizeof(float));
    int64_t *idx = (int64_t *)malloc(seq * sizeof(int64_t));
    if (!norms || !idx) {
        if (x != y) memcpy(y, x, batch * seq * dim * sizeof(float));
        if (indices) for (int64_t i = 0; i < batch * seq; i++) indices[i] = i % seq;
        free(norms); free(idx);
        return;
    }
    for (int64_t b = 0; b < batch; b++) {
        const float *xb = x + b * seq * dim;
        float *yb = y + b * seq * dim;
        /* Compute norms */
        for (int64_t s = 0; s < seq; s++) {
            float norm = 0.0f;
            for (int64_t d = 0; d < dim; d++) norm += xb[s * dim + d] * xb[s * dim + d];
            norms[s] = norm;
            idx[s] = s;
        }
        /* Insertion sort (stable, good for small seq) */
        for (int64_t i = 1; i < seq; i++) {
            float key = norms[i];
            int64_t ki = idx[i];
            int64_t j = i - 1;
            while (j >= 0 && norms[j] > key) {
                norms[j + 1] = norms[j];
                idx[j + 1] = idx[j];
                j--;
            }
            norms[j + 1] = key;
            idx[j + 1] = ki;
        }
        /* Scatter sorted tokens */
        for (int64_t s = 0; s < seq; s++) {
            memcpy(yb + s * dim, xb + idx[s] * dim, dim * sizeof(float));
            if (indices) indices[b * seq + s] = idx[s];
        }
    }
    free(norms);
    free(idx);
}

void aria_conv1d_seq_f32(const float *x, const float *weight, const float *bias,
                          float *y, int64_t batch, int64_t seq, int64_t dim) {
    /* Depthwise 1D conv along sequence dim, kernel_size=3, padding=1 */
#ifdef ARIA_HAS_OPENMP
    #pragma omp parallel for collapse(2) if(batch * dim > 64) schedule(static)
#endif
    for (int64_t b = 0; b < batch; b++) {
        for (int64_t d = 0; d < dim; d++) {
            for (int64_t s = 0; s < seq; s++) {
                float acc = 0.0f;
                for (int k = 0; k < 3; k++) {
                    int64_t si = s + k - 1; /* padding=1 */
                    if (si >= 0 && si < seq) {
                        float w = weight ? weight[d * 3 + k] : (k == 1 ? 1.0f : 0.0f);
                        acc += x[b * seq * dim + si * dim + d] * w;
                    }
                }
                if (bias) acc += bias[d];
                y[b * seq * dim + s * dim + d] = acc;
            }
        }
    }
}

/* ══════════════════════════════════════════════════════════════════════
 * Fused Ops (compose existing kernels)
 * ══════════════════════════════════════════════════════════════════════ */

void aria_fused_linear_gelu_f32(const float *x, const float *W, const float *bias,
                                  float *y, int64_t batch, int64_t dim_in, int64_t dim_out) {
    aria_linear_f32(x, W, bias, y, batch, dim_in, dim_out);
    aria_gelu_f32(y, y, batch * dim_out);
}

void aria_matmul_relu_f32(const float *A, const float *B, float *C,
                           int64_t M, int64_t K, int64_t N) {
    aria_matmul_f32(A, B, C, M, K, N);
    aria_relu_f32(C, C, M * N);
}

void aria_matmul_bias_relu_f32(const float *A, const float *B, const float *bias,
                                float *C, int64_t M, int64_t K, int64_t N) {
    aria_matmul_f32(A, B, C, M, K, N);
    if (bias) {
        for (int64_t i = 0; i < M; i++)
            for (int64_t j = 0; j < N; j++)
                C[i * N + j] += bias[j];
    }
    aria_relu_f32(C, C, M * N);
}

void aria_matmul_gelu_f32(const float *A, const float *B, float *C,
                           int64_t M, int64_t K, int64_t N) {
    aria_matmul_f32(A, B, C, M, K, N);
    aria_gelu_f32(C, C, M * N);
}

void aria_layernorm_residual_f32(const float *x, const float *residual,
                                  const float *gamma, const float *beta,
                                  float *y, int64_t rows, int64_t cols,
                                  float eps) {
    for (int64_t i = 0; i < rows * cols; i++) y[i] = x[i] + residual[i];
    aria_layernorm_f32(y, gamma, beta, y, rows, cols, eps);
}

/* ══════════════════════════════════════════════════════════════════════
 * SwiGLU MLP
 * ══════════════════════════════════════════════════════════════════════ */

void aria_swiglu_f32(const float *x,
                      const float *W_gate, const float *W_up, const float *W_down,
                      const float *bias_gate, const float *bias_up, const float *bias_down,
                      float *y, float *tmp_gate, float *tmp_up,
                      int64_t batch, int64_t dim, int64_t hidden_dim) {
    /* gate = SiLU(x @ W_gate^T + bias_gate) */
    aria_linear_f32(x, W_gate, bias_gate, tmp_gate, batch, dim, hidden_dim);
    aria_silu_f32(tmp_gate, tmp_gate, batch * hidden_dim);
    /* up = x @ W_up^T + bias_up */
    aria_linear_f32(x, W_up, bias_up, tmp_up, batch, dim, hidden_dim);
    /* element-wise: gate * up */
    aria_mul_f32(tmp_gate, tmp_up, tmp_gate, batch * hidden_dim);
    /* down: y = hidden @ W_down^T + bias_down */
    aria_linear_f32(tmp_gate, W_down, bias_down, y, batch, hidden_dim, dim);
}

/* ══════════════════════════════════════════════════════════════════════
 * RWKV Channel Mixing
 * ══════════════════════════════════════════════════════════════════════ */

void aria_rwkv_channel_f32(const float *x,
                            const float *mix_k, const float *mix_r,
                            const float *W_k, const float *W_r, const float *W_v,
                            float *y, float *tmp_xk, float *tmp_xr, float *tmp_k,
                            int64_t batch, int64_t seq, int64_t dim, int64_t hidden_dim) {
    /* RWKV channel mixing:
     *   xk[t] = mix_k * x[t] + (1-mix_k) * x[t-1]
     *   xr[t] = mix_r * x[t] + (1-mix_r) * x[t-1]
     *   k = square(relu(W_k @ xk))
     *   r = sigmoid(W_r @ xr)
     *   y = r * (W_v @ k)
     */
    for (int64_t b = 0; b < batch; b++) {
        for (int64_t t = 0; t < seq; t++) {
            const float *xt = x + (b * seq + t) * dim;
            const float *xprev = (t > 0) ? x + (b * seq + t - 1) * dim : xt;
            /* Time-shift mix */
            for (int64_t d = 0; d < dim; d++) {
                float mk = mix_k ? mix_k[d] : 0.5f;
                float mr = mix_r ? mix_r[d] : 0.5f;
                tmp_xk[d] = mk * xt[d] + (1.0f - mk) * xprev[d];
                tmp_xr[d] = mr * xt[d] + (1.0f - mr) * xprev[d];
            }
            /* k = square(relu(W_k @ xk)) */
            aria_linear_f32(tmp_xk, W_k, NULL, tmp_k, 1, dim, hidden_dim);
            aria_relu_f32(tmp_k, tmp_k, hidden_dim);
            for (int64_t h = 0; h < hidden_dim; h++) tmp_k[h] *= tmp_k[h];
            /* r = sigmoid(W_r @ xr) — reuse tmp_xr as buffer */
            float *r_buf = tmp_xr; /* safe: dim-sized, we need dim output */
            aria_linear_f32(tmp_xr, W_r, NULL, r_buf, 1, dim, dim);
            aria_sigmoid_f32(r_buf, r_buf, dim);
            /* y = r * (W_v @ k) */
            float *yt = y + (b * seq + t) * dim;
            aria_linear_f32(tmp_k, W_v, NULL, yt, 1, hidden_dim, dim);
            aria_mul_f32(r_buf, yt, yt, dim);
        }
    }
}

/* ══════════════════════════════════════════════════════════════════════
 * Token Pool / Restore
 * ══════════════════════════════════════════════════════════════════════ */

void aria_token_pool_restore_f32(const float *x, float *y,
                                   int64_t batch, int64_t seq, int64_t dim) {
    /* Pool adjacent pairs via mean, then restore via repeat.
     * Since output must be same shape, pool pairs and duplicate. */
#ifdef ARIA_HAS_OPENMP
    #pragma omp parallel for if(batch > 4) schedule(static)
#endif
    for (int64_t b = 0; b < batch; b++) {
        for (int64_t s = 0; s < seq - 1; s += 2) {
            const float *x0 = x + (b * seq + s) * dim;
            const float *x1 = x + (b * seq + s + 1) * dim;
            float *y0 = y + (b * seq + s) * dim;
            float *y1 = y + (b * seq + s + 1) * dim;
            for (int64_t d = 0; d < dim; d++) {
                float avg = 0.5f * (x0[d] + x1[d]);
                y0[d] = avg;
                y1[d] = avg;
            }
        }
        /* Handle odd last token */
        if (seq % 2 == 1) {
            const float *xl = x + (b * seq + seq - 1) * dim;
            float *yl = y + (b * seq + seq - 1) * dim;
            if (xl != yl) memcpy(yl, xl, dim * sizeof(float));
        }
    }
}

/* ══════════════════════════════════════════════════════════════════════
 * Selective Scan (SSM / Mamba-style)
 * ══════════════════════════════════════════════════════════════════════ */

void aria_selective_scan_f32(const float *x, const float *A, const float *B,
                              const float *C, const float *D,
                              float *y, int64_t batch, int64_t seq, int64_t dim) {
    /* Linear recurrence: h[t] = A[d] * h[t-1] + B[d] * x[t,d]
     *                    y[t,d] = C[d] * h[t] + D[d] * x[t,d]
     * A, B, C, D are [dim]-shaped (per-channel parameters).
     * This is the "diagonal" SSM variant (S4D / Mamba simplified). */
    for (int64_t b = 0; b < batch; b++) {
        for (int64_t d = 0; d < dim; d++) {
            float h = 0.0f;
            float a = A ? A[d] : 0.9f;    /* decay */
            float bv = B ? B[d] : 1.0f;   /* input scale */
            float c = C ? C[d] : 1.0f;    /* output scale */
            float dv = D ? D[d] : 0.0f;   /* skip connection */
            for (int64_t t = 0; t < seq; t++) {
                float xt = x[(b * seq + t) * dim + d];
                h = a * h + bv * xt;
                y[(b * seq + t) * dim + d] = c * h + dv * xt;
            }
        }
    }
}

/* ══════════════════════════════════════════════════════════════════════
 * Top-K Gating
 * ══════════════════════════════════════════════════════════════════════ */

void aria_topk_gate_f32(const float *x, const float *W_gate, float *y,
                          int64_t batch, int64_t seq, int64_t dim, int64_t k) {
    /* Project x to scores via W_gate, keep top-k, zero rest.
     * W_gate: [dim, dim], scores per-dim. Output: sparse gated version of x. */
    int64_t total = batch * seq;
    float *scores = (float *)malloc(dim * sizeof(float));
    if (!scores) { if (x != y) memcpy(y, x, total * dim * sizeof(float)); return; }

    for (int64_t bs = 0; bs < total; bs++) {
        const float *xr = x + bs * dim;
        float *yr = y + bs * dim;
        /* Compute scores = x * W_gate (use as dot product per dim) */
        if (W_gate) {
            for (int64_t d = 0; d < dim; d++) {
                float s = 0.0f;
                for (int64_t i = 0; i < dim; i++) s += xr[i] * W_gate[d * dim + i];
                scores[d] = s;
            }
        } else {
            memcpy(scores, xr, dim * sizeof(float));
        }
        /* Find k-th largest score via partial sort */
        float *tmp = (float *)malloc(dim * sizeof(float));
        if (!tmp) { memcpy(yr, xr, dim * sizeof(float)); continue; }
        memcpy(tmp, scores, dim * sizeof(float));
        int64_t kk = k < dim ? k : dim;
        std::partial_sort(tmp, tmp + kk, tmp + dim, std::greater<float>());
        float threshold = tmp[kk - 1];
        free(tmp);
        /* Gate: keep values where score >= threshold */
        for (int64_t d = 0; d < dim; d++) {
            yr[d] = scores[d] >= threshold ? xr[d] : 0.0f;
        }
    }
    free(scores);
}

/* ══════════════════════════════════════════════════════════════════════
 * Basis Expansion (sinusoidal)
 * ══════════════════════════════════════════════════════════════════════ */

void aria_basis_expansion_f32(const float *x, const float *freqs, float *y,
                                int64_t batch, int64_t seq, int64_t dim,
                                int64_t n_bases) {
    /* Sinusoidal basis expansion: y = [sin(freq_0*x), cos(freq_0*x), sin(freq_1*x), ...].
     * Output dim = dim (we write sin/cos pairs for up to dim/2 bases, or wrap). */
    int64_t total = batch * seq;
#ifdef ARIA_HAS_OPENMP
    #pragma omp parallel for if(total > 64) schedule(static)
#endif
    for (int64_t bs = 0; bs < total; bs++) {
        const float *xr = x + bs * dim;
        float *yr = y + bs * dim;
        for (int64_t d = 0; d < dim; d++) {
            int64_t basis_idx = d / 2;
            float freq = (freqs && basis_idx < n_bases) ? freqs[basis_idx] : (float)(basis_idx + 1);
            float val = xr[d] * freq;
            yr[d] = (d % 2 == 0) ? sinf(val) : cosf(val);
        }
    }
}

/* ══════════════════════════════════════════════════════════════════════
 * Sparse Threshold
 * ══════════════════════════════════════════════════════════════════════ */

void aria_sparse_threshold_f32(const float *x, float *y,
                                 int64_t batch, int64_t seq, int64_t dim) {
    /* Zero values below adaptive per-token median of absolute values. */
    int64_t total = batch * seq;
    float *abs_vals = (float *)malloc(dim * sizeof(float));
    if (!abs_vals) { if (x != y) memcpy(y, x, total * dim * sizeof(float)); return; }

    for (int64_t bs = 0; bs < total; bs++) {
        const float *xr = x + bs * dim;
        float *yr = y + bs * dim;
        for (int64_t d = 0; d < dim; d++) abs_vals[d] = fabsf(xr[d]);
        std::nth_element(abs_vals, abs_vals + dim / 2, abs_vals + dim);
        float threshold = abs_vals[dim / 2];
        for (int64_t d = 0; d < dim; d++) {
            yr[d] = fabsf(xr[d]) >= threshold ? xr[d] : 0.0f;
        }
    }
    free(abs_vals);
}

/* ══════════════════════════════════════════════════════════════════════
 * Routing Kernels
 * ══════════════════════════════════════════════════════════════════════ */

void aria_route_topk_indices_f32(const float *scores, int64_t *indices, float *weights,
                                   int64_t batch, int64_t seq, int64_t k) {
    /* Top-k over scores[batch, seq] → indices[batch, k], weights[batch, k] */
    int64_t kk = k < seq ? k : seq;
    for (int64_t b = 0; b < batch; b++) {
        const float *sb = scores + b * seq;
        int64_t *ib = indices + b * k;
        float *wb = weights + b * k;
        /* Initialize with first k elements */
        for (int64_t i = 0; i < kk; i++) { ib[i] = i; wb[i] = sb[i]; }
        /* Insertion-sort style: find min in topk, replace if current is larger */
        for (int64_t s = kk; s < seq; s++) {
            /* Find min in current topk */
            int64_t min_idx = 0;
            for (int64_t i = 1; i < kk; i++) {
                if (wb[i] < wb[min_idx]) min_idx = i;
            }
            if (sb[s] > wb[min_idx]) {
                ib[min_idx] = s;
                wb[min_idx] = sb[s];
            }
        }
        /* Softmax normalize weights */
        float max_w = wb[0];
        for (int64_t i = 1; i < kk; i++) if (wb[i] > max_w) max_w = wb[i];
        float sum = 0.0f;
        for (int64_t i = 0; i < kk; i++) { wb[i] = expf(wb[i] - max_w); sum += wb[i]; }
        if (sum > 0.0f) for (int64_t i = 0; i < kk; i++) wb[i] /= sum;
        /* Zero-fill remaining if k > seq */
        for (int64_t i = kk; i < k; i++) { ib[i] = 0; wb[i] = 0.0f; }
    }
}

void aria_route_lane_argmax_f32(const float *scores, int64_t *lane_idx,
                                  int64_t batch, int64_t seq, int64_t lanes) {
    /* Argmax over lanes dimension: scores[batch, seq, lanes] → lane_idx[batch, seq] */
#ifdef ARIA_HAS_OPENMP
    #pragma omp parallel for if(batch * seq > 256) schedule(static)
#endif
    for (int64_t bs = 0; bs < batch * seq; bs++) {
        const float *sr = scores + bs * lanes;
        int64_t best = 0;
        for (int64_t l = 1; l < lanes; l++) {
            if (sr[l] > sr[best]) best = l;
        }
        lane_idx[bs] = best;
    }
}

void aria_route_recursion_depth_f32(const float *scores, int64_t *depth,
                                      int64_t batch, int64_t seq, int64_t max_depth) {
    /* Argmax over depth dimension + 1 (1-based): scores[B,S,D] → depth[B,S] */
#ifdef ARIA_HAS_OPENMP
    #pragma omp parallel for if(batch * seq > 256) schedule(static)
#endif
    for (int64_t bs = 0; bs < batch * seq; bs++) {
        const float *sr = scores + bs * max_depth;
        int64_t best = 0;
        for (int64_t d = 1; d < max_depth; d++) {
            if (sr[d] > sr[best]) best = d;
        }
        depth[bs] = best + 1; /* 1-based */
    }
}

/* ══════════════════════════════════════════════════════════════════════
 * Token Merge
 * ══════════════════════════════════════════════════════════════════════ */

void aria_token_merge_simple_f32(const float *x, float *y, int64_t *restore_map,
                                   int64_t batch, int64_t seq, int64_t dim, int64_t n_keep) {
    /* Keep first n_keep tokens. restore_map projects original positions to kept range. */
    if (!x || !y || batch <= 0 || seq <= 0 || dim <= 0) {
        return;
    }
    int64_t nk = n_keep < seq ? n_keep : seq;
    if (nk < 1) nk = 1;
    for (int64_t b = 0; b < batch; b++) {
        const float *xb = x + b * seq * dim;
        float *yb = y + b * nk * dim;
        /* Copy only kept tokens into compact output [B, nk, D]. */
        memcpy(yb, xb, nk * dim * sizeof(float));
        /* Build restore map [B, seq] with dropped tokens mapped to last kept. */
        if (restore_map) {
            int64_t *rm = restore_map + b * seq;
            for (int64_t s = 0; s < nk; s++) rm[s] = s;
            for (int64_t s = nk; s < seq; s++) rm[s] = nk - 1;
        }
    }
}

/* ══════════════════════════════════════════════════════════════════════
 * Cosine Similarity
 * ══════════════════════════════════════════════════════════════════════ */

void aria_cosine_similarity_f32(const float *a, const float *b, float *out,
                                  int64_t batch, int64_t seq, int64_t dim) {
#ifdef ARIA_HAS_OPENMP
    #pragma omp parallel for if(batch * seq > 64) schedule(static)
#endif
    for (int64_t bs = 0; bs < batch * seq; bs++) {
        const float *ar = a + bs * dim;
        const float *br = b + bs * dim;
        float dot = 0.0f, na = 0.0f, nb = 0.0f;
        for (int64_t d = 0; d < dim; d++) {
            dot += ar[d] * br[d];
            na += ar[d] * ar[d];
            nb += br[d] * br[d];
        }
        out[bs] = dot / (sqrtf(na * nb) + 1e-8f);
    }
}

/* ══════════════════════════════════════════════════════════════════════
 * Gather Top-K (real selection, not first-k)
 * ══════════════════════════════════════════════════════════════════════ */

void aria_gather_topk_f32(const float *scores, const float *values,
                            float *out, int32_t *out_indices,
                            int64_t batch, int64_t n_items, int64_t dim,
                            int64_t k) {
    int64_t kk = k < n_items ? k : n_items;
    int64_t *idx = (int64_t *)malloc(n_items * sizeof(int64_t));
    float *sc = (float *)malloc(n_items * sizeof(float));
    if (!idx || !sc) {
        /* Fallback: first k */
        for (int64_t b = 0; b < batch; b++) {
            for (int64_t i = 0; i < kk; i++) {
                out_indices[b * k + i] = (int32_t)i;
                memcpy(out + (b * k + i) * dim, values + (b * n_items + i) * dim, dim * sizeof(float));
            }
        }
        free(idx); free(sc);
        return;
    }
    for (int64_t b = 0; b < batch; b++) {
        const float *sb = scores + b * n_items;
        memcpy(sc, sb, n_items * sizeof(float));
        for (int64_t i = 0; i < n_items; i++) idx[i] = i;
        /* Partial sort to find top-k */
        for (int64_t i = 0; i < kk; i++) {
            int64_t best = i;
            for (int64_t j = i + 1; j < n_items; j++) {
                if (sc[j] > sc[best]) best = j;
            }
            if (best != i) {
                float tmp_s = sc[i]; sc[i] = sc[best]; sc[best] = tmp_s;
                int64_t tmp_i = idx[i]; idx[i] = idx[best]; idx[best] = tmp_i;
            }
            out_indices[b * k + i] = (int32_t)idx[i];
            memcpy(out + (b * k + i) * dim,
                   values + (b * n_items + idx[i]) * dim,
                   dim * sizeof(float));
        }
        for (int64_t i = kk; i < k; i++) {
            out_indices[b * k + i] = 0;
            memset(out + (b * k + i) * dim, 0, dim * sizeof(float));
        }
    }
    free(idx);
    free(sc);
}

/* ══════════════════════════════════════════════════════════════════════
 * Gated Linear
 * ══════════════════════════════════════════════════════════════════════ */

void aria_gated_linear_f32(const float *x,
                            const float *W, const float *b,
                            const float *W_gate, const float *b_gate,
                            float *y, float *tmp_gate,
                            int64_t batch, int64_t dim_in, int64_t dim_out) {
    aria_linear_f32(x, W, b, y, batch, dim_in, dim_out);
    aria_linear_f32(x, W_gate, b_gate, tmp_gate, batch, dim_in, dim_out);
    aria_sigmoid_f32(tmp_gate, tmp_gate, batch * dim_out);
    aria_mul_f32(y, tmp_gate, y, batch * dim_out);
}

/* ══════════════════════════════════════════════════════════════════════
 * RWKV Time Mixing (WKV kernel)
 * ══════════════════════════════════════════════════════════════════════ */

void aria_rwkv_time_mixing_f32(const float *x,
                                 const float *w_decay, const float *u_bonus,
                                 const float *W_k, const float *W_v, const float *W_r,
                                 float *y,
                                 int64_t batch, int64_t seq, int64_t dim) {
    /* RWKV WKV attention:
     *   k = W_k @ x,  v = W_v @ x,  r = sigmoid(W_r @ x)
     *   For each channel d and time t:
     *     wkv[t,d] = (a[t-1] + exp(u[d] + k[t,d]) * v[t,d]) /
     *                (b[t-1] + exp(u[d] + k[t,d]))
     *     a[t] = exp(-w[d]) * a[t-1] + exp(k[t,d]) * v[t,d]
     *     b[t] = exp(-w[d]) * b[t-1] + exp(k[t,d])
     *   y[t] = r[t] * wkv[t]
     */
    float *k_buf = (float *)malloc(seq * dim * sizeof(float));
    float *v_buf = (float *)malloc(seq * dim * sizeof(float));
    float *r_buf = (float *)malloc(seq * dim * sizeof(float));
    if (!k_buf || !v_buf || !r_buf) {
        if (x != y) memcpy(y, x, batch * seq * dim * sizeof(float));
        free(k_buf); free(v_buf); free(r_buf);
        return;
    }

    for (int64_t b = 0; b < batch; b++) {
        const float *xb = x + b * seq * dim;
        float *yb = y + b * seq * dim;
        /* Project: k, v, r for all time steps */
        aria_linear_f32(xb, W_k, NULL, k_buf, seq, dim, dim);
        aria_linear_f32(xb, W_v, NULL, v_buf, seq, dim, dim);
        aria_linear_f32(xb, W_r, NULL, r_buf, seq, dim, dim);
        aria_sigmoid_f32(r_buf, r_buf, seq * dim);

        /* WKV recurrence per channel */
        for (int64_t d = 0; d < dim; d++) {
            float w = w_decay ? w_decay[d] : 0.1f;
            float u = u_bonus ? u_bonus[d] : 0.0f;
            float a = 0.0f, bp = 0.0f; /* numerator, denominator accumulators */
            float ew = expf(-w); /* decay factor */
            for (int64_t t = 0; t < seq; t++) {
                float kt = k_buf[t * dim + d];
                float vt = v_buf[t * dim + d];
                float rt = r_buf[t * dim + d];
                /* Clamp to prevent overflow */
                float eku = expf(fminf(u + kt, 30.0f));
                float ek = expf(fminf(kt, 30.0f));
                float wkv = (a + eku * vt) / (bp + eku + 1e-8f);
                yb[t * dim + d] = rt * wkv;
                a = ew * a + ek * vt;
                bp = ew * bp + ek;
            }
        }
    }
    free(k_buf);
    free(v_buf);
    free(r_buf);
}

/* ══════════════════════════════════════════════════════════════════════
 * Compression Projections
 * ══════════════════════════════════════════════════════════════════════ */

void aria_grouped_linear_f32(const float *x, const float *W, float *y,
                               int64_t batch, int64_t seq, int64_t dim,
                               int64_t groups, int64_t group_dim) {
    /* Block-diagonal grouped linear: each group handles group_dim channels.
     * W: [groups, group_dim, group_dim] */
    int64_t total = batch * seq;
    int64_t gd = group_dim > 0 ? group_dim : (dim / (groups > 0 ? groups : 1));
    int64_t ng = groups > 0 ? groups : 1;
#ifdef ARIA_HAS_OPENMP
    #pragma omp parallel for if(total > 32) schedule(static)
#endif
    for (int64_t bs = 0; bs < total; bs++) {
        const float *xr = x + bs * dim;
        float *yr = y + bs * dim;
        for (int64_t g = 0; g < ng; g++) {
            int64_t offset = g * gd;
            const float *Wg = W + g * gd * gd;
            for (int64_t o = 0; o < gd && offset + o < dim; o++) {
                float acc = 0.0f;
                for (int64_t i = 0; i < gd && offset + i < dim; i++) {
                    acc += xr[offset + i] * Wg[o * gd + i];
                }
                yr[offset + o] = acc;
            }
        }
    }
}

void aria_bottleneck_proj_f32(const float *x, const float *down, const float *up, float *y,
                                int64_t batch, int64_t seq, int64_t dim, int64_t rank) {
    /* Bottleneck: y = GELU(x @ down^T) @ up^T
     * down: [rank, dim], up: [dim, rank] */
    int64_t total = batch * seq;
    float *tmp = (float *)malloc(total * rank * sizeof(float));
    if (!tmp) { if (x != y) memcpy(y, x, total * dim * sizeof(float)); return; }
    aria_linear_f32(x, down, NULL, tmp, total, dim, rank);
    aria_gelu_f32(tmp, tmp, total * rank);
    aria_linear_f32(tmp, up, NULL, y, total, rank, dim);
    free(tmp);
}

void aria_shared_basis_proj_f32(const float *x, const float *mixing, const float *basis, float *y,
                                  int64_t batch, int64_t seq, int64_t dim, int64_t k) {
    /* Shared basis: y = (x @ mixing^T) @ basis^T
     * mixing: [k, dim], basis: [dim, k] */
    int64_t total = batch * seq;
    float *tmp = (float *)malloc(total * k * sizeof(float));
    if (!tmp) { if (x != y) memcpy(y, x, total * dim * sizeof(float)); return; }
    aria_linear_f32(x, mixing, NULL, tmp, total, dim, k);
    aria_linear_f32(tmp, basis, NULL, y, total, k, dim);
    free(tmp);
}

void aria_tied_proj_f32(const float *x, const float *W, float *y,
                          int64_t batch, int64_t seq, int64_t dim, int64_t rank) {
    /* Tied: y = GELU(x @ W^T) @ W
     * W: [rank, dim]. Down = W^T: [dim, rank], Up = W: [rank, dim] → need W transposed for up. */
    int64_t total = batch * seq;
    float *tmp = (float *)malloc(total * rank * sizeof(float));
    float *WT = (float *)malloc(dim * rank * sizeof(float));
    if (!tmp || !WT) {
        if (x != y) memcpy(y, x, total * dim * sizeof(float));
        free(tmp); free(WT);
        return;
    }
    /* W: [rank, dim] → down projection: x[total, dim] @ W^T[dim, rank] = tmp[total, rank] */
    aria_linear_f32(x, W, NULL, tmp, total, dim, rank);
    aria_gelu_f32(tmp, tmp, total * rank);
    /* Up projection: tmp[total, rank] @ W[rank, dim] — need W as [dim, rank] transposed for aria_linear_f32
     * aria_linear_f32(input, weight, bias, out, batch, dim_in, dim_out) does input @ weight^T
     * We want tmp @ W where W is [rank, dim]. So we need a transposed version. */
    aria_transpose2d_f32(W, WT, rank, dim); /* WT: [dim, rank] */
    aria_linear_f32(tmp, WT, NULL, y, total, rank, dim);
    free(tmp);
    free(WT);
}

/* ══════════════════════════════════════════════════════════════════════
 * Compression Kernels (Tier 2)
 * ══════════════════════════════════════════════════════════════════════ */

void aria_linear_low_rank_f32(const float *x, const float *U, const float *V, const float *bias,
                                float *y, int64_t batch, int64_t dim_in, int64_t dim_out, int64_t rank) {
    /* Low-rank: y = x @ U^T @ V^T + bias
     * U: [rank, dim_in], V: [dim_out, rank]
     * Step 1: tmp = x @ U^T (down-project to rank)
     * Step 2: y = tmp @ V^T + bias (up-project to dim_out) */
    float *tmp = (float *)malloc(batch * rank * sizeof(float));
    if (!tmp) { memset(y, 0, batch * dim_out * sizeof(float)); return; }
    aria_linear_f32(x, U, NULL, tmp, batch, dim_in, rank);
    aria_linear_f32(tmp, V, bias, y, batch, rank, dim_out);
    free(tmp);
}

void aria_linear_block_sparse_f32(const float *x, const float *W, const float *bias,
                                   const uint8_t *block_mask,
                                   float *y, int64_t batch, int64_t dim_in, int64_t dim_out,
                                   int64_t block_size) {
    /* Block-sparse linear: only compute blocks where mask[bi][bo] == 1.
     * block_mask: [dim_out/block_size, dim_in/block_size] */
    int64_t n_block_out = (dim_out + block_size - 1) / block_size;
    int64_t n_block_in = (dim_in + block_size - 1) / block_size;
    memset(y, 0, batch * dim_out * sizeof(float));
#ifdef ARIA_HAS_OPENMP
    #pragma omp parallel for if(batch > 4) schedule(static)
#endif
    for (int64_t b = 0; b < batch; b++) {
        const float *xb = x + b * dim_in;
        float *yb = y + b * dim_out;
        for (int64_t bo = 0; bo < n_block_out; bo++) {
            for (int64_t bi = 0; bi < n_block_in; bi++) {
                if (block_mask && !block_mask[bo * n_block_in + bi]) continue;
                int64_t o_start = bo * block_size;
                int64_t o_end = o_start + block_size < dim_out ? o_start + block_size : dim_out;
                int64_t i_start = bi * block_size;
                int64_t i_end = i_start + block_size < dim_in ? i_start + block_size : dim_in;
                for (int64_t o = o_start; o < o_end; o++) {
                    float acc = 0.0f;
                    for (int64_t i = i_start; i < i_end; i++) {
                        acc += xb[i] * W[o * dim_in + i];
                    }
                    yb[o] += acc;
                }
            }
        }
        if (bias) for (int64_t o = 0; o < dim_out; o++) yb[o] += bias[o];
    }
}

void aria_nm_sparse_mask_f32(const float *W, uint8_t *mask,
                               int64_t rows, int64_t cols, int32_t n, int32_t m) {
    /* N:M structured sparsity: in every group of M consecutive weights,
     * keep only the top-N by magnitude, set rest to 0. */
    if (n <= 0 || m <= 0) { memset(mask, 1, rows * cols); return; }
    for (int64_t r = 0; r < rows; r++) {
        for (int64_t c = 0; c < cols; c += m) {
            int64_t group_end = c + m < cols ? c + m : cols;
            int64_t group_size = group_end - c;
            /* Find top-N by magnitude */
            float mag[32]; /* m is typically 4 or 8 */
            int64_t idx[32];
            int64_t gs = group_size < 32 ? group_size : 32;
            for (int64_t i = 0; i < gs; i++) {
                mag[i] = fabsf(W[r * cols + c + i]);
                idx[i] = i;
            }
            /* Simple selection sort for top-N */
            int32_t nn = n < (int32_t)gs ? n : (int32_t)gs;
            for (int32_t i = 0; i < nn; i++) {
                int64_t best = i;
                for (int64_t j = i + 1; j < gs; j++) {
                    if (mag[j] > mag[best]) best = j;
                }
                if (best != i) {
                    float tmp_m = mag[i]; mag[i] = mag[best]; mag[best] = tmp_m;
                    int64_t tmp_i = idx[i]; idx[i] = idx[best]; idx[best] = tmp_i;
                }
            }
            /* Set mask */
            for (int64_t i = 0; i < gs; i++) mask[r * cols + c + i] = 0;
            for (int32_t i = 0; i < nn; i++) mask[r * cols + c + idx[i]] = 1;
        }
    }
}

void aria_linear_grouped_f32(const float *x, const float *W, const float *bias,
                               float *y, int64_t batch, int64_t dim, int64_t groups) {
    /* Block-diagonal grouped linear: W stored as [groups, gd, gd] (compact).
     * Each group transforms gd=dim/groups channels independently.
     * Computes: y[off..off+gd] = x[off..off+gd] @ W_g^T for each group g. */
    int64_t gd = dim / (groups > 0 ? groups : 1);
#ifdef ARIA_HAS_OPENMP
    #pragma omp parallel for if(batch > 4) schedule(static)
#endif
    for (int64_t b = 0; b < batch; b++) {
        const float *xb = x + b * dim;
        float *yb = y + b * dim;
        for (int64_t g = 0; g < groups; g++) {
            int64_t off = g * gd;
            const float *Wg = W + g * gd * gd; /* W_g: [gd, gd] */
            for (int64_t o = 0; o < gd; o++) {
                float acc = 0.0f;
                for (int64_t i = 0; i < gd; i++) {
                    acc += xb[off + i] * Wg[o * gd + i];
                }
                yb[off + o] = acc + (bias ? bias[off + o] : 0.0f);
            }
        }
    }
}

void aria_linear_bottleneck_f32(const float *x, const float *W_down, const float *W_up,
                                  const float *b_down, const float *b_up,
                                  float *y, int64_t batch, int64_t dim_in, int64_t dim_out, int64_t rank) {
    /* Bottleneck: y = GELU(x @ W_down^T + b_down) @ W_up^T + b_up
     * W_down: [rank, dim_in], W_up: [dim_out, rank] */
    float *tmp = (float *)malloc(batch * rank * sizeof(float));
    if (!tmp) { memset(y, 0, batch * dim_out * sizeof(float)); return; }
    aria_linear_f32(x, W_down, b_down, tmp, batch, dim_in, rank);
    aria_gelu_f32(tmp, tmp, batch * rank);
    aria_linear_f32(tmp, W_up, b_up, y, batch, rank, dim_out);
    free(tmp);
}

void aria_linear_shared_basis_f32(const float *x, const float *Mixing, const float *Basis,
                                    float *y, int64_t batch, int64_t dim, int64_t k_basis) {
    /* Shared basis: y = (x @ Mixing^T) @ Basis^T
     * Mixing: [k_basis, dim], Basis: [dim, k_basis] */
    float *tmp = (float *)malloc(batch * k_basis * sizeof(float));
    if (!tmp) { if (x != y) memcpy(y, x, batch * dim * sizeof(float)); return; }
    aria_linear_f32(x, Mixing, NULL, tmp, batch, dim, k_basis);
    aria_linear_f32(tmp, Basis, NULL, y, batch, k_basis, dim);
    free(tmp);
}

void aria_linear_tied_f32(const float *x, const float *W, const float *b_down, const float *b_up,
                            float *y, int64_t batch, int64_t dim_in, int64_t rank) {
    /* Tied: y = GELU(x @ W^T + b_down) @ W + b_up
     * W: [rank, dim_in] */
    float *tmp = (float *)malloc(batch * rank * sizeof(float));
    float *WT = (float *)malloc(dim_in * rank * sizeof(float));
    if (!tmp || !WT) {
        memset(y, 0, batch * dim_in * sizeof(float));
        free(tmp); free(WT);
        return;
    }
    aria_linear_f32(x, W, b_down, tmp, batch, dim_in, rank);
    aria_gelu_f32(tmp, tmp, batch * rank);
    /* tmp[batch, rank] @ W[rank, dim_in] — we need WT[dim_in, rank] */
    aria_transpose2d_f32(W, WT, rank, dim_in);
    aria_linear_f32(tmp, WT, b_up, y, batch, rank, dim_in);
    free(tmp);
    free(WT);
}
