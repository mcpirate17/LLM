#ifndef ARIA_KERNELS_COMMON_H
#include "kernels_common.h"
#endif
#include <cmath>
#include <cstring>
#include <algorithm>

#ifdef __cplusplus
extern "C" {
#endif

/* ── Softmax Attention (QKV self-attention) ────────────────────────── */

void aria_softmax_attention_f32(const float *x, const float *Wq, const float *Wk,
                                 const float *Wv, const float *Wo,
                                 float *y, int64_t batch, int64_t seq,
                                 int64_t dim, int64_t n_heads) {
    int64_t head_dim = dim / n_heads;
    if (head_dim <= 0 || n_heads <= 0) return;

    float scale = 1.0f / sqrtf((float)head_dim);

    // Per batch
    for (int64_t b = 0; b < batch; b++) {
        const float *xb = x + b * seq * dim;
        float *yb = y + b * seq * dim;

        // Allocate temp buffers
        std::vector<float> Q(seq * dim), K(seq * dim), V(seq * dim);
        std::vector<float> attn_out(seq * dim);

        // Q = x @ Wq^T, K = x @ Wk^T, V = x @ Wv^T
        for (int64_t s = 0; s < seq; s++) {
            for (int64_t d = 0; d < dim; d++) {
                float sq = 0, sk = 0, sv = 0;
                for (int64_t k = 0; k < dim; k++) {
                    float xval = xb[s * dim + k];
                    sq += xval * Wq[d * dim + k];
                    sk += xval * Wk[d * dim + k];
                    sv += xval * Wv[d * dim + k];
                }
                Q[s * dim + d] = sq;
                K[s * dim + d] = sk;
                V[s * dim + d] = sv;
            }
        }

        // Per-head attention
        for (int64_t h = 0; h < n_heads; h++) {
            int64_t off = h * head_dim;

            // Compute QK^T and apply softmax
            std::vector<float> scores(seq * seq);
            for (int64_t i = 0; i < seq; i++) {
                float max_val = -1e30f;
                for (int64_t j = 0; j < seq; j++) {
                    float dot = 0;
                    for (int64_t d = 0; d < head_dim; d++) {
                        dot += Q[i * dim + off + d] * K[j * dim + off + d];
                    }
                    scores[i * seq + j] = dot * scale;
                    if (scores[i * seq + j] > max_val) max_val = scores[i * seq + j];
                }
                // Softmax
                float sum_exp = 0;
                for (int64_t j = 0; j < seq; j++) {
                    scores[i * seq + j] = expf(scores[i * seq + j] - max_val);
                    sum_exp += scores[i * seq + j];
                }
                for (int64_t j = 0; j < seq; j++) {
                    scores[i * seq + j] /= (sum_exp + 1e-8f);
                }
            }

            // Attn output = scores @ V
            for (int64_t i = 0; i < seq; i++) {
                for (int64_t d = 0; d < head_dim; d++) {
                    float val = 0;
                    for (int64_t j = 0; j < seq; j++) {
                        val += scores[i * seq + j] * V[j * dim + off + d];
                    }
                    attn_out[i * dim + off + d] = val;
                }
            }
        }

        // Output projection: y = attn_out @ Wo^T
        for (int64_t s = 0; s < seq; s++) {
            for (int64_t d = 0; d < dim; d++) {
                float val = 0;
                for (int64_t k = 0; k < dim; k++) {
                    val += attn_out[s * dim + k] * Wo[d * dim + k];
                }
                yb[s * dim + d] = val;
            }
        }
    }
}

/* ── Linear Attention (ELU feature map, O(SD)) ────────────────────── */

void aria_linear_attention_f32(const float *x, const float *Wq, const float *Wk,
                                const float *Wv, float *y,
                                int64_t batch, int64_t seq, int64_t dim) {
    for (int64_t b = 0; b < batch; b++) {
        const float *xb = x + b * seq * dim;
        float *yb = y + b * seq * dim;

        std::vector<float> Q(seq * dim), K(seq * dim), V(seq * dim);

        // Project Q, K, V and apply ELU+1 to Q, K
        for (int64_t s = 0; s < seq; s++) {
            for (int64_t d = 0; d < dim; d++) {
                float sq = 0, sk = 0, sv = 0;
                for (int64_t k = 0; k < dim; k++) {
                    float xval = xb[s * dim + k];
                    sq += xval * Wq[d * dim + k];
                    sk += xval * Wk[d * dim + k];
                    sv += xval * Wv[d * dim + k];
                }
                // ELU + 1 feature map
                Q[s * dim + d] = sq > 0 ? sq + 1.0f : expf(sq);
                K[s * dim + d] = sk > 0 ? sk + 1.0f : expf(sk);
                V[s * dim + d] = sv;
            }
        }

        // KV = K^T @ V (dim x dim)
        std::vector<float> KV(dim * dim, 0.0f);
        for (int64_t s = 0; s < seq; s++) {
            for (int64_t i = 0; i < dim; i++) {
                for (int64_t j = 0; j < dim; j++) {
                    KV[i * dim + j] += K[s * dim + i] * V[s * dim + j];
                }
            }
        }

        // K_sum = sum(K, dim=seq)
        std::vector<float> K_sum(dim, 0.0f);
        for (int64_t s = 0; s < seq; s++) {
            for (int64_t d = 0; d < dim; d++) {
                K_sum[d] += K[s * dim + d];
            }
        }

        // y = Q @ KV / (Q @ K_sum)
        for (int64_t s = 0; s < seq; s++) {
            float z = 0;
            for (int64_t d = 0; d < dim; d++) {
                z += Q[s * dim + d] * K_sum[d];
            }
            z = std::max(z, 1e-6f);
            for (int64_t d = 0; d < dim; d++) {
                float val = 0;
                for (int64_t k = 0; k < dim; k++) {
                    val += Q[s * dim + k] * KV[k * dim + d];
                }
                yb[s * dim + d] = val / z;
            }
        }
    }
}

/* ── Fourier Mixing ────────────────────────────────────────────────── */
/* Simplified: uses real FFT-like transform via DCT approximation      */

void aria_fourier_mixing_f32(const float *x, const float *weight, float *y,
                              int64_t batch, int64_t seq, int64_t dim) {
    // Simple spectral mixing: per-channel DCT-II → elementwise scale → DCT-III
    float scale = sqrtf(2.0f / (float)seq);

    for (int64_t b = 0; b < batch; b++) {
        for (int64_t d = 0; d < dim; d++) {
            // Forward DCT-II (per channel along seq)
            std::vector<float> freq(seq);
            for (int64_t k = 0; k < seq; k++) {
                float sum = 0;
                for (int64_t n = 0; n < seq; n++) {
                    sum += x[b * seq * dim + n * dim + d] *
                           cosf(M_PI * (float)k * (2.0f * (float)n + 1.0f) / (2.0f * (float)seq));
                }
                freq[k] = sum * scale;
            }

            // Elementwise multiply with learnable weight
            for (int64_t k = 0; k < seq; k++) {
                freq[k] *= weight[k % dim];
            }

            // Inverse DCT-III
            for (int64_t n = 0; n < seq; n++) {
                float sum = freq[0] * 0.5f;
                for (int64_t k = 1; k < seq; k++) {
                    sum += freq[k] * cosf(M_PI * (float)k * (2.0f * (float)n + 1.0f) / (2.0f * (float)seq));
                }
                y[b * seq * dim + n * dim + d] = sum * scale;
            }
        }
    }
}

/* ── MoE Top-K Gating ─────────────────────────────────────────────── */

void aria_moe_topk_f32(const float *x, const float *gate_weight, float *y,
                        const float **expert_weights, int64_t batch, int64_t seq,
                        int64_t dim, int64_t n_experts, int64_t k) {
    for (int64_t b = 0; b < batch; b++) {
        for (int64_t s = 0; s < seq; s++) {
            const float *xs = x + (b * seq + s) * dim;
            float *ys = y + (b * seq + s) * dim;

            // Compute gate logits
            std::vector<float> logits(n_experts);
            for (int64_t e = 0; e < n_experts; e++) {
                float val = 0;
                for (int64_t d = 0; d < dim; d++) {
                    val += xs[d] * gate_weight[e * dim + d];
                }
                logits[e] = val;
            }

            // Top-k selection
            std::vector<int64_t> topk_idx(k);
            std::vector<float> topk_val(k, -1e30f);
            for (int64_t e = 0; e < n_experts; e++) {
                for (int64_t i = 0; i < k; i++) {
                    if (logits[e] > topk_val[i]) {
                        for (int64_t j = k - 1; j > i; j--) {
                            topk_val[j] = topk_val[j - 1];
                            topk_idx[j] = topk_idx[j - 1];
                        }
                        topk_val[i] = logits[e];
                        topk_idx[i] = e;
                        break;
                    }
                }
            }

            // Softmax over top-k values
            float max_v = topk_val[0];
            for (int64_t i = 1; i < k; i++) max_v = std::max(max_v, topk_val[i]);
            float sum_exp = 0;
            for (int64_t i = 0; i < k; i++) {
                topk_val[i] = expf(topk_val[i] - max_v);
                sum_exp += topk_val[i];
            }
            for (int64_t i = 0; i < k; i++) topk_val[i] /= (sum_exp + 1e-8f);

            // Weighted sum of expert outputs
            memset(ys, 0, dim * sizeof(float));
            for (int64_t i = 0; i < k; i++) {
                float w = topk_val[i];
                int64_t eidx = topk_idx[i];
                if (expert_weights && expert_weights[eidx]) {
                    for (int64_t d = 0; d < dim; d++) {
                        float expert_val = 0;
                        for (int64_t dd = 0; dd < dim; dd++) {
                            expert_val += xs[dd] * expert_weights[eidx][d * dim + dd];
                        }
                        ys[d] += w * expert_val;
                    }
                } else {
                    for (int64_t d = 0; d < dim; d++) {
                        ys[d] += w * xs[d];
                    }
                }
            }
        }
    }
}

/* ── ALiBi (Attention with Linear Biases) ─────────────────────────── */

void aria_alibi_f32(float *bias, int64_t n_heads, int64_t seq) {
    for (int64_t h = 0; h < n_heads; h++) {
        float slope = powf(2.0f, -(float)(h + 1) * 8.0f / (float)n_heads);
        for (int64_t i = 0; i < seq; i++) {
            for (int64_t j = 0; j < seq; j++) {
                bias[h * seq * seq + i * seq + j] = -slope * fabsf((float)(i - j));
            }
        }
    }
}

/* ── Group Norm ────────────────────────────────────────────────────── */

void aria_group_norm_f32(const float *x, const float *gamma, const float *beta,
                          float *y, int64_t batch, int64_t channels, int64_t spatial,
                          int64_t groups, float eps) {
    int64_t channels_per_group = channels / groups;
    for (int64_t b = 0; b < batch; b++) {
        for (int64_t g = 0; g < groups; g++) {
            int64_t c_start = g * channels_per_group;
            int64_t count = channels_per_group * spatial;

            // Compute mean
            double sum = 0.0;
            for (int64_t c = c_start; c < c_start + channels_per_group; c++) {
                for (int64_t s = 0; s < spatial; s++) {
                    sum += (double)x[b * channels * spatial + c * spatial + s];
                }
            }
            double mean = sum / (double)count;

            // Compute variance
            double var = 0.0;
            for (int64_t c = c_start; c < c_start + channels_per_group; c++) {
                for (int64_t s = 0; s < spatial; s++) {
                    double d = (double)x[b * channels * spatial + c * spatial + s] - mean;
                    var += d * d;
                }
            }
            var = var / (double)count;
            float inv_std = 1.0f / sqrtf((float)var + eps);

            // Normalize
            for (int64_t c = c_start; c < c_start + channels_per_group; c++) {
                for (int64_t s = 0; s < spatial; s++) {
                    int64_t idx = b * channels * spatial + c * spatial + s;
                    float normed = ((float)x[idx] - (float)mean) * inv_std;
                    y[idx] = normed * gamma[c] + beta[c];
                }
            }
        }
    }
}

/* ── Dynamic Norm (interpolation between RMS + Layer) ─────────────── */

void aria_dynamic_norm_f32(const float *x, const float *gamma, float *y,
                            int64_t batch, int64_t dim, float alpha, float eps) {
    for (int64_t b = 0; b < batch; b++) {
        const float *xb = x + b * dim;
        float *yb = y + b * dim;

        // RMSNorm
        double ss = 0.0;
        for (int64_t i = 0; i < dim; i++) ss += (double)xb[i] * (double)xb[i];
        float rms_inv = 1.0f / sqrtf((float)(ss / (double)dim) + eps);

        // LayerNorm mean + var
        double sum = 0.0;
        for (int64_t i = 0; i < dim; i++) sum += (double)xb[i];
        float mean = (float)(sum / (double)dim);
        double var = 0.0;
        for (int64_t i = 0; i < dim; i++) {
            double d = (double)xb[i] - (double)mean;
            var += d * d;
        }
        float ln_inv = 1.0f / sqrtf((float)(var / (double)dim) + eps);

        for (int64_t i = 0; i < dim; i++) {
            float rms_out = xb[i] * rms_inv * gamma[i];
            float ln_out = (xb[i] - mean) * ln_inv * gamma[i];
            yb[i] = alpha * rms_out + (1.0f - alpha) * ln_out;
        }
    }
}

/* ── Sigmoid Norm ─────────────────────────────────────────────────── */

void aria_sigmoid_norm_f32(const float *x, const float *gamma, float *y,
                            int64_t batch, int64_t dim, float eps) {
    for (int64_t b = 0; b < batch; b++) {
        const float *xb = x + b * dim;
        float *yb = y + b * dim;

        double ss = 0.0;
        for (int64_t i = 0; i < dim; i++) ss += (double)xb[i] * (double)xb[i];
        float rms_inv = 1.0f / sqrtf((float)(ss / (double)dim) + eps);

        for (int64_t i = 0; i < dim; i++) {
            float normed = xb[i] * rms_inv * gamma[i];
            float gate = 1.0f / (1.0f + expf(-normed));
            yb[i] = normed * gate;
        }
    }
}

/* ── Cumulative Sum (prefix scan) ─────────────────────────────────── */

#if defined(__AVX2__)
/*
 * SIMD prefix sum for a single row of `dim` floats.
 *
 * Strategy: blocked scan.
 *   1. Process 8-element chunks with an in-register AVX2 prefix sum.
 *   2. Add the running total from the previous chunk to each chunk.
 *
 * The in-register prefix sum for 8 floats uses a classic shift-and-add
 * pattern (3 stages for 8 lanes).
 */
static void cumsum_row_avx2(const float *x, float *y, int64_t dim) {
    __m256 running = _mm256_setzero_ps();
    int64_t i = 0;

    for (; i + 7 < dim; i += 8) {
        __m256 v = _mm256_loadu_ps(x + i);

        /* In-register prefix sum (Hillis-Steele style, 3 stages) */
        /* Stage 1: shift by 1 lane */
        __m256 s1 = _mm256_castsi256_ps(
            _mm256_slli_si256(_mm256_castps_si256(v), 4));
        v = _mm256_add_ps(v, s1);

        /* Stage 2: shift by 2 lanes */
        __m256 s2 = _mm256_castsi256_ps(
            _mm256_slli_si256(_mm256_castps_si256(v), 8));
        v = _mm256_add_ps(v, s2);

        /* Stage 3: cross-lane — propagate lane 3 to lanes 4-7 */
        /* Extract element [3] and broadcast to upper 128 bits */
        __m256 lane3 = _mm256_permute_ps(v, 0xFF);          /* [3,3,3,3, 7,7,7,7] */
        lane3 = _mm256_permute2f128_ps(lane3, lane3, 0x00); /* [3,3,3,3, 3,3,3,3] */
        __m256 mask = _mm256_cmp_ps(
            _mm256_set_ps(1,1,1,1, 0,0,0,0),
            _mm256_setzero_ps(), _CMP_GT_OQ);               /* [0,0,0,0, 1,1,1,1] */
        lane3 = _mm256_and_ps(lane3, mask);
        v = _mm256_add_ps(v, lane3);

        /* Add running offset from previous chunks */
        v = _mm256_add_ps(v, running);
        _mm256_storeu_ps(y + i, v);

        /* Update running total = broadcast of last element (lane 7) */
        __m256 last = _mm256_permute_ps(v, 0xFF);           /* [3,3,3,3, 7,7,7,7] */
        running = _mm256_permute2f128_ps(last, last, 0x11); /* [7,7,7,7, 7,7,7,7] */
    }

    /* Scalar tail */
    float tail_sum = (i > 0) ? y[i - 1] : 0.0f;
    for (; i < dim; i++) {
        tail_sum += x[i];
        y[i] = tail_sum;
    }
}
#endif /* __AVX2__ */

void aria_cumsum_f32(const float *x, float *y, int64_t batch, int64_t dim) {
#if defined(ARIA_HAS_OPENMP) && defined(__AVX2__)
    if (batch >= ARIA_OMP_BATCH_THRESHOLD && dim >= 32) {
        #pragma omp parallel for schedule(static)
        for (int64_t b = 0; b < batch; b++) {
            cumsum_row_avx2(x + b * dim, y + b * dim, dim);
        }
        return;
    }
#endif

#if defined(__AVX2__)
    for (int64_t b = 0; b < batch; b++) {
        cumsum_row_avx2(x + b * dim, y + b * dim, dim);
    }
#else
    for (int64_t b = 0; b < batch; b++) {
        float sum = 0.0f;
        for (int64_t i = 0; i < dim; i++) {
            sum += x[b * dim + i];
            y[b * dim + i] = sum;
        }
    }
#endif
}

/* ── Cumulative Product (safe, with clamping) ─────────────────────── */

void aria_cumprod_safe_f32(const float *x, float *y, int64_t batch, int64_t dim,
                            float clamp_min, float clamp_max) {
    for (int64_t b = 0; b < batch; b++) {
        float prod = 1.0f;
        for (int64_t i = 0; i < dim; i++) {
            float val = x[b * dim + i];
            if (val < clamp_min) val = clamp_min;
            if (val > clamp_max) val = clamp_max;
            prod *= val;
            y[b * dim + i] = prod;
        }
    }
}

/* ── Roll (circular shift along seq dim) ──────────────────────────── */

void aria_roll_seq_f32(const float *x, float *y, int64_t batch, int64_t seq,
                        int64_t dim, int64_t shift) {
    int64_t s = ((shift % seq) + seq) % seq;
    for (int64_t b = 0; b < batch; b++) {
        for (int64_t i = 0; i < seq; i++) {
            int64_t src = ((i - s) % seq + seq) % seq;
            memcpy(y + (b * seq + i) * dim,
                   x + (b * seq + src) * dim,
                   dim * sizeof(float));
        }
    }
}

/* ── Early Exit (confidence threshold → token mask) ───────────────── */

void aria_early_exit_f32(const float *x, float *y, int64_t batch, int64_t seq,
                          int64_t dim, float threshold) {
    for (int64_t b = 0; b < batch; b++) {
        for (int64_t s = 0; s < seq; s++) {
            float sum = 0;
            for (int64_t d = 0; d < dim; d++) {
                sum += x[(b * seq + s) * dim + d];
            }
            float confidence = 1.0f / (1.0f + expf(-sum / (float)dim));
            float mask = confidence > threshold ? 1.0f : 0.0f;
            for (int64_t d = 0; d < dim; d++) {
                y[(b * seq + s) * dim + d] = x[(b * seq + s) * dim + d] * mask;
            }
        }
    }
}

/* ── Entropy Router ───────────────────────────────────────────────── */

void aria_entropy_router_f32(const float *logits, float *entropy,
                              int64_t batch, int64_t n_routes) {
    for (int64_t b = 0; b < batch; b++) {
        const float *lb = logits + b * n_routes;
        // Softmax
        float max_v = lb[0];
        for (int64_t i = 1; i < n_routes; i++) max_v = std::max(max_v, lb[i]);
        float sum_exp = 0;
        std::vector<float> probs(n_routes);
        for (int64_t i = 0; i < n_routes; i++) {
            probs[i] = expf(lb[i] - max_v);
            sum_exp += probs[i];
        }
        float ent = 0;
        for (int64_t i = 0; i < n_routes; i++) {
            float p = probs[i] / (sum_exp + 1e-8f);
            if (p > 1e-8f) ent -= p * logf(p);
        }
        entropy[b] = ent;
    }
}

/* ── RWKV WKV Parallel Scan ────────────────────────────────────────── */
/*
 * RWKV time-mixing WKV computation (sequential scan along S dimension).
 *
 * For each (b, d):
 *   wkv[0] = 0, wkv_denom[0] = 0
 *   for t in 0..S:
 *     kt = k[b,t,d], vt = v[b,t,d], rt = sigmoid(r[b,t,d])
 *     p = exp(u + kt)
 *     out[b,t,d] = rt * (wkv + p * vt) / max(wkv_denom + p, 1e-8)
 *     wkv = wkv * exp(w) + exp(kt) * vt
 *     wkv_denom = wkv_denom * exp(w) + exp(kt)
 *
 * Inputs:
 *   k, v, r:     (batch * seq * dim)  — row-major [B, S, D]
 *   w_decay:     (dim)                — per-channel decay (negative log)
 *   u_bonus:     (dim)                — per-channel bonus
 *   out:         (batch * seq * dim)  — output buffer
 */
void aria_rwkv_wkv_scan_f32(const float *k, const float *v, const float *r,
                              const float *w_decay, const float *u_bonus,
                              float *out,
                              int64_t batch, int64_t seq, int64_t dim) {
    /* exp_w[d] = exp(-exp(w_decay[d])) — precompute once */
    float *exp_w = (float *)malloc((size_t)dim * sizeof(float));
    if (!exp_w) return;
    for (int64_t d = 0; d < dim; d++) {
        exp_w[d] = expf(-expf(w_decay[d]));
    }

#ifdef ARIA_HAS_OPENMP
    #pragma omp parallel for collapse(2) schedule(static) if(batch * dim > 256)
#endif
    for (int64_t b = 0; b < batch; b++) {
        for (int64_t d = 0; d < dim; d++) {
            float wkv = 0.0f;
            float wkv_denom = 0.0f;
            float ew = exp_w[d];
            float u = u_bonus[d];

            for (int64_t t = 0; t < seq; t++) {
                int64_t idx = (b * seq + t) * dim + d;
                float kt = k[idx];
                float vt = v[idx];
                float rt_raw = r[idx];

                /* sigmoid(rt_raw) */
                float rt = 1.0f / (1.0f + expf(-rt_raw));

                /* p = exp(u + kt) */
                float ekt = expf(kt);
                float p = expf(u + kt);

                /* output */
                float num = wkv + p * vt;
                float den = wkv_denom + p;
                if (den < 1e-8f) den = 1e-8f;
                out[idx] = rt * num / den;

                /* state update */
                wkv = wkv * ew + ekt * vt;
                wkv_denom = wkv_denom * ew + ekt;
            }
        }
    }

    free(exp_w);
}

#ifdef __cplusplus
}
#endif
