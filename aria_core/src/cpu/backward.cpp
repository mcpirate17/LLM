#ifndef ARIA_KERNELS_COMMON_H
#include "kernels_common.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* ── Unary backward ops ───────────────────────────────────────────── */

void aria_relu_backward_f32(const float *grad_out, const float *input,
                             float *grad_in, int64_t n) {
#ifdef __AVX2__
    {
        const __m256 zero = _mm256_setzero_ps();
        int64_t vec_end = n - (n % 8);
#ifdef ARIA_HAS_OPENMP
        #pragma omp parallel for if(n > ARIA_OMP_THRESHOLD) schedule(static)
#endif
        for (int64_t i = 0; i < vec_end; i += 8) {
            __m256 vx = _mm256_loadu_ps(input + i);
            __m256 vg = _mm256_loadu_ps(grad_out + i);
            __m256 mask = _mm256_cmp_ps(vx, zero, _CMP_GT_OQ);
            __m256 result = _mm256_and_ps(vg, mask);
            _mm256_storeu_ps(grad_in + i, result);
        }
        for (int64_t i = vec_end; i < n; i++) grad_in[i] = input[i] > 0.0f ? grad_out[i] : 0.0f;
    }
#else
#ifdef ARIA_HAS_OPENMP
    #pragma omp parallel for if(n > ARIA_OMP_THRESHOLD) schedule(static)
#endif
    for (int64_t i = 0; i < n; i++) grad_in[i] = input[i] > 0.0f ? grad_out[i] : 0.0f;
#endif
}

void aria_sigmoid_backward_f32(const float *grad_out, const float *output,
                                float *grad_in, int64_t n) {
#ifdef __AVX2__
    {
        const __m256 one = _mm256_set1_ps(1.0f);
        int64_t vec_end = n - (n % 8);
#ifdef ARIA_HAS_OPENMP
        #pragma omp parallel for if(n > ARIA_OMP_THRESHOLD) schedule(static)
#endif
        for (int64_t i = 0; i < vec_end; i += 8) {
            __m256 vo = _mm256_loadu_ps(output + i);
            __m256 vg = _mm256_loadu_ps(grad_out + i);
            __m256 one_minus_o = _mm256_sub_ps(one, vo);
            __m256 result = _mm256_mul_ps(vg, _mm256_mul_ps(vo, one_minus_o));
            _mm256_storeu_ps(grad_in + i, result);
        }
        for (int64_t i = vec_end; i < n; i++) grad_in[i] = grad_out[i] * output[i] * (1.0f - output[i]);
    }
#else
#ifdef ARIA_HAS_OPENMP
    #pragma omp parallel for if(n > ARIA_OMP_THRESHOLD) schedule(static)
#endif
    for (int64_t i = 0; i < n; i++) grad_in[i] = grad_out[i] * output[i] * (1.0f - output[i]);
#endif
}

void aria_tanh_backward_f32(const float *grad_out, const float *output,
                             float *grad_in, int64_t n) {
#ifdef __AVX2__
    {
        const __m256 one = _mm256_set1_ps(1.0f);
        int64_t vec_end = n - (n % 8);
#ifdef ARIA_HAS_OPENMP
        #pragma omp parallel for if(n > ARIA_OMP_THRESHOLD) schedule(static)
#endif
        for (int64_t i = 0; i < vec_end; i += 8) {
            __m256 vo = _mm256_loadu_ps(output + i);
            __m256 vg = _mm256_loadu_ps(grad_out + i);
            __m256 o2 = _mm256_mul_ps(vo, vo);
            __m256 result = _mm256_mul_ps(vg, _mm256_sub_ps(one, o2));
            _mm256_storeu_ps(grad_in + i, result);
        }
        for (int64_t i = vec_end; i < n; i++) grad_in[i] = grad_out[i] * (1.0f - output[i] * output[i]);
    }
#else
#ifdef ARIA_HAS_OPENMP
    #pragma omp parallel for if(n > ARIA_OMP_THRESHOLD) schedule(static)
#endif
    for (int64_t i = 0; i < n; i++) grad_in[i] = grad_out[i] * (1.0f - output[i] * output[i]);
#endif
}

void aria_gelu_backward_f32(const float *grad_out, const float *input,
                             float *grad_in, int64_t n) {
#ifdef __AVX2__
    {
        const __m256 half   = _mm256_set1_ps(0.5f);
        const __m256 one    = _mm256_set1_ps(1.0f);
        const __m256 two    = _mm256_set1_ps(2.0f);
        const __m256 three  = _mm256_set1_ps(3.0f);
        const __m256 coeff  = _mm256_set1_ps(GELU_COEFF);
        const __m256 cubic  = _mm256_set1_ps(GELU_CUBIC);
        int64_t vec_end = n - (n % 8);
#ifdef ARIA_HAS_OPENMP
        #pragma omp parallel for if(n > ARIA_OMP_THRESHOLD) schedule(static)
#endif
        for (int64_t i = 0; i < vec_end; i += 8) {
            __m256 vx = _mm256_loadu_ps(input + i);
            __m256 vg = _mm256_loadu_ps(grad_out + i);
            __m256 x2 = _mm256_mul_ps(vx, vx);
            __m256 x3 = _mm256_mul_ps(x2, vx);
            __m256 inner = _mm256_fmadd_ps(cubic, x3, vx);
            inner = _mm256_mul_ps(coeff, inner);
            __m256 sig = _mm256_sigmoid_ps(_mm256_mul_ps(two, inner));
            __m256 t = _mm256_fmsub_ps(two, sig, one);
            __m256 d_inner = _mm256_fmadd_ps(_mm256_mul_ps(three, cubic), x2, one);
            d_inner = _mm256_mul_ps(coeff, d_inner);
            __m256 dgelu = _mm256_add_ps(_mm256_mul_ps(half, _mm256_add_ps(one, t)), 
                                         _mm256_mul_ps(half, _mm256_mul_ps(vx, _mm256_mul_ps(_mm256_sub_ps(one, _mm256_mul_ps(t, t)), d_inner))));
            _mm256_storeu_ps(grad_in + i, _mm256_mul_ps(vg, dgelu));
        }
        for (int64_t i = vec_end; i < n; i++) {
            float v = input[i]; float inner = GELU_COEFF * (v + GELU_CUBIC * v * v * v);
            float t = tanhf(inner); float d_inner = GELU_COEFF * (1.0f + 3.0f * GELU_CUBIC * v * v);
            float dgelu = 0.5f * (1.0f + t) + 0.5f * v * (1.0f - t * t) * d_inner;
            grad_in[i] = grad_out[i] * dgelu;
        }
    }
#else
#ifdef ARIA_HAS_OPENMP
    #pragma omp parallel for if(n > ARIA_OMP_THRESHOLD) schedule(static)
#endif
    for (int64_t i = 0; i < n; i++) {
        float v = input[i]; float inner = GELU_COEFF * (v + GELU_CUBIC * v * v * v);
        float t = tanhf(inner); float d_inner = GELU_COEFF * (1.0f + 3.0f * GELU_CUBIC * v * v);
        float dgelu = 0.5f * (1.0f + t) + 0.5f * v * (1.0f - t * t) * d_inner;
        grad_in[i] = grad_out[i] * dgelu;
    }
#endif
}

void aria_silu_backward_f32(const float *grad_out, const float *input,
                             float *grad_in, int64_t n) {
#ifdef __AVX2__
    {
        const __m256 one = _mm256_set1_ps(1.0f);
        int64_t vec_end = n - (n % 8);
#ifdef ARIA_HAS_OPENMP
        #pragma omp parallel for if(n > ARIA_OMP_THRESHOLD) schedule(static)
#endif
        for (int64_t i = 0; i < vec_end; i += 8) {
            __m256 vx = _mm256_loadu_ps(input + i);
            __m256 vg = _mm256_loadu_ps(grad_out + i);
            __m256 sig = _mm256_sigmoid_ps(vx);
            __m256 dsilu = _mm256_mul_ps(sig, _mm256_fmadd_ps(vx, _mm256_sub_ps(one, sig), one));
            _mm256_storeu_ps(grad_in + i, _mm256_mul_ps(vg, dsilu));
        }
        for (int64_t i = vec_end; i < n; i++) {
            float v = input[i]; float sig = 1.0f / (1.0f + expf(-v));
            grad_in[i] = grad_out[i] * sig * (1.0f + v * (1.0f - sig));
        }
    }
#else
#ifdef ARIA_HAS_OPENMP
    #pragma omp parallel for if(n > ARIA_OMP_THRESHOLD) schedule(static)
#endif
    for (int64_t i = 0; i < n; i++) {
        float v = input[i]; float sig = 1.0f / (1.0f + expf(-v));
        grad_in[i] = grad_out[i] * sig * (1.0f + v * (1.0f - sig));
    }
#endif
}

void aria_add_backward_f32(const float *grad_out,
                            float *grad_a, float *grad_b, int64_t n) {
#ifdef ARIA_HAS_OPENMP
    #pragma omp parallel for if(n > ARIA_OMP_THRESHOLD) schedule(static)
#endif
    for (int64_t i = 0; i < n; i++) { grad_a[i] = grad_out[i]; grad_b[i] = grad_out[i]; }
}

void aria_mul_backward_f32(const float *grad_out,
                            const float *a, const float *b,
                            float *grad_a, float *grad_b, int64_t n) {
#ifdef __AVX2__
    {
        int64_t vec_end = n - (n % 8);
#ifdef ARIA_HAS_OPENMP
        #pragma omp parallel for if(n > ARIA_OMP_THRESHOLD) schedule(static)
#endif
        for (int64_t i = 0; i < vec_end; i += 8) {
            __m256 vg = _mm256_loadu_ps(grad_out + i);
            __m256 va = _mm256_loadu_ps(a + i);
            __m256 vb = _mm256_loadu_ps(b + i);
            _mm256_storeu_ps(grad_a + i, _mm256_mul_ps(vg, vb));
            _mm256_storeu_ps(grad_b + i, _mm256_mul_ps(vg, va));
        }
        for (int64_t i = vec_end; i < n; i++) { grad_a[i] = grad_out[i] * b[i]; grad_b[i] = grad_out[i] * a[i]; }
    }
#else
#ifdef ARIA_HAS_OPENMP
    #pragma omp parallel for if(n > ARIA_OMP_THRESHOLD) schedule(static)
#endif
    for (int64_t i = 0; i < n; i++) { grad_a[i] = grad_out[i] * b[i]; grad_b[i] = grad_out[i] * a[i]; }
#endif
}

void aria_sub_backward_f32(const float *grad_out,
                            float *grad_a, float *grad_b, int64_t n) {
#ifdef __AVX2__
    {
        const __m256 neg_one = _mm256_set1_ps(-1.0f);
        int64_t vec_end = n - (n % 8);
#ifdef ARIA_HAS_OPENMP
        #pragma omp parallel for if(n > ARIA_OMP_THRESHOLD) schedule(static)
#endif
        for (int64_t i = 0; i < vec_end; i += 8) {
            __m256 vg = _mm256_loadu_ps(grad_out + i);
            _mm256_storeu_ps(grad_a + i, vg);
            _mm256_storeu_ps(grad_b + i, _mm256_mul_ps(vg, neg_one));
        }
        for (int64_t i = vec_end; i < n; i++) { grad_a[i] = grad_out[i]; grad_b[i] = -grad_out[i]; }
    }
#else
#ifdef ARIA_HAS_OPENMP
    #pragma omp parallel for if(n > ARIA_OMP_THRESHOLD) schedule(static)
#endif
    for (int64_t i = 0; i < n; i++) { grad_a[i] = grad_out[i]; grad_b[i] = -grad_out[i]; }
#endif
}

void aria_matmul_backward_f32(const float *grad_out,
                               const float *A, const float *B,
                               float *grad_A, float *grad_B,
                               int64_t M, int64_t K, int64_t N) {
#ifdef ARIA_HAS_BLAS
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, (int)M, (int)K, (int)N, 1.0f, grad_out, (int)N, B, (int)N, 0.0f, grad_A, (int)K);
    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, (int)K, (int)N, (int)M, 1.0f, A, (int)K, grad_out, (int)N, 0.0f, grad_B, (int)N);
#else
    memset(grad_A, 0, sizeof(float) * M * K);
    for (int64_t i = 0; i < M; i++) for (int64_t j = 0; j < N; j++) {
        float g = grad_out[i * N + j]; for (int64_t k = 0; k < K; k++) grad_A[i * K + k] += g * B[k * N + j];
    }
    memset(grad_B, 0, sizeof(float) * K * N);
    for (int64_t k = 0; k < K; k++) for (int64_t i = 0; i < M; i++) {
        float a_val = A[i * K + k]; for (int64_t j = 0; j < N; j++) grad_B[k * N + j] += a_val * grad_out[i * N + j];
    }
#endif
}

void aria_softmax_backward_f32(const float *grad_out, const float *output,
                                float *grad_in, int64_t batch, int64_t dim) {
#ifdef ARIA_HAS_OPENMP
    #pragma omp parallel for if(batch > ARIA_OMP_BATCH_THRESHOLD) schedule(static)
#endif
    for (int64_t b = 0; b < batch; b++) {
        const float *go = grad_out + b * dim; const float *y = output + b * dim; float *gi = grad_in + b * dim;
        float dot = 0.0f; for (int64_t i = 0; i < dim; i++) dot += go[i] * y[i];
        for (int64_t i = 0; i < dim; i++) gi[i] = y[i] * (go[i] - dot);
    }
}

void aria_layernorm_backward_f32(const float *grad_out, const float *input,
                                  const float *gamma,
                                  float *grad_in, float *grad_gamma,
                                  float *grad_beta,
                                  int64_t batch, int64_t dim, float eps) {
    memset(grad_gamma, 0, sizeof(float) * dim); memset(grad_beta, 0, sizeof(float) * dim);
    for (int64_t b = 0; b < batch; b++) {
        const float *go = grad_out + b * dim; const float *x = input + b * dim; float *gi = grad_in + b * dim;
        float mean = 0.0f; for (int64_t i = 0; i < dim; i++) mean += x[i];
        mean /= (float)dim; float var = 0.0f; for (int64_t i = 0; i < dim; i++) { float d = x[i] - mean; var += d * d; }
        var /= (float)dim; float inv_std = 1.0f / sqrtf(var + eps);
        for (int64_t i = 0; i < dim; i++) {
            float x_hat = (x[i] - mean) * inv_std; grad_gamma[i] += go[i] * x_hat; grad_beta[i] += go[i];
        }
        float mean_g = 0.0f, mean_gx = 0.0f;
        for (int64_t i = 0; i < dim; i++) {
            float g = go[i] * gamma[i]; float x_hat = (x[i] - mean) * inv_std; mean_g += g; mean_gx += g * x_hat;
        }
        mean_g /= (float)dim; mean_gx /= (float)dim;
        for (int64_t i = 0; i < dim; i++) {
            float g = go[i] * gamma[i]; float x_hat = (x[i] - mean) * inv_std;
            gi[i] = inv_std * (g - mean_g - x_hat * mean_gx);
        }
    }
}

void aria_rmsnorm_backward_f32(const float *grad_out, const float *input,
                                const float *gamma,
                                float *grad_in, float *grad_gamma,
                                int64_t batch, int64_t dim, float eps) {
    memset(grad_gamma, 0, sizeof(float) * dim);
    for (int64_t b = 0; b < batch; b++) {
        const float *go = grad_out + b * dim; const float *x = input + b * dim; float *gi = grad_in + b * dim;
        float ss = 0.0f; for (int64_t i = 0; i < dim; i++) ss += x[i] * x[i];
        float rms_sq = ss / (float)dim + eps; float inv_rms = 1.0f / sqrtf(rms_sq);
        for (int64_t i = 0; i < dim; i++) grad_gamma[i] += go[i] * x[i] * inv_rms;
        float sum_gx = 0.0f; for (int64_t i = 0; i < dim; i++) sum_gx += go[i] * gamma[i] * x[i];
        float coeff = (sum_gx / (float)dim) / rms_sq;
        for (int64_t i = 0; i < dim; i++) gi[i] = inv_rms * (go[i] * gamma[i] - x[i] * coeff);
    }
}

void aria_maximum_backward_f32(const float *grad_out,
                                const float *a, const float *b,
                                float *grad_a, float *grad_b, int64_t n) {
#ifdef ARIA_HAS_OPENMP
    #pragma omp parallel for if(n > ARIA_OMP_THRESHOLD) schedule(static)
#endif
    for (int64_t i = 0; i < n; i++) { if (a[i] >= b[i]) { grad_a[i] = grad_out[i]; grad_b[i] = 0.0f; } else { grad_a[i] = 0.0f; grad_b[i] = grad_out[i]; } }
}

void aria_minimum_backward_f32(const float *grad_out,
                                const float *a, const float *b,
                                float *grad_a, float *grad_b, int64_t n) {
#ifdef ARIA_HAS_OPENMP
    #pragma omp parallel for if(n > ARIA_OMP_THRESHOLD) schedule(static)
#endif
    for (int64_t i = 0; i < n; i++) { if (a[i] <= b[i]) { grad_a[i] = grad_out[i]; grad_b[i] = 0.0f; } else { grad_a[i] = 0.0f; grad_b[i] = grad_out[i]; } }
}

void aria_div_safe_backward_f32(const float *grad_out,
                                 const float *a, const float *b,
                                 float *grad_a, float *grad_b, int64_t n) {
    static const float EPS = 1e-7f;
#ifdef ARIA_HAS_OPENMP
    #pragma omp parallel for if(n > ARIA_OMP_THRESHOLD) schedule(static)
#endif
    for (int64_t i = 0; i < n; i++) {
        float denom = b[i]; if (denom >= 0.0f && denom < EPS) denom = EPS; else if (denom < 0.0f && denom > -EPS) denom = -EPS;
        grad_a[i] = grad_out[i] / denom; grad_b[i] = -grad_out[i] * a[i] / (denom * denom);
    }
}

void aria_embedding_lookup_backward_f32(const float *grad_out, const int32_t *indices,
                                          float *grad_table, float *grad_pos_embed,
                                          int64_t batch, int64_t dim,
                                          int64_t vocab_size) {
    (void)vocab_size;
    for (int64_t b = 0; b < batch; b++) {
        int32_t idx = indices[b]; const float *gb = grad_out + b * dim; float *gt = grad_table + (int64_t)idx * dim;
        for (int64_t d = 0; d < dim; d++) gt[d] += gb[d];
        if (grad_pos_embed) { float *gp = grad_pos_embed + b * dim; for (int64_t d = 0; d < dim; d++) gp[d] += gb[d]; }
    }
}

void aria_gated_linear_backward_f32(const float *grad_out,
                                      const float *x, const float *W, const float *W_gate,
                                      const float *gate_sigmoid,
                                      float *grad_x, float *grad_W, float *grad_W_gate,
                                      float *grad_b, float *grad_b_gate,
                                      int64_t batch, int64_t dim_in, int64_t dim_out) {
    float *linear_out = (float *)malloc(batch * dim_out * sizeof(float));
    float *grad_linear = (float *)malloc(batch * dim_out * sizeof(float));
    float *grad_gate_pre = (float *)malloc(batch * dim_out * sizeof(float));
    if (!linear_out || !grad_linear || !grad_gate_pre) { free(linear_out); free(grad_linear); free(grad_gate_pre); return; }
    aria_linear_f32(x, W, NULL, linear_out, batch, dim_in, dim_out);
    int64_t total = batch * dim_out;
    for (int64_t i = 0; i < total; i++) { grad_linear[i] = grad_out[i] * gate_sigmoid[i]; float g = gate_sigmoid[i]; grad_gate_pre[i] = grad_out[i] * linear_out[i] * g * (1.0f - g); }
    if (grad_x) {
        memset(grad_x, 0, batch * dim_in * sizeof(float));
        for (int64_t b = 0; b < batch; b++) {
            const float *gl = grad_linear + b * dim_out; const float *gg = grad_gate_pre + b * dim_out; float *gx = grad_x + b * dim_in;
            for (int64_t o = 0; o < dim_out; o++) {
                const float *Wo = W + o * dim_in; const float *Wgo = W_gate + o * dim_in;
                for (int64_t i = 0; i < dim_in; i++) gx[i] += gl[o] * Wo[i] + gg[o] * Wgo[i];
            }
        }
    }
    if (grad_W) for (int64_t b = 0; b < batch; b++) {
        const float *gl = grad_linear + b * dim_out; const float *xb = x + b * dim_in;
        for (int64_t o = 0; o < dim_out; o++) { float *gWo = grad_W + o * dim_in; for (int64_t i = 0; i < dim_in; i++) gWo[i] += gl[o] * xb[i]; }
    }
    if (grad_W_gate) for (int64_t b = 0; b < batch; b++) {
        const float *gg = grad_gate_pre + b * dim_out; const float *xb = x + b * dim_in;
        for (int64_t o = 0; o < dim_out; o++) { float *gWgo = grad_W_gate + o * dim_in; for (int64_t i = 0; i < dim_in; i++) gWgo[i] += gg[o] * xb[i]; }
    }
    if (grad_b) for (int64_t b = 0; b < batch; b++) {
        const float *gl = grad_linear + b * dim_out; for (int64_t o = 0; o < dim_out; o++) grad_b[o] += gl[o];
    }
    if (grad_b_gate) for (int64_t b = 0; b < batch; b++) {
        const float *gg = grad_gate_pre + b * dim_out; for (int64_t o = 0; o < dim_out; o++) grad_b_gate[o] += gg[o];
    }
    free(linear_out); free(grad_linear); free(grad_gate_pre);
}

#ifdef __cplusplus
}
#endif
