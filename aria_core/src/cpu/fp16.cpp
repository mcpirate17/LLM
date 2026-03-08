#include "kernels_common.h"

/* ══════════════════════════════════════════════════════════════════════
 * FP16 (HALF-PRECISION) KERNELS
 *
 * Strategy: F16C convert-at-boundaries.
 * - Load fp16 (uint16_t) → convert to f32 via _mm256_cvtph_ps
 * - Compute in f32 (reuse existing SIMD paths)
 * - Convert f32 → fp16 via _mm256_cvtps_ph → store
 * ══════════════════════════════════════════════════════════════════════ */

/* ── Scalar fp16 ↔ fp32 conversion (fallback) ────────────────────── */

static inline float aria_f16_to_f32(uint16_t h) {
    uint32_t sign = (uint32_t)(h >> 15) << 31;
    uint32_t exp  = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;
    uint32_t f;
    if (exp == 0) {
        if (mant == 0) {
            f = sign;
        } else {
            exp = 1;
            while (!(mant & 0x400)) { mant <<= 1; exp--; }
            mant &= 0x3FF;
            f = sign | (uint32_t)((127 - 15 + exp) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        f = sign | 0x7F800000u | (mant << 13); /* Inf/NaN */
    } else {
        f = sign | (uint32_t)((exp - 15 + 127) << 23) | (mant << 13);
    }
    union { uint32_t u; float f; } u;
    u.u = f;
    return u.f;
}

static inline uint16_t aria_f32_to_f16(float val) {
    union { float f; uint32_t u; } u;
    u.f = val;
    uint32_t f = u.u;
    uint32_t sign = (f >> 16) & 0x8000;
    int32_t  exp  = (int32_t)((f >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = f & 0x7FFFFF;
    if (exp <= 0) {
        if (exp < -10) return (uint16_t)sign;
        mant = (mant | 0x800000) >> (1 - exp);
        return (uint16_t)(sign | (mant >> 13));
    } else if (exp >= 31) {
        if (exp == 143 && mant) return (uint16_t)(sign | 0x7C00 | (mant >> 13));
        return (uint16_t)(sign | 0x7C00);
    }
    return (uint16_t)(sign | (uint32_t)(exp << 10) | (mant >> 13));
}

/* ── Unary fp16 kernels ──────────────────────────────────────────── */

void aria_relu_f16(const uint16_t *x, uint16_t *y, int64_t n) {
#ifdef __F16C__
    {
        int64_t vec_end = n - (n % 8);
        const __m256 zero = _mm256_setzero_ps();
        for (int64_t i = 0; i < vec_end; i += 8) {
            __m128i vh = _mm_loadu_si128((const __m128i *)(x + i));
            __m256 vf = _mm256_cvtph_ps(vh);
            vf = _mm256_max_ps(vf, zero);
            __m128i out = _mm256_cvtps_ph(vf, _MM_FROUND_TO_NEAREST_INT);
            _mm_storeu_si128((__m128i *)(y + i), out);
        }
        for (int64_t i = vec_end; i < n; i++) {
            float v = aria_f16_to_f32(x[i]);
            y[i] = aria_f32_to_f16(v > 0.0f ? v : 0.0f);
        }
    }
#else
    for (int64_t i = 0; i < n; i++) {
        float v = aria_f16_to_f32(x[i]);
        y[i] = aria_f32_to_f16(v > 0.0f ? v : 0.0f);
    }
#endif
}

void aria_gelu_f16(const uint16_t *x, uint16_t *y, int64_t n) {
#if defined(__F16C__) && defined(__AVX2__)
    {
        const __m256 half  = _mm256_set1_ps(0.5f);
        const __m256 one   = _mm256_set1_ps(1.0f);
        const __m256 two   = _mm256_set1_ps(2.0f);
        const __m256 coeff = _mm256_set1_ps(GELU_COEFF);
        const __m256 cubic = _mm256_set1_ps(GELU_CUBIC);
        int64_t vec_end = n - (n % 8);
        for (int64_t i = 0; i < vec_end; i += 8) {
            __m128i vh = _mm_loadu_si128((const __m128i *)(x + i));
            __m256 vx = _mm256_cvtph_ps(vh);
            __m256 x2 = _mm256_mul_ps(vx, vx);
            __m256 x3 = _mm256_mul_ps(x2, vx);
            __m256 inner = _mm256_fmadd_ps(cubic, x3, vx);
            inner = _mm256_mul_ps(coeff, inner);
            __m256 two_inner = _mm256_mul_ps(two, inner);
            __m256 sig = _mm256_sigmoid_ps(two_inner);
            __m256 tanh_val = _mm256_fmsub_ps(two, sig, one);
            __m256 vy = _mm256_mul_ps(half, _mm256_mul_ps(vx, _mm256_add_ps(one, tanh_val)));
            __m128i out = _mm256_cvtps_ph(vy, _MM_FROUND_TO_NEAREST_INT);
            _mm_storeu_si128((__m128i *)(y + i), out);
        }
        for (int64_t i = vec_end; i < n; i++) {
            float v = aria_f16_to_f32(x[i]);
            float inner = GELU_COEFF * (v + GELU_CUBIC * v * v * v);
            y[i] = aria_f32_to_f16(0.5f * v * (1.0f + tanhf(inner)));
        }
    }
#else
    for (int64_t i = 0; i < n; i++) {
        float v = aria_f16_to_f32(x[i]);
        float inner = GELU_COEFF * (v + GELU_CUBIC * v * v * v);
        y[i] = aria_f32_to_f16(0.5f * v * (1.0f + tanhf(inner)));
    }
#endif
}

void aria_silu_f16(const uint16_t *x, uint16_t *y, int64_t n) {
#if defined(__F16C__) && defined(__AVX2__)
    {
        int64_t vec_end = n - (n % 8);
        for (int64_t i = 0; i < vec_end; i += 8) {
            __m128i vh = _mm_loadu_si128((const __m128i *)(x + i));
            __m256 vx = _mm256_cvtph_ps(vh);
            __m256 sig = _mm256_sigmoid_ps(vx);
            __m256 vy = _mm256_mul_ps(vx, sig);
            __m128i out = _mm256_cvtps_ph(vy, _MM_FROUND_TO_NEAREST_INT);
            _mm_storeu_si128((__m128i *)(y + i), out);
        }
        for (int64_t i = vec_end; i < n; i++) {
            float v = aria_f16_to_f32(x[i]);
            y[i] = aria_f32_to_f16(v / (1.0f + expf(-v)));
        }
    }
#else
    for (int64_t i = 0; i < n; i++) {
        float v = aria_f16_to_f32(x[i]);
        y[i] = aria_f32_to_f16(v / (1.0f + expf(-v)));
    }
#endif
}

void aria_sigmoid_f16(const uint16_t *x, uint16_t *y, int64_t n) {
#if defined(__F16C__) && defined(__AVX2__)
    {
        int64_t vec_end = n - (n % 8);
        for (int64_t i = 0; i < vec_end; i += 8) {
            __m128i vh = _mm_loadu_si128((const __m128i *)(x + i));
            __m256 vx = _mm256_cvtph_ps(vh);
            __m256 vy = _mm256_sigmoid_ps(vx);
            __m128i out = _mm256_cvtps_ph(vy, _MM_FROUND_TO_NEAREST_INT);
            _mm_storeu_si128((__m128i *)(y + i), out);
        }
        for (int64_t i = vec_end; i < n; i++) {
            float v = aria_f16_to_f32(x[i]);
            y[i] = aria_f32_to_f16(1.0f / (1.0f + expf(-v)));
        }
    }
#else
    for (int64_t i = 0; i < n; i++) {
        float v = aria_f16_to_f32(x[i]);
        y[i] = aria_f32_to_f16(1.0f / (1.0f + expf(-v)));
    }
#endif
}

void aria_add_f16(const uint16_t *a, const uint16_t *b, uint16_t *y, int64_t n) {
#ifdef __F16C__
    {
        int64_t vec_end = n - (n % 8);
        for (int64_t i = 0; i < vec_end; i += 8) {
            __m128i va_h = _mm_loadu_si128((const __m128i *)(a + i));
            __m128i vb_h = _mm_loadu_si128((const __m128i *)(b + i));
            __m256 va_f = _mm256_cvtph_ps(va_h);
            __m256 vb_f = _mm256_cvtph_ps(vb_h);
            __m256 vy = _mm256_add_ps(va_f, vb_f);
            __m128i out = _mm256_cvtps_ph(vy, _MM_FROUND_TO_NEAREST_INT);
            _mm_storeu_si128((__m128i *)(y + i), out);
        }
        for (int64_t i = vec_end; i < n; i++) {
            float fa = aria_f16_to_f32(a[i]);
            float fb = aria_f16_to_f32(b[i]);
            y[i] = aria_f32_to_f16(fa + fb);
        }
    }
#else
    for (int64_t i = 0; i < n; i++) {
        float fa = aria_f16_to_f32(a[i]);
        float fb = aria_f16_to_f32(b[i]);
        y[i] = aria_f32_to_f16(fa + fb);
    }
#endif
}

void aria_mul_f16(const uint16_t *a, const uint16_t *b, uint16_t *y, int64_t n) {
#ifdef __F16C__
    {
        int64_t vec_end = n - (n % 8);
        for (int64_t i = 0; i < vec_end; i += 8) {
            __m128i va_h = _mm_loadu_si128((const __m128i *)(a + i));
            __m128i vb_h = _mm_loadu_si128((const __m128i *)(b + i));
            __m256 va_f = _mm256_cvtph_ps(va_h);
            __m256 vb_f = _mm256_cvtph_ps(vb_h);
            __m256 vy = _mm256_mul_ps(va_f, vb_f);
            __m128i out = _mm256_cvtps_ph(vy, _MM_FROUND_TO_NEAREST_INT);
            _mm_storeu_si128((__m128i *)(y + i), out);
        }
        for (int64_t i = vec_end; i < n; i++) {
            float fa = aria_f16_to_f32(a[i]);
            float fb = aria_f16_to_f32(b[i]);
            y[i] = aria_f32_to_f16(fa * fb);
        }
    }
#else
    for (int64_t i = 0; i < n; i++) {
        float fa = aria_f16_to_f32(a[i]);
        float fb = aria_f16_to_f32(b[i]);
        y[i] = aria_f32_to_f16(fa * fb);
    }
#endif
}

void aria_matmul_f16(const uint16_t *A, const uint16_t *B, uint16_t *C,
                     int64_t M, int64_t K, int64_t N) {
    float *Af = (float *)malloc(sizeof(float) * (size_t)(M * K));
    float *Bf = (float *)malloc(sizeof(float) * (size_t)(K * N));
    float *Cf = (float *)malloc(sizeof(float) * (size_t)(M * N));
    int64_t total_a = M * K;
#ifdef __F16C__
    {
        int64_t vec_end = total_a - (total_a % 8);
        for (int64_t i = 0; i < vec_end; i += 8) {
            Af[i/8*8] = 0; // dummy to use i
            __m128i vh = _mm_loadu_si128((const __m128i *)(A + i));
            _mm256_storeu_ps(Af + i, _mm256_cvtph_ps(vh));
        }
        for (int64_t i = vec_end; i < total_a; i++) Af[i] = aria_f16_to_f32(A[i]);
    }
#else
    for (int64_t i = 0; i < total_a; i++) Af[i] = aria_f16_to_f32(A[i]);
#endif
    int64_t total_b = K * N;
#ifdef __F16C__
    {
        int64_t vec_end = total_b - (total_b % 8);
        for (int64_t i = 0; i < vec_end; i += 8) {
            __m128i vh = _mm_loadu_si128((const __m128i *)(B + i));
            _mm256_storeu_ps(Bf + i, _mm256_cvtph_ps(vh));
        }
        for (int64_t i = vec_end; i < total_b; i++) Bf[i] = aria_f16_to_f32(B[i]);
    }
#else
    for (int64_t i = 0; i < total_b; i++) Bf[i] = aria_f16_to_f32(B[i]);
#endif
    aria_matmul_f32(Af, Bf, Cf, M, K, N);
    int64_t total_c = M * N;
#ifdef __F16C__
    {
        int64_t vec_end = total_c - (total_c % 8);
        for (int64_t i = 0; i < vec_end; i += 8) {
            __m256 vf = _mm256_loadu_ps(Cf + i);
            _mm_storeu_si128((__m128i *)(C + i), _mm256_cvtps_ph(vf, _MM_FROUND_TO_NEAREST_INT));
        }
        for (int64_t i = vec_end; i < total_c; i++) C[i] = aria_f32_to_f16(Cf[i]);
    }
#else
    for (int64_t i = 0; i < total_c; i++) C[i] = aria_f32_to_f16(Cf[i]);
#endif
    free(Af); free(Bf); free(Cf);
}

void aria_softmax_f16(const uint16_t *x, uint16_t *y, int64_t batch, int64_t dim) {
    float *xf = (float *)malloc(sizeof(float) * (size_t)dim);
    float *yf = (float *)malloc(sizeof(float) * (size_t)dim);
    for (int64_t b = 0; b < batch; b++) {
        const uint16_t *xb = x + b * dim;
        uint16_t *yb = y + b * dim;
#ifdef __F16C__
        {
            int64_t vec_end = dim - (dim % 8);
            for (int64_t i = 0; i < vec_end; i += 8) {
                __m128i vh = _mm_loadu_si128((const __m128i *)(xb + i));
                _mm256_storeu_ps(xf + i, _mm256_cvtph_ps(vh));
            }
            for (int64_t i = vec_end; i < dim; i++) xf[i] = aria_f16_to_f32(xb[i]);
        }
#else
        for (int64_t i = 0; i < dim; i++) xf[i] = aria_f16_to_f32(xb[i]);
#endif
        aria_softmax_f32(xf, yf, 1, dim);
#ifdef __F16C__
        {
            int64_t vec_end = dim - (dim % 8);
            for (int64_t i = 0; i < vec_end; i += 8) {
                __m256 vf = _mm256_loadu_ps(yf + i);
                _mm_storeu_si128((__m128i *)(yb + i), _mm256_cvtps_ph(vf, _MM_FROUND_TO_NEAREST_INT));
            }
            for (int64_t i = vec_end; i < dim; i++) yb[i] = aria_f32_to_f16(yf[i]);
        }
#else
        for (int64_t i = 0; i < dim; i++) yb[i] = aria_f32_to_f16(yf[i]);
#endif
    }
    free(xf); free(yf);
}

void aria_rmsnorm_f16(const uint16_t *x, const uint16_t *weight, uint16_t *y,
                      int64_t batch, int64_t dim, float eps) {
    float *wf = (float *)malloc(sizeof(float) * (size_t)dim);
#ifdef __F16C__
    {
        int64_t vec_end = dim - (dim % 8);
        for (int64_t i = 0; i < vec_end; i += 8) {
            __m128i vh = _mm_loadu_si128((const __m128i *)(weight + i));
            _mm256_storeu_ps(wf + i, _mm256_cvtph_ps(vh));
        }
        for (int64_t i = vec_end; i < dim; i++) wf[i] = aria_f16_to_f32(weight[i]);
    }
#else
    for (int64_t i = 0; i < dim; i++) wf[i] = aria_f16_to_f32(weight[i]);
#endif
    float *xf = (float *)malloc(sizeof(float) * (size_t)dim);
    float *yf = (float *)malloc(sizeof(float) * (size_t)dim);
    for (int64_t b = 0; b < batch; b++) {
        const uint16_t *xb = x + b * dim;
        uint16_t *yb = y + b * dim;
#ifdef __F16C__
        {
            int64_t vec_end = dim - (dim % 8);
            for (int64_t i = 0; i < vec_end; i += 8) {
                __m128i vh = _mm_loadu_si128((const __m128i *)(xb + i));
                _mm256_storeu_ps(xf + i, _mm256_cvtph_ps(vh));
            }
            for (int64_t i = vec_end; i < dim; i++) xf[i] = aria_f16_to_f32(xb[i]);
        }
#else
        for (int64_t i = 0; i < dim; i++) xf[i] = aria_f16_to_f32(xb[i]);
#endif
        aria_rmsnorm_f32(xf, wf, yf, 1, dim, eps);
#ifdef __F16C__
        {
            int64_t vec_end = dim - (dim % 8);
            for (int64_t i = 0; i < vec_end; i += 8) {
                __m256 vf = _mm256_loadu_ps(yf + i);
                _mm_storeu_si128((__m128i *)(yb + i), _mm256_cvtps_ph(vf, _MM_FROUND_TO_NEAREST_INT));
            }
            for (int64_t i = vec_end; i < dim; i++) yb[i] = aria_f32_to_f16(yf[i]);
        }
#else
        for (int64_t i = 0; i < dim; i++) yb[i] = aria_f32_to_f16(yf[i]);
#endif
    }
    free(wf); free(xf); free(yf);
}
