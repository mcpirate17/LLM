#ifndef ARIA_SIMD_MATH_H
#define ARIA_SIMD_MATH_H

#ifdef __AVX512F__
#include <immintrin.h>
#elif defined(__AVX2__)
#include <immintrin.h>
#endif

/* ── AVX-512 / AVX2 Abstractions ───────────────────────────────────── */

#ifdef __AVX512F__
typedef __m512 aria_simd_ps;
#define ARIA_SIMD_WIDTH 16
#define aria_simd_set1_ps _mm512_set1_ps
#define aria_simd_loadu_ps _mm512_loadu_ps
#define aria_simd_storeu_ps _mm512_storeu_ps
#define aria_simd_add_ps _mm512_add_ps
#define aria_simd_sub_ps _mm512_sub_ps
#define aria_simd_mul_ps _mm512_mul_ps
#define aria_simd_div_ps _mm512_div_ps
#define aria_simd_fmadd_ps _mm512_fmadd_ps
#define aria_simd_fnmadd_ps _mm512_fnmadd_ps
#define aria_simd_fmsub_ps _mm512_fmsub_ps
#define aria_simd_min_ps _mm512_min_ps
#define aria_simd_max_ps _mm512_max_ps
#define aria_simd_and_ps _mm512_and_ps
#define aria_simd_castsi_ps _mm512_castsi512_ps
#define aria_simd_set1_epi32 _mm512_set1_epi32
#define aria_simd_zero_ps _mm512_setzero_ps()
#elif defined(__AVX2__)
typedef __m256 aria_simd_ps;
#define ARIA_SIMD_WIDTH 8
#define aria_simd_set1_ps _mm256_set1_ps
#define aria_simd_loadu_ps _mm256_loadu_ps
#define aria_simd_storeu_ps _mm256_storeu_ps
#define aria_simd_add_ps _mm256_add_ps
#define aria_simd_sub_ps _mm256_sub_ps
#define aria_simd_mul_ps _mm256_mul_ps
#define aria_simd_div_ps _mm256_div_ps
#define aria_simd_fmadd_ps _mm256_fmadd_ps
#define aria_simd_fnmadd_ps _mm256_fnmadd_ps
#define aria_simd_fmsub_ps _mm256_fmsub_ps
#define aria_simd_min_ps _mm256_min_ps
#define aria_simd_max_ps _mm256_max_ps
#define aria_simd_and_ps _mm256_and_ps
#define aria_simd_castsi_ps _mm256_castsi256_ps
#define aria_simd_set1_epi32 _mm256_set1_epi32
#define aria_simd_zero_ps _mm256_setzero_ps()
#endif

/*
 * Fast vectorized exp(x) for 8/16 floats.
 */
#ifdef __AVX512F__
static inline __m512 _mm512_exp_ps(__m512 x) {
    const __m512 log2e = _mm512_set1_ps(1.44269504088896341f);
    const __m512 half  = _mm512_set1_ps(0.5f);
    const __m512 hi = _mm512_set1_ps(88.3762626647949f);
    const __m512 lo = _mm512_set1_ps(-88.3762626647949f);
    x = _mm512_min_ps(x, hi);
    x = _mm512_max_ps(x, lo);
    __m512 t = _mm512_mul_ps(x, log2e);
    __m512 n = _mm512_roundscale_ps(_mm512_add_ps(t, half), _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC);
    __m512 f = _mm512_sub_ps(t, n);
    const __m512 c0 = _mm512_set1_ps(1.0f);
    const __m512 c1 = _mm512_set1_ps(6.93145751953125e-1f);
    const __m512 c2 = _mm512_set1_ps(2.4027631282311e-1f);
    const __m512 c3 = _mm512_set1_ps(5.5505126898933e-2f);
    const __m512 c4 = _mm512_set1_ps(9.6178371864552e-3f);
    const __m512 c5 = _mm512_set1_ps(1.3333558146428e-3f);
    __m512 p = _mm512_fmadd_ps(f, c5, c4);
    p = _mm512_fmadd_ps(f, p, c3);
    p = _mm512_fmadd_ps(f, p, c2);
    p = _mm512_fmadd_ps(f, p, c1);
    p = _mm512_fmadd_ps(f, p, c0);
    __m512i ni = _mm512_cvtps_epi32(n);
    ni = _mm512_add_epi32(ni, _mm512_set1_epi32(127));
    ni = _mm512_slli_epi32(ni, 23);
    __m512 pow2n = _mm512_castsi512_ps(ni);
    return _mm512_mul_ps(p, pow2n);
}

static inline __m512 _mm512_sigmoid_ps(__m512 x) {
    __m512 one = _mm512_set1_ps(1.0f);
    __m512 neg_x = _mm512_sub_ps(_mm512_setzero_ps(), x);
    __m512 exp_neg = _mm512_exp_ps(neg_x);
    return _mm512_div_ps(one, _mm512_add_ps(one, exp_neg));
}
#endif

#ifdef __AVX2__
static inline __m256 _mm256_exp_ps(__m256 x) {
    /* Constants */
    const __m256 log2e = _mm256_set1_ps(1.44269504088896341f);
    const __m256 half  = _mm256_set1_ps(0.5f);

    /* Clamp to avoid overflow/underflow */
    const __m256 hi = _mm256_set1_ps(88.3762626647949f);
    const __m256 lo = _mm256_set1_ps(-88.3762626647949f);
    x = _mm256_min_ps(x, hi);
    x = _mm256_max_ps(x, lo);

    /* t = x * log2(e) */
    __m256 t = _mm256_mul_ps(x, log2e);

    /* n = floor(t + 0.5) = round(t) */
    __m256 n = _mm256_round_ps(_mm256_add_ps(t, half),
                                _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC);

    /* f = t - n (fractional part, in [-0.5, 0.5]) */
    __m256 f = _mm256_sub_ps(t, n);

    /* 2^f approximation using minimax polynomial (degree 5) */
    const __m256 c0 = _mm256_set1_ps(1.0f);
    const __m256 c1 = _mm256_set1_ps(6.93145751953125e-1f);
    const __m256 c2 = _mm256_set1_ps(2.4027631282311e-1f);
    const __m256 c3 = _mm256_set1_ps(5.5505126898933e-2f);
    const __m256 c4 = _mm256_set1_ps(9.6178371864552e-3f);
    const __m256 c5 = _mm256_set1_ps(1.3333558146428e-3f);

    /* Horner's method */
    __m256 p = _mm256_fmadd_ps(f, c5, c4);
    p = _mm256_fmadd_ps(f, p, c3);
    p = _mm256_fmadd_ps(f, p, c2);
    p = _mm256_fmadd_ps(f, p, c1);
    p = _mm256_fmadd_ps(f, p, c0);

    __m256i ni = _mm256_cvtps_epi32(n);
    ni = _mm256_add_epi32(ni, _mm256_set1_epi32(127));
    ni = _mm256_slli_epi32(ni, 23);
    __m256 pow2n = _mm256_castsi256_ps(ni);

    return _mm256_mul_ps(p, pow2n);
}

/* Fast vectorized log(x) for 8 floats (AVX2) */
static inline __m256 _mm256_log_ps(__m256 x) {
    const __m256 one = _mm256_set1_ps(1.0f);
    const __m256 half = _mm256_set1_ps(0.5f);
    const __m256 ln2_hi = _mm256_set1_ps(0.693359375f);
    const __m256 ln2_lo = _mm256_set1_ps(-0.00021219444f);

    /* Extract exponent and mantissa */
    __m256i xi = _mm256_castps_si256(x);
    __m256i e = _mm256_srli_epi32(_mm256_and_si256(xi, _mm256_set1_epi32(0x7f800000)), 23);
    e = _mm256_sub_epi32(e, _mm256_set1_epi32(127));
    __m256 m = _mm256_castsi256_ps(_mm256_or_si256(_mm256_and_si256(xi, _mm256_set1_epi32(0x007fffff)), _mm256_set1_epi32(0x3f800000)));

    /* m = m - 1 if m > sqrt(2) */
    const __m256 sqrt2 = _mm256_set1_ps(1.41421356f);
    __m256 mask = _mm256_cmp_ps(m, sqrt2, _CMP_GT_OQ);
    __m256 e_adj = _mm256_and_ps(mask, one);
    m = _mm256_sub_ps(m, e_adj);
    m = _mm256_sub_ps(m, one);
    __m256 fe = _mm256_add_ps(_mm256_cvtepi32_ps(e), e_adj);

    /* Rational approximation: ln(1+x) approx x - x^2/2 + x^3/3 ... */
    __m256 x2 = _mm256_mul_ps(m, m);
    __m256 y = _mm256_fmadd_ps(m, _mm256_set1_ps(0.07037608f), _mm256_set1_ps(-0.11509069f));
    y = _mm256_fmadd_ps(m, y, _mm256_set1_ps(0.11696452f));
    y = _mm256_fmadd_ps(m, y, _mm256_set1_ps(-0.16201473f));
    y = _mm256_fmadd_ps(m, y, _mm256_set1_ps(0.20033587f));
    y = _mm256_fmadd_ps(m, y, _mm256_set1_ps(-0.24992420f));
    y = _mm256_fmadd_ps(m, y, _mm256_set1_ps(0.33333192f));
    y = _mm256_mul_ps(_mm256_mul_ps(x2, m), y);
    y = _mm256_sub_ps(y, _mm256_mul_ps(x2, half));
    y = _mm256_add_ps(y, m);

    /* result = y + e * ln(2) */
    y = _mm256_fmadd_ps(fe, ln2_hi, y);
    y = _mm256_fmadd_ps(fe, ln2_lo, y);
    return y;
}

static inline __m256 _mm256_tanh_ps(__m256 x) {
    __m256 e2x = _mm256_exp_ps(_mm256_mul_ps(x, _mm256_set1_ps(2.0f)));
    __m256 one = _mm256_set1_ps(1.0f);
    return _mm256_div_ps(_mm256_sub_ps(e2x, one), _mm256_add_ps(e2x, one));
}

static inline __m256 _mm256_atanh_ps(__m256 x) {
    __m256 one = _mm256_set1_ps(1.0f);
    __m256 num = _mm256_add_ps(one, x);
    __m256 den = _mm256_sub_ps(one, x);
    return _mm256_mul_ps(_mm256_set1_ps(0.5f), _mm256_log_ps(_mm256_div_ps(num, den)));
}

/* Fast vectorized sigmoid: 1 / (1 + exp(-x)) */
static inline __m256 _mm256_sigmoid_ps(__m256 x) {
    __m256 one = _mm256_set1_ps(1.0f);
    __m256 neg_x = _mm256_sub_ps(_mm256_setzero_ps(), x);
    __m256 exp_neg = _mm256_exp_ps(neg_x);
    return _mm256_div_ps(one, _mm256_add_ps(one, exp_neg));
}
#endif

#ifdef __AVX512F__
static inline __m512 _mm512_log_ps(__m512 x) {
    const __m512 one = _mm512_set1_ps(1.0f);
    const __m512 half = _mm512_set1_ps(0.5f);
    const __m512 ln2_hi = _mm512_set1_ps(0.693359375f);
    const __m512 ln2_lo = _mm512_set1_ps(-0.00021219444f);
    __m512i xi = _mm512_castps_si512(x);
    __m512i e = _mm512_srli_epi32(_mm512_and_si512(xi, _mm512_set1_epi32(0x7f800000)), 23);
    e = _mm512_sub_epi32(e, _mm512_set1_epi32(127));
    __m512 m = _mm512_castsi512_ps(_mm512_or_si512(_mm512_and_si512(xi, _mm512_set1_epi32(0x007fffff)), _mm512_set1_epi32(0x3f800000)));
    const __m512 sqrt2 = _mm512_set1_ps(1.41421356f);
    __mmask16 gt = _mm512_cmp_ps_mask(m, sqrt2, _CMP_GT_OQ);
    __m512 e_adj = _mm512_maskz_mov_ps(gt, one);
    m = _mm512_sub_ps(m, e_adj);
    m = _mm512_sub_ps(m, one);
    __m512 fe = _mm512_add_ps(_mm512_cvtepi32_ps(e), e_adj);
    __m512 x2 = _mm512_mul_ps(m, m);
    __m512 y = _mm512_fmadd_ps(m, _mm512_set1_ps(0.07037608f), _mm512_set1_ps(-0.11509069f));
    y = _mm512_fmadd_ps(m, y, _mm512_set1_ps(0.11696452f));
    y = _mm512_fmadd_ps(m, y, _mm512_set1_ps(-0.16201473f));
    y = _mm512_fmadd_ps(m, y, _mm512_set1_ps(0.20033587f));
    y = _mm512_fmadd_ps(m, y, _mm512_set1_ps(-0.24992420f));
    y = _mm512_fmadd_ps(m, y, _mm512_set1_ps(0.33333192f));
    y = _mm512_mul_ps(_mm512_mul_ps(x2, m), y);
    y = _mm512_sub_ps(y, _mm512_mul_ps(x2, half));
    y = _mm512_add_ps(y, m);
    y = _mm512_fmadd_ps(fe, ln2_hi, y);
    y = _mm512_fmadd_ps(fe, ln2_lo, y);
    return y;
}

static inline __m512 _mm512_tanh_ps(__m512 x) {
    __m512 e2x = _mm512_exp_ps(_mm512_mul_ps(x, _mm512_set1_ps(2.0f)));
    __m512 one = _mm512_set1_ps(1.0f);
    return _mm512_div_ps(_mm512_sub_ps(e2x, one), _mm512_add_ps(e2x, one));
}

static inline __m512 _mm512_atanh_ps(__m512 x) {
    __m512 one = _mm512_set1_ps(1.0f);
    __m512 num = _mm512_add_ps(one, x);
    __m512 den = _mm512_sub_ps(one, x);
    return _mm512_mul_ps(_mm512_set1_ps(0.5f), _mm512_log_ps(_mm512_div_ps(num, den)));
}

#define aria_simd_exp_ps _mm512_exp_ps
#define aria_simd_log_ps _mm512_log_ps
#define aria_simd_tanh_ps _mm512_tanh_ps
#define aria_simd_atanh_ps _mm512_atanh_ps
#define aria_simd_sigmoid_ps _mm512_sigmoid_ps
#elif defined(__AVX2__)
#define aria_simd_exp_ps _mm256_exp_ps
#define aria_simd_log_ps _mm256_log_ps
#define aria_simd_tanh_ps _mm256_tanh_ps
#define aria_simd_atanh_ps _mm256_atanh_ps
#define aria_simd_sigmoid_ps _mm256_sigmoid_ps
#endif

/* ── Scalar fallback for non-SIMD platforms (ARM, older x86) ────────── */

#if !defined(__AVX512F__) && !defined(__AVX2__)
#include <math.h>

static inline void aria_scalar_exp_f32(const float *in, float *out, int n) {
    for (int i = 0; i < n; i++) out[i] = expf(in[i]);
}
static inline void aria_scalar_log_f32(const float *in, float *out, int n) {
    for (int i = 0; i < n; i++) out[i] = logf(in[i]);
}
static inline void aria_scalar_tanh_f32(const float *in, float *out, int n) {
    for (int i = 0; i < n; i++) out[i] = tanhf(in[i]);
}
static inline void aria_scalar_sigmoid_f32(const float *in, float *out, int n) {
    for (int i = 0; i < n; i++) out[i] = 1.0f / (1.0f + expf(-in[i]));
}
static inline void aria_scalar_atanh_f32(const float *in, float *out, int n) {
    for (int i = 0; i < n; i++) out[i] = atanhf(in[i]);
}
#endif

#endif /* ARIA_SIMD_MATH_H */
