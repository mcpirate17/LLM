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

/* Fast vectorized sigmoid: 1 / (1 + exp(-x)) */
static inline __m256 _mm256_sigmoid_ps(__m256 x) {
    __m256 one = _mm256_set1_ps(1.0f);
    __m256 neg_x = _mm256_sub_ps(_mm256_setzero_ps(), x);
    __m256 exp_neg = _mm256_exp_ps(neg_x);
    return _mm256_div_ps(one, _mm256_add_ps(one, exp_neg));
}
#endif

#ifdef __AVX512F__
#define aria_simd_exp_ps _mm512_exp_ps
#define aria_simd_sigmoid_ps _mm512_sigmoid_ps
#elif defined(__AVX2__)
#define aria_simd_exp_ps _mm256_exp_ps
#define aria_simd_sigmoid_ps _mm256_sigmoid_ps
#endif

#endif /* ARIA_SIMD_MATH_H */
