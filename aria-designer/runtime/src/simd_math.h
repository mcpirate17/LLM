#ifndef ARIA_SIMD_MATH_H
#define ARIA_SIMD_MATH_H

#ifdef __AVX2__
#include <immintrin.h>

/*
 * Fast vectorized exp(x) for 8 floats using AVX2.
 *
 * Uses the Schraudolph/Cephes approach:
 *   exp(x) = 2^(x * log2(e))
 *   Split into integer + fractional parts, use polynomial for 2^frac.
 *
 * Accuracy: max relative error ~1e-6 for |x| < 88
 * This matches f32 precision needs and is ~4-8x faster than scalar expf().
 */
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

    /* 2^f approximation using minimax polynomial (degree 5)
     * p(f) = 2^f for f in [-0.5, 0.5]
     * Coefficients from Cephes library */
    const __m256 c0 = _mm256_set1_ps(1.0f);
    const __m256 c1 = _mm256_set1_ps(6.93145751953125e-1f);
    const __m256 c2 = _mm256_set1_ps(2.4027631282311e-1f);
    const __m256 c3 = _mm256_set1_ps(5.5505126898933e-2f);
    const __m256 c4 = _mm256_set1_ps(9.6178371864552e-3f);
    const __m256 c5 = _mm256_set1_ps(1.3333558146428e-3f);

    /* Horner's method: p = c0 + f*(c1 + f*(c2 + f*(c3 + f*(c4 + f*c5)))) */
    __m256 p = _mm256_fmadd_ps(f, c5, c4);
    p = _mm256_fmadd_ps(f, p, c3);
    p = _mm256_fmadd_ps(f, p, c2);
    p = _mm256_fmadd_ps(f, p, c1);
    p = _mm256_fmadd_ps(f, p, c0);

    /* Scale by 2^n: multiply p by 2^n using integer bit manipulation
     * 2^n = reinterpret((n + 127) << 23) as float */
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

#endif /* __AVX2__ */
#endif /* ARIA_SIMD_MATH_H */
