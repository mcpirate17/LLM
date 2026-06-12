#ifndef ARIA_SIMD_ELEMENTWISE_H
#define ARIA_SIMD_ELEMENTWISE_H

#ifndef ARIA_KERNELS_COMMON_H
#include "kernels_common.h"
#endif

/* Shared scaffolding for elementwise f32 kernels: one place owns the SIMD
 * main loop, the scalar tail, the no-SIMD fallback, and the OpenMP gating.
 * Kernels supply the op twice — a vector lambda over aria_simd_ps and a
 * scalar lambda — through the ARIA_EW_* macros. On platforms without
 * ARIA_SIMD_WIDTH the macros drop the vector lambda argument before it is
 * ever compiled, so call sites may reference aria_simd_* names freely.
 * Wrap each lambda argument in parentheses to protect commas. */

template <typename ScalarOp>
static inline void aria_ew_unary_scalar_f32(const float *x, float *y, int64_t n,
                                            ScalarOp sop) {
#ifdef ARIA_HAS_OPENMP
    #pragma omp parallel for if(n > ARIA_OMP_THRESHOLD) schedule(static)
#endif
    for (int64_t i = 0; i < n; i++) y[i] = sop(x[i]);
}

template <typename ScalarOp>
static inline void aria_ew_binary_scalar_f32(const float *a, const float *b,
                                             float *y, int64_t n, ScalarOp sop) {
#ifdef ARIA_HAS_OPENMP
    #pragma omp parallel for if(n > ARIA_OMP_THRESHOLD) schedule(static)
#endif
    for (int64_t i = 0; i < n; i++) y[i] = sop(a[i], b[i]);
}

#if defined(ARIA_SIMD_WIDTH)

template <typename VecOp, typename ScalarOp>
static inline void aria_ew_unary_f32(const float *x, float *y, int64_t n,
                                     VecOp vop, ScalarOp sop) {
    int64_t vec_end = n - (n % ARIA_SIMD_WIDTH);
#ifdef ARIA_HAS_OPENMP
    #pragma omp parallel for if(n > ARIA_OMP_THRESHOLD) schedule(static)
#endif
    for (int64_t i = 0; i < vec_end; i += ARIA_SIMD_WIDTH) {
        aria_simd_storeu_ps(y + i, vop(aria_simd_loadu_ps(x + i)));
    }
    for (int64_t i = vec_end; i < n; i++) y[i] = sop(x[i]);
}

template <typename VecOp, typename ScalarOp>
static inline void aria_ew_binary_f32(const float *a, const float *b, float *y,
                                      int64_t n, VecOp vop, ScalarOp sop) {
    int64_t vec_end = n - (n % ARIA_SIMD_WIDTH);
#ifdef ARIA_HAS_OPENMP
    #pragma omp parallel for if(n > ARIA_OMP_THRESHOLD) schedule(static)
#endif
    for (int64_t i = 0; i < vec_end; i += ARIA_SIMD_WIDTH) {
        aria_simd_storeu_ps(
            y + i, vop(aria_simd_loadu_ps(a + i), aria_simd_loadu_ps(b + i)));
    }
    for (int64_t i = vec_end; i < n; i++) y[i] = sop(a[i], b[i]);
}

#define ARIA_EW_UNARY(x, y, n, vop, sop) \
    aria_ew_unary_f32((x), (y), (n), vop, sop)
#define ARIA_EW_BINARY(a, b, y, n, vop, sop) \
    aria_ew_binary_f32((a), (b), (y), (n), vop, sop)

#else /* !ARIA_SIMD_WIDTH */

#define ARIA_EW_UNARY(x, y, n, vop, sop) \
    aria_ew_unary_scalar_f32((x), (y), (n), sop)
#define ARIA_EW_BINARY(a, b, y, n, vop, sop) \
    aria_ew_binary_scalar_f32((a), (b), (y), (n), sop)

#endif /* ARIA_SIMD_WIDTH */

/* Ops used by both the f32 (unary.cpp / binary.cpp) and f16 (fp16.cpp)
 * kernels — vector and scalar math defined exactly once. */

struct AriaOpRelu {
    static inline float scalar(float v) { return v > 0.0f ? v : 0.0f; }
#if defined(ARIA_SIMD_WIDTH)
    static inline aria_simd_ps vec(aria_simd_ps vx) {
        return aria_simd_max_ps(vx, aria_simd_zero_ps);
    }
#endif
};

/* GELU — tanh approximation; tanh(x) = 2*sigmoid(2x) - 1 in the SIMD body */
struct AriaOpGelu {
    static inline float scalar(float v) {
        float v3 = v * v * v;
        return 0.5f * v * (1.0f + tanhf(GELU_COEFF * (v + GELU_CUBIC * v3)));
    }
#if defined(ARIA_SIMD_WIDTH)
    static inline aria_simd_ps vec(aria_simd_ps vx) {
        aria_simd_ps half = aria_simd_set1_ps(0.5f);
        aria_simd_ps one = aria_simd_set1_ps(1.0f);
        aria_simd_ps two = aria_simd_set1_ps(2.0f);
        aria_simd_ps coeff = aria_simd_set1_ps(GELU_COEFF);
        aria_simd_ps cubic = aria_simd_set1_ps(GELU_CUBIC);
        aria_simd_ps vx2 = aria_simd_mul_ps(vx, vx);
        aria_simd_ps vx3 = aria_simd_mul_ps(vx2, vx);
        aria_simd_ps inner =
            aria_simd_mul_ps(coeff, aria_simd_fmadd_ps(cubic, vx3, vx));
        aria_simd_ps sig = aria_simd_sigmoid_ps(aria_simd_mul_ps(two, inner));
        aria_simd_ps tanh_val = aria_simd_sub_ps(aria_simd_mul_ps(two, sig), one);
        return aria_simd_mul_ps(
            half, aria_simd_mul_ps(vx, aria_simd_add_ps(one, tanh_val)));
    }
#endif
};

struct AriaOpSilu {
    static inline float scalar(float v) { return v / (1.0f + expf(-v)); }
#if defined(ARIA_SIMD_WIDTH)
    static inline aria_simd_ps vec(aria_simd_ps vx) {
        return aria_simd_mul_ps(vx, aria_simd_sigmoid_ps(vx));
    }
#endif
};

struct AriaOpSigmoid {
    static inline float scalar(float v) { return 1.0f / (1.0f + expf(-v)); }
#if defined(ARIA_SIMD_WIDTH)
    static inline aria_simd_ps vec(aria_simd_ps vx) {
        return aria_simd_sigmoid_ps(vx);
    }
#endif
};

struct AriaOpAdd {
    static inline float scalar(float av, float bv) { return av + bv; }
#if defined(ARIA_SIMD_WIDTH)
    static inline aria_simd_ps vec(aria_simd_ps va, aria_simd_ps vb) {
        return aria_simd_add_ps(va, vb);
    }
#endif
};

struct AriaOpMul {
    static inline float scalar(float av, float bv) { return av * bv; }
#if defined(ARIA_SIMD_WIDTH)
    static inline aria_simd_ps vec(aria_simd_ps va, aria_simd_ps vb) {
        return aria_simd_mul_ps(va, vb);
    }
#endif
};

#endif /* ARIA_SIMD_ELEMENTWISE_H */
