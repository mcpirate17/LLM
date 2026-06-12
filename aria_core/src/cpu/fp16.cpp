#include "kernels_common.h"
#include "simd_elementwise.h"

/* ══════════════════════════════════════════════════════════════════════
 * FP16 (HALF-PRECISION) KERNELS
 *
 * Strategy: convert-at-boundaries.
 * - Load fp16 (uint16_t) → convert to f32 (F16C / AVX-512)
 * - Compute in f32 — the same AriaOp* structs the f32 kernels use
 * - Convert f32 → fp16 → store
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

/* ── SIMD fp16 ↔ fp32 conversion at the native vector width ────────── */

#if defined(__AVX512F__)
typedef __m256i aria_simd_h;
#define aria_simd_loadu_h(p) _mm256_loadu_si256((const __m256i *)(p))
#define aria_simd_storeu_h(p, v) _mm256_storeu_si256((__m256i *)(p), (v))
#define aria_simd_cvtph(vh) _mm512_cvtph_ps(vh)
#define aria_simd_cvtps(vf) _mm512_cvtps_ph((vf), _MM_FROUND_TO_NEAREST_INT)
#define ARIA_F16_SIMD 1
#elif defined(__AVX2__) && defined(__F16C__)
typedef __m128i aria_simd_h;
#define aria_simd_loadu_h(p) _mm_loadu_si128((const __m128i *)(p))
#define aria_simd_storeu_h(p, v) _mm_storeu_si128((__m128i *)(p), (v))
#define aria_simd_cvtph(vh) _mm256_cvtph_ps(vh)
#define aria_simd_cvtps(vf) _mm256_cvtps_ph((vf), _MM_FROUND_TO_NEAREST_INT)
#define ARIA_F16_SIMD 1
#endif

/* Elementwise loop scaffolding — fp16 mirror of ARIA_EW_UNARY/BINARY. */

#ifdef ARIA_F16_SIMD

template <typename VecOp, typename ScalarOp>
static inline void aria_ew_unary_f16_impl(const uint16_t *x, uint16_t *y,
                                          int64_t n, VecOp vop, ScalarOp sop) {
    int64_t vec_end = n - (n % ARIA_SIMD_WIDTH);
    for (int64_t i = 0; i < vec_end; i += ARIA_SIMD_WIDTH) {
        aria_simd_h vh = aria_simd_loadu_h(x + i);
        aria_simd_storeu_h(y + i, aria_simd_cvtps(vop(aria_simd_cvtph(vh))));
    }
    for (int64_t i = vec_end; i < n; i++) {
        y[i] = aria_f32_to_f16(sop(aria_f16_to_f32(x[i])));
    }
}

template <typename VecOp, typename ScalarOp>
static inline void aria_ew_binary_f16_impl(const uint16_t *a, const uint16_t *b,
                                           uint16_t *y, int64_t n, VecOp vop,
                                           ScalarOp sop) {
    int64_t vec_end = n - (n % ARIA_SIMD_WIDTH);
    for (int64_t i = 0; i < vec_end; i += ARIA_SIMD_WIDTH) {
        aria_simd_ps va = aria_simd_cvtph(aria_simd_loadu_h(a + i));
        aria_simd_ps vb = aria_simd_cvtph(aria_simd_loadu_h(b + i));
        aria_simd_storeu_h(y + i, aria_simd_cvtps(vop(va, vb)));
    }
    for (int64_t i = vec_end; i < n; i++) {
        y[i] = aria_f32_to_f16(sop(aria_f16_to_f32(a[i]), aria_f16_to_f32(b[i])));
    }
}

#define ARIA_EW_UNARY_F16(x, y, n, vop, sop) \
    aria_ew_unary_f16_impl((x), (y), (n), vop, sop)
#define ARIA_EW_BINARY_F16(a, b, y, n, vop, sop) \
    aria_ew_binary_f16_impl((a), (b), (y), (n), vop, sop)

#else /* !ARIA_F16_SIMD */

template <typename ScalarOp>
static inline void aria_ew_unary_f16_scalar(const uint16_t *x, uint16_t *y,
                                            int64_t n, ScalarOp sop) {
    for (int64_t i = 0; i < n; i++) {
        y[i] = aria_f32_to_f16(sop(aria_f16_to_f32(x[i])));
    }
}

template <typename ScalarOp>
static inline void aria_ew_binary_f16_scalar(const uint16_t *a, const uint16_t *b,
                                             uint16_t *y, int64_t n, ScalarOp sop) {
    for (int64_t i = 0; i < n; i++) {
        y[i] = aria_f32_to_f16(sop(aria_f16_to_f32(a[i]), aria_f16_to_f32(b[i])));
    }
}

#define ARIA_EW_UNARY_F16(x, y, n, vop, sop) \
    aria_ew_unary_f16_scalar((x), (y), (n), sop)
#define ARIA_EW_BINARY_F16(a, b, y, n, vop, sop) \
    aria_ew_binary_f16_scalar((a), (b), (y), (n), sop)

#endif /* ARIA_F16_SIMD */

/* Whole-buffer conversion — used by the convert→f32-kernel→convert wrappers. */

static inline void aria_f16_to_f32_buf(const uint16_t *h, float *f, int64_t n) {
#ifdef ARIA_F16_SIMD
    int64_t vec_end = n - (n % ARIA_SIMD_WIDTH);
    for (int64_t i = 0; i < vec_end; i += ARIA_SIMD_WIDTH) {
        aria_simd_storeu_ps(f + i, aria_simd_cvtph(aria_simd_loadu_h(h + i)));
    }
    for (int64_t i = vec_end; i < n; i++) f[i] = aria_f16_to_f32(h[i]);
#else
    for (int64_t i = 0; i < n; i++) f[i] = aria_f16_to_f32(h[i]);
#endif
}

static inline void aria_f32_to_f16_buf(const float *f, uint16_t *h, int64_t n) {
#ifdef ARIA_F16_SIMD
    int64_t vec_end = n - (n % ARIA_SIMD_WIDTH);
    for (int64_t i = 0; i < vec_end; i += ARIA_SIMD_WIDTH) {
        aria_simd_storeu_h(h + i, aria_simd_cvtps(aria_simd_loadu_ps(f + i)));
    }
    for (int64_t i = vec_end; i < n; i++) h[i] = aria_f32_to_f16(f[i]);
#else
    for (int64_t i = 0; i < n; i++) h[i] = aria_f32_to_f16(f[i]);
#endif
}

/* ── Elementwise fp16 kernels ───────────────────────────────────────── */

void aria_relu_f16(const uint16_t *x, uint16_t *y, int64_t n) {
    ARIA_EW_UNARY_F16(x, y, n, (AriaOpRelu::vec), (AriaOpRelu::scalar));
}

void aria_gelu_f16(const uint16_t *x, uint16_t *y, int64_t n) {
    ARIA_EW_UNARY_F16(x, y, n, (AriaOpGelu::vec), (AriaOpGelu::scalar));
}

void aria_silu_f16(const uint16_t *x, uint16_t *y, int64_t n) {
    ARIA_EW_UNARY_F16(x, y, n, (AriaOpSilu::vec), (AriaOpSilu::scalar));
}

void aria_sigmoid_f16(const uint16_t *x, uint16_t *y, int64_t n) {
    ARIA_EW_UNARY_F16(x, y, n, (AriaOpSigmoid::vec), (AriaOpSigmoid::scalar));
}

void aria_add_f16(const uint16_t *a, const uint16_t *b, uint16_t *y, int64_t n) {
    ARIA_EW_BINARY_F16(a, b, y, n, (AriaOpAdd::vec), (AriaOpAdd::scalar));
}

void aria_mul_f16(const uint16_t *a, const uint16_t *b, uint16_t *y, int64_t n) {
    ARIA_EW_BINARY_F16(a, b, y, n, (AriaOpMul::vec), (AriaOpMul::scalar));
}

/* ── Convert → f32 kernel → convert wrappers ────────────────────────── */

void aria_matmul_f16(const uint16_t *A, const uint16_t *B, uint16_t *C,
                     int64_t M, int64_t K, int64_t N) {
    float *Af = (float *)malloc(sizeof(float) * (size_t)(M * K));
    float *Bf = (float *)malloc(sizeof(float) * (size_t)(K * N));
    float *Cf = (float *)malloc(sizeof(float) * (size_t)(M * N));
    if (!Af || !Bf || !Cf) {
        memset(C, 0, sizeof(uint16_t) * (size_t)(M * N));
        free(Af); free(Bf); free(Cf);
        return;
    }
    aria_f16_to_f32_buf(A, Af, M * K);
    aria_f16_to_f32_buf(B, Bf, K * N);
    aria_matmul_f32(Af, Bf, Cf, M, K, N);
    aria_f32_to_f16_buf(Cf, C, M * N);
    free(Af); free(Bf); free(Cf);
}

void aria_softmax_f16(const uint16_t *x, uint16_t *y, int64_t batch, int64_t dim) {
    float *xf = (float *)malloc(sizeof(float) * (size_t)dim);
    float *yf = (float *)malloc(sizeof(float) * (size_t)dim);
    if (!xf || !yf) { free(xf); free(yf); return; }
    for (int64_t b = 0; b < batch; b++) {
        aria_f16_to_f32_buf(x + b * dim, xf, dim);
        aria_softmax_f32(xf, yf, 1, dim);
        aria_f32_to_f16_buf(yf, y + b * dim, dim);
    }
    free(xf); free(yf);
}

void aria_rmsnorm_f16(const uint16_t *x, const uint16_t *weight, uint16_t *y,
                      int64_t batch, int64_t dim, float eps) {
    float *wf = (float *)malloc(sizeof(float) * (size_t)dim);
    float *xf = (float *)malloc(sizeof(float) * (size_t)dim);
    float *yf = (float *)malloc(sizeof(float) * (size_t)dim);
    if (!wf || !xf || !yf) { free(wf); free(xf); free(yf); return; }
    aria_f16_to_f32_buf(weight, wf, dim);
    for (int64_t b = 0; b < batch; b++) {
        aria_f16_to_f32_buf(x + b * dim, xf, dim);
        aria_rmsnorm_f32(xf, wf, yf, 1, dim, eps);
        aria_f32_to_f16_buf(yf, y + b * dim, dim);
    }
    free(wf); free(xf); free(yf);
}
