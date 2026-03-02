#include "clifford.h"
#include <string.h>

#include "clifford.h"
#include "simd_math.h"
#include <string.h>

#ifdef __AVX2__
// 8x8 transpose of ps registers
static inline void _transpose8_ps(__m256 &v0, __m256 &v1, __m256 &v2, __m256 &v3,
                                  __m256 &v4, __m256 &v5, __m256 &v6, __m256 &v7) {
    __m256 m0 = _mm256_unpacklo_ps(v0, v1);
    __m256 m1 = _mm256_unpackhi_ps(v0, v1);
    __m256 m2 = _mm256_unpacklo_ps(v2, v3);
    __m256 m3 = _mm256_unpackhi_ps(v2, v3);
    __m256 m4 = _mm256_unpacklo_ps(v4, v5);
    __m256 m5 = _mm256_unpackhi_ps(v4, v5);
    __m256 m6 = _mm256_unpacklo_ps(v6, v7);
    __m256 m7 = _mm256_unpackhi_ps(v6, v7);

    __m256 n0 = _mm256_shuffle_ps(m0, m2, _MM_SHUFFLE(1, 0, 1, 0));
    __m256 n1 = _mm256_shuffle_ps(m0, m2, _MM_SHUFFLE(3, 2, 3, 2));
    __m256 n2 = _mm256_shuffle_ps(m1, m3, _MM_SHUFFLE(1, 0, 1, 0));
    __m256 n3 = _mm256_shuffle_ps(m1, m3, _MM_SHUFFLE(3, 2, 3, 2));
    __m256 n4 = _mm256_shuffle_ps(m4, m6, _MM_SHUFFLE(1, 0, 1, 0));
    __m256 n5 = _mm256_shuffle_ps(m4, m6, _MM_SHUFFLE(3, 2, 3, 2));
    __m256 n6 = _mm256_shuffle_ps(m5, m7, _MM_SHUFFLE(1, 0, 1, 0));
    __m256 n7 = _mm256_shuffle_ps(m5, m7, _MM_SHUFFLE(3, 2, 3, 2));

    v0 = _mm256_permute2f128_ps(n0, n4, 0x20);
    v1 = _mm256_permute2f128_ps(n1, n5, 0x20);
    v2 = _mm256_permute2f128_ps(n2, n6, 0x20);
    v3 = _mm256_permute2f128_ps(n3, n7, 0x20);
    v4 = _mm256_permute2f128_ps(n0, n4, 0x31);
    v5 = _mm256_permute2f128_ps(n1, n5, 0x31);
    v6 = _mm256_permute2f128_ps(n2, n6, 0x31);
    v7 = _mm256_permute2f128_ps(n3, n7, 0x31);
}
#endif

static inline void _gp_cl30_single(const float *ai, const float *bi, float *yi) {
    float a0 = ai[0], a1 = ai[1], a2 = ai[2], a3 = ai[3];
    float a12 = ai[4], a13 = ai[5], a23 = ai[6], a123 = ai[7];

    float b0 = bi[0], b1 = bi[1], b2 = bi[2], b3 = bi[3];
    float b12 = bi[4], b13 = bi[5], b23 = bi[6], b123 = bi[7];

    // Scalar part (grade 0)
    yi[0] = (a0*b0 + a1*b1 + a2*b2 + a3*b3
           - a12*b12 - a13*b13 - a23*b23 - a123*b123);

    // Vector parts (grade 1)
    yi[1] = (a0*b1 + a1*b0 - a2*b12 + a12*b2
           - a3*b13 + a13*b3 + a23*b123 - a123*b23);
    yi[2] = (a0*b2 + a1*b12 + a2*b0 - a12*b1
           - a3*b23 - a13*b123 + a23*b3 + a123*b13);
    yi[3] = (a0*b3 - a1*b13 + a2*b23 + a3*b0
           + a12*b123 + a13*b1 - a23*b2 - a123*b12);

    // Bivector parts (grade 2)
    yi[4] = (a0*b12 + a1*b2 - a2*b1 + a12*b0
            + a3*b123 - a13*b23 + a23*b13 + a123*b3);
    yi[5] = (a0*b13 + a1*b3 - a3*b1 + a13*b0
            - a2*b123 + a12*b23 - a23*b12 - a123*b2);
    yi[6] = (a0*b23 + a2*b3 - a3*b2 + a23*b0
            + a1*b123 - a12*b13 + a13*b12 + a123*b1);

    // Pseudoscalar (grade 3)
    yi[7] = (a0*b123 + a1*b23 - a2*b13 + a3*b12
             + a12*b3 - a13*b2 + a23*b1 + a123*b0);
}

void aria_clifford_geometric_product_cl30_f32(const float *a, const float *b, float *y, int64_t n_multivectors) {
#ifdef __AVX2__
    int64_t i = 0;
    for (; i + 8 <= n_multivectors; i += 8) {
        __m256 va0 = _mm256_loadu_ps(a + (i + 0) * 8);
        __m256 va1 = _mm256_loadu_ps(a + (i + 1) * 8);
        __m256 va2 = _mm256_loadu_ps(a + (i + 2) * 8);
        __m256 va3 = _mm256_loadu_ps(a + (i + 3) * 8);
        __m256 va4 = _mm256_loadu_ps(a + (i + 4) * 8);
        __m256 va5 = _mm256_loadu_ps(a + (i + 5) * 8);
        __m256 va6 = _mm256_loadu_ps(a + (i + 6) * 8);
        __m256 va7 = _mm256_loadu_ps(a + (i + 7) * 8);

        __m256 vb0 = _mm256_loadu_ps(b + (i + 0) * 8);
        __m256 vb1 = _mm256_loadu_ps(b + (i + 1) * 8);
        __m256 vb2 = _mm256_loadu_ps(b + (i + 2) * 8);
        __m256 vb3 = _mm256_loadu_ps(b + (i + 3) * 8);
        __m256 vb4 = _mm256_loadu_ps(b + (i + 4) * 8);
        __m256 vb5 = _mm256_loadu_ps(b + (i + 5) * 8);
        __m256 vb6 = _mm256_loadu_ps(b + (i + 6) * 8);
        __m256 vb7 = _mm256_loadu_ps(b + (i + 7) * 8);

        _transpose8_ps(va0, va1, va2, va3, va4, va5, va6, va7);
        _transpose8_ps(vb0, vb1, vb2, vb3, vb4, vb5, vb6, vb7);

        // Compute 8 multivector results in parallel
        // res[j] = sum_k coeff[j,k] * a[k] * b[perm[j,k]]
        
        // y0 = a0*b0 + a1*b1 + a2*b2 + a3*b3 - a4*b4 - a5*b5 - a6*b6 - a7*b7
        __m256 vy0 = _mm256_mul_ps(va0, vb0);
        vy0 = _mm256_fmadd_ps(va1, vb1, vy0);
        vy0 = _mm256_fmadd_ps(va2, vb2, vy0);
        vy0 = _mm256_fmadd_ps(va3, vb3, vy0);
        vy0 = _mm256_fnmadd_ps(va4, vb4, vy0);
        vy0 = _mm256_fnmadd_ps(va5, vb5, vy0);
        vy0 = _mm256_fnmadd_ps(va6, vb6, vy0);
        vy0 = _mm256_fnmadd_ps(va7, vb7, vy0);

        // y1 = a0*b1 + a1*b0 - a2*b4 + a4*b2 - a3*b5 + a5*b3 + a6*b7 - a7*b6
        __m256 vy1 = _mm256_mul_ps(va0, vb1);
        vy1 = _mm256_fmadd_ps(va1, vb0, vy1);
        vy1 = _mm256_fnmadd_ps(va2, vb4, vy1);
        vy1 = _mm256_fmadd_ps(va4, vb2, vy1);
        vy1 = _mm256_fnmadd_ps(va3, vb5, vy1);
        vy1 = _mm256_fmadd_ps(va5, vb3, vy1);
        vy1 = _mm256_fmadd_ps(va6, vb7, vy1);
        vy1 = _mm256_fnmadd_ps(va7, vb6, vy1);

        // y2 = a0*b2 + a1*b4 + a2*b0 - a4*b1 - a3*b6 - a5*b7 + a6*b3 + a7*b5
        __m256 vy2 = _mm256_mul_ps(va0, vb2);
        vy2 = _mm256_fmadd_ps(va1, vb4, vy2);
        vy2 = _mm256_fmadd_ps(va2, vb0, vy2);
        vy2 = _mm256_fnmadd_ps(va4, vb1, vy2);
        vy2 = _mm256_fnmadd_ps(va3, vb6, vy2);
        vy2 = _mm256_fnmadd_ps(va5, vb7, vy2);
        vy2 = _mm256_fmadd_ps(va6, vb3, vy2);
        vy2 = _mm256_fmadd_ps(va7, vb5, vy2);

        // y3 = a0*b3 - a1*b5 + a2*b6 + a3*b0 + a4*b7 + a5*b1 - a6*b2 - a7*b4
        __m256 vy3 = _mm256_mul_ps(va0, vb3);
        vy3 = _mm256_fnmadd_ps(va1, vb5, vy3);
        vy3 = _mm256_fmadd_ps(va2, vb6, vy3);
        vy3 = _mm256_fmadd_ps(va3, vb0, vy3);
        vy3 = _mm256_fmadd_ps(va4, vb7, vy3);
        vy3 = _mm256_fmadd_ps(va5, vb1, vy3);
        vy3 = _mm256_fnmadd_ps(va6, vb2, vy3);
        vy3 = _mm256_fnmadd_ps(va7, vb4, vy3);

        // y4 = a0*b4 + a1*b2 - a2*b1 + a4*b0 + a3*b7 - a5*b6 + a6*b5 + a7*b3
        __m256 vy4 = _mm256_mul_ps(va0, vb4);
        vy4 = _mm256_fmadd_ps(va1, vb2, vy4);
        vy4 = _mm256_fnmadd_ps(va2, vb1, vy4);
        vy4 = _mm256_fmadd_ps(va4, vb0, vy4);
        vy4 = _mm256_fmadd_ps(va3, vb7, vy4);
        vy4 = _mm256_fnmadd_ps(va5, vb6, vy4);
        vy4 = _mm256_fmadd_ps(va6, vb5, vy4);
        vy4 = _mm256_fmadd_ps(va7, vb3, vy4);

        // y5 = a0*b5 + a1*b3 - a3*b1 + a5*b0 - a2*b7 + a4*b6 - a6*b4 - a7*b2
        __m256 vy5 = _mm256_mul_ps(va0, vb5);
        vy5 = _mm256_fmadd_ps(va1, vb3, vy5);
        vy5 = _mm256_fnmadd_ps(va3, vb1, vy5);
        vy5 = _mm256_fmadd_ps(va5, vb0, vy5);
        vy5 = _mm256_fnmadd_ps(va2, vb7, vy5);
        vy5 = _mm256_fmadd_ps(va4, vb6, vy5);
        vy5 = _mm256_fnmadd_ps(va6, vb4, vy5);
        vy5 = _mm256_fnmadd_ps(va7, vb2, vy5);

        // y6 = a0*b6 + a2*b3 - a3*b2 + a6*b0 + a1*b7 - a4*b5 + a5*b4 + a7*b1
        __m256 vy6 = _mm256_mul_ps(va0, vb6);
        vy6 = _mm256_fmadd_ps(va2, vb3, vy6);
        vy6 = _mm256_fnmadd_ps(va3, vb2, vy6);
        vy6 = _mm256_fmadd_ps(va6, vb0, vy6);
        vy6 = _mm256_fmadd_ps(va1, vb7, vy6);
        vy6 = _mm256_fnmadd_ps(va4, vb5, vy6);
        vy6 = _mm256_fmadd_ps(va5, vb4, vy6);
        vy6 = _mm256_fmadd_ps(va7, vb1, vy6);

        // y7 = a0*b7 + a1*b6 - a2*b5 + a3*b4 + a4*b3 - a5*b2 + a6*b1 + a7*b0
        __m256 vy7 = _mm256_mul_ps(va0, vb7);
        vy7 = _mm256_fmadd_ps(va1, vb6, vy7);
        vy7 = _mm256_fnmadd_ps(va2, vb5, vy7);
        vy7 = _mm256_fmadd_ps(va3, vb4, vy7);
        vy7 = _mm256_fmadd_ps(va4, vb3, vy7);
        vy7 = _mm256_fnmadd_ps(va5, vb2, vy7);
        vy7 = _mm256_fmadd_ps(va6, vb1, vy7);
        vy7 = _mm256_fmadd_ps(va7, vb0, vy7);

        _transpose8_ps(vy0, vy1, vy2, vy3, vy4, vy5, vy6, vy7);

        _mm256_storeu_ps(y + (i + 0) * 8, vy0);
        _mm256_storeu_ps(y + (i + 1) * 8, vy1);
        _mm256_storeu_ps(y + (i + 2) * 8, vy2);
        _mm256_storeu_ps(y + (i + 3) * 8, vy3);
        _mm256_storeu_ps(y + (i + 4) * 8, vy4);
        _mm256_storeu_ps(y + (i + 5) * 8, vy5);
        _mm256_storeu_ps(y + (i + 6) * 8, vy6);
        _mm256_storeu_ps(y + (i + 7) * 8, vy7);
    }
    // Scalar tail
    for (; i < n_multivectors; i++) {
        _gp_cl30_single(a + i * 8, b + i * 8, y + i * 8);
    }
#else
    #pragma omp parallel for if(n_multivectors > 1024) schedule(static)
    for (int64_t i = 0; i < n_multivectors; i++) {
        _gp_cl30_single(a + i * 8, b + i * 8, y + i * 8);
    }
#endif
}

void aria_clifford_rotor_transform_cl30_f32(const float *x, const float *rotor, float *y, int64_t n_multivectors) {
    #pragma omp parallel for if(n_multivectors > 512) schedule(static)
    for (int64_t i = 0; i < n_multivectors; i++) {
        const float *xi = x + i * 8;
        const float *ri = rotor + i * 8;
        float *yi = y + i * 8;

        float r_rev[8];
        r_rev[0] = ri[0];
        r_rev[1] = ri[1];
        r_rev[2] = ri[2];
        r_rev[3] = ri[3];
        r_rev[4] = -ri[4];
        r_rev[5] = -ri[5];
        r_rev[6] = -ri[6];
        r_rev[7] = -ri[7];

        float tmp[8];
        _gp_cl30_single(ri, xi, tmp);
        _gp_cl30_single(tmp, r_rev, yi);
    }
}
