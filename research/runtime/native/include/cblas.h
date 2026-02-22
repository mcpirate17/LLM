/**
 * cblas.h — Minimal CBLAS compatibility header for OpenBLAS linkage.
 *
 * Only declares cblas_sgemm, which is all we need for matmul/linear.
 * This avoids requiring a system-installed cblas-dev package.
 *
 * Scipy-bundled OpenBLAS prefixes symbols with "scipy_". When
 * ARIA_BLAS_SCIPY_PREFIX is defined, we redirect cblas_sgemm to
 * scipy_cblas_sgemm via a macro.
 */
#ifndef ARIA_CBLAS_COMPAT_H
#define ARIA_CBLAS_COMPAT_H

enum CBLAS_ORDER { CblasRowMajor = 101, CblasColMajor = 102 };
enum CBLAS_TRANSPOSE { CblasNoTrans = 111, CblasTrans = 112, CblasConjTrans = 113 };

#ifdef ARIA_BLAS_SCIPY_PREFIX

/* Scipy-bundled OpenBLAS uses scipy_ prefix on all symbols */
void scipy_cblas_sgemm(enum CBLAS_ORDER Order, enum CBLAS_TRANSPOSE TransA,
                       enum CBLAS_TRANSPOSE TransB, int M, int N, int K,
                       float alpha, const float *A, int lda,
                       const float *B, int ldb,
                       float beta, float *C, int ldc);

#define cblas_sgemm scipy_cblas_sgemm

#else

void cblas_sgemm(enum CBLAS_ORDER Order, enum CBLAS_TRANSPOSE TransA,
                 enum CBLAS_TRANSPOSE TransB, int M, int N, int K,
                 float alpha, const float *A, int lda,
                 const float *B, int ldb,
                 float beta, float *C, int ldc);

#endif /* ARIA_BLAS_SCIPY_PREFIX */

#endif /* ARIA_CBLAS_COMPAT_H */
