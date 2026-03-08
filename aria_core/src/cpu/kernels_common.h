#ifndef ARIA_KERNELS_COMMON_H
#define ARIA_KERNELS_COMMON_H

#include "kernels.h"
#include "simd_math.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <algorithm>
#include <vector>

#ifdef ARIA_HAS_OPENMP
#include <omp.h>
#define ARIA_OMP_THRESHOLD 16384
#define ARIA_OMP_BATCH_THRESHOLD 4
#endif

#ifdef ARIA_HAS_BLAS
#ifdef ARIA_BLAS_SCIPY_PREFIX
/* scipy-bundled OpenBLAS uses prefixed symbols; provide our own declarations */
enum CBLAS_ORDER { CblasRowMajor = 101 };
enum CBLAS_TRANSPOSE { CblasNoTrans = 111, CblasTrans = 112 };
extern "C" void scipy_cblas_sgemm(enum CBLAS_ORDER, enum CBLAS_TRANSPOSE, enum CBLAS_TRANSPOSE,
                               int, int, int, float, const float *, int,
                               const float *, int, float, float *, int);
#define cblas_sgemm scipy_cblas_sgemm
#else
#include <cblas.h>
#endif
#endif

/* ── Constants ─────────────────────────────────────────────────────── */

static const float GELU_COEFF = 0.7978845608028654f;  /* sqrt(2/pi) */
static const float GELU_CUBIC = 0.044715f;

#endif /* ARIA_KERNELS_COMMON_H */
