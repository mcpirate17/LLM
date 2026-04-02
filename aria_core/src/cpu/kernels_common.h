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

/* Stack allocation threshold: dims above this use heap malloc */
#define ARIA_STACK_ALLOC_THRESHOLD 4096

/* OMP parallelization thresholds for compute-bound loops */
#define ARIA_OMP_COMPUTE_THRESHOLD 256

/* Default numerical epsilon for stability */
#define ARIA_EPSILON_DEFAULT 1e-8f

/* ── Thread-Local Arena Allocator ────────────────────────────────────
 * Replaces per-call malloc/free for temporary buffers that are allocated
 * and freed within the same kernel call. arena_reset() is called at the
 * top of each kernel entry point; all arena_alloc() memory is implicitly
 * freed on the next reset. Falls back to malloc when the arena is full.
 * ──────────────────────────────────────────────────────────────────── */

static constexpr size_t ARENA_ALIGN = 16;
static constexpr size_t MAX_TEMP_BYTES = 4u * 1024u * 1024u; /* 4 MiB */

struct ArenaState {
    alignas(ARENA_ALIGN) char buf[MAX_TEMP_BYTES];
    size_t offset;
};

static thread_local ArenaState tl_arena = {{}, 0};

static inline void arena_reset() {
    tl_arena.offset = 0;
}

static inline void *arena_alloc(size_t n) {
    size_t aligned = (n + ARENA_ALIGN - 1) & ~(ARENA_ALIGN - 1);
    size_t new_offset = tl_arena.offset + aligned;
    if (new_offset <= MAX_TEMP_BYTES) {
        void *ptr = tl_arena.buf + tl_arena.offset;
        tl_arena.offset = new_offset;
        return ptr;
    }
    return malloc(n);
}

static inline void arena_free(void *ptr) {
    char *p = static_cast<char *>(ptr);
    if (p < tl_arena.buf || p >= tl_arena.buf + MAX_TEMP_BYTES) {
        free(ptr);
    }
}

#endif /* ARIA_KERNELS_COMMON_H */
