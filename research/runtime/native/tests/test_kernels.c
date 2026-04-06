/**
 * test_kernels.c — Correctness tests for native kernels.
 *
 * Uses simple assert-based testing. Run via: cd build && ctest
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#ifdef NDEBUG
#undef NDEBUG
#endif
#include <assert.h>
#include <string.h>
#include "kernels.h"  /* from aria_designer/runtime/src/ via CMake include path */
#include "../src/kernels_ext.h"
#include "../src/registry.h"

#define TOLERANCE 1e-5f
#define ASSERT_NEAR(a, b) assert(fabsf((a) - (b)) < TOLERANCE)

static void test_relu(void) {
    float x[] = {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f};
    float y[5];
    aria_relu_f32(x, y, 5);
    ASSERT_NEAR(y[0], 0.0f);
    ASSERT_NEAR(y[1], 0.0f);
    ASSERT_NEAR(y[2], 0.0f);
    ASSERT_NEAR(y[3], 1.0f);
    ASSERT_NEAR(y[4], 2.0f);
    printf("  PASS: relu\n");
}

static void test_gelu(void) {
    float x[] = {0.0f, 1.0f, -1.0f};
    float y[3];
    aria_gelu_f32(x, y, 3);
    ASSERT_NEAR(y[0], 0.0f);
    assert(y[1] > 0.8f && y[1] < 0.85f);  /* GELU(1) ≈ 0.8412 */
    assert(y[2] > -0.17f && y[2] < -0.15f);  /* GELU(-1) ≈ -0.1588 */
    printf("  PASS: gelu\n");
}

static void test_silu(void) {
    float x[] = {0.0f, 1.0f, -1.0f};
    float y[3];
    aria_silu_f32(x, y, 3);
    ASSERT_NEAR(y[0], 0.0f);
    assert(y[1] > 0.73f && y[1] < 0.74f);  /* SiLU(1) ≈ 0.7311 */
    printf("  PASS: silu\n");
}

static void test_add_mul_sub(void) {
    float a[] = {1.0f, 2.0f, 3.0f};
    float b[] = {4.0f, 5.0f, 6.0f};
    float y[3];

    aria_add_f32(a, b, y, 3);
    ASSERT_NEAR(y[0], 5.0f); ASSERT_NEAR(y[1], 7.0f); ASSERT_NEAR(y[2], 9.0f);

    aria_mul_f32(a, b, y, 3);
    ASSERT_NEAR(y[0], 4.0f); ASSERT_NEAR(y[1], 10.0f); ASSERT_NEAR(y[2], 18.0f);

    aria_sub_f32(a, b, y, 3);
    ASSERT_NEAR(y[0], -3.0f); ASSERT_NEAR(y[1], -3.0f); ASSERT_NEAR(y[2], -3.0f);

    printf("  PASS: add/mul/sub\n");
}

static void test_matmul(void) {
    /* A[2,3] @ B[3,2] = C[2,2] */
    float A[] = {1, 2, 3, 4, 5, 6};
    float B[] = {7, 8, 9, 10, 11, 12};
    float C[4];
    aria_matmul_f32(A, B, C, 2, 3, 2);
    ASSERT_NEAR(C[0], 58.0f);   /* 1*7 + 2*9 + 3*11 */
    ASSERT_NEAR(C[1], 64.0f);   /* 1*8 + 2*10 + 3*12 */
    ASSERT_NEAR(C[2], 139.0f);  /* 4*7 + 5*9 + 6*11 */
    ASSERT_NEAR(C[3], 154.0f);  /* 4*8 + 5*10 + 6*12 */
    printf("  PASS: matmul\n");
}

static void test_rmsnorm(void) {
    float x[] = {1.0f, 2.0f, 3.0f, 4.0f};  /* batch=2, dim=2 */
    float w[] = {1.0f, 1.0f};
    float y[4];
    aria_rmsnorm_f32(x, w, y, 2, 2, 1e-5f);
    /* RMS of [1,2] = sqrt((1+4)/2) = sqrt(2.5) ≈ 1.5811 */
    float rms = sqrtf(2.5f);
    ASSERT_NEAR(y[0], 1.0f / rms);
    ASSERT_NEAR(y[1], 2.0f / rms);
    printf("  PASS: rmsnorm\n");
}

static void test_softmax(void) {
    float x[] = {1.0f, 2.0f, 3.0f};
    float y[3];
    aria_softmax_f32(x, y, 1, 3);
    /* Check probabilities sum to 1 */
    float sum = y[0] + y[1] + y[2];
    ASSERT_NEAR(sum, 1.0f);
    /* Check ordering preserved */
    assert(y[0] < y[1] && y[1] < y[2]);
    printf("  PASS: softmax\n");
}

static void test_layernorm(void) {
    float x[] = {1.0f, 2.0f, 3.0f, 4.0f};  /* batch=2, dim=2 */
    float w[] = {1.0f, 1.0f};
    float b[] = {0.0f, 0.0f};
    float y[4];
    aria_layernorm_f32(x, w, b, y, 2, 2, 1e-5f);
    /* Normalized: each pair should have mean≈0, var≈1 */
    ASSERT_NEAR(y[0] + y[1], 0.0f);
    printf("  PASS: layernorm\n");
}

static void test_transpose(void) {
    float x[] = {1, 2, 3, 4, 5, 6};  /* 2x3 */
    float y[6];  /* 3x2 */
    aria_transpose2d_f32(x, y, 2, 3);
    ASSERT_NEAR(y[0], 1.0f); ASSERT_NEAR(y[1], 4.0f);
    ASSERT_NEAR(y[2], 2.0f); ASSERT_NEAR(y[3], 5.0f);
    ASSERT_NEAR(y[4], 3.0f); ASSERT_NEAR(y[5], 6.0f);
    printf("  PASS: transpose\n");
}

static void test_registry(void) {
    aria_registry_init();
    assert(aria_registry_count() >= 9);
    assert(aria_registry_is_native("relu") == 1);
    assert(aria_registry_is_native("gelu") == 1);
    assert(aria_registry_is_native("nonexistent_op") == 0);

    nk_unary_f32_fn fn = NULL;
    assert(aria_registry_lookup_unary("relu", &fn) == 1);
    assert(fn != NULL);

    /* Test dispatch through registry */
    float x[] = {-1.0f, 0.0f, 1.0f};
    float y[3];
    nk_status_t status = fn(x, y, 3);
    assert(status == NK_OK);
    ASSERT_NEAR(y[0], 0.0f);
    ASSERT_NEAR(y[2], 1.0f);

    printf("  PASS: registry\n");
}

int main(void) {
    printf("Running native kernel tests...\n");
    test_relu();
    test_gelu();
    test_silu();
    test_add_mul_sub();
    test_matmul();
    test_rmsnorm();
    test_softmax();
    test_layernorm();
    test_transpose();
    test_registry();
    printf("\nAll %d tests passed.\n", 10);
    return 0;
}
