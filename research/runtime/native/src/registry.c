#include "registry.h"
#include "kernels.h"
#include "kernels_ext.h"
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

static nk_registration_t g_registry[ARIA_MAX_KERNELS];
static int32_t g_count = 0;
static int g_initialized = 0;

/* --- Forward adapter wrappers --- */
static nk_status_t wrap_relu(const float *x, float *y, int64_t n) { aria_relu_f32(x, y, n); return NK_OK; }
static nk_status_t wrap_gelu(const float *x, float *y, int64_t n) { aria_gelu_f32(x, y, n); return NK_OK; }
static nk_status_t wrap_silu(const float *x, float *y, int64_t n) { aria_silu_f32(x, y, n); return NK_OK; }
static nk_status_t wrap_square(const float *x, float *y, int64_t n) { aria_square_f32(x, y, n); return NK_OK; }
static nk_status_t wrap_abs(const float *x, float *y, int64_t n) { aria_abs_f32(x, y, n); return NK_OK; }
static nk_status_t wrap_neg(const float *x, float *y, int64_t n) { aria_neg_f32(x, y, n); return NK_OK; }
static nk_status_t wrap_reciprocal(const float *x, float *y, int64_t n) { aria_reciprocal_f32(x, y, n); return NK_OK; }
static nk_status_t wrap_log(const float *x, float *y, int64_t n) { aria_log_f32(x, y, n); return NK_OK; }
static nk_status_t wrap_sqrt(const float *x, float *y, int64_t n) { aria_sqrt_f32(x, y, n); return NK_OK; }
static nk_status_t wrap_sin(const float *x, float *y, int64_t n) { aria_sin_f32(x, y, n); return NK_OK; }
static nk_status_t wrap_cos(const float *x, float *y, int64_t n) { aria_cos_f32(x, y, n); return NK_OK; }
static nk_status_t wrap_sigmoid(const float *x, float *y, int64_t n) { aria_sigmoid_f32(x, y, n); return NK_OK; }
static nk_status_t wrap_tanh(const float *x, float *y, int64_t n) { aria_tanh_f32(x, y, n); return NK_OK; }
static nk_status_t wrap_exp(const float *x, float *y, int64_t n) { aria_exp_f32(x, y, n); return NK_OK; }
static nk_status_t wrap_add(const float *a, const float *b, float *y, int64_t n) { aria_add_f32(a, b, y, n); return NK_OK; }
static nk_status_t wrap_mul(const float *a, const float *b, float *y, int64_t n) { aria_mul_f32(a, b, y, n); return NK_OK; }
static nk_status_t wrap_sub(const float *a, const float *b, float *y, int64_t n) { aria_sub_f32(a, b, y, n); return NK_OK; }

static nk_status_t wrap_matmul(const float* A, const float* B, float* C, int64_t M, int64_t K, int64_t N) {
    aria_matmul_f32(A, B, C, M, K, N); return NK_OK;
}
static nk_status_t wrap_linear(const float* x, const float* W, const float* bias, float* y, int64_t b, int64_t in, int64_t out) {
    aria_linear_f32(x, W, bias, y, b, in, out); return NK_OK;
}
static nk_status_t wrap_rmsnorm(const float* x, const float* w, float* y, int64_t b, int64_t d, float eps) {
    aria_rmsnorm_f32(x, w, y, b, d, eps); return NK_OK;
}
static nk_status_t wrap_softmax(const float* x, float* y, int64_t b, int64_t d) {
    aria_softmax_f32(x, y, b, d); return NK_OK;
}

/* --- FP16 adapter wrappers --- */
static nk_status_t wrap_relu_f16(const uint16_t *x, uint16_t *y, int64_t n) { aria_relu_f16(x, y, n); return NK_OK; }
static nk_status_t wrap_gelu_f16(const uint16_t *x, uint16_t *y, int64_t n) { aria_gelu_f16(x, y, n); return NK_OK; }
static nk_status_t wrap_silu_f16(const uint16_t *x, uint16_t *y, int64_t n) { aria_silu_f16(x, y, n); return NK_OK; }
static nk_status_t wrap_sigmoid_f16(const uint16_t *x, uint16_t *y, int64_t n) { aria_sigmoid_f16(x, y, n); return NK_OK; }
static nk_status_t wrap_add_f16(const uint16_t *a, const uint16_t *b, uint16_t *y, int64_t n) { aria_add_f16(a, b, y, n); return NK_OK; }
static nk_status_t wrap_mul_f16(const uint16_t *a, const uint16_t *b, uint16_t *y, int64_t n) { aria_mul_f16(a, b, y, n); return NK_OK; }
static nk_status_t wrap_matmul_f16(const uint16_t *A, const uint16_t *B, uint16_t *C, int64_t M, int64_t K, int64_t N) {
    aria_matmul_f16(A, B, C, M, K, N); return NK_OK;
}
static nk_status_t wrap_softmax_f16(const uint16_t *x, uint16_t *y, int64_t b, int64_t d) {
    aria_softmax_f16(x, y, b, d); return NK_OK;
}
static nk_status_t wrap_rmsnorm_f16(const uint16_t *x, const uint16_t *w, uint16_t *y, int64_t b, int64_t d, float eps) {
    aria_rmsnorm_f16(x, w, y, b, d, eps); return NK_OK;
}

/* --- Fused kernel adapter wrappers --- */
static nk_status_t wrap_matmul_relu(const float *A, const float *B, float *C, int64_t M, int64_t K, int64_t N) {
    aria_matmul_relu_f32(A, B, C, M, K, N); return NK_OK;
}
static nk_status_t wrap_matmul_bias_relu(const float *A, const float *B, const float *bias, float *C, int64_t M, int64_t K, int64_t N) {
    aria_matmul_bias_relu_f32(A, B, bias, C, M, K, N); return NK_OK;
}
static nk_status_t wrap_layernorm_residual(const float *x, const float *residual, const float *gamma, const float *beta, float *y, int64_t rows, int64_t cols, float eps) {
    aria_layernorm_residual_f32(x, residual, gamma, beta, y, rows, cols, eps); return NK_OK;
}
static nk_status_t wrap_matmul_gelu(const float *A, const float *B, float *C, int64_t M, int64_t K, int64_t N) {
    aria_matmul_gelu_f32(A, B, C, M, K, N); return NK_OK;
}

/* --- Backward adapter wrappers --- */
static nk_status_t wrap_relu_bwd(const float *go, const float *fwd, float *gi, int64_t n) {
    aria_relu_backward_f32(go, fwd, gi, n); return NK_OK;
}
static nk_status_t wrap_sigmoid_bwd(const float *go, const float *fwd, float *gi, int64_t n) {
    aria_sigmoid_backward_f32(go, fwd, gi, n); return NK_OK;
}
static nk_status_t wrap_tanh_bwd(const float *go, const float *fwd, float *gi, int64_t n) {
    aria_tanh_backward_f32(go, fwd, gi, n); return NK_OK;
}
static nk_status_t wrap_gelu_bwd(const float *go, const float *fwd, float *gi, int64_t n) {
    aria_gelu_backward_f32(go, fwd, gi, n); return NK_OK;
}
static nk_status_t wrap_silu_bwd(const float *go, const float *fwd, float *gi, int64_t n) {
    aria_silu_backward_f32(go, fwd, gi, n); return NK_OK;
}
static nk_status_t wrap_add_bwd(const float *go, float *ga, float *gb, int64_t n) {
    aria_add_backward_f32(go, ga, gb, n); return NK_OK;
}
static nk_status_t wrap_sub_bwd(const float *go, float *ga, float *gb, int64_t n) {
    aria_sub_backward_f32(go, ga, gb, n); return NK_OK;
}
static nk_status_t wrap_mul_bwd(const float *go, const float *a, const float *b, float *ga, float *gb, int64_t n) {
    aria_mul_backward_f32(go, a, b, ga, gb, n); return NK_OK;
}
static nk_status_t wrap_matmul_bwd(const float *go, const float *A, const float *B, float *gA, float *gB, int64_t M, int64_t K, int64_t N) {
    aria_matmul_backward_f32(go, A, B, gA, gB, M, K, N); return NK_OK;
}

void aria_registry_init(void) {
    if (g_initialized) return;
    g_count = 0;
    memset(g_registry, 0, sizeof(g_registry));

    nk_registration_t r;
    
    #define REG_UNARY(NAME, FN) \
        memset(&r, 0, sizeof(r)); r.op_name = NAME; r.unary_fn = FN; nk_register(&r);
    #define REG_UNARY_WITH_BWD(NAME, FN, BWD) \
        memset(&r, 0, sizeof(r)); r.op_name = NAME; r.unary_fn = FN; r.unary_backward_fn = BWD; nk_register(&r);
    #define REG_BINARY(NAME, FN) \
        memset(&r, 0, sizeof(r)); r.op_name = NAME; r.binary_fn = FN; nk_register(&r);

    /* Unary ops with f32 + f16 + backward */
    memset(&r, 0, sizeof(r)); r.op_name = "relu"; r.unary_fn = wrap_relu; r.unary_f16_fn = wrap_relu_f16; r.unary_backward_fn = wrap_relu_bwd; nk_register(&r);
    memset(&r, 0, sizeof(r)); r.op_name = "gelu"; r.unary_fn = wrap_gelu; r.unary_f16_fn = wrap_gelu_f16; r.unary_backward_fn = wrap_gelu_bwd; nk_register(&r);
    memset(&r, 0, sizeof(r)); r.op_name = "silu"; r.unary_fn = wrap_silu; r.unary_f16_fn = wrap_silu_f16; r.unary_backward_fn = wrap_silu_bwd; nk_register(&r);
    REG_UNARY("square", wrap_square);
    REG_UNARY("abs", wrap_abs);
    REG_UNARY("neg", wrap_neg);
    REG_UNARY("reciprocal", wrap_reciprocal);
    REG_UNARY("log", wrap_log);
    REG_UNARY("sqrt", wrap_sqrt);
    REG_UNARY("sin", wrap_sin);
    REG_UNARY("cos", wrap_cos);
    memset(&r, 0, sizeof(r)); r.op_name = "sigmoid"; r.unary_fn = wrap_sigmoid; r.unary_f16_fn = wrap_sigmoid_f16; r.unary_backward_fn = wrap_sigmoid_bwd; nk_register(&r);
    REG_UNARY_WITH_BWD("tanh", wrap_tanh, wrap_tanh_bwd);
    REG_UNARY("exp", wrap_exp);

    /* Binary ops with f16 */
    memset(&r, 0, sizeof(r)); r.op_name = "add"; r.binary_fn = wrap_add; r.binary_f16_fn = wrap_add_f16; r.binary_backward_simple_fn = wrap_add_bwd; nk_register(&r);
    memset(&r, 0, sizeof(r)); r.op_name = "mul"; r.binary_fn = wrap_mul; r.binary_f16_fn = wrap_mul_f16; r.binary_backward_fn = wrap_mul_bwd; nk_register(&r);
    memset(&r, 0, sizeof(r)); r.op_name = "sub"; r.binary_fn = wrap_sub; r.binary_backward_simple_fn = wrap_sub_bwd; nk_register(&r);

    /* Matmul/linear/norm with f16 */
    memset(&r, 0, sizeof(r)); r.op_name = "matmul"; r.matmul_fn = wrap_matmul; r.matmul_f16_fn = wrap_matmul_f16; r.matmul_backward_fn = wrap_matmul_bwd; nk_register(&r);
    memset(&r, 0, sizeof(r)); r.op_name = "linear"; r.linear_fn = wrap_linear; nk_register(&r);
    memset(&r, 0, sizeof(r)); r.op_name = "rmsnorm"; r.rmsnorm_fn = wrap_rmsnorm; r.rmsnorm_f16_fn = wrap_rmsnorm_f16; nk_register(&r);
    memset(&r, 0, sizeof(r)); r.op_name = "softmax"; r.softmax_fn = wrap_softmax; r.softmax_f16_fn = wrap_softmax_f16; nk_register(&r);

    /* Fused kernels */
    memset(&r, 0, sizeof(r)); r.op_name = "matmul_relu"; r.matmul_relu_fn = wrap_matmul_relu; nk_register(&r);
    memset(&r, 0, sizeof(r)); r.op_name = "matmul_bias_relu"; r.matmul_bias_relu_fn = wrap_matmul_bias_relu; nk_register(&r);
    memset(&r, 0, sizeof(r)); r.op_name = "layernorm_residual"; r.layernorm_residual_fn = wrap_layernorm_residual; nk_register(&r);
    memset(&r, 0, sizeof(r)); r.op_name = "matmul_gelu"; r.matmul_gelu_fn = wrap_matmul_gelu; nk_register(&r);

    g_initialized = 1;
}

nk_status_t nk_register(const nk_registration_t* reg) {
    if (g_count >= ARIA_MAX_KERNELS) return NK_ERR_INTERNAL;
    g_registry[g_count] = *reg;
    g_registry[g_count].op_name = strdup(reg->op_name);
    g_count++;
    return NK_OK;
}

int32_t nk_is_registered(const char* op_name) {
    return nk_dispatch(op_name) != NULL ? 1 : 0;
}

const nk_registration_t* nk_dispatch(const char* op_name) {
    for (int32_t i = 0; i < g_count; i++) {
        if (strcmp(g_registry[i].op_name, op_name) == 0) {
            return &g_registry[i];
        }
    }
    return NULL;
}

/* Backward compatibility for existing C callers */
int aria_registry_is_native(const char *op_name) {
    return nk_is_registered(op_name);
}
int32_t aria_registry_count(void) {
    return g_count;
}
void aria_registry_list(const char **names, int32_t max_count, int32_t *out_count) {
    int32_t n = g_count < max_count ? g_count : max_count;
    for (int32_t i = 0; i < n; i++) {
        names[i] = g_registry[i].op_name;
    }
    *out_count = n;
}

int aria_registry_lookup_unary(const char *op_name, nk_unary_f32_fn *out) {
    const nk_registration_t *reg = nk_dispatch(op_name);
    if (reg && reg->unary_fn) {
        *out = reg->unary_fn;
        return 1;
    }
    return 0;
}

int aria_registry_lookup_binary(const char *op_name, nk_binary_f32_fn *out) {
    const nk_registration_t *reg = nk_dispatch(op_name);
    if (reg && reg->binary_fn) {
        *out = reg->binary_fn;
        return 1;
    }
    return 0;
}
