/**
 * test_kernels.c — Unit tests for C kernels and graph validator.
 *
 * Build: gcc -O2 -o test_kernels test_kernels.c kernels.c graph_validator.c shape_inference.c -lm
 * Run:   ./test_kernels
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "kernels.h"
#include "graph_validator.h"
#include "shape_inference.h"

static int tests_run = 0;
static int tests_passed = 0;

#define ASSERT_NEAR(a, b, tol, msg) do { \
    tests_run++; \
    if (fabsf((a) - (b)) > (tol)) { \
        printf("FAIL: %s (got %.6f, expected %.6f)\n", msg, (float)(a), (float)(b)); \
    } else { tests_passed++; } \
} while(0)

#define ASSERT_EQ(a, b, msg) do { \
    tests_run++; \
    if ((a) != (b)) { \
        printf("FAIL: %s (got %d, expected %d)\n", msg, (int)(a), (int)(b)); \
    } else { tests_passed++; } \
} while(0)

/* ── Kernel tests ──────────────────────────────────────────────────── */

static void test_relu(void) {
    float x[] = {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f};
    float y[5];
    aria_relu_f32(x, y, 5);
    ASSERT_NEAR(y[0], 0.0f, 1e-6, "relu(-2)");
    ASSERT_NEAR(y[1], 0.0f, 1e-6, "relu(-1)");
    ASSERT_NEAR(y[2], 0.0f, 1e-6, "relu(0)");
    ASSERT_NEAR(y[3], 1.0f, 1e-6, "relu(1)");
    ASSERT_NEAR(y[4], 2.0f, 1e-6, "relu(2)");
}

static void test_gelu(void) {
    float x[] = {0.0f, 1.0f, -1.0f};
    float y[3];
    aria_gelu_f32(x, y, 3);
    ASSERT_NEAR(y[0], 0.0f, 1e-4, "gelu(0)");
    ASSERT_NEAR(y[1], 0.8412f, 1e-3, "gelu(1)");
    ASSERT_NEAR(y[2], -0.1588f, 1e-3, "gelu(-1)");
}

static void test_silu(void) {
    float x[] = {0.0f, 1.0f, -1.0f};
    float y[3];
    aria_silu_f32(x, y, 3);
    ASSERT_NEAR(y[0], 0.0f, 1e-6, "silu(0)");
    ASSERT_NEAR(y[1], 0.7311f, 1e-3, "silu(1)");
    ASSERT_NEAR(y[2], -0.2689f, 1e-3, "silu(-1)");
}

static void test_sin_cos(void) {
    float x[] = {0.0f, 1.5708f, 3.14159f};  /* 0, pi/2, pi */
    float ys[3], yc[3];
    aria_sin_f32(x, ys, 3);
    aria_cos_f32(x, yc, 3);
    ASSERT_NEAR(ys[0], 0.0f, 1e-5, "sin(0)");
    ASSERT_NEAR(ys[1], 1.0f, 1e-3, "sin(pi/2)");
    ASSERT_NEAR(yc[0], 1.0f, 1e-5, "cos(0)");
    ASSERT_NEAR(yc[2], -1.0f, 1e-3, "cos(pi)");
}

static void test_add_mul(void) {
    float a[] = {1.0f, 2.0f, 3.0f};
    float b[] = {4.0f, 5.0f, 6.0f};
    float y[3];

    aria_add_f32(a, b, y, 3);
    ASSERT_NEAR(y[0], 5.0f, 1e-6, "add[0]");
    ASSERT_NEAR(y[2], 9.0f, 1e-6, "add[2]");

    aria_mul_f32(a, b, y, 3);
    ASSERT_NEAR(y[0], 4.0f, 1e-6, "mul[0]");
    ASSERT_NEAR(y[2], 18.0f, 1e-6, "mul[2]");
}

static void test_sum_mean(void) {
    float x[] = {1.0f, 2.0f, 3.0f, 4.0f};
    ASSERT_NEAR(aria_sum_f32(x, 4), 10.0f, 1e-5, "sum");
    ASSERT_NEAR(aria_mean_f32(x, 4), 2.5f, 1e-5, "mean");
}

static void test_matmul(void) {
    /* 2x3 @ 3x2 -> 2x2 */
    float A[] = {1,2,3, 4,5,6};
    float B[] = {7,8, 9,10, 11,12};
    float C[4];
    aria_matmul_f32(A, B, C, 2, 3, 2);
    ASSERT_NEAR(C[0], 58.0f, 1e-3, "matmul[0,0]");   /* 1*7+2*9+3*11 */
    ASSERT_NEAR(C[1], 64.0f, 1e-3, "matmul[0,1]");   /* 1*8+2*10+3*12 */
    ASSERT_NEAR(C[2], 139.0f, 1e-3, "matmul[1,0]");   /* 4*7+5*9+6*11 */
    ASSERT_NEAR(C[3], 154.0f, 1e-3, "matmul[1,1]");   /* 4*8+5*10+6*12 */
}

static void test_linear(void) {
    /* batch=2, dim_in=3, dim_out=2 */
    float x[] = {1,2,3, 4,5,6};
    float W[] = {1,0,0, 0,1,0};  /* W[0]=[1,0,0], W[1]=[0,1,0] */
    float bias[] = {10, 20};
    float y[4];
    aria_linear_f32(x, W, bias, y, 2, 3, 2);
    ASSERT_NEAR(y[0], 11.0f, 1e-5, "linear[0,0]");  /* 1*1+2*0+3*0+10 */
    ASSERT_NEAR(y[1], 22.0f, 1e-5, "linear[0,1]");  /* 1*0+2*1+3*0+20 */
    ASSERT_NEAR(y[2], 14.0f, 1e-5, "linear[1,0]");  /* 4*1+5*0+6*0+10 */
    ASSERT_NEAR(y[3], 25.0f, 1e-5, "linear[1,1]");  /* 4*0+5*1+6*0+20 */
}

static void test_rmsnorm(void) {
    float x[] = {1.0f, 2.0f, 3.0f, 4.0f};  /* batch=1, dim=4 */
    float weight[] = {1.0f, 1.0f, 1.0f, 1.0f};
    float y[4];
    aria_rmsnorm_f32(x, weight, y, 1, 4, 1e-6f);
    /* rms = sqrt((1+4+9+16)/4) = sqrt(7.5) ≈ 2.7386 */
    float rms = sqrtf(7.5f);
    ASSERT_NEAR(y[0], 1.0f / rms, 1e-4, "rmsnorm[0]");
    ASSERT_NEAR(y[3], 4.0f / rms, 1e-4, "rmsnorm[3]");
}

static void test_softmax(void) {
    float x[] = {1.0f, 2.0f, 3.0f};
    float y[3];
    aria_softmax_f32(x, y, 1, 3);
    /* Probabilities must sum to 1 */
    float sum = y[0] + y[1] + y[2];
    ASSERT_NEAR(sum, 1.0f, 1e-5, "softmax:sum=1");
    /* Ordering preserved */
    tests_run++;
    if (y[0] < y[1] && y[1] < y[2]) { tests_passed++; }
    else { printf("FAIL: softmax:ordering\n"); }
}

static void test_layernorm(void) {
    float x[] = {1.0f, 2.0f, 3.0f, 4.0f};  /* batch=2, dim=2 */
    float w[] = {1.0f, 1.0f};
    float b[] = {0.0f, 0.0f};
    float y[4];
    aria_layernorm_f32(x, w, b, y, 2, 2, 1e-5f);
    /* Each pair should be normalized to mean ~0 */
    ASSERT_NEAR(y[0] + y[1], 0.0f, 1e-4, "layernorm:mean0");
    ASSERT_NEAR(y[2] + y[3], 0.0f, 1e-4, "layernorm:mean0[1]");
}

static void test_concat_split(void) {
    float a[] = {1.0f, 2.0f};
    float b[] = {3.0f, 4.0f, 5.0f};
    const float *inputs[] = {a, b};
    int64_t sizes[] = {2, 3};
    float output[5];
    aria_concat_f32(inputs, sizes, 2, output);
    ASSERT_NEAR(output[0], 1.0f, 1e-6, "concat[0]");
    ASSERT_NEAR(output[2], 3.0f, 1e-6, "concat[2]");
    ASSERT_NEAR(output[4], 5.0f, 1e-6, "concat[4]");

    /* Split back */
    float out_a[2], out_b[3];
    float *outputs[] = {out_a, out_b};
    aria_split_f32(output, outputs, sizes, 2);
    ASSERT_NEAR(out_a[0], 1.0f, 1e-6, "split[a][0]");
    ASSERT_NEAR(out_a[1], 2.0f, 1e-6, "split[a][1]");
    ASSERT_NEAR(out_b[0], 3.0f, 1e-6, "split[b][0]");
    ASSERT_NEAR(out_b[2], 5.0f, 1e-6, "split[b][2]");
}

static void test_transpose(void) {
    float x[] = {1, 2, 3, 4, 5, 6};  /* 2x3 matrix */
    float y[6];  /* 3x2 result */
    aria_transpose2d_f32(x, y, 2, 3);
    ASSERT_NEAR(y[0], 1.0f, 1e-6, "transpose[0,0]");
    ASSERT_NEAR(y[1], 4.0f, 1e-6, "transpose[0,1]");
    ASSERT_NEAR(y[2], 2.0f, 1e-6, "transpose[1,0]");
    ASSERT_NEAR(y[3], 5.0f, 1e-6, "transpose[1,1]");
    ASSERT_NEAR(y[4], 3.0f, 1e-6, "transpose[2,0]");
    ASSERT_NEAR(y[5], 6.0f, 1e-6, "transpose[2,1]");
}

/* ── Graph validator tests ─────────────────────────────────────────── */

static void test_valid_dag(void) {
    /* Linear chain: 0 -> 1 -> 2 -> 3 */
    AriaGraph g = {0};
    g.n_nodes = 4;
    g.n_edges = 3;
    g.edges[0] = (AriaEdge){0, 1, 0, 0};
    g.edges[1] = (AriaEdge){1, 2, 0, 0};
    g.edges[2] = (AriaEdge){2, 3, 0, 0};

    AriaValidationResult result;
    AriaResult rc = aria_validate_graph(&g, &result);
    ASSERT_EQ(rc, ARIA_OK, "valid_dag:result");
    ASSERT_EQ(result.topo_len, 4, "valid_dag:topo_len");
    ASSERT_EQ(result.topo_order[0], 0, "valid_dag:topo[0]");
    ASSERT_EQ(result.topo_order[3], 3, "valid_dag:topo[3]");
}

static void test_cycle_detection(void) {
    /* Full cycle: 0 -> 1 -> 2 -> 0 (all nodes in cycle, no source)
     * This returns ARIA_ERR_NO_SOURCE because Kahn's can't start */
    AriaGraph g = {0};
    g.n_nodes = 3;
    g.n_edges = 3;
    g.edges[0] = (AriaEdge){0, 1, 0, 0};
    g.edges[1] = (AriaEdge){1, 2, 0, 0};
    g.edges[2] = (AriaEdge){2, 0, 0, 0};

    AriaValidationResult result;
    AriaResult rc = aria_validate_graph(&g, &result);
    ASSERT_EQ(rc, ARIA_ERR_NO_SOURCE, "full_cycle:no_source");

    /* Partial cycle with a source: 0 -> 1 -> 2 -> 1
     * Node 0 is a valid source, but 1<->2 form a cycle */
    AriaGraph g2 = {0};
    g2.n_nodes = 3;
    g2.n_edges = 3;
    g2.edges[0] = (AriaEdge){0, 1, 0, 0};
    g2.edges[1] = (AriaEdge){1, 2, 0, 0};
    g2.edges[2] = (AriaEdge){2, 1, 0, 0};

    AriaResult rc2 = aria_validate_graph(&g2, &result);
    ASSERT_EQ(rc2, ARIA_ERR_CYCLE_DETECTED, "partial_cycle:detected");
}

static void test_self_loop(void) {
    AriaGraph g = {0};
    g.n_nodes = 2;
    g.n_edges = 1;
    g.edges[0] = (AriaEdge){0, 0, 0, 0};

    AriaValidationResult result;
    AriaResult rc = aria_validate_graph(&g, &result);
    ASSERT_EQ(rc, ARIA_ERR_SELF_LOOP, "self_loop:detected");
}

static void test_dangling_edge(void) {
    AriaGraph g = {0};
    g.n_nodes = 2;
    g.n_edges = 1;
    g.edges[0] = (AriaEdge){0, 5, 0, 0};  /* node 5 doesn't exist */

    AriaValidationResult result;
    AriaResult rc = aria_validate_graph(&g, &result);
    ASSERT_EQ(rc, ARIA_ERR_DANGLING_EDGE, "dangling:detected");
}

static void test_branching_dag(void) {
    /*     0
     *    / \
     *   1   2
     *    \ /
     *     3
     */
    AriaGraph g = {0};
    g.n_nodes = 4;
    g.n_edges = 4;
    g.edges[0] = (AriaEdge){0, 1, 0, 0};
    g.edges[1] = (AriaEdge){0, 2, 0, 0};
    g.edges[2] = (AriaEdge){1, 3, 0, 0};
    g.edges[3] = (AriaEdge){2, 3, 0, 0};

    AriaValidationResult result;
    AriaResult rc = aria_validate_graph(&g, &result);
    ASSERT_EQ(rc, ARIA_OK, "branch_dag:valid");
    ASSERT_EQ(result.topo_len, 4, "branch_dag:topo_len");
    /* Node 0 must come first, node 3 must come last */
    ASSERT_EQ(result.topo_order[0], 0, "branch_dag:first");
    ASSERT_EQ(result.topo_order[3], 3, "branch_dag:last");
}

static void test_sources_sinks(void) {
    /* 0 -> 1 -> 3, 2 -> 3 */
    AriaGraph g = {0};
    g.n_nodes = 4;
    g.n_edges = 3;
    g.edges[0] = (AriaEdge){0, 1, 0, 0};
    g.edges[1] = (AriaEdge){1, 3, 0, 0};
    g.edges[2] = (AriaEdge){2, 3, 0, 0};

    AriaValidationResult result;
    aria_validate_graph(&g, &result);

    int32_t sources[4], sinks[4];
    int32_t n_src = aria_find_sources(&result, 4, sources, 4);
    int32_t n_snk = aria_find_sinks(&result, 4, sinks, 4);

    ASSERT_EQ(n_src, 2, "sources:count");  /* nodes 0, 2 */
    ASSERT_EQ(n_snk, 1, "sinks:count");    /* node 3 */
    ASSERT_EQ(sinks[0], 3, "sinks[0]");
}

/* ── Shape inference tests ─────────────────────────────────────────── */

static void test_shape_identity(void) {
    NodeShapeSpec spec = {0};
    spec.rule = SHAPE_IDENTITY;
    spec.n_inputs = 1;
    spec.n_outputs = 1;
    spec.input_shapes[0].shape = (TensorShape){{2, 64, 256}, 3, 1};

    int rc = aria_apply_shape_rule(&spec);
    ASSERT_EQ(rc, 0, "shape_identity:ok");
    ASSERT_EQ(spec.output_shapes[0].shape.dims[0], 2, "shape_identity:B");
    ASSERT_EQ(spec.output_shapes[0].shape.dims[1], 64, "shape_identity:S");
    ASSERT_EQ(spec.output_shapes[0].shape.dims[2], 256, "shape_identity:D");
}

static void test_shape_reduce_last(void) {
    NodeShapeSpec spec = {0};
    spec.rule = SHAPE_REDUCE_LAST;
    spec.n_inputs = 1;
    spec.n_outputs = 1;
    spec.input_shapes[0].shape = (TensorShape){{2, 64, 256}, 3, 1};

    int rc = aria_apply_shape_rule(&spec);
    ASSERT_EQ(rc, 0, "shape_reduce_last:ok");
    ASSERT_EQ(spec.output_shapes[0].shape.dims[2], 1, "shape_reduce_last:D=1");
}

static void test_shape_matmul(void) {
    NodeShapeSpec spec = {0};
    spec.rule = SHAPE_MATMUL;
    spec.n_inputs = 2;
    spec.n_outputs = 1;
    spec.input_shapes[0].shape = (TensorShape){{2, 64, 256}, 3, 1};
    spec.input_shapes[1].shape = (TensorShape){{2, 256, 128}, 3, 1};

    int rc = aria_apply_shape_rule(&spec);
    ASSERT_EQ(rc, 0, "shape_matmul:ok");
    ASSERT_EQ(spec.output_shapes[0].shape.dims[0], 2, "shape_matmul:B");
    ASSERT_EQ(spec.output_shapes[0].shape.dims[1], 64, "shape_matmul:S");
    ASSERT_EQ(spec.output_shapes[0].shape.dims[2], 128, "shape_matmul:K");
}

static void test_shape_split(void) {
    NodeShapeSpec spec = {0};
    spec.rule = SHAPE_SPLIT;
    spec.n_inputs = 1;
    spec.n_outputs = 2;
    spec.split_n = 2;
    spec.input_shapes[0].shape = (TensorShape){{2, 64, 256}, 3, 1};

    int rc = aria_apply_shape_rule(&spec);
    ASSERT_EQ(rc, 0, "shape_split:ok");
    ASSERT_EQ(spec.output_shapes[0].shape.dims[2], 128, "shape_split:D/2");
    ASSERT_EQ(spec.output_shapes[1].shape.dims[2], 128, "shape_split:D/2[1]");
}

static void test_shape_linear(void) {
    NodeShapeSpec spec = {0};
    spec.rule = SHAPE_LINEAR;
    spec.n_inputs = 1;
    spec.n_outputs = 1;
    spec.out_dim = 512;
    spec.input_shapes[0].shape = (TensorShape){{2, 64, 256}, 3, 1};

    int rc = aria_apply_shape_rule(&spec);
    ASSERT_EQ(rc, 0, "shape_linear:ok");
    ASSERT_EQ(spec.output_shapes[0].shape.dims[2], 512, "shape_linear:D_out");
}

static void test_shape_rfft(void) {
    NodeShapeSpec spec = {0};
    spec.rule = SHAPE_RFFT;
    spec.n_inputs = 1;
    spec.n_outputs = 1;
    spec.input_shapes[0].shape = (TensorShape){{2, 64, 256}, 3, 1};

    int rc = aria_apply_shape_rule(&spec);
    ASSERT_EQ(rc, 0, "shape_rfft:ok");
    ASSERT_EQ(spec.output_shapes[0].shape.dims[1], 33, "shape_rfft:S/2+1");
}

/* ── Main ──────────────────────────────────────────────────────────── */

int main(void) {
    printf("Running aria-designer runtime tests...\n\n");

    /* Kernel tests */
    test_relu();
    test_gelu();
    test_silu();
    test_sin_cos();
    test_add_mul();
    test_sum_mean();
    test_matmul();
    test_linear();
    test_rmsnorm();
    test_softmax();
    test_layernorm();
    test_concat_split();
    test_transpose();

    /* Graph validator tests */
    test_valid_dag();
    test_cycle_detection();
    test_self_loop();
    test_dangling_edge();
    test_branching_dag();
    test_sources_sinks();

    /* Shape inference tests */
    test_shape_identity();
    test_shape_reduce_last();
    test_shape_matmul();
    test_shape_split();
    test_shape_linear();
    test_shape_rfft();

    printf("\n%d/%d tests passed\n", tests_passed, tests_run);
    return tests_passed == tests_run ? 0 : 1;
}
