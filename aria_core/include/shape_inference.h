/**
 * shape_inference.h — Shape propagation through computation graphs.
 *
 * Propagates symbolic tensor shapes through graph edges based on
 * component shape rules. Used for validation and memory estimation.
 */
#ifndef ARIA_SHAPE_INFERENCE_H
#define ARIA_SHAPE_INFERENCE_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define ARIA_MAX_DIMS 8
#define ARIA_MAX_PORTS 8
#define ARIA_SHAPE_MAX_NODES 1024

/* ── Shape rule enum ───────────────────────────────────────────────── */

typedef enum {
    SHAPE_IDENTITY = 0,      /* output = input shape */
    SHAPE_BINARY_BROADCAST,  /* broadcast two inputs */
    SHAPE_REDUCE_LAST,       /* (B,S,D) -> (B,S,1) */
    SHAPE_REDUCE_SEQ,        /* (B,S,D) -> (B,1,D) */
    SHAPE_MATMUL,            /* (B,S,D) x (B,D,K) -> (B,S,K) */
    SHAPE_OUTER,             /* elementwise product, same shape */
    SHAPE_TRANSPOSE_SD,      /* (B,S,D) -> (B,D,S) */
    SHAPE_SPLIT,             /* (B,S,D) -> N x (B,S,D/N) */
    SHAPE_CONCAT,            /* N x (B,S,D_i) -> (B,S,sum(D_i)) */
    SHAPE_LINEAR,            /* (B,S,D_in) -> (B,S,D_out) */
    SHAPE_RFFT,              /* (B,S,D) -> (B,S/2+1,D) */
    SHAPE_IRFFT,             /* (B,S/2+1,D) -> (B,S,D) */
    SHAPE_CUMULATIVE,        /* shape unchanged */
    SHAPE_SOFTMAX,           /* shape unchanged */
    SHAPE_CAUSAL_MASK,       /* shape unchanged */
    SHAPE_SCALE,             /* shape unchanged */
    SHAPE_BIAS,              /* shape unchanged */
    SHAPE_ROLL,              /* shape unchanged */
    SHAPE_GATHER,            /* shape unchanged */
    SHAPE_SCATTER,           /* shape unchanged */
    SHAPE_SORT,              /* output: sorted + indices */
    SHAPE_UNSORT,            /* shape unchanged */
    SHAPE_RULE_COUNT
} ShapeRule;

/* ── Tensor shape ──────────────────────────────────────────────────── */

typedef struct {
    int32_t dims[ARIA_MAX_DIMS]; /* -1 = symbolic/unknown */
    int32_t ndim;
    int32_t valid;               /* 0 = not yet inferred, 1 = valid */
} TensorShape;

/* ── Port shape spec ───────────────────────────────────────────────── */

typedef struct {
    TensorShape shape;
    int32_t     port_index;
} PortShape;

/* ── Node shape spec ───────────────────────────────────────────────── */

typedef struct {
    ShapeRule  rule;
    int32_t    n_inputs;
    int32_t    n_outputs;
    int32_t    split_n;          /* for SHAPE_SPLIT: number of splits */
    int32_t    out_dim;          /* for SHAPE_LINEAR: output dimension (-1 = same) */
    int32_t    orig_seq_len;     /* for SHAPE_IRFFT: original sequence length */
    PortShape  input_shapes[ARIA_MAX_PORTS];
    PortShape  output_shapes[ARIA_MAX_PORTS];
} NodeShapeSpec;

/* ── Inference result ──────────────────────────────────────────────── */

typedef struct {
    int32_t       valid;
    char          error[512];
    NodeShapeSpec nodes[ARIA_SHAPE_MAX_NODES];
    int32_t       n_nodes;
} ShapeInferenceResult;

/* ── API ───────────────────────────────────────────────────────────── */

/**
 * Apply a shape rule to infer output shapes from input shapes.
 * Returns 0 on success, -1 on shape mismatch.
 */
int aria_apply_shape_rule(NodeShapeSpec *spec);

/**
 * Propagate shapes through a graph in topological order.
 * topo_order: node indices in topological order (from graph_validator).
 * edges: edge list (source_node, target_node, src_port, tgt_port).
 *
 * Returns 0 on success.
 */
int aria_propagate_shapes(ShapeInferenceResult *result,
                          const int32_t *topo_order, int32_t topo_len,
                          const int32_t edges[][4], int32_t n_edges);

#ifdef __cplusplus
}
#endif

#endif /* ARIA_SHAPE_INFERENCE_H */
