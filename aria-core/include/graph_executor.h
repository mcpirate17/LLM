#ifndef ARIA_GRAPH_EXECUTOR_H
#define ARIA_GRAPH_EXECUTOR_H

#include <stdint.h>
#include <vector>
#include <string>
#include <map>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    ARIA_OP_RELU,
    ARIA_OP_GELU,
    ARIA_OP_SILU,
    ARIA_OP_ADD,
    ARIA_OP_MUL,
    ARIA_OP_SUB,
    ARIA_OP_RMSNORM,
    ARIA_OP_LAYERNORM,
    ARIA_OP_MATMUL,
    ARIA_OP_LINEAR,
    ARIA_OP_SOFTMAX,
    // Add more as needed
} AriaOpType;

typedef struct {
    AriaOpType type;
    int32_t n_inputs;
    int32_t n_outputs;
    int32_t input_indices[8];  /* index into global tensor pool */
    int32_t output_indices[8];
    float params[8];           /* eps, temperature, etc. */
} AriaExecutableNode;

typedef struct {
    int32_t n_nodes;
    int32_t n_tensors;
    AriaExecutableNode *nodes;
} AriaExecutableGraph;

#ifdef __cplusplus
}
#endif

#endif /* ARIA_GRAPH_EXECUTOR_H */
