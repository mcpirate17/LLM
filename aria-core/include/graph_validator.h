/**
 * graph_validator.h — Fast DAG validation, topological sort, cycle detection.
 *
 * Designed for high throughput: validates hundreds of graphs/sec during
 * architecture search. All memory is stack/arena allocated — no malloc
 * in the hot path.
 */
#ifndef ARIA_GRAPH_VALIDATOR_H
#define ARIA_GRAPH_VALIDATOR_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ── Limits ────────────────────────────────────────────────────────── */

#define ARIA_MAX_NODES 1024
#define ARIA_MAX_EDGES 4096
#define ARIA_MAX_ERROR_LEN 512

/* ── Data structures ───────────────────────────────────────────────── */

typedef struct {
    int32_t source;    /* source node index */
    int32_t target;    /* target node index */
    int32_t src_port;  /* source port index */
    int32_t tgt_port;  /* target port index */
} AriaEdge;

typedef struct {
    int32_t n_nodes;
    int32_t n_edges;
    AriaEdge edges[ARIA_MAX_EDGES];
} AriaGraph;

/* ── Result codes ──────────────────────────────────────────────────── */

typedef enum {
    ARIA_OK = 0,
    ARIA_ERR_TOO_MANY_NODES = -1,
    ARIA_ERR_TOO_MANY_EDGES = -2,
    ARIA_ERR_CYCLE_DETECTED  = -3,
    ARIA_ERR_DANGLING_EDGE   = -4,
    ARIA_ERR_DUPLICATE_EDGE  = -5,
    ARIA_ERR_SELF_LOOP       = -6,
    ARIA_ERR_NO_SOURCE       = -7,   /* graph has no root/source node */
    ARIA_ERR_DISCONNECTED    = -8,
} AriaResult;

/* ── Validation result ─────────────────────────────────────────────── */

typedef struct {
    AriaResult code;
    char       error[ARIA_MAX_ERROR_LEN];
    int32_t    topo_order[ARIA_MAX_NODES];  /* topological sort result */
    int32_t    topo_len;                    /* number of nodes in topo order */
    int32_t    in_degree[ARIA_MAX_NODES];   /* in-degree per node */
    int32_t    out_degree[ARIA_MAX_NODES];  /* out-degree per node */
} AriaValidationResult;

/* ── API ───────────────────────────────────────────────────────────── */

/**
 * Validate a directed graph: check for cycles, dangling edges, self-loops.
 * Produces a topological ordering if valid.
 *
 * Returns ARIA_OK on success. On failure, result->code and result->error
 * describe the issue.
 */
AriaResult aria_validate_graph(const AriaGraph *graph, AriaValidationResult *result);

/**
 * Compute in-degree and out-degree for each node.
 * Useful for identifying source/sink nodes.
 */
void aria_compute_degrees(const AriaGraph *graph, AriaValidationResult *result);

/**
 * Find source nodes (in-degree == 0). Returns count.
 * source_nodes[] is filled with node indices.
 */
int32_t aria_find_sources(const AriaValidationResult *result, int32_t n_nodes,
                          int32_t source_nodes[], int32_t max_sources);

/**
 * Find sink nodes (out-degree == 0). Returns count.
 */
int32_t aria_find_sinks(const AriaValidationResult *result, int32_t n_nodes,
                        int32_t sink_nodes[], int32_t max_sinks);

#ifdef __cplusplus
}
#endif

#endif /* ARIA_GRAPH_VALIDATOR_H */
