#include "../src/graph_validator.h"
#include "../src/shape_inference.h"
#include <stdio.h>
#include <assert.h>
#include <string.h>

void test_validator_cycle() {
    printf("Testing cycle detection... ");
    AriaGraph g = {0};
    g.n_nodes = 3;
    g.n_edges = 3;

    /* 0 -> 1 -> 2 -> 0 (Pure cycle, no source) */
    g.edges[0] = (AriaEdge){0, 1, 0, 0};
    g.edges[1] = (AriaEdge){1, 2, 0, 0};
    g.edges[2] = (AriaEdge){2, 0, 0, 0};

    AriaValidationResult res;
    AriaResult rc = aria_validate_graph(&g, &res);

    assert(rc == ARIA_ERR_NO_SOURCE || rc == ARIA_ERR_CYCLE_DETECTED);

    /* 0 -> 1 -> 2 -> 1 (Cycle with source 0) */
    g.n_nodes = 3;
    g.n_edges = 3;
    g.edges[0] = (AriaEdge){0, 1, 0, 0};
    g.edges[1] = (AriaEdge){1, 2, 0, 0};
    g.edges[2] = (AriaEdge){2, 1, 0, 0};

    rc = aria_validate_graph(&g, &res);
    assert(rc == ARIA_ERR_CYCLE_DETECTED);

    printf("OK\n");
}

void test_validator_dag() {
    printf("Testing DAG validation... ");
    AriaGraph g = {0};
    g.n_nodes = 4;
    g.n_edges = 4;

    /* 0 -> 1, 0 -> 2, 1 -> 3, 2 -> 3 */
    g.edges[0] = (AriaEdge){0, 1, 0, 0};
    g.edges[1] = (AriaEdge){0, 2, 0, 0};
    g.edges[2] = (AriaEdge){1, 3, 0, 0};
    g.edges[3] = (AriaEdge){2, 3, 0, 0};

    AriaValidationResult res;
    AriaResult rc = aria_validate_graph(&g, &res);

    assert(rc == ARIA_OK);
    assert(res.topo_len == 4);
    assert(res.topo_order[0] == 0);
    assert(res.topo_order[3] == 3);
    printf("OK\n");
}

void test_shape_inference() {
    printf("Testing shape inference... ");
    ShapeInferenceResult res = {0};
    res.n_nodes = 2;

    /* Node 0: Input (Identity) */
    res.nodes[0].rule = SHAPE_IDENTITY;
    res.nodes[0].n_inputs = 1;
    res.nodes[0].n_outputs = 1;
    res.nodes[0].input_shapes[0].shape.ndim = 3;
    res.nodes[0].input_shapes[0].shape.dims[0] = 1;  /* B */
    res.nodes[0].input_shapes[0].shape.dims[1] = 16; /* S */
    res.nodes[0].input_shapes[0].shape.dims[2] = 32; /* D */
    res.nodes[0].input_shapes[0].shape.valid = 1;

    /* Node 1: Linear (B,S,32) -> (B,S,64) */
    res.nodes[1].rule = SHAPE_LINEAR;
    res.nodes[1].n_inputs = 1;
    res.nodes[1].n_outputs = 1;
    res.nodes[1].out_dim = 64;

    int32_t topo[] = {0, 1};
    int32_t edges[][4] = {{0, 1, 0, 0}};

    int rc = aria_propagate_shapes(&res, topo, 2, edges, 1);
    assert(rc == 0);
    assert(res.valid == 1);
    assert(res.nodes[1].output_shapes[0].shape.dims[2] == 64);
    assert(res.nodes[1].output_shapes[0].shape.dims[1] == 16);
    assert(res.nodes[1].output_shapes[0].shape.dims[0] == 1);
    printf("OK\n");
}

int main() {
    test_validator_cycle();
    test_validator_dag();
    test_shape_inference();
    printf("All C tests passed!\n");
    return 0;
}
