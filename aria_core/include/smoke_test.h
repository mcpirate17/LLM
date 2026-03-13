#ifndef ARIA_SMOKE_TEST_H
#define ARIA_SMOKE_TEST_H

#include <cstdint>

struct SmokeTestResult {
    int32_t ok;
    int32_t has_params;
    int32_t grad_flows;
    int32_t no_unsafe;
};

#ifdef __cplusplus
extern "C" {
#endif

SmokeTestResult smoke_test_graph(
    int32_t n_nodes,
    const int32_t *edges,
    const int32_t *op_roles,
    const int32_t *has_params_flag,
    const int32_t *preserves_grad,
    int32_t output_node
);

#ifdef __cplusplus
}
#endif

#endif /* ARIA_SMOKE_TEST_H */
