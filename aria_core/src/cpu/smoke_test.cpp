/**
 * smoke_test.cpp — Fast structural smoke test for computation graphs.
 *
 * Validates graph structure without tensor allocation:
 *   1. Gradient flow: exists path from input→output where all ops preserve grad
 *   2. Has parameters: at least one op has learnable weights
 *   3. No unsafe ops: no standalone gradient-killing ops in the graph
 *   4. Connected: output is reachable from input
 *
 * ~0.01ms per graph. Exposed via pybind11.
 */

#include <cstdint>
#include <cstring>
#include <vector>

/* Result struct packed as 4 int32s for easy Python interop */
struct SmokeTestResult {
    int32_t ok;           /* 1 if all checks pass */
    int32_t has_params;   /* 1 if graph has learnable parameters */
    int32_t grad_flows;   /* 1 if gradient can flow input→output */
    int32_t no_unsafe;    /* 1 if no unsafe ops are standalone */
};

/**
 * smoke_test_graph — Structural validation of a computation graph.
 *
 * @param n_nodes       Number of nodes (including input node at index 0)
 * @param edges         Flattened (n_nodes × 2) array of input indices per node.
 *                      -1 means no input. Node 0 is the input node.
 * @param op_roles      Role code per node: 0=project, 1=normalize, 2=activate,
 *                      3=mix, 4=route, 5=gate, 6=position, 7=reduce,
 *                      8=residual, 9=unsafe, 10=input
 * @param has_params_flag  1 per node if the op has learnable parameters, 0 otherwise
 * @param preserves_grad   1 per node if the op preserves gradient, 0 otherwise
 * @param output_node   Index of the output node
 */
extern "C" SmokeTestResult smoke_test_graph(
    int32_t n_nodes,
    const int32_t *edges,       /* n_nodes * 2 */
    const int32_t *op_roles,    /* n_nodes */
    const int32_t *has_params_flag, /* n_nodes */
    const int32_t *preserves_grad,  /* n_nodes */
    int32_t output_node
) {
    SmokeTestResult result;
    result.ok = 0;
    result.has_params = 0;
    result.grad_flows = 0;
    result.no_unsafe = 1;

    if (n_nodes < 2 || output_node < 0 || output_node >= n_nodes) {
        return result;
    }

    /* Check has_params: any node with learnable weights? */
    for (int32_t i = 0; i < n_nodes; i++) {
        if (has_params_flag[i]) {
            result.has_params = 1;
            break;
        }
    }

    /* Check no_unsafe: any standalone unsafe op (role == 9)? */
    for (int32_t i = 1; i < n_nodes; i++) {  /* skip input node 0 */
        if (op_roles[i] == 9) {
            result.no_unsafe = 0;
            break;
        }
    }

    /*
     * Gradient flow check: BFS backward from output to input.
     * Only traverse edges where the node preserves gradient.
     * If we can reach node 0 (input), gradient flows.
     */
    std::vector<uint8_t> visited(n_nodes, 0);
    std::vector<int32_t> queue;
    queue.reserve(n_nodes);

    /* Start from output node */
    if (preserves_grad[output_node]) {
        visited[output_node] = 1;
        queue.push_back(output_node);
    }

    size_t head = 0;
    while (head < queue.size()) {
        int32_t node = queue[head++];
        if (node == 0) {
            result.grad_flows = 1;
            break;
        }
        /* Walk backward through edges */
        int32_t in0 = edges[node * 2];
        int32_t in1 = edges[node * 2 + 1];
        if (in0 >= 0 && in0 < n_nodes && !visited[in0] && preserves_grad[in0]) {
            visited[in0] = 1;
            queue.push_back(in0);
        }
        if (in1 >= 0 && in1 < n_nodes && !visited[in1] && preserves_grad[in1]) {
            visited[in1] = 1;
            queue.push_back(in1);
        }
    }

    /* Also check basic connectivity (ignoring grad preservation) */
    if (!result.grad_flows) {
        /* Try again without grad constraint — is graph at least connected? */
        memset(visited.data(), 0, n_nodes);
        queue.clear();
        visited[output_node] = 1;
        queue.push_back(output_node);
        head = 0;
        while (head < queue.size()) {
            int32_t node = queue[head++];
            if (node == 0) break;
            int32_t in0 = edges[node * 2];
            int32_t in1 = edges[node * 2 + 1];
            if (in0 >= 0 && in0 < n_nodes && !visited[in0]) {
                visited[in0] = 1;
                queue.push_back(in0);
            }
            if (in1 >= 0 && in1 < n_nodes && !visited[in1]) {
                visited[in1] = 1;
                queue.push_back(in1);
            }
        }
        /* If not even connected, fail hard */
        if (!visited[0]) {
            return result;
        }
    }

    /* Overall pass: has params AND gradient flows AND no unsafe */
    result.ok = result.has_params && result.grad_flows && result.no_unsafe;
    return result;
}
