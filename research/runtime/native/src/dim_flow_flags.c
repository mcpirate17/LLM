#include "../include/graph_analysis.h"

#include <stddef.h>
#include <stdint.h>

int32_t aria_graph_build_dim_flow_flags(
    int32_t n_nodes,
    const int32_t* op_codes,
    const int64_t* param_estimates,
    const int32_t* opcode_has_params,
    const int32_t* opcode_nontrivial,
    const int32_t* opcode_kv_breaking,
    const int32_t* opcode_kind,
    const int32_t* opcode_full_dim,
    int32_t* has_params_flags,
    int32_t* nontrivial_flags,
    int32_t* kv_breaking_flags,
    int32_t* op_kind_flags,
    int32_t* full_dim_flags) {
  int32_t idx;

  if (n_nodes < 0 || op_codes == NULL || param_estimates == NULL ||
      opcode_has_params == NULL || opcode_nontrivial == NULL ||
      opcode_kv_breaking == NULL || opcode_kind == NULL ||
      opcode_full_dim == NULL || has_params_flags == NULL ||
      nontrivial_flags == NULL || kv_breaking_flags == NULL ||
      op_kind_flags == NULL || full_dim_flags == NULL) {
    return -1;
  }

  for (idx = 0; idx < n_nodes; ++idx) {
    int32_t opcode = op_codes[idx];
    int32_t has_params = 0;
    if (opcode < 0) {
      return -1;
    }

    has_params = opcode_has_params[opcode];
    has_params_flags[idx] = (has_params != 0 && param_estimates[idx] > 0) ? 1 : 0;
    nontrivial_flags[idx] = opcode_nontrivial[opcode];
    kv_breaking_flags[idx] = opcode_kv_breaking[opcode];
    op_kind_flags[idx] = opcode_kind[opcode];
    full_dim_flags[idx] = opcode_full_dim[opcode];
  }

  return 0;
}
