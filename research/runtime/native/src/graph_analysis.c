#include "../include/graph_analysis.h"

#include <stdlib.h>
#include <string.h>

static void _zero_result(aria_graph_analysis_result_t* out) {
  if (out == NULL) {
    return;
  }
  out->has_gradient_path = 0;
  out->reachable_count = 0;
  out->depth = 0;
  out->has_cycle = 0;
  out->param_estimate = 0;
}

int32_t aria_graph_analyze_ir(
    int32_t n_nodes,
    const int32_t* op_codes,
    const int32_t* input_indices,
    int32_t output_node_idx,
    const int64_t* param_estimates,
    aria_graph_analysis_result_t* out,
    int32_t* reachable_mask) {
  int32_t i;
  int32_t edge_count = 0;
  int32_t visited_count = 0;
  int32_t reachable_count = 0;
  int64_t param_total = 0;
  int32_t max_depth = 0;
  int32_t* child_counts = NULL;
  int32_t* child_offsets = NULL;
  int32_t* child_cursor = NULL;
  int32_t* children = NULL;
  int32_t* indegree = NULL;
  int32_t* topo_depth = NULL;
  int32_t* queue = NULL;
  int32_t* stack = NULL;
  int8_t* seen = NULL;

  if (out == NULL || n_nodes < 0 || op_codes == NULL || input_indices == NULL) {
    return -1;
  }
  _zero_result(out);

  if (n_nodes == 0) {
    return 0;
  }

  child_counts = (int32_t*)calloc((size_t)n_nodes, sizeof(int32_t));
  child_offsets = (int32_t*)calloc((size_t)n_nodes + 1U, sizeof(int32_t));
  child_cursor = (int32_t*)calloc((size_t)n_nodes, sizeof(int32_t));
  indegree = (int32_t*)calloc((size_t)n_nodes, sizeof(int32_t));
  topo_depth = (int32_t*)calloc((size_t)n_nodes, sizeof(int32_t));
  queue = (int32_t*)malloc((size_t)n_nodes * sizeof(int32_t));
  stack = (int32_t*)malloc((size_t)n_nodes * sizeof(int32_t));
  seen = (int8_t*)calloc((size_t)n_nodes, sizeof(int8_t));

  if (child_counts == NULL || child_offsets == NULL || child_cursor == NULL ||
      indegree == NULL || topo_depth == NULL || queue == NULL || stack == NULL ||
      seen == NULL) {
    free(child_counts);
    free(child_offsets);
    free(child_cursor);
    free(indegree);
    free(topo_depth);
    free(queue);
    free(stack);
    free(seen);
    return -1;
  }

  if (reachable_mask != NULL) {
    memset(reachable_mask, 0, (size_t)n_nodes * sizeof(int32_t));
  }

  for (i = 0; i < n_nodes; ++i) {
    int32_t j;
    for (j = 0; j < 2; ++j) {
      int32_t parent = input_indices[(i * 2) + j];
      if (parent >= 0 && parent < n_nodes) {
        child_counts[parent] += 1;
        indegree[i] += 1;
        edge_count += 1;
      }
    }
  }

  child_offsets[0] = 0;
  for (i = 0; i < n_nodes; ++i) {
    child_offsets[i + 1] = child_offsets[i] + child_counts[i];
  }

  children = (int32_t*)malloc((size_t)(edge_count > 0 ? edge_count : 1) * sizeof(int32_t));
  if (children == NULL) {
    free(child_counts);
    free(child_offsets);
    free(child_cursor);
    free(indegree);
    free(topo_depth);
    free(queue);
    free(stack);
    free(seen);
    return -1;
  }

  memcpy(child_cursor, child_offsets, (size_t)n_nodes * sizeof(int32_t));
  for (i = 0; i < n_nodes; ++i) {
    int32_t j;
    for (j = 0; j < 2; ++j) {
      int32_t parent = input_indices[(i * 2) + j];
      if (parent >= 0 && parent < n_nodes) {
        children[child_cursor[parent]++] = i;
      }
    }
  }

  if (output_node_idx >= 0 && output_node_idx < n_nodes) {
    int32_t stack_size = 0;
    stack[stack_size++] = output_node_idx;
    while (stack_size > 0) {
      int32_t node = stack[--stack_size];
      int32_t j;
      if (seen[node]) {
        continue;
      }
      seen[node] = 1;
      if (reachable_mask != NULL) {
        reachable_mask[node] = 1;
      }
      reachable_count += 1;
      if (op_codes[node] == 0) {
        out->has_gradient_path = 1;
      }
      for (j = 0; j < 2; ++j) {
        int32_t parent = input_indices[(node * 2) + j];
        if (parent >= 0 && parent < n_nodes && !seen[parent]) {
          stack[stack_size++] = parent;
        }
      }
    }
  }
  out->reachable_count = reachable_count;

  if (param_estimates != NULL) {
    for (i = 0; i < n_nodes; ++i) {
      if (seen[i] && param_estimates[i] > 0) {
        param_total += param_estimates[i];
      }
    }
  }
  out->param_estimate = param_total;

  {
    int32_t head = 0;
    int32_t tail = 0;
    for (i = 0; i < n_nodes; ++i) {
      if (indegree[i] == 0) {
        queue[tail++] = i;
      }
    }

    while (head < tail) {
      int32_t node = queue[head++];
      int32_t next_depth = topo_depth[node] + 1;
      int32_t child_idx;
      visited_count += 1;

      if (seen[node] && topo_depth[node] > max_depth) {
        max_depth = topo_depth[node];
      }

      for (child_idx = child_offsets[node]; child_idx < child_offsets[node + 1];
           ++child_idx) {
        int32_t child = children[child_idx];
        if (next_depth > topo_depth[child]) {
          topo_depth[child] = next_depth;
        }
        indegree[child] -= 1;
        if (indegree[child] == 0) {
          queue[tail++] = child;
        }
      }
    }
  }

  out->depth = max_depth;
  out->has_cycle = visited_count < n_nodes ? 1 : 0;

  free(child_counts);
  free(child_offsets);
  free(child_cursor);
  free(children);
  free(indegree);
  free(topo_depth);
  free(queue);
  free(stack);
  free(seen);
  return 0;
}

int32_t aria_graph_dim_flow_summary(
    int32_t n_nodes,
    const int32_t* reachable_mask,
    const int32_t* has_params_flags,
    const int64_t* param_estimates,
    const int32_t* nontrivial_flags,
    const int32_t* kv_breaking_flags,
    aria_dim_flow_summary_t* out) {
  int32_t i;
  if (n_nodes < 0 || reachable_mask == NULL || out == NULL) {
    return -1;
  }

  out->reachable_param_count = 0;
  out->reachable_param_estimate = 0;
  out->reachable_nontrivial_ops = 0;
  out->reachable_ops = 0;
  out->kv_cacheable = 1;

  for (i = 0; i < n_nodes; ++i) {
    if (reachable_mask[i] == 0) {
      continue;
    }
    out->reachable_ops += 1;
    if (nontrivial_flags != NULL && nontrivial_flags[i] != 0) {
      out->reachable_nontrivial_ops += 1;
    }
    if (has_params_flags != NULL && has_params_flags[i] != 0) {
      out->reachable_param_count += 1;
      if (param_estimates != NULL && param_estimates[i] > 0) {
        out->reachable_param_estimate += param_estimates[i];
      }
    }
    if (kv_breaking_flags != NULL && kv_breaking_flags[i] != 0) {
      out->kv_cacheable = 0;
    }
  }
  return 0;
}

int32_t aria_graph_validate_edges(
    int32_t n_nodes,
    const int32_t* reachable_mask,
    const int32_t* input_indices,
    const int32_t* node_dims,
    const int32_t* node_seq_flags,
    const int32_t* op_kind_flags,
    const int32_t* full_dim_flags,
    int32_t model_dim,
    aria_edge_validation_t* out) {
  int32_t i;
  if (n_nodes < 0 || reachable_mask == NULL || input_indices == NULL ||
      node_dims == NULL || node_seq_flags == NULL || op_kind_flags == NULL ||
      full_dim_flags == NULL || out == NULL) {
    return -1;
  }

  for (i = 0; i < n_nodes; ++i) {
    int32_t in0;
    int32_t in1;
    int32_t input_slot;
    out[i].freq_mismatch_bits = 0;
    out[i].reduce_full_dim_bits = 0;
    out[i].binary_dim_mismatch = 0;
    out[i].full_dim_input_bits = 0;

    if (reachable_mask[i] == 0) {
      continue;
    }

    in0 = input_indices[(i * 2)];
    in1 = input_indices[(i * 2) + 1];

    for (input_slot = 0; input_slot < 2; ++input_slot) {
      int32_t parent = input_slot == 0 ? in0 : in1;
      if (parent < 0 || parent >= n_nodes) {
        continue;
      }
      if (node_seq_flags[parent] != 0 && op_kind_flags[i] != 1 && op_kind_flags[i] != 2) {
        out[i].freq_mismatch_bits |= (1 << input_slot);
      }
      if (node_dims[parent] == 1 && full_dim_flags[i] != 0) {
        out[i].reduce_full_dim_bits |= (1 << input_slot);
      }
      if (full_dim_flags[i] != 0 && node_dims[parent] != model_dim) {
        out[i].full_dim_input_bits |= (1 << input_slot);
      }
    }

    if (op_kind_flags[i] == 3 && in0 >= 0 && in1 >= 0) {
      int32_t d0 = node_dims[in0];
      int32_t d1 = node_dims[in1];
      if (d0 != d1 && d0 != 1 && d1 != 1) {
        out[i].binary_dim_mismatch = 1;
      }
    }
  }

  return 0;
}

int32_t aria_graph_validation_summary(
    int32_t n_nodes,
    const int32_t* known_op_flags,
    const int32_t* risky_op_flags,
    const int32_t* parameterized_op_flags,
    const int32_t* norm_op_flags,
    const int32_t* linear_op_flags,
    aria_validation_summary_t* out) {
  int32_t i;
  int32_t projection_chain_depth = 0;

  if (n_nodes < 0 || known_op_flags == NULL || risky_op_flags == NULL ||
      parameterized_op_flags == NULL || norm_op_flags == NULL ||
      linear_op_flags == NULL || out == NULL) {
    return -1;
  }

  out->risky_op_count = 0;
  out->parameterized_op_count = 0;
  out->unknown_op_count = 0;
  out->max_projection_chain_depth = 0;

  for (i = 0; i < n_nodes; ++i) {
    if (known_op_flags[i] == 0) {
      out->unknown_op_count += 1;
      continue;
    }

    if (risky_op_flags[i] != 0) {
      out->risky_op_count += 1;
    }
    if (parameterized_op_flags[i] != 0) {
      out->parameterized_op_count += 1;
    }

    if (norm_op_flags[i] != 0) {
      projection_chain_depth = 0;
      continue;
    }

    if (linear_op_flags[i] != 0) {
      projection_chain_depth += 1;
      if (projection_chain_depth > out->max_projection_chain_depth) {
        out->max_projection_chain_depth = projection_chain_depth;
      }
    }
  }

  return 0;
}

int32_t aria_graph_dead_parameterized_mask(
    int32_t n_nodes,
    const int32_t* reachable_mask,
    const int32_t* parameterized_flags,
    int32_t* dead_mask) {
  int32_t i;

  if (n_nodes < 0 || reachable_mask == NULL || parameterized_flags == NULL ||
      dead_mask == NULL) {
    return -1;
  }

  for (i = 0; i < n_nodes; ++i) {
    dead_mask[i] = (reachable_mask[i] == 0 && parameterized_flags[i] != 0) ? 1 : 0;
  }

  return 0;
}
