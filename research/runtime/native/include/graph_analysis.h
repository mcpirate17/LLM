#ifndef ARIA_GRAPH_ANALYSIS_H
#define ARIA_GRAPH_ANALYSIS_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  int32_t has_gradient_path;
  int32_t reachable_count;
  int32_t depth;
  int32_t has_cycle;
  int64_t param_estimate;
} aria_graph_analysis_result_t;

typedef struct {
  int32_t reachable_param_count;
  int64_t reachable_param_estimate;
  int32_t reachable_nontrivial_ops;
  int32_t reachable_ops;
  int32_t kv_cacheable;
} aria_dim_flow_summary_t;

typedef struct {
  int32_t freq_mismatch_bits;
  int32_t reduce_full_dim_bits;
  int32_t binary_dim_mismatch;
  int32_t full_dim_input_bits;
} aria_edge_validation_t;

typedef struct {
  int32_t risky_op_count;
  int32_t parameterized_op_count;
  int32_t unknown_op_count;
  int32_t max_projection_chain_depth;
} aria_validation_summary_t;

int32_t aria_graph_analyze_ir(
    int32_t n_nodes,
    const int32_t* op_codes,
    const int32_t* input_indices,
    int32_t output_node_idx,
    const int64_t* param_estimates,
    aria_graph_analysis_result_t* out,
    int32_t* reachable_mask);

int32_t aria_graph_dim_flow_summary(
    int32_t n_nodes,
    const int32_t* reachable_mask,
    const int32_t* has_params_flags,
    const int64_t* param_estimates,
    const int32_t* nontrivial_flags,
    const int32_t* kv_breaking_flags,
    aria_dim_flow_summary_t* out);

int32_t aria_graph_validate_edges(
    int32_t n_nodes,
    const int32_t* reachable_mask,
    const int32_t* input_indices,
    const int32_t* node_dims,
    const int32_t* node_seq_flags,
    const int32_t* op_kind_flags,
    const int32_t* full_dim_flags,
    int32_t model_dim,
    aria_edge_validation_t* out);

int32_t aria_graph_validation_summary(
    int32_t n_nodes,
    const int32_t* known_op_flags,
    const int32_t* risky_op_flags,
    const int32_t* parameterized_op_flags,
    const int32_t* norm_op_flags,
    const int32_t* linear_op_flags,
    aria_validation_summary_t* out);

int32_t aria_graph_dead_parameterized_mask(
    int32_t n_nodes,
    const int32_t* reachable_mask,
    const int32_t* parameterized_flags,
    int32_t* dead_mask);

int32_t aria_eval_param_formula(const char* formula, int64_t* out_value);

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
    int32_t* full_dim_flags);

int32_t aria_graph_mutation_plan(
    int32_t n_nodes,
    const int32_t* op_codes,
    int32_t n_opcodes,
    const int32_t* opcode_category_ids,
    const int32_t* opcode_input_arities,
    uint64_t rng_seed,
    int32_t max_pairs,
    int32_t* out_node_indices,
    int32_t* out_candidate_opcodes,
    int32_t* out_pair_count);

#ifdef __cplusplus
}
#endif

#endif
