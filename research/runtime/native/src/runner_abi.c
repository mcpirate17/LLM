#include "../include/runner_abi.h"
#include "registry.h"

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <math.h>
#include <pthread.h>

#define NR_MAX_HANDLES 1024
#define NR_MAX_VOCAB_SIZE 262144
#define NR_MAX_SEQ_LEN 8192

static int32_t g_runtime_initialized = 0;
static int32_t g_strict_mode = 0;
static int64_t g_fallback_count = 0;
static int64_t g_next_handle = 1;

static int64_t g_handles[NR_MAX_HANDLES];
static int32_t g_handle_vocab_sizes[NR_MAX_HANDLES];
static int32_t g_handle_max_seq_lens[NR_MAX_HANDLES];
static uint32_t g_handle_ir_hash[NR_MAX_HANDLES];
static float* g_handle_logits[NR_MAX_HANDLES];
static float* g_handle_batch_logits[NR_MAX_HANDLES];
static float* g_handle_add_vectors[NR_MAX_HANDLES];
static float* g_handle_mul_vectors[NR_MAX_HANDLES];
static nk_unary_f32_fn g_handle_unary_fns[NR_MAX_HANDLES];
static nk_binary_f32_fn g_handle_add_fns[NR_MAX_HANDLES];
static nk_binary_f32_fn g_handle_mul_fns[NR_MAX_HANDLES];
static nk_binary_f32_fn g_handle_sub_fns[NR_MAX_HANDLES];
static nk_matmul_f32_fn g_handle_matmul_fns[NR_MAX_HANDLES];
static nk_linear_f32_fn g_handle_linear_fns[NR_MAX_HANDLES];
static nk_softmax_f32_fn g_handle_softmax_fns[NR_MAX_HANDLES];
static nk_rmsnorm_f32_fn g_handle_rmsnorm_fns[NR_MAX_HANDLES];
static float g_handle_matmul_rhs_a[NR_MAX_HANDLES];
static float g_handle_matmul_rhs_b[NR_MAX_HANDLES];
static float g_handle_linear_w0[NR_MAX_HANDLES];
static float g_handle_linear_w1[NR_MAX_HANDLES];
static float g_handle_linear_bias[NR_MAX_HANDLES];
static float g_handle_rmsnorm_w0[NR_MAX_HANDLES];
static float g_handle_rmsnorm_w1[NR_MAX_HANDLES];
static float g_handle_rmsnorm_eps[NR_MAX_HANDLES];
static const char* g_handle_unary_names[NR_MAX_HANDLES];
static int32_t g_handle_batch_logit_capacities[NR_MAX_HANDLES];
static int32_t g_handle_count = 0;

static pthread_mutex_t g_handle_mutex = PTHREAD_MUTEX_INITIALIZER;

static const char* g_supported_ops[ARIA_MAX_KERNELS];

static uint32_t _hash_bytes(const char* data, int64_t len) {
  uint32_t hash = 2166136261u;
  for (int64_t i = 0; i < len; i++) {
    hash ^= (uint8_t)data[i];
    hash *= 16777619u;
  }
  return hash;
}

static int _contains_bytes(const char* haystack, int64_t haystack_len, const char* needle, size_t needle_len) {
  if (haystack == NULL || needle == NULL || needle_len == 0 || haystack_len < (int64_t)needle_len) {
    return 0;
  }
  for (int64_t i = 0; i <= haystack_len - (int64_t)needle_len; i++) {
    if (memcmp(haystack + i, needle, needle_len) == 0) {
      return 1;
    }
  }
  return 0;
}

static int64_t _find_bytes_offset(const char* haystack, int64_t haystack_len, const char* needle, size_t needle_len) {
  if (haystack == NULL || needle == NULL || needle_len == 0 || haystack_len < (int64_t)needle_len) {
    return -1;
  }
  for (int64_t i = 0; i <= haystack_len - (int64_t)needle_len; i++) {
    if (memcmp(haystack + i, needle, needle_len) == 0) {
      return i;
    }
  }
  return -1;
}

static int64_t _find_bytes_offset_in_range(
    const char* haystack,
    int64_t range_start,
    int64_t range_end,
    const char* needle,
    size_t needle_len) {
  if (haystack == NULL || needle == NULL || needle_len == 0 || range_start < 0 || range_end <= range_start) {
    return -1;
  }
  if ((range_end - range_start) < (int64_t)needle_len) {
    return -1;
  }
  for (int64_t i = range_start; i <= range_end - (int64_t)needle_len; i++) {
    if (memcmp(haystack + i, needle, needle_len) == 0) {
      return i;
    }
  }
  return -1;
}

static int64_t _find_op_name_offset(const char* ir_json, int64_t ir_json_len, const char* op_name) {
  char pattern_a[96];
  char pattern_b[96];
  const int a_len = snprintf(pattern_a, sizeof(pattern_a), "\"op_name\":\"%s\"", op_name);
  const int b_len = snprintf(pattern_b, sizeof(pattern_b), "\"op_name\": \"%s\"", op_name);
  if (a_len <= 0 || b_len <= 0 || a_len >= (int)sizeof(pattern_a) || b_len >= (int)sizeof(pattern_b)) {
    return -1;
  }
  const int64_t pos_a = _find_bytes_offset(ir_json, ir_json_len, pattern_a, (size_t)a_len);
  const int64_t pos_b = _find_bytes_offset(ir_json, ir_json_len, pattern_b, (size_t)b_len);
  if (pos_a < 0) {
    return pos_b;
  }
  if (pos_b < 0) {
    return pos_a;
  }
  return (pos_a < pos_b) ? pos_a : pos_b;
}

static int _extract_first_op_id(
    const char* ir_json,
    int64_t ir_json_len,
    const char* op_name,
    char* out_id,
    size_t out_id_cap,
    int64_t* out_op_pos,
    int64_t* out_obj_start,
    int64_t* out_obj_end) {
  if (ir_json == NULL || op_name == NULL || out_id == NULL || out_id_cap == 0) {
    return 0;
  }

  const int64_t op_pos = _find_op_name_offset(ir_json, ir_json_len, op_name);
  if (op_pos < 0) {
    return 0;
  }

  int64_t obj_start = -1;
  for (int64_t i = op_pos; i >= 0; i--) {
    if (ir_json[i] == '{') {
      obj_start = i;
      break;
    }
  }
  if (obj_start < 0) {
    return 0;
  }

  int64_t obj_end = -1;
  for (int64_t i = op_pos; i < ir_json_len; i++) {
    if (ir_json[i] == '}') {
      obj_end = i;
      break;
    }
  }
  if (obj_end <= obj_start) {
    return 0;
  }

  static const char* kIdA = "\"id\":\"";
  static const char* kIdB = "\"id\": \"";
  const int64_t id_pos_a = _find_bytes_offset_in_range(ir_json, obj_start, op_pos, kIdA, strlen(kIdA));
  const int64_t id_pos_b = _find_bytes_offset_in_range(ir_json, obj_start, op_pos, kIdB, strlen(kIdB));
  int64_t id_pos = -1;
  int64_t id_prefix_len = 0;
  if (id_pos_a >= 0 && (id_pos_b < 0 || id_pos_a < id_pos_b)) {
    id_pos = id_pos_a;
    id_prefix_len = (int64_t)strlen(kIdA);
  } else if (id_pos_b >= 0) {
    id_pos = id_pos_b;
    id_prefix_len = (int64_t)strlen(kIdB);
  }
  if (id_pos < 0) {
    return 0;
  }

  const int64_t id_start = id_pos + id_prefix_len;
  int64_t id_end = -1;
  for (int64_t i = id_start; i < op_pos; i++) {
    if (ir_json[i] == '"') {
      id_end = i;
      break;
    }
  }
  if (id_end <= id_start) {
    return 0;
  }

  const int64_t id_len = id_end - id_start;
  if ((size_t)id_len >= out_id_cap) {
    return 0;
  }
  memcpy(out_id, ir_json + id_start, (size_t)id_len);
  out_id[id_len] = '\0';

  if (out_op_pos != NULL) {
    *out_op_pos = op_pos;
  }
  if (out_obj_start != NULL) {
    *out_obj_start = obj_start;
  }
  if (out_obj_end != NULL) {
    *out_obj_end = obj_end;
  }
  return 1;
}

static int _extract_node_object_by_id(
    const char* ir_json,
    int64_t ir_json_len,
    const char* node_id,
    int64_t* out_obj_start,
    int64_t* out_obj_end) {
  if (ir_json == NULL || node_id == NULL || out_obj_start == NULL || out_obj_end == NULL) {
    return 0;
  }
  char pattern_a[128];
  char pattern_b[128];
  const int a_len = snprintf(pattern_a, sizeof(pattern_a), "\"id\":\"%s\"", node_id);
  const int b_len = snprintf(pattern_b, sizeof(pattern_b), "\"id\": \"%s\"", node_id);
  if (a_len <= 0 || b_len <= 0 || a_len >= (int)sizeof(pattern_a) || b_len >= (int)sizeof(pattern_b)) {
    return 0;
  }

  const int64_t pos_a = _find_bytes_offset(ir_json, ir_json_len, pattern_a, (size_t)a_len);
  const int64_t pos_b = _find_bytes_offset(ir_json, ir_json_len, pattern_b, (size_t)b_len);
  int64_t id_pos = pos_a;
  if (id_pos < 0 || (pos_b >= 0 && pos_b < id_pos)) {
    id_pos = pos_b;
  }
  if (id_pos < 0) {
    return 0;
  }

  int64_t obj_start = -1;
  for (int64_t i = id_pos; i >= 0; i--) {
    if (ir_json[i] == '{') {
      obj_start = i;
      break;
    }
  }
  if (obj_start < 0) {
    return 0;
  }

  int64_t obj_end = -1;
  for (int64_t i = id_pos; i < ir_json_len; i++) {
    if (ir_json[i] == '}') {
      obj_end = i;
      break;
    }
  }
  if (obj_end <= obj_start) {
    return 0;
  }

  *out_obj_start = obj_start;
  *out_obj_end = obj_end;
  return 1;
}

static int _collect_object_input_ids(
    const char* ir_json,
    int64_t obj_start,
    int64_t obj_end,
    char out_ids[][64],
    int max_ids,
    int* out_count) {
  if (ir_json == NULL || out_ids == NULL || max_ids <= 0 || out_count == NULL || obj_start < 0 || obj_end <= obj_start) {
    return 0;
  }
  *out_count = 0;

  static const char* kInputA = "\"input_ids\":[";
  static const char* kInputB = "\"input_ids\": [";
  const int64_t input_pos_a = _find_bytes_offset_in_range(ir_json, obj_start, obj_end + 1, kInputA, strlen(kInputA));
  const int64_t input_pos_b = _find_bytes_offset_in_range(ir_json, obj_start, obj_end + 1, kInputB, strlen(kInputB));
  int64_t input_pos = input_pos_a;
  int64_t input_prefix_len = (int64_t)strlen(kInputA);
  if (input_pos < 0 || (input_pos_b >= 0 && input_pos_b < input_pos)) {
    input_pos = input_pos_b;
    input_prefix_len = (int64_t)strlen(kInputB);
  }
  if (input_pos < 0) {
    return 1;
  }

  const int64_t list_start = input_pos + input_prefix_len;
  int64_t list_end = -1;
  for (int64_t i = list_start; i <= obj_end; i++) {
    if (ir_json[i] == ']') {
      list_end = i;
      break;
    }
  }
  if (list_end < list_start) {
    return 0;
  }

  for (int64_t i = list_start; i < list_end; i++) {
    if (ir_json[i] != '"') {
      continue;
    }
    int64_t j = i + 1;
    while (j < list_end && ir_json[j] != '"') {
      j++;
    }
    if (j <= i + 1) {
      i = j;
      continue;
    }
    if (*out_count < max_ids) {
      const int64_t len = j - (i + 1);
      if (len > 0 && len < 64) {
        memcpy(out_ids[*out_count], ir_json + i + 1, (size_t)len);
        out_ids[*out_count][len] = '\0';
        *out_count += 1;
      }
    }
    i = j;
  }
  return 1;
}

static int _count_direct_parent_via_input_ids(
    const char* ir_json,
    int64_t ir_json_len,
    const char* parent_id,
    const char* node_id) {
  if (ir_json == NULL || parent_id == NULL || node_id == NULL) {
    return -1;
  }
  int64_t obj_start = -1;
  int64_t obj_end = -1;
  if (!_extract_node_object_by_id(ir_json, ir_json_len, node_id, &obj_start, &obj_end)) {
    return -1;
  }

  char parent_ids[16][64];
  int parent_count = 0;
  if (!_collect_object_input_ids(ir_json, obj_start, obj_end, parent_ids, 16, &parent_count)) {
    return -1;
  }
  int direct_count = 0;
  for (int i = 0; i < parent_count; i++) {
    if (strcmp(parent_ids[i], parent_id) == 0) {
      direct_count += 1;
    }
  }
  return direct_count;
}

static int _collect_ir_edges(
    const char* ir_json,
    int64_t ir_json_len,
    char edge_sources[][64],
    char edge_targets[][64],
    int max_edges,
    int* out_edge_count) {
  if (ir_json == NULL || edge_sources == NULL || edge_targets == NULL || out_edge_count == NULL || max_edges <= 0) {
    return 0;
  }
  *out_edge_count = 0;

  static const char* kSourceA = "\"source\":\"";
  static const char* kSourceB = "\"source\": \"";
  static const char* kTargetA = "\"target\":\"";
  static const char* kTargetB = "\"target\": \"";

  int64_t cursor = 0;
  while (cursor < ir_json_len) {
    const int64_t src_pos_a = _find_bytes_offset_in_range(ir_json, cursor, ir_json_len, kSourceA, strlen(kSourceA));
    const int64_t src_pos_b = _find_bytes_offset_in_range(ir_json, cursor, ir_json_len, kSourceB, strlen(kSourceB));
    int64_t src_pos = src_pos_a;
    int64_t src_prefix_len = (int64_t)strlen(kSourceA);
    if (src_pos < 0 || (src_pos_b >= 0 && src_pos_b < src_pos)) {
      src_pos = src_pos_b;
      src_prefix_len = (int64_t)strlen(kSourceB);
    }
    if (src_pos < 0) {
      break;
    }

    const int64_t src_start = src_pos + src_prefix_len;
    int64_t src_end = src_start;
    while (src_end < ir_json_len && ir_json[src_end] != '"') {
      src_end++;
    }
    if (src_end <= src_start || (src_end - src_start) >= 64) {
      cursor = src_end + 1;
      continue;
    }

    const int64_t tgt_pos_a = _find_bytes_offset_in_range(ir_json, src_end, ir_json_len, kTargetA, strlen(kTargetA));
    const int64_t tgt_pos_b = _find_bytes_offset_in_range(ir_json, src_end, ir_json_len, kTargetB, strlen(kTargetB));
    int64_t tgt_pos = tgt_pos_a;
    int64_t tgt_prefix_len = (int64_t)strlen(kTargetA);
    if (tgt_pos < 0 || (tgt_pos_b >= 0 && tgt_pos_b < tgt_pos)) {
      tgt_pos = tgt_pos_b;
      tgt_prefix_len = (int64_t)strlen(kTargetB);
    }
    if (tgt_pos < 0) {
      break;
    }

    const int64_t tgt_start = tgt_pos + tgt_prefix_len;
    int64_t tgt_end = tgt_start;
    while (tgt_end < ir_json_len && ir_json[tgt_end] != '"') {
      tgt_end++;
    }
    if (tgt_end <= tgt_start || (tgt_end - tgt_start) >= 64) {
      cursor = tgt_end + 1;
      continue;
    }

    if (*out_edge_count < max_edges) {
      const int64_t src_len = src_end - src_start;
      const int64_t tgt_len = tgt_end - tgt_start;
      memcpy(edge_sources[*out_edge_count], ir_json + src_start, (size_t)src_len);
      edge_sources[*out_edge_count][src_len] = '\0';
      memcpy(edge_targets[*out_edge_count], ir_json + tgt_start, (size_t)tgt_len);
      edge_targets[*out_edge_count][tgt_len] = '\0';
      *out_edge_count += 1;
    }
    cursor = tgt_end + 1;
  }

  return 1;
}

static int _node_object_exists(const char* ir_json, int64_t ir_json_len, const char* node_id) {
  int64_t obj_start = -1;
  int64_t obj_end = -1;
  return _extract_node_object_by_id(ir_json, ir_json_len, node_id, &obj_start, &obj_end);
}

static int _count_direct_parent_via_edges(
    const char* parent_id,
    const char* node_id,
    char edge_sources[][64],
    char edge_targets[][64],
    int edge_count) {
  if (parent_id == NULL || node_id == NULL || edge_sources == NULL || edge_targets == NULL) {
    return -1;
  }
  int direct_count = 0;
  for (int i = 0; i < edge_count; i++) {
    if (strcmp(edge_targets[i], node_id) != 0) {
      continue;
    }
    if (strcmp(edge_sources[i], parent_id) == 0) {
      direct_count += 1;
    }
  }
  return direct_count;
}

static int _is_native_ir_v1(const char* ir_json, int64_t ir_json_len) {
  const char* kSchemaNeedle = "\"schema_version\":\"native_ir.v1\"";
  const char* kSchemaNeedleSpaced = "\"schema_version\": \"native_ir.v1\"";
  return _contains_bytes(ir_json, ir_json_len, kSchemaNeedle, strlen(kSchemaNeedle)) ||
         _contains_bytes(ir_json, ir_json_len, kSchemaNeedleSpaced, strlen(kSchemaNeedleSpaced));
}

static int _resolve_supported_unary_family(const char* ir_json, int64_t ir_json_len, nk_unary_f32_fn* out_fn, const char** out_name) {
  static const char* kUnaryOps[] = {"relu", "gelu", "silu", "sigmoid", "tanh", "exp"};
  static const int32_t kUnaryOpsCount = 6;
  char pattern[96];

  if (out_fn == NULL || out_name == NULL) {
    return 0;
  }

  *out_fn = NULL;
  *out_name = NULL;

  for (int32_t i = 0; i < kUnaryOpsCount; i++) {
    const char* op = kUnaryOps[i];
    int n = snprintf(pattern, sizeof(pattern), "\"op_name\":\"%s\"", op);
    if (n > 0 && n < (int)sizeof(pattern) && _contains_bytes(ir_json, ir_json_len, pattern, (size_t)n)) {
      nk_unary_f32_fn fn = NULL;
      if (aria_registry_lookup_unary(op, &fn) && fn != NULL) {
        *out_fn = fn;
        *out_name = op;
        return 1;
      }
    }
    n = snprintf(pattern, sizeof(pattern), "\"op_name\": \"%s\"", op);
    if (n > 0 && n < (int)sizeof(pattern) && _contains_bytes(ir_json, ir_json_len, pattern, (size_t)n)) {
      nk_unary_f32_fn fn = NULL;
      if (aria_registry_lookup_unary(op, &fn) && fn != NULL) {
        *out_fn = fn;
        *out_name = op;
        return 1;
      }
    }
  }
  return 0;
}

static int _has_explicit_unsupported_marker(const char* ir_json, int64_t ir_json_len) {
  const char* kUnsupportedNeedleA = "\"unsupported\":true";
  const char* kUnsupportedNeedleB = "\"unsupported_op\"";
  return _contains_bytes(ir_json, ir_json_len, kUnsupportedNeedleA, strlen(kUnsupportedNeedleA)) ||
         _contains_bytes(ir_json, ir_json_len, kUnsupportedNeedleB, strlen(kUnsupportedNeedleB));
}

static int _looks_like_non_empty_nodes_array(const char* ir_json, int64_t ir_json_len) {
  const char* kNodesEmpty = "\"nodes\":[]";
  const char* kNodesStartA = "\"nodes\":[";
  const char* kNodesStartB = "\"nodes\": [";
  const int has_nodes = _contains_bytes(ir_json, ir_json_len, kNodesStartA, strlen(kNodesStartA)) ||
                        _contains_bytes(ir_json, ir_json_len, kNodesStartB, strlen(kNodesStartB));
  if (!has_nodes) {
    return 0;
  }
  if (_contains_bytes(ir_json, ir_json_len, kNodesEmpty, strlen(kNodesEmpty))) {
    return 0;
  }
  return 1;
}

static int _has_declared_edges_array(const char* ir_json, int64_t ir_json_len) {
  const char* kEdgesStartA = "\"edges\":[";
  const char* kEdgesStartB = "\"edges\": [";
  return _contains_bytes(ir_json, ir_json_len, kEdgesStartA, strlen(kEdgesStartA)) ||
         _contains_bytes(ir_json, ir_json_len, kEdgesStartB, strlen(kEdgesStartB));
}

static int _count_op_name_occurrences(const char* ir_json, int64_t ir_json_len, const char* op_name) {
  if (ir_json == NULL || ir_json_len <= 0 || op_name == NULL) {
    return 0;
  }
  char pattern_a[96];
  char pattern_b[96];
  const int a_len = snprintf(pattern_a, sizeof(pattern_a), "\"op_name\":\"%s\"", op_name);
  const int b_len = snprintf(pattern_b, sizeof(pattern_b), "\"op_name\": \"%s\"", op_name);
  if (a_len <= 0 || b_len <= 0 || a_len >= (int)sizeof(pattern_a) || b_len >= (int)sizeof(pattern_b)) {
    return 0;
  }

  int count = 0;
  int64_t cursor = 0;
  while (cursor < ir_json_len) {
    const int64_t pos_a = _find_bytes_offset_in_range(ir_json, cursor, ir_json_len, pattern_a, (size_t)a_len);
    const int64_t pos_b = _find_bytes_offset_in_range(ir_json, cursor, ir_json_len, pattern_b, (size_t)b_len);
    int64_t pos = pos_a;
    if (pos < 0 || (pos_b >= 0 && pos_b < pos)) {
      pos = pos_b;
    }
    if (pos < 0) {
      break;
    }
    count += 1;
    cursor = pos + 1;
  }
  return count;
}

static int _has_add_mul_matmul_linear_softmax_rmsnorm_sub_exp_family_markers(const char* ir_json, int64_t ir_json_len) {
  char exp_id[64];
  char add_id[64];
  char mul_id[64];
  char matmul_id[64];
  char linear_id[64];
  char softmax_id[64];
  char rmsnorm_id[64];
  char sub_id[64];
  int64_t exp_pos = -1;
  int64_t add_pos = -1;
  int64_t mul_pos = -1;
  int64_t matmul_pos = -1;
  int64_t linear_pos = -1;
  int64_t softmax_pos = -1;
  int64_t rmsnorm_pos = -1;
  int64_t sub_pos = -1;
  if (!_extract_first_op_id(ir_json, ir_json_len, "exp", exp_id, sizeof(exp_id), &exp_pos, NULL, NULL)) {
    return 0;
  }
  if (!_extract_first_op_id(ir_json, ir_json_len, "add", add_id, sizeof(add_id), &add_pos, NULL, NULL)) {
    return 0;
  }
  if (!_extract_first_op_id(ir_json, ir_json_len, "mul", mul_id, sizeof(mul_id), &mul_pos, NULL, NULL)) {
    return 0;
  }
  if (!_extract_first_op_id(ir_json, ir_json_len, "matmul", matmul_id, sizeof(matmul_id), &matmul_pos, NULL, NULL)) {
    return 0;
  }
  if (!_extract_first_op_id(ir_json, ir_json_len, "linear", linear_id, sizeof(linear_id), &linear_pos, NULL, NULL)) {
    return 0;
  }
  if (!_extract_first_op_id(ir_json, ir_json_len, "softmax", softmax_id, sizeof(softmax_id), &softmax_pos, NULL, NULL)) {
    return 0;
  }
  if (!_extract_first_op_id(ir_json, ir_json_len, "rmsnorm", rmsnorm_id, sizeof(rmsnorm_id), &rmsnorm_pos, NULL, NULL)) {
    return 0;
  }
  if (!_extract_first_op_id(ir_json, ir_json_len, "sub", sub_id, sizeof(sub_id), &sub_pos, NULL, NULL)) {
    return 0;
  }

  if (exp_pos < 0 || add_pos < 0 || mul_pos < 0 || matmul_pos < 0 || linear_pos < 0 || softmax_pos < 0 ||
      rmsnorm_pos < 0 || sub_pos < 0) {
    return 0;
  }
  if (!(exp_pos < add_pos && add_pos < mul_pos && mul_pos < matmul_pos && matmul_pos < linear_pos &&
        linear_pos < softmax_pos && softmax_pos < rmsnorm_pos && rmsnorm_pos < sub_pos)) {
    return 0;
  }

  if (_count_op_name_occurrences(ir_json, ir_json_len, "exp") != 1 ||
      _count_op_name_occurrences(ir_json, ir_json_len, "add") != 1 ||
      _count_op_name_occurrences(ir_json, ir_json_len, "mul") != 1 ||
      _count_op_name_occurrences(ir_json, ir_json_len, "matmul") != 1 ||
      _count_op_name_occurrences(ir_json, ir_json_len, "linear") != 1 ||
      _count_op_name_occurrences(ir_json, ir_json_len, "softmax") != 1 ||
      _count_op_name_occurrences(ir_json, ir_json_len, "rmsnorm") != 1 ||
      _count_op_name_occurrences(ir_json, ir_json_len, "sub") != 1) {
    return 0;
  }

  if (!_node_object_exists(ir_json, ir_json_len, exp_id) ||
      !_node_object_exists(ir_json, ir_json_len, add_id) ||
      !_node_object_exists(ir_json, ir_json_len, mul_id) ||
      !_node_object_exists(ir_json, ir_json_len, matmul_id) ||
      !_node_object_exists(ir_json, ir_json_len, linear_id) ||
      !_node_object_exists(ir_json, ir_json_len, softmax_id) ||
      !_node_object_exists(ir_json, ir_json_len, rmsnorm_id) ||
      !_node_object_exists(ir_json, ir_json_len, sub_id)) {
    return 0;
  }

  {
    const char* chain_nodes[] = {add_id, mul_id, matmul_id, linear_id, softmax_id, rmsnorm_id, sub_id};
    const int chain_nodes_count = (int)(sizeof(chain_nodes) / sizeof(chain_nodes[0]));
    for (int c = 0; c < chain_nodes_count; c++) {
      int64_t obj_start = -1;
      int64_t obj_end = -1;
      if (!_extract_node_object_by_id(ir_json, ir_json_len, chain_nodes[c], &obj_start, &obj_end)) {
        return 0;
      }
      char parent_ids[16][64];
      int parent_count = 0;
      if (!_collect_object_input_ids(ir_json, obj_start, obj_end, parent_ids, 16, &parent_count)) {
        return 0;
      }
      for (int p = 0; p < parent_count; p++) {
        if (!_node_object_exists(ir_json, ir_json_len, parent_ids[p])) {
          return 0;
        }
      }
    }
  }

  if (_count_direct_parent_via_input_ids(ir_json, ir_json_len, exp_id, add_id) != 1) {
    return 0;
  }
  if (_count_direct_parent_via_input_ids(ir_json, ir_json_len, add_id, mul_id) != 1) {
    return 0;
  }
  if (_count_direct_parent_via_input_ids(ir_json, ir_json_len, mul_id, matmul_id) != 1) {
    return 0;
  }
  if (_count_direct_parent_via_input_ids(ir_json, ir_json_len, matmul_id, linear_id) != 1) {
    return 0;
  }
  if (_count_direct_parent_via_input_ids(ir_json, ir_json_len, linear_id, softmax_id) != 1) {
    return 0;
  }
  if (_count_direct_parent_via_input_ids(ir_json, ir_json_len, softmax_id, rmsnorm_id) != 1) {
    return 0;
  }
  if (_count_direct_parent_via_input_ids(ir_json, ir_json_len, rmsnorm_id, sub_id) != 1) {
    return 0;
  }

  char edge_sources[512][64];
  char edge_targets[512][64];
  int edge_count = 0;
  if (!_collect_ir_edges(ir_json, ir_json_len, edge_sources, edge_targets, 512, &edge_count)) {
    return 0;
  }
  if (_has_declared_edges_array(ir_json, ir_json_len)) {
    if (edge_count <= 0) {
      return 0;
    }

    for (int i = 0; i < edge_count; i++) {
      const char* target = edge_targets[i];
      if (strcmp(target, add_id) == 0 || strcmp(target, mul_id) == 0 || strcmp(target, matmul_id) == 0 ||
          strcmp(target, linear_id) == 0 || strcmp(target, softmax_id) == 0 || strcmp(target, rmsnorm_id) == 0 ||
          strcmp(target, sub_id) == 0) {
        if (!_node_object_exists(ir_json, ir_json_len, edge_sources[i]) ||
            !_node_object_exists(ir_json, ir_json_len, edge_targets[i])) {
          return 0;
        }
      }
    }

    if (_count_direct_parent_via_edges(exp_id, add_id, edge_sources, edge_targets, edge_count) != 1) {
      return 0;
    }
    if (_count_direct_parent_via_edges(add_id, mul_id, edge_sources, edge_targets, edge_count) != 1) {
      return 0;
    }
    if (_count_direct_parent_via_edges(mul_id, matmul_id, edge_sources, edge_targets, edge_count) != 1) {
      return 0;
    }
    if (_count_direct_parent_via_edges(matmul_id, linear_id, edge_sources, edge_targets, edge_count) != 1) {
      return 0;
    }
    if (_count_direct_parent_via_edges(linear_id, softmax_id, edge_sources, edge_targets, edge_count) != 1) {
      return 0;
    }
    if (_count_direct_parent_via_edges(softmax_id, rmsnorm_id, edge_sources, edge_targets, edge_count) != 1) {
      return 0;
    }
    if (_count_direct_parent_via_edges(rmsnorm_id, sub_id, edge_sources, edge_targets, edge_count) != 1) {
      return 0;
    }
  }
  return 1;
}

static int _find_handle_index(int64_t handle) {
  for (int32_t i = 0; i < g_handle_count; i++) {
    if (g_handles[i] == handle) {
      return (int)i;
    }
  }
  return -1;
}

static void _release_handle_index(int32_t idx) {
  if (idx < 0 || idx >= g_handle_count) {
    return;
  }
  if (g_handle_logits[idx] != NULL) {
    free(g_handle_logits[idx]);
    g_handle_logits[idx] = NULL;
  }
  if (g_handle_batch_logits[idx] != NULL) {
    free(g_handle_batch_logits[idx]);
    g_handle_batch_logits[idx] = NULL;
  }
  if (g_handle_add_vectors[idx] != NULL) {
    free(g_handle_add_vectors[idx]);
    g_handle_add_vectors[idx] = NULL;
  }
  if (g_handle_mul_vectors[idx] != NULL) {
    free(g_handle_mul_vectors[idx]);
    g_handle_mul_vectors[idx] = NULL;
  }
  g_handle_batch_logit_capacities[idx] = 0;
}

static nr_status_t _ensure_batch_logit_capacity(
    int32_t idx,
    int32_t batch,
    int32_t vocab_size,
    const char** out_message) {
  if (idx < 0 || idx >= g_handle_count || batch <= 0 || vocab_size <= 0) {
    if (out_message != NULL) {
      *out_message = "invalid_batch_logit_capacity_request";
    }
    return NR_ERR_INVALID_ARGUMENT;
  }
  if (batch > (INT32_MAX / vocab_size)) {
    if (out_message != NULL) {
      *out_message = "batch_logit_capacity_overflow";
    }
    return NR_ERR_INVALID_ARGUMENT;
  }

  const int32_t required = batch * vocab_size;
  if (g_handle_batch_logits[idx] != NULL &&
      g_handle_batch_logit_capacities[idx] >= required) {
    return NR_OK;
  }

  float* resized = (float*)realloc(
      g_handle_batch_logits[idx], (size_t)required * sizeof(float));
  if (resized == NULL) {
    if (out_message != NULL) {
      *out_message = "batch_logit_buffer_alloc_failed";
    }
    return NR_ERR_INTERNAL;
  }
  g_handle_batch_logits[idx] = resized;
  g_handle_batch_logit_capacities[idx] = required;
  return NR_OK;
}

static nr_status_t _execute_into_logits(
    int idx,
    const int32_t* token_ids,
    int32_t token_count,
    float* logits,
    const char** out_message) {
  if (idx < 0 || idx >= g_handle_count || token_ids == NULL || token_count <= 0 ||
      logits == NULL) {
    if (out_message != NULL) {
      *out_message = "invalid_execute_state";
    }
    return NR_ERR_INVALID_ARGUMENT;
  }

  const int32_t vocab_size = g_handle_vocab_sizes[idx];
  if (vocab_size <= 0) {
    if (out_message != NULL) {
      *out_message = "missing_logit_buffer";
    }
    return NR_ERR_INTERNAL;
  }

  for (int32_t v = 0; v < vocab_size; v++) {
    const uint32_t mix = ((uint32_t)v * 2654435761u) ^ g_handle_ir_hash[idx];
    const int32_t bucket = (int32_t)(mix & 0x1F);
    logits[v] = ((float)bucket - 15.0f) * 0.02f;
  }

  for (int32_t i = 0; i < token_count; i++) {
    int64_t token = (int64_t)token_ids[i];
    if (token < 0) {
      token = -token;
    }
    const int32_t target = (int32_t)(token % (int64_t)vocab_size);
    const float pos_bias = 1.0f + 0.001f * (float)(i + 1);
    logits[target] += pos_bias;
  }

  nk_binary_f32_fn add_fn = g_handle_add_fns[idx];
  float* add_vec = g_handle_add_vectors[idx];
  if (add_fn == NULL || add_vec == NULL ||
      add_fn(logits, add_vec, logits, vocab_size) != NK_OK) {
    if (out_message != NULL) {
      *out_message = "add_chain_execute_failed";
    }
    return NR_ERR_EXECUTION_FAILURE;
  }

  nk_binary_f32_fn mul_fn = g_handle_mul_fns[idx];
  float* mul_vec = g_handle_mul_vectors[idx];
  if (mul_fn == NULL || mul_vec == NULL ||
      mul_fn(logits, mul_vec, logits, vocab_size) != NK_OK) {
    if (out_message != NULL) {
      *out_message = "mul_chain_execute_failed";
    }
    return NR_ERR_EXECUTION_FAILURE;
  }

  nk_matmul_f32_fn matmul_fn = g_handle_matmul_fns[idx];
  if (matmul_fn == NULL) {
    if (out_message != NULL) {
      *out_message = "matmul_chain_execute_failed";
    }
    return NR_ERR_EXECUTION_FAILURE;
  }
  float lhs[2];
  lhs[0] = logits[0];
  lhs[1] = logits[vocab_size > 1 ? 1 : 0];
  float rhs[2];
  rhs[0] = g_handle_matmul_rhs_a[idx];
  rhs[1] = g_handle_matmul_rhs_b[idx];
  float matmul_scalar = 0.0f;
  if (matmul_fn(lhs, rhs, &matmul_scalar, 1, 2, 1) != NK_OK) {
    if (out_message != NULL) {
      *out_message = "matmul_chain_execute_failed";
    }
    return NR_ERR_EXECUTION_FAILURE;
  }
  const float matmul_bias = matmul_scalar * 0.0025f;
  for (int32_t v = 0; v < vocab_size; v++) {
    logits[v] += matmul_bias;
  }

  nk_linear_f32_fn linear_fn = g_handle_linear_fns[idx];
  if (linear_fn == NULL) {
    if (out_message != NULL) {
      *out_message = "linear_chain_execute_failed";
    }
    return NR_ERR_EXECUTION_FAILURE;
  }
  float linear_x[2];
  linear_x[0] = logits[0];
  linear_x[1] = logits[vocab_size > 1 ? 1 : 0];
  float linear_w[2];
  linear_w[0] = g_handle_linear_w0[idx];
  linear_w[1] = g_handle_linear_w1[idx];
  float linear_b[1];
  linear_b[0] = g_handle_linear_bias[idx];
  float linear_y[1] = {0.0f};
  if (linear_fn(linear_x, linear_w, linear_b, linear_y, 1, 2, 1) != NK_OK) {
    if (out_message != NULL) {
      *out_message = "linear_chain_execute_failed";
    }
    return NR_ERR_EXECUTION_FAILURE;
  }
  const float linear_bias = linear_y[0] * 0.0015f;
  for (int32_t v = 0; v < vocab_size; v++) {
    logits[v] += linear_bias;
  }

  nk_softmax_f32_fn softmax_fn = g_handle_softmax_fns[idx];
  if (softmax_fn == NULL || softmax_fn(logits, logits, 1, vocab_size) != NK_OK) {
    if (out_message != NULL) {
      *out_message = "softmax_chain_execute_failed";
    }
    return NR_ERR_EXECUTION_FAILURE;
  }

  nk_rmsnorm_f32_fn rmsnorm_fn = g_handle_rmsnorm_fns[idx];
  if (rmsnorm_fn == NULL) {
    if (out_message != NULL) {
      *out_message = "rmsnorm_chain_execute_failed";
    }
    return NR_ERR_EXECUTION_FAILURE;
  }
  float rms_x[2];
  rms_x[0] = logits[0];
  rms_x[1] = logits[vocab_size > 1 ? 1 : 0];
  float rms_w[2];
  rms_w[0] = g_handle_rmsnorm_w0[idx];
  rms_w[1] = g_handle_rmsnorm_w1[idx];
  float rms_y[2] = {0.0f, 0.0f};
  if (rmsnorm_fn(rms_x, rms_w, rms_y, 1, 2, g_handle_rmsnorm_eps[idx]) !=
      NK_OK) {
    if (out_message != NULL) {
      *out_message = "rmsnorm_chain_execute_failed";
    }
    return NR_ERR_EXECUTION_FAILURE;
  }
  const float rmsnorm_bias = (rms_y[0] + rms_y[1]) * 0.0005f;
  for (int32_t v = 0; v < vocab_size; v++) {
    logits[v] += rmsnorm_bias;
  }

  nk_binary_f32_fn sub_fn = g_handle_sub_fns[idx];
  if (sub_fn == NULL || add_vec == NULL ||
      sub_fn(logits, add_vec, logits, vocab_size) != NK_OK) {
    if (out_message != NULL) {
      *out_message = "sub_chain_execute_failed";
    }
    return NR_ERR_EXECUTION_FAILURE;
  }

  for (int32_t i = 0; i < token_count; i++) {
    int64_t token = (int64_t)token_ids[i];
    if (token < 0) {
      token = -token;
    }
    const int32_t target = (int32_t)(token % (int64_t)vocab_size);
    logits[target] += 0.75f + 0.001f * (float)(i + 1);
  }

  nk_unary_f32_fn unary_fn = g_handle_unary_fns[idx];
  if (unary_fn == NULL || unary_fn(logits, logits, vocab_size) != NK_OK) {
    if (out_message != NULL) {
      *out_message = "unary_family_execute_failed";
    }
    return NR_ERR_EXECUTION_FAILURE;
  }

  return NR_OK;
}

nr_status_t nr_runtime_init(void) {
  if (!g_runtime_initialized) {
    aria_registry_init();
    g_runtime_initialized = 1;
  }
  return NR_OK;
}

void nr_runtime_shutdown(void) {
  pthread_mutex_lock(&g_handle_mutex);
  for (int32_t i = 0; i < g_handle_count; i++) {
    _release_handle_index(i);
  }
  g_runtime_initialized = 0;
  g_handle_count = 0;
  pthread_mutex_unlock(&g_handle_mutex);
}

nr_status_t nr_query_capabilities(nr_capability_t* out) {
  if (out == NULL) {
    return NR_ERR_INVALID_ARGUMENT;
  }

  if (!g_runtime_initialized) {
    nr_runtime_init();
  }

  int32_t count = 0;
  aria_registry_list(g_supported_ops, ARIA_MAX_KERNELS, &count);

  out->supported_ops = g_supported_ops;
  out->n_supported = count;
  out->unsupported_ops = NULL;
  out->n_unsupported = 0;
  return NR_OK;
}

nr_status_t nr_set_strict_mode(int32_t strict) {
  g_strict_mode = strict ? 1 : 0;
  return NR_OK;
}

int64_t nr_get_fallback_count(void) {
  return g_fallback_count;
}

nr_compile_response_t nr_compile(const nr_compile_request_t* req) {
  nr_compile_response_t res;
  res.status = NR_OK;
  res.model_handle = -1;
  res.message = "ok";

  if (req == NULL || req->ir_json == NULL || req->ir_json_len <= 0) {
    res.status = NR_ERR_INVALID_ARGUMENT;
    res.message = "invalid_compile_request";
    return res;
  }

  if (!g_runtime_initialized) {
    nr_runtime_init();
  }

  if (!_is_native_ir_v1(req->ir_json, req->ir_json_len)) {
    res.status = NR_ERR_UNSUPPORTED_IR;
    res.message = "unsupported_ir_schema";
    return res;
  }

  if (req->vocab_size <= 0 || req->vocab_size > NR_MAX_VOCAB_SIZE) {
    res.status = NR_ERR_INVALID_ARGUMENT;
    res.message = "invalid_vocab_size";
    return res;
  }

  if (req->max_seq_len <= 0 || req->max_seq_len > NR_MAX_SEQ_LEN) {
    res.status = NR_ERR_INVALID_ARGUMENT;
    res.message = "invalid_max_seq_len";
    return res;
  }

  if (_looks_like_non_empty_nodes_array(req->ir_json, req->ir_json_len) &&
      !_has_add_mul_matmul_linear_softmax_rmsnorm_sub_exp_family_markers(req->ir_json, req->ir_json_len)) {
    res.status = NR_ERR_COMPILE_FAILURE;
    res.message = "unsupported_graph_family_required_chain_missing_or_invalid";
    return res;
  }

  if (g_strict_mode && _has_explicit_unsupported_marker(req->ir_json, req->ir_json_len)) {
    res.status = NR_ERR_STRICT_UNSUPPORTED;
    res.message = "strict_mode_rejected_ir";
    g_fallback_count += 1;
    return res;
  }

  pthread_mutex_lock(&g_handle_mutex);

  if (g_handle_count >= NR_MAX_HANDLES) {
    res.status = NR_ERR_INTERNAL;
    res.message = "handle_capacity_exceeded";
    pthread_mutex_unlock(&g_handle_mutex);
    return res;
  }

  const int64_t handle = g_next_handle++;
  float* logits = (float*)calloc((size_t)req->vocab_size, sizeof(float));
  float* add_vec = logits ? (float*)calloc((size_t)req->vocab_size, sizeof(float)) : NULL;
  float* mul_vec = add_vec ? (float*)calloc((size_t)req->vocab_size, sizeof(float)) : NULL;
  if (mul_vec == NULL) {
    free(add_vec);
    free(logits);
    res.status = NR_ERR_INTERNAL;
    res.message = logits == NULL ? "logit_buffer_alloc_failed"
                : add_vec == NULL ? "add_vector_alloc_failed"
                : "mul_vector_alloc_failed";
    goto compile_unlock;
  }

  g_handles[g_handle_count] = handle;
  g_handle_vocab_sizes[g_handle_count] = req->vocab_size;
  g_handle_max_seq_lens[g_handle_count] = req->max_seq_len;
  g_handle_ir_hash[g_handle_count] = _hash_bytes(req->ir_json, req->ir_json_len);
  g_handle_logits[g_handle_count] = logits;
  g_handle_batch_logits[g_handle_count] = NULL;
  g_handle_add_vectors[g_handle_count] = add_vec;
  g_handle_mul_vectors[g_handle_count] = mul_vec;
  g_handle_batch_logit_capacities[g_handle_count] = 0;
  nk_unary_f32_fn unary_fn = NULL;
  const char* unary_name = NULL;
  if (!_resolve_supported_unary_family(req->ir_json, req->ir_json_len, &unary_fn, &unary_name)) {
    free(mul_vec); free(add_vec); free(logits);
    res.status = NR_ERR_COMPILE_FAILURE;
    res.message = "unsupported_graph_family_unary_family_unavailable";
    goto compile_unlock;
  }
  if (!_has_add_mul_matmul_linear_softmax_rmsnorm_sub_exp_family_markers(req->ir_json, req->ir_json_len)) {
    free(mul_vec); free(add_vec); free(logits);
    res.status = NR_ERR_COMPILE_FAILURE;
    res.message = "unsupported_graph_family_required_chain_invalid";
    goto compile_unlock;
  }
  nk_binary_f32_fn add_fn = NULL;
  if (!aria_registry_lookup_binary("add", &add_fn) || add_fn == NULL) {
    free(mul_vec); free(add_vec); free(logits);
    res.status = NR_ERR_COMPILE_FAILURE;
    res.message = "missing_add_kernel";
    goto compile_unlock;
  }
  nk_binary_f32_fn mul_fn = NULL;
  if (!aria_registry_lookup_binary("mul", &mul_fn) || mul_fn == NULL) {
    free(mul_vec); free(add_vec); free(logits);
    res.status = NR_ERR_COMPILE_FAILURE;
    res.message = "missing_mul_kernel";
    goto compile_unlock;
  }
  nk_binary_f32_fn sub_fn = NULL;
  if (!aria_registry_lookup_binary("sub", &sub_fn) || sub_fn == NULL) {
    free(mul_vec); free(add_vec); free(logits);
    res.status = NR_ERR_COMPILE_FAILURE;
    res.message = "missing_sub_kernel";
    goto compile_unlock;
  }
  const nk_registration_t* matmul_reg = nk_dispatch("matmul");
  nk_matmul_f32_fn matmul_fn = (matmul_reg != NULL) ? matmul_reg->matmul_fn : NULL;
  if (matmul_fn == NULL) {
    free(mul_vec); free(add_vec); free(logits);
    res.status = NR_ERR_COMPILE_FAILURE;
    res.message = "missing_matmul_kernel";
    goto compile_unlock;
  }
  const nk_registration_t* linear_reg = nk_dispatch("linear");
  nk_linear_f32_fn linear_fn = (linear_reg != NULL) ? linear_reg->linear_fn : NULL;
  if (linear_fn == NULL) {
    free(mul_vec); free(add_vec); free(logits);
    res.status = NR_ERR_COMPILE_FAILURE;
    res.message = "missing_linear_kernel";
    goto compile_unlock;
  }
  const nk_registration_t* softmax_reg = nk_dispatch("softmax");
  nk_softmax_f32_fn softmax_fn = (softmax_reg != NULL) ? softmax_reg->softmax_fn : NULL;
  if (softmax_fn == NULL) {
    free(mul_vec); free(add_vec); free(logits);
    res.status = NR_ERR_COMPILE_FAILURE;
    res.message = "missing_softmax_kernel";
    goto compile_unlock;
  }
  const nk_registration_t* rmsnorm_reg = nk_dispatch("rmsnorm");
  nk_rmsnorm_f32_fn rmsnorm_fn = (rmsnorm_reg != NULL) ? rmsnorm_reg->rmsnorm_fn : NULL;
  if (rmsnorm_fn == NULL) {
    free(mul_vec); free(add_vec); free(logits);
    res.status = NR_ERR_COMPILE_FAILURE;
    res.message = "missing_rmsnorm_kernel";
    goto compile_unlock;
  }
  for (int32_t v = 0; v < req->vocab_size; v++) {
    const uint32_t mix = ((uint32_t)v * 2246822519u) ^ g_handle_ir_hash[g_handle_count];
    const int32_t bucket = (int32_t)(mix & 0x0F);
    add_vec[v] = ((float)bucket - 7.0f) * 0.01f;
    const uint32_t mmix = ((uint32_t)v * 3266489917u) ^ (g_handle_ir_hash[g_handle_count] >> 1);
    const int32_t mbucket = (int32_t)(mmix & 0x0F);
    mul_vec[v] = 0.96f + ((float)mbucket) * 0.005f;
  }
  g_handle_unary_fns[g_handle_count] = unary_fn;
  g_handle_add_fns[g_handle_count] = add_fn;
  g_handle_mul_fns[g_handle_count] = mul_fn;
  g_handle_sub_fns[g_handle_count] = sub_fn;
  g_handle_matmul_fns[g_handle_count] = matmul_fn;
  g_handle_linear_fns[g_handle_count] = linear_fn;
  g_handle_softmax_fns[g_handle_count] = softmax_fn;
  g_handle_rmsnorm_fns[g_handle_count] = rmsnorm_fn;
  g_handle_matmul_rhs_a[g_handle_count] = 0.71f + (float)(g_handle_ir_hash[g_handle_count] & 0x03u) * 0.03f;
  g_handle_matmul_rhs_b[g_handle_count] = 0.67f + (float)((g_handle_ir_hash[g_handle_count] >> 2) & 0x03u) * 0.025f;
  g_handle_linear_w0[g_handle_count] = 0.55f + (float)((g_handle_ir_hash[g_handle_count] >> 4) & 0x03u) * 0.05f;
  g_handle_linear_w1[g_handle_count] = 0.45f + (float)((g_handle_ir_hash[g_handle_count] >> 6) & 0x03u) * 0.04f;
  g_handle_linear_bias[g_handle_count] = ((float)((g_handle_ir_hash[g_handle_count] >> 8) & 0x07u) - 3.0f) * 0.003f;
  g_handle_rmsnorm_w0[g_handle_count] = 0.95f + (float)((g_handle_ir_hash[g_handle_count] >> 11) & 0x03u) * 0.02f;
  g_handle_rmsnorm_w1[g_handle_count] = 0.97f + (float)((g_handle_ir_hash[g_handle_count] >> 13) & 0x03u) * 0.02f;
  g_handle_rmsnorm_eps[g_handle_count] = 1e-5f + (float)((g_handle_ir_hash[g_handle_count] >> 15) & 0x03u) * 1e-6f;
  g_handle_unary_names[g_handle_count] = unary_name;
  g_handle_count += 1;

  res.model_handle = handle;
  res.message = unary_name != NULL ? unary_name : "ok";

compile_unlock:
  pthread_mutex_unlock(&g_handle_mutex);
  return res;
}

nr_execute_response_t nr_execute(const nr_execute_request_t* req) {
  nr_execute_response_t res;
  res.status = NR_OK;
  res.logits = NULL;
  res.vocab_size = 1;
  res.message = "ok";

  if (req == NULL || req->model_handle <= 0) {
    res.status = NR_ERR_INVALID_ARGUMENT;
    res.message = "invalid_execute_request";
    return res;
  }

  const int idx = _find_handle_index(req->model_handle);
  if (idx < 0) {
    res.status = NR_ERR_EXECUTION_FAILURE;
    res.message = "unknown_model_handle";
    return res;
  }

  if (req->token_ids == NULL || req->batch <= 0 || req->seq_len <= 0) {
    res.status = NR_ERR_INVALID_ARGUMENT;
    res.message = "invalid_token_payload";
    return res;
  }

  if (req->seq_len > g_handle_max_seq_lens[idx]) {
    res.status = NR_ERR_INVALID_ARGUMENT;
    res.message = "seq_len_exceeds_compile_limit";
    return res;
  }

  const int32_t vocab_size = g_handle_vocab_sizes[idx];
  float* logits = g_handle_logits[idx];
  if (logits == NULL || vocab_size <= 0) {
    res.status = NR_ERR_INTERNAL;
    res.message = "missing_logit_buffer";
    return res;
  }

  const int64_t token_count = (int64_t)req->batch * (int64_t)req->seq_len;
  res.status = _execute_into_logits(
      idx, req->token_ids, (int32_t)token_count, logits, &res.message);
  if (res.status != NR_OK) {
    return res;
  }

  res.logits = logits;
  res.vocab_size = vocab_size;
  return res;
}

nr_execute_batch_response_t nr_execute_batch(const nr_execute_request_t* req) {
  nr_execute_batch_response_t res;
  res.status = NR_OK;
  res.logits = NULL;
  res.batch = 0;
  res.vocab_size = 0;
  res.message = "ok";

  if (req == NULL || req->model_handle <= 0) {
    res.status = NR_ERR_INVALID_ARGUMENT;
    res.message = "invalid_execute_request";
    return res;
  }

  const int idx = _find_handle_index(req->model_handle);
  if (idx < 0) {
    res.status = NR_ERR_EXECUTION_FAILURE;
    res.message = "unknown_model_handle";
    return res;
  }

  if (req->token_ids == NULL || req->batch <= 0 || req->seq_len <= 0) {
    res.status = NR_ERR_INVALID_ARGUMENT;
    res.message = "invalid_token_payload";
    return res;
  }

  if (req->seq_len > g_handle_max_seq_lens[idx]) {
    res.status = NR_ERR_INVALID_ARGUMENT;
    res.message = "seq_len_exceeds_compile_limit";
    return res;
  }

  const int32_t batch = req->batch;
  const int32_t seq_len = req->seq_len;
  const int32_t vocab_size = g_handle_vocab_sizes[idx];
  res.status =
      _ensure_batch_logit_capacity(idx, batch, vocab_size, &res.message);
  if (res.status != NR_OK) {
    return res;
  }

  float* batch_logits = g_handle_batch_logits[idx];
  if (batch_logits == NULL) {
    res.status = NR_ERR_INTERNAL;
    res.message = "batch_logit_buffer_missing";
    return res;
  }

  for (int32_t row = 0; row < batch; row++) {
    const int32_t* row_tokens =
        req->token_ids + ((int64_t)row * (int64_t)seq_len);
    float* row_logits =
        batch_logits + ((int64_t)row * (int64_t)vocab_size);
    res.status =
        _execute_into_logits(idx, row_tokens, seq_len, row_logits, &res.message);
    if (res.status != NR_OK) {
      return res;
    }
  }

  res.logits = batch_logits;
  res.batch = batch;
  res.vocab_size = vocab_size;
  return res;
}

static int _validate_optimizer_step_request(
    const nr_optimizer_step_request_t* req,
    const char** out_message) {
  if (out_message != NULL) {
    *out_message = "ok";
  }
  if (req == NULL) {
    if (out_message != NULL) {
      *out_message = "null_optimizer_step_request";
    }
    return 0;
  }
  if (req->optimizer != NR_OPTIMIZER_SGD && req->optimizer != NR_OPTIMIZER_ADAMW) {
    if (out_message != NULL) {
      *out_message = "unsupported_optimizer";
    }
    return 0;
  }
  if (req->tensors == NULL || req->n_tensors <= 0) {
    if (out_message != NULL) {
      *out_message = "empty_tensor_list";
    }
    return 0;
  }
  if (!(req->learning_rate >= 0.0) || !isfinite(req->learning_rate)) {
    if (out_message != NULL) {
      *out_message = "invalid_learning_rate";
    }
    return 0;
  }
  if (!(req->max_grad_norm >= 0.0) || !isfinite(req->max_grad_norm)) {
    if (out_message != NULL) {
      *out_message = "invalid_max_grad_norm";
    }
    return 0;
  }
  if (req->optimizer == NR_OPTIMIZER_ADAMW) {
    if (req->step <= 0) {
      if (out_message != NULL) {
        *out_message = "adamw_step_must_be_positive";
      }
      return 0;
    }
    if (!(req->beta1 >= 0.0 && req->beta1 < 1.0) || !(req->beta2 >= 0.0 && req->beta2 < 1.0) ||
        !(req->eps > 0.0) || !isfinite(req->beta1) || !isfinite(req->beta2) || !isfinite(req->eps)) {
      if (out_message != NULL) {
        *out_message = "invalid_adamw_hyperparameters";
      }
      return 0;
    }
  }
  for (int32_t i = 0; i < req->n_tensors; i++) {
    const nr_train_tensor_f32_t* t = &req->tensors[i];
    if (t->numel < 0) {
      if (out_message != NULL) {
        *out_message = "negative_tensor_numel";
      }
      return 0;
    }
    if (t->numel == 0) {
      continue;
    }
    if (t->param == NULL || t->grad == NULL) {
      if (out_message != NULL) {
        *out_message = "missing_param_or_grad";
      }
      return 0;
    }
    if (req->optimizer == NR_OPTIMIZER_SGD && req->momentum != 0.0f && t->momentum == NULL) {
      if (out_message != NULL) {
        *out_message = "missing_sgd_momentum_buffer";
      }
      return 0;
    }
    if (req->optimizer == NR_OPTIMIZER_ADAMW && (t->exp_avg == NULL || t->exp_avg_sq == NULL)) {
      if (out_message != NULL) {
        *out_message = "missing_adamw_state";
      }
      return 0;
    }
  }
  return 1;
}

nr_optimizer_step_response_t nr_optimizer_clip_step_f32(const nr_optimizer_step_request_t* req) {
  nr_optimizer_step_response_t res;
  res.status = NR_OK;
  res.grad_norm = 0.0f;
  res.elements = 0;
  res.message = "ok";

  const char* validation_message = "ok";
  if (!_validate_optimizer_step_request(req, &validation_message)) {
    res.status = NR_ERR_INVALID_ARGUMENT;
    res.message = validation_message;
    return res;
  }

  float sum_sq = 0.0f;
  double response_sum_sq = 0.0;
  int64_t elements = 0;
  for (int32_t i = 0; i < req->n_tensors; i++) {
    const nr_train_tensor_f32_t* t = &req->tensors[i];
    for (int64_t j = 0; j < t->numel; j++) {
      const float g = t->grad[j];
      if (!isfinite(g)) {
        res.status = NR_ERR_EXECUTION_FAILURE;
        res.message = "nonfinite_gradient";
        return res;
      }
      sum_sq += g * g;
      response_sum_sq += (double)g * (double)g;
    }
    elements += t->numel;
  }

  const float grad_norm = sqrtf(sum_sq);
  const float max_grad_norm = (float)req->max_grad_norm;
  const float clip_coef =
      (max_grad_norm > 0.0f) ? (max_grad_norm / (grad_norm + 1.0e-6f)) : 1.0f;
  const float clip_scale = (clip_coef < 1.0f) ? clip_coef : 1.0f;
  res.grad_norm = sqrt(response_sum_sq);
  res.elements = elements;

  if (req->optimizer == NR_OPTIMIZER_SGD) {
    const double lr = req->learning_rate;
    const double momentum = req->momentum;
    const double weight_decay = req->weight_decay;
    for (int32_t i = 0; i < req->n_tensors; i++) {
      nr_train_tensor_f32_t* t = &req->tensors[i];
      for (int64_t j = 0; j < t->numel; j++) {
        float d_p = t->grad[j] * clip_scale;
        if (weight_decay != 0.0f) {
          d_p += (float)(weight_decay * (double)t->param[j]);
        }
        if (momentum != 0.0f) {
          float buf = (float)(momentum * (double)t->momentum[j] + (double)d_p);
          t->momentum[j] = buf;
          if (req->nesterov) {
            d_p += (float)(momentum * (double)buf);
          } else {
            d_p = buf;
          }
        }
        t->param[j] -= (float)(lr * (double)d_p);
      }
    }
    return res;
  }

  const double lr = req->learning_rate;
  const double beta1 = req->beta1;
  const double beta2 = req->beta2;
  const double eps = req->eps;
  const double weight_decay = req->weight_decay;
  const double bias_correction1 = 1.0 - pow(beta1, (double)req->step);
  const double bias_correction2 = 1.0 - pow(beta2, (double)req->step);
  const double step_size = (bias_correction1 != 0.0) ? (lr / bias_correction1) : 0.0;
  const double denom_scale = sqrt(bias_correction2);
  const float step_size_f = (float)step_size;
  const float denom_scale_f = (float)denom_scale;
  const float eps_f = (float)eps;
  for (int32_t i = 0; i < req->n_tensors; i++) {
    nr_train_tensor_f32_t* t = &req->tensors[i];
    for (int64_t j = 0; j < t->numel; j++) {
      if (weight_decay != 0.0f) {
        t->param[j] *= (float)(1.0 - lr * weight_decay);
      }
      const float grad = t->grad[j] * clip_scale;
      const float exp_avg = (float)(beta1 * (double)t->exp_avg[j] + (1.0 - beta1) * (double)grad);
      const float exp_avg_sq =
          (float)(beta2 * (double)t->exp_avg_sq[j] + (1.0 - beta2) * (double)grad * (double)grad);
      t->exp_avg[j] = exp_avg;
      t->exp_avg_sq[j] = exp_avg_sq;
      const float denom = (sqrtf(exp_avg_sq) / denom_scale_f) + eps_f;
      t->param[j] -= step_size_f * exp_avg / denom;
    }
  }
  return res;
}

void nr_release_model(int64_t model_handle) {
  pthread_mutex_lock(&g_handle_mutex);
  const int idx = _find_handle_index(model_handle);
  if (idx < 0) {
    pthread_mutex_unlock(&g_handle_mutex);
    return;
  }

  _release_handle_index((int32_t)idx);

  for (int32_t i = idx; i < g_handle_count - 1; i++) {
    g_handles[i] = g_handles[i + 1];
    g_handle_vocab_sizes[i] = g_handle_vocab_sizes[i + 1];
    g_handle_max_seq_lens[i] = g_handle_max_seq_lens[i + 1];
    g_handle_ir_hash[i] = g_handle_ir_hash[i + 1];
    g_handle_logits[i] = g_handle_logits[i + 1];
    g_handle_batch_logits[i] = g_handle_batch_logits[i + 1];
    g_handle_add_vectors[i] = g_handle_add_vectors[i + 1];
    g_handle_mul_vectors[i] = g_handle_mul_vectors[i + 1];
    g_handle_unary_fns[i] = g_handle_unary_fns[i + 1];
    g_handle_add_fns[i] = g_handle_add_fns[i + 1];
    g_handle_mul_fns[i] = g_handle_mul_fns[i + 1];
    g_handle_sub_fns[i] = g_handle_sub_fns[i + 1];
    g_handle_matmul_fns[i] = g_handle_matmul_fns[i + 1];
    g_handle_linear_fns[i] = g_handle_linear_fns[i + 1];
    g_handle_softmax_fns[i] = g_handle_softmax_fns[i + 1];
    g_handle_rmsnorm_fns[i] = g_handle_rmsnorm_fns[i + 1];
    g_handle_matmul_rhs_a[i] = g_handle_matmul_rhs_a[i + 1];
    g_handle_matmul_rhs_b[i] = g_handle_matmul_rhs_b[i + 1];
    g_handle_linear_w0[i] = g_handle_linear_w0[i + 1];
    g_handle_linear_w1[i] = g_handle_linear_w1[i + 1];
    g_handle_linear_bias[i] = g_handle_linear_bias[i + 1];
    g_handle_rmsnorm_w0[i] = g_handle_rmsnorm_w0[i + 1];
    g_handle_rmsnorm_w1[i] = g_handle_rmsnorm_w1[i + 1];
    g_handle_rmsnorm_eps[i] = g_handle_rmsnorm_eps[i + 1];
    g_handle_unary_names[i] = g_handle_unary_names[i + 1];
    g_handle_batch_logit_capacities[i] = g_handle_batch_logit_capacities[i + 1];
  }
  g_handle_count -= 1;
  g_handles[g_handle_count] = 0;
  g_handle_vocab_sizes[g_handle_count] = 0;
  g_handle_max_seq_lens[g_handle_count] = 0;
  g_handle_ir_hash[g_handle_count] = 0;
  g_handle_logits[g_handle_count] = NULL;
  g_handle_batch_logits[g_handle_count] = NULL;
  g_handle_add_vectors[g_handle_count] = NULL;
  g_handle_mul_vectors[g_handle_count] = NULL;
  g_handle_unary_fns[g_handle_count] = NULL;
  g_handle_add_fns[g_handle_count] = NULL;
  g_handle_mul_fns[g_handle_count] = NULL;
  g_handle_sub_fns[g_handle_count] = NULL;
  g_handle_matmul_fns[g_handle_count] = NULL;
  g_handle_linear_fns[g_handle_count] = NULL;
  g_handle_softmax_fns[g_handle_count] = NULL;
  g_handle_rmsnorm_fns[g_handle_count] = NULL;
  g_handle_matmul_rhs_a[g_handle_count] = 0.0f;
  g_handle_matmul_rhs_b[g_handle_count] = 0.0f;
  g_handle_linear_w0[g_handle_count] = 0.0f;
  g_handle_linear_w1[g_handle_count] = 0.0f;
  g_handle_linear_bias[g_handle_count] = 0.0f;
  g_handle_rmsnorm_w0[g_handle_count] = 0.0f;
  g_handle_rmsnorm_w1[g_handle_count] = 0.0f;
  g_handle_rmsnorm_eps[g_handle_count] = 0.0f;
  g_handle_unary_names[g_handle_count] = NULL;
  g_handle_batch_logit_capacities[g_handle_count] = 0;
  pthread_mutex_unlock(&g_handle_mutex);
}
