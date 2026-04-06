#ifndef RESEARCH_RUNTIME_NATIVE_INTELLIGENT_ROUTER_ABI_H_
#define RESEARCH_RUNTIME_NATIVE_INTELLIGENT_ROUTER_ABI_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
  ARIA_IROUTER_OK = 0,
  ARIA_IROUTER_ERR_INVALID_ARGUMENT = -1,
  ARIA_IROUTER_ERR_NOT_FOUND = -2,
  ARIA_IROUTER_ERR_INTERNAL = -3,
} aria_irouter_status_t;

typedef struct {
  int32_t span_count;
  int32_t required_span_capacity;
} aria_irouter_route_meta_t;

aria_irouter_status_t aria_irouter_create(
    int32_t vocab,
    int32_t lanes,
    int64_t* out_handle);

aria_irouter_status_t aria_irouter_destroy(int64_t handle);

aria_irouter_status_t aria_irouter_train_token_gate(
    int64_t handle,
    int32_t token,
    int32_t keep,
    float strength);

aria_irouter_status_t aria_irouter_train_span_router(
    int64_t handle,
    const int32_t* sequence,
    int32_t seq_len,
    int32_t lane,
    float strength);

aria_irouter_status_t aria_irouter_route(
    int64_t handle,
    const int32_t* sequence,
    int32_t seq_len,
    int32_t* token_actions_out,
    float* token_keep_probability_out,
    int32_t* span_token_indices_out,
    int32_t span_token_indices_capacity,
    int32_t* span_lanes_out,
    float* span_confidences_out,
    aria_irouter_route_meta_t* out_meta);

aria_irouter_status_t aria_irouter_save(int64_t handle, const char* path);

aria_irouter_status_t aria_irouter_load(const char* path, int64_t* out_handle);

const char* aria_irouter_last_error(void);

#ifdef __cplusplus
}
#endif

#endif
