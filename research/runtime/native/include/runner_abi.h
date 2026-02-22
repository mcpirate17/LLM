#ifndef ARIA_RESEARCH_RUNNER_ABI_H
#define ARIA_RESEARCH_RUNNER_ABI_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
  NR_OK = 0,
  NR_ERR_INVALID_ARGUMENT = -1,
  NR_ERR_UNSUPPORTED_IR = -2,
  NR_ERR_COMPILE_FAILURE = -3,
  NR_ERR_EXECUTION_FAILURE = -4,
  NR_ERR_INTERNAL = -5,
  NR_ERR_STRICT_UNSUPPORTED = -6
} nr_status_t;

typedef struct {
  const char* ir_json;
  int64_t ir_json_len;
  int32_t vocab_size;
  int32_t max_seq_len;
} nr_compile_request_t;

typedef struct {
  nr_status_t status;
  int64_t model_handle;
  const char* message;
} nr_compile_response_t;

typedef struct {
  int64_t model_handle;
  const int32_t* token_ids;
  int32_t batch;
  int32_t seq_len;
} nr_execute_request_t;

typedef struct {
  nr_status_t status;
  const float* logits;
  int32_t vocab_size;
  const char* message;
} nr_execute_response_t;

/* --------------- capability query --------------- */

typedef struct {
  const char** supported_ops;
  int32_t n_supported;
  const char** unsupported_ops;
  int32_t n_unsupported;
} nr_capability_t;

nr_status_t nr_query_capabilities(nr_capability_t* out);

/* --------------- strict mode & telemetry --------------- */

nr_status_t nr_set_strict_mode(int32_t strict);
int64_t nr_get_fallback_count(void);

/* --------------- lifecycle --------------- */

nr_status_t nr_runtime_init(void);
void nr_runtime_shutdown(void);
nr_compile_response_t nr_compile(const nr_compile_request_t* req);
nr_execute_response_t nr_execute(const nr_execute_request_t* req);
void nr_release_model(int64_t model_handle);

#ifdef __cplusplus
}
#endif

#endif
