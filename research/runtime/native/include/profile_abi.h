#ifndef ARIA_RESEARCH_PROFILE_ABI_H
#define ARIA_RESEARCH_PROFILE_ABI_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  const char* event_name;
  const char* op_name;
  int32_t node_id;
  int64_t start_ns;
  int64_t end_ns;
  int32_t thread_id;
} np_event_t;

typedef void (*np_event_sink_fn)(const np_event_t* evt, void* user_data);

void np_set_event_sink(np_event_sink_fn sink, void* user_data);
void np_emit_event(const np_event_t* evt);

/* --------------- memory profiling --------------- */

typedef struct {
  const char* tag;
  int64_t bytes_allocated;
  int64_t bytes_freed;
  int64_t peak_bytes;
  int64_t timestamp_ns;
} np_memory_event_t;

typedef void (*np_memory_sink_fn)(const np_memory_event_t* evt, void* user_data);

void np_emit_memory_event(const np_memory_event_t* evt);
void np_set_memory_sink(np_memory_sink_fn sink, void* user_data);
int64_t np_get_peak_memory(void);
void np_reset_counters(void);

/* --------------- profiler control --------------- */

void np_profiler_enable(int enable);
int  np_profiler_enabled(void);
int64_t np_clock_ns(void);

/* --------------- ring buffer drain --------------- */

int32_t np_event_count(void);
int32_t np_drain_events(np_event_t* out, int32_t max_out);
int32_t np_memory_event_count(void);
int32_t np_drain_memory_events(np_memory_event_t* out, int32_t max_out);

#ifdef __cplusplus
}
#endif

#endif
