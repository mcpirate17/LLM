/**
 * profiler.h — Internal helpers for instrumenting kernel dispatch.
 *
 * The profiled_dispatch_* functions wrap kernel function pointers with
 * timing events. They are no-ops when profiling is disabled.
 */

#ifndef ARIA_PROFILER_H
#define ARIA_PROFILER_H

#include "../include/profile_abi.h"
#include "../include/kernel_abi.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Wrap a nk_dispatch() call with profiling events.
 * Returns the same registration pointer as nk_dispatch().
 * When profiling is enabled, emits start/end events for the lookup itself.
 *
 * Instead of wrapping nk_dispatch, we provide helper macros/functions
 * that callers can use around their kernel invocations.
 */

/** Begin a profiled kernel span. Returns start_ns (0 if profiling off). */
static inline int64_t np_kernel_begin(const char* op_name, int32_t node_id) {
    if (!np_profiler_enabled()) return 0;
    (void)op_name; (void)node_id;
    return np_clock_ns();
}

/** End a profiled kernel span. No-op if start_ns == 0. */
static inline void np_kernel_end(const char* op_name, int32_t node_id, int64_t start_ns) {
    if (start_ns == 0) return;
    np_event_t evt;
    evt.event_name = "kernel";
    evt.op_name    = op_name;
    evt.node_id    = node_id;
    evt.start_ns   = start_ns;
    evt.end_ns     = np_clock_ns();
    evt.thread_id  = 0;  /* single-threaded for now */
    np_emit_event(&evt);
}

#ifdef __cplusplus
}
#endif

#endif /* ARIA_PROFILER_H */
