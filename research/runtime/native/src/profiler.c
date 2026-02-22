/**
 * profiler.c — Implementation of profile_abi.h
 *
 * Ring buffer of profiling events with opt-in activation.
 * Zero overhead when disabled: all public functions check the global enable
 * flag first and return immediately when profiling is off.
 *
 * Enable at runtime: call np_profiler_enable(1) or set env NATIVE_RUNNER_PROFILE=1.
 */

#include "../include/profile_abi.h"
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <pthread.h>

/* ---------- Ring buffer configuration ---------- */

#define NP_RING_CAPACITY 8192

static np_event_t       g_event_ring[NP_RING_CAPACITY];
static int32_t          g_ring_head  = 0;   /* next write slot */
static int32_t          g_ring_count = 0;   /* number of valid entries (<=CAPACITY) */

static np_memory_event_t g_mem_ring[NP_RING_CAPACITY];
static int32_t           g_mem_head  = 0;
static int32_t           g_mem_count = 0;

/* ---------- Global state ---------- */

static int              g_profiler_enabled = -1;  /* -1 = not yet checked */
static np_event_sink_fn g_event_sink       = NULL;
static void*            g_event_sink_data  = NULL;
static np_memory_sink_fn g_memory_sink     = NULL;
static void*            g_memory_sink_data = NULL;

static int64_t g_mem_allocated = 0;
static int64_t g_mem_freed     = 0;
static int64_t g_mem_peak      = 0;

static pthread_mutex_t  g_lock = PTHREAD_MUTEX_INITIALIZER;

/* ---------- Helpers ---------- */

static int profiler_is_enabled(void) {
    if (g_profiler_enabled == -1) {
        const char *env = getenv("NATIVE_RUNNER_PROFILE");
        g_profiler_enabled = (env && (env[0] == '1'));
    }
    return g_profiler_enabled;
}

int64_t np_clock_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (int64_t)ts.tv_sec * 1000000000LL + (int64_t)ts.tv_nsec;
}

/* ---------- Public API: enable/disable ---------- */

void np_profiler_enable(int enable) {
    g_profiler_enabled = enable ? 1 : 0;
}

int np_profiler_enabled(void) {
    return profiler_is_enabled();
}

/* ---------- Event sink ---------- */

void np_set_event_sink(np_event_sink_fn sink, void* user_data) {
    g_event_sink      = sink;
    g_event_sink_data = user_data;
}

void np_emit_event(const np_event_t* evt) {
    if (!profiler_is_enabled()) return;
    if (!evt) return;

    /* Forward to user sink if registered */
    if (g_event_sink) {
        g_event_sink(evt, g_event_sink_data);
    }

    /* Write into ring buffer with deep-copied strings */
    pthread_mutex_lock(&g_lock);
    int idx = g_ring_head;
    /* Free old strings if slot is being reused */
    if (g_ring_count >= NP_RING_CAPACITY) {
        free((void*)g_event_ring[idx].event_name);
        free((void*)g_event_ring[idx].op_name);
    }
    g_event_ring[idx] = *evt;
    g_event_ring[idx].event_name = evt->event_name ? strdup(evt->event_name) : NULL;
    g_event_ring[idx].op_name    = evt->op_name    ? strdup(evt->op_name)    : NULL;
    g_ring_head = (g_ring_head + 1) % NP_RING_CAPACITY;
    if (g_ring_count < NP_RING_CAPACITY) g_ring_count++;
    pthread_mutex_unlock(&g_lock);
}

/* ---------- Memory profiling ---------- */

void np_set_memory_sink(np_memory_sink_fn sink, void* user_data) {
    g_memory_sink      = sink;
    g_memory_sink_data = user_data;
}

void np_emit_memory_event(const np_memory_event_t* evt) {
    if (!profiler_is_enabled()) return;
    if (!evt) return;

    if (g_memory_sink) {
        g_memory_sink(evt, g_memory_sink_data);
    }

    pthread_mutex_lock(&g_lock);

    /* Update counters */
    g_mem_allocated += evt->bytes_allocated;
    g_mem_freed     += evt->bytes_freed;
    int64_t current  = g_mem_allocated - g_mem_freed;
    if (current > g_mem_peak) g_mem_peak = current;

    /* Write into memory ring with deep-copied tag */
    int idx = g_mem_head;
    if (g_mem_count >= NP_RING_CAPACITY) {
        free((void*)g_mem_ring[idx].tag);
    }
    g_mem_ring[idx] = *evt;
    g_mem_ring[idx].tag = evt->tag ? strdup(evt->tag) : NULL;
    /* Backfill peak_bytes with the current peak */
    g_mem_ring[idx].peak_bytes = g_mem_peak;
    g_mem_head = (g_mem_head + 1) % NP_RING_CAPACITY;
    if (g_mem_count < NP_RING_CAPACITY) g_mem_count++;

    pthread_mutex_unlock(&g_lock);
}

int64_t np_get_peak_memory(void) {
    return g_mem_peak;
}

void np_reset_counters(void) {
    pthread_mutex_lock(&g_lock);
    /* Free deep-copied strings in event ring */
    for (int32_t i = 0; i < g_ring_count; i++) {
        int32_t idx = (g_ring_head - g_ring_count + i + NP_RING_CAPACITY) % NP_RING_CAPACITY;
        free((void*)g_event_ring[idx].event_name);
        free((void*)g_event_ring[idx].op_name);
        g_event_ring[idx].event_name = NULL;
        g_event_ring[idx].op_name    = NULL;
    }
    /* Free deep-copied strings in memory ring */
    for (int32_t i = 0; i < g_mem_count; i++) {
        int32_t idx = (g_mem_head - g_mem_count + i + NP_RING_CAPACITY) % NP_RING_CAPACITY;
        free((void*)g_mem_ring[idx].tag);
        g_mem_ring[idx].tag = NULL;
    }
    g_ring_head  = 0;
    g_ring_count = 0;
    g_mem_head   = 0;
    g_mem_count  = 0;
    g_mem_allocated = 0;
    g_mem_freed     = 0;
    g_mem_peak      = 0;
    pthread_mutex_unlock(&g_lock);
}

/* ---------- Query API (used by Rust FFI) ---------- */

int32_t np_event_count(void) {
    return g_ring_count;
}

/**
 * Copy up to `max_out` events into `out`, returning the number copied.
 * Events are returned oldest-first.
 */
int32_t np_drain_events(np_event_t* out, int32_t max_out) {
    if (!out || max_out <= 0) return 0;

    pthread_mutex_lock(&g_lock);
    int32_t n = g_ring_count < max_out ? g_ring_count : max_out;
    /* Oldest event is at (head - count) mod CAPACITY */
    int32_t start = (g_ring_head - g_ring_count + NP_RING_CAPACITY) % NP_RING_CAPACITY;
    for (int32_t i = 0; i < n; i++) {
        out[i] = g_event_ring[(start + i) % NP_RING_CAPACITY];
    }
    /* Reset after drain */
    g_ring_head  = 0;
    g_ring_count = 0;
    pthread_mutex_unlock(&g_lock);
    return n;
}

int32_t np_memory_event_count(void) {
    return g_mem_count;
}

int32_t np_drain_memory_events(np_memory_event_t* out, int32_t max_out) {
    if (!out || max_out <= 0) return 0;

    pthread_mutex_lock(&g_lock);
    int32_t n = g_mem_count < max_out ? g_mem_count : max_out;
    int32_t start = (g_mem_head - g_mem_count + NP_RING_CAPACITY) % NP_RING_CAPACITY;
    for (int32_t i = 0; i < n; i++) {
        out[i] = g_mem_ring[(start + i) % NP_RING_CAPACITY];
    }
    g_mem_head  = 0;
    g_mem_count = 0;
    pthread_mutex_unlock(&g_lock);
    return n;
}
