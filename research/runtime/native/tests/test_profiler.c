/**
 * test_profiler.c — Tests for the profiling subsystem (profile_abi.h).
 *
 * Validates:
 *  1. Profiler enable/disable controls
 *  2. Event emission and ring buffer drain
 *  3. Memory event tracking and peak calculation
 *  4. Reset clears all state
 *  5. Zero-overhead when disabled
 */
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include "../include/profile_abi.h"

static void test_profiler_default_disabled(void) {
    /* Without NATIVE_RUNNER_PROFILE=1 env var, profiler should be disabled
       after a fresh enable(0) call. */
    np_profiler_enable(0);
    assert(np_profiler_enabled() == 0);
    printf("  PASS: profiler default disabled\n");
}

static void test_profiler_enable_disable(void) {
    np_profiler_enable(1);
    assert(np_profiler_enabled() != 0);
    np_profiler_enable(0);
    assert(np_profiler_enabled() == 0);
    printf("  PASS: profiler enable/disable\n");
}

static void test_event_emission(void) {
    np_profiler_enable(1);
    np_reset_counters();

    np_event_t evt;
    evt.event_name = "kernel";
    evt.op_name    = "relu";
    evt.node_id    = 42;
    evt.start_ns   = 1000;
    evt.end_ns     = 2000;
    evt.thread_id  = 0;
    np_emit_event(&evt);

    assert(np_event_count() == 1);

    /* Drain and verify */
    np_event_t out[8];
    int32_t n = np_drain_events(out, 8);
    assert(n == 1);
    assert(out[0].node_id == 42);
    assert(out[0].start_ns == 1000);
    assert(out[0].end_ns == 2000);
    assert(strcmp(out[0].op_name, "relu") == 0);

    /* After drain, count should be 0 */
    assert(np_event_count() == 0);

    np_profiler_enable(0);
    printf("  PASS: event emission and drain\n");
}

static void test_multiple_events_ordering(void) {
    np_profiler_enable(1);
    np_reset_counters();

    for (int i = 0; i < 5; i++) {
        np_event_t evt;
        evt.event_name = "kernel";
        evt.op_name    = "add";
        evt.node_id    = i;
        evt.start_ns   = i * 100;
        evt.end_ns     = i * 100 + 50;
        evt.thread_id  = 0;
        np_emit_event(&evt);
    }

    assert(np_event_count() == 5);

    np_event_t out[10];
    int32_t n = np_drain_events(out, 10);
    assert(n == 5);

    /* Verify oldest-first ordering */
    for (int i = 0; i < 5; i++) {
        assert(out[i].node_id == i);
        assert(out[i].start_ns == i * 100);
    }

    np_profiler_enable(0);
    printf("  PASS: multiple events ordering\n");
}

static void test_no_events_when_disabled(void) {
    np_profiler_enable(0);
    np_reset_counters();

    np_event_t evt;
    evt.event_name = "kernel";
    evt.op_name    = "relu";
    evt.node_id    = 99;
    evt.start_ns   = 5000;
    evt.end_ns     = 6000;
    evt.thread_id  = 0;
    np_emit_event(&evt);

    /* Should not have recorded anything */
    assert(np_event_count() == 0);
    printf("  PASS: no events when disabled\n");
}

static void test_memory_events(void) {
    np_profiler_enable(1);
    np_reset_counters();

    np_memory_event_t mevt;
    mevt.tag = "arena_alloc";
    mevt.bytes_allocated = 4096;
    mevt.bytes_freed     = 0;
    mevt.peak_bytes      = 0;  /* will be filled by profiler */
    mevt.timestamp_ns    = np_clock_ns();
    np_emit_memory_event(&mevt);

    assert(np_get_peak_memory() == 4096);

    /* Allocate more */
    mevt.bytes_allocated = 8192;
    mevt.bytes_freed     = 0;
    mevt.timestamp_ns    = np_clock_ns();
    np_emit_memory_event(&mevt);
    assert(np_get_peak_memory() == 4096 + 8192);

    /* Free some */
    mevt.bytes_allocated = 0;
    mevt.bytes_freed     = 4096;
    mevt.timestamp_ns    = np_clock_ns();
    np_emit_memory_event(&mevt);
    /* Peak should still be 12288 (previous high water mark) */
    assert(np_get_peak_memory() == 12288);

    assert(np_memory_event_count() == 3);

    /* Drain */
    np_memory_event_t mout[8];
    int32_t mn = np_drain_memory_events(mout, 8);
    assert(mn == 3);
    assert(mout[0].bytes_allocated == 4096);
    assert(mout[2].bytes_freed == 4096);

    np_profiler_enable(0);
    printf("  PASS: memory events\n");
}

static void test_reset_clears_all(void) {
    np_profiler_enable(1);

    np_event_t evt;
    evt.event_name = "kernel";
    evt.op_name    = "relu";
    evt.node_id    = 1;
    evt.start_ns   = 100;
    evt.end_ns     = 200;
    evt.thread_id  = 0;
    np_emit_event(&evt);

    np_memory_event_t mevt;
    mevt.tag = "test";
    mevt.bytes_allocated = 1024;
    mevt.bytes_freed     = 0;
    mevt.peak_bytes      = 0;
    mevt.timestamp_ns    = 100;
    np_emit_memory_event(&mevt);

    assert(np_event_count() > 0);
    assert(np_memory_event_count() > 0);
    assert(np_get_peak_memory() > 0);

    np_reset_counters();

    assert(np_event_count() == 0);
    assert(np_memory_event_count() == 0);
    assert(np_get_peak_memory() == 0);

    np_profiler_enable(0);
    printf("  PASS: reset clears all\n");
}

static void test_clock_ns_monotonic(void) {
    int64_t t1 = np_clock_ns();
    int64_t t2 = np_clock_ns();
    assert(t2 >= t1);
    assert(t1 > 0);
    printf("  PASS: clock_ns monotonic\n");
}

/* Custom sink to verify callback is called */
static int g_sink_called = 0;
static void test_sink_fn(const np_event_t* evt, void* user_data) {
    (void)evt;
    (void)user_data;
    g_sink_called++;
}

static void test_event_sink_callback(void) {
    np_profiler_enable(1);
    np_reset_counters();
    g_sink_called = 0;

    np_set_event_sink(test_sink_fn, NULL);

    np_event_t evt;
    evt.event_name = "kernel";
    evt.op_name    = "gelu";
    evt.node_id    = 7;
    evt.start_ns   = 500;
    evt.end_ns     = 600;
    evt.thread_id  = 0;
    np_emit_event(&evt);

    assert(g_sink_called == 1);

    /* Clean up */
    np_set_event_sink(NULL, NULL);
    np_profiler_enable(0);
    printf("  PASS: event sink callback\n");
}

int main(void) {
    printf("Running profiler tests...\n");
    test_profiler_default_disabled();
    test_profiler_enable_disable();
    test_event_emission();
    test_multiple_events_ordering();
    test_no_events_when_disabled();
    test_memory_events();
    test_reset_clears_all();
    test_clock_ns_monotonic();
    test_event_sink_callback();
    printf("\nAll 9 profiler tests passed.\n");
    return 0;
}
