#![cfg(feature = "python")]

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use std::collections::HashMap;

use crate::arena::Arena;
use crate::ffi;
use crate::graph::GraphIR;
use crate::executor::{
    execute_with_arena, execute_forward_saving_activations,
    execute_backward_with_arena, NativeKernelDispatch,
};

/// Parse a graph IR JSON string, validate it, and return the node count.
#[pyfunction]
fn parse_graph_ir(json: &str) -> PyResult<String> {
    let graph = GraphIR::from_json(json)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    graph
        .validate()
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    Ok(format!("{} nodes", graph.nodes.len()))
}

/// Return the topological execution order as a list of node id integers.
#[pyfunction]
fn topological_order(json: &str) -> PyResult<Vec<u32>> {
    let graph = GraphIR::from_json(json)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    let order = graph
        .topological_order()
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    Ok(order.into_iter().map(|id| id.0).collect())
}

/// Execute a graph IR using the native kernel library.
///
/// Returns the output tensor as a list of floats (backward-compatible API).
/// Use `execute_graph_with_stats` to also get arena memory statistics.
#[pyfunction]
fn execute_graph(json: &str, input: Vec<f32>) -> PyResult<Vec<f32>> {
    let graph = GraphIR::from_json(json)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    let result = execute_with_arena(&graph, &NativeKernelDispatch, &input)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
    Ok(result.output)
}

/// Execute a graph IR using the native kernel library, returning arena stats.
///
/// Returns a dict with keys:
///   - "output": list[float]  -- the output tensor
///   - "arena_bytes_used": int
///   - "arena_capacity": int
///   - "arena_alloc_count": int
///   - "heap_fallback_count": int
#[pyfunction]
fn execute_graph_with_stats(py: Python<'_>, json: &str, input: Vec<f32>) -> PyResult<PyObject> {
    let graph = GraphIR::from_json(json)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    let result = execute_with_arena(&graph, &NativeKernelDispatch, &input)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

    let dict = PyDict::new_bound(py);
    dict.set_item("output", result.output)?;
    dict.set_item("arena_bytes_used", result.arena_stats.arena_bytes_used)?;
    dict.set_item("arena_capacity", result.arena_stats.arena_capacity)?;
    dict.set_item("arena_alloc_count", result.arena_stats.arena_alloc_count)?;
    dict.set_item("heap_fallback_count", result.arena_stats.heap_fallback_count)?;

    // Include profiling data when available.
    if !result.node_profiles.is_empty() {
        let profiles = PyList::empty_bound(py);
        for np in &result.node_profiles {
            let d = PyDict::new_bound(py);
            d.set_item("node_id", np.node_id)?;
            d.set_item("op_name", &np.op_name)?;
            d.set_item("start_ns", np.start_ns)?;
            d.set_item("end_ns", np.end_ns)?;
            d.set_item("duration_us", np.duration_us)?;
            profiles.append(d)?;
        }
        dict.set_item("node_profiles", profiles)?;
        dict.set_item("peak_memory_bytes", result.peak_memory_bytes)?;
    }

    Ok(dict.into())
}

/// Smoke-test the arena allocator: allocate `size_mb` megabytes, report
/// (peak_bytes, used_bytes_after_reset).
#[pyfunction]
fn arena_test(size_mb: usize) -> PyResult<(usize, usize)> {
    let capacity = size_mb * 1024 * 1024;
    let mut arena = Arena::new(capacity);
    let float_count = capacity / 4; // f32 = 4 bytes
    arena
        .alloc_f32(float_count)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
    let peak = arena.peak_bytes();
    arena.reset();
    let used = arena.used_bytes();
    Ok((peak, used))
}

/// Enable or disable the native profiler.
///
/// When enabled, `execute_graph_with_stats` will include per-node timing
/// data in its return dict under the "node_profiles" key.
#[pyfunction]
fn profiler_enable(enable: bool) {
    unsafe { ffi::np_profiler_enable(if enable { 1 } else { 0 }) };
}

/// Check whether the native profiler is currently enabled.
#[pyfunction]
fn profiler_enabled() -> bool {
    unsafe { ffi::np_profiler_enabled() != 0 }
}

/// Reset all profiler counters and ring buffers.
#[pyfunction]
fn profiler_reset() {
    unsafe { ffi::np_reset_counters() };
}

/// Execute the forward pass and return saved activations alongside the output.
///
/// Returns a dict with keys:
///   - "output": list[float]
///   - "saved_activations": dict[int, list[float]]
///   - "arena_bytes_used": int
///   - "arena_capacity": int
#[pyfunction]
fn execute_graph_forward_saved(py: Python<'_>, json: &str, input: Vec<f32>) -> PyResult<PyObject> {
    let graph = GraphIR::from_json(json)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    let result = execute_forward_saving_activations(&graph, &NativeKernelDispatch, &input)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

    let dict = PyDict::new_bound(py);
    dict.set_item("output", result.output)?;

    let saved_dict = PyDict::new_bound(py);
    for (node_id, activation) in &result.saved_activations {
        saved_dict.set_item(*node_id, activation.clone())?;
    }
    dict.set_item("saved_activations", saved_dict)?;
    dict.set_item("arena_bytes_used", result.arena_stats.arena_bytes_used)?;
    dict.set_item("arena_capacity", result.arena_stats.arena_capacity)?;

    Ok(dict.into())
}

/// Execute the backward pass through a graph.
///
/// Args:
///   json: Graph IR JSON string.
///   grad_output: Gradient w.r.t. the graph output (list[float]).
///   saved_activations: dict[int, list[float]] from execute_graph_forward_saved.
///
/// Returns a dict with keys:
///   - "grads": dict[int, list[float]]  -- gradient for each node
///   - "arena_bytes_used": int
#[pyfunction]
fn execute_graph_backward(
    py: Python<'_>,
    json: &str,
    grad_output: Vec<f32>,
    saved_activations: &Bound<'_, PyDict>,
) -> PyResult<PyObject> {
    let graph = GraphIR::from_json(json)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

    // Convert Python dict to HashMap<u32, Vec<f32>>.
    let mut saved: HashMap<u32, Vec<f32>> = HashMap::new();
    for (key, value) in saved_activations.iter() {
        let node_id: u32 = key.extract()?;
        let activation: Vec<f32> = value.extract()?;
        saved.insert(node_id, activation);
    }

    let result = execute_backward_with_arena(&graph, &grad_output, &saved)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

    let dict = PyDict::new_bound(py);
    let grads_dict = PyDict::new_bound(py);
    for (node_id, grad) in &result.grads {
        grads_dict.set_item(*node_id, grad.clone())?;
    }
    dict.set_item("grads", grads_dict)?;
    dict.set_item("arena_bytes_used", result.arena_stats.arena_bytes_used)?;

    Ok(dict.into())
}

/// The Python module definition.
#[pymodule]
fn aria_scheduler(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(parse_graph_ir, m)?)?;
    m.add_function(wrap_pyfunction!(topological_order, m)?)?;
    m.add_function(wrap_pyfunction!(execute_graph, m)?)?;
    m.add_function(wrap_pyfunction!(execute_graph_with_stats, m)?)?;
    m.add_function(wrap_pyfunction!(execute_graph_forward_saved, m)?)?;
    m.add_function(wrap_pyfunction!(execute_graph_backward, m)?)?;
    m.add_function(wrap_pyfunction!(arena_test, m)?)?;
    m.add_function(wrap_pyfunction!(profiler_enable, m)?)?;
    m.add_function(wrap_pyfunction!(profiler_enabled, m)?)?;
    m.add_function(wrap_pyfunction!(profiler_reset, m)?)?;
    Ok(())
}
