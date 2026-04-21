#![cfg(feature = "python")]

use numpy::{
    Element, IntoPyArray, PyArray1, PyArrayMethods, PyReadonlyArrayDyn, PyUntypedArrayMethods,
};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use serde::{Deserialize, Serialize};

use std::collections::HashMap;
use std::path::Path;

use crate::arena::Arena;
use crate::corpus::{
    build_graph_training_corpus_json, build_predictor_training_corpus_json,
    fingerprint_notebook_graph_json,
};
use crate::executor::{
    execute_backward_with_arena, execute_backward_with_arena_slices,
    execute_forward_saving_activations,
    execute_forward_saving_activations_multi_input, execute_with_arena,
    execute_with_arena_multi_input, NativeKernelDispatch,
};
use crate::ffi;
use crate::graph::GraphIR;
use crate::intelligence::{
    extract_edge_op_pairs_batch_json, extract_edge_op_pairs_json, extract_graph_segments_json,
    extract_graph_segments_map,
    extract_topology_feature_map,
    extract_topology_feature_map_with_imodel,
    extract_topology_features_batch_json,
    extract_topology_feature_maps_batch,
    extract_topology_feature_maps_with_imodel_batch,
    extract_topology_features_json,
    extract_topology_features_with_imodel_json,
    extract_topology_features_with_imodel_batch,
    train_interaction_model_native,
    train_op_embeddings_epoch_native,
};
use crate::notebook_graph::{
    analyze_graph_provenance, analyze_graph_provenance_json, extract_graph_feature_payload_json, extract_graph_ops_json,
    extract_graph_structure_features_json,
};
use crate::template_selection::TemplateSelector;

#[pyclass]
struct SavedActivationStore {
    saved_activations: HashMap<u32, Vec<f32>>,
}

#[pyclass]
struct CompiledGraphHandle {
    graph: GraphIR,
}

#[pyclass]
struct TemplateSelectorHandle {
    selector: TemplateSelector,
}

enum SavedActivationInput<'py> {
    Array(PyReadonlyArrayDyn<'py, f32>),
    Owned(Vec<f32>),
}

impl<'py> SavedActivationInput<'py> {
    fn as_slice(&self) -> PyResult<&[f32]> {
        match self {
            SavedActivationInput::Array(array) => readonly_array_slice(array),
            SavedActivationInput::Owned(values) => Ok(values.as_slice()),
        }
    }
}

fn readonly_array_slices<'py>(
    inputs: &'py [PyReadonlyArrayDyn<'py, f32>],
) -> PyResult<Vec<&'py [f32]>> {
    let mut slices = Vec::with_capacity(inputs.len());
    for input in inputs {
        let slice = input
            .as_slice()
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        slices.push(slice);
    }
    Ok(slices)
}

fn readonly_array_slice<'py, T: Element>(
    input: &'py PyReadonlyArrayDyn<'py, T>,
) -> PyResult<&'py [T]> {
    input
        .as_slice()
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
}

fn feature_map_to_pydict<'py>(
    py: Python<'py>,
    features: HashMap<String, f64>,
) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new_bound(py);
    for (key, value) in features {
        dict.set_item(key, value)?;
    }
    Ok(dict)
}

fn feature_maps_to_pylist<'py>(
    py: Python<'py>,
    features_batch: Vec<HashMap<String, f64>>,
) -> PyResult<Bound<'py, PyList>> {
    let items = PyList::empty_bound(py);
    for features in features_batch {
        items.append(feature_map_to_pydict(py, features)?)?;
    }
    Ok(items)
}

fn matrix_to_numpy<'py>(
    py: Python<'py>,
    flat: Vec<f64>,
    rows: usize,
    cols: usize,
) -> PyResult<Bound<'py, numpy::PyArray2<f64>>> {
    if rows * cols != flat.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "flat matrix shape mismatch",
        ));
    }
    let array = flat.into_pyarray_bound(py);
    array.reshape([rows, cols])
}

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

#[pyfunction]
fn compile_graph_ir_handle(py: Python<'_>, json: &str) -> PyResult<Py<CompiledGraphHandle>> {
    let graph = GraphIR::from_json(json)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    graph
        .validate()
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    Py::new(py, CompiledGraphHandle { graph })
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

/// Compute the canonical notebook-graph fingerprint from stored graph_json.
#[pyfunction]
fn fingerprint_notebook_graph(json: &str) -> PyResult<String> {
    fingerprint_notebook_graph_json(json)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
}

#[pyfunction]
fn extract_graph_ops(json: &str) -> PyResult<Vec<String>> {
    extract_graph_ops_json(json)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
}

#[pyfunction]
fn extract_graph_ops_batch(graphs: Vec<String>) -> PyResult<Vec<Vec<String>>> {
    graphs
        .into_iter()
        .map(|graph| {
            extract_graph_ops_json(&graph)
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
        })
        .collect()
}

#[pyfunction]
fn extract_graph_feature_payload(
    json: &str,
) -> PyResult<(String, Vec<String>, Vec<String>, String, String, String)> {
    let payload = extract_graph_feature_payload_json(json)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    Ok((
        payload.template_name,
        payload.op_names,
        payload.pair_signatures,
        payload.templates_json,
        payload.motifs_json,
        payload.slot_usage_json,
    ))
}

#[pyfunction(signature = (graph_json, generic_sink_ops, failure_op=None))]
fn analyze_graph_provenance_native(
    graph_json: &str,
    generic_sink_ops: Vec<String>,
    failure_op: Option<&str>,
) -> PyResult<String> {
    analyze_graph_provenance_json(graph_json, failure_op, &generic_sink_ops)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
}

#[pyfunction(signature = (graph_json, generic_sink_ops, failure_op=None))]
fn analyze_graph_provenance_native_py<'py>(
    py: Python<'py>,
    graph_json: &str,
    generic_sink_ops: Vec<String>,
    failure_op: Option<&str>,
) -> PyResult<PyObject> {
    let payload = analyze_graph_provenance(graph_json, failure_op, &generic_sink_ops)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
    let dict = PyDict::new_bound(py);
    dict.set_item("op_names", payload.op_names)?;
    dict.set_item("source_op", payload.source_op)?;
    Ok(dict.into())
}

#[pyfunction]
fn extract_graph_structure_features_native(graph_json: &str) -> PyResult<String> {
    extract_graph_structure_features_json(graph_json)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
}

#[pyfunction]
fn compile_template_selector_handle(
    py: Python<'_>,
    names: Vec<String>,
    default_weights: Vec<f64>,
) -> PyResult<Py<TemplateSelectorHandle>> {
    let selector = TemplateSelector::new(names, default_weights)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    Py::new(py, TemplateSelectorHandle { selector })
}

#[pyfunction(signature = (
    handle,
    exploration_budget,
    exploration_draw,
    selection_draw,
    override_weights=None,
    allowed_names=None
))]
fn select_template_index_compiled(
    handle: PyRef<'_, TemplateSelectorHandle>,
    exploration_budget: f64,
    exploration_draw: f64,
    selection_draw: f64,
    override_weights: Option<HashMap<String, f64>>,
    allowed_names: Option<Vec<String>>,
) -> PyResult<(usize, bool)> {
    handle
        .selector
        .select_index(
            override_weights.as_ref(),
            allowed_names.as_ref(),
            exploration_budget,
            exploration_draw,
            selection_draw,
        )
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
}

#[derive(Serialize)]
struct OpRateRecord {
    op: String,
    n: u64,
    s0: u64,
    s1: u64,
}

#[derive(Serialize)]
struct CorrectedRateRecord {
    op: String,
    n: u64,
    s0: u64,
    s1: u64,
    excluded: u64,
}

#[derive(Serialize)]
struct PairCountRecord {
    op_a: String,
    op_b: String,
    n: u64,
    s0: u64,
    s1: u64,
}

#[derive(Serialize)]
struct LossByOpRecord {
    op: String,
    values: Vec<f64>,
}

#[derive(Serialize)]
struct FailureOpRecord {
    op: String,
    count: u64,
}

#[derive(Serialize)]
struct FailureGroupRecord {
    name: String,
    count: u64,
    ops: Vec<FailureOpRecord>,
}

#[derive(Serialize)]
struct OpIndexPayload {
    pair_counts: Vec<PairCountRecord>,
    loss_by_op: Vec<LossByOpRecord>,
    failure_groups: Vec<FailureGroupRecord>,
    stored_rates: Vec<OpRateRecord>,
    corrected_rates: Vec<CorrectedRateRecord>,
}

#[derive(Default)]
struct RateStats {
    n: u64,
    s0: u64,
    s1: u64,
}

#[derive(Default)]
struct CorrectedRateStats {
    n: u64,
    s0: u64,
    s1: u64,
    excluded: u64,
}

#[derive(Default)]
struct PairStats {
    n: u64,
    s0: u64,
    s1: u64,
}

#[derive(Default)]
struct FailureGroupStats {
    count: u64,
    ops: HashMap<String, u64>,
}

#[derive(Default, serde::Deserialize)]
struct FailureDetails {
    root_cause_code: Option<String>,
    failure_op: Option<String>,
}

#[derive(Deserialize)]
struct OpIndexInputRow {
    graph_json: String,
    stage0_passed: bool,
    stage1_passed: bool,
    loss_ratio: Option<f64>,
    error_type: Option<String>,
    failure_op: Option<String>,
    failure_details_json: Option<String>,
}

#[pyfunction]
fn build_op_index_from_rows(rows_json: &str) -> PyResult<String> {
    let rows: Vec<OpIndexInputRow> = serde_json::from_str(rows_json)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    let mut stored_rates: HashMap<String, RateStats> = HashMap::new();
    let mut corrected_rates: HashMap<String, CorrectedRateStats> = HashMap::new();
    let mut pair_counts: HashMap<(String, String), PairStats> = HashMap::new();
    let mut loss_by_op: HashMap<String, Vec<f64>> = HashMap::new();
    let mut failure_groups: HashMap<String, FailureGroupStats> = HashMap::new();
    let mut graph_ops_cache: HashMap<String, Vec<String>> = HashMap::new();

    let non_op_errors = ["RuntimeError", "causality_violation"];

    for row in rows {
        let ops = graph_ops_cache
            .entry(row.graph_json.clone())
            .or_insert_with(|| extract_graph_ops_json(&row.graph_json).unwrap_or_default());
        if ops.is_empty() {
            continue;
        }

        let error_type = row.error_type.unwrap_or_default();
        let is_non_op_failure =
            !row.stage0_passed && non_op_errors.contains(&error_type.as_str());

        for op in ops.iter() {
            let stored = stored_rates.entry(op.clone()).or_default();
            stored.n += 1;
            if row.stage0_passed {
                stored.s0 += 1;
            }
            if row.stage1_passed {
                stored.s1 += 1;
            }

            let corrected = corrected_rates.entry(op.clone()).or_default();
            if is_non_op_failure {
                corrected.excluded += 1;
            } else {
                corrected.n += 1;
                if row.stage0_passed {
                    corrected.s0 += 1;
                }
                if row.stage1_passed {
                    corrected.s1 += 1;
                }
            }
        }

        for (idx, left) in ops.iter().enumerate() {
            for right in ops.iter().skip(idx + 1) {
                let pair = pair_counts
                    .entry((left.clone(), right.clone()))
                    .or_default();
                pair.n += 1;
                if row.stage0_passed {
                    pair.s0 += 1;
                }
                if row.stage1_passed {
                    pair.s1 += 1;
                }
            }
        }

        if let Some(loss_value) = row.loss_ratio {
            if row.stage0_passed {
                for op in ops.iter() {
                    loss_by_op.entry(op.clone()).or_default().push(loss_value);
                }
            }
        }

        if !error_type.is_empty() {
            let details = row
                .failure_details_json
                .as_deref()
                .and_then(|raw| serde_json::from_str::<FailureDetails>(raw).ok())
                .unwrap_or_default();
            let root_cause_code = details
                .root_cause_code
                .unwrap_or_else(|| error_type.clone());
            let failure_op = details
                .failure_op
                .or(row.failure_op)
                .unwrap_or_default();

            if !row.stage0_passed {
                let group = failure_groups.entry(root_cause_code.clone()).or_default();
                group.count += 1;
                if !failure_op.is_empty() {
                    *group.ops.entry(failure_op.clone()).or_insert(0) += 1;
                } else {
                    for op in ops.iter() {
                        *group.ops.entry(op.clone()).or_insert(0) += 1;
                    }
                }
            }
            if row.stage0_passed && !row.stage1_passed {
                let group = failure_groups
                    .entry(format!("s1_{}", root_cause_code))
                    .or_default();
                group.count += 1;
                if !failure_op.is_empty() {
                    *group.ops.entry(failure_op).or_insert(0) += 1;
                } else {
                    for op in ops.iter() {
                        *group.ops.entry(op.clone()).or_insert(0) += 1;
                    }
                }
            }
        }
    }

    let mut pair_counts_out: Vec<PairCountRecord> = pair_counts
        .into_iter()
        .map(|((op_a, op_b), stats)| PairCountRecord {
            op_a,
            op_b,
            n: stats.n,
            s0: stats.s0,
            s1: stats.s1,
        })
        .collect();
    pair_counts_out.sort_by(|a, b| a.op_a.cmp(&b.op_a).then_with(|| a.op_b.cmp(&b.op_b)));

    let mut loss_by_op_out: Vec<LossByOpRecord> = loss_by_op
        .into_iter()
        .map(|(op, values)| LossByOpRecord { op, values })
        .collect();
    loss_by_op_out.sort_by(|a, b| a.op.cmp(&b.op));

    let mut failure_groups_out: Vec<FailureGroupRecord> = failure_groups
        .into_iter()
        .map(|(name, stats)| {
            let mut ops: Vec<FailureOpRecord> = stats
                .ops
                .into_iter()
                .map(|(op, count)| FailureOpRecord { op, count })
                .collect();
            ops.sort_by(|a, b| b.count.cmp(&a.count).then_with(|| a.op.cmp(&b.op)));
            FailureGroupRecord {
                name,
                count: stats.count,
                ops,
            }
        })
        .collect();
    failure_groups_out.sort_by(|a, b| b.count.cmp(&a.count).then_with(|| a.name.cmp(&b.name)));

    let mut stored_rates_out: Vec<OpRateRecord> = stored_rates
        .into_iter()
        .map(|(op, stats)| OpRateRecord {
            op,
            n: stats.n,
            s0: stats.s0,
            s1: stats.s1,
        })
        .collect();
    stored_rates_out.sort_by(|a, b| a.op.cmp(&b.op));

    let mut corrected_rates_out: Vec<CorrectedRateRecord> = corrected_rates
        .into_iter()
        .map(|(op, stats)| CorrectedRateRecord {
            op,
            n: stats.n,
            s0: stats.s0,
            s1: stats.s1,
            excluded: stats.excluded,
        })
        .collect();
    corrected_rates_out.sort_by(|a, b| a.op.cmp(&b.op));

    let payload = OpIndexPayload {
        pair_counts: pair_counts_out,
        loss_by_op: loss_by_op_out,
        failure_groups: failure_groups_out,
        stored_rates: stored_rates_out,
        corrected_rates: corrected_rates_out,
    };
    serde_json::to_string(&payload)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
}

/// Build a deduped graph-training corpus from the notebook DB and return it as JSON.
#[pyfunction]
fn build_graph_training_corpus(db_path: &str) -> PyResult<String> {
    build_graph_training_corpus_json(Path::new(db_path))
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
}

/// Build a deduped predictor-training corpus from the notebook DB and return it as JSON.
#[pyfunction]
fn build_predictor_training_corpus(db_path: &str) -> PyResult<String> {
    build_predictor_training_corpus_json(Path::new(db_path))
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
}

#[pyfunction]
fn extract_topology_features_native(
    graph_json: &str,
    op_profiles_json: &str,
    pair_stability_json: &str,
    op_metadata_json: &str,
) -> PyResult<String> {
    extract_topology_features_json(
        graph_json,
        op_profiles_json,
        pair_stability_json,
        op_metadata_json,
    )
    .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
}

#[pyfunction]
fn extract_topology_feature_map_native_py<'py>(
    py: Python<'py>,
    graph_json: &str,
    op_profiles_json: &str,
    pair_stability_json: &str,
    op_metadata_json: &str,
) -> PyResult<PyObject> {
    let features = extract_topology_feature_map(
        graph_json,
        op_profiles_json,
        pair_stability_json,
        op_metadata_json,
    )
    .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
    Ok(feature_map_to_pydict(py, features)?.into())
}

#[pyfunction]
fn extract_topology_features_with_imodel_native<'py>(
    graph_json: &str,
    op_profiles_json: &str,
    pair_stability_json: &str,
    op_metadata_json: &str,
    op_names: Vec<String>,
    u: PyReadonlyArrayDyn<'py, f32>,
    v: PyReadonlyArrayDyn<'py, f32>,
    w_s: PyReadonlyArrayDyn<'py, f32>,
    w_l: PyReadonlyArrayDyn<'py, f32>,
    b_s: f64,
    b_l: f64,
) -> PyResult<String> {
    let u_shape = u.shape();
    let v_shape = v.shape();
    let ws_shape = w_s.shape();
    let wl_shape = w_l.shape();
    if u_shape.len() != 2 || v_shape.len() != 2 || ws_shape.len() != 2 || wl_shape.len() != 2 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "interaction model arrays must be 2D",
        ));
    }
    extract_topology_features_with_imodel_json(
        graph_json,
        op_profiles_json,
        pair_stability_json,
        op_metadata_json,
        &op_names,
        readonly_array_slice(&u)?,
        u_shape[0],
        u_shape[1],
        readonly_array_slice(&v)?,
        v_shape[0],
        v_shape[1],
        readonly_array_slice(&w_s)?,
        ws_shape[0],
        ws_shape[1],
        readonly_array_slice(&w_l)?,
        wl_shape[0],
        wl_shape[1],
        b_s,
        b_l,
    )
    .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
}

#[pyfunction]
fn extract_topology_feature_map_with_imodel_native_py<'py>(
    py: Python<'py>,
    graph_json: &str,
    op_profiles_json: &str,
    pair_stability_json: &str,
    op_metadata_json: &str,
    op_names: Vec<String>,
    u: PyReadonlyArrayDyn<'py, f32>,
    v: PyReadonlyArrayDyn<'py, f32>,
    w_s: PyReadonlyArrayDyn<'py, f32>,
    w_l: PyReadonlyArrayDyn<'py, f32>,
    b_s: f64,
    b_l: f64,
) -> PyResult<PyObject> {
    let u_shape = u.shape();
    let v_shape = v.shape();
    let ws_shape = w_s.shape();
    let wl_shape = w_l.shape();
    if u_shape.len() != 2 || v_shape.len() != 2 || ws_shape.len() != 2 || wl_shape.len() != 2 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "interaction model arrays must be 2D",
        ));
    }
    let features = extract_topology_feature_map_with_imodel(
        graph_json,
        op_profiles_json,
        pair_stability_json,
        op_metadata_json,
        &op_names,
        readonly_array_slice(&u)?,
        u_shape[0],
        u_shape[1],
        readonly_array_slice(&v)?,
        v_shape[0],
        v_shape[1],
        readonly_array_slice(&w_s)?,
        ws_shape[0],
        ws_shape[1],
        readonly_array_slice(&w_l)?,
        wl_shape[0],
        wl_shape[1],
        b_s,
        b_l,
    )
    .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
    Ok(feature_map_to_pydict(py, features)?.into())
}

#[pyfunction]
fn extract_topology_features_batch_native(
    graphs: Vec<String>,
    op_profiles_json: &str,
    pair_stability_json: &str,
    op_metadata_json: &str,
) -> PyResult<Vec<String>> {
    extract_topology_features_batch_json(
        &graphs,
        op_profiles_json,
        pair_stability_json,
        op_metadata_json,
    )
    .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
}

#[pyfunction]
fn extract_topology_feature_maps_batch_native_py<'py>(
    py: Python<'py>,
    graphs: Vec<String>,
    op_profiles_json: &str,
    pair_stability_json: &str,
    op_metadata_json: &str,
) -> PyResult<PyObject> {
    let features_batch = extract_topology_feature_maps_batch(
        &graphs,
        op_profiles_json,
        pair_stability_json,
        op_metadata_json,
    )
    .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
    Ok(feature_maps_to_pylist(py, features_batch)?.into())
}

#[pyfunction]
fn extract_topology_features_with_imodel_batch_native<'py>(
    graphs: Vec<String>,
    op_profiles_json: &str,
    pair_stability_json: &str,
    op_metadata_json: &str,
    op_names: Vec<String>,
    u: PyReadonlyArrayDyn<'py, f32>,
    v: PyReadonlyArrayDyn<'py, f32>,
    w_s: PyReadonlyArrayDyn<'py, f32>,
    w_l: PyReadonlyArrayDyn<'py, f32>,
    b_s: f64,
    b_l: f64,
) -> PyResult<Vec<String>> {
    let u_shape = u.shape();
    let v_shape = v.shape();
    let ws_shape = w_s.shape();
    let wl_shape = w_l.shape();
    if u_shape.len() != 2 || v_shape.len() != 2 || ws_shape.len() != 2 || wl_shape.len() != 2 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "interaction model arrays must be 2D",
        ));
    }
    extract_topology_features_with_imodel_batch(
        &graphs,
        op_profiles_json,
        pair_stability_json,
        op_metadata_json,
        &op_names,
        readonly_array_slice(&u)?,
        u_shape[0],
        u_shape[1],
        readonly_array_slice(&v)?,
        v_shape[0],
        v_shape[1],
        readonly_array_slice(&w_s)?,
        ws_shape[0],
        ws_shape[1],
        readonly_array_slice(&w_l)?,
        wl_shape[0],
        wl_shape[1],
        b_s,
        b_l,
    )
    .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
}

#[pyfunction]
fn extract_topology_feature_maps_with_imodel_batch_native_py<'py>(
    py: Python<'py>,
    graphs: Vec<String>,
    op_profiles_json: &str,
    pair_stability_json: &str,
    op_metadata_json: &str,
    op_names: Vec<String>,
    u: PyReadonlyArrayDyn<'py, f32>,
    v: PyReadonlyArrayDyn<'py, f32>,
    w_s: PyReadonlyArrayDyn<'py, f32>,
    w_l: PyReadonlyArrayDyn<'py, f32>,
    b_s: f64,
    b_l: f64,
) -> PyResult<PyObject> {
    let u_shape = u.shape();
    let v_shape = v.shape();
    let ws_shape = w_s.shape();
    let wl_shape = w_l.shape();
    if u_shape.len() != 2 || v_shape.len() != 2 || ws_shape.len() != 2 || wl_shape.len() != 2 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "interaction model arrays must be 2D",
        ));
    }
    let features_batch = extract_topology_feature_maps_with_imodel_batch(
        &graphs,
        op_profiles_json,
        pair_stability_json,
        op_metadata_json,
        &op_names,
        readonly_array_slice(&u)?,
        u_shape[0],
        u_shape[1],
        readonly_array_slice(&v)?,
        v_shape[0],
        v_shape[1],
        readonly_array_slice(&w_s)?,
        ws_shape[0],
        ws_shape[1],
        readonly_array_slice(&w_l)?,
        wl_shape[0],
        wl_shape[1],
        b_s,
        b_l,
    )
    .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
    Ok(feature_maps_to_pylist(py, features_batch)?.into())
}

#[pyfunction]
fn extract_edge_op_pairs_native(graph_json: &str) -> PyResult<String> {
    extract_edge_op_pairs_json(graph_json)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
}

#[pyfunction]
fn extract_edge_op_pairs_batch_native(graphs: Vec<String>) -> PyResult<Vec<String>> {
    extract_edge_op_pairs_batch_json(&graphs)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
}

#[pyfunction]
fn extract_graph_segments_native(
    graph_json: &str,
    min_len: usize,
    max_len: usize,
) -> PyResult<String> {
    extract_graph_segments_json(graph_json, min_len, max_len)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
}

#[pyfunction]
fn extract_graph_segments_map_native_py<'py>(
    py: Python<'py>,
    graph_json: &str,
    min_len: usize,
    max_len: usize,
) -> PyResult<PyObject> {
    let counts = extract_graph_segments_map(graph_json, min_len, max_len)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
    let dict = PyDict::new_bound(py);
    for (key, value) in counts {
        dict.set_item(key, value)?;
    }
    Ok(dict.into())
}

#[pyfunction]
fn train_interaction_model_native_py<'py>(
    py: Python<'py>,
    u: PyReadonlyArrayDyn<'py, f64>,
    v: PyReadonlyArrayDyn<'py, f64>,
    w_s: PyReadonlyArrayDyn<'py, f64>,
    w_l: PyReadonlyArrayDyn<'py, f64>,
    b_s: f64,
    b_l: f64,
    stab_idx: PyReadonlyArrayDyn<'py, i32>,
    stab_labels: PyReadonlyArrayDyn<'py, f64>,
    stab_weights: PyReadonlyArrayDyn<'py, f64>,
    loss_idx: PyReadonlyArrayDyn<'py, i32>,
    loss_labels: PyReadonlyArrayDyn<'py, f64>,
    loss_weights: PyReadonlyArrayDyn<'py, f64>,
    n_epochs: usize,
    lr: f64,
    batch_size: usize,
    seed: u64,
) -> PyResult<PyObject> {
    let u_shape = u.shape();
    let v_shape = v.shape();
    let ws_shape = w_s.shape();
    let wl_shape = w_l.shape();
    let stab_shape = stab_idx.shape();
    let loss_shape = loss_idx.shape();
    if u_shape.len() != 2
        || v_shape.len() != 2
        || ws_shape.len() != 2
        || wl_shape.len() != 2
        || stab_shape.len() != 2
        || loss_shape.len() != 2
    {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "expected rank-2 arrays for matrices and index arrays",
        ));
    }
    let u_vec = readonly_array_slice(&u)?.to_vec();
    let v_vec = readonly_array_slice(&v)?.to_vec();
    let ws_vec = readonly_array_slice(&w_s)?.to_vec();
    let wl_vec = readonly_array_slice(&w_l)?.to_vec();
    let stab_idx_vec = readonly_array_slice(&stab_idx)?.to_vec();
    let stab_labels_vec = readonly_array_slice(&stab_labels)?.to_vec();
    let stab_weights_vec = readonly_array_slice(&stab_weights)?.to_vec();
    let loss_idx_vec = readonly_array_slice(&loss_idx)?.to_vec();
    let loss_labels_vec = readonly_array_slice(&loss_labels)?.to_vec();
    let loss_weights_vec = readonly_array_slice(&loss_weights)?.to_vec();

    let result = py.allow_threads(move || {
        train_interaction_model_native(
            &u_vec,
            u_shape[0],
            u_shape[1],
            &v_vec,
            v_shape[0],
            v_shape[1],
            &ws_vec,
            ws_shape[0],
            ws_shape[1],
            &wl_vec,
            wl_shape[0],
            wl_shape[1],
            b_s,
            b_l,
            &stab_idx_vec,
            stab_shape[0],
            &stab_labels_vec,
            &stab_weights_vec,
            &loss_idx_vec,
            loss_shape[0],
            &loss_labels_vec,
            &loss_weights_vec,
            n_epochs,
            lr,
            batch_size,
            seed,
        )
    })
    .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

    let dict = PyDict::new_bound(py);
    dict.set_item("u", matrix_to_numpy(py, result.u, result.u_rows, result.u_cols)?)?;
    dict.set_item("v", matrix_to_numpy(py, result.v, result.v_rows, result.v_cols)?)?;
    dict.set_item("W_s", matrix_to_numpy(py, result.w_s, result.ws_rows, result.ws_cols)?)?;
    dict.set_item("W_l", matrix_to_numpy(py, result.w_l, result.wl_rows, result.wl_cols)?)?;
    dict.set_item("b_s", result.b_s)?;
    dict.set_item("b_l", result.b_l)?;
    dict.set_item("best_loss", result.best_loss)?;
    Ok(dict.into())
}

#[pyfunction]
fn train_op_embeddings_epoch_native_py<'py>(
    py: Python<'py>,
    embeddings: PyReadonlyArrayDyn<'py, f64>,
    positive_pairs: PyReadonlyArrayDyn<'py, i32>,
    negative_pairs: PyReadonlyArrayDyn<'py, i32>,
    pair_idx: PyReadonlyArrayDyn<'py, i32>,
    pair_labels: PyReadonlyArrayDyn<'py, f64>,
    lr: f64,
    batch_size: usize,
    margin: f64,
    pair_weight: f64,
    seed: u64,
) -> PyResult<PyObject> {
    let emb_shape = embeddings.shape();
    let pos_shape = positive_pairs.shape();
    let neg_shape = negative_pairs.shape();
    let pair_shape = pair_idx.shape();
    if emb_shape.len() != 2 || pos_shape.len() != 2 || neg_shape.len() != 2 || pair_shape.len() != 2
    {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "expected rank-2 arrays for embedding and pair inputs",
        ));
    }
    let embeddings_vec = readonly_array_slice(&embeddings)?.to_vec();
    let positive_vec = readonly_array_slice(&positive_pairs)?.to_vec();
    let negative_vec = readonly_array_slice(&negative_pairs)?.to_vec();
    let pair_idx_vec = readonly_array_slice(&pair_idx)?.to_vec();
    let pair_labels_vec = readonly_array_slice(&pair_labels)?.to_vec();

    let result = py.allow_threads(move || {
        train_op_embeddings_epoch_native(
            &embeddings_vec,
            emb_shape[0],
            emb_shape[1],
            &positive_vec,
            pos_shape[0],
            &negative_vec,
            neg_shape[0],
            &pair_idx_vec,
            pair_shape[0],
            &pair_labels_vec,
            lr,
            batch_size,
            margin,
            pair_weight,
            seed,
        )
    })
    .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

    let dict = PyDict::new_bound(py);
    dict.set_item(
        "embeddings",
        matrix_to_numpy(py, result.embeddings, result.emb_rows, result.emb_cols)?,
    )?;
    dict.set_item("total_loss", result.total_loss)?;
    dict.set_item("n_samples", result.n_samples)?;
    Ok(dict.into())
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
    dict.set_item(
        "heap_fallback_count",
        result.arena_stats.heap_fallback_count,
    )?;

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

#[pyfunction]
fn execute_graph_with_stats_arrays<'py>(
    py: Python<'py>,
    json: &str,
    input: PyReadonlyArrayDyn<'py, f32>,
) -> PyResult<PyObject> {
    let graph = GraphIR::from_json(json)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    let input_slice = readonly_array_slice(&input)?;
    let result = execute_with_arena(&graph, &NativeKernelDispatch, input_slice)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

    let dict = PyDict::new_bound(py);
    dict.set_item("output", result.output.into_pyarray_bound(py))?;
    dict.set_item("arena_bytes_used", result.arena_stats.arena_bytes_used)?;
    dict.set_item("arena_capacity", result.arena_stats.arena_capacity)?;
    dict.set_item("arena_alloc_count", result.arena_stats.arena_alloc_count)?;
    dict.set_item(
        "heap_fallback_count",
        result.arena_stats.heap_fallback_count,
    )?;

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

#[pyfunction]
fn execute_graph_with_stats_compiled_arrays<'py>(
    py: Python<'py>,
    graph: PyRef<'py, CompiledGraphHandle>,
    input: PyReadonlyArrayDyn<'py, f32>,
) -> PyResult<PyObject> {
    let input_slice = readonly_array_slice(&input)?;
    let result = execute_with_arena(&graph.graph, &NativeKernelDispatch, input_slice)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

    let dict = PyDict::new_bound(py);
    dict.set_item("output", result.output.into_pyarray_bound(py))?;
    dict.set_item("arena_bytes_used", result.arena_stats.arena_bytes_used)?;
    dict.set_item("arena_capacity", result.arena_stats.arena_capacity)?;
    dict.set_item("arena_alloc_count", result.arena_stats.arena_alloc_count)?;
    dict.set_item(
        "heap_fallback_count",
        result.arena_stats.heap_fallback_count,
    )?;
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

#[pyfunction]
fn execute_graph_compiled_arrays_handle<'py>(
    py: Python<'py>,
    graph: PyRef<'py, CompiledGraphHandle>,
    input: PyReadonlyArrayDyn<'py, f32>,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    let input_slice = readonly_array_slice(&input)?;
    let result = execute_with_arena(&graph.graph, &NativeKernelDispatch, input_slice)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
    Ok(result.output.into_pyarray_bound(py))
}

/// Execute a graph IR using multiple distinct input buffers.
///
/// Inputs are bound to input nodes in ascending input-node order.
#[pyfunction]
fn execute_graph_multi_input(json: &str, inputs: Vec<Vec<f32>>) -> PyResult<Vec<f32>> {
    let graph = GraphIR::from_json(json)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    let input_slices: Vec<&[f32]> = inputs.iter().map(|input| input.as_slice()).collect();
    let result = execute_with_arena_multi_input(&graph, &NativeKernelDispatch, &input_slices)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
    Ok(result.output)
}

/// Execute a graph IR using multiple contiguous float32 array inputs.
#[pyfunction]
fn execute_graph_multi_input_arrays<'py>(
    py: Python<'py>,
    json: &str,
    inputs: Vec<PyReadonlyArrayDyn<'py, f32>>,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    let graph = GraphIR::from_json(json)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    let input_slices = readonly_array_slices(&inputs)?;
    let result = execute_with_arena_multi_input(&graph, &NativeKernelDispatch, &input_slices)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
    Ok(result.output.into_pyarray_bound(py))
}

/// Execute a compiled graph IR using multiple contiguous float32 array inputs.
#[pyfunction]
fn execute_graph_multi_input_compiled_arrays_handle<'py>(
    py: Python<'py>,
    graph: PyRef<'py, CompiledGraphHandle>,
    inputs: Vec<PyReadonlyArrayDyn<'py, f32>>,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    let input_slices = readonly_array_slices(&inputs)?;
    let result = execute_with_arena_multi_input(&graph.graph, &NativeKernelDispatch, &input_slices)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
    Ok(result.output.into_pyarray_bound(py))
}

/// Execute a graph IR with distinct input buffers and return arena stats.
#[pyfunction]
fn execute_graph_multi_input_with_stats(
    py: Python<'_>,
    json: &str,
    inputs: Vec<Vec<f32>>,
) -> PyResult<PyObject> {
    let graph = GraphIR::from_json(json)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    let input_slices: Vec<&[f32]> = inputs.iter().map(|input| input.as_slice()).collect();
    let result = execute_with_arena_multi_input(&graph, &NativeKernelDispatch, &input_slices)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

    let dict = PyDict::new_bound(py);
    dict.set_item("output", result.output)?;
    dict.set_item("arena_bytes_used", result.arena_stats.arena_bytes_used)?;
    dict.set_item("arena_capacity", result.arena_stats.arena_capacity)?;
    dict.set_item("arena_alloc_count", result.arena_stats.arena_alloc_count)?;
    dict.set_item(
        "heap_fallback_count",
        result.arena_stats.heap_fallback_count,
    )?;
    Ok(dict.into())
}

/// Execute a compiled graph IR with contiguous float32 array inputs and return arena stats.
#[pyfunction]
fn execute_graph_multi_input_compiled_arrays_with_stats<'py>(
    py: Python<'py>,
    graph: PyRef<'py, CompiledGraphHandle>,
    inputs: Vec<PyReadonlyArrayDyn<'py, f32>>,
) -> PyResult<PyObject> {
    let input_slices = readonly_array_slices(&inputs)?;
    let result = execute_with_arena_multi_input(&graph.graph, &NativeKernelDispatch, &input_slices)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

    let dict = PyDict::new_bound(py);
    dict.set_item("output", result.output.into_pyarray_bound(py))?;
    dict.set_item("arena_bytes_used", result.arena_stats.arena_bytes_used)?;
    dict.set_item("arena_capacity", result.arena_stats.arena_capacity)?;
    dict.set_item("arena_alloc_count", result.arena_stats.arena_alloc_count)?;
    dict.set_item(
        "heap_fallback_count",
        result.arena_stats.heap_fallback_count,
    )?;
    Ok(dict.into())
}

/// Execute a graph IR with contiguous float32 array inputs and return arena stats.
#[pyfunction]
fn execute_graph_multi_input_arrays_with_stats<'py>(
    py: Python<'py>,
    json: &str,
    inputs: Vec<PyReadonlyArrayDyn<'py, f32>>,
) -> PyResult<PyObject> {
    let graph = GraphIR::from_json(json)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    let input_slices = readonly_array_slices(&inputs)?;
    let result = execute_with_arena_multi_input(&graph, &NativeKernelDispatch, &input_slices)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

    let dict = PyDict::new_bound(py);
    dict.set_item("output", result.output.into_pyarray_bound(py))?;
    dict.set_item("arena_bytes_used", result.arena_stats.arena_bytes_used)?;
    dict.set_item("arena_capacity", result.arena_stats.arena_capacity)?;
    dict.set_item("arena_alloc_count", result.arena_stats.arena_alloc_count)?;
    dict.set_item(
        "heap_fallback_count",
        result.arena_stats.heap_fallback_count,
    )?;
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

#[pyfunction]
fn execute_graph_forward_saved_handle(
    py: Python<'_>,
    json: &str,
    input: Vec<f32>,
) -> PyResult<PyObject> {
    let graph = GraphIR::from_json(json)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    let result = execute_forward_saving_activations(&graph, &NativeKernelDispatch, &input)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

    let dict = PyDict::new_bound(py);
    dict.set_item("output", result.output)?;
    dict.set_item(
        "saved_state",
        Py::new(
            py,
            SavedActivationStore {
                saved_activations: result.saved_activations,
            },
        )?,
    )?;
    dict.set_item("arena_bytes_used", result.arena_stats.arena_bytes_used)?;
    dict.set_item("arena_capacity", result.arena_stats.arena_capacity)?;
    Ok(dict.into())
}

#[pyfunction]
fn execute_graph_forward_saved_arrays<'py>(
    py: Python<'py>,
    json: &str,
    input: PyReadonlyArrayDyn<'py, f32>,
) -> PyResult<PyObject> {
    let graph = GraphIR::from_json(json)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    let input_slice = readonly_array_slice(&input)?;
    let result = execute_forward_saving_activations(&graph, &NativeKernelDispatch, input_slice)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

    let dict = PyDict::new_bound(py);
    dict.set_item("output", result.output.into_pyarray_bound(py))?;

    let saved_dict = PyDict::new_bound(py);
    for (node_id, activation) in &result.saved_activations {
        saved_dict.set_item(*node_id, activation.clone().into_pyarray_bound(py))?;
    }
    dict.set_item("saved_activations", saved_dict)?;
    dict.set_item("arena_bytes_used", result.arena_stats.arena_bytes_used)?;
    dict.set_item("arena_capacity", result.arena_stats.arena_capacity)?;

    Ok(dict.into())
}

#[pyfunction]
fn execute_graph_forward_saved_arrays_handle<'py>(
    py: Python<'py>,
    json: &str,
    input: PyReadonlyArrayDyn<'py, f32>,
) -> PyResult<PyObject> {
    let graph = GraphIR::from_json(json)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    let input_slice = readonly_array_slice(&input)?;
    let result = execute_forward_saving_activations(&graph, &NativeKernelDispatch, input_slice)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

    let dict = PyDict::new_bound(py);
    dict.set_item("output", result.output.into_pyarray_bound(py))?;
    dict.set_item(
        "saved_state",
        Py::new(
            py,
            SavedActivationStore {
                saved_activations: result.saved_activations,
            },
        )?,
    )?;
    dict.set_item("arena_bytes_used", result.arena_stats.arena_bytes_used)?;
    dict.set_item("arena_capacity", result.arena_stats.arena_capacity)?;
    Ok(dict.into())
}

#[pyfunction]
fn execute_graph_forward_saved_compiled_arrays_handle<'py>(
    py: Python<'py>,
    graph: PyRef<'py, CompiledGraphHandle>,
    input: PyReadonlyArrayDyn<'py, f32>,
) -> PyResult<PyObject> {
    let input_slice = readonly_array_slice(&input)?;
    let result =
        execute_forward_saving_activations(&graph.graph, &NativeKernelDispatch, input_slice)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

    let dict = PyDict::new_bound(py);
    dict.set_item("output", result.output.into_pyarray_bound(py))?;
    dict.set_item(
        "saved_state",
        Py::new(
            py,
            SavedActivationStore {
                saved_activations: result.saved_activations,
            },
        )?,
    )?;
    dict.set_item("arena_bytes_used", result.arena_stats.arena_bytes_used)?;
    dict.set_item("arena_capacity", result.arena_stats.arena_capacity)?;
    Ok(dict.into())
}

#[pyfunction]
fn execute_graph_forward_saved_multi_input(
    py: Python<'_>,
    json: &str,
    inputs: Vec<Vec<f32>>,
) -> PyResult<PyObject> {
    let graph = GraphIR::from_json(json)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    let input_slices: Vec<&[f32]> = inputs.iter().map(|input| input.as_slice()).collect();
    let result = execute_forward_saving_activations_multi_input(
        &graph,
        &NativeKernelDispatch,
        &input_slices,
    )
    .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

    let dict = PyDict::new_bound(py);
    dict.set_item("output", result.output.into_pyarray_bound(py))?;

    let saved_dict = PyDict::new_bound(py);
    for (node_id, activation) in result.saved_activations {
        saved_dict.set_item(node_id, activation.into_pyarray_bound(py))?;
    }
    dict.set_item("saved_activations", saved_dict)?;
    dict.set_item("arena_bytes_used", result.arena_stats.arena_bytes_used)?;
    dict.set_item("arena_capacity", result.arena_stats.arena_capacity)?;
    Ok(dict.into())
}

#[pyfunction]
fn execute_graph_forward_saved_multi_input_arrays<'py>(
    py: Python<'py>,
    json: &str,
    inputs: Vec<PyReadonlyArrayDyn<'py, f32>>,
) -> PyResult<PyObject> {
    let graph = GraphIR::from_json(json)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    let input_slices = readonly_array_slices(&inputs)?;
    let result = execute_forward_saving_activations_multi_input(
        &graph,
        &NativeKernelDispatch,
        &input_slices,
    )
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

#[pyfunction]
fn execute_graph_forward_saved_multi_input_arrays_handle<'py>(
    py: Python<'py>,
    json: &str,
    inputs: Vec<PyReadonlyArrayDyn<'py, f32>>,
) -> PyResult<PyObject> {
    let graph = GraphIR::from_json(json)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    let input_slices = readonly_array_slices(&inputs)?;
    let result = execute_forward_saving_activations_multi_input(
        &graph,
        &NativeKernelDispatch,
        &input_slices,
    )
    .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

    let dict = PyDict::new_bound(py);
    dict.set_item("output", result.output.into_pyarray_bound(py))?;
    dict.set_item(
        "saved_state",
        Py::new(
            py,
            SavedActivationStore {
                saved_activations: result.saved_activations,
            },
        )?,
    )?;
    dict.set_item("arena_bytes_used", result.arena_stats.arena_bytes_used)?;
    dict.set_item("arena_capacity", result.arena_stats.arena_capacity)?;
    Ok(dict.into())
}

#[pyfunction]
fn execute_graph_forward_saved_multi_input_compiled_arrays_handle<'py>(
    py: Python<'py>,
    graph: PyRef<'py, CompiledGraphHandle>,
    inputs: Vec<PyReadonlyArrayDyn<'py, f32>>,
) -> PyResult<PyObject> {
    let input_slices = readonly_array_slices(&inputs)?;
    let result = execute_forward_saving_activations_multi_input(
        &graph.graph,
        &NativeKernelDispatch,
        &input_slices,
    )
    .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

    let dict = PyDict::new_bound(py);
    dict.set_item("output", result.output.into_pyarray_bound(py))?;
    dict.set_item(
        "saved_state",
        Py::new(
            py,
            SavedActivationStore {
                saved_activations: result.saved_activations,
            },
        )?,
    )?;
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

#[pyfunction]
fn execute_graph_backward_handle(
    py: Python<'_>,
    json: &str,
    grad_output: Vec<f32>,
    saved_state: PyRef<'_, SavedActivationStore>,
) -> PyResult<PyObject> {
    let graph = GraphIR::from_json(json)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    let result = execute_backward_with_arena(&graph, &grad_output, &saved_state.saved_activations)
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

#[pyfunction]
fn execute_graph_backward_arrays<'py>(
    py: Python<'py>,
    json: &str,
    grad_output: PyReadonlyArrayDyn<'py, f32>,
    saved_activations: &Bound<'py, PyDict>,
) -> PyResult<PyObject> {
    let graph = GraphIR::from_json(json)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    let grad_output_slice = readonly_array_slice(&grad_output)?;

    let mut saved_inputs: HashMap<u32, SavedActivationInput<'py>> = HashMap::new();
    for (key, value) in saved_activations.iter() {
        let node_id: u32 = key.extract()?;
        if let Ok(array) = value.extract::<PyReadonlyArrayDyn<'py, f32>>() {
            saved_inputs.insert(node_id, SavedActivationInput::Array(array));
        } else {
            let activation: Vec<f32> = value.extract()?;
            saved_inputs.insert(node_id, SavedActivationInput::Owned(activation));
        }
    }

    let mut saved: HashMap<u32, &[f32]> = HashMap::with_capacity(saved_inputs.len());
    for (node_id, values) in &saved_inputs {
        saved.insert(*node_id, values.as_slice()?);
    }

    let result = execute_backward_with_arena_slices(&graph, grad_output_slice, &saved)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

    let dict = PyDict::new_bound(py);
    let grads_dict = PyDict::new_bound(py);
    for (node_id, grad) in &result.grads {
        grads_dict.set_item(*node_id, grad.clone().into_pyarray_bound(py))?;
    }
    dict.set_item("grads", grads_dict)?;
    dict.set_item("arena_bytes_used", result.arena_stats.arena_bytes_used)?;

    Ok(dict.into())
}

#[pyfunction]
fn execute_graph_backward_arrays_handle<'py>(
    py: Python<'py>,
    json: &str,
    grad_output: PyReadonlyArrayDyn<'py, f32>,
    saved_state: PyRef<'py, SavedActivationStore>,
) -> PyResult<PyObject> {
    let graph = GraphIR::from_json(json)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    let grad_output_slice = readonly_array_slice(&grad_output)?;
    let saved: HashMap<u32, &[f32]> = saved_state
        .saved_activations
        .iter()
        .map(|(node_id, values)| (*node_id, values.as_slice()))
        .collect();
    let result = execute_backward_with_arena_slices(&graph, grad_output_slice, &saved)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

    let dict = PyDict::new_bound(py);
    let grads_dict = PyDict::new_bound(py);
    for (node_id, grad) in &result.grads {
        grads_dict.set_item(*node_id, grad.clone().into_pyarray_bound(py))?;
    }
    dict.set_item("grads", grads_dict)?;
    dict.set_item("arena_bytes_used", result.arena_stats.arena_bytes_used)?;
    Ok(dict.into())
}

#[pyfunction]
fn execute_graph_backward_compiled_arrays_handle<'py>(
    py: Python<'py>,
    graph: PyRef<'py, CompiledGraphHandle>,
    grad_output: PyReadonlyArrayDyn<'py, f32>,
    saved_state: PyRef<'py, SavedActivationStore>,
) -> PyResult<PyObject> {
    let grad_output_slice = readonly_array_slice(&grad_output)?;
    let saved: HashMap<u32, &[f32]> = saved_state
        .saved_activations
        .iter()
        .map(|(node_id, values)| (*node_id, values.as_slice()))
        .collect();
    let result = execute_backward_with_arena_slices(&graph.graph, grad_output_slice, &saved)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

    let dict = PyDict::new_bound(py);
    let grads_dict = PyDict::new_bound(py);
    for (node_id, grad) in &result.grads {
        grads_dict.set_item(*node_id, grad.clone().into_pyarray_bound(py))?;
    }
    dict.set_item("grads", grads_dict)?;
    dict.set_item("arena_bytes_used", result.arena_stats.arena_bytes_used)?;
    Ok(dict.into())
}

/// The Python module definition.
#[pymodule]
fn aria_scheduler(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<CompiledGraphHandle>()?;
    m.add_class::<SavedActivationStore>()?;
    m.add_class::<TemplateSelectorHandle>()?;
    m.add_function(wrap_pyfunction!(parse_graph_ir, m)?)?;
    m.add_function(wrap_pyfunction!(compile_graph_ir_handle, m)?)?;
    m.add_function(wrap_pyfunction!(compile_template_selector_handle, m)?)?;
    m.add_function(wrap_pyfunction!(select_template_index_compiled, m)?)?;
    m.add_function(wrap_pyfunction!(topological_order, m)?)?;
    m.add_function(wrap_pyfunction!(fingerprint_notebook_graph, m)?)?;
    m.add_function(wrap_pyfunction!(extract_graph_ops, m)?)?;
    m.add_function(wrap_pyfunction!(extract_graph_ops_batch, m)?)?;
    m.add_function(wrap_pyfunction!(extract_graph_feature_payload, m)?)?;
    m.add_function(wrap_pyfunction!(extract_graph_structure_features_native, m)?)?;
    m.add_function(wrap_pyfunction!(analyze_graph_provenance_native, m)?)?;
    m.add_function(wrap_pyfunction!(analyze_graph_provenance_native_py, m)?)?;
    m.add_function(wrap_pyfunction!(build_op_index_from_rows, m)?)?;
    m.add_function(wrap_pyfunction!(build_graph_training_corpus, m)?)?;
    m.add_function(wrap_pyfunction!(build_predictor_training_corpus, m)?)?;
    m.add_function(wrap_pyfunction!(extract_topology_features_native, m)?)?;
    m.add_function(wrap_pyfunction!(extract_topology_feature_map_native_py, m)?)?;
    m.add_function(wrap_pyfunction!(
        extract_topology_features_with_imodel_native,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        extract_topology_feature_map_with_imodel_native_py,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(extract_topology_features_batch_native, m)?)?;
    m.add_function(wrap_pyfunction!(
        extract_topology_feature_maps_batch_native_py,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        extract_topology_features_with_imodel_batch_native,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        extract_topology_feature_maps_with_imodel_batch_native_py,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(extract_edge_op_pairs_native, m)?)?;
    m.add_function(wrap_pyfunction!(extract_edge_op_pairs_batch_native, m)?)?;
    m.add_function(wrap_pyfunction!(extract_graph_segments_native, m)?)?;
    m.add_function(wrap_pyfunction!(extract_graph_segments_map_native_py, m)?)?;
    m.add_function(wrap_pyfunction!(train_interaction_model_native_py, m)?)?;
    m.add_function(wrap_pyfunction!(train_op_embeddings_epoch_native_py, m)?)?;
    m.add_function(wrap_pyfunction!(execute_graph, m)?)?;
    m.add_function(wrap_pyfunction!(execute_graph_with_stats, m)?)?;
    m.add_function(wrap_pyfunction!(execute_graph_with_stats_arrays, m)?)?;
    m.add_function(wrap_pyfunction!(execute_graph_with_stats_compiled_arrays, m)?)?;
    m.add_function(wrap_pyfunction!(execute_graph_compiled_arrays_handle, m)?)?;
    m.add_function(wrap_pyfunction!(execute_graph_multi_input, m)?)?;
    m.add_function(wrap_pyfunction!(execute_graph_multi_input_with_stats, m)?)?;
    m.add_function(wrap_pyfunction!(execute_graph_multi_input_arrays, m)?)?;
    m.add_function(wrap_pyfunction!(
        execute_graph_multi_input_compiled_arrays_handle,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        execute_graph_multi_input_arrays_with_stats,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        execute_graph_multi_input_compiled_arrays_with_stats,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(execute_graph_forward_saved, m)?)?;
    m.add_function(wrap_pyfunction!(execute_graph_forward_saved_handle, m)?)?;
    m.add_function(wrap_pyfunction!(execute_graph_forward_saved_arrays, m)?)?;
    m.add_function(wrap_pyfunction!(execute_graph_forward_saved_arrays_handle, m)?)?;
    m.add_function(wrap_pyfunction!(
        execute_graph_forward_saved_compiled_arrays_handle,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        execute_graph_forward_saved_multi_input,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        execute_graph_forward_saved_multi_input_arrays,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        execute_graph_forward_saved_multi_input_arrays_handle,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        execute_graph_forward_saved_multi_input_compiled_arrays_handle,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(execute_graph_backward, m)?)?;
    m.add_function(wrap_pyfunction!(execute_graph_backward_handle, m)?)?;
    m.add_function(wrap_pyfunction!(execute_graph_backward_arrays, m)?)?;
    m.add_function(wrap_pyfunction!(execute_graph_backward_arrays_handle, m)?)?;
    m.add_function(wrap_pyfunction!(
        execute_graph_backward_compiled_arrays_handle,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(arena_test, m)?)?;
    m.add_function(wrap_pyfunction!(profiler_enable, m)?)?;
    m.add_function(wrap_pyfunction!(profiler_enabled, m)?)?;
    m.add_function(wrap_pyfunction!(profiler_reset, m)?)?;
    Ok(())
}
