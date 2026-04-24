use std::collections::BTreeMap;
use std::collections::HashMap;

use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};

use crate::error::AriaError;
use crate::notebook_graph::{NotebookGraph, NotebookNode};

const HUBER_DELTA: f64 = 0.2;

#[derive(Default, Deserialize)]
struct OpProfile {
    #[serde(default)]
    output_std: Option<f64>,
    #[serde(default)]
    grad_norm: Option<f64>,
    #[serde(default)]
    lipschitz: Option<f64>,
    #[serde(default)]
    grad_vanishing: Option<f64>,
    #[serde(default)]
    grad_exploding: Option<f64>,
    #[serde(default)]
    has_nan: Option<f64>,
}

#[derive(Default, Deserialize)]
struct OpCategoryMeta {
    #[serde(default)]
    category: Option<String>,
    #[serde(default)]
    has_params: Option<bool>,
}

#[derive(Serialize)]
pub struct NativeInteractionTrainResult {
    pub u: Vec<f64>,
    pub u_rows: usize,
    pub u_cols: usize,
    pub v: Vec<f64>,
    pub v_rows: usize,
    pub v_cols: usize,
    pub w_s: Vec<f64>,
    pub ws_rows: usize,
    pub ws_cols: usize,
    pub w_l: Vec<f64>,
    pub wl_rows: usize,
    pub wl_cols: usize,
    pub b_s: f64,
    pub b_l: f64,
    pub best_loss: f64,
}

#[derive(Serialize)]
pub struct NativeEmbeddingEpochResult {
    pub embeddings: Vec<f64>,
    pub emb_rows: usize,
    pub emb_cols: usize,
    pub total_loss: f64,
    pub n_samples: usize,
}

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x.clamp(-15.0, 15.0)).exp())
}

fn parse_pair_stability_map(payload: &str) -> Result<HashMap<(String, String), f64>, AriaError> {
    let raw: HashMap<String, f64> =
        serde_json::from_str(payload).map_err(|e| AriaError::InvalidIR(e.to_string()))?;
    let mut out = HashMap::with_capacity(raw.len());
    for (key, value) in raw {
        if let Some((left, right)) = key.split_once('\t') {
            out.insert((left.to_string(), right.to_string()), value);
        }
    }
    Ok(out)
}

fn profile_lipschitz(profiles: &HashMap<String, OpProfile>, op: &str) -> f64 {
    profiles
        .get(op)
        .and_then(|p| p.lipschitz)
        .unwrap_or(1.0)
        .min(100.0)
}

fn profile_output_std(profiles: &HashMap<String, OpProfile>, op: &str) -> f64 {
    profiles
        .get(op)
        .and_then(|p| p.output_std)
        .unwrap_or(1.0)
        .min(10.0)
}

fn profile_grad_norm(profiles: &HashMap<String, OpProfile>, op: &str) -> f64 {
    profiles
        .get(op)
        .and_then(|p| p.grad_norm)
        .unwrap_or(1.0)
        .min(1000.0)
}

fn profile_grad_risk(profiles: &HashMap<String, OpProfile>, op: &str) -> f64 {
    let Some(profile) = profiles.get(op) else {
        return 0.0;
    };
    profile.grad_vanishing.unwrap_or(0.0)
        + profile.grad_exploding.unwrap_or(0.0)
        + profile.has_nan.unwrap_or(0.0)
}

struct TopologyIndex<'a> {
    sorted_nodes: Vec<&'a NotebookNode>,
    id_to_idx: HashMap<i64, usize>,
    op_names: Vec<&'a str>,
    children: Vec<Vec<usize>>,
    parents: Vec<Vec<usize>>,
}

fn build_topology_index(graph: &NotebookGraph) -> Result<TopologyIndex<'_>, AriaError> {
    let mut sorted_nodes: Vec<&NotebookNode> = graph.nodes.values().collect();
    sorted_nodes.sort_unstable_by_key(|node| node.id);
    if sorted_nodes.len() < 2 {
        return Err(AriaError::InvalidIR("graph has fewer than 2 nodes".into()));
    }

    let mut id_to_idx = HashMap::with_capacity(sorted_nodes.len());
    for (idx, node) in sorted_nodes.iter().enumerate() {
        id_to_idx.insert(node.id, idx);
    }

    let mut children: Vec<Vec<usize>> = vec![Vec::new(); sorted_nodes.len()];
    let mut parents: Vec<Vec<usize>> = vec![Vec::new(); sorted_nodes.len()];
    for (idx, node) in sorted_nodes.iter().enumerate() {
        for input_id in &node.input_ids {
            if let Some(&parent_idx) = id_to_idx.get(input_id) {
                children[parent_idx].push(idx);
                parents[idx].push(parent_idx);
            }
        }
    }

    let op_names = sorted_nodes
        .iter()
        .map(|node| node.op_name.as_str())
        .collect();

    Ok(TopologyIndex {
        sorted_nodes,
        id_to_idx,
        op_names,
        children,
        parents,
    })
}

pub fn extract_topology_features_json(
    graph_json: &str,
    op_profiles_json: &str,
    pair_stability_json: &str,
    op_metadata_json: &str,
) -> Result<String, AriaError> {
    let op_profiles: HashMap<String, OpProfile> =
        serde_json::from_str(op_profiles_json).map_err(|e| AriaError::InvalidIR(e.to_string()))?;
    let pair_stability = parse_pair_stability_map(pair_stability_json)?;
    let op_metadata: HashMap<String, OpCategoryMeta> =
        serde_json::from_str(op_metadata_json).map_err(|e| AriaError::InvalidIR(e.to_string()))?;
    let graph = NotebookGraph::from_json(graph_json)?;
    extract_topology_features_for_graph(&graph, &op_profiles, &pair_stability, &op_metadata)
}

pub fn extract_topology_features_with_imodel_json(
    graph_json: &str,
    op_profiles_json: &str,
    pair_stability_json: &str,
    op_metadata_json: &str,
    op_names: &[String],
    u: &[f32],
    u_rows: usize,
    u_cols: usize,
    v: &[f32],
    v_rows: usize,
    v_cols: usize,
    w_s: &[f32],
    ws_rows: usize,
    ws_cols: usize,
    w_l: &[f32],
    wl_rows: usize,
    wl_cols: usize,
    b_s: f64,
    b_l: f64,
) -> Result<String, AriaError> {
    let op_profiles: HashMap<String, OpProfile> =
        serde_json::from_str(op_profiles_json).map_err(|e| AriaError::InvalidIR(e.to_string()))?;
    let pair_stability = parse_pair_stability_map(pair_stability_json)?;
    let op_metadata: HashMap<String, OpCategoryMeta> =
        serde_json::from_str(op_metadata_json).map_err(|e| AriaError::InvalidIR(e.to_string()))?;
    let kernel = InteractionModelKernel::new(
        op_names, u, u_rows, u_cols, v, v_rows, v_cols, w_s, ws_rows, ws_cols, w_l, wl_rows,
        wl_cols, b_s, b_l,
    )?;
    let graph = NotebookGraph::from_json(graph_json)?;
    let topo = build_topology_index(&graph)?;
    let mut features = collect_topology_features_for_graph(
        &graph,
        &topo,
        &op_profiles,
        &pair_stability,
        &op_metadata,
    )?;
    let (min_stability, mean_stability, mean_loss) = kernel.pair_stats_for_topology(&topo);
    features.insert("imodel_min_stability".into(), min_stability);
    features.insert("imodel_mean_stability".into(), mean_stability);
    features.insert("imodel_mean_loss".into(), mean_loss);
    serde_json::to_string(&features).map_err(|e| AriaError::ExecutionFailed(e.to_string()))
}

pub fn extract_topology_feature_map(
    graph_json: &str,
    op_profiles_json: &str,
    pair_stability_json: &str,
    op_metadata_json: &str,
) -> Result<HashMap<String, f64>, AriaError> {
    let op_profiles: HashMap<String, OpProfile> =
        serde_json::from_str(op_profiles_json).map_err(|e| AriaError::InvalidIR(e.to_string()))?;
    let pair_stability = parse_pair_stability_map(pair_stability_json)?;
    let op_metadata: HashMap<String, OpCategoryMeta> =
        serde_json::from_str(op_metadata_json).map_err(|e| AriaError::InvalidIR(e.to_string()))?;
    let graph = NotebookGraph::from_json(graph_json)?;
    let topo = build_topology_index(&graph)?;
    collect_topology_features_for_graph(&graph, &topo, &op_profiles, &pair_stability, &op_metadata)
}

pub fn extract_topology_feature_map_with_imodel(
    graph_json: &str,
    op_profiles_json: &str,
    pair_stability_json: &str,
    op_metadata_json: &str,
    op_names: &[String],
    u: &[f32],
    u_rows: usize,
    u_cols: usize,
    v: &[f32],
    v_rows: usize,
    v_cols: usize,
    w_s: &[f32],
    ws_rows: usize,
    ws_cols: usize,
    w_l: &[f32],
    wl_rows: usize,
    wl_cols: usize,
    b_s: f64,
    b_l: f64,
) -> Result<HashMap<String, f64>, AriaError> {
    let op_profiles: HashMap<String, OpProfile> =
        serde_json::from_str(op_profiles_json).map_err(|e| AriaError::InvalidIR(e.to_string()))?;
    let pair_stability = parse_pair_stability_map(pair_stability_json)?;
    let op_metadata: HashMap<String, OpCategoryMeta> =
        serde_json::from_str(op_metadata_json).map_err(|e| AriaError::InvalidIR(e.to_string()))?;
    let kernel = InteractionModelKernel::new(
        op_names, u, u_rows, u_cols, v, v_rows, v_cols, w_s, ws_rows, ws_cols, w_l, wl_rows,
        wl_cols, b_s, b_l,
    )?;
    let graph = NotebookGraph::from_json(graph_json)?;
    let topo = build_topology_index(&graph)?;
    let mut features = collect_topology_features_for_graph(
        &graph,
        &topo,
        &op_profiles,
        &pair_stability,
        &op_metadata,
    )?;
    let (min_stability, mean_stability, mean_loss) = kernel.pair_stats_for_topology(&topo);
    features.insert("imodel_min_stability".into(), min_stability);
    features.insert("imodel_mean_stability".into(), mean_stability);
    features.insert("imodel_mean_loss".into(), mean_loss);
    Ok(features)
}

pub fn extract_topology_features_batch_json(
    graphs: &[String],
    op_profiles_json: &str,
    pair_stability_json: &str,
    op_metadata_json: &str,
) -> Result<Vec<String>, AriaError> {
    let op_profiles: HashMap<String, OpProfile> =
        serde_json::from_str(op_profiles_json).map_err(|e| AriaError::InvalidIR(e.to_string()))?;
    let pair_stability = parse_pair_stability_map(pair_stability_json)?;
    let op_metadata: HashMap<String, OpCategoryMeta> =
        serde_json::from_str(op_metadata_json).map_err(|e| AriaError::InvalidIR(e.to_string()))?;
    let mut results = Vec::with_capacity(graphs.len());
    for graph_json in graphs {
        let graph = NotebookGraph::from_json(graph_json)?;
        results.push(extract_topology_features_for_graph(
            &graph,
            &op_profiles,
            &pair_stability,
            &op_metadata,
        )?);
    }
    Ok(results)
}

pub fn extract_topology_feature_maps_batch(
    graphs: &[String],
    op_profiles_json: &str,
    pair_stability_json: &str,
    op_metadata_json: &str,
) -> Result<Vec<HashMap<String, f64>>, AriaError> {
    let op_profiles: HashMap<String, OpProfile> =
        serde_json::from_str(op_profiles_json).map_err(|e| AriaError::InvalidIR(e.to_string()))?;
    let pair_stability = parse_pair_stability_map(pair_stability_json)?;
    let op_metadata: HashMap<String, OpCategoryMeta> =
        serde_json::from_str(op_metadata_json).map_err(|e| AriaError::InvalidIR(e.to_string()))?;
    let mut results = Vec::with_capacity(graphs.len());
    for graph_json in graphs {
        let graph = NotebookGraph::from_json(graph_json)?;
        let topo = build_topology_index(&graph)?;
        results.push(collect_topology_features_for_graph(
            &graph,
            &topo,
            &op_profiles,
            &pair_stability,
            &op_metadata,
        )?);
    }
    Ok(results)
}

struct InteractionModelKernel<'a> {
    op_to_idx: HashMap<&'a str, usize>,
    u: &'a [f32],
    v: &'a [f32],
    w_s: &'a [f32],
    w_l: &'a [f32],
    n_ops: usize,
    dim: usize,
    b_s: f64,
    b_l: f64,
}

impl<'a> InteractionModelKernel<'a> {
    fn new(
        op_names: &'a [String],
        u: &'a [f32],
        u_rows: usize,
        u_cols: usize,
        v: &'a [f32],
        v_rows: usize,
        v_cols: usize,
        w_s: &'a [f32],
        ws_rows: usize,
        ws_cols: usize,
        w_l: &'a [f32],
        wl_rows: usize,
        wl_cols: usize,
        b_s: f64,
        b_l: f64,
    ) -> Result<Self, AriaError> {
        if op_names.len() != u_rows || op_names.len() != v_rows {
            return Err(AriaError::InvalidIR(
                "interaction model op registry shape mismatch".into(),
            ));
        }
        if u_cols == 0 || v_cols == 0 || u_cols != v_cols {
            return Err(AriaError::InvalidIR(
                "interaction model embedding dims must match".into(),
            ));
        }
        if ws_rows != u_cols || ws_cols != u_cols || wl_rows != u_cols || wl_cols != u_cols {
            return Err(AriaError::InvalidIR(
                "interaction model bilinear kernels must be square embedding-dim matrices".into(),
            ));
        }
        if u.len() != u_rows * u_cols
            || v.len() != v_rows * v_cols
            || w_s.len() != ws_rows * ws_cols
            || w_l.len() != wl_rows * wl_cols
        {
            return Err(AriaError::InvalidIR(
                "interaction model tensor buffer shape mismatch".into(),
            ));
        }
        let mut op_to_idx = HashMap::with_capacity(op_names.len());
        for (idx, op_name) in op_names.iter().enumerate() {
            op_to_idx.insert(op_name.as_str(), idx);
        }
        Ok(Self {
            op_to_idx,
            u,
            v,
            w_s,
            w_l,
            n_ops: op_names.len(),
            dim: u_cols,
            b_s,
            b_l,
        })
    }

    fn bilinear_score(&self, left_idx: usize, right_idx: usize, weights: &[f32]) -> f64 {
        let left_start = left_idx * self.dim;
        let right_start = right_idx * self.dim;
        let left = &self.u[left_start..left_start + self.dim];
        let right = &self.v[right_start..right_start + self.dim];
        let mut total = 0.0_f64;
        for row in 0..self.dim {
            let lhs = left[row] as f64;
            let weight_row = &weights[row * self.dim..(row + 1) * self.dim];
            let mut inner = 0.0_f64;
            for col in 0..self.dim {
                inner += weight_row[col] as f64 * right[col] as f64;
            }
            total += lhs * inner;
        }
        total
    }

    fn pair_stats_for_topology(&self, topo: &TopologyIndex<'_>) -> (f64, f64, f64) {
        let mut pair_count = 0usize;
        let mut min_stability = 1.0_f64;
        let mut sum_stability = 0.0_f64;
        let mut sum_loss = 0.0_f64;
        for (idx, children) in topo.children.iter().enumerate() {
            let left = topo.op_names[idx];
            if left.is_empty() || left == "input" {
                continue;
            }
            for &child_idx in children {
                let right = topo.op_names[child_idx];
                if right.is_empty() || right == "input" {
                    continue;
                }
                let mut stability = 0.5_f64;
                let mut loss = 0.7_f64;
                if let (Some(&left_idx), Some(&right_idx)) =
                    (self.op_to_idx.get(left), self.op_to_idx.get(right))
                {
                    if left_idx < self.n_ops && right_idx < self.n_ops {
                        stability =
                            sigmoid(self.bilinear_score(left_idx, right_idx, self.w_s) + self.b_s);
                        loss = self.bilinear_score(left_idx, right_idx, self.w_l) + self.b_l;
                    }
                }
                pair_count += 1;
                min_stability = min_stability.min(stability);
                sum_stability += stability;
                sum_loss += loss;
            }
        }
        if pair_count == 0 {
            return (0.5, 0.5, 0.7);
        }
        (
            min_stability,
            sum_stability / pair_count as f64,
            sum_loss / pair_count as f64,
        )
    }
}

pub fn extract_topology_features_with_imodel_batch(
    graphs: &[String],
    op_profiles_json: &str,
    pair_stability_json: &str,
    op_metadata_json: &str,
    op_names: &[String],
    u: &[f32],
    u_rows: usize,
    u_cols: usize,
    v: &[f32],
    v_rows: usize,
    v_cols: usize,
    w_s: &[f32],
    ws_rows: usize,
    ws_cols: usize,
    w_l: &[f32],
    wl_rows: usize,
    wl_cols: usize,
    b_s: f64,
    b_l: f64,
) -> Result<Vec<String>, AriaError> {
    let op_profiles: HashMap<String, OpProfile> =
        serde_json::from_str(op_profiles_json).map_err(|e| AriaError::InvalidIR(e.to_string()))?;
    let pair_stability = parse_pair_stability_map(pair_stability_json)?;
    let op_metadata: HashMap<String, OpCategoryMeta> =
        serde_json::from_str(op_metadata_json).map_err(|e| AriaError::InvalidIR(e.to_string()))?;
    let kernel = InteractionModelKernel::new(
        op_names, u, u_rows, u_cols, v, v_rows, v_cols, w_s, ws_rows, ws_cols, w_l, wl_rows,
        wl_cols, b_s, b_l,
    )?;
    let mut results = Vec::with_capacity(graphs.len());
    for graph_json in graphs {
        let graph = NotebookGraph::from_json(graph_json)?;
        let topo = build_topology_index(&graph)?;
        let mut features = collect_topology_features_for_graph(
            &graph,
            &topo,
            &op_profiles,
            &pair_stability,
            &op_metadata,
        )?;
        let (min_stability, mean_stability, mean_loss) = kernel.pair_stats_for_topology(&topo);
        features.insert("imodel_min_stability".into(), min_stability);
        features.insert("imodel_mean_stability".into(), mean_stability);
        features.insert("imodel_mean_loss".into(), mean_loss);
        results.push(
            serde_json::to_string(&features)
                .map_err(|e| AriaError::ExecutionFailed(e.to_string()))?,
        );
    }
    Ok(results)
}

pub fn extract_topology_feature_maps_with_imodel_batch(
    graphs: &[String],
    op_profiles_json: &str,
    pair_stability_json: &str,
    op_metadata_json: &str,
    op_names: &[String],
    u: &[f32],
    u_rows: usize,
    u_cols: usize,
    v: &[f32],
    v_rows: usize,
    v_cols: usize,
    w_s: &[f32],
    ws_rows: usize,
    ws_cols: usize,
    w_l: &[f32],
    wl_rows: usize,
    wl_cols: usize,
    b_s: f64,
    b_l: f64,
) -> Result<Vec<HashMap<String, f64>>, AriaError> {
    let op_profiles: HashMap<String, OpProfile> =
        serde_json::from_str(op_profiles_json).map_err(|e| AriaError::InvalidIR(e.to_string()))?;
    let pair_stability = parse_pair_stability_map(pair_stability_json)?;
    let op_metadata: HashMap<String, OpCategoryMeta> =
        serde_json::from_str(op_metadata_json).map_err(|e| AriaError::InvalidIR(e.to_string()))?;
    let kernel = InteractionModelKernel::new(
        op_names, u, u_rows, u_cols, v, v_rows, v_cols, w_s, ws_rows, ws_cols, w_l, wl_rows,
        wl_cols, b_s, b_l,
    )?;
    let mut results = Vec::with_capacity(graphs.len());
    for graph_json in graphs {
        let graph = NotebookGraph::from_json(graph_json)?;
        let topo = build_topology_index(&graph)?;
        let mut features = collect_topology_features_for_graph(
            &graph,
            &topo,
            &op_profiles,
            &pair_stability,
            &op_metadata,
        )?;
        let (min_stability, mean_stability, mean_loss) = kernel.pair_stats_for_topology(&topo);
        features.insert("imodel_min_stability".into(), min_stability);
        features.insert("imodel_mean_stability".into(), mean_stability);
        features.insert("imodel_mean_loss".into(), mean_loss);
        results.push(features);
    }
    Ok(results)
}

fn extract_topology_features_for_graph(
    graph: &NotebookGraph,
    op_profiles: &HashMap<String, OpProfile>,
    pair_stability: &HashMap<(String, String), f64>,
    op_metadata: &HashMap<String, OpCategoryMeta>,
) -> Result<String, AriaError> {
    let topo = build_topology_index(&graph)?;
    let features = collect_topology_features_for_graph(
        graph,
        &topo,
        op_profiles,
        pair_stability,
        op_metadata,
    )?;
    serde_json::to_string(&features).map_err(|e| AriaError::ExecutionFailed(e.to_string()))
}

fn collect_topology_features_for_graph(
    graph: &NotebookGraph,
    topo: &TopologyIndex<'_>,
    op_profiles: &HashMap<String, OpProfile>,
    pair_stability: &HashMap<(String, String), f64>,
    op_metadata: &HashMap<String, OpCategoryMeta>,
) -> Result<HashMap<String, f64>, AriaError> {
    let n = topo.sorted_nodes.len();
    let op_names = &topo.op_names;
    let id_to_idx = &topo.id_to_idx;
    let children = &topo.children;
    let parents = &topo.parents;

    let mut roots: Vec<usize> = parents
        .iter()
        .enumerate()
        .filter_map(|(idx, p)| if p.is_empty() { Some(idx) } else { None })
        .collect();
    if roots.is_empty() {
        roots.push(0);
    }

    let mut depth = vec![-1_i32; n];
    let mut queue = roots.clone();
    for &root in &roots {
        depth[root] = 0;
    }
    let mut qi = 0usize;
    while qi < queue.len() {
        let idx = queue[qi];
        qi += 1;
        let next_depth = depth[idx] + 1;
        for &child in &children[idx] {
            if depth[child] < 0 {
                depth[child] = next_depth;
                queue.push(child);
            }
        }
    }
    let max_depth = depth.iter().copied().max().unwrap_or(1).max(1) as f64;

    let n_ops = op_names
        .iter()
        .copied()
        .filter(|op| !op.is_empty() && *op != "input")
        .count() as f64;
    let n_edges: usize = children.iter().map(|ch| ch.len()).sum();

    let mut features: HashMap<String, f64> = HashMap::new();
    features.insert("topo_n_ops".into(), n_ops);
    features.insert("topo_depth".into(), max_depth);
    features.insert("topo_edge_density".into(), n_edges as f64 / n.max(1) as f64);
    features.insert("topo_edges_per_op".into(), n_edges as f64 / n_ops.max(1.0));
    features.insert("topo_depth_per_op".into(), max_depth / n_ops.max(1.0));

    let fan_ins: Vec<usize> = parents.iter().map(|p| p.len()).collect();
    let fan_outs: Vec<usize> = children.iter().map(|ch| ch.len()).collect();
    let max_fan_in = fan_ins.iter().copied().max().unwrap_or(0) as f64;
    let max_fan_out = fan_outs.iter().copied().max().unwrap_or(0) as f64;
    let mean_fan_in = if fan_ins.is_empty() {
        0.0
    } else {
        fan_ins.iter().sum::<usize>() as f64 / fan_ins.len() as f64
    };
    let mean_fan_out = if fan_outs.is_empty() {
        0.0
    } else {
        fan_outs.iter().sum::<usize>() as f64 / fan_outs.len() as f64
    };
    features.insert("topo_max_fan_in".into(), max_fan_in);
    features.insert("topo_max_fan_out".into(), max_fan_out);
    features.insert("topo_mean_fan_in".into(), mean_fan_in);
    features.insert("topo_mean_fan_out".into(), mean_fan_out);
    features.insert(
        "topo_n_merge_nodes".into(),
        fan_ins.iter().filter(|&&v| v > 1).count() as f64,
    );
    features.insert(
        "topo_n_split_nodes".into(),
        fan_outs.iter().filter(|&&v| v > 1).count() as f64,
    );
    features.insert(
        "topo_leaf_fraction".into(),
        fan_outs.iter().filter(|&&v| v == 0).count() as f64 / n.max(1) as f64,
    );
    features.insert(
        "topo_root_fraction".into(),
        fan_ins.iter().filter(|&&v| v == 0).count() as f64 / n.max(1) as f64,
    );

    let mut lip_values = vec![1.0_f64; n];
    let mut grad_risks = vec![0.0_f64; n];
    for (idx, &op) in op_names.iter().enumerate() {
        lip_values[idx] = profile_lipschitz(&op_profiles, op);
        grad_risks[idx] = profile_grad_risk(&op_profiles, op);
    }

    let mut lip_product = vec![1.0_f64; n];
    for &idx in &queue {
        if !parents[idx].is_empty() {
            let max_parent_lip = parents[idx]
                .iter()
                .map(|&p| lip_product[p])
                .fold(1.0, f64::max);
            lip_product[idx] = max_parent_lip * lip_values[idx];
        }
    }
    let max_lip_product = lip_product
        .iter()
        .copied()
        .fold(0.0_f64, f64::max)
        .clamp(0.0, 1e6);
    let mean_lip_product =
        (lip_product.iter().sum::<f64>() / lip_product.len().max(1) as f64).clamp(0.0, 1e6);
    features.insert("path_max_lip_product".into(), max_lip_product);
    features.insert("path_mean_lip_product".into(), mean_lip_product);
    features.insert("path_max_lip_log".into(), (1.0 + max_lip_product).ln());

    let mut risk_accum = vec![0.0_f64; n];
    for &idx in &queue {
        if !parents[idx].is_empty() {
            let max_parent_risk = parents[idx]
                .iter()
                .map(|&p| risk_accum[p])
                .fold(0.0, f64::max);
            risk_accum[idx] = max_parent_risk + grad_risks[idx];
        }
    }
    features.insert(
        "path_max_risk_accum".into(),
        risk_accum.iter().copied().fold(0.0_f64, f64::max),
    );
    features.insert(
        "path_mean_risk_accum".into(),
        risk_accum.iter().sum::<f64>() / risk_accum.len().max(1) as f64,
    );

    let mut depth_weights = vec![0.0_f64; n];
    for idx in 0..n {
        if depth[idx] >= 0 {
            depth_weights[idx] = (depth[idx] as f64 + 1.0) / (max_depth + 1.0);
        }
    }

    let mut weighted_lip = 0.0_f64;
    let mut weighted_std = 0.0_f64;
    let mut weighted_grad = 0.0_f64;
    let mut weight_sum = 0.0_f64;
    for (idx, &op) in op_names.iter().enumerate() {
        if op.is_empty() || op == "input" {
            continue;
        }
        let weight = depth_weights[idx];
        weighted_lip += weight * profile_lipschitz(&op_profiles, op);
        weighted_std += weight * profile_output_std(&op_profiles, op);
        weighted_grad += weight * profile_grad_norm(&op_profiles, op);
        weight_sum += weight;
    }
    let denom = weight_sum.max(1e-8);
    features.insert("depth_weighted_lip".into(), weighted_lip / denom);
    features.insert("depth_weighted_std".into(), weighted_std / denom);
    features.insert(
        "depth_weighted_grad".into(),
        (1.0 + (weighted_grad / denom)).ln(),
    );

    let mut pair_values = Vec::new();
    for idx in 0..n {
        for &child_idx in &children[idx] {
            let a = op_names[idx];
            let b = op_names[child_idx];
            if a.is_empty() || b.is_empty() || a == "input" || b == "input" {
                continue;
            }
            pair_values.push(
                *pair_stability
                    .get(&(a.to_string(), b.to_string()))
                    .unwrap_or(&0.5),
            );
        }
    }
    if pair_values.is_empty() {
        features.insert("pair_min_stability".into(), 0.5);
        features.insert("pair_mean_stability".into(), 0.5);
        features.insert("pair_frac_unstable".into(), 0.5);
    } else {
        let min_stability = pair_values.iter().copied().fold(1.0_f64, f64::min);
        let mean_stability = pair_values.iter().sum::<f64>() / pair_values.len() as f64;
        let frac_unstable =
            pair_values.iter().filter(|&&v| v < 0.5).count() as f64 / pair_values.len() as f64;
        features.insert("pair_min_stability".into(), min_stability);
        features.insert("pair_mean_stability".into(), mean_stability);
        features.insert("pair_frac_unstable".into(), frac_unstable);
    }

    let mut child_lip_means = Vec::new();
    for child_list in children {
        if child_list.is_empty() {
            continue;
        }
        let mean = child_list
            .iter()
            .map(|&child_idx| lip_values[child_idx])
            .sum::<f64>()
            / child_list.len() as f64;
        child_lip_means.push(mean);
    }
    features.insert(
        "neighbor_mean_child_lip".into(),
        if child_lip_means.is_empty() {
            1.0
        } else {
            child_lip_means.iter().sum::<f64>() / child_lip_means.len() as f64
        },
    );
    features.insert(
        "neighbor_max_child_lip".into(),
        child_lip_means.iter().copied().fold(1.0_f64, f64::max),
    );

    let mut n_skip = 0usize;
    let mut skip_spans = Vec::new();
    for idx in 0..n {
        if op_names[idx] == "add" && parents[idx].len() >= 2 {
            let mut local_depths = Vec::new();
            for &parent_idx in &parents[idx] {
                if depth[parent_idx] >= 0 {
                    local_depths.push(depth[parent_idx] as f64);
                }
            }
            if !local_depths.is_empty() {
                let min_depth = local_depths.iter().copied().fold(f64::INFINITY, f64::min);
                let max_depth_local = local_depths
                    .iter()
                    .copied()
                    .fold(f64::NEG_INFINITY, f64::max);
                if max_depth_local - min_depth > 0.0 {
                    n_skip += 1;
                    skip_spans.push(max_depth_local - min_depth);
                }
            }
        }
    }
    features.insert("residual_coverage".into(), n_skip as f64 / n_ops.max(1.0));
    features.insert(
        "residual_span_mean".into(),
        if skip_spans.is_empty() {
            0.0
        } else {
            skip_spans.iter().sum::<f64>() / skip_spans.len() as f64
        },
    );
    features.insert(
        "residual_span_max".into(),
        skip_spans.iter().copied().fold(0.0_f64, f64::max),
    );

    let mut early = 0usize;
    let mut mid = 0usize;
    let mut late = 0usize;
    let mut mixing = 0usize;
    let mut math_space = 0usize;
    let mut parameterized = 0usize;
    let mut reduction = 0usize;
    let mut param_ops = 0usize;
    let mut math_late = 0.0_f64;
    let mut mixing_late = 0.0_f64;
    for (idx, &op) in op_names.iter().enumerate() {
        if op.is_empty() || op == "input" {
            continue;
        }
        let d_norm = depth_weights[idx];
        if d_norm <= 0.34 {
            early += 1;
        } else if d_norm <= 0.67 {
            mid += 1;
        } else {
            late += 1;
        }
        if let Some(meta) = op_metadata.get(op) {
            let category = meta.category.as_deref().unwrap_or("").to_ascii_lowercase();
            if category.contains("mix") {
                mixing += 1;
                mixing_late += d_norm;
            } else if category.contains("math") {
                math_space += 1;
                math_late += d_norm;
            } else if category.contains("param") {
                parameterized += 1;
            } else if category.contains("reduction") {
                reduction += 1;
            }
            if meta.has_params.unwrap_or(false) {
                param_ops += 1;
            }
        }
    }
    features.insert("depth_frac_early".into(), early as f64 / n_ops.max(1.0));
    features.insert("depth_frac_mid".into(), mid as f64 / n_ops.max(1.0));
    features.insert("depth_frac_late".into(), late as f64 / n_ops.max(1.0));
    features.insert("cat_frac_mixing".into(), mixing as f64 / n_ops.max(1.0));
    features.insert(
        "cat_frac_math_space".into(),
        math_space as f64 / n_ops.max(1.0),
    );
    features.insert(
        "cat_frac_parameterized".into(),
        parameterized as f64 / n_ops.max(1.0),
    );
    features.insert(
        "cat_frac_reduction".into(),
        reduction as f64 / n_ops.max(1.0),
    );
    features.insert(
        "param_op_fraction".into(),
        param_ops as f64 / n_ops.max(1.0),
    );
    features.insert("late_mixing_density".into(), mixing_late / n_ops.max(1.0));
    features.insert("late_math_density".into(), math_late / n_ops.max(1.0));

    let templates_len = graph
        .metadata
        .get("templates_used")
        .and_then(|value| value.as_array())
        .map(|items| items.len())
        .unwrap_or(0);
    let motifs_len = graph
        .metadata
        .get("motifs_used")
        .and_then(|value| value.as_array())
        .map(|items| items.len())
        .unwrap_or(0);
    features.insert("meta_n_templates".into(), templates_len as f64);
    features.insert("meta_n_motifs".into(), motifs_len as f64);
    features.insert(
        "meta_template_per_op".into(),
        templates_len as f64 / n_ops.max(1.0),
    );
    features.insert(
        "meta_motif_per_op".into(),
        motifs_len as f64 / n_ops.max(1.0),
    );

    if let Some(output_id) = graph.output_node_id {
        if let Some(&output_idx) = id_to_idx.get(&output_id) {
            let output_parents = &parents[output_idx];
            let has_norm = output_parents.iter().any(|&parent_idx| {
                matches!(
                    op_names[parent_idx],
                    "layernorm" | "rmsnorm" | "layer_norm" | "rms_norm"
                )
            });
            let mut parent_depths = Vec::new();
            for &parent_idx in output_parents {
                if depth[parent_idx] >= 0 {
                    parent_depths.push(depth[parent_idx] as f64);
                }
            }
            features.insert(
                "has_norm_before_output".into(),
                if has_norm { 1.0 } else { 0.0 },
            );
            features.insert(
                "output_parent_depth_mean".into(),
                if parent_depths.is_empty() {
                    0.0
                } else {
                    parent_depths.iter().sum::<f64>() / parent_depths.len() as f64
                },
            );
        } else {
            features.insert("has_norm_before_output".into(), 0.0);
            features.insert("output_parent_depth_mean".into(), 0.0);
        }
    } else {
        features.insert("has_norm_before_output".into(), 0.0);
        features.insert("output_parent_depth_mean".into(), 0.0);
    }

    Ok(features)
}

pub fn extract_edge_op_pairs_json(graph_json: &str) -> Result<String, AriaError> {
    let graph = NotebookGraph::from_json(graph_json)?;
    extract_edge_op_pairs_for_graph(&graph)
}

pub fn extract_edge_op_pairs_batch_json(graphs: &[String]) -> Result<Vec<String>, AriaError> {
    let mut results = Vec::with_capacity(graphs.len());
    for graph_json in graphs {
        let graph = NotebookGraph::from_json(graph_json)?;
        results.push(extract_edge_op_pairs_for_graph(&graph)?);
    }
    Ok(results)
}

fn extract_edge_op_pairs_for_graph(graph: &NotebookGraph) -> Result<String, AriaError> {
    let topo = build_topology_index(&graph)?;
    let mut pairs: Vec<(String, String)> = Vec::new();
    for (idx, children) in topo.children.iter().enumerate() {
        let left = topo.op_names[idx];
        if left.is_empty() || left == "input" {
            continue;
        }
        for &child_idx in children {
            let right = topo.op_names[child_idx];
            if right.is_empty() || right == "input" {
                continue;
            }
            pairs.push((left.to_string(), right.to_string()));
        }
    }
    serde_json::to_string(&pairs).map_err(|e| AriaError::ExecutionFailed(e.to_string()))
}

pub fn extract_graph_segments_json(
    graph_json: &str,
    min_len: usize,
    max_len: usize,
) -> Result<String, AriaError> {
    let counts = extract_graph_segments_map(graph_json, min_len, max_len)?;
    serde_json::to_string(&counts).map_err(|e| AriaError::ExecutionFailed(e.to_string()))
}

pub fn extract_graph_segments_map(
    graph_json: &str,
    min_len: usize,
    max_len: usize,
) -> Result<BTreeMap<String, usize>, AriaError> {
    let graph = NotebookGraph::from_json(graph_json)?;
    let topo = build_topology_index(&graph)?;
    let mut counts: BTreeMap<String, usize> = BTreeMap::new();
    let mut path: Vec<usize> = Vec::new();

    fn visit(
        idx: usize,
        topo: &TopologyIndex<'_>,
        min_len: usize,
        max_len: usize,
        path: &mut Vec<usize>,
        counts: &mut BTreeMap<String, usize>,
    ) {
        let op_name = topo.op_names[idx];
        if op_name.is_empty() || op_name == "input" {
            return;
        }
        path.push(idx);

        let path_len = path.len();
        if path_len >= min_len {
            let mut fragment = format!("seg_p{}:", path_len);
            for (pos, &node_idx) in path.iter().enumerate() {
                if pos > 0 {
                    fragment.push('>');
                }
                fragment.push_str(topo.op_names[node_idx]);
            }
            *counts.entry(fragment).or_insert(0) += 1;
        }

        if path_len < max_len {
            for &child_idx in &topo.children[idx] {
                let child_op = topo.op_names[child_idx];
                if child_op.is_empty() || child_op == "input" || path.contains(&child_idx) {
                    continue;
                }
                visit(child_idx, topo, min_len, max_len, path, counts);
            }
        }

        path.pop();
    }

    for idx in 0..topo.sorted_nodes.len() {
        let op_name = topo.op_names[idx];
        if op_name.is_empty() || op_name == "input" {
            continue;
        }
        visit(idx, &topo, min_len, max_len, &mut path, &mut counts);
    }

    Ok(counts)
}

fn checked_flat_copy(flat: &[f64], rows: usize, cols: usize) -> Result<Vec<f64>, AriaError> {
    if rows * cols != flat.len() {
        return Err(AriaError::ExecutionFailed(format!(
            "matrix shape mismatch rows={} cols={} len={}",
            rows,
            cols,
            flat.len()
        )));
    }
    Ok(flat.to_vec())
}

fn dot_row_matrix_flat(
    row: &[f64],
    matrix: &[f64],
    matrix_rows: usize,
    matrix_cols: usize,
    out: &mut [f64],
) {
    for item in out.iter_mut() {
        *item = 0.0;
    }
    let rows = matrix_rows.min(row.len());
    for (k, &row_value) in row.iter().take(rows).enumerate() {
        let base = k * matrix_cols;
        for col in 0..matrix_cols.min(out.len()) {
            out[col] += row_value * matrix[base + col];
        }
    }
}

#[inline]
fn row_offset(row_idx: usize, cols: usize) -> usize {
    row_idx * cols
}

pub fn train_interaction_model_native(
    u_init: &[f64],
    u_rows: usize,
    u_cols: usize,
    v_init: &[f64],
    v_rows: usize,
    v_cols: usize,
    w_s_init: &[f64],
    ws_rows: usize,
    ws_cols: usize,
    w_l_init: &[f64],
    wl_rows: usize,
    wl_cols: usize,
    mut b_s: f64,
    mut b_l: f64,
    stab_idx: &[i32],
    stab_rows: usize,
    stab_labels: &[f64],
    stab_weights: &[f64],
    loss_idx: &[i32],
    loss_rows: usize,
    loss_labels: &[f64],
    loss_weights: &[f64],
    n_epochs: usize,
    lr: f64,
    batch_size: usize,
    seed: u64,
) -> Result<NativeInteractionTrainResult, AriaError> {
    let mut u = checked_flat_copy(u_init, u_rows, u_cols)?;
    let mut v = checked_flat_copy(v_init, v_rows, v_cols)?;
    let mut w_s = checked_flat_copy(w_s_init, ws_rows, ws_cols)?;
    let mut w_l = checked_flat_copy(w_l_init, wl_rows, wl_cols)?;
    if stab_idx.len() != stab_rows * 2
        || stab_labels.len() != stab_rows
        || stab_weights.len() != stab_rows
        || loss_idx.len() != loss_rows * 2
        || loss_labels.len() != loss_rows
        || loss_weights.len() != loss_rows
    {
        return Err(AriaError::ExecutionFailed(
            "interaction input shape mismatch".into(),
        ));
    }

    let mut rng = StdRng::seed_from_u64(seed);
    let d = u_cols;
    let mut best_loss = f64::INFINITY;
    let mut stab_perm: Vec<usize> = (0..stab_rows).collect();
    let mut loss_perm: Vec<usize> = (0..loss_rows).collect();

    for _epoch in 0..n_epochs {
        let mut total_loss = 0.0_f64;
        stab_perm.shuffle(&mut rng);

        let mut u_w = vec![0.0_f64; d];
        let mut du = vec![0.0_f64; d];
        let mut dv = vec![0.0_f64; d];
        let mut d_w = vec![0.0_f64; d * d];

        for batch_start in (0..stab_rows).step_by(batch_size.max(1)) {
            for value in &mut d_w {
                *value = 0.0;
            }
            let end = (batch_start + batch_size.max(1)).min(stab_rows);
            let bs = (end - batch_start).max(1) as f64;
            let mut db_accum = 0.0_f64;
            for &sample_idx in &stab_perm[batch_start..end] {
                let i = stab_idx[sample_idx * 2] as usize;
                let j = stab_idx[sample_idx * 2 + 1] as usize;
                let target = stab_labels[sample_idx];
                let weight = stab_weights[sample_idx];
                let u_base = row_offset(i, d);
                let v_base = row_offset(j, d);
                dot_row_matrix_flat(&u[u_base..u_base + d], &w_s, ws_rows, ws_cols, &mut u_w);
                let mut logit = b_s;
                for k in 0..d {
                    logit += u_w[k] * v[v_base + k];
                }
                let pred = sigmoid(logit);
                let eps = 1e-8_f64;
                total_loss += -weight
                    * (target * (pred + eps).ln() + (1.0 - target) * (1.0 - pred + eps).ln());
                let d_logit = (pred - target) * weight;
                db_accum += d_logit;
                for a in 0..d {
                    let u_val = u[u_base + a];
                    for b in 0..d {
                        d_w[a * d + b] += u_val * d_logit * v[v_base + b];
                    }
                }
                for k in 0..d {
                    du[k] = 0.0;
                    dv[k] = d_logit * u_w[k];
                }
                for a in 0..d {
                    let mut acc = 0.0_f64;
                    for b in 0..d {
                        acc += v[v_base + b] * w_s[a * d + b];
                    }
                    du[a] = d_logit * acc;
                }
                for k in 0..d {
                    u[u_base + k] -= lr * du[k] / bs;
                    v[v_base + k] -= lr * dv[k] / bs;
                }
            }
            for a in 0..d {
                for b in 0..d {
                    w_s[a * d + b] -= lr * d_w[a * d + b] / bs;
                }
            }
            b_s -= lr * (db_accum / bs);
        }

        if loss_rows > 0 {
            loss_perm.shuffle(&mut rng);
            let capped = loss_rows.min(2000);
            let mut d_wl = vec![0.0_f64; d * d];
            let mut u_wl = vec![0.0_f64; d];
            for batch_start in (0..capped).step_by(batch_size.max(1)) {
                for value in &mut d_wl {
                    *value = 0.0;
                }
                let end = (batch_start + batch_size.max(1)).min(capped);
                let bs = (end - batch_start).max(1) as f64;
                let mut db_accum = 0.0_f64;
                for &perm_idx in &loss_perm[batch_start..end] {
                    let i = loss_idx[perm_idx * 2] as usize;
                    let j = loss_idx[perm_idx * 2 + 1] as usize;
                    let target = loss_labels[perm_idx];
                    let weight = loss_weights[perm_idx];
                    let u_base = row_offset(i, d);
                    let v_base = row_offset(j, d);
                    dot_row_matrix_flat(&u[u_base..u_base + d], &w_l, wl_rows, wl_cols, &mut u_wl);
                    let mut pred = b_l;
                    for k in 0..d {
                        pred += u_wl[k] * v[v_base + k];
                    }
                    let diff = pred - target;
                    let abs_diff = diff.abs();
                    let grad = if abs_diff <= HUBER_DELTA {
                        diff
                    } else {
                        HUBER_DELTA * diff.signum()
                    } * weight
                        * 0.5;
                    total_loss += 0.5 * grad.abs();
                    db_accum += grad;
                    for a in 0..d {
                        let u_val = u[u_base + a];
                        for b in 0..d {
                            d_wl[a * d + b] += u_val * grad * v[v_base + b];
                        }
                    }
                    for a in 0..d {
                        let mut acc = 0.0_f64;
                        for b in 0..d {
                            acc += v[v_base + b] * w_l[a * d + b];
                        }
                        du[a] = grad * acc;
                        dv[a] = grad * u_wl[a];
                    }
                    for k in 0..d {
                        u[u_base + k] -= lr * du[k] / bs;
                        v[v_base + k] -= lr * dv[k] / bs;
                    }
                }
                for a in 0..d {
                    for b in 0..d {
                        w_l[a * d + b] -= lr * d_wl[a * d + b] / bs;
                    }
                }
                b_l -= lr * (db_accum / bs);
            }
        }

        if total_loss < best_loss {
            best_loss = total_loss;
        }
    }

    Ok(NativeInteractionTrainResult {
        u,
        u_rows,
        u_cols,
        v,
        v_rows,
        v_cols,
        w_s,
        ws_rows,
        ws_cols,
        w_l,
        wl_rows,
        wl_cols,
        b_s,
        b_l,
        best_loss,
    })
}

fn normalize_rows_flat(matrix: &mut [f64], rows: usize, cols: usize) {
    for row_idx in 0..rows {
        let base = row_idx * cols;
        let mut norm = 0.0_f64;
        for col in 0..cols {
            let value = matrix[base + col];
            norm += value * value;
        }
        let norm = norm.sqrt().max(1e-8);
        for col in 0..cols {
            matrix[base + col] /= norm;
        }
    }
}

pub fn train_op_embeddings_epoch_native(
    embeddings_init: &[f64],
    emb_rows: usize,
    emb_cols: usize,
    positive_pairs: &[i32],
    positive_rows: usize,
    negative_pairs: &[i32],
    negative_rows: usize,
    pair_idx: &[i32],
    pair_rows: usize,
    pair_labels: &[f64],
    lr: f64,
    batch_size: usize,
    margin: f64,
    pair_weight: f64,
    seed: u64,
) -> Result<NativeEmbeddingEpochResult, AriaError> {
    if positive_pairs.len() != positive_rows * 2
        || negative_pairs.len() != negative_rows * 2
        || pair_idx.len() != pair_rows * 2
        || pair_labels.len() != pair_rows
    {
        return Err(AriaError::ExecutionFailed(
            "embedding input shape mismatch".into(),
        ));
    }
    let mut embeddings = checked_flat_copy(embeddings_init, emb_rows, emb_cols)?;
    let mut rng = StdRng::seed_from_u64(seed);
    let mut total_loss = 0.0_f64;
    let mut n_samples = 0usize;
    let mut positive_perm: Vec<usize> = (0..positive_rows).collect();
    positive_perm.shuffle(&mut rng);
    let d = emb_cols;
    let capped_pos = positive_rows.min(2000);

    for batch_start in (0..capped_pos).step_by(batch_size.max(1)) {
        let end = (batch_start + batch_size.max(1)).min(capped_pos);
        for &perm_idx in &positive_perm[batch_start..end] {
            if negative_rows == 0 {
                continue;
            }
            let anchor_idx = positive_pairs[perm_idx * 2] as usize;
            let pos_idx = positive_pairs[perm_idx * 2 + 1] as usize;
            let neg_pair_idx = rng.gen_range(0..negative_rows);
            let neg_left = negative_pairs[neg_pair_idx * 2] as usize;
            let neg_right = negative_pairs[neg_pair_idx * 2 + 1] as usize;
            let neg_idx = if neg_left == anchor_idx {
                neg_right
            } else {
                neg_left
            };
            let anchor_base = row_offset(anchor_idx, d);
            let pos_base = row_offset(pos_idx, d);
            let neg_base = row_offset(neg_idx, d);

            let mut d_pos = 0.0_f64;
            let mut d_neg = 0.0_f64;
            for k in 0..d {
                let ap = embeddings[anchor_base + k] - embeddings[pos_base + k];
                let an = embeddings[anchor_base + k] - embeddings[neg_base + k];
                d_pos += ap * ap;
                d_neg += an * an;
            }
            let margin_loss = (d_pos - d_neg + margin).max(0.0);
            if margin_loss > 0.0 {
                for k in 0..d {
                    let a = embeddings[anchor_base + k];
                    let p = embeddings[pos_base + k];
                    let n = embeddings[neg_base + k];
                    let grad_a = 2.0 * ((a - p) - (a - n));
                    let grad_p = 2.0 * (p - a);
                    let grad_n = 2.0 * (a - n);
                    embeddings[anchor_base + k] -= lr * grad_a;
                    embeddings[pos_base + k] -= lr * grad_p;
                    embeddings[neg_base + k] += lr * grad_n;
                }
                total_loss += margin_loss;
                n_samples += 1;
            }
        }
    }

    let capped_pair = pair_rows.min(1000);
    for pair_row in 0..capped_pair {
        let idx_a = pair_idx[pair_row * 2] as usize;
        let idx_b = pair_idx[pair_row * 2 + 1] as usize;
        let base_a = row_offset(idx_a, d);
        let base_b = row_offset(idx_b, d);
        let target = pair_labels[pair_row];
        let mut logit = 0.0_f64;
        for k in 0..d {
            logit += embeddings[base_a + k] * embeddings[base_b + k];
        }
        let pred = 1.0 / (1.0 + (-logit.clamp(-10.0, 10.0)).exp());
        let d_logit = pred - target;
        let eps = 1e-8_f64;
        total_loss += pair_weight
            * -(target * (pred.max(eps)).ln() + (1.0 - target) * ((1.0 - pred).max(eps)).ln());
        for k in 0..d {
            let grad_a = pair_weight * d_logit * embeddings[base_b + k];
            let grad_b = pair_weight * d_logit * embeddings[base_a + k];
            embeddings[base_a + k] -= lr * grad_a;
            embeddings[base_b + k] -= lr * grad_b;
        }
    }

    normalize_rows_flat(&mut embeddings, emb_rows, emb_cols);
    Ok(NativeEmbeddingEpochResult {
        embeddings,
        emb_rows,
        emb_cols,
        total_loss,
        n_samples,
    })
}
