use std::cmp::Ordering;
use std::collections::{BTreeSet, BinaryHeap, HashMap, HashSet, VecDeque};

use serde::Deserialize;
use serde::Serialize;
use serde_json::Value;
use xxhash_rust::xxh64::xxh64;

use crate::error::AriaError;

#[derive(Debug, Deserialize)]
pub struct NotebookGraph {
    #[serde(default)]
    pub model_dim: u32,
    pub nodes: HashMap<String, NotebookNode>,
    #[serde(default)]
    pub metadata: Value,
    #[serde(default)]
    pub output_node_id: Option<i64>,
}

#[derive(Debug, Deserialize)]
pub struct NotebookNode {
    pub id: i64,
    pub op_name: String,
    #[serde(default)]
    pub input_ids: Vec<i64>,
    #[serde(default)]
    pub config: Value,
}

pub struct GraphFeaturePayload {
    pub template_name: String,
    pub op_names: Vec<String>,
    pub pair_signatures: Vec<String>,
    pub templates_json: String,
    pub motifs_json: String,
    pub slot_usage_json: String,
}

#[derive(Serialize)]
pub struct GraphProvenancePayload {
    pub op_names: Vec<String>,
    pub source_op: Option<String>,
}

#[derive(Serialize)]
pub struct GraphStructurePayload {
    pub op_names: Vec<String>,
    pub n_nodes: f64,
    pub n_edges: f64,
    pub n_ops: f64,
    pub depth: f64,
    pub width: f64,
    pub n_unique_ops: f64,
    pub n_skip_connections: f64,
    pub edge_density: f64,
    pub n_templates_used: f64,
    pub model_dim: f64,
}

#[derive(Clone, Eq, PartialEq)]
struct ReadyNode {
    op_name: String,
    input_keys: Vec<usize>,
    config_str: String,
    orig_id: i64,
}

impl Ord for ReadyNode {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .op_name
            .cmp(&self.op_name)
            .then_with(|| other.input_keys.cmp(&self.input_keys))
            .then_with(|| other.config_str.cmp(&self.config_str))
            .then_with(|| other.orig_id.cmp(&self.orig_id))
    }
}

impl PartialOrd for ReadyNode {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl NotebookGraph {
    pub fn from_json(json: &str) -> Result<Self, AriaError> {
        let value: Value =
            serde_json::from_str(json).map_err(|e| AriaError::InvalidIR(e.to_string()))?;
        notebook_graph_from_value(value)
    }

    pub fn canonical_topological_order(&self) -> Vec<i64> {
        let mut in_degree: HashMap<i64, usize> = HashMap::with_capacity(self.nodes.len());
        let mut children: HashMap<i64, Vec<i64>> = HashMap::with_capacity(self.nodes.len());
        let mut static_keys: HashMap<i64, (String, String, i64)> =
            HashMap::with_capacity(self.nodes.len());

        for node in self.nodes.values() {
            in_degree.insert(node.id, node.input_ids.len());
            children.entry(node.id).or_default();
            static_keys.insert(
                node.id,
                (node.op_name.clone(), config_string(&node.config), node.id),
            );
        }

        for node in self.nodes.values() {
            for &input_id in &node.input_ids {
                children.entry(input_id).or_default().push(node.id);
            }
        }

        let mut ready = BinaryHeap::new();
        for (&node_id, &deg) in &in_degree {
            if deg == 0 {
                let (op_name, config_str, orig_id) = static_keys
                    .get(&node_id)
                    .cloned()
                    .unwrap_or_else(|| (String::new(), String::new(), node_id));
                ready.push(ReadyNode {
                    op_name,
                    input_keys: Vec::new(),
                    config_str,
                    orig_id,
                });
            }
        }

        let mut order = Vec::with_capacity(self.nodes.len());
        let mut canonical_id_map: HashMap<i64, usize> = HashMap::with_capacity(self.nodes.len());

        while let Some(next) = ready.pop() {
            let node_id = next.orig_id;
            if canonical_id_map.contains_key(&node_id) {
                continue;
            }

            canonical_id_map.insert(node_id, order.len());
            order.push(node_id);

            if let Some(node_children) = children.get(&node_id) {
                for &child_id in node_children {
                    let Some(child_deg) = in_degree.get_mut(&child_id) else {
                        continue;
                    };
                    if *child_deg == 0 {
                        continue;
                    }
                    *child_deg -= 1;
                    if *child_deg != 0 {
                        continue;
                    }

                    let Some(child) = self.node_by_id(child_id) else {
                        continue;
                    };
                    let mut input_keys = Vec::with_capacity(child.input_ids.len());
                    let mut complete = true;
                    for input_id in &child.input_ids {
                        if let Some(rank) = canonical_id_map.get(input_id) {
                            input_keys.push(*rank);
                        } else {
                            complete = false;
                            break;
                        }
                    }
                    if !complete {
                        continue;
                    }
                    let (op_name, config_str, orig_id) = static_keys
                        .get(&child_id)
                        .cloned()
                        .unwrap_or_else(|| (String::new(), String::new(), child_id));
                    ready.push(ReadyNode {
                        op_name,
                        input_keys,
                        config_str,
                        orig_id,
                    });
                }
            }
        }

        if order.len() < self.nodes.len() {
            let mut fallback: Vec<i64> = self.nodes.values().map(|node| node.id).collect();
            fallback.sort_unstable();
            return fallback;
        }

        order
    }

    pub fn fingerprint(&self) -> Result<String, AriaError> {
        let order = self.canonical_topological_order();
        let mut id_to_rank = HashMap::with_capacity(order.len());
        for (idx, node_id) in order.iter().enumerate() {
            id_to_rank.insert(*node_id, idx);
        }

        let mut desc: Vec<String> = Vec::with_capacity(order.len());
        for node_id in order {
            let Some(node) = self.node_by_id(node_id) else {
                return Err(AriaError::InvalidIR(format!(
                    "node {} missing from graph",
                    node_id
                )));
            };
            let mut rank_inputs = Vec::with_capacity(node.input_ids.len());
            for input_id in &node.input_ids {
                let Some(rank) = id_to_rank.get(input_id) else {
                    return Err(AriaError::InvalidIR(format!(
                        "node {} references unknown input {}",
                        node.id, input_id
                    )));
                };
                rank_inputs.push(rank.to_string());
            }
            let config_str = config_string(&node.config);
            desc.push(format!(
                "{}{}({})",
                node.op_name,
                config_str,
                rank_inputs.join(",")
            ));
        }

        let rc_str = routing_compression_string(&self.metadata);
        let key = format!("dim={}{}|{}", self.model_dim, rc_str, desc.join("|"));
        Ok(format!("{:016x}", xxh64(key.as_bytes(), 0)))
    }

    fn node_by_id(&self, node_id: i64) -> Option<&NotebookNode> {
        self.nodes.values().find(|node| node.id == node_id)
    }
}

pub fn extract_graph_ops_json(json: &str) -> Result<Vec<String>, AriaError> {
    let value: Value =
        serde_json::from_str(json).map_err(|e| AriaError::InvalidIR(e.to_string()))?;
    Ok(graph_ops_from_value(&value, false))
}

fn notebook_graph_from_value(value: Value) -> Result<NotebookGraph, AriaError> {
    let Some(graph) = value.as_object() else {
        return Err(AriaError::InvalidIR("notebook graph must be a JSON object".into()));
    };
    let model_dim = graph
        .get("model_dim")
        .and_then(Value::as_u64)
        .unwrap_or(0)
        .min(u32::MAX as u64) as u32;
    let metadata = graph.get("metadata").cloned().unwrap_or(Value::Null);
    let mut nodes: HashMap<String, NotebookNode> = HashMap::new();
    let mut aliases: HashMap<String, i64> = HashMap::new();

    match graph.get("nodes") {
        Some(Value::Object(node_map)) => {
            for (idx, (key, node)) in node_map.iter().enumerate() {
                let id = assigned_node_id(node, Some(key), idx);
                aliases.insert(key.clone(), id);
                aliases.insert(id.to_string(), id);
                if let Some(identifier) = node_identifier(node, Some(key), idx) {
                    aliases.insert(identifier, id);
                }
            }
        }
        Some(Value::Array(node_list)) => {
            for (idx, node) in node_list.iter().enumerate() {
                let id = assigned_node_id(node, None, idx);
                aliases.insert(id.to_string(), id);
                if let Some(identifier) = node_identifier(node, None, idx) {
                    aliases.insert(identifier, id);
                }
            }
        }
        _ => {}
    }

    match graph.get("nodes") {
        Some(Value::Object(node_map)) => {
            for (idx, (key, node)) in node_map.iter().enumerate() {
                if let Some(parsed) = notebook_node_from_value(node, Some(key), idx, &aliases) {
                    nodes.insert(key.clone(), parsed);
                }
            }
        }
        Some(Value::Array(node_list)) => {
            for (idx, node) in node_list.iter().enumerate() {
                if let Some(parsed) = notebook_node_from_value(node, None, idx, &aliases) {
                    nodes.insert(parsed.id.to_string(), parsed);
                }
            }
        }
        _ => {}
    }

    let output_node_id = graph
        .get("output_node_id")
        .and_then(|value| value_to_node_id(value, &aliases));

    Ok(NotebookGraph {
        model_dim,
        nodes,
        metadata,
        output_node_id,
    })
}

fn notebook_node_from_value(
    value: &Value,
    key: Option<&str>,
    index: usize,
    aliases: &HashMap<String, i64>,
) -> Option<NotebookNode> {
    if let Some(raw_op) = value.as_str() {
        return Some(NotebookNode {
            id: assigned_node_id(value, key, index),
            op_name: raw_op.trim().to_string(),
            input_ids: Vec::new(),
            config: Value::Null,
        });
    }
    let obj = value.as_object()?;
    let id = assigned_node_id(value, key, index);
    let op_name = cleaned_op_name(
        obj.get("op_name")
            .or_else(|| obj.get("op"))
            .or_else(|| obj.get("op_type")),
    );
    let input_ids = obj
        .get("input_ids")
        .and_then(Value::as_array)
        .map(|values| {
            values
                .iter()
                .filter_map(|value| value_to_node_id(value, aliases))
                .collect()
        })
        .unwrap_or_default();
    Some(NotebookNode {
        id,
        op_name,
        input_ids,
        config: obj.get("config").cloned().unwrap_or(Value::Null),
    })
}

fn assigned_node_id(value: &Value, key: Option<&str>, index: usize) -> i64 {
    value
        .as_object()
        .and_then(|obj| obj.get("id"))
        .and_then(value_to_i64)
        .or_else(|| key.and_then(|raw| raw.parse::<i64>().ok()))
        .unwrap_or(index as i64)
}

fn node_identifier(value: &Value, key: Option<&str>, index: usize) -> Option<String> {
    if let Some(obj) = value.as_object() {
        if let Some(raw_id) = obj.get("id") {
            let identifier = value_to_lookup_key(raw_id);
            if !identifier.is_empty() {
                return Some(identifier);
            }
        }
    }
    key.map(str::to_string).or_else(|| Some(index.to_string()))
}

fn graph_ops_from_value(value: &Value, unique: bool) -> Vec<String> {
    let Some(graph) = value.as_object() else {
        return Vec::new();
    };
    let mut ops = Vec::new();
    match graph.get("nodes") {
        Some(Value::Object(node_map)) => {
            for node in node_map.values() {
                if let Some(op_name) = op_name_from_node_value(node) {
                    ops.push(op_name);
                }
            }
        }
        Some(Value::Array(node_list)) => {
            for node in node_list {
                if let Some(op_name) = op_name_from_node_value(node) {
                    ops.push(op_name);
                }
            }
        }
        _ => {}
    }
    if unique {
        ops.sort();
        ops.dedup();
    }
    ops
}

fn op_name_from_node_value(value: &Value) -> Option<String> {
    let op_name = if let Some(raw) = value.as_str() {
        raw.trim().to_string()
    } else {
        let obj = value.as_object()?;
        cleaned_op_name(
            obj.get("op_name")
                .or_else(|| obj.get("op"))
                .or_else(|| obj.get("op_type")),
        )
    };
    if op_name.is_empty() || op_name == "input" {
        None
    } else {
        Some(op_name)
    }
}

fn value_to_i64(value: &Value) -> Option<i64> {
    value
        .as_i64()
        .or_else(|| value.as_u64().and_then(|raw| i64::try_from(raw).ok()))
        .or_else(|| value.as_str().and_then(|raw| raw.parse::<i64>().ok()))
}

fn value_to_node_id(value: &Value, aliases: &HashMap<String, i64>) -> Option<i64> {
    value_to_i64(value).or_else(|| aliases.get(&value_to_lookup_key(value)).copied())
}

pub fn extract_graph_feature_payload_json(json: &str) -> Result<GraphFeaturePayload, AriaError> {
    if json.trim().is_empty() {
        return Ok(empty_graph_feature_payload());
    }
    let value: Value =
        serde_json::from_str(json).map_err(|e| AriaError::InvalidIR(e.to_string()))?;
    let Some(graph) = value.as_object() else {
        return Ok(empty_graph_feature_payload());
    };

    let metadata = graph
        .get("metadata")
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default();
    let nodes = graph.get("nodes");

    let mut op_names: BTreeSet<String> = BTreeSet::new();
    let mut pair_signatures: BTreeSet<String> = BTreeSet::new();

    match nodes {
        Some(Value::Object(node_map)) => {
            for node in node_map.values() {
                let Some(node_obj) = node.as_object() else {
                    continue;
                };
                let op_name = cleaned_op_name(
                    node_obj
                        .get("op_name")
                        .or_else(|| node_obj.get("op"))
                        .or_else(|| node_obj.get("op_type")),
                );
                if op_name.is_empty() || op_name == "input" {
                    continue;
                }
                op_names.insert(op_name.clone());
                let Some(input_ids) = node_obj.get("input_ids").and_then(Value::as_array) else {
                    continue;
                };
                for raw_parent in input_ids {
                    let parent_key = value_to_lookup_key(raw_parent);
                    let Some(parent_obj) = node_map.get(&parent_key).and_then(Value::as_object)
                    else {
                        continue;
                    };
                    let parent_op = cleaned_op_name(
                        parent_obj
                            .get("op_name")
                            .or_else(|| parent_obj.get("op"))
                            .or_else(|| parent_obj.get("op_type")),
                    );
                    if !parent_op.is_empty() && parent_op != "input" {
                        pair_signatures.insert(format!("{}->{}", parent_op, op_name));
                    }
                }
            }
        }
        Some(Value::Array(node_list)) => {
            let mut node_map: HashMap<String, &serde_json::Map<String, Value>> =
                HashMap::with_capacity(node_list.len());
            for (idx, node) in node_list.iter().enumerate() {
                let Some(node_obj) = node.as_object() else {
                    continue;
                };
                let id_key = node_obj
                    .get("id")
                    .map(value_to_lookup_key)
                    .unwrap_or_else(|| idx.to_string());
                node_map.insert(id_key, node_obj);
            }
            for node_obj in node_map.values() {
                let op_name = cleaned_op_name(
                    node_obj
                        .get("op_name")
                        .or_else(|| node_obj.get("op"))
                        .or_else(|| node_obj.get("op_type")),
                );
                if op_name.is_empty() || op_name == "input" {
                    continue;
                }
                op_names.insert(op_name.clone());
                let Some(input_ids) = node_obj.get("input_ids").and_then(Value::as_array) else {
                    continue;
                };
                for raw_parent in input_ids {
                    let parent_key = value_to_lookup_key(raw_parent);
                    let Some(parent_obj) = node_map.get(&parent_key) else {
                        continue;
                    };
                    let parent_op = cleaned_op_name(
                        parent_obj
                            .get("op_name")
                            .or_else(|| parent_obj.get("op"))
                            .or_else(|| parent_obj.get("op_type")),
                    );
                    if !parent_op.is_empty() && parent_op != "input" {
                        pair_signatures.insert(format!("{}->{}", parent_op, op_name));
                    }
                }
            }
        }
        _ => {}
    }

    let template_name = metadata
        .get("template")
        .or_else(|| metadata.get("template_name"))
        .map(value_to_string)
        .unwrap_or_default()
        .trim()
        .to_string();

    Ok(GraphFeaturePayload {
        template_name,
        op_names: op_names.into_iter().collect(),
        pair_signatures: pair_signatures.into_iter().collect(),
        templates_json: clean_string_list_json(metadata.get("templates_used")),
        motifs_json: clean_string_list_json(metadata.get("motifs_used")),
        slot_usage_json: clean_dict_list_json(metadata.get("template_slot_usage")),
    })
}

pub fn analyze_graph_provenance_json(
    json: &str,
    failure_op: Option<&str>,
    generic_sink_ops: &[String],
) -> Result<String, AriaError> {
    let payload = analyze_graph_provenance(json, failure_op, generic_sink_ops)?;
    serde_json::to_string(&payload).map_err(|e| AriaError::InvalidIR(e.to_string()))
}

pub fn analyze_graph_provenance(
    json: &str,
    failure_op: Option<&str>,
    generic_sink_ops: &[String],
) -> Result<GraphProvenancePayload, AriaError> {
    let graph = NotebookGraph::from_json(json)?;
    let mut sorted_nodes: Vec<&NotebookNode> = graph.nodes.values().collect();
    sorted_nodes.sort_unstable_by_key(|node| node.id);

    let generic_sink_set: HashSet<&str> = generic_sink_ops.iter().map(String::as_str).collect();
    let mut op_names = Vec::new();
    let mut id_to_idx = HashMap::with_capacity(sorted_nodes.len());
    for (idx, node) in sorted_nodes.iter().enumerate() {
        id_to_idx.insert(node.id, idx);
        let op_name = node.op_name.trim();
        if !op_name.is_empty() && op_name != "input" {
            op_names.push(op_name.to_string());
        }
    }

    let source_op = failure_op
        .map(str::trim)
        .filter(|op| !op.is_empty())
        .and_then(|failure_name| {
            let start_indices: Vec<usize> = sorted_nodes
                .iter()
                .enumerate()
                .filter_map(|(idx, node)| (node.op_name == failure_name).then_some(idx))
                .collect();
            if start_indices.is_empty() {
                return None;
            }

            let mut queue = VecDeque::new();
            let mut seen = HashSet::new();
            for idx in start_indices {
                for input_id in &sorted_nodes[idx].input_ids {
                    if let Some(&parent_idx) = id_to_idx.get(input_id) {
                        queue.push_back(parent_idx);
                    }
                }
            }

            while let Some(idx) = queue.pop_front() {
                if !seen.insert(idx) {
                    continue;
                }
                let op_name = sorted_nodes[idx].op_name.trim();
                if !op_name.is_empty() && op_name != "input" && !generic_sink_set.contains(op_name)
                {
                    return Some(op_name.to_string());
                }
                for input_id in &sorted_nodes[idx].input_ids {
                    if let Some(&parent_idx) = id_to_idx.get(input_id) {
                        queue.push_back(parent_idx);
                    }
                }
            }
            None
        });

    Ok(GraphProvenancePayload {
        op_names,
        source_op,
    })
}

pub fn extract_graph_structure_features_json(json: &str) -> Result<String, AriaError> {
    let graph = NotebookGraph::from_json(json)?;
    let mut sorted_nodes: Vec<&NotebookNode> = graph.nodes.values().collect();
    sorted_nodes.sort_unstable_by_key(|node| node.id);
    let n = sorted_nodes.len();

    if n == 0 {
        return serde_json::to_string(&GraphStructurePayload {
            op_names: Vec::new(),
            n_nodes: 0.0,
            n_edges: 0.0,
            n_ops: 0.0,
            depth: 0.0,
            width: 0.0,
            n_unique_ops: 0.0,
            n_skip_connections: 0.0,
            edge_density: 0.0,
            n_templates_used: 0.0,
            model_dim: 0.0,
        })
        .map_err(|e| AriaError::InvalidIR(e.to_string()));
    }

    let mut id_to_idx = HashMap::with_capacity(n);
    for (idx, node) in sorted_nodes.iter().enumerate() {
        id_to_idx.insert(node.id, idx);
    }

    let mut children: Vec<Vec<usize>> = vec![Vec::new(); n];
    let mut in_degree = vec![0usize; n];
    for (idx, node) in sorted_nodes.iter().enumerate() {
        for input_id in &node.input_ids {
            if let Some(&parent_idx) = id_to_idx.get(input_id) {
                children[parent_idx].push(idx);
                in_degree[idx] += 1;
            }
        }
    }

    let mut depth_map: Vec<Option<usize>> = vec![None; n];
    let mut bfs_queue = VecDeque::new();
    for (idx, deg) in in_degree.iter().enumerate() {
        if *deg == 0 {
            depth_map[idx] = Some(0);
            bfs_queue.push_back(idx);
        }
    }
    while let Some(idx) = bfs_queue.pop_front() {
        let current_depth = depth_map[idx].unwrap_or(0);
        for &child in &children[idx] {
            if depth_map[child].is_none() {
                depth_map[child] = Some(current_depth + 1);
                bfs_queue.push_back(child);
            }
        }
    }

    let mut topo_in_degree = in_degree.clone();
    let mut topo_queue = VecDeque::new();
    let mut longest = vec![0usize; n];
    for (idx, deg) in topo_in_degree.iter().enumerate() {
        if *deg == 0 {
            topo_queue.push_back(idx);
        }
    }
    while let Some(idx) = topo_queue.pop_front() {
        let next_depth = longest[idx] + 1;
        for &child in &children[idx] {
            if next_depth > longest[child] {
                longest[child] = next_depth;
            }
            topo_in_degree[child] -= 1;
            if topo_in_degree[child] == 0 {
                topo_queue.push_back(child);
            }
        }
    }

    let mut width_counts: HashMap<usize, usize> = HashMap::new();
    for depth in depth_map.iter().flatten() {
        *width_counts.entry(*depth).or_insert(0) += 1;
    }

    let mut op_names = Vec::new();
    let mut unique_ops = BTreeSet::new();
    let mut n_edges = 0usize;
    let mut n_skip_connections = 0usize;
    for (idx, node) in sorted_nodes.iter().enumerate() {
        n_edges += children[idx].len();
        let op_name = canonicalize_op_name(node.op_name.trim());
        if !op_name.is_empty() && op_name != "input" {
            op_names.push(op_name.to_string());
            unique_ops.insert(op_name.to_string());
        }
        if op_name == "add" && node.input_ids.len() >= 2 {
            let mut depths = HashSet::new();
            for input_id in &node.input_ids {
                if let Some(&parent_idx) = id_to_idx.get(input_id) {
                    depths.insert(depth_map[parent_idx].unwrap_or(0));
                }
            }
            if depths.len() > 1 {
                n_skip_connections += 1;
            }
        }
    }

    let templates_used = graph
        .metadata
        .get("templates_used")
        .and_then(Value::as_array)
        .map(|items| items.len())
        .unwrap_or(0);
    let n_ops = op_names.len();

    serde_json::to_string(&GraphStructurePayload {
        op_names,
        n_nodes: n as f64,
        n_edges: n_edges as f64,
        n_ops: n_ops as f64,
        depth: longest.into_iter().max().unwrap_or(0) as f64,
        width: width_counts.values().copied().max().unwrap_or(1) as f64,
        n_unique_ops: unique_ops.len() as f64,
        n_skip_connections: n_skip_connections as f64,
        edge_density: n_edges as f64 / n.max(1) as f64,
        n_templates_used: templates_used as f64,
        model_dim: graph.model_dim as f64,
    })
    .map_err(|e| AriaError::InvalidIR(e.to_string()))
}

fn empty_graph_feature_payload() -> GraphFeaturePayload {
    GraphFeaturePayload {
        template_name: String::new(),
        op_names: Vec::new(),
        pair_signatures: Vec::new(),
        templates_json: "[]".to_string(),
        motifs_json: "[]".to_string(),
        slot_usage_json: "[]".to_string(),
    }
}

fn cleaned_op_name(value: Option<&Value>) -> String {
    canonicalize_op_name(value.map(value_to_string).unwrap_or_default().trim()).to_string()
}

fn value_to_lookup_key(value: &Value) -> String {
    match value {
        Value::String(raw) => raw.clone(),
        Value::Number(raw) => raw.to_string(),
        Value::Bool(raw) => raw.to_string(),
        Value::Null => String::new(),
        _ => value_to_string(value),
    }
}

fn value_to_string(value: &Value) -> String {
    match value {
        Value::String(raw) => raw.clone(),
        Value::Number(raw) => raw.to_string(),
        Value::Bool(raw) => raw.to_string(),
        Value::Null => String::new(),
        _ => value.to_string(),
    }
}

fn clean_string_list_json(value: Option<&Value>) -> String {
    let Some(items) = value.and_then(Value::as_array) else {
        return "[]".to_string();
    };
    let cleaned: Vec<String> = items
        .iter()
        .filter(|item| !item.is_null())
        .map(value_to_string)
        .collect();
    if cleaned.is_empty() {
        "[]".to_string()
    } else {
        serde_json::to_string(&cleaned).unwrap_or_else(|_| "[]".to_string())
    }
}

fn clean_dict_list_json(value: Option<&Value>) -> String {
    let Some(items) = value.and_then(Value::as_array) else {
        return "[]".to_string();
    };
    let cleaned: Vec<Value> = items
        .iter()
        .filter(|item| item.is_object())
        .cloned()
        .collect();
    if cleaned.is_empty() {
        "[]".to_string()
    } else {
        serde_json::to_string(&cleaned).unwrap_or_else(|_| "[]".to_string())
    }
}

fn routing_compression_string(metadata: &Value) -> String {
    let Some(rc) = metadata.get("routing_compression") else {
        return String::new();
    };
    let routing_kind = rc
        .get("routing")
        .and_then(|v| v.get("kind"))
        .and_then(Value::as_str)
        .unwrap_or("unknown");
    let compression_kind = rc
        .get("compression")
        .and_then(|v| v.get("kind"))
        .and_then(Value::as_str)
        .unwrap_or("unknown");
    format!("|rc={}:{}", routing_kind, compression_kind)
}

fn canonicalize_op_name(op_name: &str) -> &str {
    match op_name {
        "route_topk" => "feature_sparsity",
        "route_lanes" => "gated_lane_blend",
        "route_recursion" => "depth_gated_transform",
        "routing_conditioned_compression" => "signal_conditioned_compression",
        "relu_gate_routing" => "relu_gated_moe",
        "adaptive_lane_mixer" => "difficulty_blend_3way",
        "adaptive_recursion" => "depth_weighted_proj",
        "cascade" => "learned_token_gate",
        "compression_mixture_experts" => "dual_compression_blend",
        "early_exit" => "confidence_token_gate",
        "entropy_score" => "token_entropy",
        "mixed_recursion_gate" => "score_depth_blend",
        "mod_topk" => "depth_token_mask",
        "progressive_compression_gate" => "adaptive_rank_gate",
        "speculative" => "cheap_verify_blend",
        "token_merge" => "adjacent_token_merge",
        "token_type_classifier" => "token_class_proj",
        "n_way_sparse_router" => "sparse_bottleneck_moe",
        _ => op_name,
    }
}

fn config_string(config: &Value) -> String {
    let Some(config_obj) = config.as_object() else {
        return String::new();
    };
    if config_obj.is_empty() {
        return String::new();
    }
    let mut items: Vec<String> = config_obj
        .iter()
        .map(|(k, v)| format!("{}={}", k, python_like_scalar(v)))
        .collect();
    items.sort();
    format!("[{}]", items.join(","))
}

fn python_like_scalar(value: &Value) -> String {
    match value {
        Value::Null => "None".to_string(),
        Value::Bool(flag) => {
            if *flag {
                "True".to_string()
            } else {
                "False".to_string()
            }
        }
        Value::Number(number) => number.to_string(),
        Value::String(text) => text.clone(),
        Value::Array(values) => {
            let inner: Vec<String> = values.iter().map(python_like_scalar).collect();
            format!("[{}]", inner.join(", "))
        }
        Value::Object(map) => {
            let mut items: Vec<String> = map
                .iter()
                .map(|(k, v)| format!("'{}': {}", k, python_like_scalar(v)))
                .collect();
            items.sort();
            format!("{{{}}}", items.join(", "))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::NotebookGraph;

    #[test]
    fn fingerprint_ignores_metadata_only_changes() {
        let base = r#"{
            "model_dim":256,
            "nodes":{
                "0":{"id":0,"op_name":"input","input_ids":[],"config":{}},
                "1":{"id":1,"op_name":"layernorm","input_ids":[0],"config":{}},
                "2":{"id":2,"op_name":"add","input_ids":[0,1],"config":{}}
            },
            "metadata":{"templates_used":["a"]}
        }"#;
        let changed = r#"{
            "model_dim":256,
            "nodes":{
                "0":{"id":0,"op_name":"input","input_ids":[],"config":{}},
                "1":{"id":1,"op_name":"layernorm","input_ids":[0],"config":{}},
                "2":{"id":2,"op_name":"add","input_ids":[0,1],"config":{}}
            },
            "metadata":{"templates_used":["b"],"lineage":{"parent":"x"}}
        }"#;
        let base_fp = NotebookGraph::from_json(base)
            .unwrap()
            .fingerprint()
            .unwrap();
        let changed_fp = NotebookGraph::from_json(changed)
            .unwrap()
            .fingerprint()
            .unwrap();
        assert_eq!(base_fp, changed_fp);
    }

    #[test]
    fn fingerprint_uses_config_and_routing_compression() {
        let a = r#"{
            "model_dim":256,
            "nodes":{
                "0":{"id":0,"op_name":"input","input_ids":[],"config":{}},
                "1":{"id":1,"op_name":"topk_gate","input_ids":[0],"config":{"k":2}}
            },
            "metadata":{"routing_compression":{"routing":{"kind":"topk"},"compression":{"kind":"none"}}}
        }"#;
        let b = r#"{
            "model_dim":256,
            "nodes":{
                "0":{"id":0,"op_name":"input","input_ids":[],"config":{}},
                "1":{"id":1,"op_name":"topk_gate","input_ids":[0],"config":{"k":4}}
            },
            "metadata":{"routing_compression":{"routing":{"kind":"topk"},"compression":{"kind":"none"}}}
        }"#;
        let c = r#"{
            "model_dim":256,
            "nodes":{
                "0":{"id":0,"op_name":"input","input_ids":[],"config":{}},
                "1":{"id":1,"op_name":"topk_gate","input_ids":[0],"config":{"k":2}}
            },
            "metadata":{"routing_compression":{"routing":{"kind":"soft"},"compression":{"kind":"none"}}}
        }"#;
        let fp_a = NotebookGraph::from_json(a).unwrap().fingerprint().unwrap();
        let fp_b = NotebookGraph::from_json(b).unwrap().fingerprint().unwrap();
        let fp_c = NotebookGraph::from_json(c).unwrap().fingerprint().unwrap();
        assert_ne!(fp_a, fp_b);
        assert_ne!(fp_a, fp_c);
    }
}
