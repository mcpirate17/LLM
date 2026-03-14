use std::collections::{HashMap, HashSet, VecDeque};

use serde::{Deserialize, Serialize};

use crate::error::AriaError;

/// Opaque node identifier (index into the graph's node list).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct NodeId(pub u32);

/// A single computation node in the graph IR.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Node {
    pub id: NodeId,
    pub op_name: String,
    pub input_ids: Vec<NodeId>,
    pub config: serde_json::Value,
    pub is_input: bool,
    pub is_output: bool,
}

/// A directed edge between two nodes, with optional port names.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Edge {
    pub source: NodeId,
    pub target: NodeId,
    pub source_port: Option<String>,
    pub target_port: Option<String>,
}

/// The full graph intermediate representation, deserialized from JSON.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphIR {
    pub schema_version: String,
    pub model_dim: u32,
    pub nodes: Vec<Node>,
    pub edges: Vec<Edge>,
    pub output_node_id: NodeId,
    pub metadata: Option<serde_json::Value>,
}

fn fused_pattern_name(parent_op: &str, child_op: &str) -> Option<&'static str> {
    match (parent_op, child_op) {
        ("matmul", "relu") => Some("matmul_relu"),
        ("matmul", "gelu") => Some("matmul_gelu"),
        _ => None,
    }
}

impl GraphIR {
    /// Parse a `GraphIR` from its JSON representation.
    pub fn from_json(json: &str) -> Result<Self, AriaError> {
        serde_json::from_str(json).map_err(|e| AriaError::InvalidIR(e.to_string()))
    }

    /// Compute a topological ordering of node ids using Kahn's algorithm.
    ///
    /// Returns `Err(AriaError::CyclicGraph)` if the graph contains a cycle.
    pub fn topological_order(&self) -> Result<Vec<NodeId>, AriaError> {
        let node_ids: HashSet<NodeId> = self.nodes.iter().map(|n| n.id).collect();
        let mut in_degree: HashMap<NodeId, usize> = node_ids.iter().map(|&id| (id, 0)).collect();
        let mut adjacency: HashMap<NodeId, Vec<NodeId>> = node_ids.iter().map(|&id| (id, Vec::new())).collect();

        for edge in &self.edges {
            *in_degree.entry(edge.target).or_insert(0) += 1;
            adjacency.entry(edge.source).or_default().push(edge.target);
        }

        let mut queue: VecDeque<NodeId> = in_degree
            .iter()
            .filter(|(_, &deg)| deg == 0)
            .map(|(&id, _)| id)
            .collect();

        // Deterministic ordering: sort the initial queue by id.
        let mut start: Vec<NodeId> = queue.drain(..).collect();
        start.sort_by_key(|n| n.0);
        queue.extend(start);

        let mut order = Vec::with_capacity(self.nodes.len());

        while let Some(node) = queue.pop_front() {
            order.push(node);
            if let Some(successors) = adjacency.get(&node) {
                let mut next_ready = Vec::new();
                for &succ in successors {
                    if let Some(deg) = in_degree.get_mut(&succ) {
                        *deg -= 1;
                        if *deg == 0 {
                            next_ready.push(succ);
                        }
                    }
                }
                // Sort for deterministic output.
                next_ready.sort_by_key(|n| n.0);
                queue.extend(next_ready);
            }
        }

        if order.len() != self.nodes.len() {
            return Err(AriaError::CyclicGraph);
        }

        Ok(order)
    }

    /// Validate structural invariants:
    /// - All node ids are unique.
    /// - Every edge references existing nodes.
    /// - The output_node_id exists in the node set.
    pub fn validate(&self) -> Result<(), AriaError> {
        let mut seen = HashSet::with_capacity(self.nodes.len());
        for node in &self.nodes {
            if !seen.insert(node.id) {
                return Err(AriaError::InvalidIR(format!(
                    "duplicate node id: {}",
                    node.id.0
                )));
            }
        }

        for edge in &self.edges {
            if !seen.contains(&edge.source) {
                return Err(AriaError::InvalidIR(format!(
                    "edge references unknown source node: {}",
                    edge.source.0
                )));
            }
            if !seen.contains(&edge.target) {
                return Err(AriaError::InvalidIR(format!(
                    "edge references unknown target node: {}",
                    edge.target.0
                )));
            }
        }

        if !seen.contains(&self.output_node_id) {
            return Err(AriaError::InvalidIR(format!(
                "output_node_id {} does not exist in nodes",
                self.output_node_id.0
            )));
        }

        Ok(())
    }

    /// Fuse simple supported patterns into native fused kernels.
    pub fn fuse_supported_patterns(&self) -> Self {
        let mut consumer_counts: HashMap<NodeId, usize> = HashMap::new();
        for node in &self.nodes {
            for input_id in &node.input_ids {
                *consumer_counts.entry(*input_id).or_insert(0) += 1;
            }
        }

        let node_map: HashMap<NodeId, &Node> = self.nodes.iter().map(|n| (n.id, n)).collect();
        let mut removed = HashSet::new();
        let mut rewrites: HashMap<NodeId, String> = HashMap::new();
        let mut output_node_id = self.output_node_id;

        for node in &self.nodes {
            if node.is_input || removed.contains(&node.id) || node.input_ids.len() != 1 {
                continue;
            }
            let parent_id = node.input_ids[0];
            let Some(parent) = node_map.get(&parent_id) else {
                continue;
            };
            if parent.is_input || removed.contains(&parent_id) {
                continue;
            }
            if consumer_counts.get(&parent_id).copied().unwrap_or(0) != 1 {
                continue;
            }
            if let Some(fused_name) = fused_pattern_name(parent.op_name.as_str(), node.op_name.as_str()) {
                rewrites.insert(parent_id, fused_name.to_string());
                removed.insert(node.id);
                if output_node_id == node.id {
                    output_node_id = parent_id;
                }
            }
        }

        if removed.is_empty() {
            return self.clone();
        }

        let nodes = self
            .nodes
            .iter()
            .filter(|node| !removed.contains(&node.id))
            .map(|node| {
                let mut next = node.clone();
                if let Some(name) = rewrites.get(&node.id) {
                    next.op_name = name.clone();
                }
                next.input_ids = next
                    .input_ids
                    .iter()
                    .copied()
                    .filter(|id| !removed.contains(id))
                    .collect();
                next.is_output = next.id == output_node_id;
                next
            })
            .collect();

        let edges = self
            .edges
            .iter()
            .filter(|edge| !removed.contains(&edge.source) && !removed.contains(&edge.target))
            .cloned()
            .collect();

        Self {
            schema_version: self.schema_version.clone(),
            model_dim: self.model_dim,
            nodes,
            edges,
            output_node_id,
            metadata: self.metadata.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_graph_json() -> String {
        r#"{
            "schema_version": "0.1",
            "model_dim": 64,
            "nodes": [
                {"id": 0, "op_name": "input", "input_ids": [], "config": {}, "is_input": true, "is_output": false},
                {"id": 1, "op_name": "linear", "input_ids": [0], "config": {"out_features": 32}, "is_input": false, "is_output": false},
                {"id": 2, "op_name": "relu", "input_ids": [1], "config": {}, "is_input": false, "is_output": true}
            ],
            "edges": [
                {"source": 0, "target": 1, "source_port": null, "target_port": null},
                {"source": 1, "target": 2, "source_port": null, "target_port": null}
            ],
            "output_node_id": 2,
            "metadata": null
        }"#
        .to_string()
    }

    #[test]
    fn test_from_json() {
        let g = GraphIR::from_json(&sample_graph_json()).unwrap();
        assert_eq!(g.nodes.len(), 3);
        assert_eq!(g.edges.len(), 2);
    }

    #[test]
    fn test_topological_order() {
        let g = GraphIR::from_json(&sample_graph_json()).unwrap();
        let order = g.topological_order().unwrap();
        assert_eq!(order, vec![NodeId(0), NodeId(1), NodeId(2)]);
    }

    #[test]
    fn test_validate_ok() {
        let g = GraphIR::from_json(&sample_graph_json()).unwrap();
        g.validate().unwrap();
    }

    #[test]
    fn test_validate_duplicate_id() {
        let json = r#"{
            "schema_version": "0.1", "model_dim": 64,
            "nodes": [
                {"id": 0, "op_name": "a", "input_ids": [], "config": {}, "is_input": true, "is_output": false},
                {"id": 0, "op_name": "b", "input_ids": [], "config": {}, "is_input": false, "is_output": true}
            ],
            "edges": [], "output_node_id": 0, "metadata": null
        }"#;
        let g = GraphIR::from_json(json).unwrap();
        assert!(g.validate().is_err());
    }

    #[test]
    fn test_fuse_supported_patterns_matmul_relu() {
        let json = r#"{
            "schema_version": "0.1",
            "model_dim": 64,
            "nodes": [
                {"id": 0, "op_name": "input", "input_ids": [], "config": {}, "is_input": true, "is_output": false},
                {"id": 1, "op_name": "input", "input_ids": [], "config": {}, "is_input": true, "is_output": false},
                {"id": 2, "op_name": "matmul", "input_ids": [0, 1], "config": {"m": 2, "k": 3, "n": 4}, "is_input": false, "is_output": false},
                {"id": 3, "op_name": "relu", "input_ids": [2], "config": {}, "is_input": false, "is_output": true}
            ],
            "edges": [
                {"source": 0, "target": 2, "source_port": null, "target_port": null},
                {"source": 1, "target": 2, "source_port": null, "target_port": null},
                {"source": 2, "target": 3, "source_port": null, "target_port": null}
            ],
            "output_node_id": 3,
            "metadata": null
        }"#;
        let g = GraphIR::from_json(json).unwrap();
        let fused = g.fuse_supported_patterns();
        assert_eq!(fused.nodes.len(), 3);
        assert_eq!(fused.output_node_id, NodeId(2));
        let fused_node = fused.nodes.iter().find(|n| n.id == NodeId(2)).unwrap();
        assert_eq!(fused_node.op_name, "matmul_relu");
        assert!(fused.nodes.iter().all(|n| n.id != NodeId(3)));
    }
}
