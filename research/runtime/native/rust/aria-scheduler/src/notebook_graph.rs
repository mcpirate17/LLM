use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};

use serde::Deserialize;
use serde_json::Value;
use xxhash_rust::xxh64::xxh64;

use crate::error::AriaError;

#[derive(Debug, Deserialize)]
pub struct NotebookGraph {
    pub model_dim: u32,
    pub nodes: HashMap<String, NotebookNode>,
    #[serde(default)]
    pub metadata: Value,
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
        serde_json::from_str(json).map_err(|e| AriaError::InvalidIR(e.to_string()))
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
