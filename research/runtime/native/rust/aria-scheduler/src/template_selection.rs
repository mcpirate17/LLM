use std::collections::HashMap;

#[derive(Clone)]
pub struct TemplateSelector {
    names: Vec<String>,
    default_weights: Vec<f64>,
    name_to_index: HashMap<String, usize>,
}

impl TemplateSelector {
    pub fn new(names: Vec<String>, default_weights: Vec<f64>) -> Result<Self, String> {
        if names.len() != default_weights.len() {
            return Err("names/default_weights length mismatch".to_string());
        }
        let mut name_to_index = HashMap::with_capacity(names.len());
        for (idx, name) in names.iter().enumerate() {
            if name_to_index.insert(name.clone(), idx).is_some() {
                return Err(format!("duplicate template name '{}'", name));
            }
        }
        Ok(Self {
            names,
            default_weights,
            name_to_index,
        })
    }

    pub fn len(&self) -> usize {
        self.names.len()
    }

    pub fn select_index(
        &self,
        override_weights: Option<&HashMap<String, f64>>,
        allowed_names: Option<&Vec<String>>,
        exploration_budget: f64,
        exploration_draw: f64,
        selection_draw: f64,
    ) -> Result<(usize, bool), String> {
        if self.names.is_empty() {
            return Err("template selector has no names".to_string());
        }
        let allowed_indices = self.allowed_indices(allowed_names)?;

        let explored = exploration_budget > 0.0 && exploration_draw < exploration_budget;

        if explored {
            let pool: Vec<usize> = self.positive_pool(override_weights, allowed_indices.as_ref());
            let picked = uniform_pick(&pool, selection_draw);
            return Ok((picked, true));
        }

        let mut total = 0.0_f64;
        let mut effective_weights = Vec::with_capacity(self.names.len());
        for (idx, name) in self.names.iter().enumerate() {
            let weight = if allowed_indices
                .as_ref()
                .map(|allowed| !allowed.contains(&idx))
                .unwrap_or(false)
            {
                0.0
            } else {
                self.effective_weight(idx, name, override_weights)
            };
            effective_weights.push(weight);
            total += weight;
        }

        if total <= 0.0 || !total.is_finite() {
            let fallback_pool = allowed_indices.unwrap_or_else(|| (0..self.names.len()).collect());
            return Ok((uniform_pick(&fallback_pool, selection_draw), false));
        }

        let mut threshold = selection_draw.clamp(0.0, 0.999_999_999_999) * total;
        for (idx, weight) in effective_weights.iter().enumerate() {
            if *weight <= 0.0 {
                continue;
            }
            if threshold < *weight {
                return Ok((idx, false));
            }
            threshold -= *weight;
        }
        Ok((self.names.len() - 1, false))
    }

    fn effective_weight(
        &self,
        idx: usize,
        name: &str,
        override_weights: Option<&HashMap<String, f64>>,
    ) -> f64 {
        let mut weight = override_weights
            .and_then(|weights| weights.get(name).copied())
            .unwrap_or(self.default_weights[idx]);
        if !weight.is_finite() || weight <= 0.0 {
            weight = 0.0;
        }
        weight
    }

    pub fn name_at(&self, idx: usize) -> Option<&str> {
        self.names.get(idx).map(String::as_str)
    }

    pub fn index_of(&self, name: &str) -> Option<usize> {
        self.name_to_index.get(name).copied()
    }

    fn allowed_indices(&self, allowed_names: Option<&Vec<String>>) -> Result<Option<Vec<usize>>, String> {
        let Some(names) = allowed_names else {
            return Ok(None);
        };
        let mut indices = Vec::with_capacity(names.len());
        for name in names {
            let idx = self
                .index_of(name)
                .ok_or_else(|| format!("unknown template name '{}'", name))?;
            indices.push(idx);
        }
        indices.sort_unstable();
        indices.dedup();
        Ok(Some(indices))
    }

    fn positive_pool(
        &self,
        override_weights: Option<&HashMap<String, f64>>,
        allowed_indices: Option<&Vec<usize>>,
    ) -> Vec<usize> {
        let source_indices: Vec<usize> = allowed_indices
            .cloned()
            .unwrap_or_else(|| (0..self.names.len()).collect());
        let mut indices = Vec::new();
        for idx in &source_indices {
            let name = &self.names[*idx];
            if self.effective_weight(*idx, name, override_weights) > 0.0 {
                indices.push(*idx);
            }
        }
        if indices.is_empty() {
            source_indices
        } else {
            indices
        }
    }
}

fn uniform_pick(indices: &[usize], draw: f64) -> usize {
    if indices.is_empty() {
        return 0;
    }
    let offset = uniform_index(indices.len(), draw);
    indices[offset]
}

fn uniform_index(len: usize, draw: f64) -> usize {
    if len <= 1 {
        return 0;
    }
    let scaled = (draw.clamp(0.0, 0.999_999_999_999) * len as f64) as usize;
    scaled.min(len - 1)
}
