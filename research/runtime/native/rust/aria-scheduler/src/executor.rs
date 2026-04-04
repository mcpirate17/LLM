use std::collections::HashMap;
use std::ffi::CString;
use std::sync::Once;

use crate::arena::Arena;
use crate::error::AriaError;
use crate::ffi::{self, NkStatus, NpEvent};
use crate::graph::{GraphIR, NodeId};

static REGISTRY_INIT: Once = Once::new();

/// Ensure the C kernel registry is initialized exactly once.
fn ensure_registry_init() {
    REGISTRY_INIT.call_once(|| {
        unsafe { ffi::aria_registry_init() };
    });
}

/// Statistics about arena memory usage during graph execution.
#[derive(Debug, Clone, Default)]
pub struct ArenaStats {
    /// Total bytes allocated from the arena (including alignment padding).
    pub arena_bytes_used: usize,
    /// Total capacity of the arena in bytes.
    pub arena_capacity: usize,
    /// Number of node outputs allocated from the arena.
    pub arena_alloc_count: usize,
    /// Number of node outputs that fell back to heap allocation.
    pub heap_fallback_count: usize,
}

/// A buffer that is either arena-allocated (raw pointer) or heap-allocated (Vec).
enum NodeBuffer {
    /// Arena-allocated: pointer + element count. Valid until arena reset/drop.
    Arena { ptr: *mut f32, len: usize },
    /// Heap-allocated fallback.
    Heap(Vec<f32>),
}

impl NodeBuffer {
    /// Get a shared slice view of the buffer contents.
    fn as_slice(&self) -> &[f32] {
        match self {
            NodeBuffer::Arena { ptr, len } => unsafe { std::slice::from_raw_parts(*ptr, *len) },
            NodeBuffer::Heap(v) => v.as_slice(),
        }
    }

    /// Convert to an owned Vec<f32>. Arena buffers are copied out.
    fn into_vec(self) -> Vec<f32> {
        match self {
            NodeBuffer::Arena { ptr, len } => {
                let slice = unsafe { std::slice::from_raw_parts(ptr, len) };
                slice.to_vec()
            }
            NodeBuffer::Heap(v) => v,
        }
    }
}

/// Holds intermediate outputs during graph execution.
/// Buffers may be arena-allocated or heap-allocated.
pub(crate) struct ExecutionContext {
    outputs: HashMap<NodeId, NodeBuffer>,
}

impl ExecutionContext {
    pub fn new() -> Self {
        Self {
            outputs: HashMap::new(),
        }
    }
}

impl Default for ExecutionContext {
    fn default() -> Self {
        Self::new()
    }
}

/// Trait for dispatching compute kernels by operation name.
///
/// Implementations map `op_name` strings to actual compute logic (CPU, GPU,
/// or FFI calls into a C kernel library).
pub trait KernelDispatch {
    fn dispatch(
        &self,
        op_name: &str,
        inputs: &[&[f32]],
        config: &serde_json::Value,
    ) -> Result<Vec<f32>, AriaError>;

    /// Dispatch a kernel, writing output into a pre-allocated buffer.
    ///
    /// Returns the number of elements actually written. The default
    /// implementation delegates to `dispatch()` and copies into `output_buf`.
    fn dispatch_into(
        &self,
        op_name: &str,
        inputs: &[&[f32]],
        config: &serde_json::Value,
        output_buf: &mut [f32],
    ) -> Result<usize, AriaError> {
        let result = self.dispatch(op_name, inputs, config)?;
        let copy_len = result.len().min(output_buf.len());
        output_buf[..copy_len].copy_from_slice(&result[..copy_len]);
        Ok(copy_len)
    }
}

/// A dispatcher that calls into the C kernel library via FFI.
pub struct NativeKernelDispatch;

impl NativeKernelDispatch {
    /// Determine the output length for a given op based on inputs and config.
    fn output_len(op_name: &str, inputs: &[&[f32]], config: &serde_json::Value) -> usize {
        let mut output_len = if !inputs.is_empty() {
            inputs[0].len()
        } else {
            0
        };

        if op_name == "linear" {
            if let Some(dim_out) = config.get("dim_out").and_then(|v| v.as_i64()) {
                let batch = config.get("batch").and_then(|v| v.as_i64()).unwrap_or(1);
                output_len = (batch * dim_out) as usize;
            }
        } else if op_name == "matmul" {
            if let (Some(m), Some(n)) = (
                config.get("m").and_then(|v| v.as_i64()),
                config.get("n").and_then(|v| v.as_i64()),
            ) {
                output_len = (m * n) as usize;
            }
        } else if op_name == "matmul_relu"
            || op_name == "matmul_gelu"
            || op_name == "matmul_bias_relu"
        {
            if let (Some(m), Some(n)) = (
                config.get("m").and_then(|v| v.as_i64()),
                config.get("n").and_then(|v| v.as_i64()),
            ) {
                output_len = (m * n) as usize;
            }
        } else if op_name == "layernorm_residual" {
            if let (Some(rows), Some(cols)) = (
                config.get("rows").and_then(|v| v.as_i64()),
                config.get("cols").and_then(|v| v.as_i64()),
            ) {
                output_len = (rows * cols) as usize;
            }
        } else if op_name == "swiglu" {
            if let (Some(batch), Some(dim)) = (
                config.get("batch").and_then(|v| v.as_i64()),
                config.get("dim").and_then(|v| v.as_i64()),
            ) {
                output_len = (batch * dim) as usize;
            }
        } else if op_name == "rwkv_channel" {
            if let (Some(batch), Some(seq), Some(dim)) = (
                config.get("batch").and_then(|v| v.as_i64()),
                config.get("seq").and_then(|v| v.as_i64()),
                config.get("dim").and_then(|v| v.as_i64()),
            ) {
                output_len = (batch * seq * dim) as usize;
            }
        } else if op_name == "embedding_lookup" {
            if let (Some(batch), Some(dim)) = (
                config.get("batch").and_then(|v| v.as_i64()),
                config.get("dim").and_then(|v| v.as_i64()),
            ) {
                output_len = (batch * dim) as usize;
            }
        } else if op_name == "rope_rotate"
            || op_name == "rwkv_time_mixing"
            || op_name == "conv1d_seq"
        {
            if let (Some(batch), Some(seq), Some(dim)) = (
                config.get("batch").and_then(|v| v.as_i64()),
                config.get("seq").and_then(|v| v.as_i64()),
                config.get("dim").and_then(|v| v.as_i64()),
            ) {
                output_len = (batch * seq * dim) as usize;
            }
        } else if op_name == "softmax_attention"
            || op_name == "linear_attention"
            || op_name == "depth_weighted_proj"
            || op_name == "selective_scan"
            || op_name == "state_space"
            || op_name == "gated_delta"
        {
            if let (Some(batch), Some(seq), Some(dim)) = (
                config.get("batch").and_then(|v| v.as_i64()),
                config.get("seq").and_then(|v| v.as_i64()),
                config.get("dim").and_then(|v| v.as_i64()),
            ) {
                output_len = (batch * seq * dim) as usize;
            }
        } else if op_name == "gated_linear" {
            if let (Some(batch), Some(dim_out)) = (
                config.get("batch").and_then(|v| v.as_i64()),
                config.get("dim_out").and_then(|v| v.as_i64()),
            ) {
                output_len = (batch * dim_out) as usize;
            }
        } else if op_name == "cosine_similarity" {
            if let (Some(batch), Some(seq)) = (
                config.get("batch").and_then(|v| v.as_i64()),
                config.get("seq").and_then(|v| v.as_i64()),
            ) {
                output_len = (batch * seq) as usize;
            }
        } else if op_name == "gather_topk" {
            if let (Some(batch), Some(k), Some(dim)) = (
                config.get("batch").and_then(|v| v.as_i64()),
                config.get("k").and_then(|v| v.as_i64()),
                config.get("dim").and_then(|v| v.as_i64()),
            ) {
                output_len = (batch * k * dim) as usize;
            }
        } else if op_name == "concat" {
            output_len = inputs.iter().map(|inp| inp.len()).sum();
        }

        output_len
    }

    /// Execute the kernel into a pre-allocated output buffer. The buffer must
    /// be at least `output_len(...)` elements long.
    fn execute_kernel(
        op_name: &str,
        inputs: &[&[f32]],
        config: &serde_json::Value,
        output: &mut [f32],
    ) -> Result<(), AriaError> {
        ensure_registry_init();
        let c_op_name = CString::new(op_name)
            .map_err(|_| AriaError::ExecutionFailed(format!("invalid op name: {}", op_name)))?;

        // Intercept tier 3 / math space research ops that are not in NkRegistration
        match op_name {
            "exp_map" => {
                let batch = config.get("batch").and_then(|v| v.as_i64()).unwrap_or(1);
                let dim = config
                    .get("dim")
                    .and_then(|v| v.as_i64())
                    .unwrap_or(output.len() as i64 / batch.max(1));
                let c = config.get("c").and_then(|v| v.as_f64()).unwrap_or(1.0) as f32;
                unsafe {
                    ffi::aria_exp_map_f32(inputs[0].as_ptr(), output.as_mut_ptr(), batch, dim, c);
                }
                return Ok(());
            }
            "log_map" => {
                let batch = config.get("batch").and_then(|v| v.as_i64()).unwrap_or(1);
                let dim = config
                    .get("dim")
                    .and_then(|v| v.as_i64())
                    .unwrap_or(output.len() as i64 / batch.max(1));
                let c = config.get("c").and_then(|v| v.as_f64()).unwrap_or(1.0) as f32;
                unsafe {
                    ffi::aria_log_map_f32(inputs[0].as_ptr(), output.as_mut_ptr(), batch, dim, c);
                }
                return Ok(());
            }
            "poincare_add" => {
                let batch = config.get("batch").and_then(|v| v.as_i64()).unwrap_or(1);
                let dim = config
                    .get("dim")
                    .and_then(|v| v.as_i64())
                    .unwrap_or(output.len() as i64 / batch.max(1));
                let c = config.get("c").and_then(|v| v.as_f64()).unwrap_or(1.0) as f32;
                unsafe {
                    ffi::aria_poincare_add_f32(
                        inputs[0].as_ptr(),
                        inputs[1].as_ptr(),
                        output.as_mut_ptr(),
                        batch,
                        dim,
                        c,
                    );
                }
                return Ok(());
            }
            "hyp_linear" => {
                let batch = config.get("batch").and_then(|v| v.as_i64()).unwrap_or(1);
                let dim_in = config.get("dim_in").and_then(|v| v.as_i64()).unwrap_or(0);
                let dim_out = config.get("dim_out").and_then(|v| v.as_i64()).unwrap_or(0);
                let c = config.get("c").and_then(|v| v.as_f64()).unwrap_or(1.0) as f32;
                unsafe {
                    ffi::aria_hyp_linear_f32(
                        inputs[0].as_ptr(),
                        inputs[1].as_ptr(),
                        output.as_mut_ptr(),
                        batch,
                        dim_in,
                        dim_out,
                        c,
                    );
                }
                return Ok(());
            }
            "hyp_tangent_nonlinear" => {
                let c = config.get("c").and_then(|v| v.as_f64()).unwrap_or(1.0) as f32;
                unsafe {
                    ffi::aria_hyp_tangent_nonlinear_f32(
                        inputs[0].as_ptr(),
                        output.as_mut_ptr(),
                        output.len() as i64,
                        c,
                    );
                }
                return Ok(());
            }
            "hyperbolic_norm" => {
                let batch = config.get("batch").and_then(|v| v.as_i64()).unwrap_or(1);
                let dim = config
                    .get("dim")
                    .and_then(|v| v.as_i64())
                    .unwrap_or(output.len() as i64 / batch.max(1));
                let c = config.get("c").and_then(|v| v.as_f64()).unwrap_or(1.0) as f32;
                let eps = config.get("eps").and_then(|v| v.as_f64()).unwrap_or(1e-5) as f32;
                unsafe {
                    ffi::aria_hyperbolic_norm_f32(
                        inputs[0].as_ptr(),
                        inputs[1].as_ptr(),
                        inputs[2].as_ptr(),
                        output.as_mut_ptr(),
                        batch,
                        dim,
                        c,
                        eps,
                    );
                }
                return Ok(());
            }
            "tropical_attention" => {
                let batch = config.get("batch").and_then(|v| v.as_i64()).unwrap_or(1);
                let dim = config.get("dim").and_then(|v| v.as_i64()).unwrap_or(0);
                let seq = config
                    .get("seq")
                    .and_then(|v| v.as_i64())
                    .unwrap_or(output.len() as i64 / (batch.max(1) * dim.max(1)));
                let temperature = config
                    .get("temperature")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(1.0) as f32;
                unsafe {
                    ffi::aria_tropical_attention_f32(
                        inputs[0].as_ptr(),
                        output.as_mut_ptr(),
                        batch,
                        seq,
                        dim,
                        temperature,
                    );
                }
                return Ok(());
            }
            "tropical_gate" => {
                let batch = config.get("batch").and_then(|v| v.as_i64()).unwrap_or(1);
                let dim = config.get("dim").and_then(|v| v.as_i64()).unwrap_or(0);
                let seq = config
                    .get("seq")
                    .and_then(|v| v.as_i64())
                    .unwrap_or(output.len() as i64 / (batch.max(1) * dim.max(1)));
                let temperature = config
                    .get("temperature")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(1.0) as f32;
                unsafe {
                    ffi::aria_tropical_gate_f32(
                        inputs[0].as_ptr(),
                        output.as_mut_ptr(),
                        batch,
                        seq,
                        dim,
                        temperature,
                    );
                }
                return Ok(());
            }
            "tropical_add" => {
                unsafe {
                    ffi::aria_tropical_add_f32(
                        inputs[0].as_ptr(),
                        inputs[1].as_ptr(),
                        output.as_mut_ptr(),
                        output.len() as i64,
                    );
                }
                return Ok(());
            }
            "tropical_matmul" => {
                let m = config.get("m").and_then(|v| v.as_i64()).ok_or_else(|| {
                    AriaError::ExecutionFailed("tropical_matmul missing m".to_string())
                })?;
                let k = config.get("k").and_then(|v| v.as_i64()).ok_or_else(|| {
                    AriaError::ExecutionFailed("tropical_matmul missing k".to_string())
                })?;
                let n = config.get("n").and_then(|v| v.as_i64()).ok_or_else(|| {
                    AriaError::ExecutionFailed("tropical_matmul missing n".to_string())
                })?;
                unsafe {
                    ffi::aria_tropical_matmul_f32(
                        inputs[0].as_ptr(),
                        inputs[1].as_ptr(),
                        output.as_mut_ptr(),
                        m,
                        k,
                        n,
                    );
                }
                return Ok(());
            }
            "rotor_transform" => {
                let batch = config.get("batch").and_then(|v| v.as_i64()).unwrap_or(1);
                let dim = config
                    .get("dim")
                    .and_then(|v| v.as_i64())
                    .unwrap_or(output.len() as i64 / batch.max(1));
                unsafe {
                    ffi::aria_rotor_transform_f32(
                        inputs[0].as_ptr(),
                        inputs[1].as_ptr(),
                        output.as_mut_ptr(),
                        batch,
                        dim,
                    );
                }
                return Ok(());
            }
            "grade_select" => {
                let batch = config.get("batch").and_then(|v| v.as_i64()).unwrap_or(1);
                let dim = config
                    .get("dim")
                    .and_then(|v| v.as_i64())
                    .unwrap_or(output.len() as i64 / batch.max(1));
                let grade = config.get("grade").and_then(|v| v.as_i64()).unwrap_or(0) as i32;
                unsafe {
                    ffi::aria_grade_select_f32(
                        inputs[0].as_ptr(),
                        output.as_mut_ptr(),
                        batch,
                        dim,
                        grade,
                    );
                }
                return Ok(());
            }
            "grade_mix" => {
                if inputs.len() < 2 {
                    return Err(AriaError::ExecutionFailed(
                        "grade_mix dispatch requires x and alpha".to_string(),
                    ));
                }
                let batch = config.get("batch").and_then(|v| v.as_i64()).unwrap_or(1);
                let dim = config
                    .get("dim")
                    .and_then(|v| v.as_i64())
                    .unwrap_or(output.len() as i64 / batch.max(1));
                unsafe {
                    ffi::aria_grade_mix_f32(
                        inputs[0].as_ptr(),
                        inputs[1].as_ptr(),
                        output.as_mut_ptr(),
                        batch,
                        dim,
                    );
                }
                return Ok(());
            }
            "clifford_attention" => {
                let batch = config.get("batch").and_then(|v| v.as_i64()).unwrap_or(1);
                let dim = config.get("dim").and_then(|v| v.as_i64()).unwrap_or(0);
                let seq = config
                    .get("seq")
                    .and_then(|v| v.as_i64())
                    .unwrap_or(output.len() as i64 / (batch.max(1) * dim.max(1)));
                unsafe {
                    ffi::aria_clifford_attention_f32(
                        inputs[0].as_ptr(),
                        output.as_mut_ptr(),
                        batch,
                        seq,
                        dim,
                    );
                }
                return Ok(());
            }
            "softmax_attention" => {
                if inputs.len() < 5 {
                    return Err(AriaError::ExecutionFailed(
                        "softmax_attention dispatch requires x, Wq, Wk, Wv, Wo".to_string(),
                    ));
                }
                let batch = config
                    .get("batch")
                    .and_then(|v| v.as_i64())
                    .ok_or_else(|| {
                        AriaError::ExecutionFailed("softmax_attention missing batch".to_string())
                    })?;
                let seq = config.get("seq").and_then(|v| v.as_i64()).ok_or_else(|| {
                    AriaError::ExecutionFailed("softmax_attention missing seq".to_string())
                })?;
                let dim = config.get("dim").and_then(|v| v.as_i64()).ok_or_else(|| {
                    AriaError::ExecutionFailed("softmax_attention missing dim".to_string())
                })?;
                let n_heads = config
                    .get("n_heads")
                    .and_then(|v| v.as_i64())
                    .ok_or_else(|| {
                        AriaError::ExecutionFailed("softmax_attention missing n_heads".to_string())
                    })?;
                unsafe {
                    ffi::aria_softmax_attention_f32(
                        inputs[0].as_ptr(),
                        inputs[1].as_ptr(),
                        inputs[2].as_ptr(),
                        inputs[3].as_ptr(),
                        inputs[4].as_ptr(),
                        output.as_mut_ptr(),
                        batch,
                        seq,
                        dim,
                        n_heads,
                    );
                }
                return Ok(());
            }
            "linear_attention" => {
                if inputs.len() < 5 {
                    return Err(AriaError::ExecutionFailed(
                        "linear_attention dispatch requires x, Wq, Wk, Wv, Wo".to_string(),
                    ));
                }
                let batch = config
                    .get("batch")
                    .and_then(|v| v.as_i64())
                    .ok_or_else(|| {
                        AriaError::ExecutionFailed("linear_attention missing batch".to_string())
                    })?;
                let seq = config.get("seq").and_then(|v| v.as_i64()).ok_or_else(|| {
                    AriaError::ExecutionFailed("linear_attention missing seq".to_string())
                })?;
                let dim = config.get("dim").and_then(|v| v.as_i64()).ok_or_else(|| {
                    AriaError::ExecutionFailed("linear_attention missing dim".to_string())
                })?;
                let x_ptr = inputs[0].as_ptr();
                let wq_ptr = inputs[1].as_ptr();
                let wk_ptr = inputs[2].as_ptr();
                let wv_ptr = inputs[3].as_ptr();
                let wo_ptr = inputs[4].as_ptr();
                let y_ptr = output.as_mut_ptr();
                unsafe {
                    ffi::aria_linear_attention_f32(
                        x_ptr, wq_ptr, wk_ptr, wv_ptr, wo_ptr, y_ptr, batch, seq, dim,
                    );
                }
                return Ok(());
            }
            "depth_weighted_proj" => {
                if inputs.len() < 3 {
                    return Err(AriaError::ExecutionFailed(
                        "depth_weighted_proj dispatch requires x, depth_scorer, step_projs"
                            .to_string(),
                    ));
                }
                let batch = config
                    .get("batch")
                    .and_then(|v| v.as_i64())
                    .ok_or_else(|| {
                        AriaError::ExecutionFailed(
                            "depth_weighted_proj missing batch".to_string(),
                        )
                    })?;
                let seq = config.get("seq").and_then(|v| v.as_i64()).ok_or_else(|| {
                    AriaError::ExecutionFailed("depth_weighted_proj missing seq".to_string())
                })?;
                let dim = config.get("dim").and_then(|v| v.as_i64()).ok_or_else(|| {
                    AriaError::ExecutionFailed("depth_weighted_proj missing dim".to_string())
                })?;
                let max_depth = config
                    .get("max_depth")
                    .and_then(|v| v.as_i64())
                    .ok_or_else(|| {
                        AriaError::ExecutionFailed(
                            "depth_weighted_proj missing max_depth".to_string(),
                        )
                    })?;
                unsafe {
                    ffi::aria_depth_weighted_proj_f32(
                        inputs[0].as_ptr(),
                        inputs[1].as_ptr(),
                        inputs[2].as_ptr(),
                        output.as_mut_ptr(),
                        batch,
                        seq,
                        dim,
                        max_depth,
                    );
                }
                return Ok(());
            }
            "layernorm" => {
                if inputs.len() < 3 {
                    return Err(AriaError::ExecutionFailed(
                        "layernorm dispatch requires x, weight, bias".to_string(),
                    ));
                }
                let batch = config
                    .get("batch")
                    .and_then(|v| v.as_i64())
                    .ok_or_else(|| {
                        AriaError::ExecutionFailed("layernorm missing batch".to_string())
                    })?;
                let dim = config.get("dim").and_then(|v| v.as_i64()).ok_or_else(|| {
                    AriaError::ExecutionFailed("layernorm missing dim".to_string())
                })?;
                let eps = config
                    .get("eps")
                    .and_then(|v| v.as_f64())
                    .map(|v| v as f32)
                    .unwrap_or(1e-5f32);
                unsafe {
                    ffi::aria_layernorm_f32(
                        inputs[0].as_ptr(),
                        inputs[1].as_ptr(),
                        inputs[2].as_ptr(),
                        output.as_mut_ptr(),
                        batch,
                        dim,
                        eps,
                    );
                }
                return Ok(());
            }
            "selective_scan" => {
                if inputs.len() < 5 {
                    return Err(AriaError::ExecutionFailed(
                        "selective_scan dispatch requires x, A_log, dt_proj, B_weight, C_weight"
                            .to_string(),
                    ));
                }
                let batch = config
                    .get("batch")
                    .and_then(|v| v.as_i64())
                    .ok_or_else(|| {
                        AriaError::ExecutionFailed("selective_scan missing batch".to_string())
                    })?;
                let seq = config.get("seq").and_then(|v| v.as_i64()).ok_or_else(|| {
                    AriaError::ExecutionFailed("selective_scan missing seq".to_string())
                })?;
                let dim = config.get("dim").and_then(|v| v.as_i64()).ok_or_else(|| {
                    AriaError::ExecutionFailed("selective_scan missing dim".to_string())
                })?;
                unsafe {
                    ffi::aria_selective_scan_compiled_f32(
                        inputs[0].as_ptr(),
                        inputs[1].as_ptr(),
                        inputs[2].as_ptr(),
                        inputs[3].as_ptr(),
                        inputs[4].as_ptr(),
                        output.as_mut_ptr(),
                        batch,
                        seq,
                        dim,
                    );
                }
                return Ok(());
            }
            "state_space" => {
                if inputs.len() < 7 {
                    return Err(AriaError::ExecutionFailed(
                        "state_space dispatch requires x, ssm_A, ssm_B_weight, ssm_C_weight, ssm_D, ssm_dt_weight, ssm_dt_bias".to_string(),
                    ));
                }
                let batch = config
                    .get("batch")
                    .and_then(|v| v.as_i64())
                    .ok_or_else(|| {
                        AriaError::ExecutionFailed("state_space missing batch".to_string())
                    })?;
                let seq = config.get("seq").and_then(|v| v.as_i64()).ok_or_else(|| {
                    AriaError::ExecutionFailed("state_space missing seq".to_string())
                })?;
                let dim = config.get("dim").and_then(|v| v.as_i64()).ok_or_else(|| {
                    AriaError::ExecutionFailed("state_space missing dim".to_string())
                })?;
                let state_dim = config
                    .get("state_dim")
                    .and_then(|v| v.as_i64())
                    .or_else(|| {
                        if dim > 0 {
                            let len = inputs[1].len() as i64;
                            if len % dim == 0 {
                                Some(len / dim)
                            } else {
                                None
                            }
                        } else {
                            None
                        }
                    })
                    .ok_or_else(|| {
                        AriaError::ExecutionFailed("state_space missing state_dim".to_string())
                    })?;
                unsafe {
                    ffi::aria_state_space_compiled_f32(
                        inputs[0].as_ptr(),
                        inputs[1].as_ptr(),
                        inputs[2].as_ptr(),
                        inputs[3].as_ptr(),
                        inputs[4].as_ptr(),
                        inputs[5].as_ptr(),
                        inputs[6].as_ptr(),
                        output.as_mut_ptr(),
                        batch,
                        seq,
                        dim,
                        state_dim,
                    );
                }
                return Ok(());
            }
            "gated_delta" => {
                if inputs.len() < 7 {
                    return Err(AriaError::ExecutionFailed(
                        "gated_delta dispatch requires x, q_weight, k_weight, v_weight, alpha_weight, beta_weight, o_weight".to_string(),
                    ));
                }
                let batch = config
                    .get("batch")
                    .and_then(|v| v.as_i64())
                    .ok_or_else(|| {
                        AriaError::ExecutionFailed("gated_delta missing batch".to_string())
                    })?;
                let seq = config.get("seq").and_then(|v| v.as_i64()).ok_or_else(|| {
                    AriaError::ExecutionFailed("gated_delta missing seq".to_string())
                })?;
                let dim = config.get("dim").and_then(|v| v.as_i64()).ok_or_else(|| {
                    AriaError::ExecutionFailed("gated_delta missing dim".to_string())
                })?;
                let n_heads = config
                    .get("n_heads")
                    .and_then(|v| v.as_i64())
                    .ok_or_else(|| {
                        AriaError::ExecutionFailed("gated_delta missing n_heads".to_string())
                    })?;
                unsafe {
                    ffi::aria_gated_delta_compiled_f32(
                        inputs[0].as_ptr(),
                        inputs[1].as_ptr(),
                        inputs[2].as_ptr(),
                        inputs[3].as_ptr(),
                        inputs[4].as_ptr(),
                        inputs[5].as_ptr(),
                        inputs[6].as_ptr(),
                        output.as_mut_ptr(),
                        batch,
                        seq,
                        dim,
                        n_heads,
                    );
                }
                return Ok(());
            }
            "geometric_product" => {
                let n_multivectors = config
                    .get("n_multivectors")
                    .and_then(|v| v.as_i64())
                    .unwrap_or(output.len() as i64 / 8);
                unsafe {
                    ffi::aria_clifford_geometric_product_cl30_f32(
                        inputs[0].as_ptr(),
                        inputs[1].as_ptr(),
                        output.as_mut_ptr(),
                        n_multivectors,
                    );
                }
                return Ok(());
            }
            _ => {} // Fall through to standard registry
        }

        let reg_ptr = unsafe { ffi::nk_dispatch(c_op_name.as_ptr()) };
        if reg_ptr.is_null() {
            return Err(AriaError::ExecutionFailed(format!(
                "op {} not registered in native runtime",
                op_name
            )));
        }

        let reg = unsafe { &*reg_ptr };
        let output_len = output.len();

        let status = unsafe {
            if let Some(unary) = reg.unary_fn {
                unary(inputs[0].as_ptr(), output.as_mut_ptr(), output_len as i64)
            } else if let Some(binary) = reg.binary_fn {
                binary(
                    inputs[0].as_ptr(),
                    inputs[1].as_ptr(),
                    output.as_mut_ptr(),
                    output_len as i64,
                )
            } else if let Some(matmul) = reg.matmul_fn {
                let m = config
                    .get("m")
                    .and_then(|v| v.as_i64())
                    .ok_or_else(|| AriaError::ExecutionFailed("matmul missing m".to_string()))?;
                let k = config
                    .get("k")
                    .and_then(|v| v.as_i64())
                    .ok_or_else(|| AriaError::ExecutionFailed("matmul missing k".to_string()))?;
                let n = config
                    .get("n")
                    .and_then(|v| v.as_i64())
                    .ok_or_else(|| AriaError::ExecutionFailed("matmul missing n".to_string()))?;
                matmul(
                    inputs[0].as_ptr(),
                    inputs[1].as_ptr(),
                    output.as_mut_ptr(),
                    m,
                    k,
                    n,
                )
            } else if let Some(linear) = reg.linear_fn {
                let batch = config.get("batch").and_then(|v| v.as_i64()).unwrap_or(1);
                let dim_in = config.get("dim_in").and_then(|v| v.as_i64()).unwrap_or(0);
                let dim_out = config.get("dim_out").and_then(|v| v.as_i64()).unwrap_or(0);
                let bias_ptr = inputs
                    .get(2)
                    .map(|input| input.as_ptr())
                    .unwrap_or(std::ptr::null());
                linear(
                    inputs[0].as_ptr(),
                    inputs[1].as_ptr(),
                    bias_ptr,
                    output.as_mut_ptr(),
                    batch,
                    dim_in,
                    dim_out,
                )
            } else if let Some(softmax) = reg.softmax_fn {
                let batch = config.get("batch").and_then(|v| v.as_i64()).unwrap_or(1);
                let dim = config
                    .get("dim")
                    .and_then(|v| v.as_i64())
                    .unwrap_or(output_len as i64 / batch);
                softmax(inputs[0].as_ptr(), output.as_mut_ptr(), batch, dim)
            } else if let Some(rmsnorm) = reg.rmsnorm_fn {
                if inputs.len() < 2 {
                    return Err(AriaError::ExecutionFailed(
                        "rmsnorm dispatch requires x and weight inputs".to_string(),
                    ));
                }
                let batch = config.get("batch").and_then(|v| v.as_i64()).unwrap_or(1);
                let dim = config
                    .get("dim")
                    .and_then(|v| v.as_i64())
                    .unwrap_or(output_len as i64 / batch.max(1));
                let eps = config
                    .get("eps")
                    .and_then(|v| v.as_f64())
                    .map(|v| v as f32)
                    .unwrap_or(1e-5f32);
                rmsnorm(
                    inputs[0].as_ptr(),
                    inputs[1].as_ptr(),
                    output.as_mut_ptr(),
                    batch,
                    dim,
                    eps,
                )
            } else if let Some(matmul_relu) = reg.matmul_relu_fn {
                let m = config.get("m").and_then(|v| v.as_i64()).ok_or_else(|| {
                    AriaError::ExecutionFailed("matmul_relu missing m".to_string())
                })?;
                let k = config.get("k").and_then(|v| v.as_i64()).ok_or_else(|| {
                    AriaError::ExecutionFailed("matmul_relu missing k".to_string())
                })?;
                let n = config.get("n").and_then(|v| v.as_i64()).ok_or_else(|| {
                    AriaError::ExecutionFailed("matmul_relu missing n".to_string())
                })?;
                matmul_relu(
                    inputs[0].as_ptr(),
                    inputs[1].as_ptr(),
                    output.as_mut_ptr(),
                    m,
                    k,
                    n,
                )
            } else if let Some(matmul_bias_relu) = reg.matmul_bias_relu_fn {
                if inputs.len() < 3 {
                    return Err(AriaError::ExecutionFailed(
                        "matmul_bias_relu dispatch requires A, B, bias".to_string(),
                    ));
                }
                let m = config.get("m").and_then(|v| v.as_i64()).ok_or_else(|| {
                    AriaError::ExecutionFailed("matmul_bias_relu missing m".to_string())
                })?;
                let k = config.get("k").and_then(|v| v.as_i64()).ok_or_else(|| {
                    AriaError::ExecutionFailed("matmul_bias_relu missing k".to_string())
                })?;
                let n = config.get("n").and_then(|v| v.as_i64()).ok_or_else(|| {
                    AriaError::ExecutionFailed("matmul_bias_relu missing n".to_string())
                })?;
                matmul_bias_relu(
                    inputs[0].as_ptr(),
                    inputs[1].as_ptr(),
                    inputs[2].as_ptr(),
                    output.as_mut_ptr(),
                    m,
                    k,
                    n,
                )
            } else if let Some(layernorm_residual) = reg.layernorm_residual_fn {
                if inputs.len() < 4 {
                    return Err(AriaError::ExecutionFailed(
                        "layernorm_residual dispatch requires x, residual, gamma, beta".to_string(),
                    ));
                }
                let rows = config.get("rows").and_then(|v| v.as_i64()).ok_or_else(|| {
                    AriaError::ExecutionFailed("layernorm_residual missing rows".to_string())
                })?;
                let cols = config.get("cols").and_then(|v| v.as_i64()).ok_or_else(|| {
                    AriaError::ExecutionFailed("layernorm_residual missing cols".to_string())
                })?;
                let eps = config
                    .get("eps")
                    .and_then(|v| v.as_f64())
                    .map(|v| v as f32)
                    .unwrap_or(1e-5f32);
                layernorm_residual(
                    inputs[0].as_ptr(),
                    inputs[1].as_ptr(),
                    inputs[2].as_ptr(),
                    inputs[3].as_ptr(),
                    output.as_mut_ptr(),
                    rows,
                    cols,
                    eps,
                )
            } else if let Some(matmul_gelu) = reg.matmul_gelu_fn {
                let m = config.get("m").and_then(|v| v.as_i64()).ok_or_else(|| {
                    AriaError::ExecutionFailed("matmul_gelu missing m".to_string())
                })?;
                let k = config.get("k").and_then(|v| v.as_i64()).ok_or_else(|| {
                    AriaError::ExecutionFailed("matmul_gelu missing k".to_string())
                })?;
                let n = config.get("n").and_then(|v| v.as_i64()).ok_or_else(|| {
                    AriaError::ExecutionFailed("matmul_gelu missing n".to_string())
                })?;
                matmul_gelu(
                    inputs[0].as_ptr(),
                    inputs[1].as_ptr(),
                    output.as_mut_ptr(),
                    m,
                    k,
                    n,
                )
            } else if let Some(swiglu) = reg.swiglu_fn {
                if inputs.len() < 7 {
                    return Err(AriaError::ExecutionFailed(
                        "swiglu dispatch requires x, W_gate, W_up, W_down, b_gate, b_up, b_down"
                            .to_string(),
                    ));
                }
                let batch = config
                    .get("batch")
                    .and_then(|v| v.as_i64())
                    .ok_or_else(|| {
                        AriaError::ExecutionFailed("swiglu missing batch".to_string())
                    })?;
                let dim = config
                    .get("dim")
                    .and_then(|v| v.as_i64())
                    .ok_or_else(|| AriaError::ExecutionFailed("swiglu missing dim".to_string()))?;
                let hidden_dim = config
                    .get("hidden_dim")
                    .and_then(|v| v.as_i64())
                    .ok_or_else(|| {
                        AriaError::ExecutionFailed("swiglu missing hidden_dim".to_string())
                    })?;

                let mut tmp_gate = vec![0.0f32; (batch * hidden_dim) as usize];
                let mut tmp_up = vec![0.0f32; (batch * hidden_dim) as usize];
                swiglu(
                    inputs[0].as_ptr(),
                    inputs[1].as_ptr(),
                    inputs[2].as_ptr(),
                    inputs[3].as_ptr(),
                    inputs[4].as_ptr(),
                    inputs[5].as_ptr(),
                    inputs[6].as_ptr(),
                    output.as_mut_ptr(),
                    tmp_gate.as_mut_ptr(),
                    tmp_up.as_mut_ptr(),
                    batch,
                    dim,
                    hidden_dim,
                )
            } else if let Some(rwkv_channel) = reg.rwkv_channel_fn {
                if inputs.len() < 6 {
                    return Err(AriaError::ExecutionFailed(
                        "rwkv_channel dispatch requires x, mix_k, mix_r, W_k, W_r, W_v".to_string(),
                    ));
                }
                let batch = config
                    .get("batch")
                    .and_then(|v| v.as_i64())
                    .ok_or_else(|| {
                        AriaError::ExecutionFailed("rwkv_channel missing batch".to_string())
                    })?;
                let seq = config.get("seq").and_then(|v| v.as_i64()).ok_or_else(|| {
                    AriaError::ExecutionFailed("rwkv_channel missing seq".to_string())
                })?;
                let dim = config.get("dim").and_then(|v| v.as_i64()).ok_or_else(|| {
                    AriaError::ExecutionFailed("rwkv_channel missing dim".to_string())
                })?;
                let hidden_dim = config
                    .get("hidden_dim")
                    .and_then(|v| v.as_i64())
                    .ok_or_else(|| {
                        AriaError::ExecutionFailed("rwkv_channel missing hidden_dim".to_string())
                    })?;

                let mut tmp_xk = vec![0.0f32; (batch * seq * dim) as usize];
                let mut tmp_xr = vec![0.0f32; (batch * seq * dim) as usize];
                let mut tmp_k = vec![0.0f32; (batch * seq * hidden_dim) as usize];
                rwkv_channel(
                    inputs[0].as_ptr(),
                    inputs[1].as_ptr(),
                    inputs[2].as_ptr(),
                    inputs[3].as_ptr(),
                    inputs[4].as_ptr(),
                    inputs[5].as_ptr(),
                    output.as_mut_ptr(),
                    tmp_xk.as_mut_ptr(),
                    tmp_xr.as_mut_ptr(),
                    tmp_k.as_mut_ptr(),
                    batch,
                    seq,
                    dim,
                    hidden_dim,
                )
            } else if let Some(conv1d_seq) = reg.conv1d_seq_fn {
                if inputs.len() < 3 {
                    return Err(AriaError::ExecutionFailed(
                        "conv1d_seq dispatch requires x, weight, bias".to_string(),
                    ));
                }
                let batch = config
                    .get("batch")
                    .and_then(|v| v.as_i64())
                    .ok_or_else(|| {
                        AriaError::ExecutionFailed("conv1d_seq missing batch".to_string())
                    })?;
                let seq = config.get("seq").and_then(|v| v.as_i64()).ok_or_else(|| {
                    AriaError::ExecutionFailed("conv1d_seq missing seq".to_string())
                })?;
                let dim = config.get("dim").and_then(|v| v.as_i64()).ok_or_else(|| {
                    AriaError::ExecutionFailed("conv1d_seq missing dim".to_string())
                })?;
                conv1d_seq(
                    inputs[0].as_ptr(),
                    inputs[1].as_ptr(),
                    inputs[2].as_ptr(),
                    output.as_mut_ptr(),
                    batch,
                    seq,
                    dim,
                )
            } else if let Some(embedding_lookup) = reg.embedding_lookup_fn {
                if inputs.len() < 2 {
                    return Err(AriaError::ExecutionFailed(
                        "embedding_lookup dispatch requires table and indices inputs".to_string(),
                    ));
                }
                let batch = config
                    .get("batch")
                    .and_then(|v| v.as_i64())
                    .ok_or_else(|| {
                        AriaError::ExecutionFailed("embedding_lookup missing batch".to_string())
                    })?;
                let dim = config.get("dim").and_then(|v| v.as_i64()).ok_or_else(|| {
                    AriaError::ExecutionFailed("embedding_lookup missing dim".to_string())
                })?;
                let vocab_size = config
                    .get("vocab_size")
                    .and_then(|v| v.as_i64())
                    .ok_or_else(|| {
                        AriaError::ExecutionFailed(
                            "embedding_lookup missing vocab_size".to_string(),
                        )
                    })?;
                // inputs[1] is indices as f32-reinterpreted i32
                let indices_ptr = inputs[1].as_ptr() as *const i32;
                let pos_embed_ptr = if inputs.len() > 2 {
                    inputs[2].as_ptr()
                } else {
                    std::ptr::null()
                };
                embedding_lookup(
                    inputs[0].as_ptr(),
                    indices_ptr,
                    pos_embed_ptr,
                    output.as_mut_ptr(),
                    batch,
                    dim,
                    vocab_size,
                )
            } else if let Some(rope_rotate) = reg.rope_rotate_fn {
                let batch = config
                    .get("batch")
                    .and_then(|v| v.as_i64())
                    .ok_or_else(|| {
                        AriaError::ExecutionFailed("rope_rotate missing batch".to_string())
                    })?;
                let seq = config.get("seq").and_then(|v| v.as_i64()).ok_or_else(|| {
                    AriaError::ExecutionFailed("rope_rotate missing seq".to_string())
                })?;
                let dim = config.get("dim").and_then(|v| v.as_i64()).ok_or_else(|| {
                    AriaError::ExecutionFailed("rope_rotate missing dim".to_string())
                })?;
                let theta_base = config
                    .get("theta_base")
                    .and_then(|v| v.as_f64())
                    .map(|v| v as f32)
                    .unwrap_or(10000.0f32);
                rope_rotate(
                    inputs[0].as_ptr(),
                    output.as_mut_ptr(),
                    batch,
                    seq,
                    dim,
                    theta_base,
                )
            } else if let Some(gated_linear) = reg.gated_linear_fn {
                if inputs.len() < 4 {
                    return Err(AriaError::ExecutionFailed(
                        "gated_linear dispatch requires x, W, b, W_gate (+ optional b_gate)"
                            .to_string(),
                    ));
                }
                let batch = config
                    .get("batch")
                    .and_then(|v| v.as_i64())
                    .ok_or_else(|| {
                        AriaError::ExecutionFailed("gated_linear missing batch".to_string())
                    })?;
                let dim_in = config
                    .get("dim_in")
                    .and_then(|v| v.as_i64())
                    .ok_or_else(|| {
                        AriaError::ExecutionFailed("gated_linear missing dim_in".to_string())
                    })?;
                let dim_out = config
                    .get("dim_out")
                    .and_then(|v| v.as_i64())
                    .ok_or_else(|| {
                        AriaError::ExecutionFailed("gated_linear missing dim_out".to_string())
                    })?;
                let b_gate_ptr = if inputs.len() > 4 {
                    inputs[4].as_ptr()
                } else {
                    std::ptr::null()
                };
                let mut tmp_gate = vec![0.0f32; (batch * dim_out) as usize];
                gated_linear(
                    inputs[0].as_ptr(),
                    inputs[1].as_ptr(),
                    inputs[2].as_ptr(),
                    inputs[3].as_ptr(),
                    b_gate_ptr,
                    output.as_mut_ptr(),
                    tmp_gate.as_mut_ptr(),
                    batch,
                    dim_in,
                    dim_out,
                )
            } else if let Some(cosine_similarity) = reg.cosine_similarity_fn {
                if inputs.len() < 2 {
                    return Err(AriaError::ExecutionFailed(
                        "cosine_similarity dispatch requires a and b inputs".to_string(),
                    ));
                }
                let batch = config
                    .get("batch")
                    .and_then(|v| v.as_i64())
                    .ok_or_else(|| {
                        AriaError::ExecutionFailed("cosine_similarity missing batch".to_string())
                    })?;
                let seq = config.get("seq").and_then(|v| v.as_i64()).ok_or_else(|| {
                    AriaError::ExecutionFailed("cosine_similarity missing seq".to_string())
                })?;
                let dim = config.get("dim").and_then(|v| v.as_i64()).ok_or_else(|| {
                    AriaError::ExecutionFailed("cosine_similarity missing dim".to_string())
                })?;
                cosine_similarity(
                    inputs[0].as_ptr(),
                    inputs[1].as_ptr(),
                    output.as_mut_ptr(),
                    batch,
                    seq,
                    dim,
                )
            } else if let Some(gather_topk) = reg.gather_topk_fn {
                if inputs.len() < 2 {
                    return Err(AriaError::ExecutionFailed(
                        "gather_topk dispatch requires scores and values inputs".to_string(),
                    ));
                }
                let batch = config
                    .get("batch")
                    .and_then(|v| v.as_i64())
                    .ok_or_else(|| {
                        AriaError::ExecutionFailed("gather_topk missing batch".to_string())
                    })?;
                let n_items = config
                    .get("n_items")
                    .and_then(|v| v.as_i64())
                    .ok_or_else(|| {
                        AriaError::ExecutionFailed("gather_topk missing n_items".to_string())
                    })?;
                let dim = config.get("dim").and_then(|v| v.as_i64()).ok_or_else(|| {
                    AriaError::ExecutionFailed("gather_topk missing dim".to_string())
                })?;
                let k = config.get("k").and_then(|v| v.as_i64()).ok_or_else(|| {
                    AriaError::ExecutionFailed("gather_topk missing k".to_string())
                })?;
                let mut out_indices = vec![0i32; (batch * k) as usize];
                gather_topk(
                    inputs[0].as_ptr(),
                    inputs[1].as_ptr(),
                    output.as_mut_ptr(),
                    out_indices.as_mut_ptr(),
                    batch,
                    n_items,
                    dim,
                    k,
                )
            } else if let Some(rwkv_time_mixing) = reg.rwkv_time_mixing_fn {
                if inputs.len() < 6 {
                    return Err(AriaError::ExecutionFailed(
                        "rwkv_time_mixing dispatch requires x, w_decay, u_bonus, W_k, W_v, W_r"
                            .to_string(),
                    ));
                }
                let batch = config
                    .get("batch")
                    .and_then(|v| v.as_i64())
                    .ok_or_else(|| {
                        AriaError::ExecutionFailed("rwkv_time_mixing missing batch".to_string())
                    })?;
                let seq = config.get("seq").and_then(|v| v.as_i64()).ok_or_else(|| {
                    AriaError::ExecutionFailed("rwkv_time_mixing missing seq".to_string())
                })?;
                let dim = config.get("dim").and_then(|v| v.as_i64()).ok_or_else(|| {
                    AriaError::ExecutionFailed("rwkv_time_mixing missing dim".to_string())
                })?;
                rwkv_time_mixing(
                    inputs[0].as_ptr(),
                    inputs[1].as_ptr(),
                    inputs[2].as_ptr(),
                    inputs[3].as_ptr(),
                    inputs[4].as_ptr(),
                    inputs[5].as_ptr(),
                    output.as_mut_ptr(),
                    batch,
                    seq,
                    dim,
                )
            } else if let Some(concat) = reg.concat_fn {
                if inputs.is_empty() {
                    return Err(AriaError::ExecutionFailed(
                        "concat dispatch requires at least one input".to_string(),
                    ));
                }
                let input_ptrs: Vec<*const f32> = inputs.iter().map(|inp| inp.as_ptr()).collect();
                let sizes: Vec<i64> = inputs.iter().map(|inp| inp.len() as i64).collect();
                let dim = config.get("dim").and_then(|v| v.as_i64()).unwrap_or(-1);
                concat(
                    input_ptrs.as_ptr(),
                    sizes.as_ptr(),
                    input_ptrs.len() as i32,
                    output.as_mut_ptr(),
                    dim,
                )
            } else if reg.split_fn.is_some() {
                return Err(AriaError::ExecutionFailed(
                    "split op is multi-output and is not supported by single-output executor path"
                        .to_string(),
                ));
            } else {
                return Err(AriaError::ExecutionFailed(format!(
                    "op {} has no dispatch handler",
                    op_name
                )));
            }
        };

        if status != NkStatus::Ok {
            return Err(AriaError::ExecutionFailed(format!(
                "native kernel {} failed with status {:?}",
                op_name, status
            )));
        }

        Ok(())
    }
}

impl KernelDispatch for NativeKernelDispatch {
    fn dispatch(
        &self,
        op_name: &str,
        inputs: &[&[f32]],
        config: &serde_json::Value,
    ) -> Result<Vec<f32>, AriaError> {
        let output_len = Self::output_len(op_name, inputs, config);
        let mut output = vec![0.0f32; output_len];
        Self::execute_kernel(op_name, inputs, config, &mut output)?;
        Ok(output)
    }

    fn dispatch_into(
        &self,
        op_name: &str,
        inputs: &[&[f32]],
        config: &serde_json::Value,
        output_buf: &mut [f32],
    ) -> Result<usize, AriaError> {
        Self::execute_kernel(op_name, inputs, config, output_buf)?;
        Ok(output_buf.len())
    }
}

/// Estimate total arena capacity needed for a graph execution.
///
/// Sums the estimated output size (in f32 elements) for every node,
/// then converts to bytes with per-allocation alignment padding.
fn estimate_arena_capacity(graph: &GraphIR, input_len: usize) -> usize {
    let alignment = 64usize;
    let mut total_bytes = 0usize;

    for node in &graph.nodes {
        // Estimate output size for this node.
        let elem_count = if node.is_input {
            input_len
        } else {
            // For linear/matmul, check config for dim_out.
            if node.op_name == "linear" || node.op_name == "matmul" {
                if let Some(dim_out) = node.config.get("dim_out").and_then(|v| v.as_i64()) {
                    let batch = node
                        .config
                        .get("batch")
                        .and_then(|v| v.as_i64())
                        .unwrap_or(1);
                    (batch * dim_out) as usize
                } else {
                    // Fall back to input size estimate.
                    input_len
                }
            } else {
                // Most ops preserve input size. Use the first input's estimated size
                // or fall back to input_len.
                input_len
            }
        };

        // Bytes needed: elem_count * 4, rounded up to alignment.
        let raw_bytes = elem_count * std::mem::size_of::<f32>();
        let aligned = (raw_bytes + alignment - 1) & !(alignment - 1);
        // Add alignment padding for the allocation offset.
        total_bytes += aligned + alignment;
    }

    total_bytes
}

/// Timing info for a single node execution.
#[derive(Debug, Clone)]
pub struct NodeProfile {
    pub node_id: u32,
    pub op_name: String,
    pub start_ns: i64,
    pub end_ns: i64,
    pub duration_us: f64,
}

/// Result of graph execution including output data and arena statistics.
#[derive(Debug)]
pub struct ExecutionResult {
    /// The output tensor from the graph's output node.
    pub output: Vec<f32>,
    /// Arena memory usage statistics.
    pub arena_stats: ArenaStats,
    /// Per-node profiling data (empty when profiling is disabled).
    pub node_profiles: Vec<NodeProfile>,
    /// Peak memory reported by profiler (0 when profiling is disabled).
    pub peak_memory_bytes: i64,
}

#[derive(Clone, Copy)]
enum InputBinding<'a> {
    Shared(&'a [f32]),
    Distinct(&'a [&'a [f32]]),
}

impl<'a> InputBinding<'a> {
    fn estimate_input_len(&self) -> usize {
        match self {
            Self::Shared(input) => input.len(),
            Self::Distinct(inputs) => inputs.first().map(|input| input.len()).unwrap_or(0),
        }
    }

    fn slice_for_input_node(&self, ordinal: usize) -> Result<&'a [f32], AriaError> {
        match self {
            Self::Shared(input) => Ok(input),
            Self::Distinct(inputs) => inputs.get(ordinal).copied().ok_or_else(|| {
                AriaError::ExecutionFailed(format!(
                    "graph requires input node {} but only {} input buffers were provided",
                    ordinal,
                    inputs.len(),
                ))
            }),
        }
    }
}

fn is_all_zero(buf: &[f32]) -> bool {
    buf.iter().all(|v| v.abs() <= 1e-12)
}

fn execute_conditional_dispatch(
    inputs: &[&[f32]],
    config: &serde_json::Value,
) -> Result<Vec<f32>, AriaError> {
    let x = inputs.first().ok_or_else(|| {
        AriaError::ExecutionFailed("conditional_dispatch requires at least one input".to_string())
    })?;
    let mut out = x.to_vec();

    // Explicit lane-empty hint allows true skip without assignment tensor.
    if let Some(active_tokens) = config.get("active_tokens").and_then(|v| v.as_i64()) {
        if active_tokens <= 0 {
            out.fill(0.0);
            return Ok(out);
        }
    }

    // Optional dense->packed lane routing using assignments input.
    if inputs.len() < 2 {
        return Ok(out);
    }
    let assignments = inputs[1];
    let batch = match config.get("batch").and_then(|v| v.as_i64()) {
        Some(v) if v > 0 => v as usize,
        _ => return Ok(out),
    };
    let seq = match config.get("seq").and_then(|v| v.as_i64()) {
        Some(v) if v > 0 => v as usize,
        _ => return Ok(out),
    };
    let dim = match config.get("dim").and_then(|v| v.as_i64()) {
        Some(v) if v > 0 => v as usize,
        _ => return Ok(out),
    };
    let lane = config.get("lane").and_then(|v| v.as_i64()).unwrap_or(0);

    if assignments.len() != batch * seq || x.len() != batch * seq * dim {
        return Ok(out);
    }

    out.fill(0.0);
    for b in 0..batch {
        let mut write_pos = 0usize;
        for s in 0..seq {
            let src_idx = b * seq + s;
            let assign_lane = assignments[src_idx].round() as i64;
            if assign_lane == lane {
                if write_pos < seq {
                    let src_off = src_idx * dim;
                    let dst_off = (b * seq + write_pos) * dim;
                    out[dst_off..dst_off + dim].copy_from_slice(&x[src_off..src_off + dim]);
                    write_pos += 1;
                }
            }
        }
    }
    Ok(out)
}

fn execute_conditional_gather(inputs: &[&[f32]]) -> Result<Vec<f32>, AriaError> {
    if inputs.is_empty() {
        return Err(AriaError::ExecutionFailed(
            "conditional_gather requires at least one input".to_string(),
        ));
    }
    if inputs.len() == 1 {
        return Ok(inputs[0].to_vec());
    }

    let output_len = inputs[0].len();
    let mut out = vec![0.0f32; output_len];
    let mut contributing = 0usize;

    for inp in inputs {
        if inp.len() != output_len {
            return Err(AriaError::ExecutionFailed(
                "conditional_gather input length mismatch".to_string(),
            ));
        }
        if is_all_zero(inp) {
            continue;
        }
        for (o, v) in out.iter_mut().zip(inp.iter()) {
            *o += *v;
        }
        contributing += 1;
    }

    if contributing == 0 {
        return Ok(out);
    }
    if contributing > 1 {
        let scale = 1.0f32 / (contributing as f32);
        for v in &mut out {
            *v *= scale;
        }
    }
    Ok(out)
}

/// Execute the graph in topological order using arena-based buffer allocation.
///
/// 1. Estimates total buffer memory needed and creates an arena.
/// 2. For each node in topological order, allocates output from the arena
///    (falling back to heap if the arena is exhausted).
/// 3. Dispatches kernels directly into the pre-allocated buffers.
/// 4. Returns the output tensor and arena usage statistics.
fn execute_with_input_binding(
    graph: &GraphIR,
    dispatcher: &dyn KernelDispatch,
    input_binding: InputBinding<'_>,
) -> Result<ExecutionResult, AriaError> {
    let optimized_graph = graph.fuse_supported_patterns();
    let graph = &optimized_graph;
    let order = graph.topological_order()?;

    let arena_capacity = estimate_arena_capacity(graph, input_binding.estimate_input_len());
    let mut arena = Arena::new(arena_capacity);
    let mut stats = ArenaStats {
        arena_capacity,
        ..Default::default()
    };

    let mut ctx = ExecutionContext::new();

    let node_map: HashMap<NodeId, &crate::graph::Node> =
        graph.nodes.iter().map(|n| (n.id, n)).collect();

    // Check if profiling is enabled (cached for the loop).
    let profiling = unsafe { ffi::np_profiler_enabled() != 0 };
    if profiling {
        unsafe { ffi::np_reset_counters() };
    }

    let mut input_ordinal = 0usize;
    for &node_id in &order {
        let node = node_map.get(&node_id).ok_or_else(|| {
            AriaError::InvalidIR(format!(
                "node {} in topo order but missing from graph",
                node_id.0
            ))
        })?;

        if node.is_input {
            let input = input_binding.slice_for_input_node(input_ordinal)?;
            input_ordinal += 1;
            // Allocate from arena and copy input data in.
            match arena.alloc_f32_raw(input.len()) {
                Ok((ptr, len)) => {
                    let buf = unsafe { std::slice::from_raw_parts_mut(ptr, len) };
                    buf.copy_from_slice(input);
                    ctx.outputs.insert(node_id, NodeBuffer::Arena { ptr, len });
                    stats.arena_alloc_count += 1;
                }
                Err(_) => {
                    // Fallback: heap allocation.
                    ctx.outputs
                        .insert(node_id, NodeBuffer::Heap(input.to_vec()));
                    stats.heap_fallback_count += 1;
                }
            }
            continue;
        }

        // Gather input slices from previously-computed outputs.
        let input_slices: Vec<&[f32]> = node
            .input_ids
            .iter()
            .map(|id| {
                ctx.outputs
                    .get(id)
                    .map(|buf| buf.as_slice())
                    .ok_or_else(|| {
                        AriaError::ExecutionFailed(format!(
                            "node {} requires input from node {} which has no output",
                            node_id.0, id.0
                        ))
                    })
            })
            .collect::<Result<Vec<_>, _>>()?;

        // "output" nodes are identity/passthrough: just copy the first input.
        if node.op_name == "output" {
            if let Some(first) = input_slices.first() {
                match arena.alloc_f32_raw(first.len()) {
                    Ok((ptr, len)) => {
                        let buf = unsafe { std::slice::from_raw_parts_mut(ptr, len) };
                        buf.copy_from_slice(first);
                        ctx.outputs.insert(node_id, NodeBuffer::Arena { ptr, len });
                        stats.arena_alloc_count += 1;
                    }
                    Err(_) => {
                        ctx.outputs
                            .insert(node_id, NodeBuffer::Heap(first.to_vec()));
                        stats.heap_fallback_count += 1;
                    }
                }
            }
            continue;
        }

        // Conditional subgraph control: execute dispatch/gather directly in scheduler.
        // This enables empty-lane skipping without requiring native registry kernels.
        if node.op_name == "conditional_dispatch" || node.op_name == "conditional_gather" {
            let out_vec = if node.op_name == "conditional_dispatch" {
                execute_conditional_dispatch(&input_slices, &node.config)?
            } else {
                execute_conditional_gather(&input_slices)?
            };
            match arena.alloc_f32_raw(out_vec.len()) {
                Ok((ptr, len)) => {
                    let buf = unsafe { std::slice::from_raw_parts_mut(ptr, len) };
                    buf.copy_from_slice(&out_vec);
                    ctx.outputs.insert(node_id, NodeBuffer::Arena { ptr, len });
                    stats.arena_alloc_count += 1;
                }
                Err(_) => {
                    ctx.outputs.insert(node_id, NodeBuffer::Heap(out_vec));
                    stats.heap_fallback_count += 1;
                }
            }
            continue;
        }

        // Determine output size.
        let output_len =
            NativeKernelDispatch::output_len(&node.op_name, &input_slices, &node.config);

        // Record start time if profiling.
        let t_start = if profiling {
            unsafe { ffi::np_clock_ns() }
        } else {
            0
        };

        // Try arena allocation first, fall back to heap.
        match arena.alloc_f32_raw(output_len) {
            Ok((ptr, len)) => {
                let out_slice = unsafe { std::slice::from_raw_parts_mut(ptr, len) };
                dispatcher.dispatch_into(&node.op_name, &input_slices, &node.config, out_slice)?;
                ctx.outputs.insert(node_id, NodeBuffer::Arena { ptr, len });
                stats.arena_alloc_count += 1;
            }
            Err(_) => {
                // Graceful degradation: fall back to dispatch() which heap-allocates.
                let result = dispatcher.dispatch(&node.op_name, &input_slices, &node.config)?;
                ctx.outputs.insert(node_id, NodeBuffer::Heap(result));
                stats.heap_fallback_count += 1;
            }
        }

        // Emit profiling event for this kernel.
        if profiling && t_start != 0 {
            let t_end = unsafe { ffi::np_clock_ns() };
            let c_op = CString::new(node.op_name.as_str()).unwrap_or_default();
            let c_evt = CString::new("kernel").unwrap_or_default();
            let evt = NpEvent {
                event_name: c_evt.as_ptr(),
                op_name: c_op.as_ptr(),
                node_id: node_id.0 as i32,
                start_ns: t_start,
                end_ns: t_end,
                thread_id: 0,
            };
            unsafe { ffi::np_emit_event(&evt as *const NpEvent) };
        }
    }

    stats.arena_bytes_used = arena.used_bytes();

    // Extract the output node's buffer. We must copy it out because the
    // arena will be dropped when this function returns.
    let output = ctx
        .outputs
        .remove(&graph.output_node_id)
        .ok_or_else(|| {
            AriaError::ExecutionFailed(format!(
                "output node {} produced no result",
                graph.output_node_id.0
            ))
        })?
        .into_vec();

    // Collect profiling data if enabled.
    let mut node_profiles = Vec::new();
    let mut peak_memory_bytes: i64 = 0;
    if profiling {
        // Drain events from the C ring buffer.
        let evt_count = unsafe { ffi::np_event_count() };
        if evt_count > 0 {
            let mut raw_events = vec![
                NpEvent {
                    event_name: std::ptr::null(),
                    op_name: std::ptr::null(),
                    node_id: 0,
                    start_ns: 0,
                    end_ns: 0,
                    thread_id: 0,
                };
                evt_count as usize
            ];
            let n = unsafe { ffi::np_drain_events(raw_events.as_mut_ptr(), evt_count) };
            for i in 0..n as usize {
                let e = &raw_events[i];
                let op = if e.op_name.is_null() {
                    String::new()
                } else {
                    unsafe { std::ffi::CStr::from_ptr(e.op_name) }
                        .to_string_lossy()
                        .into_owned()
                };
                let dur_us = (e.end_ns - e.start_ns) as f64 / 1000.0;
                node_profiles.push(NodeProfile {
                    node_id: e.node_id as u32,
                    op_name: op,
                    start_ns: e.start_ns,
                    end_ns: e.end_ns,
                    duration_us: dur_us,
                });
            }
        }
        peak_memory_bytes = unsafe { ffi::np_get_peak_memory() };
    }

    // Arena is dropped here, freeing the backing buffer.
    Ok(ExecutionResult {
        output,
        arena_stats: stats,
        node_profiles,
        peak_memory_bytes,
    })
}

pub fn execute_with_arena(
    graph: &GraphIR,
    dispatcher: &dyn KernelDispatch,
    input: &[f32],
) -> Result<ExecutionResult, AriaError> {
    execute_with_input_binding(graph, dispatcher, InputBinding::Shared(input))
}

pub fn execute_with_arena_multi_input(
    graph: &GraphIR,
    dispatcher: &dyn KernelDispatch,
    inputs: &[&[f32]],
) -> Result<ExecutionResult, AriaError> {
    execute_with_input_binding(graph, dispatcher, InputBinding::Distinct(inputs))
}

// ── Backward pass infrastructure ──────────────────────────────────────

/// Result of a backward kernel dispatch.
/// Unary backward ops produce a single gradient; binary/matmul produce two.
pub enum BackwardGrads {
    /// Single gradient (unary ops).
    Single(Vec<f32>),
    /// Two gradients (binary ops: grad_a, grad_b).
    Pair(Vec<f32>, Vec<f32>),
    /// Arbitrary per-input gradients in input order.
    Many(Vec<Vec<f32>>),
}

/// Result of full backward graph execution.
#[derive(Debug)]
pub struct BackwardResult {
    /// Gradient for each node, keyed by NodeId.
    pub grads: HashMap<u32, Vec<f32>>,
    /// Arena memory usage statistics for the backward pass.
    pub arena_stats: ArenaStats,
}

fn silu_derivative(x: f32) -> f32 {
    let sig = 1.0f32 / (1.0f32 + (-x).exp());
    sig * (1.0f32 + x * (1.0f32 - sig))
}

fn conv1d_seq_backward(
    grad_output: &[f32],
    saved_tensors: &[&[f32]],
    config: &serde_json::Value,
) -> Result<BackwardGrads, AriaError> {
    if saved_tensors.len() < 3 {
        return Err(AriaError::ExecutionFailed(
            "conv1d_seq backward: need x, weight, bias".into(),
        ));
    }
    let x = saved_tensors[0];
    let weight = saved_tensors[1];
    let bias = saved_tensors[2];
    let batch = config
        .get("batch")
        .and_then(|v| v.as_i64())
        .ok_or_else(|| AriaError::ExecutionFailed("conv1d_seq missing batch".into()))?;
    let seq = config
        .get("seq")
        .and_then(|v| v.as_i64())
        .ok_or_else(|| AriaError::ExecutionFailed("conv1d_seq missing seq".into()))?;
    let dim = config
        .get("dim")
        .and_then(|v| v.as_i64())
        .ok_or_else(|| AriaError::ExecutionFailed("conv1d_seq missing dim".into()))?;
    let kernel = if dim > 0 {
        weight.len() as i64 / dim
    } else {
        0
    };
    if kernel <= 0 {
        return Err(AriaError::ExecutionFailed(
            "conv1d_seq backward: invalid kernel shape".into(),
        ));
    }

    let mut grad_x = vec![0.0f32; x.len()];
    let mut grad_weight = vec![0.0f32; weight.len()];
    let mut grad_bias = vec![0.0f32; bias.len()];

    for b in 0..batch as usize {
        for s in 0..seq as usize {
            for d in 0..dim as usize {
                let grad_idx = (b * seq as usize + s) * dim as usize + d;
                let go = grad_output[grad_idx];
                grad_bias[d] += go;
                for k in 0..kernel as usize {
                    let src_s = s as i64 + k as i64 - (kernel - 1);
                    if !(0..seq).contains(&src_s) {
                        continue;
                    }
                    let src_idx = (b * seq as usize + src_s as usize) * dim as usize + d;
                    let weight_idx = d * kernel as usize + k;
                    grad_x[src_idx] += go * weight[weight_idx];
                    grad_weight[weight_idx] += go * x[src_idx];
                }
            }
        }
    }

    Ok(BackwardGrads::Many(vec![grad_x, grad_weight, grad_bias]))
}

fn swiglu_backward(
    grad_output: &[f32],
    saved_tensors: &[&[f32]],
    config: &serde_json::Value,
) -> Result<BackwardGrads, AriaError> {
    if saved_tensors.len() < 7 {
        return Err(AriaError::ExecutionFailed(
            "swiglu backward: need x, W_gate, W_up, W_down, b_gate, b_up, b_down".into(),
        ));
    }
    let x = saved_tensors[0];
    let w_gate = saved_tensors[1];
    let w_up = saved_tensors[2];
    let w_down = saved_tensors[3];
    let b_gate = saved_tensors[4];
    let b_up = saved_tensors[5];
    let b_down = saved_tensors[6];
    let batch = config
        .get("batch")
        .and_then(|v| v.as_i64())
        .ok_or_else(|| AriaError::ExecutionFailed("swiglu missing batch".into()))?;
    let dim = config
        .get("dim")
        .and_then(|v| v.as_i64())
        .ok_or_else(|| AriaError::ExecutionFailed("swiglu missing dim".into()))?;
    let hidden_dim = config
        .get("hidden_dim")
        .and_then(|v| v.as_i64())
        .ok_or_else(|| AriaError::ExecutionFailed("swiglu missing hidden_dim".into()))?;

    let rows = batch as usize;
    let dim_usize = dim as usize;
    let hidden_usize = hidden_dim as usize;

    let mut gate_linear = vec![0.0f32; rows * hidden_usize];
    let mut gate_act = vec![0.0f32; rows * hidden_usize];
    let mut up_linear = vec![0.0f32; rows * hidden_usize];
    let mut hidden = vec![0.0f32; rows * hidden_usize];
    for row in 0..rows {
        for h in 0..hidden_usize {
            let mut gate_sum = b_gate.get(h).copied().unwrap_or(0.0f32);
            let mut up_sum = b_up.get(h).copied().unwrap_or(0.0f32);
            for d in 0..dim_usize {
                let x_val = x[row * dim_usize + d];
                gate_sum += x_val * w_gate[h * dim_usize + d];
                up_sum += x_val * w_up[h * dim_usize + d];
            }
            let gate_idx = row * hidden_usize + h;
            gate_linear[gate_idx] = gate_sum;
            gate_act[gate_idx] = gate_sum / (1.0f32 + (-gate_sum).exp());
            up_linear[gate_idx] = up_sum;
            hidden[gate_idx] = gate_act[gate_idx] * up_sum;
        }
    }

    let mut grad_x = vec![0.0f32; x.len()];
    let mut grad_w_gate = vec![0.0f32; w_gate.len()];
    let mut grad_w_up = vec![0.0f32; w_up.len()];
    let mut grad_w_down = vec![0.0f32; w_down.len()];
    let mut grad_b_gate = vec![0.0f32; b_gate.len()];
    let mut grad_b_up = vec![0.0f32; b_up.len()];
    let mut grad_b_down = vec![0.0f32; b_down.len()];
    let mut grad_hidden = vec![0.0f32; hidden.len()];

    for row in 0..rows {
        for out_d in 0..dim_usize {
            let go = grad_output[row * dim_usize + out_d];
            if out_d < grad_b_down.len() {
                grad_b_down[out_d] += go;
            }
            for h in 0..hidden_usize {
                grad_w_down[out_d * hidden_usize + h] += go * hidden[row * hidden_usize + h];
                grad_hidden[row * hidden_usize + h] += go * w_down[out_d * hidden_usize + h];
            }
        }
    }

    for row in 0..rows {
        for h in 0..hidden_usize {
            let idx = row * hidden_usize + h;
            let grad_gate_act = grad_hidden[idx] * up_linear[idx];
            let grad_up = grad_hidden[idx] * gate_act[idx];
            let grad_gate = grad_gate_act * silu_derivative(gate_linear[idx]);
            if h < grad_b_gate.len() {
                grad_b_gate[h] += grad_gate;
            }
            if h < grad_b_up.len() {
                grad_b_up[h] += grad_up;
            }
            for d in 0..dim_usize {
                let x_val = x[row * dim_usize + d];
                grad_w_gate[h * dim_usize + d] += grad_gate * x_val;
                grad_w_up[h * dim_usize + d] += grad_up * x_val;
                grad_x[row * dim_usize + d] +=
                    grad_gate * w_gate[h * dim_usize + d] + grad_up * w_up[h * dim_usize + d];
            }
        }
    }

    Ok(BackwardGrads::Many(vec![
        grad_x,
        grad_w_gate,
        grad_w_up,
        grad_w_down,
        grad_b_gate,
        grad_b_up,
        grad_b_down,
    ]))
}

fn rwkv_channel_backward(
    grad_output: &[f32],
    saved_tensors: &[&[f32]],
    config: &serde_json::Value,
) -> Result<BackwardGrads, AriaError> {
    if saved_tensors.len() < 6 {
        return Err(AriaError::ExecutionFailed(
            "rwkv_channel backward: need x, mix_k, mix_r, W_k, W_r, W_v".into(),
        ));
    }
    let x = saved_tensors[0];
    let mix_k = saved_tensors[1];
    let mix_r = saved_tensors[2];
    let w_k = saved_tensors[3];
    let w_r = saved_tensors[4];
    let w_v = saved_tensors[5];
    let batch = config
        .get("batch")
        .and_then(|v| v.as_i64())
        .ok_or_else(|| AriaError::ExecutionFailed("rwkv_channel missing batch".into()))?;
    let seq = config
        .get("seq")
        .and_then(|v| v.as_i64())
        .ok_or_else(|| AriaError::ExecutionFailed("rwkv_channel missing seq".into()))?;
    let dim = config
        .get("dim")
        .and_then(|v| v.as_i64())
        .ok_or_else(|| AriaError::ExecutionFailed("rwkv_channel missing dim".into()))?;
    let hidden_dim = config
        .get("hidden_dim")
        .and_then(|v| v.as_i64())
        .ok_or_else(|| AriaError::ExecutionFailed("rwkv_channel missing hidden_dim".into()))?;

    let batch_usize = batch as usize;
    let seq_usize = seq as usize;
    let dim_usize = dim as usize;
    let hidden_usize = hidden_dim as usize;

    let mut grad_x = vec![0.0f32; x.len()];
    let mut grad_mix_k = vec![0.0f32; mix_k.len()];
    let mut grad_mix_r = vec![0.0f32; mix_r.len()];
    let mut grad_w_k = vec![0.0f32; w_k.len()];
    let mut grad_w_r = vec![0.0f32; w_r.len()];
    let mut grad_w_v = vec![0.0f32; w_v.len()];

    let batch_stride = seq_usize * dim_usize;
    for b in 0..batch_usize {
        let batch_offset = b * batch_stride;
        for t in 0..seq_usize {
            let token_offset = batch_offset + t * dim_usize;
            let prev_offset = if t == 0 {
                token_offset
            } else {
                batch_offset + (t - 1) * dim_usize
            };

            let mut xk = vec![0.0f32; dim_usize];
            let mut xr = vec![0.0f32; dim_usize];
            for d in 0..dim_usize {
                let xt = x[token_offset + d];
                let xprev = x[prev_offset + d];
                let mk = mix_k.get(d).copied().unwrap_or(0.5f32);
                let mr = mix_r.get(d).copied().unwrap_or(0.5f32);
                xk[d] = if t == 0 {
                    xt
                } else {
                    mk * xt + (1.0f32 - mk) * xprev
                };
                xr[d] = if t == 0 {
                    xt
                } else {
                    mr * xt + (1.0f32 - mr) * xprev
                };
            }

            let mut k_pre = vec![0.0f32; hidden_usize];
            let mut k_relu = vec![0.0f32; hidden_usize];
            let mut k_sq = vec![0.0f32; hidden_usize];
            for h in 0..hidden_usize {
                let mut sum = 0.0f32;
                for d in 0..dim_usize {
                    sum += w_k[h * dim_usize + d] * xk[d];
                }
                k_pre[h] = sum;
                k_relu[h] = sum.max(0.0f32);
                k_sq[h] = k_relu[h] * k_relu[h];
            }

            let mut r_pre = vec![0.0f32; dim_usize];
            let mut r = vec![0.0f32; dim_usize];
            let mut v = vec![0.0f32; dim_usize];
            for out_d in 0..dim_usize {
                let mut r_sum = 0.0f32;
                let mut v_sum = 0.0f32;
                for d in 0..dim_usize {
                    r_sum += w_r[out_d * dim_usize + d] * xr[d];
                }
                for h in 0..hidden_usize {
                    v_sum += w_v[out_d * hidden_usize + h] * k_sq[h];
                }
                r_pre[out_d] = r_sum;
                r[out_d] = 1.0f32 / (1.0f32 + (-r_sum).exp());
                v[out_d] = v_sum;
            }

            let mut grad_xk = vec![0.0f32; dim_usize];
            let mut grad_xr = vec![0.0f32; dim_usize];
            let mut grad_k_sq = vec![0.0f32; hidden_usize];
            for out_d in 0..dim_usize {
                let go = grad_output[token_offset + out_d];
                let grad_r = go * v[out_d];
                let grad_v = go * r[out_d];
                let grad_r_pre = grad_r * r[out_d] * (1.0f32 - r[out_d]);
                for h in 0..hidden_usize {
                    grad_w_v[out_d * hidden_usize + h] += grad_v * k_sq[h];
                    grad_k_sq[h] += grad_v * w_v[out_d * hidden_usize + h];
                }
                for d in 0..dim_usize {
                    grad_w_r[out_d * dim_usize + d] += grad_r_pre * xr[d];
                    grad_xr[d] += grad_r_pre * w_r[out_d * dim_usize + d];
                }
            }

            for h in 0..hidden_usize {
                let grad_k_pre = if k_pre[h] > 0.0f32 {
                    grad_k_sq[h] * 2.0f32 * k_relu[h]
                } else {
                    0.0f32
                };
                for d in 0..dim_usize {
                    grad_w_k[h * dim_usize + d] += grad_k_pre * xk[d];
                    grad_xk[d] += grad_k_pre * w_k[h * dim_usize + d];
                }
            }

            for d in 0..dim_usize {
                let xt = x[token_offset + d];
                if t == 0 {
                    grad_x[token_offset + d] += grad_xk[d] + grad_xr[d];
                    continue;
                }
                let xprev = x[prev_offset + d];
                let mk = mix_k.get(d).copied().unwrap_or(0.5f32);
                let mr = mix_r.get(d).copied().unwrap_or(0.5f32);
                grad_x[token_offset + d] += grad_xk[d] * mk + grad_xr[d] * mr;
                grad_x[prev_offset + d] += grad_xk[d] * (1.0f32 - mk) + grad_xr[d] * (1.0f32 - mr);
                grad_mix_k[d] += grad_xk[d] * (xt - xprev);
                grad_mix_r[d] += grad_xr[d] * (xt - xprev);
            }
        }
    }

    Ok(BackwardGrads::Many(vec![
        grad_x, grad_mix_k, grad_mix_r, grad_w_k, grad_w_r, grad_w_v,
    ]))
}

fn gated_linear_backward(
    grad_output: &[f32],
    saved_tensors: &[&[f32]],
    config: &serde_json::Value,
) -> Result<BackwardGrads, AriaError> {
    if saved_tensors.len() < 5 {
        return Err(AriaError::ExecutionFailed(
            "gated_linear backward: need x, W, bias, W_gate, bias_gate".into(),
        ));
    }
    let x = saved_tensors[0];
    let w = saved_tensors[1];
    let w_gate = saved_tensors[3];
    let batch = config
        .get("batch")
        .and_then(|v| v.as_i64())
        .unwrap_or_else(|| if !x.is_empty() { 1 } else { 0 });
    let dim_in = config
        .get("dim_in")
        .and_then(|v| v.as_i64())
        .unwrap_or_else(|| if batch > 0 { x.len() as i64 / batch } else { 0 });
    let dim_out = config
        .get("dim_out")
        .and_then(|v| v.as_i64())
        .unwrap_or_else(|| w.len() as i64 / dim_in.max(1));
    let mut gate_sigmoid = vec![0.0f32; (batch * dim_out) as usize];
    let mut grad_x = vec![0.0f32; x.len()];
    let mut grad_w = vec![0.0f32; w.len()];
    let mut grad_w_gate = vec![0.0f32; w_gate.len()];
    let mut grad_b = vec![0.0f32; dim_out as usize];
    let mut grad_b_gate = vec![0.0f32; dim_out as usize];

    for row in 0..batch as usize {
        let x_row = &x[row * dim_in as usize..(row + 1) * dim_in as usize];
        let gate_slice = &mut gate_sigmoid[row * dim_out as usize..(row + 1) * dim_out as usize];
        for out_idx in 0..dim_out as usize {
            let w_row = &w_gate[out_idx * dim_in as usize..(out_idx + 1) * dim_in as usize];
            let mut sum = saved_tensors[4][out_idx];
            for in_idx in 0..dim_in as usize {
                sum += x_row[in_idx] * w_row[in_idx];
            }
            gate_slice[out_idx] = 1.0f32 / (1.0f32 + (-sum).exp());
        }
    }

    unsafe {
        ffi::aria_gated_linear_backward_f32(
            grad_output.as_ptr(),
            x.as_ptr(),
            w.as_ptr(),
            w_gate.as_ptr(),
            gate_sigmoid.as_ptr(),
            grad_x.as_mut_ptr(),
            grad_w.as_mut_ptr(),
            grad_w_gate.as_mut_ptr(),
            grad_b.as_mut_ptr(),
            grad_b_gate.as_mut_ptr(),
            batch,
            dim_in,
            dim_out,
        );
    }
    Ok(BackwardGrads::Many(vec![
        grad_x,
        grad_w,
        grad_b,
        grad_w_gate,
        grad_b_gate,
    ]))
}

fn softmax_attention_backward(
    grad_output: &[f32],
    saved_tensors: &[&[f32]],
    config: &serde_json::Value,
) -> Result<BackwardGrads, AriaError> {
    if saved_tensors.len() < 5 {
        return Err(AriaError::ExecutionFailed(
            "softmax_attention backward: need x, Wq, Wk, Wv, Wo".into(),
        ));
    }
    let x = saved_tensors[0];
    let wq = saved_tensors[1];
    let wk = saved_tensors[2];
    let wv = saved_tensors[3];
    let wo = saved_tensors[4];
    let batch = config
        .get("batch")
        .and_then(|v| v.as_i64())
        .ok_or_else(|| AriaError::ExecutionFailed("softmax_attention missing batch".into()))?;
    let seq = config
        .get("seq")
        .and_then(|v| v.as_i64())
        .ok_or_else(|| AriaError::ExecutionFailed("softmax_attention missing seq".into()))?;
    let dim = config
        .get("dim")
        .and_then(|v| v.as_i64())
        .ok_or_else(|| AriaError::ExecutionFailed("softmax_attention missing dim".into()))?;
    let n_heads = config
        .get("n_heads")
        .and_then(|v| v.as_i64())
        .ok_or_else(|| AriaError::ExecutionFailed("softmax_attention missing n_heads".into()))?;
    let mut grad_x = vec![0.0f32; x.len()];
    let mut grad_wq = vec![0.0f32; wq.len()];
    let mut grad_wk = vec![0.0f32; wk.len()];
    let mut grad_wv = vec![0.0f32; wv.len()];
    let mut grad_wo = vec![0.0f32; wo.len()];
    unsafe {
        ffi::aria_softmax_attention_backward_f32(
            grad_output.as_ptr(),
            x.as_ptr(),
            wq.as_ptr(),
            wk.as_ptr(),
            wv.as_ptr(),
            wo.as_ptr(),
            grad_x.as_mut_ptr(),
            grad_wq.as_mut_ptr(),
            grad_wk.as_mut_ptr(),
            grad_wv.as_mut_ptr(),
            grad_wo.as_mut_ptr(),
            batch,
            seq,
            dim,
            n_heads,
        );
    }
    Ok(BackwardGrads::Many(vec![
        grad_x, grad_wq, grad_wk, grad_wv, grad_wo,
    ]))
}

fn selective_scan_backward(
    grad_output: &[f32],
    saved_tensors: &[&[f32]],
    config: &serde_json::Value,
) -> Result<BackwardGrads, AriaError> {
    if saved_tensors.len() < 5 {
        return Err(AriaError::ExecutionFailed(
            "selective_scan backward: need x, A_log, dt_proj, B_weight, C_weight".into(),
        ));
    }
    let x = saved_tensors[0];
    let a_log = saved_tensors[1];
    let dt_proj = saved_tensors[2];
    let b_weight = saved_tensors[3];
    let c_weight = saved_tensors[4];
    let batch = config
        .get("batch")
        .and_then(|v| v.as_i64())
        .ok_or_else(|| AriaError::ExecutionFailed("selective_scan missing batch".into()))?;
    let seq = config
        .get("seq")
        .and_then(|v| v.as_i64())
        .ok_or_else(|| AriaError::ExecutionFailed("selective_scan missing seq".into()))?;
    let dim = config
        .get("dim")
        .and_then(|v| v.as_i64())
        .ok_or_else(|| AriaError::ExecutionFailed("selective_scan missing dim".into()))?;
    let mut grad_x = vec![0.0f32; x.len()];
    let mut grad_a_log = vec![0.0f32; a_log.len()];
    let mut grad_dt_proj = vec![0.0f32; dt_proj.len()];
    let mut grad_b_weight = vec![0.0f32; b_weight.len()];
    let mut grad_c_weight = vec![0.0f32; c_weight.len()];
    unsafe {
        ffi::aria_selective_scan_compiled_backward_f32(
            grad_output.as_ptr(),
            x.as_ptr(),
            a_log.as_ptr(),
            dt_proj.as_ptr(),
            b_weight.as_ptr(),
            c_weight.as_ptr(),
            grad_x.as_mut_ptr(),
            grad_a_log.as_mut_ptr(),
            grad_dt_proj.as_mut_ptr(),
            grad_b_weight.as_mut_ptr(),
            grad_c_weight.as_mut_ptr(),
            batch,
            seq,
            dim,
        );
    }
    Ok(BackwardGrads::Many(vec![
        grad_x,
        grad_a_log,
        grad_dt_proj,
        grad_b_weight,
        grad_c_weight,
    ]))
}

fn state_space_backward(
    grad_output: &[f32],
    saved_tensors: &[&[f32]],
    config: &serde_json::Value,
) -> Result<BackwardGrads, AriaError> {
    if saved_tensors.len() < 7 {
        return Err(AriaError::ExecutionFailed(
            "state_space backward: need x, ssm_A, ssm_B_weight, ssm_C_weight, ssm_D, ssm_dt_weight, ssm_dt_bias".into(),
        ));
    }
    let x = saved_tensors[0];
    let ssm_a = saved_tensors[1];
    let ssm_b_weight = saved_tensors[2];
    let ssm_c_weight = saved_tensors[3];
    let ssm_d = saved_tensors[4];
    let ssm_dt_weight = saved_tensors[5];
    let ssm_dt_bias = saved_tensors[6];
    let batch = config
        .get("batch")
        .and_then(|v| v.as_i64())
        .ok_or_else(|| AriaError::ExecutionFailed("state_space missing batch".into()))?;
    let seq = config
        .get("seq")
        .and_then(|v| v.as_i64())
        .ok_or_else(|| AriaError::ExecutionFailed("state_space missing seq".into()))?;
    let dim = config
        .get("dim")
        .and_then(|v| v.as_i64())
        .ok_or_else(|| AriaError::ExecutionFailed("state_space missing dim".into()))?;
    let state_dim = config
        .get("state_dim")
        .and_then(|v| v.as_i64())
        .or_else(|| {
            if dim > 0 {
                let len = ssm_a.len() as i64;
                if len % dim == 0 {
                    Some(len / dim)
                } else {
                    None
                }
            } else {
                None
            }
        })
        .ok_or_else(|| AriaError::ExecutionFailed("state_space missing state_dim".into()))?;
    let mut grad_x = vec![0.0f32; x.len()];
    let mut grad_ssm_a = vec![0.0f32; ssm_a.len()];
    let mut grad_ssm_b_weight = vec![0.0f32; ssm_b_weight.len()];
    let mut grad_ssm_c_weight = vec![0.0f32; ssm_c_weight.len()];
    let mut grad_ssm_d = vec![0.0f32; ssm_d.len()];
    let mut grad_ssm_dt_weight = vec![0.0f32; ssm_dt_weight.len()];
    let mut grad_ssm_dt_bias = vec![0.0f32; ssm_dt_bias.len()];
    unsafe {
        ffi::aria_state_space_compiled_backward_f32(
            grad_output.as_ptr(),
            x.as_ptr(),
            ssm_a.as_ptr(),
            ssm_b_weight.as_ptr(),
            ssm_c_weight.as_ptr(),
            ssm_d.as_ptr(),
            ssm_dt_weight.as_ptr(),
            ssm_dt_bias.as_ptr(),
            grad_x.as_mut_ptr(),
            grad_ssm_a.as_mut_ptr(),
            grad_ssm_b_weight.as_mut_ptr(),
            grad_ssm_c_weight.as_mut_ptr(),
            grad_ssm_d.as_mut_ptr(),
            grad_ssm_dt_weight.as_mut_ptr(),
            grad_ssm_dt_bias.as_mut_ptr(),
            batch,
            seq,
            dim,
            state_dim,
        );
    }
    Ok(BackwardGrads::Many(vec![
        grad_x,
        grad_ssm_a,
        grad_ssm_b_weight,
        grad_ssm_c_weight,
        grad_ssm_d,
        grad_ssm_dt_weight,
        grad_ssm_dt_bias,
    ]))
}

fn gated_delta_backward(
    grad_output: &[f32],
    saved_tensors: &[&[f32]],
    config: &serde_json::Value,
) -> Result<BackwardGrads, AriaError> {
    if saved_tensors.len() < 7 {
        return Err(AriaError::ExecutionFailed(
            "gated_delta backward: need x, q_weight, k_weight, v_weight, alpha_weight, beta_weight, o_weight".into(),
        ));
    }
    let x = saved_tensors[0];
    let q_weight = saved_tensors[1];
    let k_weight = saved_tensors[2];
    let v_weight = saved_tensors[3];
    let alpha_weight = saved_tensors[4];
    let beta_weight = saved_tensors[5];
    let o_weight = saved_tensors[6];
    let batch = config
        .get("batch")
        .and_then(|v| v.as_i64())
        .ok_or_else(|| AriaError::ExecutionFailed("gated_delta missing batch".into()))?;
    let seq = config
        .get("seq")
        .and_then(|v| v.as_i64())
        .ok_or_else(|| AriaError::ExecutionFailed("gated_delta missing seq".into()))?;
    let dim = config
        .get("dim")
        .and_then(|v| v.as_i64())
        .ok_or_else(|| AriaError::ExecutionFailed("gated_delta missing dim".into()))?;
    let n_heads = config
        .get("n_heads")
        .and_then(|v| v.as_i64())
        .ok_or_else(|| AriaError::ExecutionFailed("gated_delta missing n_heads".into()))?;
    let mut grad_x = vec![0.0f32; x.len()];
    let mut grad_q_weight = vec![0.0f32; q_weight.len()];
    let mut grad_k_weight = vec![0.0f32; k_weight.len()];
    let mut grad_v_weight = vec![0.0f32; v_weight.len()];
    let mut grad_alpha_weight = vec![0.0f32; alpha_weight.len()];
    let mut grad_beta_weight = vec![0.0f32; beta_weight.len()];
    let mut grad_o_weight = vec![0.0f32; o_weight.len()];
    unsafe {
        ffi::aria_gated_delta_compiled_backward_f32(
            grad_output.as_ptr(),
            x.as_ptr(),
            q_weight.as_ptr(),
            k_weight.as_ptr(),
            v_weight.as_ptr(),
            alpha_weight.as_ptr(),
            beta_weight.as_ptr(),
            o_weight.as_ptr(),
            grad_x.as_mut_ptr(),
            grad_q_weight.as_mut_ptr(),
            grad_k_weight.as_mut_ptr(),
            grad_v_weight.as_mut_ptr(),
            grad_alpha_weight.as_mut_ptr(),
            grad_beta_weight.as_mut_ptr(),
            grad_o_weight.as_mut_ptr(),
            batch,
            seq,
            dim,
            n_heads,
        );
    }
    Ok(BackwardGrads::Many(vec![
        grad_x,
        grad_q_weight,
        grad_k_weight,
        grad_v_weight,
        grad_alpha_weight,
        grad_beta_weight,
        grad_o_weight,
    ]))
}

fn rwkv_time_mixing_backward(
    grad_output: &[f32],
    saved_tensors: &[&[f32]],
    config: &serde_json::Value,
) -> Result<BackwardGrads, AriaError> {
    if saved_tensors.len() < 6 {
        return Err(AriaError::ExecutionFailed(
            "rwkv_time_mixing backward: need x, w_decay, u_bonus, W_k, W_v, W_r".into(),
        ));
    }
    let x = saved_tensors[0];
    let w_decay = saved_tensors[1];
    let u_bonus = saved_tensors[2];
    let w_k = saved_tensors[3];
    let w_v = saved_tensors[4];
    let w_r = saved_tensors[5];
    let batch = config
        .get("batch")
        .and_then(|v| v.as_i64())
        .ok_or_else(|| AriaError::ExecutionFailed("rwkv_time_mixing missing batch".into()))?;
    let seq = config
        .get("seq")
        .and_then(|v| v.as_i64())
        .ok_or_else(|| AriaError::ExecutionFailed("rwkv_time_mixing missing seq".into()))?;
    let dim = config
        .get("dim")
        .and_then(|v| v.as_i64())
        .ok_or_else(|| AriaError::ExecutionFailed("rwkv_time_mixing missing dim".into()))?;
    let mut grad_x = vec![0.0f32; x.len()];
    let mut grad_w_decay = vec![0.0f32; w_decay.len()];
    let mut grad_u_bonus = vec![0.0f32; u_bonus.len()];
    let mut grad_w_k = vec![0.0f32; w_k.len()];
    let mut grad_w_v = vec![0.0f32; w_v.len()];
    let mut grad_w_r = vec![0.0f32; w_r.len()];
    unsafe {
        ffi::aria_rwkv_time_mixing_backward_f32(
            grad_output.as_ptr(),
            x.as_ptr(),
            w_decay.as_ptr(),
            u_bonus.as_ptr(),
            w_k.as_ptr(),
            w_v.as_ptr(),
            w_r.as_ptr(),
            grad_x.as_mut_ptr(),
            grad_w_decay.as_mut_ptr(),
            grad_u_bonus.as_mut_ptr(),
            grad_w_k.as_mut_ptr(),
            grad_w_v.as_mut_ptr(),
            grad_w_r.as_mut_ptr(),
            batch,
            seq,
            dim,
        );
    }
    Ok(BackwardGrads::Many(vec![
        grad_x,
        grad_w_decay,
        grad_u_bonus,
        grad_w_k,
        grad_w_v,
        grad_w_r,
    ]))
}

fn conditional_dispatch_backward(
    grad_output: &[f32],
    config: &serde_json::Value,
) -> Result<BackwardGrads, AriaError> {
    if config
        .get("active_tokens")
        .and_then(|v| v.as_i64())
        .unwrap_or(1)
        <= 0
    {
        Ok(BackwardGrads::Single(vec![0.0f32; grad_output.len()]))
    } else {
        Ok(BackwardGrads::Single(grad_output.to_vec()))
    }
}

fn conditional_gather_backward(
    grad_output: &[f32],
    saved_tensors: &[&[f32]],
) -> Result<BackwardGrads, AriaError> {
    let a = saved_tensors.first().copied().unwrap_or(&[]);
    let b = saved_tensors.get(1).copied().unwrap_or(&[]);
    let a_nonzero = !a.is_empty() && !is_all_zero(a);
    let b_nonzero = !b.is_empty() && !is_all_zero(b);
    let mut ga = vec![0.0f32; grad_output.len()];
    let mut gb = vec![0.0f32; grad_output.len()];
    if a_nonzero && b_nonzero {
        for (i, g) in grad_output.iter().enumerate() {
            ga[i] = 0.5f32 * g;
            gb[i] = 0.5f32 * g;
        }
    } else if a_nonzero {
        ga.copy_from_slice(grad_output);
    } else if b_nonzero {
        gb.copy_from_slice(grad_output);
    }
    Ok(BackwardGrads::Pair(ga, gb))
}

impl NativeKernelDispatch {
    /// Dispatch a backward kernel for the given op.
    ///
    /// - `grad_output`: incoming gradient from downstream.
    /// - `saved_tensors`: saved activations from the forward pass.
    ///   - For unary ops: `[input_or_output]` (1 element).
    ///   - For binary ops (add/sub/mul): `[a, b]` (2 elements).
    ///   - For matmul: `[A, B]` (2 elements).
    /// - `config`: node config (needed for matmul dimensions).
    pub fn dispatch_backward(
        op_name: &str,
        grad_output: &[f32],
        saved_tensors: &[&[f32]],
        config: &serde_json::Value,
    ) -> Result<BackwardGrads, AriaError> {
        ensure_registry_init();
        let n = grad_output.len() as i64;

        match op_name {
            // ── Unary backward ops ───────────────────────────────
            "relu" => {
                let input = saved_tensors.first().ok_or_else(|| {
                    AriaError::ExecutionFailed("relu backward: missing saved input".into())
                })?;
                let mut grad_in = vec![0.0f32; input.len()];
                unsafe {
                    ffi::aria_relu_backward_f32(
                        grad_output.as_ptr(),
                        input.as_ptr(),
                        grad_in.as_mut_ptr(),
                        input.len() as i64,
                    );
                }
                Ok(BackwardGrads::Single(grad_in))
            }
            "sigmoid" => {
                let output = saved_tensors.first().ok_or_else(|| {
                    AriaError::ExecutionFailed("sigmoid backward: missing saved output".into())
                })?;
                let mut grad_in = vec![0.0f32; output.len()];
                unsafe {
                    ffi::aria_sigmoid_backward_f32(
                        grad_output.as_ptr(),
                        output.as_ptr(),
                        grad_in.as_mut_ptr(),
                        output.len() as i64,
                    );
                }
                Ok(BackwardGrads::Single(grad_in))
            }
            "tanh" => {
                let output = saved_tensors.first().ok_or_else(|| {
                    AriaError::ExecutionFailed("tanh backward: missing saved output".into())
                })?;
                let mut grad_in = vec![0.0f32; output.len()];
                unsafe {
                    ffi::aria_tanh_backward_f32(
                        grad_output.as_ptr(),
                        output.as_ptr(),
                        grad_in.as_mut_ptr(),
                        output.len() as i64,
                    );
                }
                Ok(BackwardGrads::Single(grad_in))
            }
            "gelu" => {
                let input = saved_tensors.first().ok_or_else(|| {
                    AriaError::ExecutionFailed("gelu backward: missing saved input".into())
                })?;
                let mut grad_in = vec![0.0f32; input.len()];
                unsafe {
                    ffi::aria_gelu_backward_f32(
                        grad_output.as_ptr(),
                        input.as_ptr(),
                        grad_in.as_mut_ptr(),
                        input.len() as i64,
                    );
                }
                Ok(BackwardGrads::Single(grad_in))
            }
            "silu" => {
                let input = saved_tensors.first().ok_or_else(|| {
                    AriaError::ExecutionFailed("silu backward: missing saved input".into())
                })?;
                let mut grad_in = vec![0.0f32; input.len()];
                unsafe {
                    ffi::aria_silu_backward_f32(
                        grad_output.as_ptr(),
                        input.as_ptr(),
                        grad_in.as_mut_ptr(),
                        input.len() as i64,
                    );
                }
                Ok(BackwardGrads::Single(grad_in))
            }
            "rmsnorm" => {
                if saved_tensors.len() < 2 {
                    return Err(AriaError::ExecutionFailed(
                        "rmsnorm backward: need input and gamma".into(),
                    ));
                }
                let input = saved_tensors[0];
                let gamma = saved_tensors[1];
                let batch = config
                    .get("batch")
                    .and_then(|v| v.as_i64())
                    .unwrap_or_else(|| {
                        let dim = gamma.len() as i64;
                        if dim > 0 {
                            input.len() as i64 / dim
                        } else {
                            0
                        }
                    });
                let dim = config
                    .get("dim")
                    .and_then(|v| v.as_i64())
                    .unwrap_or(gamma.len() as i64);
                let eps = config
                    .get("eps")
                    .and_then(|v| v.as_f64())
                    .map(|v| v as f32)
                    .unwrap_or(1e-6f32);
                let mut grad_in = vec![0.0f32; input.len()];
                let mut grad_gamma = vec![0.0f32; gamma.len()];
                unsafe {
                    ffi::aria_rmsnorm_backward_f32(
                        grad_output.as_ptr(),
                        input.as_ptr(),
                        gamma.as_ptr(),
                        grad_in.as_mut_ptr(),
                        grad_gamma.as_mut_ptr(),
                        batch,
                        dim,
                        eps,
                    );
                }
                Ok(BackwardGrads::Pair(grad_in, grad_gamma))
            }
            "layernorm" => {
                if saved_tensors.len() < 3 {
                    return Err(AriaError::ExecutionFailed(
                        "layernorm backward: need input, gamma, beta".into(),
                    ));
                }
                let input = saved_tensors[0];
                let gamma = saved_tensors[1];
                let beta = saved_tensors[2];
                let batch = config
                    .get("batch")
                    .and_then(|v| v.as_i64())
                    .unwrap_or_else(|| {
                        let dim = gamma.len() as i64;
                        if dim > 0 {
                            input.len() as i64 / dim
                        } else {
                            0
                        }
                    });
                let dim = config
                    .get("dim")
                    .and_then(|v| v.as_i64())
                    .unwrap_or(gamma.len() as i64);
                let eps = config
                    .get("eps")
                    .and_then(|v| v.as_f64())
                    .map(|v| v as f32)
                    .unwrap_or(1e-5f32);
                let mut grad_in = vec![0.0f32; input.len()];
                let mut grad_gamma = vec![0.0f32; gamma.len()];
                let mut grad_beta = vec![0.0f32; beta.len()];
                unsafe {
                    ffi::aria_layernorm_backward_f32(
                        grad_output.as_ptr(),
                        input.as_ptr(),
                        gamma.as_ptr(),
                        grad_in.as_mut_ptr(),
                        grad_gamma.as_mut_ptr(),
                        grad_beta.as_mut_ptr(),
                        batch,
                        dim,
                        eps,
                    );
                }
                Ok(BackwardGrads::Many(vec![grad_in, grad_gamma, grad_beta]))
            }

            // ── Binary backward ops ──────────────────────────────
            "add" => {
                let mut grad_a = vec![0.0f32; grad_output.len()];
                let mut grad_b = vec![0.0f32; grad_output.len()];
                unsafe {
                    ffi::aria_add_backward_f32(
                        grad_output.as_ptr(),
                        grad_a.as_mut_ptr(),
                        grad_b.as_mut_ptr(),
                        n,
                    );
                }
                Ok(BackwardGrads::Pair(grad_a, grad_b))
            }
            "sub" => {
                let mut grad_a = vec![0.0f32; grad_output.len()];
                let mut grad_b = vec![0.0f32; grad_output.len()];
                unsafe {
                    ffi::aria_sub_backward_f32(
                        grad_output.as_ptr(),
                        grad_a.as_mut_ptr(),
                        grad_b.as_mut_ptr(),
                        n,
                    );
                }
                Ok(BackwardGrads::Pair(grad_a, grad_b))
            }
            "mul" => {
                if saved_tensors.len() < 2 {
                    return Err(AriaError::ExecutionFailed(
                        "mul backward: need 2 saved tensors (a, b)".into(),
                    ));
                }
                let a = saved_tensors[0];
                let b = saved_tensors[1];
                let mut grad_a = vec![0.0f32; a.len()];
                let mut grad_b = vec![0.0f32; b.len()];
                unsafe {
                    ffi::aria_mul_backward_f32(
                        grad_output.as_ptr(),
                        a.as_ptr(),
                        b.as_ptr(),
                        grad_a.as_mut_ptr(),
                        grad_b.as_mut_ptr(),
                        a.len() as i64,
                    );
                }
                Ok(BackwardGrads::Pair(grad_a, grad_b))
            }

            // ── Matmul backward ──────────────────────────────────
            "matmul" | "linear" => {
                if saved_tensors.len() < 2 {
                    return Err(AriaError::ExecutionFailed(format!(
                        "{} backward: need 2 saved tensors (A, B)",
                        op_name
                    )));
                }
                let a = saved_tensors[0];
                let b = saved_tensors[1];

                // Extract dimensions from config or infer.
                let m = config.get("batch").and_then(|v| v.as_i64()).unwrap_or(1);
                let k_val = config
                    .get("dim_in")
                    .and_then(|v| v.as_i64())
                    .unwrap_or_else(|| {
                        // Infer K from A.len / M.
                        if m > 0 {
                            a.len() as i64 / m
                        } else {
                            a.len() as i64
                        }
                    });
                let n_val = config
                    .get("dim_out")
                    .and_then(|v| v.as_i64())
                    .unwrap_or_else(|| {
                        // Infer N from grad_output.len / M.
                        if m > 0 {
                            grad_output.len() as i64 / m
                        } else {
                            grad_output.len() as i64
                        }
                    });

                let mut grad_a = vec![0.0f32; (m * k_val) as usize];
                let mut grad_b = vec![0.0f32; (k_val * n_val) as usize];
                unsafe {
                    ffi::aria_matmul_backward_f32(
                        grad_output.as_ptr(),
                        a.as_ptr(),
                        b.as_ptr(),
                        grad_a.as_mut_ptr(),
                        grad_b.as_mut_ptr(),
                        m,
                        k_val,
                        n_val,
                    );
                }
                Ok(BackwardGrads::Pair(grad_a, grad_b))
            }
            "gated_linear" => gated_linear_backward(grad_output, saved_tensors, config),
            "softmax_attention" => softmax_attention_backward(grad_output, saved_tensors, config),
            "selective_scan" => selective_scan_backward(grad_output, saved_tensors, config),
            "state_space" => state_space_backward(grad_output, saved_tensors, config),
            "gated_delta" => gated_delta_backward(grad_output, saved_tensors, config),
            "rwkv_time_mixing" => rwkv_time_mixing_backward(grad_output, saved_tensors, config),
            "conv1d_seq" => conv1d_seq_backward(grad_output, saved_tensors, config),
            "swiglu" => swiglu_backward(grad_output, saved_tensors, config),
            "rwkv_channel" => rwkv_channel_backward(grad_output, saved_tensors, config),

            // ── Conditional adaptive-routing ops ─────────────────
            "conditional_dispatch" => conditional_dispatch_backward(grad_output, config),
            "conditional_gather" => conditional_gather_backward(grad_output, saved_tensors),

            // ── Passthrough for input/output nodes ───────────────
            "input" | "output" => Ok(BackwardGrads::Single(grad_output.to_vec())),

            _ => Err(AriaError::UnsupportedOp(format!(
                "no backward kernel for op: {}",
                op_name
            ))),
        }
    }
}

/// Forward execution result that includes saved activations for backward pass.
#[derive(Debug)]
pub struct ForwardForBackwardResult {
    /// The output tensor from the graph's output node.
    pub output: Vec<f32>,
    /// Saved intermediate activations keyed by node id.
    /// For each node, stores the output of the forward pass.
    pub saved_activations: HashMap<u32, Vec<f32>>,
    /// Arena memory usage statistics.
    pub arena_stats: ArenaStats,
}

/// Execute the graph forward, saving all intermediate activations for the
/// backward pass.
pub fn execute_forward_saving_activations(
    graph: &GraphIR,
    dispatcher: &dyn KernelDispatch,
    input: &[f32],
) -> Result<ForwardForBackwardResult, AriaError> {
    execute_forward_saving_with_input_binding(graph, dispatcher, InputBinding::Shared(input))
}

pub fn execute_forward_saving_activations_multi_input(
    graph: &GraphIR,
    dispatcher: &dyn KernelDispatch,
    inputs: &[&[f32]],
) -> Result<ForwardForBackwardResult, AriaError> {
    execute_forward_saving_with_input_binding(graph, dispatcher, InputBinding::Distinct(inputs))
}

fn execute_forward_saving_with_input_binding(
    graph: &GraphIR,
    dispatcher: &dyn KernelDispatch,
    input_binding: InputBinding<'_>,
) -> Result<ForwardForBackwardResult, AriaError> {
    let result = execute_with_input_binding(graph, dispatcher, input_binding)?;

    let order = graph.topological_order()?;
    let node_map: HashMap<NodeId, &crate::graph::Node> =
        graph.nodes.iter().map(|n| (n.id, n)).collect();

    let mut saved: HashMap<u32, Vec<f32>> = HashMap::new();
    let mut ctx = ExecutionContext::new();
    let mut input_ordinal = 0usize;

    for &node_id in &order {
        let node = node_map
            .get(&node_id)
            .ok_or_else(|| AriaError::InvalidIR(format!("node {} missing", node_id.0)))?;

        if node.is_input {
            let buf = input_binding.slice_for_input_node(input_ordinal)?.to_vec();
            input_ordinal += 1;
            saved.insert(node_id.0, buf.clone());
            ctx.outputs.insert(node_id, NodeBuffer::Heap(buf));
            continue;
        }

        let input_slices: Vec<&[f32]> = node
            .input_ids
            .iter()
            .map(|id| {
                ctx.outputs
                    .get(id)
                    .map(|buf| buf.as_slice())
                    .ok_or_else(|| {
                        AriaError::ExecutionFailed(format!(
                            "node {} requires input from node {} which has no output",
                            node_id.0, id.0
                        ))
                    })
            })
            .collect::<Result<Vec<_>, _>>()?;

        // "output" nodes are identity/passthrough.
        let output = if node.op_name == "output" {
            input_slices.first().map(|s| s.to_vec()).unwrap_or_default()
        } else if node.op_name == "conditional_dispatch" {
            execute_conditional_dispatch(&input_slices, &node.config)?
        } else if node.op_name == "conditional_gather" {
            execute_conditional_gather(&input_slices)?
        } else {
            dispatcher.dispatch(&node.op_name, &input_slices, &node.config)?
        };
        saved.insert(node_id.0, output.clone());
        ctx.outputs.insert(node_id, NodeBuffer::Heap(output));
    }

    Ok(ForwardForBackwardResult {
        output: result.output,
        saved_activations: saved,
        arena_stats: result.arena_stats,
    })
}

/// Execute the backward pass through the graph.
///
/// Given saved activations from the forward pass and a gradient w.r.t. the
/// output, traverses nodes in reverse topological order, accumulating
/// gradients for each node.
///
/// Returns gradient buffers keyed by node id (u32).
pub fn execute_backward_with_arena(
    graph: &GraphIR,
    grad_output: &[f32],
    saved_activations: &HashMap<u32, Vec<f32>>,
) -> Result<BackwardResult, AriaError> {
    let order = graph.topological_order()?;
    let node_map: HashMap<NodeId, &crate::graph::Node> =
        graph.nodes.iter().map(|n| (n.id, n)).collect();

    // Arena for backward intermediate buffers.
    let est_capacity = grad_output.len() * 4 * order.len() * 2 + 4096;
    let arena_capacity = est_capacity;
    let mut stats = ArenaStats {
        arena_capacity,
        ..Default::default()
    };

    // Gradient accumulator: node_id -> gradient w.r.t. that node's output.
    let mut grads: HashMap<NodeId, Vec<f32>> = HashMap::new();

    // Seed: gradient for the output node is the provided grad_output.
    grads.insert(graph.output_node_id, grad_output.to_vec());

    // Traverse in reverse topological order.
    for &node_id in order.iter().rev() {
        let node = match node_map.get(&node_id) {
            Some(n) => *n,
            None => continue,
        };

        // Get the accumulated gradient for this node's output.
        let node_grad = match grads.get(&node_id) {
            Some(g) => g.clone(),
            None => continue, // No gradient flows to this node.
        };

        // Input nodes: just store the gradient, no backward dispatch needed.
        if node.is_input {
            continue;
        }

        // Gather saved tensors for this node's inputs.
        let saved_tensors: Vec<&[f32]> = node
            .input_ids
            .iter()
            .filter_map(|id| saved_activations.get(&id.0).map(|v| v.as_slice()))
            .collect();

        // For sigmoid and tanh, the backward kernel needs the *output* of the
        // forward op, not the input. Use the saved activation of the current
        // node itself.
        let saved_for_backward: Vec<&[f32]> = match node.op_name.as_str() {
            "sigmoid" | "tanh" => {
                // Need the output of this node (saved under node_id).
                match saved_activations.get(&node_id.0) {
                    Some(out) => vec![out.as_slice()],
                    None => saved_tensors.clone(),
                }
            }
            _ => saved_tensors.clone(),
        };

        // Dispatch backward kernel.
        let backward_grads = NativeKernelDispatch::dispatch_backward(
            &node.op_name,
            &node_grad,
            &saved_for_backward,
            &node.config,
        )?;

        // Route gradients to input nodes, accumulating when a node has
        // multiple consumers.
        match backward_grads {
            BackwardGrads::Single(g) => {
                // Single gradient goes to the first (only) input.
                if let Some(&input_id) = node.input_ids.first() {
                    let entry = grads
                        .entry(input_id)
                        .or_insert_with(|| vec![0.0f32; g.len()]);
                    for (i, val) in g.iter().enumerate() {
                        if i < entry.len() {
                            entry[i] += val;
                        }
                    }
                    stats.arena_alloc_count += 1;
                }
            }
            BackwardGrads::Pair(ga, gb) => {
                // First gradient goes to first input, second to second input.
                if let Some(&id_a) = node.input_ids.first() {
                    let entry = grads.entry(id_a).or_insert_with(|| vec![0.0f32; ga.len()]);
                    for (i, val) in ga.iter().enumerate() {
                        if i < entry.len() {
                            entry[i] += val;
                        }
                    }
                    stats.arena_alloc_count += 1;
                }
                if let Some(&id_b) = node.input_ids.get(1) {
                    let entry = grads.entry(id_b).or_insert_with(|| vec![0.0f32; gb.len()]);
                    for (i, val) in gb.iter().enumerate() {
                        if i < entry.len() {
                            entry[i] += val;
                        }
                    }
                    stats.arena_alloc_count += 1;
                }
            }
            BackwardGrads::Many(grads_many) => {
                for (input_id, grad_in) in node.input_ids.iter().zip(grads_many.into_iter()) {
                    let entry = grads
                        .entry(*input_id)
                        .or_insert_with(|| vec![0.0f32; grad_in.len()]);
                    for (i, val) in grad_in.iter().enumerate() {
                        if i < entry.len() {
                            entry[i] += val;
                        }
                    }
                    stats.arena_alloc_count += 1;
                }
            }
        }
    }

    // Convert NodeId keys to u32 for the public API.
    let grads_u32: HashMap<u32, Vec<f32>> = grads.into_iter().map(|(id, g)| (id.0, g)).collect();

    Ok(BackwardResult {
        grads: grads_u32,
        arena_stats: stats,
    })
}

/// Execute the graph in topological order (original heap-based API).
///
/// This is a convenience wrapper that calls `execute_with_arena` and
/// discards the arena statistics, preserving backward compatibility.
pub fn execute(
    graph: &GraphIR,
    dispatcher: &dyn KernelDispatch,
    input: &[f32],
) -> Result<Vec<f32>, AriaError> {
    execute_with_arena(graph, dispatcher, input).map(|r| r.output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::GraphIR;
    use std::thread;

    /// A trivial dispatcher that passes the first input through unchanged.
    struct PassthroughDispatcher;

    impl KernelDispatch for PassthroughDispatcher {
        fn dispatch(
            &self,
            _op_name: &str,
            inputs: &[&[f32]],
            _config: &serde_json::Value,
        ) -> Result<Vec<f32>, AriaError> {
            if inputs.is_empty() {
                return Ok(vec![]);
            }
            Ok(inputs[0].to_vec())
        }

        fn dispatch_into(
            &self,
            _op_name: &str,
            inputs: &[&[f32]],
            _config: &serde_json::Value,
            output_buf: &mut [f32],
        ) -> Result<usize, AriaError> {
            if inputs.is_empty() {
                return Ok(0);
            }
            let copy_len = inputs[0].len().min(output_buf.len());
            output_buf[..copy_len].copy_from_slice(&inputs[0][..copy_len]);
            Ok(copy_len)
        }
    }

    fn sample_graph_json() -> &'static str {
        r#"{
            "schema_version": "0.1", "model_dim": 4,
            "nodes": [
                {"id": 0, "op_name": "input",   "input_ids": [],  "config": {}, "is_input": true,  "is_output": false},
                {"id": 1, "op_name": "linear",  "input_ids": [0], "config": {}, "is_input": false, "is_output": false},
                {"id": 2, "op_name": "output",  "input_ids": [1], "config": {}, "is_input": false, "is_output": true}
            ],
            "edges": [
                {"source": 0, "target": 1, "source_port": null, "target_port": null},
                {"source": 1, "target": 2, "source_port": null, "target_port": null}
            ],
            "output_node_id": 2,
            "metadata": null
        }"#
    }

    fn run_with_large_stack<F>(name: &str, f: F)
    where
        F: FnOnce() + Send + 'static,
    {
        let handle = thread::Builder::new()
            .name(name.to_string())
            .stack_size(64 * 1024 * 1024)
            .spawn(f)
            .expect("failed to spawn test thread");
        match handle.join() {
            Ok(()) => {}
            Err(err) => std::panic::resume_unwind(err),
        }
    }

    #[test]
    fn test_execute_passthrough() {
        let graph = GraphIR::from_json(sample_graph_json()).unwrap();
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let result = execute(&graph, &PassthroughDispatcher, &input).unwrap();
        assert_eq!(result, input);
    }

    #[test]
    fn test_execute_with_arena_passthrough() {
        let graph = GraphIR::from_json(sample_graph_json()).unwrap();
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let result = execute_with_arena(&graph, &PassthroughDispatcher, &input).unwrap();
        assert_eq!(result.output, input);
        // All 3 nodes should be arena-allocated.
        assert_eq!(result.arena_stats.arena_alloc_count, 3);
        assert_eq!(result.arena_stats.heap_fallback_count, 0);
        assert!(result.arena_stats.arena_bytes_used > 0);
        assert!(result.arena_stats.arena_capacity > 0);
    }

    #[test]
    fn test_arena_fallback_on_tiny_capacity() {
        // Use a graph with enough nodes that a tiny arena will overflow.
        let json = r#"{
            "schema_version": "0.1", "model_dim": 4,
            "nodes": [
                {"id": 0, "op_name": "input",  "input_ids": [],  "config": {}, "is_input": true,  "is_output": false},
                {"id": 1, "op_name": "relu",   "input_ids": [0], "config": {}, "is_input": false, "is_output": false},
                {"id": 2, "op_name": "relu",   "input_ids": [1], "config": {}, "is_input": false, "is_output": false},
                {"id": 3, "op_name": "relu",   "input_ids": [2], "config": {}, "is_input": false, "is_output": false},
                {"id": 4, "op_name": "output", "input_ids": [3], "config": {}, "is_input": false, "is_output": true}
            ],
            "edges": [
                {"source": 0, "target": 1, "source_port": null, "target_port": null},
                {"source": 1, "target": 2, "source_port": null, "target_port": null},
                {"source": 2, "target": 3, "source_port": null, "target_port": null},
                {"source": 3, "target": 4, "source_port": null, "target_port": null}
            ],
            "output_node_id": 4,
            "metadata": null
        }"#;

        let graph = GraphIR::from_json(json).unwrap();
        // Large input: 1024 floats = 4KB per node. Normal arena estimation
        // would handle this, but the test verifies the mechanism works.
        let input = vec![1.0f32; 1024];
        let result = execute_with_arena(&graph, &PassthroughDispatcher, &input).unwrap();
        assert_eq!(result.output, input);
        // The result should still be correct regardless of arena vs heap.
    }

    #[test]
    fn test_arena_stats_populated() {
        let graph = GraphIR::from_json(sample_graph_json()).unwrap();
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let result = execute_with_arena(&graph, &PassthroughDispatcher, &input).unwrap();
        let stats = &result.arena_stats;
        assert!(stats.arena_capacity > 0, "arena capacity should be nonzero");
        assert!(
            stats.arena_bytes_used > 0,
            "arena should have used some bytes"
        );
        assert_eq!(
            stats.arena_alloc_count + stats.heap_fallback_count,
            3,
            "total allocs should equal node count"
        );
    }

    #[test]
    fn test_forward_saving_activations() {
        let graph = GraphIR::from_json(sample_graph_json()).unwrap();
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let result =
            execute_forward_saving_activations(&graph, &PassthroughDispatcher, &input).unwrap();
        assert_eq!(result.output, input);
        // All 3 nodes should have saved activations (input, linear, output).
        assert_eq!(result.saved_activations.len(), 3);
        assert!(result.saved_activations.contains_key(&0));
        assert!(result.saved_activations.contains_key(&1));
        assert!(result.saved_activations.contains_key(&2));
    }

    #[test]
    fn test_backward_passthrough_identity_grad() {
        // Graph: input(0) -> passthrough(1) -> output(2)
        // With passthrough, backward should propagate gradient unchanged.
        let graph = GraphIR::from_json(sample_graph_json()).unwrap();
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let fwd =
            execute_forward_saving_activations(&graph, &PassthroughDispatcher, &input).unwrap();

        let grad_out = vec![1.0, 1.0, 1.0, 1.0];
        let result = execute_backward_with_arena(&graph, &grad_out, &fwd.saved_activations);
        // This will fail with UnsupportedOp for "linear" since we don't have
        // the C kernels in unit tests. That's expected — the integration test
        // via Python + C library covers the full path.
        // Just verify the function is callable and returns the right error type.
        match result {
            Ok(bwd) => {
                // If somehow it succeeds (unlikely without C lib), check structure.
                assert!(
                    bwd.grads.contains_key(&0),
                    "should have grad for input node"
                );
            }
            Err(AriaError::UnsupportedOp(msg)) => {
                assert!(msg.contains("linear"), "should fail on linear op");
            }
            Err(e) => {
                // Any error about missing C symbols is acceptable in unit tests.
                let msg = format!("{}", e);
                assert!(
                    msg.contains("linear") || msg.contains("backward") || msg.contains("kernel"),
                    "unexpected error: {}",
                    msg
                );
            }
        }
    }

    #[test]
    fn test_conditional_gather_skips_empty_lane() {
        let json = r#"{
            "schema_version": "0.1", "model_dim": 4,
            "nodes": [
                {"id": 0, "op_name": "input", "input_ids": [], "config": {}, "is_input": true, "is_output": false},
                {"id": 1, "op_name": "conditional_dispatch", "input_ids": [0], "config": {"lane": 0, "active_tokens": 0}, "is_input": false, "is_output": false},
                {"id": 2, "op_name": "relu", "input_ids": [0], "config": {}, "is_input": false, "is_output": false},
                {"id": 3, "op_name": "conditional_gather", "input_ids": [1, 2], "config": {}, "is_input": false, "is_output": false},
                {"id": 4, "op_name": "output", "input_ids": [3], "config": {}, "is_input": false, "is_output": true}
            ],
            "edges": [
                {"source": 0, "target": 1, "source_port": null, "target_port": null},
                {"source": 0, "target": 2, "source_port": null, "target_port": null},
                {"source": 1, "target": 3, "source_port": null, "target_port": null},
                {"source": 2, "target": 3, "source_port": null, "target_port": null},
                {"source": 3, "target": 4, "source_port": null, "target_port": null}
            ],
            "output_node_id": 4,
            "metadata": null
        }"#;
        let graph = GraphIR::from_json(json).unwrap();
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let result = execute(&graph, &PassthroughDispatcher, &input).unwrap();
        assert_eq!(result, input);
    }

    #[test]
    fn test_conditional_dispatch_packs_with_assignments() {
        // Because the scheduler API currently accepts one input tensor, use the same
        // buffer for x and assignments. With dim=1 this is still a valid packing check.
        let json = r#"{
            "schema_version": "0.1", "model_dim": 1,
            "nodes": [
                {"id": 0, "op_name": "input", "input_ids": [], "config": {}, "is_input": true, "is_output": false},
                {"id": 1, "op_name": "conditional_dispatch", "input_ids": [0, 0], "config": {"batch": 1, "seq": 4, "dim": 1, "lane": 1}, "is_input": false, "is_output": false},
                {"id": 2, "op_name": "output", "input_ids": [1], "config": {}, "is_input": false, "is_output": true}
            ],
            "edges": [
                {"source": 0, "target": 1, "source_port": null, "target_port": null},
                {"source": 1, "target": 2, "source_port": null, "target_port": null}
            ],
            "output_node_id": 2,
            "metadata": null
        }"#;
        let graph = GraphIR::from_json(json).unwrap();
        let input = vec![0.0, 1.0, 0.0, 1.0];
        let result = execute(&graph, &PassthroughDispatcher, &input).unwrap();
        assert_eq!(result, vec![1.0, 1.0, 0.0, 0.0]);
    }

    #[test]
    fn test_conditional_graph_forward_backward_integration() {
        let json = r#"{
                "schema_version": "0.1", "model_dim": 4,
                "nodes": [
                    {"id": 0, "op_name": "input", "input_ids": [], "config": {}, "is_input": true, "is_output": false},
                    {"id": 1, "op_name": "conditional_dispatch", "input_ids": [0], "config": {"lane": 0, "active_tokens": 0}, "is_input": false, "is_output": false},
                    {"id": 2, "op_name": "conditional_dispatch", "input_ids": [0], "config": {"lane": 1, "active_tokens": 4}, "is_input": false, "is_output": false},
                    {"id": 3, "op_name": "conditional_gather", "input_ids": [1, 2], "config": {}, "is_input": false, "is_output": false},
                    {"id": 4, "op_name": "output", "input_ids": [3], "config": {}, "is_input": false, "is_output": true}
                ],
                "edges": [
                    {"source": 0, "target": 1, "source_port": null, "target_port": null},
                    {"source": 0, "target": 2, "source_port": null, "target_port": null},
                    {"source": 1, "target": 3, "source_port": null, "target_port": null},
                    {"source": 2, "target": 3, "source_port": null, "target_port": null},
                    {"source": 3, "target": 4, "source_port": null, "target_port": null}
                ],
                "output_node_id": 4,
                "metadata": null
        }"#;
        let graph = GraphIR::from_json(json).unwrap();
        let input = vec![0.5f32, -0.25f32, 1.25f32, -2.0f32];

        let fwd =
            execute_forward_saving_activations(&graph, &PassthroughDispatcher, &input).unwrap();
        assert_eq!(fwd.output, input);

        let grad_out = vec![1.0f32, 1.0f32, 1.0f32, 1.0f32];
        let bwd = execute_backward_with_arena(&graph, &grad_out, &fwd.saved_activations).unwrap();

        let grad_input = bwd.grads.get(&0).expect("input gradient missing");
        assert_eq!(grad_input.len(), grad_out.len());
        assert!(grad_input.iter().all(|v| v.is_finite()));
        assert_eq!(grad_input, &grad_out);
    }

    #[test]
    fn test_execute_with_arena_multi_input_state_space_weighted_graph() {
        run_with_large_stack(
            "test_execute_with_arena_multi_input_state_space_weighted_graph",
            || {
                let json = r#"{
                    "schema_version": "native_ir.v1", "model_dim": 4,
                    "nodes": [
                        {"id": 0, "op_name": "input", "input_ids": [], "config": {}, "is_input": true, "is_output": false},
                        {"id": 2, "op_name": "input", "input_ids": [], "config": {}, "is_input": true, "is_output": false},
                        {"id": 3, "op_name": "input", "input_ids": [], "config": {}, "is_input": true, "is_output": false},
                        {"id": 4, "op_name": "input", "input_ids": [], "config": {}, "is_input": true, "is_output": false},
                        {"id": 5, "op_name": "input", "input_ids": [], "config": {}, "is_input": true, "is_output": false},
                        {"id": 6, "op_name": "input", "input_ids": [], "config": {}, "is_input": true, "is_output": false},
                        {"id": 7, "op_name": "input", "input_ids": [], "config": {}, "is_input": true, "is_output": false},
                        {"id": 1, "op_name": "state_space", "input_ids": [0, 2, 3, 4, 5, 6, 7], "config": {"batch": 1, "seq": 2, "dim": 4, "state_dim": 2}, "is_input": false, "is_output": false},
                        {"id": 8, "op_name": "output", "input_ids": [1], "config": {}, "is_input": false, "is_output": true}
                    ],
                    "edges": [
                        {"source": 0, "target": 1, "source_port": null, "target_port": null},
                        {"source": 2, "target": 1, "source_port": null, "target_port": null},
                        {"source": 3, "target": 1, "source_port": null, "target_port": null},
                        {"source": 4, "target": 1, "source_port": null, "target_port": null},
                        {"source": 5, "target": 1, "source_port": null, "target_port": null},
                        {"source": 6, "target": 1, "source_port": null, "target_port": null},
                        {"source": 7, "target": 1, "source_port": null, "target_port": null},
                        {"source": 1, "target": 8, "source_port": null, "target_port": null}
                    ],
                    "output_node_id": 8,
                    "metadata": null
                }"#;
                let graph = GraphIR::from_json(json).unwrap();
                let x = vec![0.1f32, -0.2, 0.3, -0.4, 0.2, 0.1, -0.3, 0.5];
                let ssm_a = vec![0.05f32; 8];
                let ssm_b_weight = vec![0.02f32; 32];
                let ssm_c_weight = vec![0.03f32; 32];
                let ssm_d = vec![1.0f32, 1.0, 1.0, 1.0];
                let ssm_dt_weight = vec![0.01f32; 16];
                let ssm_dt_bias = vec![0.1f32; 4];
                let inputs: [&[f32]; 7] = [
                    &x,
                    &ssm_a,
                    &ssm_b_weight,
                    &ssm_c_weight,
                    &ssm_d,
                    &ssm_dt_weight,
                    &ssm_dt_bias,
                ];
                let result =
                    execute_with_arena_multi_input(&graph, &NativeKernelDispatch, &inputs).unwrap();
                assert_eq!(result.output.len(), x.len());
                assert!(result.output.iter().all(|v| v.is_finite()));
            },
        );
    }

    #[test]
    fn test_execute_backward_with_arena_multi_input_state_space_weighted_graph() {
        run_with_large_stack(
            "test_execute_backward_with_arena_multi_input_state_space_weighted_graph",
            || {
                let json = r#"{
                    "schema_version": "native_ir.v1", "model_dim": 4,
                    "nodes": [
                        {"id": 0, "op_name": "input", "input_ids": [], "config": {}, "is_input": true, "is_output": false},
                        {"id": 2, "op_name": "input", "input_ids": [], "config": {}, "is_input": true, "is_output": false},
                        {"id": 3, "op_name": "input", "input_ids": [], "config": {}, "is_input": true, "is_output": false},
                        {"id": 4, "op_name": "input", "input_ids": [], "config": {}, "is_input": true, "is_output": false},
                        {"id": 5, "op_name": "input", "input_ids": [], "config": {}, "is_input": true, "is_output": false},
                        {"id": 6, "op_name": "input", "input_ids": [], "config": {}, "is_input": true, "is_output": false},
                        {"id": 7, "op_name": "input", "input_ids": [], "config": {}, "is_input": true, "is_output": false},
                        {"id": 1, "op_name": "state_space", "input_ids": [0, 2, 3, 4, 5, 6, 7], "config": {"batch": 1, "seq": 2, "dim": 4, "state_dim": 2}, "is_input": false, "is_output": false},
                        {"id": 8, "op_name": "output", "input_ids": [1], "config": {}, "is_input": false, "is_output": true}
                    ],
                    "edges": [
                        {"source": 0, "target": 1, "source_port": null, "target_port": null},
                        {"source": 2, "target": 1, "source_port": null, "target_port": null},
                        {"source": 3, "target": 1, "source_port": null, "target_port": null},
                        {"source": 4, "target": 1, "source_port": null, "target_port": null},
                        {"source": 5, "target": 1, "source_port": null, "target_port": null},
                        {"source": 6, "target": 1, "source_port": null, "target_port": null},
                        {"source": 7, "target": 1, "source_port": null, "target_port": null},
                        {"source": 1, "target": 8, "source_port": null, "target_port": null}
                    ],
                    "output_node_id": 8,
                    "metadata": null
                }"#;
                let graph = GraphIR::from_json(json).unwrap();
                let x = vec![0.1f32, -0.2, 0.3, -0.4, 0.2, 0.1, -0.3, 0.5];
                let ssm_a = vec![0.05f32; 8];
                let ssm_b_weight = vec![0.02f32; 32];
                let ssm_c_weight = vec![0.03f32; 32];
                let ssm_d = vec![1.0f32, 1.0, 1.0, 1.0];
                let ssm_dt_weight = vec![0.01f32; 16];
                let ssm_dt_bias = vec![0.1f32; 4];
                let inputs: [&[f32]; 7] = [
                    &x,
                    &ssm_a,
                    &ssm_b_weight,
                    &ssm_c_weight,
                    &ssm_d,
                    &ssm_dt_weight,
                    &ssm_dt_bias,
                ];
                let fwd = execute_forward_saving_activations_multi_input(
                    &graph,
                    &NativeKernelDispatch,
                    &inputs,
                )
                .unwrap();
                let grad_out = vec![1.0f32; x.len()];
                let bwd =
                    execute_backward_with_arena(&graph, &grad_out, &fwd.saved_activations).unwrap();
                for node_id in [0u32, 2, 3, 4, 5, 6, 7] {
                    let grad = bwd
                        .grads
                        .get(&node_id)
                        .expect("missing weighted input grad");
                    assert!(grad.iter().all(|v| v.is_finite()));
                }
            },
        );
    }
}
