use std::os::raw::c_char;

#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum NkStatus {
    Ok = 0,
    ErrUnsupported = -1,
    ErrInvalidArgument = -2,
    ErrInternal = -3,
}

pub type NkUnaryF32Fn = Option<unsafe extern "C" fn(x: *const f32, y: *mut f32, n: i64) -> NkStatus>;
pub type NkBinaryF32Fn = Option<unsafe extern "C" fn(a: *const f32, b: *const f32, y: *mut f32, n: i64) -> NkStatus>;
pub type NkMatmulF32Fn = Option<unsafe extern "C" fn(a: *const f32, b: *const f32, c: *mut f32, m: i64, k: i64, n: i64) -> NkStatus>;
pub type NkLinearF32Fn = Option<unsafe extern "C" fn(x: *const f32, w: *const f32, bias: *const f32, y: *mut f32, batch: i64, dim_in: i64, dim_out: i64) -> NkStatus>;
pub type NkSoftmaxF32Fn = Option<unsafe extern "C" fn(x: *const f32, y: *mut f32, batch: i64, dim: i64) -> NkStatus>;
pub type NkRmsnormF32Fn = Option<unsafe extern "C" fn(x: *const f32, weight: *const f32, y: *mut f32, batch: i64, dim: i64, eps: f32) -> NkStatus>;
pub type NkConcatF32Fn = Option<unsafe extern "C" fn(inputs: *const *const f32, sizes: *const i64, n_inputs: i32, output: *mut f32, dim: i64) -> NkStatus>;
pub type NkSplitF32Fn = Option<unsafe extern "C" fn(input: *const f32, outputs: *mut *mut f32, sizes: *const i64, n_outputs: i32, dim: i64) -> NkStatus>;

pub type NkUnaryF16Fn = Option<unsafe extern "C" fn(x: *const u16, y: *mut u16, n: i64) -> NkStatus>;
pub type NkBinaryF16Fn = Option<unsafe extern "C" fn(a: *const u16, b: *const u16, y: *mut u16, n: i64) -> NkStatus>;
pub type NkMatmulF16Fn = Option<unsafe extern "C" fn(a: *const u16, b: *const u16, c: *mut u16, m: i64, k: i64, n: i64) -> NkStatus>;
pub type NkSoftmaxF16Fn = Option<unsafe extern "C" fn(x: *const u16, y: *mut u16, batch: i64, dim: i64) -> NkStatus>;
pub type NkRmsnormF16Fn = Option<unsafe extern "C" fn(x: *const u16, weight: *const u16, y: *mut u16, batch: i64, dim: i64, eps: f32) -> NkStatus>;

pub type NkMatmulReluF32Fn = Option<unsafe extern "C" fn(a: *const f32, b: *const f32, c: *mut f32, m: i64, k: i64, n: i64) -> NkStatus>;
pub type NkMatmulBiasReluF32Fn = Option<unsafe extern "C" fn(a: *const f32, b: *const f32, bias: *const f32, c: *mut f32, m: i64, k: i64, n: i64) -> NkStatus>;
pub type NkLayernormResidualF32Fn = Option<unsafe extern "C" fn(x: *const f32, residual: *const f32, gamma: *const f32, beta: *const f32, y: *mut f32, rows: i64, cols: i64, eps: f32) -> NkStatus>;
pub type NkMatmulGeluF32Fn = Option<unsafe extern "C" fn(a: *const f32, b: *const f32, c: *mut f32, m: i64, k: i64, n: i64) -> NkStatus>;

pub type NkSwigluF32Fn = Option<unsafe extern "C" fn(x: *const f32, w_gate: *const f32, w_up: *const f32, w_down: *const f32, b_gate: *const f32, b_up: *const f32, b_down: *const f32, y: *mut f32, tmp_gate: *mut f32, tmp_up: *mut f32, batch: i64, dim: i64, hidden_dim: i64) -> NkStatus>;
pub type NkRwkvChannelF32Fn = Option<unsafe extern "C" fn(x: *const f32, mix_k: *const f32, mix_r: *const f32, w_k: *const f32, w_r: *const f32, w_v: *const f32, y: *mut f32, tmp_xk: *mut f32, tmp_xr: *mut f32, tmp_k: *mut f32, batch: i64, seq: i64, dim: i64, hidden_dim: i64) -> NkStatus>;

pub type NkEmbeddingLookupF32Fn = Option<unsafe extern "C" fn(table: *const f32, indices: *const i32, pos_embed: *const f32, y: *mut f32, batch: i64, dim: i64, vocab_size: i64) -> NkStatus>;
pub type NkRopeRotateF32Fn = Option<unsafe extern "C" fn(x: *const f32, y: *mut f32, batch: i64, seq: i64, dim: i64, theta_base: f32) -> NkStatus>;
pub type NkGatedLinearF32Fn = Option<unsafe extern "C" fn(x: *const f32, w: *const f32, b: *const f32, w_gate: *const f32, b_gate: *const f32, y: *mut f32, tmp_gate: *mut f32, batch: i64, dim_in: i64, dim_out: i64) -> NkStatus>;
pub type NkCosineSimilarityF32Fn = Option<unsafe extern "C" fn(a: *const f32, b: *const f32, out: *mut f32, batch: i64, seq: i64, dim: i64) -> NkStatus>;
pub type NkGatherTopkF32Fn = Option<unsafe extern "C" fn(scores: *const f32, values: *const f32, out: *mut f32, out_indices: *mut i32, batch: i64, n_items: i64, dim: i64, k: i64) -> NkStatus>;
pub type NkRwkvTimeMixingF32Fn = Option<unsafe extern "C" fn(x: *const f32, w_decay: *const f32, u_bonus: *const f32, w_k: *const f32, w_v: *const f32, w_r: *const f32, y: *mut f32, batch: i64, seq: i64, dim: i64) -> NkStatus>;

#[repr(C)]
pub struct NkRegistration {
    pub op_name: *const c_char,
    pub unary_fn: NkUnaryF32Fn,
    pub binary_fn: NkBinaryF32Fn,
    pub matmul_fn: NkMatmulF32Fn,
    pub linear_fn: NkLinearF32Fn,
    pub softmax_fn: NkSoftmaxF32Fn,
    pub rmsnorm_fn: NkRmsnormF32Fn,
    pub concat_fn: NkConcatF32Fn,
    pub split_fn: NkSplitF32Fn,
    /* FP16 */
    pub unary_f16_fn: NkUnaryF16Fn,
    pub binary_f16_fn: NkBinaryF16Fn,
    pub matmul_f16_fn: NkMatmulF16Fn,
    pub softmax_f16_fn: NkSoftmaxF16Fn,
    pub rmsnorm_f16_fn: NkRmsnormF16Fn,
    /* Fused */
    pub matmul_relu_fn: NkMatmulReluF32Fn,
    pub matmul_bias_relu_fn: NkMatmulBiasReluF32Fn,
    pub layernorm_residual_fn: NkLayernormResidualF32Fn,
    pub matmul_gelu_fn: NkMatmulGeluF32Fn,
    pub swiglu_fn: NkSwigluF32Fn,
    pub rwkv_channel_fn: NkRwkvChannelF32Fn,
    /* Reference architecture ops */
    pub embedding_lookup_fn: NkEmbeddingLookupF32Fn,
    pub rope_rotate_fn: NkRopeRotateF32Fn,
    pub gated_linear_fn: NkGatedLinearF32Fn,
    pub cosine_similarity_fn: NkCosineSimilarityF32Fn,
    pub gather_topk_fn: NkGatherTopkF32Fn,
    pub rwkv_time_mixing_fn: NkRwkvTimeMixingF32Fn,
    /* Backward */
    pub unary_backward_fn: *const std::ffi::c_void,
    pub binary_backward_simple_fn: *const std::ffi::c_void,
    pub binary_backward_fn: *const std::ffi::c_void,
    pub matmul_backward_fn: *const std::ffi::c_void,
}

extern "C" {
    pub fn aria_registry_init();
    pub fn nk_is_registered(op_name: *const c_char) -> i32;
    pub fn nk_dispatch(op_name: *const c_char) -> *const NkRegistration;

    // ── Backward (gradient) kernels ──────────────────────────────────
    // Unary backward kernels: grad_out, input_or_output, grad_in, n
    pub fn aria_relu_backward_f32(grad_out: *const f32, input: *const f32, grad_in: *mut f32, n: i64);
    pub fn aria_sigmoid_backward_f32(grad_out: *const f32, output: *const f32, grad_in: *mut f32, n: i64);
    pub fn aria_tanh_backward_f32(grad_out: *const f32, output: *const f32, grad_in: *mut f32, n: i64);
    pub fn aria_gelu_backward_f32(grad_out: *const f32, input: *const f32, grad_in: *mut f32, n: i64);
    pub fn aria_silu_backward_f32(grad_out: *const f32, input: *const f32, grad_in: *mut f32, n: i64);

    // Binary backward kernels
    pub fn aria_add_backward_f32(grad_out: *const f32, grad_a: *mut f32, grad_b: *mut f32, n: i64);
    pub fn aria_mul_backward_f32(grad_out: *const f32, a: *const f32, b: *const f32, grad_a: *mut f32, grad_b: *mut f32, n: i64);
    pub fn aria_sub_backward_f32(grad_out: *const f32, grad_a: *mut f32, grad_b: *mut f32, n: i64);

    // Matmul backward: C = A[M,K] @ B[K,N]
    pub fn aria_matmul_backward_f32(grad_out: *const f32, a: *const f32, b: *const f32, grad_a: *mut f32, grad_b: *mut f32, m: i64, k: i64, n: i64);

    // Profiler ABI (profile_abi.h)
    pub fn np_profiler_enable(enable: i32);
    pub fn np_profiler_enabled() -> i32;
    pub fn np_clock_ns() -> i64;
    pub fn np_reset_counters();
    pub fn np_event_count() -> i32;
    pub fn np_drain_events(out: *mut NpEvent, max_out: i32) -> i32;
    pub fn np_emit_event(evt: *const NpEvent);
    pub fn np_get_peak_memory() -> i64;
    pub fn np_memory_event_count() -> i32;
    pub fn np_drain_memory_events(out: *mut NpMemoryEvent, max_out: i32) -> i32;
}

/// Mirrors np_event_t from profile_abi.h.
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct NpEvent {
    pub event_name: *const c_char,
    pub op_name: *const c_char,
    pub node_id: i32,
    pub start_ns: i64,
    pub end_ns: i64,
    pub thread_id: i32,
}

/// Mirrors np_memory_event_t from profile_abi.h.
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct NpMemoryEvent {
    pub tag: *const c_char,
    pub bytes_allocated: i64,
    pub bytes_freed: i64,
    pub peak_bytes: i64,
    pub timestamp_ns: i64,
}
