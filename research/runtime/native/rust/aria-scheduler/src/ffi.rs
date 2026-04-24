use std::os::raw::c_char;

#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum NkStatus {
    Ok = 0,
    ErrUnsupported = -1,
    ErrInvalidArgument = -2,
    ErrInternal = -3,
}

#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum NrStatus {
    Ok = 0,
    ErrInvalidArgument = -1,
    ErrUnsupportedIr = -2,
    ErrCompileFailure = -3,
    ErrExecutionFailure = -4,
    ErrInternal = -5,
    ErrStrictUnsupported = -6,
}

#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum NrOptimizer {
    Sgd = 1,
    Adamw = 2,
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct NrTrainTensorF32 {
    pub param: *mut f32,
    pub grad: *const f32,
    pub momentum: *mut f32,
    pub exp_avg: *mut f32,
    pub exp_avg_sq: *mut f32,
    pub numel: i64,
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct NrOptimizerStepRequest {
    pub optimizer: NrOptimizer,
    pub tensors: *mut NrTrainTensorF32,
    pub n_tensors: i32,
    pub learning_rate: f64,
    pub momentum: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub eps: f64,
    pub weight_decay: f64,
    pub max_grad_norm: f64,
    pub nesterov: i32,
    pub step: i64,
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct NrOptimizerStepResponse {
    pub status: NrStatus,
    pub grad_norm: f64,
    pub elements: i64,
    pub message: *const c_char,
}

pub type NkUnaryF32Fn =
    Option<unsafe extern "C" fn(x: *const f32, y: *mut f32, n: i64) -> NkStatus>;
pub type NkBinaryF32Fn =
    Option<unsafe extern "C" fn(a: *const f32, b: *const f32, y: *mut f32, n: i64) -> NkStatus>;
pub type NkMatmulF32Fn = Option<
    unsafe extern "C" fn(
        a: *const f32,
        b: *const f32,
        c: *mut f32,
        m: i64,
        k: i64,
        n: i64,
    ) -> NkStatus,
>;
pub type NkLinearF32Fn = Option<
    unsafe extern "C" fn(
        x: *const f32,
        w: *const f32,
        bias: *const f32,
        y: *mut f32,
        batch: i64,
        dim_in: i64,
        dim_out: i64,
    ) -> NkStatus,
>;
pub type NkSoftmaxF32Fn =
    Option<unsafe extern "C" fn(x: *const f32, y: *mut f32, batch: i64, dim: i64) -> NkStatus>;
pub type NkRmsnormF32Fn = Option<
    unsafe extern "C" fn(
        x: *const f32,
        weight: *const f32,
        y: *mut f32,
        batch: i64,
        dim: i64,
        eps: f32,
    ) -> NkStatus,
>;
pub type NkConcatF32Fn = Option<
    unsafe extern "C" fn(
        inputs: *const *const f32,
        sizes: *const i64,
        n_inputs: i32,
        output: *mut f32,
        dim: i64,
    ) -> NkStatus,
>;
pub type NkSplitF32Fn = Option<
    unsafe extern "C" fn(
        input: *const f32,
        outputs: *mut *mut f32,
        sizes: *const i64,
        n_outputs: i32,
        dim: i64,
    ) -> NkStatus,
>;

pub type NkUnaryF16Fn =
    Option<unsafe extern "C" fn(x: *const u16, y: *mut u16, n: i64) -> NkStatus>;
pub type NkBinaryF16Fn =
    Option<unsafe extern "C" fn(a: *const u16, b: *const u16, y: *mut u16, n: i64) -> NkStatus>;
pub type NkMatmulF16Fn = Option<
    unsafe extern "C" fn(
        a: *const u16,
        b: *const u16,
        c: *mut u16,
        m: i64,
        k: i64,
        n: i64,
    ) -> NkStatus,
>;
pub type NkSoftmaxF16Fn =
    Option<unsafe extern "C" fn(x: *const u16, y: *mut u16, batch: i64, dim: i64) -> NkStatus>;
pub type NkRmsnormF16Fn = Option<
    unsafe extern "C" fn(
        x: *const u16,
        weight: *const u16,
        y: *mut u16,
        batch: i64,
        dim: i64,
        eps: f32,
    ) -> NkStatus,
>;

pub type NkMatmulReluF32Fn = Option<
    unsafe extern "C" fn(
        a: *const f32,
        b: *const f32,
        c: *mut f32,
        m: i64,
        k: i64,
        n: i64,
    ) -> NkStatus,
>;
pub type NkMatmulBiasReluF32Fn = Option<
    unsafe extern "C" fn(
        a: *const f32,
        b: *const f32,
        bias: *const f32,
        c: *mut f32,
        m: i64,
        k: i64,
        n: i64,
    ) -> NkStatus,
>;
pub type NkLayernormResidualF32Fn = Option<
    unsafe extern "C" fn(
        x: *const f32,
        residual: *const f32,
        gamma: *const f32,
        beta: *const f32,
        y: *mut f32,
        rows: i64,
        cols: i64,
        eps: f32,
    ) -> NkStatus,
>;
pub type NkMatmulGeluF32Fn = Option<
    unsafe extern "C" fn(
        a: *const f32,
        b: *const f32,
        c: *mut f32,
        m: i64,
        k: i64,
        n: i64,
    ) -> NkStatus,
>;

pub type NkSwigluF32Fn = Option<
    unsafe extern "C" fn(
        x: *const f32,
        w_gate: *const f32,
        w_up: *const f32,
        w_down: *const f32,
        b_gate: *const f32,
        b_up: *const f32,
        b_down: *const f32,
        y: *mut f32,
        tmp_gate: *mut f32,
        tmp_up: *mut f32,
        batch: i64,
        dim: i64,
        hidden_dim: i64,
    ) -> NkStatus,
>;
pub type NkRwkvChannelF32Fn = Option<
    unsafe extern "C" fn(
        x: *const f32,
        mix_k: *const f32,
        mix_r: *const f32,
        w_k: *const f32,
        w_r: *const f32,
        w_v: *const f32,
        y: *mut f32,
        tmp_xk: *mut f32,
        tmp_xr: *mut f32,
        tmp_k: *mut f32,
        batch: i64,
        seq: i64,
        dim: i64,
        hidden_dim: i64,
    ) -> NkStatus,
>;
pub type NkConv1dSeqF32Fn = Option<
    unsafe extern "C" fn(
        x: *const f32,
        weight: *const f32,
        bias: *const f32,
        y: *mut f32,
        batch: i64,
        seq: i64,
        dim: i64,
    ) -> NkStatus,
>;

pub type NkEmbeddingLookupF32Fn = Option<
    unsafe extern "C" fn(
        table: *const f32,
        indices: *const i32,
        pos_embed: *const f32,
        y: *mut f32,
        batch: i64,
        dim: i64,
        vocab_size: i64,
    ) -> NkStatus,
>;
pub type NkRopeRotateF32Fn = Option<
    unsafe extern "C" fn(
        x: *const f32,
        y: *mut f32,
        batch: i64,
        seq: i64,
        dim: i64,
        theta_base: f32,
    ) -> NkStatus,
>;
pub type NkGatedLinearF32Fn = Option<
    unsafe extern "C" fn(
        x: *const f32,
        w: *const f32,
        b: *const f32,
        w_gate: *const f32,
        b_gate: *const f32,
        y: *mut f32,
        tmp_gate: *mut f32,
        batch: i64,
        dim_in: i64,
        dim_out: i64,
    ) -> NkStatus,
>;
pub type NkCosineSimilarityF32Fn = Option<
    unsafe extern "C" fn(
        a: *const f32,
        b: *const f32,
        out: *mut f32,
        batch: i64,
        seq: i64,
        dim: i64,
    ) -> NkStatus,
>;
pub type NkGatherTopkF32Fn = Option<
    unsafe extern "C" fn(
        scores: *const f32,
        values: *const f32,
        out: *mut f32,
        out_indices: *mut i32,
        batch: i64,
        n_items: i64,
        dim: i64,
        k: i64,
    ) -> NkStatus,
>;
pub type NkRwkvTimeMixingF32Fn = Option<
    unsafe extern "C" fn(
        x: *const f32,
        w_decay: *const f32,
        u_bonus: *const f32,
        w_k: *const f32,
        w_v: *const f32,
        w_r: *const f32,
        y: *mut f32,
        batch: i64,
        seq: i64,
        dim: i64,
    ) -> NkStatus,
>;

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
    pub conv1d_seq_fn: NkConv1dSeqF32Fn,
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
    pub fn nr_optimizer_clip_step_f32(
        req: *const NrOptimizerStepRequest,
    ) -> NrOptimizerStepResponse;

    pub fn aria_registry_init();
    pub fn nk_is_registered(op_name: *const c_char) -> i32;
    pub fn nk_dispatch(op_name: *const c_char) -> *const NkRegistration;
    // ── Tier 3 / Research Math Spaces ────────────────────────────────
    pub fn aria_exp_map_f32(x: *const f32, y: *mut f32, batch: i64, dim: i64, c: f32);
    pub fn aria_log_map_f32(x: *const f32, y: *mut f32, batch: i64, dim: i64, c: f32);
    pub fn aria_poincare_add_f32(
        x: *const f32,
        v: *const f32,
        y: *mut f32,
        batch: i64,
        dim: i64,
        c: f32,
    );
    pub fn aria_hyp_linear_f32(
        x: *const f32,
        w: *const f32,
        y: *mut f32,
        batch: i64,
        dim_in: i64,
        dim_out: i64,
        c: f32,
    );
    pub fn aria_hyperbolic_norm_f32(
        x: *const f32,
        gamma: *const f32,
        beta: *const f32,
        y: *mut f32,
        batch: i64,
        dim: i64,
        c: f32,
        eps: f32,
    );
    pub fn aria_hyp_tangent_nonlinear_f32(x: *const f32, y: *mut f32, n: i64, c: f32);

    pub fn aria_tropical_attention_f32(
        x: *const f32,
        y: *mut f32,
        batch: i64,
        seq: i64,
        dim: i64,
        temperature: f32,
    );
    pub fn aria_tropical_gate_f32(
        x: *const f32,
        y: *mut f32,
        batch: i64,
        seq: i64,
        dim: i64,
        temperature: f32,
    );
    pub fn aria_tropical_add_f32(a: *const f32, b: *const f32, y: *mut f32, n: i64);
    pub fn aria_tropical_matmul_f32(
        a: *const f32,
        b: *const f32,
        c: *mut f32,
        m: i64,
        k: i64,
        n: i64,
    );

    pub fn aria_rotor_transform_f32(
        x: *const f32,
        rotor: *const f32,
        y: *mut f32,
        batch: i64,
        dim: i64,
    );
    pub fn aria_grade_select_f32(x: *const f32, y: *mut f32, batch: i64, dim: i64, grade: i32);
    pub fn aria_grade_mix_f32(x: *const f32, alpha: *const f32, y: *mut f32, batch: i64, dim: i64);
    pub fn aria_clifford_attention_f32(x: *const f32, y: *mut f32, batch: i64, seq: i64, dim: i64);
    pub fn aria_clifford_geometric_product_cl30_f32(
        a: *const f32,
        b: *const f32,
        y: *mut f32,
        n_multivectors: i64,
    );
    pub fn aria_softmax_attention_f32(
        x: *const f32,
        wq: *const f32,
        wk: *const f32,
        wv: *const f32,
        wo: *const f32,
        y: *mut f32,
        batch: i64,
        seq: i64,
        dim: i64,
        n_heads: i64,
    );
    pub fn aria_linear_attention_f32(
        x: *const f32,
        wq: *const f32,
        wk: *const f32,
        wv: *const f32,
        wo: *const f32,
        y: *mut f32,
        batch: i64,
        seq: i64,
        dim: i64,
    );
    pub fn aria_depth_weighted_proj_f32(
        x: *const f32,
        depth_scorer: *const f32,
        step_projs: *const f32,
        y: *mut f32,
        batch: i64,
        seq: i64,
        dim: i64,
        max_depth: i64,
    );
    pub fn aria_selective_scan_compiled_f32(
        x: *const f32,
        a_log: *const f32,
        dt_proj: *const f32,
        b_weight: *const f32,
        c_weight: *const f32,
        y: *mut f32,
        batch: i64,
        seq: i64,
        dim: i64,
    );
    pub fn aria_state_space_compiled_f32(
        x: *const f32,
        ssm_a: *const f32,
        ssm_b_weight: *const f32,
        ssm_c_weight: *const f32,
        ssm_d: *const f32,
        ssm_dt_weight: *const f32,
        ssm_dt_bias: *const f32,
        y: *mut f32,
        batch: i64,
        seq: i64,
        dim: i64,
        state_dim: i64,
    );
    pub fn aria_gated_delta_compiled_f32(
        x: *const f32,
        q_weight: *const f32,
        k_weight: *const f32,
        v_weight: *const f32,
        alpha_weight: *const f32,
        beta_weight: *const f32,
        o_weight: *const f32,
        y: *mut f32,
        batch: i64,
        seq: i64,
        dim: i64,
        n_heads: i64,
    );
    pub fn aria_selective_scan_compiled_backward_f32(
        grad_out: *const f32,
        x: *const f32,
        a_log: *const f32,
        dt_proj: *const f32,
        b_weight: *const f32,
        c_weight: *const f32,
        grad_x: *mut f32,
        grad_a_log: *mut f32,
        grad_dt_proj: *mut f32,
        grad_b_weight: *mut f32,
        grad_c_weight: *mut f32,
        batch: i64,
        seq: i64,
        dim: i64,
    );
    pub fn aria_state_space_compiled_backward_f32(
        grad_out: *const f32,
        x: *const f32,
        ssm_a: *const f32,
        ssm_b_weight: *const f32,
        ssm_c_weight: *const f32,
        ssm_d: *const f32,
        ssm_dt_weight: *const f32,
        ssm_dt_bias: *const f32,
        grad_x: *mut f32,
        grad_ssm_a: *mut f32,
        grad_ssm_b_weight: *mut f32,
        grad_ssm_c_weight: *mut f32,
        grad_ssm_d: *mut f32,
        grad_ssm_dt_weight: *mut f32,
        grad_ssm_dt_bias: *mut f32,
        batch: i64,
        seq: i64,
        dim: i64,
        state_dim: i64,
    );
    pub fn aria_gated_delta_compiled_backward_f32(
        grad_out: *const f32,
        x: *const f32,
        q_weight: *const f32,
        k_weight: *const f32,
        v_weight: *const f32,
        alpha_weight: *const f32,
        beta_weight: *const f32,
        o_weight: *const f32,
        grad_x: *mut f32,
        grad_q_weight: *mut f32,
        grad_k_weight: *mut f32,
        grad_v_weight: *mut f32,
        grad_alpha_weight: *mut f32,
        grad_beta_weight: *mut f32,
        grad_o_weight: *mut f32,
        batch: i64,
        seq: i64,
        dim: i64,
        n_heads: i64,
    );
    pub fn aria_layernorm_f32(
        x: *const f32,
        weight: *const f32,
        bias: *const f32,
        y: *mut f32,
        batch: i64,
        dim: i64,
        eps: f32,
    );
    pub fn aria_layernorm_backward_f32(
        grad_out: *const f32,
        input: *const f32,
        gamma: *const f32,
        grad_in: *mut f32,
        grad_gamma: *mut f32,
        grad_beta: *mut f32,
        batch: i64,
        dim: i64,
        eps: f32,
    );

    // ── Backward (gradient) kernels ──────────────────────────────────
    // Unary backward kernels: grad_out, input_or_output, grad_in, n
    pub fn aria_relu_backward_f32(
        grad_out: *const f32,
        input: *const f32,
        grad_in: *mut f32,
        n: i64,
    );
    pub fn aria_sigmoid_backward_f32(
        grad_out: *const f32,
        output: *const f32,
        grad_in: *mut f32,
        n: i64,
    );
    pub fn aria_tanh_backward_f32(
        grad_out: *const f32,
        output: *const f32,
        grad_in: *mut f32,
        n: i64,
    );
    pub fn aria_gelu_backward_f32(
        grad_out: *const f32,
        input: *const f32,
        grad_in: *mut f32,
        n: i64,
    );
    pub fn aria_silu_backward_f32(
        grad_out: *const f32,
        input: *const f32,
        grad_in: *mut f32,
        n: i64,
    );
    pub fn aria_rmsnorm_backward_f32(
        grad_out: *const f32,
        input: *const f32,
        gamma: *const f32,
        grad_in: *mut f32,
        grad_gamma: *mut f32,
        batch: i64,
        dim: i64,
        eps: f32,
    );

    // Binary backward kernels
    pub fn aria_add_backward_f32(grad_out: *const f32, grad_a: *mut f32, grad_b: *mut f32, n: i64);
    pub fn aria_mul_backward_f32(
        grad_out: *const f32,
        a: *const f32,
        b: *const f32,
        grad_a: *mut f32,
        grad_b: *mut f32,
        n: i64,
    );
    pub fn aria_sub_backward_f32(grad_out: *const f32, grad_a: *mut f32, grad_b: *mut f32, n: i64);

    // Matmul backward: C = A[M,K] @ B[K,N]
    pub fn aria_matmul_backward_f32(
        grad_out: *const f32,
        a: *const f32,
        b: *const f32,
        grad_a: *mut f32,
        grad_b: *mut f32,
        m: i64,
        k: i64,
        n: i64,
    );
    pub fn aria_softmax_attention_backward_f32(
        grad_out: *const f32,
        x: *const f32,
        wq: *const f32,
        wk: *const f32,
        wv: *const f32,
        wo: *const f32,
        grad_x: *mut f32,
        grad_wq: *mut f32,
        grad_wk: *mut f32,
        grad_wv: *mut f32,
        grad_wo: *mut f32,
        batch: i64,
        seq: i64,
        dim: i64,
        n_heads: i64,
    );
    pub fn aria_gated_linear_backward_f32(
        grad_out: *const f32,
        x: *const f32,
        w: *const f32,
        w_gate: *const f32,
        gate_sigmoid: *const f32,
        grad_x: *mut f32,
        grad_w: *mut f32,
        grad_w_gate: *mut f32,
        grad_b: *mut f32,
        grad_b_gate: *mut f32,
        batch: i64,
        dim_in: i64,
        dim_out: i64,
    );
    pub fn aria_rwkv_time_mixing_backward_f32(
        grad_out: *const f32,
        x: *const f32,
        w_decay: *const f32,
        u_bonus: *const f32,
        w_k: *const f32,
        w_v: *const f32,
        w_r: *const f32,
        grad_x: *mut f32,
        grad_w_decay: *mut f32,
        grad_u_bonus: *mut f32,
        grad_w_k: *mut f32,
        grad_w_v: *mut f32,
        grad_w_r: *mut f32,
        batch: i64,
        seq: i64,
        dim: i64,
    );

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
