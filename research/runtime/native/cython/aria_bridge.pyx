# aria_bridge.pyx — Zero-copy Python bridge to native Aria kernels
# cython: language_level=3, boundscheck=False, wraparound=False
import numpy as np
cimport numpy as cnp
from libc.stdint cimport int32_t, int64_t
from libc.stdint cimport uint16_t
from libc.stdlib cimport malloc, free
from libc.string cimport memcpy

cnp.import_array()

cdef inline cnp.ndarray _as_float16_array(object x):
    return np.ascontiguousarray(x, dtype=np.float16)


cdef tuple _FP16_NATIVE_OPS = ('add', 'gelu', 'matmul', 'mul', 'relu', 'rmsnorm', 'sigmoid', 'silu', 'softmax')

# ── Kernel imports (all from Designer kernels.h, resolved via include_dirs) ──

cdef extern from "kernels.h":
    # Elementwise unary
    void aria_relu_f32(const float* x, float* y, int64_t n)
    void aria_gelu_f32(const float* x, float* y, int64_t n)
    void aria_silu_f32(const float* x, float* y, int64_t n)
    void aria_square_f32(const float* x, float* y, int64_t n)
    void aria_abs_f32(const float* x, float* y, int64_t n)
    void aria_neg_f32(const float* x, float* y, int64_t n)
    void aria_reciprocal_f32(const float* x, float* y, int64_t n)
    void aria_log_f32(const float* x, float* y, int64_t n)
    void aria_sqrt_f32(const float* x, float* y, int64_t n)
    void aria_sin_f32(const float* x, float* y, int64_t n)
    void aria_cos_f32(const float* x, float* y, int64_t n)
    void aria_sigmoid_f32(const float* x, float* y, int64_t n)
    void aria_tanh_f32(const float* x, float* y, int64_t n)
    void aria_exp_f32(const float* x, float* y, int64_t n)
    void aria_sign_ste_f32(const float* x, float* y, int64_t n)

    # FP16
    void aria_relu_f16(const uint16_t* x, uint16_t* y, int64_t n)
    void aria_gelu_f16(const uint16_t* x, uint16_t* y, int64_t n)
    void aria_silu_f16(const uint16_t* x, uint16_t* y, int64_t n)
    void aria_sigmoid_f16(const uint16_t* x, uint16_t* y, int64_t n)

    # Elementwise binary
    void aria_add_f32(const float* a, const float* b, float* y, int64_t n)
    void aria_mul_f32(const float* a, const float* b, float* y, int64_t n)
    void aria_sub_f32(const float* a, const float* b, float* y, int64_t n)
    void aria_maximum_f32(const float* a, const float* b, float* y, int64_t n)
    void aria_minimum_f32(const float* a, const float* b, float* y, int64_t n)
    void aria_div_safe_f32(const float* a, const float* b, float* y, int64_t n)
    void aria_outer_product_f32(const float* a, const float* b, float* y, int64_t n)

    # FP16
    void aria_add_f16(const uint16_t* a, const uint16_t* b, uint16_t* y, int64_t n)
    void aria_mul_f16(const uint16_t* a, const uint16_t* b, uint16_t* y, int64_t n)

    # Reductions
    float aria_sum_f32(const float* x, int64_t n)
    float aria_mean_f32(const float* x, int64_t n)

    # Linear algebra
    void aria_matmul_f32(const float* A, const float* B, float* C, int64_t M, int64_t K, int64_t N)
    void aria_linear_f32(const float* x, const float* W, const float* bias, float* y, int64_t batch, int64_t dim_in, int64_t dim_out)
    void aria_matmul_f16(const uint16_t* A, const uint16_t* B, uint16_t* C, int64_t M, int64_t K, int64_t N)

    # Normalization
    void aria_rmsnorm_f32(const float* x, const float* weight, float* y, int64_t batch, int64_t dim, float eps)
    void aria_layernorm_f32(const float* x, const float* weight, const float* bias, float* y, int64_t batch, int64_t dim, float eps)
    void aria_rmsnorm_f16(const uint16_t* x, const uint16_t* weight, uint16_t* y, int64_t batch, int64_t dim, float eps)

    # Softmax
    void aria_softmax_f32(const float* x, float* y, int64_t batch, int64_t dim)
    void aria_softmax_f16(const uint16_t* x, uint16_t* y, int64_t batch, int64_t dim)

    # Structural ops
    void aria_transpose2d_f32(const float* input, float* output, int64_t rows, int64_t cols)
    void aria_concat_f32(const float** inputs, const int64_t* sizes, int32_t n_inputs, float* output)
    void aria_split_f32(const float* input, float** outputs, const int64_t* sizes, int32_t n_outputs)

    # Tier 1 structural/sequence
    void aria_causal_mask_f32(const float* x, float* y, int64_t batch, int64_t seq, int64_t dim)
    void aria_softmax_seq_f32(const float* x, float* y, int64_t batch, int64_t seq, int64_t dim)

    # Tier 2 ops
    void aria_sliding_window_mask_f32(const float* x, float* y, int64_t batch, int64_t seq, int64_t dim, int64_t window_size)
    void aria_sort_seq_f32(const float* x, float* y, int64_t* indices, int64_t batch, int64_t seq, int64_t dim)
    void aria_argsort_seq_f32(const float* x, int64_t* indices, int64_t batch, int64_t seq, int64_t dim)
    void aria_conv1d_seq_f32(const float* x, const float* weight, const float* bias, float* y, int64_t batch, int64_t seq, int64_t dim)
    void aria_fused_linear_gelu_f32(const float* x, const float* W, const float* bias, float* y, int64_t batch, int64_t dim_in, int64_t dim_out)
    void aria_swiglu_f32(const float* x, const float* W_gate, const float* W_up, const float* W_down, const float* bias_gate, const float* bias_up, const float* bias_down, float* y, float* tmp_gate, float* tmp_up, int64_t batch, int64_t dim, int64_t hidden_dim)
    void aria_token_pool_restore_f32(const float* x, float* y, int64_t batch, int64_t seq, int64_t dim)
    void aria_selective_scan_f32(const float* x, const float* A, const float* B, const float* C, const float* D, float* y, int64_t batch, int64_t seq, int64_t dim)
    void aria_topk_gate_f32(const float* x, const float* W_gate, float* y, int64_t batch, int64_t seq, int64_t dim, int64_t k)
    void aria_basis_expansion_f32(const float* x, const float* freqs, float* y, int64_t batch, int64_t seq, int64_t dim, int64_t n_bases)
    void aria_sparse_threshold_f32(const float* x, float* y, int64_t batch, int64_t seq, int64_t dim)

    # Tier 3: Hyperbolic
    void aria_exp_map_f32(const float* x, float* y, int64_t batch, int64_t dim, float c)
    void aria_log_map_f32(const float* x, float* y, int64_t batch, int64_t dim, float c)
    void aria_poincare_add_f32(const float* x, const float* v, float* y, int64_t batch, int64_t dim, float c)
    void aria_hyp_linear_f32(const float* x, const float* W, float* y, int64_t batch, int64_t dim_in, int64_t dim_out, float c)
    void aria_hyperbolic_norm_f32(const float* x, const float* gamma, const float* beta, float* y, int64_t batch, int64_t dim, float c, float eps)
    void aria_hyp_tangent_nonlinear_f32(const float* x, float* y, int64_t n, float c)

    # Tier 3: Tropical (already exist)
    void aria_tropical_attention_f32(const float* x, float* y, int64_t batch, int64_t seq, int64_t dim, float temperature)
    void aria_tropical_center_f32(const float* x, float* y, int64_t batch, int64_t seq, int64_t dim)
    void aria_tropical_gate_f32(const float* x, float* y, int64_t batch, int64_t seq, int64_t dim, float temperature)
    void aria_tropical_add_f32(const float* a, const float* b, float* y, int64_t n)
    void aria_tropical_matmul_f32(const float* A, const float* B, float* C, int64_t M, int64_t K, int64_t N)

    # Tier 3: P-adic
    void aria_padic_gate_f32(const float* x, float* y, int64_t n, float p)
    void aria_padic_expand_f32(const float* x, const float* W, float* y, int64_t batch, int64_t dim, float p, int64_t n_digits)
    void aria_padic_residual_f32(const float* x, const float* W, float* y, int64_t batch, int64_t dim, float p, int64_t n_digits)
    void aria_ultrametric_attention_f32(const float* x, float* y, int64_t batch, int64_t seq, int64_t dim, float p)

    # Tier 3: Clifford
    void aria_rotor_transform_f32(const float* x, const float* rotor, float* y, int64_t batch, int64_t dim)
    void aria_grade_select_f32(const float* x, float* y, int64_t batch, int64_t dim, int32_t grade)
    void aria_grade_mix_f32(const float* x, const float* alpha, float* y, int64_t batch, int64_t dim)
    void aria_clifford_attention_f32(const float* x, float* y, int64_t batch, int64_t seq, int64_t dim)

    # Tier 3: Spiking
    void aria_lif_neuron_f32(const float* x, float* y, int64_t batch, int64_t seq, int64_t dim, float tau, float threshold)
    void aria_spike_rate_code_f32(const float* x, float* y, int64_t batch, int64_t seq, int64_t dim)
    void aria_stdp_attention_f32(const float* x, float* y, int64_t batch, int64_t seq, int64_t dim, float tau_plus, float tau_minus)

    # Reference architecture ops
    void aria_embedding_lookup_f32(const float* table, const int32_t* indices,
                                     const float* pos_embed,
                                     float* y, int64_t batch, int64_t dim,
                                     int64_t vocab_size)
    void aria_rope_rotate_f32(const float* x, float* y,
                                int64_t batch, int64_t seq, int64_t dim,
                                float theta_base)
    void aria_gated_linear_f32(const float* x,
                                 const float* W, const float* b,
                                 const float* W_gate, const float* b_gate,
                                 float* y, float* tmp_gate,
                                 int64_t batch, int64_t dim_in, int64_t dim_out)
    void aria_cosine_similarity_f32(const float* a, const float* b, float* out,
                                      int64_t batch, int64_t seq, int64_t dim)
    void aria_gather_topk_f32(const float* scores, const float* values,
                                float* out, int32_t* out_indices,
                                int64_t batch, int64_t n_items, int64_t dim,
                                int64_t k)
    void aria_rwkv_time_mixing_f32(const float* x,
                                     const float* w_decay, const float* u_bonus,
                                     const float* W_k, const float* W_v, const float* W_r,
                                     float* y,
                                     int64_t batch, int64_t seq, int64_t dim)
    void aria_embedding_lookup_backward_f32(const float* grad_out, const int32_t* indices,
                                              float* grad_table, float* grad_pos_embed,
                                              int64_t batch, int64_t dim,
                                              int64_t vocab_size)
    void aria_gated_linear_backward_f32(const float* grad_out,
                                          const float* x, const float* W, const float* W_gate,
                                          const float* gate_sigmoid,
                                          float* grad_x, float* grad_W, float* grad_W_gate,
                                          float* grad_b, float* grad_b_gate,
                                          int64_t batch, int64_t dim_in, int64_t dim_out)

    # Backward (gradient) kernels — unary
    void aria_relu_backward_f32(const float* grad_out, const float* input, float* grad_in, int64_t n)
    void aria_sigmoid_backward_f32(const float* grad_out, const float* output, float* grad_in, int64_t n)
    void aria_tanh_backward_f32(const float* grad_out, const float* output, float* grad_in, int64_t n)
    void aria_gelu_backward_f32(const float* grad_out, const float* input, float* grad_in, int64_t n)
    void aria_silu_backward_f32(const float* grad_out, const float* input, float* grad_in, int64_t n)

    # Backward (gradient) kernels — binary
    void aria_add_backward_f32(const float* grad_out, float* grad_a, float* grad_b, int64_t n)
    void aria_mul_backward_f32(const float* grad_out, const float* a, const float* b, float* grad_a, float* grad_b, int64_t n)
    void aria_sub_backward_f32(const float* grad_out, float* grad_a, float* grad_b, int64_t n)
    void aria_maximum_backward_f32(const float* grad_out, const float* a, const float* b, float* grad_a, float* grad_b, int64_t n)
    void aria_minimum_backward_f32(const float* grad_out, const float* a, const float* b, float* grad_a, float* grad_b, int64_t n)
    void aria_div_safe_backward_f32(const float* grad_out, const float* a, const float* b, float* grad_a, float* grad_b, int64_t n)

    # Backward (gradient) kernels — matmul
    void aria_matmul_backward_f32(const float* grad_out, const float* A, const float* B, float* grad_A, float* grad_B, int64_t M, int64_t K, int64_t N)

    # Backward (gradient) kernels — normalization / softmax
    void aria_softmax_backward_f32(const float* grad_out, const float* output, float* grad_in, int64_t batch, int64_t dim)
    void aria_layernorm_backward_f32(const float* grad_out, const float* input, const float* gamma, float* grad_in, float* grad_gamma, float* grad_beta, int64_t batch, int64_t dim, float eps)
    void aria_rmsnorm_backward_f32(const float* grad_out, const float* input, const float* gamma, float* grad_in, float* grad_gamma, int64_t batch, int64_t dim, float eps)


# ── Dispatch registry ───────────────────────────────────────────────

_UNARY_OPS = {
    'relu': 'aria_relu_f32',
    'gelu': 'aria_gelu_f32',
    'silu': 'aria_silu_f32',
    'square': 'aria_square_f32',
    'abs': 'aria_abs_f32',
    'neg': 'aria_neg_f32',
    'reciprocal': 'aria_reciprocal_f32',
    'log': 'aria_log_f32',
    'sqrt': 'aria_sqrt_f32',
    'sin': 'aria_sin_f32',
    'cos': 'aria_cos_f32',
    'sigmoid': 'aria_sigmoid_f32',
    'tanh': 'aria_tanh_f32',
    'exp': 'aria_exp_f32',
    'sign_ste': 'aria_sign_ste_f32',
}

_BINARY_OPS = {
    'add': 'aria_add_f32',
    'mul': 'aria_mul_f32',
    'sub': 'aria_sub_f32',
    'maximum': 'aria_maximum_f32',
    'minimum': 'aria_minimum_f32',
    'div_safe': 'aria_div_safe_f32',
    'outer_product': 'aria_outer_product_f32',
}

# Extended ops that need 3D dispatch (batch, seq, dim)
_EXTENDED_OPS = {
    'causal_mask', 'softmax_seq', 'sliding_window_mask',
    'sort_seq', 'argsort_seq', 'conv1d_seq', 'fused_linear_gelu',
    'swiglu', 'token_pool_restore', 'selective_scan', 'topk_gate',
    'basis_expansion', 'sparse_threshold',
    # Reference architecture ops
    'embedding_lookup', 'rope_rotate', 'gated_linear',
    'cosine_similarity', 'gather_topk', 'rwkv_time_mixing',
    # Math space
    'exp_map', 'log_map', 'poincare_add', 'hyp_linear',
    'hyperbolic_norm', 'hyp_tangent_nonlinear',
    'tropical_attention', 'tropical_center', 'tropical_gate',
    'padic_gate', 'padic_expand', 'padic_residual', 'ultrametric_attention',
    'rotor_transform', 'grade_select', 'grade_mix', 'clifford_attention',
    'lif_neuron', 'spike_rate_code', 'stdp_attention',
    # Adaptive routing
    'difficulty_scorer', 'lane_router_threshold',
    'conditional_dispatch', 'conditional_gather',
    'adaptive_route_dispatch',
}


def dispatch_unary(str op_name, cnp.ndarray[float, ndim=1] x):
    """Dispatch a unary op through native C kernels. Returns numpy array."""
    cdef int64_t n = x.shape[0]
    cdef cnp.ndarray[float, ndim=1] y = np.empty(n, dtype=np.float32)
    cdef float* x_ptr = <float*>x.data
    cdef float* y_ptr = <float*>y.data

    if op_name == 'relu':
        aria_relu_f32(x_ptr, y_ptr, n)
    elif op_name == 'gelu':
        aria_gelu_f32(x_ptr, y_ptr, n)
    elif op_name == 'silu':
        aria_silu_f32(x_ptr, y_ptr, n)
    elif op_name == 'square':
        aria_square_f32(x_ptr, y_ptr, n)
    elif op_name == 'abs':
        aria_abs_f32(x_ptr, y_ptr, n)
    elif op_name == 'neg':
        aria_neg_f32(x_ptr, y_ptr, n)
    elif op_name == 'reciprocal':
        aria_reciprocal_f32(x_ptr, y_ptr, n)
    elif op_name == 'log':
        aria_log_f32(x_ptr, y_ptr, n)
    elif op_name == 'sqrt':
        aria_sqrt_f32(x_ptr, y_ptr, n)
    elif op_name == 'sin':
        aria_sin_f32(x_ptr, y_ptr, n)
    elif op_name == 'cos':
        aria_cos_f32(x_ptr, y_ptr, n)
    elif op_name == 'sigmoid':
        aria_sigmoid_f32(x_ptr, y_ptr, n)
    elif op_name == 'tanh':
        aria_tanh_f32(x_ptr, y_ptr, n)
    elif op_name == 'exp':
        aria_exp_f32(x_ptr, y_ptr, n)
    elif op_name == 'sign_ste':
        aria_sign_ste_f32(x_ptr, y_ptr, n)
    else:
        raise ValueError(f"Unsupported unary op: {op_name}")
    return y


def dispatch_binary(str op_name, cnp.ndarray[float, ndim=1] a, cnp.ndarray[float, ndim=1] b):
    """Dispatch a binary op through native C kernels. Returns numpy array."""
    cdef int64_t n = a.shape[0]
    assert b.shape[0] == n, f"Shape mismatch: {a.shape[0]} vs {b.shape[0]}"
    cdef cnp.ndarray[float, ndim=1] y = np.empty(n, dtype=np.float32)
    cdef float* a_ptr = <float*>a.data
    cdef float* b_ptr = <float*>b.data
    cdef float* y_ptr = <float*>y.data

    if op_name == 'add':
        aria_add_f32(a_ptr, b_ptr, y_ptr, n)
    elif op_name == 'mul':
        aria_mul_f32(a_ptr, b_ptr, y_ptr, n)
    elif op_name == 'sub':
        aria_sub_f32(a_ptr, b_ptr, y_ptr, n)
    elif op_name == 'maximum':
        aria_maximum_f32(a_ptr, b_ptr, y_ptr, n)
    elif op_name == 'minimum':
        aria_minimum_f32(a_ptr, b_ptr, y_ptr, n)
    elif op_name == 'div_safe':
        aria_div_safe_f32(a_ptr, b_ptr, y_ptr, n)
    elif op_name == 'outer_product':
        aria_outer_product_f32(a_ptr, b_ptr, y_ptr, n)
    else:
        raise ValueError(f"Unsupported binary op: {op_name}")
    return y


def dispatch_matmul(cnp.ndarray[float, ndim=2] A, cnp.ndarray[float, ndim=2] B):
    """Matrix multiply via native tiled C kernel."""
    cdef int64_t M = A.shape[0]
    cdef int64_t K = A.shape[1]
    cdef int64_t N = B.shape[1]
    assert B.shape[0] == K, f"Shape mismatch: A cols {K} != B rows {B.shape[0]}"
    cdef cnp.ndarray[float, ndim=2] C = np.zeros((M, N), dtype=np.float32)
    aria_matmul_f32(<float*>A.data, <float*>B.data, <float*>C.data, M, K, N)
    return C


def dispatch_linear(cnp.ndarray[float, ndim=2] x, cnp.ndarray[float, ndim=2] W, bias=None):
    """Linear projection: y = x @ W^T + bias."""
    cdef int64_t batch = x.shape[0]
    cdef int64_t dim_in = x.shape[1]
    cdef int64_t dim_out = W.shape[0]
    assert W.shape[1] == dim_in
    cdef cnp.ndarray[float, ndim=2] y = np.empty((batch, dim_out), dtype=np.float32)
    cdef float* bias_ptr = NULL
    cdef cnp.ndarray[float, ndim=1] bias_arr
    if bias is not None:
        bias_arr = np.ascontiguousarray(bias, dtype=np.float32)
        bias_ptr = <float*>bias_arr.data
    aria_linear_f32(<float*>x.data, <float*>W.data, bias_ptr, <float*>y.data, batch, dim_in, dim_out)
    return y


def dispatch_rmsnorm(cnp.ndarray[float, ndim=2] x, cnp.ndarray[float, ndim=1] weight, float eps=1e-5):
    """RMSNorm via native C kernel."""
    cdef int64_t batch = x.shape[0]
    cdef int64_t dim = x.shape[1]
    assert weight.shape[0] == dim
    cdef cnp.ndarray[float, ndim=2] y = np.empty((batch, dim), dtype=np.float32)
    aria_rmsnorm_f32(<float*>x.data, <float*>weight.data, <float*>y.data, batch, dim, eps)
    return y


def dispatch_softmax(cnp.ndarray[float, ndim=2] x):
    """Softmax along last dimension via native C kernel."""
    cdef int64_t batch = x.shape[0]
    cdef int64_t dim = x.shape[1]
    cdef cnp.ndarray[float, ndim=2] y = np.empty((batch, dim), dtype=np.float32)
    aria_softmax_f32(<float*>x.data, <float*>y.data, batch, dim)
    return y


def dispatch_layernorm(cnp.ndarray[float, ndim=2] x, cnp.ndarray[float, ndim=1] weight,
                        cnp.ndarray[float, ndim=1] bias, float eps=1e-5):
    """LayerNorm via native C kernel."""
    cdef int64_t batch = x.shape[0]
    cdef int64_t dim = x.shape[1]
    assert weight.shape[0] == dim
    assert bias.shape[0] == dim
    cdef cnp.ndarray[float, ndim=2] y = np.empty((batch, dim), dtype=np.float32)
    aria_layernorm_f32(<float*>x.data, <float*>weight.data, <float*>bias.data, <float*>y.data, batch, dim, eps)
    return y


def dispatch_transpose2d(cnp.ndarray[float, ndim=2] x):
    """2D transpose via native C kernel."""
    cdef int64_t rows = x.shape[0]
    cdef int64_t cols = x.shape[1]
    cdef cnp.ndarray[float, ndim=2] y = np.empty((cols, rows), dtype=np.float32)
    aria_transpose2d_f32(<float*>x.data, <float*>y.data, rows, cols)
    return y


def native_sum(cnp.ndarray[float, ndim=1] x):
    """Sum reduction via native Kahan summation."""
    return aria_sum_f32(<float*>x.data, x.shape[0])


def native_mean(cnp.ndarray[float, ndim=1] x):
    """Mean reduction."""
    return aria_mean_f32(<float*>x.data, x.shape[0])


def dispatch_unary_fp16(str op_name, x):
    """Dispatch a unary op through native fp16 kernels."""
    cdef cnp.ndarray x_arr = _as_float16_array(x)
    cdef int64_t n = x_arr.size
    cdef cnp.ndarray y = np.empty_like(x_arr)

    if op_name == 'relu':
        aria_relu_f16(<uint16_t*>x_arr.data, <uint16_t*>y.data, n)
    elif op_name == 'gelu':
        aria_gelu_f16(<uint16_t*>x_arr.data, <uint16_t*>y.data, n)
    elif op_name == 'silu':
        aria_silu_f16(<uint16_t*>x_arr.data, <uint16_t*>y.data, n)
    elif op_name == 'sigmoid':
        aria_sigmoid_f16(<uint16_t*>x_arr.data, <uint16_t*>y.data, n)
    else:
        raise ValueError(f"Unsupported fp16 unary op: {op_name}")
    return y


def dispatch_binary_fp16(str op_name, a, b):
    """Dispatch a binary op through native fp16 kernels."""
    cdef cnp.ndarray a_arr = _as_float16_array(a)
    cdef cnp.ndarray b_arr = _as_float16_array(b)
    cdef int64_t n = a_arr.size
    if a_arr.shape != b_arr.shape:
        raise ValueError(f"Shape mismatch: {np.shape(a_arr)} vs {np.shape(b_arr)}")
    cdef cnp.ndarray y = np.empty_like(a_arr)

    if op_name == 'add':
        aria_add_f16(<uint16_t*>a_arr.data, <uint16_t*>b_arr.data, <uint16_t*>y.data, n)
    elif op_name == 'mul':
        aria_mul_f16(<uint16_t*>a_arr.data, <uint16_t*>b_arr.data, <uint16_t*>y.data, n)
    else:
        raise ValueError(f"Unsupported fp16 binary op: {op_name}")
    return y


def dispatch_matmul_fp16(A, B):
    """Matrix multiply via native fp16 kernel."""
    cdef cnp.ndarray A_arr = _as_float16_array(A)
    cdef cnp.ndarray B_arr = _as_float16_array(B)
    if A_arr.ndim != 2 or B_arr.ndim != 2:
        raise ValueError("dispatch_matmul_fp16 expects 2D arrays")
    cdef int64_t M = A_arr.shape[0]
    cdef int64_t K = A_arr.shape[1]
    cdef int64_t N = B_arr.shape[1]
    if B_arr.shape[0] != K:
        raise ValueError(f"Shape mismatch: A cols {K} != B rows {B_arr.shape[0]}")
    cdef cnp.ndarray C = np.empty((M, N), dtype=np.float16)
    aria_matmul_f16(<uint16_t*>A_arr.data, <uint16_t*>B_arr.data, <uint16_t*>C.data, M, K, N)
    return C


def dispatch_softmax_fp16(x):
    """Softmax along last dimension via native fp16 kernel."""
    cdef cnp.ndarray x_arr = _as_float16_array(x)
    if x_arr.ndim != 2:
        raise ValueError("dispatch_softmax_fp16 expects a 2D array")
    cdef int64_t batch = x_arr.shape[0]
    cdef int64_t dim = x_arr.shape[1]
    cdef cnp.ndarray y = np.empty((batch, dim), dtype=np.float16)
    aria_softmax_f16(<uint16_t*>x_arr.data, <uint16_t*>y.data, batch, dim)
    return y


def dispatch_rmsnorm_fp16(x, weight, float eps=1e-5):
    """RMSNorm via native fp16 kernel."""
    cdef cnp.ndarray x_arr = _as_float16_array(x)
    cdef cnp.ndarray w_arr = _as_float16_array(weight)
    if x_arr.ndim != 2 or w_arr.ndim != 1:
        raise ValueError("dispatch_rmsnorm_fp16 expects x as 2D and weight as 1D")
    cdef int64_t batch = x_arr.shape[0]
    cdef int64_t dim = x_arr.shape[1]
    if w_arr.shape[0] != dim:
        raise ValueError(f"Weight dim mismatch: {w_arr.shape[0]} vs {dim}")
    cdef cnp.ndarray y = np.empty((batch, dim), dtype=np.float16)
    aria_rmsnorm_f16(<uint16_t*>x_arr.data, <uint16_t*>w_arr.data, <uint16_t*>y.data, batch, dim, eps)
    return y


def list_native_fp16_ops():
    """Return the fp16 kernel surface supported by the Cython bridge."""
    return list(_FP16_NATIVE_OPS)


def list_native_ops():
    """Return list of all natively supported op names."""
    ops = list(_UNARY_OPS.keys()) + list(_BINARY_OPS.keys())
    ops.extend(['matmul', 'linear', 'rmsnorm', 'layernorm', 'softmax',
                'transpose2d', 'sum', 'mean', 'concat', 'split'])
    ops.extend(sorted(_EXTENDED_OPS))
    return sorted(set(ops))


def is_native(str op_name):
    """Check if an op has a native kernel."""
    return (op_name in _UNARY_OPS or op_name in _BINARY_OPS or
            op_name in _EXTENDED_OPS or op_name in (
                'matmul', 'linear', 'rmsnorm', 'layernorm', 'softmax',
                'transpose2d', 'sum', 'mean', 'concat', 'split'
            ))


# ── Backward (gradient) dispatch ──────────────────────────────────

# Ops that save the forward *input* for backward (relu, gelu, silu)
_UNARY_BACKWARD_INPUT_OPS = {'relu', 'gelu', 'silu'}
# Ops that save the forward *output* for backward (sigmoid, tanh)
_UNARY_BACKWARD_OUTPUT_OPS = {'sigmoid', 'tanh'}


def dispatch_unary_backward(str op_name,
                             cnp.ndarray[float, ndim=1] grad_output,
                             cnp.ndarray[float, ndim=1] forward_saved):
    """Dispatch a unary backward op. Returns grad_input (numpy array).

    For relu/gelu/silu: forward_saved is the forward *input*.
    For sigmoid/tanh:   forward_saved is the forward *output*.
    """
    cdef int64_t n = grad_output.shape[0]
    assert forward_saved.shape[0] == n, \
        f"Shape mismatch: grad_output {n} vs forward_saved {forward_saved.shape[0]}"
    cdef cnp.ndarray[float, ndim=1] grad_in = np.empty(n, dtype=np.float32)
    cdef float* go_ptr = <float*>grad_output.data
    cdef float* fs_ptr = <float*>forward_saved.data
    cdef float* gi_ptr = <float*>grad_in.data

    if op_name == 'relu':
        aria_relu_backward_f32(go_ptr, fs_ptr, gi_ptr, n)
    elif op_name == 'gelu':
        aria_gelu_backward_f32(go_ptr, fs_ptr, gi_ptr, n)
    elif op_name == 'silu':
        aria_silu_backward_f32(go_ptr, fs_ptr, gi_ptr, n)
    elif op_name == 'sigmoid':
        aria_sigmoid_backward_f32(go_ptr, fs_ptr, gi_ptr, n)
    elif op_name == 'tanh':
        aria_tanh_backward_f32(go_ptr, fs_ptr, gi_ptr, n)
    else:
        raise ValueError(f"Unsupported unary backward op: {op_name}")
    return grad_in


def dispatch_binary_backward(str op_name,
                              cnp.ndarray[float, ndim=1] grad_output,
                              cnp.ndarray[float, ndim=1] a,
                              cnp.ndarray[float, ndim=1] b):
    """Dispatch a binary backward op. Returns (grad_a, grad_b) as numpy arrays."""
    cdef int64_t n = grad_output.shape[0]
    assert a.shape[0] == n, f"Shape mismatch: grad_output {n} vs a {a.shape[0]}"
    assert b.shape[0] == n, f"Shape mismatch: grad_output {n} vs b {b.shape[0]}"
    cdef cnp.ndarray[float, ndim=1] grad_a = np.empty(n, dtype=np.float32)
    cdef cnp.ndarray[float, ndim=1] grad_b = np.empty(n, dtype=np.float32)
    cdef float* go_ptr = <float*>grad_output.data
    cdef float* a_ptr = <float*>a.data
    cdef float* b_ptr = <float*>b.data
    cdef float* ga_ptr = <float*>grad_a.data
    cdef float* gb_ptr = <float*>grad_b.data

    if op_name == 'add':
        aria_add_backward_f32(go_ptr, ga_ptr, gb_ptr, n)
    elif op_name == 'mul':
        aria_mul_backward_f32(go_ptr, a_ptr, b_ptr, ga_ptr, gb_ptr, n)
    elif op_name == 'sub':
        aria_sub_backward_f32(go_ptr, ga_ptr, gb_ptr, n)
    elif op_name == 'maximum':
        aria_maximum_backward_f32(go_ptr, a_ptr, b_ptr, ga_ptr, gb_ptr, n)
    elif op_name == 'minimum':
        aria_minimum_backward_f32(go_ptr, a_ptr, b_ptr, ga_ptr, gb_ptr, n)
    elif op_name == 'div_safe':
        aria_div_safe_backward_f32(go_ptr, a_ptr, b_ptr, ga_ptr, gb_ptr, n)
    else:
        raise ValueError(f"Unsupported binary backward op: {op_name}")
    return grad_a, grad_b


def dispatch_matmul_backward(cnp.ndarray[float, ndim=2] grad_output,
                              cnp.ndarray[float, ndim=2] A,
                              cnp.ndarray[float, ndim=2] B):
    """Matmul backward: given grad_output for C=A@B, returns (grad_A, grad_B)."""
    cdef int64_t M = A.shape[0]
    cdef int64_t K = A.shape[1]
    cdef int64_t N = B.shape[1]
    assert B.shape[0] == K, f"Shape mismatch: A cols {K} != B rows {B.shape[0]}"
    assert grad_output.shape[0] == M, f"Shape mismatch: grad_output rows {grad_output.shape[0]} != M {M}"
    assert grad_output.shape[1] == N, f"Shape mismatch: grad_output cols {grad_output.shape[1]} != N {N}"
    cdef cnp.ndarray[float, ndim=2] grad_A = np.zeros((M, K), dtype=np.float32)
    cdef cnp.ndarray[float, ndim=2] grad_B = np.zeros((K, N), dtype=np.float32)
    aria_matmul_backward_f32(
        <float*>grad_output.data,
        <float*>A.data,
        <float*>B.data,
        <float*>grad_A.data,
        <float*>grad_B.data,
        M, K, N,
    )
    return grad_A, grad_B


def dispatch_softmax_backward(cnp.ndarray[float, ndim=2] grad_output,
                               cnp.ndarray[float, ndim=2] output):
    """Softmax backward: given grad_output and forward output, returns grad_input."""
    cdef int64_t batch = grad_output.shape[0]
    cdef int64_t dim = grad_output.shape[1]
    assert output.shape[0] == batch, f"Shape mismatch: grad_output batch {batch} vs output batch {output.shape[0]}"
    assert output.shape[1] == dim, f"Shape mismatch: grad_output dim {dim} vs output dim {output.shape[1]}"
    cdef cnp.ndarray[float, ndim=2] grad_in = np.empty((batch, dim), dtype=np.float32)
    aria_softmax_backward_f32(
        <float*>grad_output.data,
        <float*>output.data,
        <float*>grad_in.data,
        batch, dim,
    )
    return grad_in


def dispatch_layernorm_backward(cnp.ndarray[float, ndim=2] grad_output,
                                 cnp.ndarray[float, ndim=2] input,
                                 cnp.ndarray[float, ndim=1] gamma,
                                 float eps=1e-5):
    """LayerNorm backward: returns (grad_input, grad_gamma, grad_beta)."""
    cdef int64_t batch = grad_output.shape[0]
    cdef int64_t dim = grad_output.shape[1]
    assert input.shape[0] == batch and input.shape[1] == dim
    assert gamma.shape[0] == dim
    cdef cnp.ndarray[float, ndim=2] grad_in = np.empty((batch, dim), dtype=np.float32)
    cdef cnp.ndarray[float, ndim=1] grad_gamma = np.zeros(dim, dtype=np.float32)
    cdef cnp.ndarray[float, ndim=1] grad_beta = np.zeros(dim, dtype=np.float32)
    aria_layernorm_backward_f32(
        <float*>grad_output.data,
        <float*>input.data,
        <float*>gamma.data,
        <float*>grad_in.data,
        <float*>grad_gamma.data,
        <float*>grad_beta.data,
        batch, dim, eps,
    )
    return grad_in, grad_gamma, grad_beta


def dispatch_rmsnorm_backward(cnp.ndarray[float, ndim=2] grad_output,
                               cnp.ndarray[float, ndim=2] input,
                               cnp.ndarray[float, ndim=1] gamma,
                               float eps=1e-5):
    """RMSNorm backward: returns (grad_input, grad_gamma)."""
    cdef int64_t batch = grad_output.shape[0]
    cdef int64_t dim = grad_output.shape[1]
    assert input.shape[0] == batch and input.shape[1] == dim
    assert gamma.shape[0] == dim
    cdef cnp.ndarray[float, ndim=2] grad_in = np.empty((batch, dim), dtype=np.float32)
    cdef cnp.ndarray[float, ndim=1] grad_gamma = np.zeros(dim, dtype=np.float32)
    aria_rmsnorm_backward_f32(
        <float*>grad_output.data,
        <float*>input.data,
        <float*>gamma.data,
        <float*>grad_in.data,
        <float*>grad_gamma.data,
        batch, dim, eps,
    )
    return grad_in, grad_gamma


# ── Reference architecture op dispatch ────────────────────────────

def dispatch_embedding_lookup(cnp.ndarray[float, ndim=2] table,
                                cnp.ndarray[int, ndim=1] indices,
                                pos_embed=None):
    """Embedding lookup: y[batch, dim] = table[indices[batch], :]."""
    cdef int64_t batch = indices.shape[0]
    cdef int64_t dim = table.shape[1]
    cdef int64_t vocab_size = table.shape[0]
    cdef cnp.ndarray[float, ndim=2] y = np.empty((batch, dim), dtype=np.float32)
    cdef float* pe_ptr = NULL
    cdef cnp.ndarray[float, ndim=2] pe_arr
    if pos_embed is not None:
        pe_arr = np.ascontiguousarray(pos_embed, dtype=np.float32)
        pe_ptr = <float*>pe_arr.data
    aria_embedding_lookup_f32(<float*>table.data, <int32_t*>indices.data,
                               pe_ptr, <float*>y.data, batch, dim, vocab_size)
    return y


def dispatch_rope_rotate(cnp.ndarray[float, ndim=3] x, float theta_base=10000.0):
    """Rotary Position Embedding on [batch, seq, dim] input."""
    cdef int64_t batch = x.shape[0]
    cdef int64_t seq = x.shape[1]
    cdef int64_t dim = x.shape[2]
    cdef cnp.ndarray[float, ndim=3] y = np.empty((batch, seq, dim), dtype=np.float32)
    aria_rope_rotate_f32(<float*>x.data, <float*>y.data, batch, seq, dim, theta_base)
    return y


def dispatch_gated_linear(cnp.ndarray[float, ndim=2] x,
                            cnp.ndarray[float, ndim=2] W,
                            cnp.ndarray[float, ndim=2] W_gate,
                            bias=None, bias_gate=None):
    """Gated linear: y = (x @ W^T + b) * sigmoid(x @ W_gate^T + b_gate)."""
    cdef int64_t batch = x.shape[0]
    cdef int64_t dim_in = x.shape[1]
    cdef int64_t dim_out = W.shape[0]
    cdef cnp.ndarray[float, ndim=2] y = np.empty((batch, dim_out), dtype=np.float32)
    cdef cnp.ndarray[float, ndim=2] tmp_gate = np.empty((batch, dim_out), dtype=np.float32)
    cdef float* b_ptr = NULL
    cdef float* bg_ptr = NULL
    cdef cnp.ndarray[float, ndim=1] b_arr, bg_arr
    if bias is not None:
        b_arr = np.ascontiguousarray(bias, dtype=np.float32)
        b_ptr = <float*>b_arr.data
    if bias_gate is not None:
        bg_arr = np.ascontiguousarray(bias_gate, dtype=np.float32)
        bg_ptr = <float*>bg_arr.data
    aria_gated_linear_f32(<float*>x.data, <float*>W.data, b_ptr,
                           <float*>W_gate.data, bg_ptr,
                           <float*>y.data, <float*>tmp_gate.data,
                           batch, dim_in, dim_out)
    return y


def dispatch_cosine_similarity(cnp.ndarray[float, ndim=3] a,
                                 cnp.ndarray[float, ndim=3] b):
    """Cosine similarity: out[batch, seq] = cos_sim(a, b) along last dim."""
    cdef int64_t batch = a.shape[0]
    cdef int64_t seq = a.shape[1]
    cdef int64_t dim = a.shape[2]
    cdef cnp.ndarray[float, ndim=2] out = np.empty((batch, seq), dtype=np.float32)
    aria_cosine_similarity_f32(<float*>a.data, <float*>b.data, <float*>out.data,
                                batch, seq, dim)
    return out


def dispatch_gather_topk(cnp.ndarray[float, ndim=2] scores,
                           cnp.ndarray[float, ndim=3] values,
                           int64_t k):
    """Gather top-k vectors from values by scores."""
    cdef int64_t batch = scores.shape[0]
    cdef int64_t n_items = scores.shape[1]
    cdef int64_t dim = values.shape[2]
    cdef cnp.ndarray[float, ndim=3] out = np.empty((batch, k, dim), dtype=np.float32)
    cdef cnp.ndarray[int, ndim=2] out_indices = np.empty((batch, k), dtype=np.int32)
    aria_gather_topk_f32(<float*>scores.data, <float*>values.data,
                          <float*>out.data, <int32_t*>out_indices.data,
                          batch, n_items, dim, k)
    return out, out_indices


def dispatch_rwkv_time_mixing(cnp.ndarray[float, ndim=3] x,
                                cnp.ndarray[float, ndim=1] w_decay,
                                cnp.ndarray[float, ndim=1] u_bonus,
                                cnp.ndarray[float, ndim=2] W_k,
                                cnp.ndarray[float, ndim=2] W_v,
                                cnp.ndarray[float, ndim=2] W_r):
    """RWKV time-mixing WKV kernel."""
    cdef int64_t batch = x.shape[0]
    cdef int64_t seq = x.shape[1]
    cdef int64_t dim = x.shape[2]
    cdef cnp.ndarray[float, ndim=3] y = np.empty((batch, seq, dim), dtype=np.float32)
    aria_rwkv_time_mixing_f32(<float*>x.data,
                                <float*>w_decay.data, <float*>u_bonus.data,
                                <float*>W_k.data, <float*>W_v.data, <float*>W_r.data,
                                <float*>y.data,
                                batch, seq, dim)
    return y


# ── Adaptive routing dispatch (C2 bridge) ─────────────────────────

def dispatch_difficulty_scorer(cnp.ndarray[float, ndim=3] x,
                                 cnp.ndarray[float, ndim=2] w1,
                                 cnp.ndarray[float, ndim=2] w2,
                                 b1=None, b2=None):
    """Bridge wrapper for difficulty scorer via aria_core bindings."""
    import aria_core
    import torch
    cdef cnp.ndarray[float, ndim=1] b1_arr
    cdef cnp.ndarray[float, ndim=1] b2_arr
    if b1 is None:
        b1_arr = np.zeros((w1.shape[0],), dtype=np.float32)
    else:
        b1_arr = np.ascontiguousarray(b1, dtype=np.float32)
    if b2 is None:
        b2_arr = np.zeros((1,), dtype=np.float32)
    else:
        b2_arr = np.ascontiguousarray(b2, dtype=np.float32)
    tx = torch.from_numpy(np.asarray(x, dtype=np.float32))
    tw1 = torch.from_numpy(np.asarray(w1, dtype=np.float32))
    tb1 = torch.from_numpy(b1_arr)
    tw2 = torch.from_numpy(np.asarray(w2, dtype=np.float32))
    tb2 = torch.from_numpy(b2_arr)
    out = aria_core.difficulty_scorer_f32(tx, tw1, tb1, tw2, tb2)
    return out.detach().cpu().numpy().astype(np.float32, copy=False)


def dispatch_lane_router_threshold(cnp.ndarray[float, ndim=2] scores,
                                     int64_t lanes,
                                     thresholds=None):
    """Bridge wrapper for threshold lane routing via aria_core bindings."""
    import aria_core
    import torch
    cdef cnp.ndarray[float, ndim=1] th_arr
    if thresholds is None:
        th_arr = np.linspace(0.0, 1.0, lanes + 1, dtype=np.float32)[1:-1]
    else:
        th_arr = np.ascontiguousarray(thresholds, dtype=np.float32)
    ts = torch.from_numpy(np.asarray(scores, dtype=np.float32))
    tt = torch.from_numpy(th_arr)
    assignments, weights = aria_core.lane_router_threshold_f32(ts, lanes, tt)
    return assignments.detach().cpu().numpy(), weights.detach().cpu().numpy().astype(np.float32, copy=False)


def dispatch_conditional_dispatch(cnp.ndarray[float, ndim=3] x,
                                    assignments,
                                    int64_t lane_id):
    """Bridge wrapper for per-lane token packing via aria_core bindings."""
    import aria_core
    import torch
    lane_out, index_map, lane_counts = aria_core.conditional_dispatch_f32(
        torch.from_numpy(np.asarray(x, dtype=np.float32)),
        torch.from_numpy(np.asarray(assignments, dtype=np.int64)),
        lane_id,
    )
    return (
        lane_out.detach().cpu().numpy().astype(np.float32, copy=False),
        index_map.detach().cpu().numpy(),
        lane_counts.detach().cpu().numpy(),
    )


def dispatch_conditional_dispatch_backward(cnp.ndarray[float, ndim=3] lane_grad, index_map):
    """Bridge wrapper for dispatch backward scatter via aria_core bindings."""
    import aria_core
    import torch
    grad_x = aria_core.conditional_dispatch_backward_f32(
        torch.from_numpy(np.asarray(lane_grad, dtype=np.float32)),
        torch.from_numpy(np.asarray(index_map, dtype=np.int64)),
    )
    return grad_x.detach().cpu().numpy().astype(np.float32, copy=False)


def dispatch_conditional_gather(cnp.ndarray[float, ndim=3] lane_out,
                                  index_map,
                                  cnp.ndarray[float, ndim=2] weights):
    """Bridge wrapper for gather contribution via aria_core bindings."""
    import aria_core
    import torch
    y = aria_core.conditional_gather_f32(
        torch.from_numpy(np.asarray(lane_out, dtype=np.float32)),
        torch.from_numpy(np.asarray(index_map, dtype=np.int64)),
        torch.from_numpy(np.asarray(weights, dtype=np.float32)),
    )
    return y.detach().cpu().numpy().astype(np.float32, copy=False)


def dispatch_conditional_gather_backward(cnp.ndarray[float, ndim=3] grad_y,
                                           cnp.ndarray[float, ndim=3] lane_out,
                                           index_map,
                                           cnp.ndarray[float, ndim=2] weights):
    """Bridge wrapper for gather backward (grad_lane, grad_weights)."""
    import aria_core
    import torch
    grad_lane, grad_weights = aria_core.conditional_gather_backward_f32(
        torch.from_numpy(np.asarray(grad_y, dtype=np.float32)),
        torch.from_numpy(np.asarray(lane_out, dtype=np.float32)),
        torch.from_numpy(np.asarray(index_map, dtype=np.int64)),
        torch.from_numpy(np.asarray(weights, dtype=np.float32)),
    )
    return (
        grad_lane.detach().cpu().numpy().astype(np.float32, copy=False),
        grad_weights.detach().cpu().numpy().astype(np.float32, copy=False),
    )


# --- Mathematical Spaces Additions ---

def dispatch_exp_map(cnp.ndarray[float, ndim=2] x, float c):
    cdef int64_t batch = x.shape[0]
    cdef int64_t dim = x.shape[1]
    cdef cnp.ndarray[float, ndim=2] y = np.empty((batch, dim), dtype=np.float32)
    aria_exp_map_f32(<float*>x.data, <float*>y.data, batch, dim, c)
    return y

def dispatch_log_map(cnp.ndarray[float, ndim=2] x, float c):
    cdef int64_t batch = x.shape[0]
    cdef int64_t dim = x.shape[1]
    cdef cnp.ndarray[float, ndim=2] y = np.empty((batch, dim), dtype=np.float32)
    aria_log_map_f32(<float*>x.data, <float*>y.data, batch, dim, c)
    return y

def dispatch_poincare_add(cnp.ndarray[float, ndim=2] x, cnp.ndarray[float, ndim=2] v, float c):
    cdef int64_t batch = x.shape[0]
    cdef int64_t dim = x.shape[1]
    cdef cnp.ndarray[float, ndim=2] y = np.empty((batch, dim), dtype=np.float32)
    aria_poincare_add_f32(<float*>x.data, <float*>v.data, <float*>y.data, batch, dim, c)
    return y

def dispatch_hyp_linear(cnp.ndarray[float, ndim=2] x, cnp.ndarray[float, ndim=2] W, float c):
    cdef int64_t batch = x.shape[0]
    cdef int64_t dim_in = x.shape[1]
    cdef int64_t dim_out = W.shape[0]
    cdef cnp.ndarray[float, ndim=2] y = np.empty((batch, dim_out), dtype=np.float32)
    aria_hyp_linear_f32(<float*>x.data, <float*>W.data, <float*>y.data, batch, dim_in, dim_out, c)
    return y

def dispatch_hyperbolic_norm(cnp.ndarray[float, ndim=2] x, cnp.ndarray[float, ndim=1] gamma, cnp.ndarray[float, ndim=1] beta, float c, float eps=1e-5):
    cdef int64_t batch = x.shape[0]
    cdef int64_t dim = x.shape[1]
    cdef cnp.ndarray[float, ndim=2] y = np.empty((batch, dim), dtype=np.float32)
    cdef float* beta_ptr = NULL
    if beta is not None:
        beta_ptr = <float*>beta.data
    aria_hyperbolic_norm_f32(<float*>x.data, <float*>gamma.data, beta_ptr, <float*>y.data, batch, dim, c, eps)
    return y

def dispatch_hyp_tangent_nonlinear(cnp.ndarray[float, ndim=2] x, float c):
    cdef int64_t batch = x.shape[0]
    cdef int64_t dim = x.shape[1]
    cdef cnp.ndarray[float, ndim=2] y = np.empty((batch, dim), dtype=np.float32)
    aria_hyp_tangent_nonlinear_f32(<float*>x.data, <float*>y.data, batch * dim, c)
    return y

def dispatch_tropical_attention(cnp.ndarray[float, ndim=3] x, float temperature=1.0):
    cdef int64_t batch = x.shape[0]
    cdef int64_t seq = x.shape[1]
    cdef int64_t dim = x.shape[2]
    cdef cnp.ndarray[float, ndim=3] y = np.empty((batch, seq, dim), dtype=np.float32)
    aria_tropical_attention_f32(<float*>x.data, <float*>y.data, batch, seq, dim, temperature)
    return y

def dispatch_tropical_center(cnp.ndarray[float, ndim=3] x):
    cdef int64_t batch = x.shape[0]
    cdef int64_t seq = x.shape[1]
    cdef int64_t dim = x.shape[2]
    cdef cnp.ndarray[float, ndim=3] y = np.empty((batch, seq, dim), dtype=np.float32)
    aria_tropical_center_f32(<float*>x.data, <float*>y.data, batch, seq, dim)
    return y

def dispatch_tropical_gate(cnp.ndarray[float, ndim=3] x, float temperature=1.0):
    cdef int64_t batch = x.shape[0]
    cdef int64_t seq = x.shape[1]
    cdef int64_t dim = x.shape[2]
    cdef cnp.ndarray[float, ndim=3] y = np.empty((batch, seq, dim), dtype=np.float32)
    aria_tropical_gate_f32(<float*>x.data, <float*>y.data, batch, seq, dim, temperature)
    return y

def dispatch_tropical_add(cnp.ndarray[float, ndim=1] a, cnp.ndarray[float, ndim=1] b):
    cdef int64_t n = a.shape[0]
    cdef cnp.ndarray[float, ndim=1] y = np.empty(n, dtype=np.float32)
    aria_tropical_add_f32(<float*>a.data, <float*>b.data, <float*>y.data, n)
    return y

def dispatch_tropical_matmul(cnp.ndarray[float, ndim=2] A, cnp.ndarray[float, ndim=2] B):
    cdef int64_t M = A.shape[0]
    cdef int64_t K = A.shape[1]
    cdef int64_t N = B.shape[1]
    cdef cnp.ndarray[float, ndim=2] C = np.empty((M, N), dtype=np.float32)
    aria_tropical_matmul_f32(<float*>A.data, <float*>B.data, <float*>C.data, M, K, N)
    return C

def dispatch_padic_gate(cnp.ndarray[float, ndim=1] x, float p=2.0):
    cdef int64_t n = x.shape[0]
    cdef cnp.ndarray[float, ndim=1] y = np.empty(n, dtype=np.float32)
    aria_padic_gate_f32(<float*>x.data, <float*>y.data, n, p)
    return y

def dispatch_padic_expand(cnp.ndarray[float, ndim=2] x, cnp.ndarray[float, ndim=2] W, float p=2.0, int64_t n_digits=3):
    cdef int64_t batch = x.shape[0]
    cdef int64_t dim_ext = W.shape[0]
    cdef cnp.ndarray[float, ndim=2] y = np.empty((batch, dim_ext), dtype=np.float32)
    aria_padic_expand_f32(<float*>x.data, <float*>W.data, <float*>y.data, batch, x.shape[1], p, n_digits)
    return y

def dispatch_padic_residual(cnp.ndarray[float, ndim=2] x, cnp.ndarray[float, ndim=2] W, float p=2.0, int64_t n_digits=3):
    cdef int64_t batch = x.shape[0]
    cdef int64_t dim = x.shape[1]
    cdef cnp.ndarray[float, ndim=2] y = np.empty((batch, dim), dtype=np.float32)
    aria_padic_residual_f32(<float*>x.data, <float*>W.data, <float*>y.data, batch, dim, p, n_digits)
    return y

def dispatch_ultrametric_attention(cnp.ndarray[float, ndim=3] x, float p=2.0):
    cdef int64_t batch = x.shape[0]
    cdef int64_t seq = x.shape[1]
    cdef int64_t dim = x.shape[2]
    cdef cnp.ndarray[float, ndim=3] y = np.empty((batch, seq, dim), dtype=np.float32)
    aria_ultrametric_attention_f32(<float*>x.data, <float*>y.data, batch, seq, dim, p)
    return y

def dispatch_rotor_transform(cnp.ndarray[float, ndim=2] x, cnp.ndarray[float, ndim=1] rotor):
    cdef int64_t batch = x.shape[0]
    cdef int64_t dim = x.shape[1]
    cdef cnp.ndarray[float, ndim=2] y = np.empty((batch, dim), dtype=np.float32)
    aria_rotor_transform_f32(<float*>x.data, <float*>rotor.data, <float*>y.data, batch, dim)
    return y

def dispatch_grade_select(cnp.ndarray[float, ndim=2] x, int32_t grade):
    cdef int64_t batch = x.shape[0]
    cdef int64_t dim = x.shape[1]
    cdef cnp.ndarray[float, ndim=2] y = np.empty((batch, dim), dtype=np.float32)
    aria_grade_select_f32(<float*>x.data, <float*>y.data, batch, dim, grade)
    return y

def dispatch_grade_mix(cnp.ndarray[float, ndim=2] x, cnp.ndarray[float, ndim=1] alpha):
    cdef int64_t batch = x.shape[0]
    cdef int64_t dim = x.shape[1]
    cdef cnp.ndarray[float, ndim=2] y = np.empty((batch, dim), dtype=np.float32)
    aria_grade_mix_f32(<float*>x.data, <float*>alpha.data, <float*>y.data, batch, dim)
    return y

def dispatch_clifford_attention(cnp.ndarray[float, ndim=3] x):
    cdef int64_t batch = x.shape[0]
    cdef int64_t seq = x.shape[1]
    cdef int64_t dim = x.shape[2]
    cdef cnp.ndarray[float, ndim=3] y = np.empty((batch, seq, dim), dtype=np.float32)
    aria_clifford_attention_f32(<float*>x.data, <float*>y.data, batch, seq, dim)
    return y


def dispatch_adaptive_route_dispatch(cnp.ndarray[float, ndim=3] x,
                                       cnp.ndarray[float, ndim=2] w1,
                                       cnp.ndarray[float, ndim=2] w2,
                                       int64_t lanes,
                                       thresholds=None,
                                       b1=None, b2=None):
    """Bridge wrapper for fused scorer->router->dispatch via aria_core bindings."""
    import aria_core
    import torch
    cdef cnp.ndarray[float, ndim=1] b1_arr
    cdef cnp.ndarray[float, ndim=1] b2_arr
    cdef cnp.ndarray[float, ndim=1] th_arr
    if b1 is None:
        b1_arr = np.zeros((w1.shape[0],), dtype=np.float32)
    else:
        b1_arr = np.ascontiguousarray(b1, dtype=np.float32)
    if b2 is None:
        b2_arr = np.zeros((1,), dtype=np.float32)
    else:
        b2_arr = np.ascontiguousarray(b2, dtype=np.float32)
    if thresholds is None:
        th_arr = np.linspace(0.0, 1.0, lanes + 1, dtype=np.float32)[1:-1]
    else:
        th_arr = np.ascontiguousarray(thresholds, dtype=np.float32)

    scores, assignments, weights, lane_out, index_map, lane_counts = aria_core.adaptive_route_dispatch_f32(
        torch.from_numpy(np.asarray(x, dtype=np.float32)),
        torch.from_numpy(np.asarray(w1, dtype=np.float32)),
        torch.from_numpy(b1_arr),
        torch.from_numpy(np.asarray(w2, dtype=np.float32)),
        torch.from_numpy(b2_arr),
        lanes,
        torch.from_numpy(th_arr),
    )
    return (
        scores.detach().cpu().numpy().astype(np.float32, copy=False),
        assignments.detach().cpu().numpy(),
        weights.detach().cpu().numpy().astype(np.float32, copy=False),
        lane_out.detach().cpu().numpy().astype(np.float32, copy=False),
        index_map.detach().cpu().numpy(),
        lane_counts.detach().cpu().numpy(),
    )


def dispatch_embedding_lookup_backward(cnp.ndarray[float, ndim=2] grad_out,
                                         cnp.ndarray[int, ndim=1] indices,
                                         int64_t vocab_size,
                                         bint has_pos_embed=False):
    """Embedding lookup backward: accumulate gradients into table rows."""
    cdef int64_t batch = grad_out.shape[0]
    cdef int64_t dim = grad_out.shape[1]
    cdef cnp.ndarray[float, ndim=2] grad_table = np.zeros((vocab_size, dim), dtype=np.float32)
    cdef cnp.ndarray[float, ndim=2] grad_pos = np.zeros((batch, dim), dtype=np.float32)
    cdef float* gp_ptr = NULL
    if has_pos_embed:
        gp_ptr = <float*>grad_pos.data
    aria_embedding_lookup_backward_f32(<float*>grad_out.data, <int32_t*>indices.data,
                                        <float*>grad_table.data, gp_ptr,
                                        batch, dim, vocab_size)
    if has_pos_embed:
        return grad_table, grad_pos
    return grad_table


def dispatch_gated_linear_backward(cnp.ndarray[float, ndim=2] grad_out,
                                     cnp.ndarray[float, ndim=2] x,
                                     cnp.ndarray[float, ndim=2] W,
                                     cnp.ndarray[float, ndim=2] W_gate,
                                     cnp.ndarray[float, ndim=2] gate_sigmoid):
    """Gated linear backward: returns (grad_x, grad_W, grad_W_gate, grad_b, grad_b_gate)."""
    cdef int64_t batch = x.shape[0]
    cdef int64_t dim_in = x.shape[1]
    cdef int64_t dim_out = W.shape[0]
    cdef cnp.ndarray[float, ndim=2] grad_x = np.zeros((batch, dim_in), dtype=np.float32)
    cdef cnp.ndarray[float, ndim=2] grad_W = np.zeros((dim_out, dim_in), dtype=np.float32)
    cdef cnp.ndarray[float, ndim=2] grad_Wg = np.zeros((dim_out, dim_in), dtype=np.float32)
    cdef cnp.ndarray[float, ndim=1] grad_b = np.zeros(dim_out, dtype=np.float32)
    cdef cnp.ndarray[float, ndim=1] grad_bg = np.zeros(dim_out, dtype=np.float32)
    aria_gated_linear_backward_f32(<float*>grad_out.data,
                                     <float*>x.data, <float*>W.data, <float*>W_gate.data,
                                     <float*>gate_sigmoid.data,
                                     <float*>grad_x.data, <float*>grad_W.data,
                                     <float*>grad_Wg.data,
                                     <float*>grad_b.data, <float*>grad_bg.data,
                                     batch, dim_in, dim_out)
    return grad_x, grad_W, grad_Wg, grad_b, grad_bg


def has_backward(str op_name):
    """Check if an op has a native backward kernel."""
    return op_name in _UNARY_BACKWARD_INPUT_OPS or \
           op_name in _UNARY_BACKWARD_OUTPUT_OPS or \
           op_name in ('add', 'mul', 'sub', 'maximum', 'minimum', 'div_safe',
                       'matmul', 'softmax', 'layernorm', 'rmsnorm',
                       'embedding_lookup', 'gated_linear')
