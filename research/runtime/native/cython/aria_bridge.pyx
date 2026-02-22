# aria_bridge.pyx — Zero-copy Python bridge to native Aria kernels
# cython: language_level=3, boundscheck=False, wraparound=False
import numpy as np
cimport numpy as cnp
from libc.stdint cimport int32_t, int64_t
from libc.stdlib cimport malloc, free
from libc.string cimport memcpy

cnp.import_array()

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

    # Elementwise binary
    void aria_add_f32(const float* a, const float* b, float* y, int64_t n)
    void aria_mul_f32(const float* a, const float* b, float* y, int64_t n)
    void aria_sub_f32(const float* a, const float* b, float* y, int64_t n)
    void aria_maximum_f32(const float* a, const float* b, float* y, int64_t n)
    void aria_minimum_f32(const float* a, const float* b, float* y, int64_t n)
    void aria_div_safe_f32(const float* a, const float* b, float* y, int64_t n)
    void aria_outer_product_f32(const float* a, const float* b, float* y, int64_t n)

    # Reductions
    float aria_sum_f32(const float* x, int64_t n)
    float aria_mean_f32(const float* x, int64_t n)

    # Linear algebra
    void aria_matmul_f32(const float* A, const float* B, float* C, int64_t M, int64_t K, int64_t N)
    void aria_linear_f32(const float* x, const float* W, const float* bias, float* y, int64_t batch, int64_t dim_in, int64_t dim_out)

    # Normalization
    void aria_rmsnorm_f32(const float* x, const float* weight, float* y, int64_t batch, int64_t dim, float eps)
    void aria_layernorm_f32(const float* x, const float* weight, const float* bias, float* y, int64_t batch, int64_t dim, float eps)

    # Softmax
    void aria_softmax_f32(const float* x, float* y, int64_t batch, int64_t dim)

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
    void aria_exp_map_f32(const float* x, float* y, int64_t n, float c)
    void aria_log_map_f32(const float* x, float* y, int64_t n, float c)
    void aria_poincare_add_f32(const float* x, const float* v, float* y, int64_t batch, int64_t dim, float c)
    void aria_hyp_linear_f32(const float* x, const float* W, float* y, int64_t batch, int64_t dim_in, int64_t dim_out, float c)
    void aria_hyperbolic_norm_f32(const float* x, const float* gamma, const float* beta, float* y, int64_t batch, int64_t dim, float c, float eps)
    void aria_hyp_tangent_nonlinear_f32(const float* x, float* y, int64_t n, float c)

    # Tier 3: Tropical (already exist)
    void aria_tropical_attention_f32(const float* x, float* y, int64_t batch, int64_t seq, int64_t dim, float temperature)
    void aria_tropical_center_f32(const float* x, float* y, int64_t batch, int64_t seq, int64_t dim)
    void aria_tropical_gate_f32(const float* x, float* y, int64_t batch, int64_t seq, int64_t dim, float temperature)

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
    # Math space
    'exp_map', 'log_map', 'poincare_add', 'hyp_linear',
    'hyperbolic_norm', 'hyp_tangent_nonlinear',
    'tropical_attention', 'tropical_center', 'tropical_gate',
    'padic_gate', 'padic_expand', 'padic_residual', 'ultrametric_attention',
    'rotor_transform', 'grade_select', 'grade_mix', 'clifford_attention',
    'lif_neuron', 'spike_rate_code', 'stdp_attention',
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


def has_backward(str op_name):
    """Check if an op has a native backward kernel."""
    return op_name in _UNARY_BACKWARD_INPUT_OPS or \
           op_name in _UNARY_BACKWARD_OUTPUT_OPS or \
           op_name in ('add', 'mul', 'sub', 'maximum', 'minimum', 'div_safe',
                       'matmul', 'softmax', 'layernorm', 'rmsnorm')
