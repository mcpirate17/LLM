export const CATEGORY_DESCRIPTIONS = {
  elementwise_unary: "Simple one-to-one transformations like activations (ReLU, GELU), exp, and log.",
  elementwise_binary: "Basic interactions between two tensors like add, multiply, and divide.",
  reduction: "Consolidating information along a dimension (mean, sum, max, norm).",
  linear_algebra: "Core matrix operations like matmuls, outer products, and transposes.",
  structural: "Graph routing and shape manipulation (split, concat, roll, gather).",
  parameterized: "Operations with learned weights: linear layers, normalization, and convolutions.",
  mixing: "Specialized information exchange mechanisms like state-spaces or local attention.",
  sequence: "Operations that process token order: causal masking, sorting, and windowed mixing.",
  frequency: "Fourier domain transformations (FFT) for global sequence mixing in O(N log N).",
  math_space: "Advanced operators from non-Euclidean spaces (hyperbolic, p-adic, tropical).",
  functional: "Higher-order components like basis expansions or fixed-point iterations.",
};

export const AVAILABLE_OPS_REFERENCE = {
  elementwise_unary: "abs, cos, exp, gelu, log, neg, reciprocal, relu, sigmoid, sign_ste, silu, sin, sqrt, square, tanh",
  elementwise_binary: "add, div_safe, maximum, minimum, mul, sub",
  reduction: "cumprod_safe, cumsum, max_last, mean_last, mean_seq, norm_last, sum_last, sum_seq",
  linear_algebra: "matmul, outer_product, transpose_sd",
  structural: "concat, gather_sorted, multi_head_mix, roll_neg, roll_seq, scatter_unsort, split2, split3",
  parameterized: "block_sparse_linear, conv1d_seq, fused_linear_gelu, learnable_bias, learnable_scale, linear_proj, linear_proj_down, linear_proj_up, moe_topk, nm_sparse_linear, rmsnorm, rwkv_channel, selective_scan, semi_structured_2_4_linear, swiglu_mlp, topk_gate",
  mixing: "conv_only, fourier_mixing, graph_attention, linear_attention, softmax_attention, state_space",
  sequence: "argsort_seq, causal_mask, local_window_attn, sliding_window_mask, softmax_last, softmax_seq, token_pool_restore",
  frequency: "irfft_seq, rfft_seq",
  math_space: "(loaded dynamically from math space modules)",
  functional: "basis_expansion, fixed_point_iter, integral_kernel",
};
