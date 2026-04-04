#include <cmath>
#include <vector>

#include <torch/extension.h>

namespace {

torch::Tensor layer_norm_affine(
    const torch::Tensor& x,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    double eps = 1e-5) {
  return torch::layer_norm(x, {x.size(-1)}, weight, bias, eps, false);
}

torch::Tensor linear_3d(
    const torch::Tensor& x,
    const torch::Tensor& weight,
    const torch::Tensor& bias) {
  auto y = torch::matmul(x, weight.transpose(0, 1));
  return y + bias.view({1, 1, -1});
}

torch::Tensor attention_forward(
    const torch::Tensor& x,
    const torch::Tensor& in_proj_weight,
    const torch::Tensor& in_proj_bias,
    const torch::Tensor& out_proj_weight,
    const torch::Tensor& out_proj_bias,
    int64_t n_heads) {
  const auto batch = x.size(0);
  const auto seq_len = x.size(1);
  const auto dim = x.size(2);
  const auto head_dim = dim / n_heads;

  auto qkv = linear_3d(x, in_proj_weight, in_proj_bias);
  auto parts = qkv.split(dim, -1);
  auto q = parts[0].view({batch, seq_len, n_heads, head_dim}).transpose(1, 2);
  auto k = parts[1].view({batch, seq_len, n_heads, head_dim}).transpose(1, 2);
  auto v = parts[2].view({batch, seq_len, n_heads, head_dim}).transpose(1, 2);

  auto ctx = torch::scaled_dot_product_attention(
                 q,
                 k,
                 v,
                 c10::optional<torch::Tensor>(),
                 0.0,
                 true,
                 c10::optional<double>(),
                 false)
                 .transpose(1, 2)
                 .contiguous()
                 .view({batch, seq_len, dim});
  return linear_3d(ctx, out_proj_weight, out_proj_bias);
}

}  // namespace

torch::Tensor baseline_transformer_forward(
    const torch::Tensor& input_ids,
    const torch::Tensor& embed_weight,
    const std::vector<torch::Tensor>& attn_in_proj_weights,
    const std::vector<torch::Tensor>& attn_in_proj_biases,
    const std::vector<torch::Tensor>& attn_out_proj_weights,
    const std::vector<torch::Tensor>& attn_out_proj_biases,
    const std::vector<torch::Tensor>& ff1_weights,
    const std::vector<torch::Tensor>& ff1_biases,
    const std::vector<torch::Tensor>& ff2_weights,
    const std::vector<torch::Tensor>& ff2_biases,
    const std::vector<torch::Tensor>& ln1_weights,
    const std::vector<torch::Tensor>& ln1_biases,
    const std::vector<torch::Tensor>& ln2_weights,
    const std::vector<torch::Tensor>& ln2_biases,
    const torch::Tensor& ln_f_weight,
    const torch::Tensor& ln_f_bias,
    const torch::Tensor& head_weight,
    int64_t n_heads) {
  TORCH_CHECK(input_ids.dim() == 2, "input_ids must be [B,S]");
  auto x = torch::embedding(embed_weight, input_ids);

  const auto n_layers = static_cast<int64_t>(attn_in_proj_weights.size());
  for (int64_t i = 0; i < n_layers; ++i) {
    auto h = layer_norm_affine(x, ln1_weights[i], ln1_biases[i]);
    h = attention_forward(
        h,
        attn_in_proj_weights[i],
        attn_in_proj_biases[i],
        attn_out_proj_weights[i],
        attn_out_proj_biases[i],
        n_heads);
    x = x + h;

    h = layer_norm_affine(x, ln2_weights[i], ln2_biases[i]);
    h = linear_3d(h, ff1_weights[i], ff1_biases[i]);
    h = torch::gelu(h);
    h = linear_3d(h, ff2_weights[i], ff2_biases[i]);
    x = x + h;
  }

  x = layer_norm_affine(x, ln_f_weight, ln_f_bias);
  return torch::matmul(x, head_weight.transpose(0, 1));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
      "baseline_transformer_forward",
      &baseline_transformer_forward,
      "Native forward for the shared baseline transformer");
}
