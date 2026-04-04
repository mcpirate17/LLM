#include <torch/extension.h>

namespace {

torch::Tensor compute_new_mask(
    const torch::Tensor& param,
    const torch::Tensor& grad,
    const torch::Tensor& mask,
    int64_t num_to_update) {
  TORCH_CHECK(param.sizes() == mask.sizes(), "param/mask shape mismatch");
  TORCH_CHECK(grad.sizes() == param.sizes(), "grad/param shape mismatch");
  TORCH_CHECK(mask.scalar_type() == torch::kBool, "mask must be bool");

  const auto flat_mask = mask.reshape({-1});
  const auto active_indices = torch::nonzero(flat_mask).reshape({-1});
  const auto num_active = active_indices.size(0);
  const auto keep_k = num_active - num_to_update;

  auto new_mask_flat = torch::zeros_like(flat_mask);
  const auto flat_weight_mag = param.abs().reshape({-1});
  const auto flat_grad_mag = grad.abs().reshape({-1});

  if (keep_k > 0) {
    auto active_weight_mag = flat_weight_mag.index_select(0, active_indices);
    auto keep_local =
        std::get<1>(torch::topk(active_weight_mag, keep_k, 0, true, true));
    auto keep_indices = active_indices.index_select(0, keep_local);
    new_mask_flat.index_put_({keep_indices}, true);
  }

  if (num_to_update > 0) {
    auto grow_candidates = torch::nonzero(~new_mask_flat).reshape({-1});
    auto candidate_grad_mag = flat_grad_mag.index_select(0, grow_candidates);
    auto grow_local =
        std::get<1>(torch::topk(candidate_grad_mag, num_to_update, 0, true, true));
    auto grow_indices = grow_candidates.index_select(0, grow_local);
    new_mask_flat.index_put_({grow_indices}, true);
  }

  return new_mask_flat.reshape(mask.sizes());
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
      "compute_new_mask",
      &compute_new_mask,
      "Compute RigL prune/grow mask");
}
