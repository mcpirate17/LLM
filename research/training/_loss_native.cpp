#include <ATen/Parallel.h>
#include <torch/extension.h>
#include <torch/nn/functional/loss.h>

#include <cmath>
#include <string>
#include <vector>

namespace F = torch::nn::functional;

namespace {

template <typename scalar_t>
torch::Tensor rank_weighted_ce_cpu_impl(
    const torch::Tensor& flat_logits,
    const torch::Tensor& flat_targets,
    const torch::Tensor& log_probs) {
  const auto rows = flat_logits.size(0);
  const auto cols = flat_logits.size(1);

  const auto logits = flat_logits.contiguous();
  const auto targets = flat_targets.contiguous();
  const auto logs = log_probs.contiguous();

  const scalar_t* logits_ptr = logits.data_ptr<scalar_t>();
  const scalar_t* log_probs_ptr = logs.data_ptr<scalar_t>();
  const int64_t* targets_ptr = targets.data_ptr<int64_t>();

  std::vector<double> partials(at::get_num_threads(), 0.0);

  at::parallel_for(0, rows, 1, [&](int64_t begin, int64_t end) {
    const auto thread_id = at::get_thread_num();
    double local_sum = 0.0;

    for (int64_t row = begin; row < end; ++row) {
      const int64_t target = targets_ptr[row];
      const int64_t offset = row * cols;
      const scalar_t target_logit = logits_ptr[offset + target];

      int64_t rank_pos = 0;
      for (int64_t col = 0; col < cols; ++col) {
        rank_pos += logits_ptr[offset + col] > target_logit;
      }

      const double nll = -static_cast<double>(log_probs_ptr[offset + target]);
      local_sum += nll * (std::log1p(static_cast<double>(rank_pos)) + 1.0);
    }

    partials[thread_id] += local_sum;
  });

  double total = 0.0;
  for (double partial : partials) {
    total += partial;
  }

  return torch::scalar_tensor(
      total / static_cast<double>(rows),
      flat_logits.options());
}

torch::Tensor rank_weighted_ce(
    const torch::Tensor& flat_logits,
    const torch::Tensor& flat_targets,
    const torch::Tensor& log_probs) {
  TORCH_CHECK(flat_logits.dim() == 2, "flat_logits must be 2D");
  TORCH_CHECK(flat_targets.dim() == 1, "flat_targets must be 1D");
  TORCH_CHECK(
      flat_logits.size(0) == flat_targets.size(0),
      "batch size mismatch between logits and targets");
  TORCH_CHECK(
      log_probs.sizes() == flat_logits.sizes(),
      "log_probs/logits shape mismatch");

  if (flat_logits.device().is_cpu() &&
      flat_targets.device().is_cpu() &&
      log_probs.device().is_cpu() &&
      flat_targets.scalar_type() == torch::kInt64) {
    switch (flat_logits.scalar_type()) {
      case torch::kFloat32:
        return rank_weighted_ce_cpu_impl<float>(
            flat_logits,
            flat_targets,
            log_probs);
      case torch::kFloat64:
        return rank_weighted_ce_cpu_impl<double>(
            flat_logits,
            flat_targets,
            log_probs);
      default:
        break;
    }
  }

  auto target_col = flat_targets.unsqueeze(1);
  auto nll = -log_probs.gather(1, target_col).squeeze(1);
  auto target_logits = flat_logits.gather(1, target_col);
  auto rank_pos =
      flat_logits.gt(target_logits).sum(1).to(flat_logits.scalar_type());
  auto weights = torch::log1p(rank_pos) + 1.0;
  return (nll * weights).mean();
}

torch::Tensor entropy_reg(const torch::Tensor& log_probs) {
  auto probs = log_probs.exp();
  auto entropy = -(probs * log_probs).sum(1).mean();
  return entropy * 0.1;
}

torch::Tensor tropical_ce(
    const torch::Tensor& flat_targets,
    const torch::Tensor& log_probs) {
  TORCH_CHECK(log_probs.size(-1) > 1, "tropical_ce requires vocab > 1");
  auto target_col = flat_targets.unsqueeze(1);
  auto target_log_probs = log_probs.gather(1, target_col).squeeze(1);
  auto topk = torch::topk(log_probs, 2, -1, true, true);
  auto top_vals = std::get<0>(topk);
  auto top_idx = std::get<1>(topk);
  auto max_other = torch::where(
      top_idx.select(1, 0).eq(flat_targets),
      top_vals.select(1, 1),
      top_vals.select(1, 0));
  auto margin = max_other - target_log_probs;
  return torch::relu(margin + 1.0).mean();
}

torch::Tensor next_token_cross_entropy(
    const torch::Tensor& logits,
    const torch::Tensor& targets,
    int64_t vocab_size,
    const std::string& reduction) {
  TORCH_CHECK(logits.dim() == 3, "logits must be [B,S,V]");
  TORCH_CHECK(targets.dim() == 2, "targets must be [B,S]");
  auto score_logits = logits.slice(1, 0, logits.size(1) - 1).contiguous();
  if (score_logits.size(-1) > vocab_size) {
    score_logits = score_logits.slice(-1, 0, vocab_size);
  }
  auto flat_logits = score_logits.view({-1, score_logits.size(-1)});
  auto flat_targets =
      targets.slice(1, 1, targets.size(1)).contiguous().view({-1});

  F::CrossEntropyFuncOptions options;
  if (reduction == "sum") {
    options = options.reduction(torch::kSum);
  } else if (reduction == "none") {
    options = options.reduction(torch::kNone);
  } else {
    options = options.reduction(torch::kMean);
  }
  return F::cross_entropy(flat_logits, flat_targets, options);
}

torch::Tensor clip_grad_norm_(
    const std::vector<torch::Tensor>& grads,
    double max_norm,
    double eps) {
  TORCH_CHECK(max_norm >= 0.0, "max_norm must be non-negative");
  if (grads.empty()) {
    return torch::zeros({}, torch::TensorOptions().dtype(torch::kFloat32));
  }

  const auto options = grads.front().options().dtype(torch::kFloat32);
  auto total_sq = torch::zeros({}, options);
  std::vector<torch::Tensor> dense_grads;
  dense_grads.reserve(grads.size());

  for (const auto& grad : grads) {
    if (!grad.defined()) {
      continue;
    }
    auto grad_view = grad.detach();
    if (grad_view.is_sparse()) {
      grad_view = grad_view.coalesce().values();
    }
    total_sq.add_(grad_view.to(torch::kFloat32).pow(2).sum());
    dense_grads.push_back(grad);
  }

  if (dense_grads.empty()) {
    return torch::zeros({}, options);
  }

  auto total_norm = total_sq.sqrt();
  auto clip_coef =
      torch::clamp(torch::full({}, max_norm, options) / (total_norm + eps), 0.0, 1.0);
  for (auto& grad : dense_grads) {
    grad.mul_(clip_coef);
  }
  return total_norm;
}

torch::Tensor rigl_compute_new_mask(
    const torch::Tensor& param,
    const torch::Tensor& grad,
    const torch::Tensor& mask,
    int64_t num_to_update) {
  TORCH_CHECK(param.sizes() == mask.sizes(), "param/mask shape mismatch");
  TORCH_CHECK(grad.sizes() == mask.sizes(), "grad/mask shape mismatch");
  TORCH_CHECK(mask.scalar_type() == torch::kBool, "mask must be bool");

  auto flat_mask = mask.reshape(-1);
  auto active_indices =
      torch::nonzero(flat_mask).select(/*dim=*/1, 0);
  const int64_t num_active = active_indices.numel();
  const int64_t keep_k = std::max<int64_t>(num_active - num_to_update, 0);

  auto new_mask_flat = torch::zeros_like(flat_mask);
  auto flat_weight_mag = param.abs().reshape(-1);
  auto flat_grad_mag = grad.abs().reshape(-1);

  if (keep_k > 0) {
    auto active_weight_mag = flat_weight_mag.index_select(0, active_indices);
    auto keep_local =
        std::get<1>(torch::topk(active_weight_mag, keep_k, 0, true, false));
    auto keep_indices = active_indices.index_select(0, keep_local);
    new_mask_flat.index_fill_(0, keep_indices, true);
  }

  if (num_to_update > 0) {
    auto grow_candidates =
        torch::nonzero(torch::logical_not(new_mask_flat)).select(1, 0);
    const int64_t cand_k =
        std::min<int64_t>(num_to_update, grow_candidates.numel());
    if (cand_k > 0) {
      auto candidate_grad_mag = flat_grad_mag.index_select(0, grow_candidates);
      auto grow_local =
          std::get<1>(torch::topk(candidate_grad_mag, cand_k, 0, true, false));
      auto grow_indices = grow_candidates.index_select(0, grow_local);
      new_mask_flat.index_fill_(0, grow_indices, true);
    }
  }

  return new_mask_flat.reshape(mask.sizes());
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
      "rank_weighted_ce",
      &rank_weighted_ce,
      "Rank-weighted cross entropy");
  m.def(
      "entropy_reg",
      &entropy_reg,
      "Entropy regularization from log probabilities");
  m.def(
      "tropical_ce",
      &tropical_ce,
      "Tropical cross entropy");
  m.def(
      "next_token_cross_entropy",
      &next_token_cross_entropy,
      "Next-token cross entropy over [B,S,V] logits and [B,S] targets");
  m.def(
      "clip_grad_norm_",
      &clip_grad_norm_,
      "In-place gradient clipping by total L2 norm");
  m.def(
      "rigl_compute_new_mask",
      &rigl_compute_new_mask,
      "RigL dynamic sparse mask update (keep by |w|, grow by |g|)");
}
