#include <ATen/Parallel.h>
#include <torch/extension.h>

#include <cmath>
#include <vector>

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
}
