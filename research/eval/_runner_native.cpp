#include <cmath>
#include <vector>

#include <torch/extension.h>
#include <torch/nn/functional/loss.h>

namespace F = torch::nn::functional;

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

void sgd_step_inplace(
    torch::Tensor param,
    const torch::Tensor& grad,
    torch::Tensor momentum_buf,
    double lr,
    double momentum,
    double weight_decay,
    bool nesterov) {
  auto update = grad;
  if (weight_decay != 0.0) {
    update = update.add(param, weight_decay);
  }
  if (momentum != 0.0) {
    momentum_buf.mul_(momentum).add_(update);
    update = nesterov ? update.add(momentum_buf, momentum) : momentum_buf;
  }
  param.add_(update, -lr);
}

void adamw_step_inplace(
    torch::Tensor param,
    const torch::Tensor& grad,
    torch::Tensor exp_avg,
    torch::Tensor exp_avg_sq,
    double lr,
    double beta1,
    double beta2,
    double eps,
    double weight_decay,
    int64_t step) {
  exp_avg.mul_(beta1).add_(grad, 1.0 - beta1);
  exp_avg_sq.mul_(beta2).addcmul_(grad, grad, 1.0 - beta2);

  const double bias_correction1 = 1.0 - std::pow(beta1, static_cast<double>(step));
  const double bias_correction2 = 1.0 - std::pow(beta2, static_cast<double>(step));
  const double step_size = lr / bias_correction1;

  auto denom = exp_avg_sq.sqrt().div_(std::sqrt(bias_correction2)).add_(eps);
  if (weight_decay != 0.0) {
    param.mul_(1.0 - lr * weight_decay);
  }
  param.addcdiv_(exp_avg, denom, -step_size);
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

py::dict summarize_training_loop(
    int64_t total_tokens,
    double total_time_ms,
    int64_t step_count,
    double step_time_sum_ms,
    double grad_norm_sum,
    double grad_norm_sq_sum,
    double grad_norm_max,
    int64_t grad_norm_count) {
  py::dict out;
  out["throughput"] = total_time_ms > 0.0
      ? py::float_(static_cast<double>(total_tokens) / (total_time_ms / 1000.0))
      : py::float_(0.0);
  out["avg_step_time_ms"] = step_count > 0
      ? py::float_(step_time_sum_ms / static_cast<double>(step_count))
      : py::float_(0.0);
  out["n_train_steps"] = step_count;
  if (grad_norm_count > 0) {
    const double mean = grad_norm_sum / static_cast<double>(grad_norm_count);
    const double var = std::max(
        (grad_norm_sq_sum / static_cast<double>(grad_norm_count)) - (mean * mean),
        0.0);
    out["max_grad_norm"] = py::float_(grad_norm_max);
    out["mean_grad_norm"] = py::float_(mean);
    out["grad_norm_std"] = py::float_(std::sqrt(var));
  } else {
    out["max_grad_norm"] = py::none();
    out["mean_grad_norm"] = py::none();
    out["grad_norm_std"] = py::none();
  }
  return out;
}

py::dict grad_stats_fused(
    const std::vector<torch::Tensor>& grads,
    const std::vector<std::string>& names) {
  py::dict layer_norms;
  py::dict out;

  if (grads.empty()) {
    out["total_norm"] = py::float_(0.0);
    out["layer_norms"] = layer_norms;
    out["max_layer"] = py::none();
    out["max_layer_norm"] = py::float_(0.0);
    out["has_nonfinite"] = py::bool_(false);
    out["has_zero"] = py::bool_(true);
    out["num_grads"] = py::int_(0);
    return out;
  }

  double total_sq = 0.0;
  bool has_nonfinite = false;
  bool has_zero = true;
  double max_norm = 0.0;
  int64_t max_idx = -1;
  const auto n = static_cast<int64_t>(grads.size());

  for (int64_t i = 0; i < n; ++i) {
    const auto& grad = grads[i];
    double lnorm = grad.detach().to(torch::kFloat32).norm().item<double>();
    const auto& name = (i < static_cast<int64_t>(names.size())) ? names[i]
        : std::string("param_") + std::to_string(i);

    if (!std::isfinite(lnorm)) {
      has_nonfinite = true;
      layer_norms[py::str(name)] = py::none();
    } else {
      layer_norms[py::str(name)] = py::float_(lnorm);
      total_sq += lnorm * lnorm;
      if (lnorm > 1e-10) has_zero = false;
      if (lnorm > max_norm) { max_norm = lnorm; max_idx = i; }
    }
  }

  double total_norm = has_nonfinite ? 0.0 : std::sqrt(total_sq);
  out["total_norm"] = py::float_(total_norm);
  out["layer_norms"] = layer_norms;
  if (max_idx >= 0 && max_idx < static_cast<int64_t>(names.size())) {
    out["max_layer"] = py::str(names[max_idx]);
  } else {
    out["max_layer"] = py::none();
  }
  out["max_layer_norm"] = py::float_(max_norm);
  out["has_nonfinite"] = py::bool_(has_nonfinite);
  out["has_zero"] = py::bool_(has_zero);
  out["num_grads"] = py::int_(n);
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
      "next_token_cross_entropy",
      &next_token_cross_entropy,
      "Next-token cross entropy over [B,S,V] logits and [B,S] targets");
  m.def("sgd_step_inplace", &sgd_step_inplace, "In-place SGD update");
  m.def("adamw_step_inplace", &adamw_step_inplace, "In-place AdamW update");
  m.def("clip_grad_norm_", &clip_grad_norm_, "In-place gradient clipping");
  m.def(
      "summarize_training_loop",
      &summarize_training_loop,
      "Summarize training-loop aggregate metrics");
  m.def(
      "grad_stats_fused",
      &grad_stats_fused,
      "Fused per-layer gradient norms, total norm, NaN/zero detection");
}
