#include <cmath>

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

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
      "next_token_cross_entropy",
      &next_token_cross_entropy,
      "Next-token cross entropy over [B,S,V] logits and [B,S] targets");
  m.def("sgd_step_inplace", &sgd_step_inplace, "In-place SGD update");
  m.def("adamw_step_inplace", &adamw_step_inplace, "In-place AdamW update");
}
