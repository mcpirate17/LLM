#include <cmath>

#include <torch/extension.h>

torch::Tensor zero_count_last_dim(const torch::Tensor& values, double threshold) {
  TORCH_CHECK(values.dim() >= 1, "values must have at least 1 dimension");
  auto flat = values.reshape({-1, values.size(-1)}).abs();
  auto mask = flat.lt(threshold);
  return mask.sum(0, false, torch::kFloat32);
}

double gini_coefficient_f64(const torch::Tensor& counts) {
  auto flat = counts.reshape({-1}).to(torch::kFloat64);
  if (flat.numel() < 2) {
    return 0.0;
  }
  auto total = flat.sum().item<double>();
  if (total <= 0.0) {
    return 0.0;
  }
  auto sorted = std::get<0>(flat.sort());
  auto n = sorted.numel();
  auto idx = torch::arange(1, n + 1, sorted.options());
  auto weighted = (idx * sorted).sum().item<double>();
  return (2.0 * weighted / (static_cast<double>(n) * total))
         - (static_cast<double>(n + 1) / static_cast<double>(n));
}

torch::Tensor routing_metrics_f64(
    const torch::Tensor& counts,
    double entropy_sum,
    int64_t sample_count) {
  auto flat = counts.reshape({-1}).to(torch::kFloat64);
  const auto n = flat.numel();
  if (n == 0) {
    return torch::tensor({0.0, 0.0, 0.0, 0.0}, torch::kFloat64);
  }

  const double total = flat.sum().item<double>();
  const double avg_entropy = sample_count > 0
      ? (entropy_sum / static_cast<double>(sample_count))
      : 0.0;
  const double max_entropy = n > 1 ? std::log(static_cast<double>(n)) : 0.0;
  const double normalized_entropy = max_entropy > 0.0 ? (avg_entropy / max_entropy) : 0.0;
  const double dominant_fraction = total > 0.0 ? (flat.max().item<double>() / total) : 0.0;
  const double gini = gini_coefficient_f64(flat);
  return torch::tensor(
      {gini, avg_entropy, normalized_entropy, dominant_fraction},
      torch::kFloat64);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
      "zero_count_last_dim",
      &zero_count_last_dim,
      "Count near-zero elements across all leading dimensions");
  m.def(
      "gini_coefficient_f64",
      &gini_coefficient_f64,
      "Compute Gini coefficient from a 1D count tensor");
  m.def(
      "routing_metrics_f64",
      &routing_metrics_f64,
      "Compute routing summary metrics from expert counts and entropy totals");
}
