#include <torch/extension.h>

namespace {

torch::Tensor orthogonalize_update(const torch::Tensor& matrix, int64_t n_steps) {
  TORCH_CHECK(matrix.dim() == 2, "matrix must be 2D");

  constexpr double a = 3.4445;
  constexpr double b = -4.7750;
  constexpr double c = 2.0315;

  const auto rows = matrix.size(0);
  const auto cols = matrix.size(1);
  const bool transposed = rows < cols;
  auto working = transposed ? matrix.transpose(0, 1) : matrix;
  const auto norm = working.norm();
  if (norm.item<double>() < 1e-8) {
    return matrix;
  }

  auto x = working / norm;
  for (int64_t i = 0; i < n_steps; ++i) {
    auto gram = x.transpose(0, 1).matmul(x);
    auto xg = x.matmul(gram);
    x = a * x + b * xg + c * xg.matmul(gram);
  }

  return transposed ? x.transpose(0, 1) : x;
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
      "orthogonalize_update",
      &orthogonalize_update,
      "Newton-Schulz orthogonalization for Muon");
}
