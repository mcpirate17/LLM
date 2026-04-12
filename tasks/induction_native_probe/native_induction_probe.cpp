#include <tuple>

#include <torch/extension.h>

using torch::indexing::Slice;

namespace {

constexpr int64_t kRestrictedVocab = 256;

torch::Tensor make_device_scalar(
    const torch::Tensor& device_ref,
    int64_t value) {
  auto opts = torch::TensorOptions()
                  .dtype(torch::kLong)
                  .device(device_ref.device());
  return torch::full({}, value, opts);
}

}  // namespace

std::tuple<torch::Tensor, torch::Tensor> induction_batch_like(
    const torch::Tensor& device_ref,
    int64_t batch_size,
    int64_t gap,
    int64_t vocab_size = kRestrictedVocab) {
  TORCH_CHECK(batch_size > 0, "batch_size must be positive");
  TORCH_CHECK(gap >= 0, "gap must be non-negative");
  TORCH_CHECK(vocab_size > 2, "vocab_size must be > 2");

  auto opts = torch::TensorOptions()
                  .dtype(torch::kLong)
                  .device(device_ref.device());
  const auto seq_len = gap + 3;
  auto batch = torch::randint(1, vocab_size, {batch_size, seq_len}, opts);
  auto token_a = torch::randint(1, vocab_size, {batch_size}, opts);
  auto token_b = torch::randint(1, vocab_size, {batch_size}, opts);

  batch.index_put_({Slice(), 0}, token_a);
  batch.index_put_({Slice(), 1}, token_b);

  if (gap > 0) {
    auto noise = batch.index({Slice(), Slice(2, gap + 2)});
    auto token_a_expanded = token_a.unsqueeze(1).expand_as(noise);
    auto collisions = noise.eq(token_a_expanded);
    if (collisions.any().item<bool>()) {
      auto offsets = torch::randint(1, vocab_size - 1, collisions.sizes(), opts);
      auto repaired =
          (token_a_expanded + offsets).remainder(vocab_size - 1) + 1;
      noise = torch::where(collisions, repaired, noise);
      batch.index_put_({Slice(), Slice(2, gap + 2)}, noise);
    }
  }

  batch.index_put_({Slice(), gap + 2}, token_a);
  return {batch, token_b};
}

std::tuple<torch::Tensor, torch::Tensor> induction_batch_pool_like(
    const torch::Tensor& device_ref,
    int64_t pool_size,
    int64_t batch_size,
    int64_t gap,
    int64_t vocab_size = kRestrictedVocab) {
  TORCH_CHECK(pool_size > 0, "pool_size must be positive");
  TORCH_CHECK(batch_size > 0, "batch_size must be positive");
  TORCH_CHECK(gap >= 0, "gap must be non-negative");
  TORCH_CHECK(vocab_size > 2, "vocab_size must be > 2");

  auto opts = torch::TensorOptions()
                  .dtype(torch::kLong)
                  .device(device_ref.device());
  const auto seq_len = gap + 3;
  auto batches = torch::randint(
      1, vocab_size, {pool_size, batch_size, seq_len}, opts);
  auto token_a = torch::randint(1, vocab_size, {pool_size, batch_size}, opts);
  auto token_b = torch::randint(1, vocab_size, {pool_size, batch_size}, opts);

  batches.index_put_({Slice(), Slice(), 0}, token_a);
  batches.index_put_({Slice(), Slice(), 1}, token_b);

  if (gap > 0) {
    auto noise = batches.index({Slice(), Slice(), Slice(2, gap + 2)});
    auto token_a_expanded = token_a.unsqueeze(-1).expand_as(noise);
    auto collisions = noise.eq(token_a_expanded);
    if (collisions.any().item<bool>()) {
      auto offsets = torch::randint(1, vocab_size - 1, collisions.sizes(), opts);
      auto repaired =
          (token_a_expanded + offsets).remainder(vocab_size - 1) + 1;
      noise = torch::where(collisions, repaired, noise);
      batches.index_put_({Slice(), Slice(), Slice(2, gap + 2)}, noise);
    }
  }

  batches.index_put_({Slice(), Slice(), gap + 2}, token_a);
  return {batches, token_b};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
      "induction_batch_like",
      &induction_batch_like,
      py::arg("device_ref"),
      py::arg("batch_size"),
      py::arg("gap"),
      py::arg("vocab_size") = kRestrictedVocab);
  m.def(
      "induction_batch_pool_like",
      &induction_batch_pool_like,
      py::arg("device_ref"),
      py::arg("pool_size"),
      py::arg("batch_size"),
      py::arg("gap"),
      py::arg("vocab_size") = kRestrictedVocab);
}
