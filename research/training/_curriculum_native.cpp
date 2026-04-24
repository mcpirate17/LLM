#include <torch/extension.h>

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace {

constexpr int64_t kFixed = 0;
constexpr int64_t kGrowing = 1;
constexpr int64_t kOscillating = 2;
constexpr double kOscillatingScale = 8.0 * M_PI;

torch::Tensor schedule_seq_lens(
    int64_t schedule_code,
    int64_t initial_seq_len,
    int64_t max_seq_len,
    int64_t warmup_steps,
    int64_t total_steps,
    int64_t start,
    int64_t stop) {
  TORCH_CHECK(stop >= start, "stop must be >= start");
  const auto count = stop - start;
  auto out = torch::empty({count}, torch::TensorOptions().dtype(torch::kLong));
  auto* dst = out.data_ptr<int64_t>();
  if (count == 0) return out;

  const int64_t span = max_seq_len - initial_seq_len;
  if (schedule_code == kFixed) {
    std::fill(dst, dst + count, max_seq_len);
    return out;
  }

  if (schedule_code == kGrowing) {
    const double warmup = static_cast<double>(std::max<int64_t>(warmup_steps, 1));
    for (int64_t i = 0; i < count; ++i) {
      const int64_t step = start + i;
      const double progress = static_cast<double>(step) / warmup;
      dst[i] = progress >= 1.0
          ? max_seq_len
          : static_cast<int64_t>(
                static_cast<double>(initial_seq_len) + progress * span);
    }
    return out;
  }

  if (schedule_code == kOscillating) {
    const double scale = total_steps > 1
        ? kOscillatingScale / static_cast<double>(total_steps)
        : kOscillatingScale;
    for (int64_t i = 0; i < count; ++i) {
      const int64_t step = start + i;
      const double frac = 0.5 + 0.5 * std::sin(static_cast<double>(step) * scale);
      dst[i] = static_cast<int64_t>(
          static_cast<double>(initial_seq_len) + frac * span);
    }
    return out;
  }

  std::fill(dst, dst + count, max_seq_len);
  return out;
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
      "schedule_seq_lens",
      &schedule_seq_lens,
      "Compute curriculum sequence lengths for [start, stop)");
}
