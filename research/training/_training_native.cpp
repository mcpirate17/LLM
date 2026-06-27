// Single native extension for research/training: data-pipeline tokenization,
// loss kernels, and curriculum schedules. Merged from the former
// _data_native.cpp / _loss_native.cpp / _curriculum_native.cpp so cold start
// pays one JIT build instead of three.
#include <ATen/Parallel.h>
#include <torch/extension.h>
#include <torch/nn/functional/loss.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace F = torch::nn::functional;

namespace {

// ── Data pipeline ─────────────────────────────────────────────────────

inline torch::Tensor empty_long_1d() {
  return torch::empty({0}, torch::TensorOptions().dtype(torch::kLong));
}

torch::Tensor byte_tokenize_utf8(const std::string& text, int64_t vocab_size) {
  if (vocab_size <= 0 || text.empty()) {
    return empty_long_1d();
  }

  auto out = torch::empty(
      {static_cast<int64_t>(text.size())},
      torch::TensorOptions().dtype(torch::kLong));
  auto* data = out.data_ptr<int64_t>();
  const auto vs = static_cast<uint64_t>(vocab_size);
  const auto* src = reinterpret_cast<const unsigned char*>(text.data());
  const auto n = text.size();
  for (size_t i = 0; i < n; ++i) {
    data[i] = static_cast<int64_t>(static_cast<uint64_t>(src[i]) % vs);
  }
  return out;
}

torch::Tensor byte_tokenize_file_prefix_utf8(
    const std::string& path,
    int64_t vocab_size,
    int64_t max_bytes) {
  if (vocab_size <= 0) {
    return empty_long_1d();
  }

  std::ifstream file(path, std::ios::binary);
  TORCH_CHECK(file.good(), "failed to open file: ", path);

  file.seekg(0, std::ios::end);
  const auto size = static_cast<size_t>(file.tellg());
  file.seekg(0, std::ios::beg);
  size_t used = size;
  if (max_bytes >= 0) {
    used = std::min(used, static_cast<size_t>(max_bytes));
  }
  if (used == 0) {
    return empty_long_1d();
  }

  std::vector<char> buffer(used);
  file.read(buffer.data(), static_cast<std::streamsize>(used));
  TORCH_CHECK(file.good() || file.eof(), "failed to read file: ", path);

  auto out = torch::empty(
      {static_cast<int64_t>(used)},
      torch::TensorOptions().dtype(torch::kLong));
  auto* data = out.data_ptr<int64_t>();
  const auto vs = static_cast<uint64_t>(vocab_size);
  const auto* src = reinterpret_cast<const unsigned char*>(buffer.data());
  for (size_t i = 0; i < used; ++i) {
    data[i] = static_cast<int64_t>(static_cast<uint64_t>(src[i]) % vs);
  }
  return out;
}

torch::Tensor byte_tokenize_file_utf8(const std::string& path, int64_t vocab_size) {
  return byte_tokenize_file_prefix_utf8(path, vocab_size, -1);
}

// Project a 1D int64 token buffer into [0, vocab_size) in-place. Avoids the
// numpy detour when ingesting pre-tokenized .npy files.
void project_int64_modulo_inplace(torch::Tensor tokens, int64_t vocab_size) {
  TORCH_CHECK(tokens.dim() == 1, "tokens must be 1D");
  TORCH_CHECK(tokens.scalar_type() == torch::kLong, "tokens must be int64");
  TORCH_CHECK(vocab_size > 0, "vocab_size must be positive");
  TORCH_CHECK(tokens.is_contiguous(), "tokens must be contiguous");
  const auto n = tokens.numel();
  auto* data = tokens.data_ptr<int64_t>();
  const auto vs = static_cast<int64_t>(vocab_size);
  at::parallel_for(0, n, 1 << 14, [&](int64_t begin, int64_t end) {
    for (int64_t i = begin; i < end; ++i) {
      int64_t v = data[i] % vs;
      if (v < 0) v += vs;
      data[i] = v;
    }
  });
}

torch::Tensor whitespace_hash_tokenize(
    const std::string& text,
    int64_t vocab_size) {
  if (vocab_size <= 0 || text.empty()) {
    return empty_long_1d();
  }

  int64_t token_count = 0;
  bool in_token = false;
  for (unsigned char ch : text) {
    if (std::isspace(ch)) {
      in_token = false;
      continue;
    }
    if (!in_token) {
      ++token_count;
      in_token = true;
    }
  }

  auto out = torch::empty({token_count}, torch::TensorOptions().dtype(torch::kLong));
  auto* data = out.data_ptr<int64_t>();
  int64_t out_idx = 0;
  uint64_t hash = 1469598103934665603ULL;
  in_token = false;
  for (unsigned char ch : text) {
    if (std::isspace(ch)) {
      if (in_token) {
        data[out_idx++] = static_cast<int64_t>(hash % static_cast<uint64_t>(vocab_size));
        hash = 1469598103934665603ULL;
        in_token = false;
      }
      continue;
    }
    in_token = true;
    hash ^= static_cast<uint64_t>(ch);
    hash *= 1099511628211ULL;
  }
  if (in_token) {
    data[out_idx++] = static_cast<int64_t>(hash % static_cast<uint64_t>(vocab_size));
  }
  return out;
}

// Append a single \uXXXX BMP escape decoded as UTF-8 to dst.
inline void append_utf8_bmp(std::string& dst, unsigned int cp) {
  if (cp < 0x80) {
    dst.push_back(static_cast<char>(cp));
  } else if (cp < 0x800) {
    dst.push_back(static_cast<char>(0xC0 | (cp >> 6)));
    dst.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
  } else {
    dst.push_back(static_cast<char>(0xE0 | (cp >> 12)));
    dst.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
    dst.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
  }
}

// Decode a JSON string body starting at `pos` (just past the opening quote)
// in `line` into `dst`. Stops at the closing unescaped quote or `max_chars`.
// Returns the number of source chars consumed (escapes count as 1).
inline int64_t decode_json_string_body(
    const std::string& line,
    size_t& pos,
    std::string& dst,
    int64_t max_chars) {
  int64_t added = 0;
  while (pos < line.size() && added < max_chars) {
    char c = line[pos++];
    if (c == '"') break;
    if (c != '\\' || pos >= line.size()) {
      dst.push_back(c);
    } else {
      char esc = line[pos++];
      switch (esc) {
        case '"': case '\\': case '/': dst.push_back(esc); break;
        case 'b': dst.push_back('\b'); break;
        case 'f': dst.push_back('\f'); break;
        case 'n': dst.push_back('\n'); break;
        case 'r': dst.push_back('\r'); break;
        case 't': dst.push_back('\t'); break;
        case 'u': {
          if (pos + 4 > line.size()) break;
          unsigned int cp = 0;
          for (int k = 0; k < 4; ++k) {
            char h = line[pos++];
            unsigned int v = 0;
            if (h >= '0' && h <= '9') v = h - '0';
            else if (h >= 'a' && h <= 'f') v = 10 + h - 'a';
            else if (h >= 'A' && h <= 'F') v = 10 + h - 'A';
            cp = (cp << 4) | v;
          }
          append_utf8_bmp(dst, cp);
          break;
        }
        default: dst.push_back(esc); break;
      }
    }
    ++added;
  }
  return added;
}

// Walk past `"<text_key>" :  "` in `line`. Returns true on match with `pos`
// pointing one past the opening quote of the value, false if the key is not
// found or the value is not a string.
inline bool find_string_value(
    const std::string& line, const std::string& needle, size_t& pos) {
  pos = line.find(needle);
  if (pos == std::string::npos) return false;
  pos += needle.size();
  while (pos < line.size() && std::isspace(static_cast<unsigned char>(line[pos]))) ++pos;
  if (pos >= line.size() || line[pos++] != ':') return false;
  while (pos < line.size() && std::isspace(static_cast<unsigned char>(line[pos]))) ++pos;
  if (pos >= line.size() || line[pos++] != '"') return false;
  return true;
}

// Fused JSONL byte-tokenizer: one pass through the file, no Python json.loads,
// no per-record tensors. Each line is parsed as a flat JSON object whose
// `text_key` value is a string. Output is the byte-tokenized concatenation
// with a single newline byte separating records, capped at `max_chars`.
torch::Tensor jsonl_byte_tokenize_file(
    const std::string& path,
    const std::string& text_key,
    int64_t vocab_size,
    int64_t max_chars) {
  if (vocab_size <= 0 || max_chars <= 0) return empty_long_1d();
  std::ifstream file(path, std::ios::binary);
  TORCH_CHECK(file.good(), "failed to open file: ", path);

  std::vector<unsigned char> out;
  out.reserve(static_cast<size_t>(std::min<int64_t>(max_chars, 4 * 1024 * 1024)));

  std::string line;
  std::string text_buf;
  line.reserve(8192);
  text_buf.reserve(8192);
  const std::string needle = std::string("\"") + text_key + "\"";
  const auto vs = static_cast<uint64_t>(vocab_size);
  bool first = true;
  int64_t chars_used = 0;

  while (std::getline(file, line) && chars_used < max_chars) {
    if (line.empty()) continue;
    size_t pos = 0;
    if (!find_string_value(line, needle, pos)) continue;

    text_buf.clear();
    int64_t added =
        decode_json_string_body(line, pos, text_buf, max_chars - chars_used);
    if (added <= 0) continue;

    if (!first) {
      out.push_back(static_cast<unsigned char>(static_cast<uint64_t>('\n') % vs));
    }
    first = false;

    const auto base = out.size();
    out.resize(base + text_buf.size());
    const auto* src = reinterpret_cast<const unsigned char*>(text_buf.data());
    for (size_t i = 0; i < text_buf.size(); ++i) {
      out[base + i] = static_cast<unsigned char>(
          static_cast<uint64_t>(src[i]) % vs);
    }
    chars_used += added;
  }

  if (out.empty()) return empty_long_1d();
  auto tensor = torch::empty(
      {static_cast<int64_t>(out.size())},
      torch::TensorOptions().dtype(torch::kLong));
  auto* dst = tensor.data_ptr<int64_t>();
  for (size_t i = 0; i < out.size(); ++i) {
    dst[i] = static_cast<int64_t>(out[i]);
  }
  return tensor;
}

// Gather token batches into a caller-provided contiguous int64 output.
// Lets the caller keep one pinned buffer and reuse it for the whole loop,
// which removes per-step pin/alloc churn.
void gather_token_batch_into(
    const torch::Tensor& tokens,
    const torch::Tensor& starts,
    int64_t seq_len,
    torch::Tensor out) {
  TORCH_CHECK(tokens.dim() == 1, "tokens must be 1D");
  TORCH_CHECK(starts.dim() == 1, "starts must be 1D");
  TORCH_CHECK(seq_len > 0, "seq_len must be positive");
  TORCH_CHECK(tokens.scalar_type() == torch::kLong, "tokens must be int64");
  TORCH_CHECK(starts.scalar_type() == torch::kLong, "starts must be int64");
  TORCH_CHECK(out.dim() == 2, "out must be 2D");
  TORCH_CHECK(out.scalar_type() == torch::kLong, "out must be int64");
  TORCH_CHECK(out.size(0) == starts.size(0), "out batch dim mismatch");
  TORCH_CHECK(out.size(1) == seq_len, "out seq_len dim mismatch");
  TORCH_CHECK(out.is_contiguous(), "out must be contiguous");
  TORCH_CHECK(out.device().is_cpu(), "out must be on CPU");

  const auto tokens_c = tokens.contiguous();
  const auto starts_c = starts.contiguous();
  const auto* tok_ptr = tokens_c.data_ptr<int64_t>();
  const auto* start_ptr = starts_c.data_ptr<int64_t>();
  auto* out_ptr = out.data_ptr<int64_t>();
  const int64_t n_tokens = tokens.numel();
  const int64_t batch = starts.size(0);

  at::parallel_for(0, batch, 1, [&](int64_t begin, int64_t end) {
    for (int64_t b = begin; b < end; ++b) {
      const int64_t s = start_ptr[b];
      TORCH_CHECK(
          s >= 0 && s + seq_len <= n_tokens,
          "start index out of range");
      std::memcpy(
          out_ptr + b * seq_len,
          tok_ptr + s,
          static_cast<size_t>(seq_len) * sizeof(int64_t));
    }
  });
}

torch::Tensor gather_token_batch(
    const torch::Tensor& tokens,
    const torch::Tensor& starts,
    int64_t seq_len) {
  auto out = torch::empty(
      {starts.size(0), seq_len},
      torch::TensorOptions().dtype(torch::kLong));
  gather_token_batch_into(tokens, starts, seq_len, out);
  return out;
}

// ── Loss kernels ──────────────────────────────────────────────────────

template <typename scalar_t>
torch::Tensor rank_weights_cpu_impl(
    const torch::Tensor& flat_logits,
    const torch::Tensor& flat_targets) {
  const auto rows = flat_logits.size(0);
  const auto cols = flat_logits.size(1);

  const auto logits = flat_logits.contiguous();
  const auto targets = flat_targets.contiguous();

  const scalar_t* logits_ptr = logits.data_ptr<scalar_t>();
  const int64_t* targets_ptr = targets.data_ptr<int64_t>();
  auto weights = torch::empty({rows}, flat_logits.options());
  scalar_t* weights_ptr = weights.data_ptr<scalar_t>();

  at::parallel_for(0, rows, 1 << 8, [&](int64_t begin, int64_t end) {
    for (int64_t row = begin; row < end; ++row) {
      const int64_t target = targets_ptr[row];
      const int64_t offset = row * cols;
      const scalar_t target_logit = logits_ptr[offset + target];

      int64_t rank_pos = 0;
      for (int64_t col = 0; col < cols; ++col) {
        rank_pos += logits_ptr[offset + col] > target_logit;
      }

      weights_ptr[row] = static_cast<scalar_t>(
          std::log1p(static_cast<double>(rank_pos)) + 1.0);
    }
  });

  return weights;
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
    torch::Tensor weights;
    switch (flat_logits.scalar_type()) {
      case torch::kFloat32:
        weights = rank_weights_cpu_impl<float>(flat_logits, flat_targets);
        break;
      case torch::kFloat64:
        weights = rank_weights_cpu_impl<double>(flat_logits, flat_targets);
        break;
      default:
        break;
    }
    if (weights.defined()) {
      auto nll = -log_probs.gather(1, flat_targets.unsqueeze(1)).squeeze(1);
      return (nll * weights).mean();
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

torch::Tensor contrastive_push(
    const torch::Tensor& flat_logits,
    const torch::Tensor& flat_targets) {
  TORCH_CHECK(flat_logits.dim() == 2, "flat_logits must be 2D");
  TORCH_CHECK(flat_targets.dim() == 1, "flat_targets must be 1D");
  TORCH_CHECK(
      flat_logits.size(0) == flat_targets.size(0),
      "batch size mismatch between logits and targets");
  const auto vocab = flat_logits.size(1);
  const auto topk_width = std::min<int64_t>(6, vocab);
  if (topk_width <= 1) {
    return flat_logits.new_zeros({});
  }
  auto target_logits = flat_logits.gather(1, flat_targets.unsqueeze(1));
  auto topk = std::get<0>(torch::topk(flat_logits, topk_width, -1, true, true));
  auto negatives = topk.slice(1, 1, topk_width);
  return torch::relu(negatives - target_logits + 0.5).mean();
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

// ── Curriculum schedule ───────────────────────────────────────────────

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
  // Data pipeline
  m.def(
      "byte_tokenize_utf8",
      &byte_tokenize_utf8,
      "Tokenize UTF-8 bytes with modulo vocab projection");
  m.def(
      "byte_tokenize_file_utf8",
      &byte_tokenize_file_utf8,
      "Tokenize a UTF-8 file with modulo vocab projection");
  m.def(
      "byte_tokenize_file_prefix_utf8",
      &byte_tokenize_file_prefix_utf8,
      "Tokenize a UTF-8 file prefix (max_bytes) with modulo vocab projection");
  m.def(
      "jsonl_byte_tokenize_file",
      &jsonl_byte_tokenize_file,
      "Single-pass JSONL byte tokenizer for the configured text field");
  m.def(
      "project_int64_modulo_inplace",
      &project_int64_modulo_inplace,
      "Project a 1D int64 token buffer into [0, vocab_size) in-place");
  m.def(
      "whitespace_hash_tokenize",
      &whitespace_hash_tokenize,
      "Tokenize whitespace-delimited text with deterministic FNV-1a hashing");
  m.def(
      "gather_token_batch",
      &gather_token_batch,
      "Gather token batches from a 1D token buffer (allocates output)");
  m.def(
      "gather_token_batch_into",
      &gather_token_batch_into,
      "Gather token batches into a caller-provided contiguous int64 output");
  // Loss kernels
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
      "contrastive_push",
      &contrastive_push,
      "Contrastive top-k margin push loss");
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
  // Curriculum
  m.def(
      "schedule_seq_lens",
      &schedule_seq_lens,
      "Compute curriculum sequence lengths for [start, stop)");
}
