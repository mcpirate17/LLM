#include <cmath>
#include <cstring>

#include <c10/core/InferenceMode.h>
#include <torch/extension.h>

std::tuple<int, int> hellaswag_score_batch_native(
    py::object model,
    const std::vector<std::vector<int64_t>>& ctx_tokens,
    const std::vector<std::vector<std::vector<int64_t>>>& ending_tokens,
    const std::vector<int64_t>& labels,
    int64_t vocab_size,
    std::string device_str,
    int64_t max_seq_len);

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
  m.def(
      "hellaswag_score_batch_native",
      &hellaswag_score_batch_native,
      "Score a batch of HellaSwag examples natively");
}

std::tuple<int, int> hellaswag_score_batch_native(
    py::object model,
    const std::vector<std::vector<int64_t>>& ctx_tokens,
    const std::vector<std::vector<std::vector<int64_t>>>& ending_tokens,
    const std::vector<int64_t>& labels,
    int64_t vocab_size,
    std::string device_str,
    int64_t max_seq_len) {
    
    int n_examples = ctx_tokens.size();
    std::vector<std::vector<int64_t>> flat_seqs;
    std::vector<int64_t> flat_starts;
    std::vector<int> group_sizes;
    
    for (int i = 0; i < n_examples; ++i) {
        const auto& c_toks = ctx_tokens[i];
        const auto& e_toks_list = ending_tokens[i];
        group_sizes.push_back(e_toks_list.size());
        
        for (const auto& e_toks : e_toks_list) {
            std::vector<int64_t> combined;
            int64_t start_pos = 0;
            
            if (e_toks.empty()) {
                combined.clear();
                start_pos = 0;
            } else {
                int64_t total_len = c_toks.size() + e_toks.size();
                if (total_len <= max_seq_len) {
                    combined = c_toks;
                    combined.insert(combined.end(), e_toks.begin(), e_toks.end());
                    start_pos = std::max<int64_t>(0, c_toks.size() - 1);
                } else {
                    int64_t excess = total_len - max_seq_len;
                    int64_t ctx_len;
                    if (excess < (int64_t)c_toks.size()) {
                        combined.insert(combined.end(), c_toks.begin() + excess, c_toks.end());
                        ctx_len = combined.size();
                        combined.insert(combined.end(), e_toks.begin(), e_toks.end());
                    } else {
                        combined.insert(combined.end(), e_toks.begin() + (excess - c_toks.size()), e_toks.end());
                        ctx_len = 0;
                    }
                    start_pos = std::max<int64_t>(0, ctx_len - 1);
                }
            }
            flat_seqs.push_back(combined);
            flat_starts.push_back(start_pos);
        }
    }
    
    int n_seq = flat_seqs.size();
    if (n_seq == 0) return {0, 0};
    
    int64_t max_len = 0;
    for (const auto& s : flat_seqs) {
        if ((int64_t)s.size() > max_len) {
            max_len = s.size();
        }
    }
    if (max_len < 2) {
        return {0, 0}; 
    }
    
    auto tensor_opts = torch::TensorOptions().dtype(torch::kInt64);
    std::vector<int64_t> padded_host(static_cast<size_t>(n_seq) * static_cast<size_t>(max_len), 0);
    std::vector<int64_t> lengths_host(static_cast<size_t>(n_seq), 0);
    for (int i = 0; i < n_seq; ++i) {
        const auto& seq = flat_seqs[i];
        const auto seq_len = static_cast<int64_t>(seq.size());
        lengths_host[static_cast<size_t>(i)] = seq_len;
        if (seq_len > 0) {
            std::memcpy(
                padded_host.data() + static_cast<size_t>(i) * static_cast<size_t>(max_len),
                seq.data(),
                static_cast<size_t>(seq_len) * sizeof(int64_t));
        }
    }

    torch::Tensor padded = torch::from_blob(
        padded_host.data(),
        {n_seq, max_len},
        tensor_opts).clone();

    torch::Device device(device_str);
    if (device.is_cuda()) {
        padded = padded.pin_memory().to(device, true);
    }

    auto starts_tensor = torch::from_blob(
        flat_starts.data(),
        {n_seq},
        tensor_opts).clone();
    auto lengths_tensor = torch::from_blob(
        lengths_host.data(),
        {n_seq},
        tensor_opts).clone();
    if (device.is_cuda()) {
        starts_tensor = starts_tensor.to(device, true);
        lengths_tensor = lengths_tensor.to(device, true);
    }

    c10::InferenceMode guard(true);
    torch::Tensor logits = model(padded).cast<torch::Tensor>();
    
    if (logits.size(-1) > vocab_size) {
        logits = logits.index({torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(0, vocab_size)});
    }
    
    auto next_logits = logits.index(
        {torch::indexing::Slice(), torch::indexing::Slice(0, -1)});
    auto targets = padded.index(
        {torch::indexing::Slice(), torch::indexing::Slice(1, torch::indexing::None)});
    auto target_logits = next_logits.gather(2, targets.unsqueeze(2)).squeeze(2);
    auto log_denom = torch::logsumexp(next_logits, -1);
    auto token_lps = target_logits - log_denom;
    
    auto positions = torch::arange(max_len - 1, torch::TensorOptions().device(device)).unsqueeze(0);
    auto span_mask = (positions >= starts_tensor.unsqueeze(1)).logical_and(positions < (lengths_tensor - 1).unsqueeze(1));
    
    auto token_counts = span_mask.sum(1);
    auto valid_spans = token_counts > 0;
    auto mean_lps = torch::full({n_seq}, -std::numeric_limits<float>::infinity(), torch::TensorOptions().dtype(torch::kFloat32).device(device));
    auto sums = (token_lps * span_mask).sum(1);
    auto denom = token_counts.clamp_min(1).to(torch::kFloat32);
    auto candidate_means = sums / denom;
    mean_lps = torch::where(valid_spans, candidate_means, mean_lps);

    mean_lps = mean_lps.cpu();
    auto mean_lps_acc = mean_lps.accessor<float, 1>();
    
    int correct = 0;
    int total = 0;
    int offset = 0;
    for (int i = 0; i < n_examples; ++i) {
        int gsize = group_sizes[i];
        if (gsize == 0) continue;
        
        float best_score = -std::numeric_limits<float>::infinity();
        int best_idx = -1;
        bool all_inf = true;
        
        for (int j = 0; j < gsize; ++j) {
            float s = mean_lps_acc[offset + j];
            if (s > best_score) {
                best_score = s;
                best_idx = j;
            }
            if (s > -std::numeric_limits<float>::infinity()) {
                all_inf = false;
            }
        }
        
        offset += gsize;
        
        if (!all_inf) {
            total++;
            if (best_idx == labels[i]) {
                correct++;
            }
        }
    }
    
    return {correct, total};
}
