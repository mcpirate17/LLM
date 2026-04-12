#include <cmath>
#include <cstring>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

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

namespace {

py::object mean_or_none(const std::vector<double>& values) {
  if (values.empty()) {
    return py::none();
  }
  double total = 0.0;
  for (double value : values) {
    total += value;
  }
  return py::float_(total / static_cast<double>(values.size()));
}

py::object min_or_none(const std::vector<double>& values) {
  if (values.empty()) {
    return py::none();
  }
  double best = values.front();
  for (double value : values) {
    if (value < best) {
      best = value;
    }
  }
  return py::float_(best);
}

std::string evidence_level_from_count(int64_t n_used) {
  if (n_used < 3) {
    return "insufficient";
  }
  if (n_used < 10) {
    return "sparse";
  }
  if (n_used < 30) {
    return "building";
  }
  return "established";
}

}  // namespace

py::dict screening_graph_analysis_native(
    const std::vector<int64_t>& node_ids,
    const std::vector<std::string>& op_names,
    const std::vector<std::vector<int64_t>>& input_ids,
    const std::vector<uint8_t>& is_input,
    const std::vector<uint8_t>& is_output,
    const std::vector<uint8_t>& has_params) {
  const auto n = node_ids.size();
  TORCH_CHECK(
      op_names.size() == n && input_ids.size() == n && is_input.size() == n
          && is_output.size() == n && has_params.size() == n,
      "screening_graph_analysis_native inputs must have matching lengths");

  std::unordered_map<int64_t, size_t> node_index;
  node_index.reserve(n);
  for (size_t i = 0; i < n; ++i) {
    node_index[node_ids[i]] = i;
  }

  std::vector<std::string> counted_ops;
  counted_ops.reserve(n);
  std::unordered_set<std::string> op_name_set;
  std::set<std::string> toxic_bigrams;
  bool has_parameterized_op = false;

  for (size_t i = 0; i < n; ++i) {
    if (is_input[i]) {
      continue;
    }
    const auto& op_name = op_names[i];
    if (!op_name.empty()) {
      counted_ops.push_back(op_name);
    }
    if (is_output[i]) {
      continue;
    }
    op_name_set.insert(op_name);
    has_parameterized_op = has_parameterized_op || (has_params[i] != 0);
    for (int64_t parent_id : input_ids[i]) {
      auto it = node_index.find(parent_id);
      if (it == node_index.end()) {
        continue;
      }
      const size_t parent_idx = it->second;
      if (is_input[parent_idx] || is_output[parent_idx]) {
        continue;
      }
      toxic_bigrams.insert(op_names[parent_idx] + "->" + op_name);
    }
  }

  py::list counted_ops_py;
  for (const auto& op_name : counted_ops) {
    counted_ops_py.append(py::str(op_name));
  }
  py::list op_names_py;
  for (const auto& op_name : op_name_set) {
    op_names_py.append(py::str(op_name));
  }
  py::list toxic_bigrams_py;
  for (const auto& bigram : toxic_bigrams) {
    toxic_bigrams_py.append(py::str(bigram));
  }

  py::dict out;
  out["counted_ops"] = counted_ops_py;
  out["op_names"] = op_names_py;
  out["toxic_bigrams"] = toxic_bigrams_py;
  out["has_parameterized_op"] = has_parameterized_op;
  return out;
}

py::dict summarize_template_stat_core(
    int64_t n_used,
    int64_t n_stage0,
    int64_t n_stage05,
    int64_t n_stage1,
    const std::vector<double>& losses,
    const std::vector<double>& validation_losses,
    const std::vector<double>& discovery_losses,
    const std::vector<double>& novelties,
    const std::vector<double>& novelty_confidences,
    const std::vector<double>& induction_aucs,
    const std::vector<double>& binding_aucs,
    const std::vector<double>& ar_aucs,
    const std::vector<double>& hellaswag_accs,
    const std::vector<double>& screening_hellaswag_accs,
    int64_t screening_wikitext_ok,
    int64_t screening_wikitext_runs,
    int64_t slot_count,
    int64_t routing_fast_lane_runs,
    int64_t routing_fast_lane_ok,
    int64_t routing_fast_lane_positive,
    const std::vector<double>& routing_fast_lane_scores,
    const std::vector<double>& routing_fast_lane_improvements,
    const std::vector<double>& routing_fast_lane_slopes) {
  const double denom = static_cast<double>(std::max<int64_t>(n_used, 1));
  py::dict out;
  out["n_used"] = n_used;
  out["s0_rate"] = static_cast<double>(n_stage0) / denom;
  out["s05_rate"] = static_cast<double>(n_stage05) / denom;
  out["s1_rate"] = static_cast<double>(n_stage1) / denom;
  out["avg_loss_ratio"] = mean_or_none(losses);
  out["best_loss_ratio"] = min_or_none(losses);
  out["avg_validation_loss_ratio"] = mean_or_none(validation_losses);
  out["avg_discovery_loss_ratio"] = mean_or_none(discovery_losses);
  out["avg_novelty"] = mean_or_none(novelties);
  out["avg_novelty_confidence"] = mean_or_none(novelty_confidences);
  out["avg_induction_auc"] = mean_or_none(induction_aucs);
  out["avg_binding_auc"] = mean_or_none(binding_aucs);
  out["avg_ar_auc"] = mean_or_none(ar_aucs);
  out["avg_hellaswag_acc"] = mean_or_none(hellaswag_accs);
  out["avg_screening_hellaswag_acc"] = mean_or_none(screening_hellaswag_accs);
  out["screening_wikitext_ok_rate"] =
      screening_wikitext_runs > 0
      ? py::object(py::float_(
            static_cast<double>(screening_wikitext_ok)
            / static_cast<double>(std::max<int64_t>(screening_wikitext_runs, 1))))
      : py::object(py::none());
  py::dict coverage;
  coverage["induction"] = static_cast<int64_t>(induction_aucs.size());
  coverage["binding"] = static_cast<int64_t>(binding_aucs.size());
  coverage["associative_recall"] = static_cast<int64_t>(ar_aucs.size());
  coverage["hellaswag"] = static_cast<int64_t>(
      hellaswag_accs.size() + screening_hellaswag_accs.size());
  coverage["wikitext"] = screening_wikitext_runs;
  out["screening_metric_coverage"] = coverage;
  out["slot_count"] = slot_count;
  out["routing_fast_lane_runs"] = routing_fast_lane_runs;
  out["routing_fast_lane_ok_rate"] =
      routing_fast_lane_runs > 0
      ? py::object(py::float_(
            static_cast<double>(routing_fast_lane_ok)
            / static_cast<double>(std::max<int64_t>(routing_fast_lane_runs, 1))))
      : py::object(py::none());
  out["routing_fast_lane_positive_rate"] =
      routing_fast_lane_runs > 0
      ? py::object(py::float_(
            static_cast<double>(routing_fast_lane_positive)
            / static_cast<double>(std::max<int64_t>(routing_fast_lane_runs, 1))))
      : py::object(py::none());
  out["routing_fast_lane_avg_score"] = mean_or_none(routing_fast_lane_scores);
  out["routing_fast_lane_avg_improvement"] =
      mean_or_none(routing_fast_lane_improvements);
  out["routing_fast_lane_avg_slope"] = mean_or_none(routing_fast_lane_slopes);
  out["evidence_level"] = evidence_level_from_count(n_used);
  return out;
}

std::tuple<torch::Tensor, torch::Tensor, int64_t> pad_sequences_native(
    const std::vector<std::vector<int64_t>>& sequences,
    const std::string& device_str) {
  const int64_t n_seq = static_cast<int64_t>(sequences.size());
  if (n_seq == 0) {
    auto opts = torch::TensorOptions().dtype(torch::kInt64);
    return {torch::empty({0, 0}, opts), torch::empty({0}, opts), 0};
  }

  int64_t max_len = 0;
  std::vector<int64_t> lengths_host(static_cast<size_t>(n_seq));
  for (int64_t i = 0; i < n_seq; ++i) {
    const auto slen = static_cast<int64_t>(sequences[i].size());
    lengths_host[i] = slen;
    if (slen > max_len) max_len = slen;
  }
  if (max_len == 0) {
    auto opts = torch::TensorOptions().dtype(torch::kInt64);
    return {torch::zeros({n_seq, 1}, opts), torch::zeros({n_seq}, opts), 0};
  }

  std::vector<int64_t> padded_host(
      static_cast<size_t>(n_seq) * static_cast<size_t>(max_len), 0);
  for (int64_t i = 0; i < n_seq; ++i) {
    const auto& seq = sequences[i];
    if (!seq.empty()) {
      std::memcpy(
          padded_host.data() + static_cast<size_t>(i) * static_cast<size_t>(max_len),
          seq.data(),
          seq.size() * sizeof(int64_t));
    }
  }

  auto opts = torch::TensorOptions().dtype(torch::kInt64);
  torch::Tensor padded = torch::from_blob(
      padded_host.data(), {n_seq, max_len}, opts).clone();
  torch::Tensor lengths = torch::from_blob(
      lengths_host.data(), {n_seq}, opts).clone();

  torch::Device device(device_str);
  if (device.is_cuda()) {
    padded = padded.pin_memory().to(device, /*non_blocking=*/true);
    lengths = lengths.to(device, /*non_blocking=*/true);
  } else if (device != torch::kCPU) {
    padded = padded.to(device);
    lengths = lengths.to(device);
  }
  return {padded, lengths, max_len};
}

torch::Tensor span_mean_log_probs_native(
    const torch::Tensor& token_log_probs,
    const torch::Tensor& start_positions,
    const torch::Tensor& lengths,
    int64_t max_len) {
  const auto device = token_log_probs.device();
  const int64_t n_seq = token_log_probs.size(0);

  auto positions = torch::arange(
      max_len - 1, torch::TensorOptions().device(device)).unsqueeze(0);
  auto span_mask = (positions >= start_positions.unsqueeze(1)).logical_and(
      positions < (lengths - 1).unsqueeze(1));
  auto token_counts = span_mask.sum(1);
  auto valid_spans = token_counts > 0;
  auto mean_lps = torch::full(
      {n_seq}, -std::numeric_limits<float>::infinity(),
      torch::TensorOptions().dtype(torch::kFloat32).device(device));
  auto sums = (token_log_probs * span_mask).sum(1);
  auto denom = token_counts.clamp_min(1).to(torch::kFloat32);
  auto candidate_means = sums / denom;
  mean_lps = torch::where(valid_spans, candidate_means, mean_lps);
  return mean_lps;
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
  m.def(
      "screening_graph_analysis_native",
      &screening_graph_analysis_native,
      "Analyze graph screening facts in native code");
  m.def(
      "summarize_template_stat_core",
      &summarize_template_stat_core,
      "Summarize template-stat arithmetic in native code");
  m.def(
      "pad_sequences_native",
      &pad_sequences_native,
      "Pad variable-length int64 sequences into (batch, max_len) tensor");
  m.def(
      "span_mean_log_probs_native",
      &span_mean_log_probs_native,
      "Compute span-masked mean token log-probs from gathered log-probs");
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
