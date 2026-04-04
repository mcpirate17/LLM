#include <cctype>
#include <string>

#include <torch/extension.h>

torch::Tensor byte_tokenize_utf8(const std::string& text, int64_t vocab_size) {
  if (vocab_size <= 0 || text.empty()) {
    return torch::empty({0}, torch::TensorOptions().dtype(torch::kLong));
  }

  auto out = torch::empty(
      {static_cast<int64_t>(text.size())},
      torch::TensorOptions().dtype(torch::kLong));
  auto* data = out.data_ptr<int64_t>();
  for (size_t i = 0; i < text.size(); ++i) {
    data[i] = static_cast<unsigned char>(text[i]) % vocab_size;
  }
  return out;
}

torch::Tensor whitespace_hash_tokenize(
    const std::string& text,
    int64_t vocab_size) {
  if (vocab_size <= 0 || text.empty()) {
    return torch::empty({0}, torch::TensorOptions().dtype(torch::kLong));
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

torch::Tensor gather_token_batch(
    const torch::Tensor& tokens,
    const torch::Tensor& starts,
    int64_t seq_len) {
  TORCH_CHECK(tokens.dim() == 1, "tokens must be 1D");
  TORCH_CHECK(starts.dim() == 1, "starts must be 1D");
  TORCH_CHECK(seq_len > 0, "seq_len must be positive");
  TORCH_CHECK(tokens.scalar_type() == torch::kLong, "tokens must be int64");
  TORCH_CHECK(starts.scalar_type() == torch::kLong, "starts must be int64");

  auto offsets = torch::arange(seq_len, starts.options());
  auto indices = starts.unsqueeze(1) + offsets.unsqueeze(0);
  return tokens.index_select(0, indices.reshape({-1}))
      .reshape({starts.size(0), seq_len});
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
      "byte_tokenize_utf8",
      &byte_tokenize_utf8,
      "Tokenize UTF-8 bytes with modulo vocab projection");
  m.def(
      "whitespace_hash_tokenize",
      &whitespace_hash_tokenize,
      "Tokenize whitespace-delimited text with deterministic FNV-1a hashing");
  m.def(
      "gather_token_batch",
      &gather_token_batch,
      "Gather token batches from 1D token buffer");
}
