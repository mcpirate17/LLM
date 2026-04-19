#include <cctype>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <algorithm>
#include <string>
#include <vector>

#include <ATen/Parallel.h>
#include <torch/extension.h>

namespace {

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

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
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
}
