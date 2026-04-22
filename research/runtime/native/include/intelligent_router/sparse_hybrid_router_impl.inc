#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <stdexcept>

namespace ir {
namespace {

constexpr std::size_t kTokenGateDim = 16;
constexpr std::size_t kSpanDim = 48;
constexpr std::uint32_t kMagic = 0x53524831U;

int wrap_token(int x, int vocab) {
    int y = x % vocab;
    if (y < 0) {
        y += vocab;
    }
    return y;
}

}  // namespace

SparseHybridRouter::SparseHybridRouter(std::size_t vocab, std::size_t lanes)
    : vocab_(vocab),
      lanes_(lanes),
      token_gate_(2, kTokenGateDim),
      span_router_(lanes, kSpanDim) {
    if (vocab == 0 || lanes == 0) {
        throw std::invalid_argument("SparseHybridRouter requires non-zero vocab and lanes");
    }
}

std::vector<float> SparseHybridRouter::encode_token(int token) const {
    if (token < 0 || static_cast<std::size_t>(token) >= vocab_) {
        throw std::invalid_argument("SparseHybridRouter token out of range");
    }
    const float inv_vocab = 1.0F / static_cast<float>(vocab_ - 1);
    std::vector<float> x(kTokenGateDim, 0.0F);
    x[0] = static_cast<float>(token) * inv_vocab;
    x[1] = static_cast<float>(token < static_cast<int>(vocab_ / 4));
    x[2] = static_cast<float>(token >= static_cast<int>(vocab_ / 2));
    x[3] = static_cast<float>(token >= static_cast<int>((3 * vocab_) / 4));
    x[4] = std::sin(6.28318530718F * static_cast<float>(token) / static_cast<float>(vocab_));
    x[5] = std::cos(6.28318530718F * static_cast<float>(token) / static_cast<float>(vocab_));
    x[6] = static_cast<float>((token & 1) != 0);
    x[7] = static_cast<float>((token & 0x3) == 0);
    x[8] = static_cast<float>((token & 0x7) == 0);
    x[9] = static_cast<float>((token % 3) == 0);
    x[10] = static_cast<float>((token % 5) == 0);
    x[11] = static_cast<float>((token ^ 0x15) & 0x1F) / 31.0F;
    x[12] = static_cast<float>((token * 3) % static_cast<int>(vocab_)) * inv_vocab;
    x[13] = static_cast<float>((token * 5) % static_cast<int>(vocab_)) * inv_vocab;
    x[14] = static_cast<float>((token + 7) % static_cast<int>(vocab_)) * inv_vocab;
    x[15] = static_cast<float>((token + 11) % static_cast<int>(vocab_)) * inv_vocab;
    return x;
}

std::vector<float> SparseHybridRouter::encode_sparse_triplet(
    const std::vector<int>& sequence,
    std::vector<int>* informative_indices
) const {
    if (sequence.empty()) {
        throw std::invalid_argument("SparseHybridRouter sequence must be non-empty");
    }
    std::vector<int> kept;
    kept.reserve(sequence.size());
    if (informative_indices != nullptr) {
        informative_indices->clear();
    }
    for (std::size_t i = 0; i < sequence.size(); ++i) {
        const auto gate = token_gate_.route(encode_token(sequence[i]));
        const float keep_prob = gate.probabilities.size() > 1 ? gate.probabilities[1] : 0.0F;
        if (keep_prob >= 0.5F) {
            kept.push_back(sequence[i]);
            if (informative_indices != nullptr) {
                informative_indices->push_back(static_cast<int>(i));
            }
            if (kept.size() == 3) {
                break;
            }
        }
    }
    if (kept.empty()) {
        kept.push_back(sequence[sequence.size() / 2]);
        if (informative_indices != nullptr) {
            informative_indices->push_back(static_cast<int>(sequence.size() / 2));
        }
    }
    while (kept.size() < 3) {
        kept.push_back(kept.back());
        if (informative_indices != nullptr) {
            informative_indices->push_back(
                informative_indices->empty() ? 0 : informative_indices->back()
            );
        }
    }

    const int a = kept[0];
    const int b = kept[1];
    const int c = kept[2];
    const int d1 = wrap_token(b - a, static_cast<int>(vocab_));
    const int d2 = wrap_token(c - b, static_cast<int>(vocab_));
    const int stop_count = static_cast<int>(std::count_if(
        sequence.begin(),
        sequence.end(),
        [&](int tok) { return token_gate_.route(encode_token(tok)).lane == 0; }
    ));
    const float inv_vocab = 1.0F / static_cast<float>(vocab_ - 1);
    const float seq_den =
        sequence.size() > 1 ? static_cast<float>(sequence.size() - 1) : 1.0F;

    std::vector<float> x(kSpanDim, 0.0F);
    x[0] = static_cast<float>(a) * inv_vocab;
    x[1] = static_cast<float>(b) * inv_vocab;
    x[2] = static_cast<float>(c) * inv_vocab;
    x[3] = static_cast<float>(d1) * inv_vocab;
    x[4] = static_cast<float>(d2) * inv_vocab;
    x[5] = static_cast<float>(a < b && b < c);
    x[6] = static_cast<float>(a > b && b > c);
    x[7] = static_cast<float>(d1 == d2);
    x[8] = static_cast<float>(std::abs(d1 - d2) >= 8);
    x[9] = static_cast<float>(
        ((a + c) % static_cast<int>(vocab_)) == ((2 * b) % static_cast<int>(vocab_))
    );
    x[10] = static_cast<float>((a ^ b ^ c) & 1);
    x[11] = static_cast<float>((b & 0x3) == 0);
    x[12] = static_cast<float>(b >= static_cast<int>(vocab_ / 2));
    x[13] = static_cast<float>(a < static_cast<int>(vocab_ / 4));
    x[14] = static_cast<float>(stop_count) / static_cast<float>(sequence.size());
    const int first_idx =
        informative_indices != nullptr && !informative_indices->empty()
            ? (*informative_indices)[0]
            : 0;
    const int last_idx =
        informative_indices != nullptr && !informative_indices->empty()
            ? (*informative_indices)[std::min<std::size_t>(2, informative_indices->size() - 1)]
            : first_idx;
    x[15] = static_cast<float>(first_idx) / seq_den;
    x[16] = static_cast<float>(last_idx) / seq_den;
    x[17] = static_cast<float>(last_idx - first_idx) / seq_den;
    x[18] = static_cast<float>((a + b + c) % static_cast<int>(vocab_)) * inv_vocab;
    x[19] = static_cast<float>((a ^ b ^ c) & 0x1F) / 31.0F;
    x[20] = std::sin(6.28318530718F * static_cast<float>(a) / static_cast<float>(vocab_));
    x[21] = std::cos(6.28318530718F * static_cast<float>(a) / static_cast<float>(vocab_));
    x[22] = std::sin(6.28318530718F * static_cast<float>(b) / static_cast<float>(vocab_));
    x[23] = std::cos(6.28318530718F * static_cast<float>(b) / static_cast<float>(vocab_));
    x[24] = std::sin(6.28318530718F * static_cast<float>(c) / static_cast<float>(vocab_));
    x[25] = std::cos(6.28318530718F * static_cast<float>(c) / static_cast<float>(vocab_));
    x[26] = static_cast<float>((a % 5) == (c % 5));
    x[27] = static_cast<float>((a & 0x3) == (b & 0x3));
    x[28] = static_cast<float>((c & 0x3) == (b & 0x3));
    x[29] = static_cast<float>((d1 < 8) && (d2 < 8));
    x[30] = static_cast<float>((d1 >= 8) && (d2 >= 8));
    x[31] = static_cast<float>(wrap_token(d1 + d2, static_cast<int>(vocab_))) * inv_vocab;
    for (std::size_t i = 0; i < 8; ++i) {
        const int tok = sequence[i % sequence.size()];
        x[32 + i] = static_cast<float>(token_gate_.route(encode_token(tok)).lane == 0);
        x[40 + i] = static_cast<float>(tok >= static_cast<int>(vocab_ / 2));
    }
    return x;
}

void SparseHybridRouter::train_token_gate(int token, bool keep, float strength) {
    token_gate_.train_supervised(encode_token(token), keep ? 1 : 0, strength);
}

void SparseHybridRouter::train_span_router(
    const std::vector<int>& sequence,
    int lane,
    float strength
) {
    std::vector<int> informative_indices;
    span_router_.train_supervised(
        encode_sparse_triplet(sequence, &informative_indices),
        lane,
        strength
    );
}

HybridRouteResult SparseHybridRouter::route(const std::vector<int>& sequence) const {
    if (sequence.empty()) {
        throw std::invalid_argument("SparseHybridRouter sequence must be non-empty");
    }
    HybridRouteResult out;
    out.token_actions.resize(sequence.size(), 0);
    out.token_keep_probability.resize(sequence.size(), 0.0F);

    std::vector<int> kept_indices;
    for (std::size_t i = 0; i < sequence.size(); ++i) {
        const auto gate = token_gate_.route(encode_token(sequence[i]));
        const float keep_prob = gate.probabilities.size() > 1 ? gate.probabilities[1] : 0.0F;
        out.token_keep_probability[i] = keep_prob;
        out.token_actions[i] = (keep_prob >= 0.5F) ? 1 : 0;
        if (keep_prob >= 0.5F && kept_indices.size() < 3) {
            kept_indices.push_back(static_cast<int>(i));
        }
    }
    if (kept_indices.empty()) {
        const int fallback = static_cast<int>(sequence.size() / 2);
        kept_indices.push_back(fallback);
        out.token_actions[static_cast<std::size_t>(fallback)] = 1;
        out.token_keep_probability[static_cast<std::size_t>(fallback)] = 1.0F;
    }
    while (kept_indices.size() < 3) {
        kept_indices.push_back(kept_indices.back());
    }

    const auto lane_decision = span_router_.route(encode_sparse_triplet(sequence, nullptr));
    HybridSparseSpan span;
    span.token_indices = {kept_indices[0], kept_indices[1], kept_indices[2]};
    span.lane = lane_decision.lane;
    span.confidence = lane_decision.probabilities.empty()
        ? 0.0F
        : lane_decision.probabilities[static_cast<std::size_t>(lane_decision.lane)];
    out.spans.push_back(span);
    return out;
}

void SparseHybridRouter::save(const std::string& path) const {
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        throw std::runtime_error("SparseHybridRouter failed to open save path");
    }
    const auto write_size = [&](std::size_t v) {
        out.write(reinterpret_cast<const char*>(&v), sizeof(v));
    };
    const auto write_vec = [&](const std::vector<float>& v) {
        const std::size_t n = v.size();
        write_size(n);
        out.write(reinterpret_cast<const char*>(v.data()), sizeof(float) * n);
    };

    out.write(reinterpret_cast<const char*>(&kMagic), sizeof(kMagic));
    write_size(vocab_);
    write_size(lanes_);
    write_vec(token_gate_.weights_);
    write_vec(token_gate_.bias_);
    write_vec(span_router_.weights_);
    write_vec(span_router_.bias_);
}

SparseHybridRouter SparseHybridRouter::load(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("SparseHybridRouter failed to open load path");
    }
    const auto read_size = [&](std::size_t* v) {
        in.read(reinterpret_cast<char*>(v), sizeof(*v));
    };
    const auto read_vec = [&](std::vector<float>* v) {
        std::size_t n = 0;
        read_size(&n);
        v->resize(n);
        in.read(reinterpret_cast<char*>(v->data()), sizeof(float) * n);
    };

    std::uint32_t magic = 0;
    in.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    if (magic != kMagic) {
        throw std::runtime_error("SparseHybridRouter invalid checkpoint magic");
    }

    std::size_t vocab = 0;
    std::size_t lanes = 0;
    read_size(&vocab);
    read_size(&lanes);
    SparseHybridRouter router(vocab, lanes);
    read_vec(&router.token_gate_.weights_);
    read_vec(&router.token_gate_.bias_);
    read_vec(&router.span_router_.weights_);
    read_vec(&router.span_router_.bias_);
    return router;
}

}  // namespace ir
