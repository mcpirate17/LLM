#include "router_distilled.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace ir {
namespace {

void softmax_inplace(std::vector<float>& xs) {
    const float max_v = *std::max_element(xs.begin(), xs.end());
    float sum = 0.0F;
    for (float& x : xs) {
        x = std::exp(x - max_v);
        sum += x;
    }
    const float inv = (sum > 0.0F) ? (1.0F / sum) : 1.0F;
    for (float& x : xs) {
        x *= inv;
    }
}

}  // namespace

RouterDistilled::RouterDistilled(std::size_t lanes, std::size_t dim)
    : lanes_(lanes), dim_(dim), weights_(lanes * dim, 0.0F), bias_(lanes, 0.0F) {
    if (lanes == 0 || dim == 0) {
        throw std::invalid_argument("RouterDistilled requires non-zero lanes and dim");
    }
}

DistilledDecision RouterDistilled::route(const std::vector<float>& token) const {
    if (token.size() != dim_) {
        throw std::invalid_argument("RouterDistilled token dimension mismatch");
    }
    std::vector<float> logits(lanes_, 0.0F);
    const float scale = 1.0F / std::sqrt(static_cast<float>(dim_));
    for (std::size_t lane = 0; lane < lanes_; ++lane) {
        const float* w = weights_.data() + lane * dim_;
        logits[lane] = std::inner_product(token.begin(), token.end(), w, 0.0F) * scale + bias_[lane];
    }
    std::vector<float> probs = logits;
    softmax_inplace(probs);
    const int lane = static_cast<int>(
        std::distance(logits.begin(), std::max_element(logits.begin(), logits.end()))
    );
    return DistilledDecision{lane, probs};
}

void RouterDistilled::train_supervised(
    const std::vector<float>& token,
    int lane,
    float strength
) {
    if (token.size() != dim_ || lane < 0 || static_cast<std::size_t>(lane) >= lanes_) {
        throw std::invalid_argument("RouterDistilled supervised input out of range");
    }
    auto decision = route(token);
    const float lr = 0.08F * std::clamp(strength, 0.25F, 4.0F);
    for (std::size_t k = 0; k < lanes_; ++k) {
        const float target = (static_cast<int>(k) == lane) ? 1.0F : 0.0F;
        const float grad = target - decision.probabilities[k];
        float* w = weights_.data() + k * dim_;
        for (std::size_t j = 0; j < dim_; ++j) {
            w[j] += lr * grad * token[j];
        }
        bias_[k] += lr * grad;
    }
}

}  // namespace ir
