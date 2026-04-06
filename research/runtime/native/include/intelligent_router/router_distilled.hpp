#ifndef RESEARCH_RUNTIME_NATIVE_INTELLIGENT_ROUTER_ROUTER_DISTILLED_HPP_
#define RESEARCH_RUNTIME_NATIVE_INTELLIGENT_ROUTER_ROUTER_DISTILLED_HPP_

#include <cstddef>
#include <vector>

namespace ir {

class SparseHybridRouter;

struct DistilledDecision {
    int lane = 0;
    std::vector<float> probabilities;
};

class RouterDistilled {
  public:
    RouterDistilled(std::size_t lanes, std::size_t dim);

    void train_supervised(const std::vector<float>& token, int lane, float strength = 1.0F);
    DistilledDecision route(const std::vector<float>& token) const;

    std::size_t lanes() const noexcept { return lanes_; }
    std::size_t dim() const noexcept { return dim_; }

  private:
    friend class SparseHybridRouter;
    std::size_t lanes_;
    std::size_t dim_;
    std::vector<float> weights_;
    std::vector<float> bias_;
};

}  // namespace ir

#endif
