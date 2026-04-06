#ifndef ARIA_DESIGNER_ROUTING_HYBRID_SPARSE_ROUTER_PROTOTYPE_SPARSE_HYBRID_ROUTER_HPP_
#define ARIA_DESIGNER_ROUTING_HYBRID_SPARSE_ROUTER_PROTOTYPE_SPARSE_HYBRID_ROUTER_HPP_

#include <cstddef>
#include <string>
#include <vector>

#include "router_distilled.hpp"

namespace ir {

struct HybridSparseSpan {
    std::vector<int> token_indices;
    int lane = 0;
    float confidence = 0.0F;
};

struct HybridRouteResult {
    std::vector<int> token_actions;
    std::vector<float> token_keep_probability;
    std::vector<HybridSparseSpan> spans;
};

class SparseHybridRouter {
  public:
    SparseHybridRouter(std::size_t vocab, std::size_t lanes);

    void train_token_gate(int token, bool keep, float strength = 1.0F);
    void train_span_router(const std::vector<int>& sequence, int lane, float strength = 1.0F);
    HybridRouteResult route(const std::vector<int>& sequence) const;
    void save(const std::string& path) const;
    static SparseHybridRouter load(const std::string& path);

    std::size_t vocab() const noexcept { return vocab_; }
    std::size_t lanes() const noexcept { return lanes_; }

  private:
    std::vector<float> encode_token(int token) const;
    std::vector<float> encode_sparse_triplet(
        const std::vector<int>& sequence,
        std::vector<int>* informative_indices
    ) const;

    std::size_t vocab_;
    std::size_t lanes_;
    RouterDistilled token_gate_;
    RouterDistilled span_router_;
};

}  // namespace ir

#endif
