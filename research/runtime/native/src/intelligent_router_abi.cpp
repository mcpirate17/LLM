#include "intelligent_router_abi.h"

#include <algorithm>
#include <cstddef>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "intelligent_router/sparse_hybrid_router.hpp"

namespace {

std::mutex& router_mutex() {
    static std::mutex mtx;
    return mtx;
}

std::unordered_map<int64_t, std::unique_ptr<ir::SparseHybridRouter>>& router_registry() {
    static std::unordered_map<int64_t, std::unique_ptr<ir::SparseHybridRouter>> registry;
    return registry;
}

int64_t& next_handle() {
    static int64_t value = 1;
    return value;
}

thread_local std::string g_last_error;

void clear_error() {
    g_last_error.clear();
}

aria_irouter_status_t fail(aria_irouter_status_t status, const std::string& message) {
    g_last_error = message;
    return status;
}

std::vector<int> copy_sequence(const int32_t* sequence, int32_t seq_len) {
    if (sequence == nullptr || seq_len <= 0) {
        throw std::invalid_argument("sequence must be non-null and non-empty");
    }
    std::vector<int> out(static_cast<std::size_t>(seq_len));
    for (int32_t i = 0; i < seq_len; ++i) {
        out[static_cast<std::size_t>(i)] = static_cast<int>(sequence[i]);
    }
    return out;
}

template <typename Fn>
aria_irouter_status_t with_router(int64_t handle, Fn&& fn) {
    if (handle <= 0) {
        return fail(ARIA_IROUTER_ERR_INVALID_ARGUMENT, "router handle must be positive");
    }
    try {
        clear_error();
        std::lock_guard<std::mutex> lock(router_mutex());
        auto it = router_registry().find(handle);
        if (it == router_registry().end()) {
            return fail(ARIA_IROUTER_ERR_NOT_FOUND, "router handle not found");
        }
        fn(*it->second);
        return ARIA_IROUTER_OK;
    } catch (const std::invalid_argument& exc) {
        return fail(ARIA_IROUTER_ERR_INVALID_ARGUMENT, exc.what());
    } catch (const std::exception& exc) {
        return fail(ARIA_IROUTER_ERR_INTERNAL, exc.what());
    } catch (...) {
        return fail(ARIA_IROUTER_ERR_INTERNAL, "unknown intelligent router error");
    }
}

}  // namespace

extern "C" {

aria_irouter_status_t aria_irouter_create(
    int32_t vocab,
    int32_t lanes,
    int64_t* out_handle) {
    if (out_handle == nullptr) {
        return fail(ARIA_IROUTER_ERR_INVALID_ARGUMENT, "out_handle must be non-null");
    }
    if (vocab <= 0 || lanes <= 0) {
        return fail(ARIA_IROUTER_ERR_INVALID_ARGUMENT, "vocab and lanes must be positive");
    }
    try {
        clear_error();
        auto router = std::make_unique<ir::SparseHybridRouter>(
            static_cast<std::size_t>(vocab),
            static_cast<std::size_t>(lanes));
        std::lock_guard<std::mutex> lock(router_mutex());
        const int64_t handle = next_handle()++;
        router_registry().emplace(handle, std::move(router));
        *out_handle = handle;
        return ARIA_IROUTER_OK;
    } catch (const std::invalid_argument& exc) {
        return fail(ARIA_IROUTER_ERR_INVALID_ARGUMENT, exc.what());
    } catch (const std::exception& exc) {
        return fail(ARIA_IROUTER_ERR_INTERNAL, exc.what());
    } catch (...) {
        return fail(ARIA_IROUTER_ERR_INTERNAL, "unknown intelligent router error");
    }
}

aria_irouter_status_t aria_irouter_destroy(int64_t handle) {
    if (handle <= 0) {
        return fail(ARIA_IROUTER_ERR_INVALID_ARGUMENT, "router handle must be positive");
    }
    clear_error();
    std::lock_guard<std::mutex> lock(router_mutex());
    const auto erased = router_registry().erase(handle);
    if (erased == 0U) {
        return fail(ARIA_IROUTER_ERR_NOT_FOUND, "router handle not found");
    }
    return ARIA_IROUTER_OK;
}

aria_irouter_status_t aria_irouter_train_token_gate(
    int64_t handle,
    int32_t token,
    int32_t keep,
    float strength) {
    return with_router(handle, [&](ir::SparseHybridRouter& router) {
        router.train_token_gate(static_cast<int>(token), keep != 0, strength);
    });
}

aria_irouter_status_t aria_irouter_train_span_router(
    int64_t handle,
    const int32_t* sequence,
    int32_t seq_len,
    int32_t lane,
    float strength) {
    return with_router(handle, [&](ir::SparseHybridRouter& router) {
        router.train_span_router(copy_sequence(sequence, seq_len), static_cast<int>(lane), strength);
    });
}

aria_irouter_status_t aria_irouter_route(
    int64_t handle,
    const int32_t* sequence,
    int32_t seq_len,
    int32_t* token_actions_out,
    float* token_keep_probability_out,
    int32_t* span_token_indices_out,
    int32_t span_token_indices_capacity,
    int32_t* span_lanes_out,
    float* span_confidences_out,
    aria_irouter_route_meta_t* out_meta) {
    if (token_actions_out == nullptr || token_keep_probability_out == nullptr || out_meta == nullptr) {
        return fail(
            ARIA_IROUTER_ERR_INVALID_ARGUMENT,
            "route outputs token_actions_out, token_keep_probability_out, and out_meta must be non-null");
    }
    return with_router(handle, [&](ir::SparseHybridRouter& router) {
        const ir::HybridRouteResult result = router.route(copy_sequence(sequence, seq_len));
        const std::size_t n_tokens = result.token_actions.size();
        for (std::size_t i = 0; i < n_tokens; ++i) {
            token_actions_out[i] = static_cast<int32_t>(result.token_actions[i]);
            token_keep_probability_out[i] = result.token_keep_probability[i];
        }

        const int32_t span_count = static_cast<int32_t>(result.spans.size());
        out_meta->span_count = span_count;
        out_meta->required_span_capacity = span_count * 3;
        if (span_count == 0) {
            return;
        }
        if (span_lanes_out == nullptr || span_confidences_out == nullptr) {
            throw std::invalid_argument("span lanes and confidences outputs must be non-null when spans exist");
        }
        if (span_token_indices_out == nullptr || span_token_indices_capacity < (span_count * 3)) {
            throw std::invalid_argument("span token output capacity is too small");
        }
        for (int32_t span_idx = 0; span_idx < span_count; ++span_idx) {
            const auto& span = result.spans[static_cast<std::size_t>(span_idx)];
            span_lanes_out[span_idx] = static_cast<int32_t>(span.lane);
            span_confidences_out[span_idx] = span.confidence;
            const std::size_t n_indices = std::min<std::size_t>(3, span.token_indices.size());
            for (std::size_t j = 0; j < n_indices; ++j) {
                span_token_indices_out[static_cast<std::size_t>(span_idx) * 3U + j] =
                    static_cast<int32_t>(span.token_indices[j]);
            }
        }
    });
}

aria_irouter_status_t aria_irouter_save(int64_t handle, const char* path) {
    if (path == nullptr || path[0] == '\0') {
        return fail(ARIA_IROUTER_ERR_INVALID_ARGUMENT, "save path must be non-empty");
    }
    return with_router(handle, [&](ir::SparseHybridRouter& router) {
        router.save(path);
    });
}

aria_irouter_status_t aria_irouter_load(const char* path, int64_t* out_handle) {
    if (path == nullptr || path[0] == '\0' || out_handle == nullptr) {
        return fail(ARIA_IROUTER_ERR_INVALID_ARGUMENT, "load path and out_handle must be non-null");
    }
    try {
        clear_error();
        auto router = std::make_unique<ir::SparseHybridRouter>(ir::SparseHybridRouter::load(path));
        std::lock_guard<std::mutex> lock(router_mutex());
        const int64_t handle = next_handle()++;
        router_registry().emplace(handle, std::move(router));
        *out_handle = handle;
        return ARIA_IROUTER_OK;
    } catch (const std::invalid_argument& exc) {
        return fail(ARIA_IROUTER_ERR_INVALID_ARGUMENT, exc.what());
    } catch (const std::exception& exc) {
        return fail(ARIA_IROUTER_ERR_INTERNAL, exc.what());
    } catch (...) {
        return fail(ARIA_IROUTER_ERR_INTERNAL, "unknown intelligent router error");
    }
}

const char* aria_irouter_last_error(void) {
    return g_last_error.c_str();
}

}  // extern "C"
