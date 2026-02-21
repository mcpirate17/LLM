
#include <torch/extension.h>
#include <vector>
#include <immintrin.h>
#include <algorithm>

/**
 * Fast mask building for N:M sparsity on CPU using AVX-512 (if available).
 */

#if defined(__AVX512F__)
// AVX-512 implementation for 2:4 sparsity (common case)
void build_24_mask_avx512(float* weight_ptr, float* mask_ptr, int64_t rows, int64_t cols) {
    // Process in chunks of 16 floats (4 chunks of 2:4)
    for (int64_t i = 0; i < rows * cols; i += 16) {
        // Load 16 weights
        __m512 w = _mm512_loadu_ps(weight_ptr + i);
        __m512 abs_w = _mm512_abs_ps(w);
        
        // For each group of 4, find top 2
        // This is a bit complex in pure SIMD, we use a hybrid approach for the prototype
        alignas(64) float vals[16];
        _mm512_store_ps(vals, abs_w);
        
        alignas(64) float m_vals[16] = {0};
        for (int g = 0; g < 4; ++g) {
            int base = g * 4;
            // Simple sort for 4 elements
            int idx[4] = {0, 1, 2, 3};
            if (vals[base+idx[0]] < vals[base+idx[1]]) std::swap(idx[0], idx[1]);
            if (vals[base+idx[2]] < vals[base+idx[3]]) std::swap(idx[2], idx[3]);
            if (vals[base+idx[0]] < vals[base+idx[2]]) std::swap(idx[0], idx[2]);
            if (vals[base+idx[1]] < vals[base+idx[3]]) std::swap(idx[1], idx[3]);
            if (vals[base+idx[1]] < vals[base+idx[2]]) std::swap(idx[1], idx[2]);
            
            m_vals[base + idx[0]] = 1.0f;
            m_vals[base + idx[1]] = 1.0f;
        }
        
        _mm512_storeu_ps(mask_ptr + i, _mm512_load_ps(m_vals));
    }
}
#endif

torch::Tensor build_nm_mask_cpu(torch::Tensor weight, int n, int m) {
    torch::NoGradGuard no_grad;
    auto rows = weight.size(0);
    auto cols = weight.size(1);
    auto total_elements = weight.numel();
    auto mask = torch::zeros_like(weight);
    
    float* weight_ptr = weight.data_ptr<float>();
    float* mask_ptr = mask.data_ptr<float>();
    
#if defined(__AVX512F__)
    // Only use AVX-512 if we have a full chunk of 16 elements and total elements is multiple of 16
    if (n == 2 && m == 4 && total_elements >= 16 && (total_elements % 16 == 0)) {
        build_24_mask_avx512(weight_ptr, mask_ptr, rows, cols);
        return mask;
    }
#endif

    // Fallback implementation
    for (int64_t r = 0; r < rows; ++r) {
        for (int64_t c = 0; c < cols; c += m) {
            std::vector<std::pair<float, int64_t>> chunk;
            for (int64_t i = 0; i < m && (c + i) < cols; ++i) {
                chunk.push_back({std::abs(weight_ptr[r * cols + c + i]), c + i});
            }
            std::sort(chunk.begin(), chunk.end(), std::greater<std::pair<float, int64_t>>());
            
            for (int64_t i = 0; i < n && i < chunk.size(); ++i) {
                mask_ptr[r * cols + chunk[i].second] = 1.0f;
            }
        }
    }
    
    return mask;
}

/**
 * Fast small reduction: sum of absolute values
 */
float fast_abs_sum_cpu(torch::Tensor x) {
    auto n = x.numel();
    float* ptr = x.data_ptr<float>();
    float sum = 0.0f;

#if defined(__AVX512F__)
    __m512 sum_vec = _mm512_setzero_ps();
    int64_t i = 0;
    for (; i <= n - 16; i += 16) {
        __m512 val = _mm512_loadu_ps(ptr + i);
        sum_vec = _mm512_add_ps(sum_vec, _mm512_abs_ps(val));
    }
    sum = _mm512_reduce_add_ps(sum_vec);
    for (; i < n; ++i) sum += std::abs(ptr[i]);
#else
    for (int64_t i = 0; i < n; ++i) sum += std::abs(ptr[i]);
#endif

    return sum;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("build_nm_mask_cpu", &build_nm_mask_cpu, "Build N:M mask on CPU");
    m.def("fast_abs_sum_cpu", &fast_abs_sum_cpu, "Fast absolute sum on CPU");
}
