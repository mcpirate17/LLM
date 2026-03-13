/**
 * fingerprint_metrics.cpp — Native kernels for behavioral fingerprint metrics.
 *
 * Replaces the Python/Torch fallbacks for interaction_metrics and
 * sensitivity_metrics in research/eval/fingerprint.py.
 *
 * All functions take flat float arrays with explicit dimensions.
 */

#ifndef ARIA_KERNELS_COMMON_H
#include "kernels_common.h"
#endif

#include <math.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Interaction metrics from an influence matrix.
 *
 * influence: [n_pos, seq_len] — perturbation influence matrix
 * positions: [n_pos] — token position indices (int64)
 * out: [4] — locality, sparsity, symmetry, hierarchy
 */
void aria_interaction_metrics_f32(
    const float *influence, const int64_t *positions,
    float *out, int64_t n_pos, int64_t seq_len
) {
    /* Locality: 1 - mean(weighted_distance / seq_len) */
    float locality_sum = 0.0f;
    int64_t locality_n = 0;
    for (int64_t i = 0; i < n_pos; i++) {
        float row_sum = 0.0f;
        float dist_sum = 0.0f;
        const float *row = influence + i * seq_len;
        int64_t pos = positions[i];
        for (int64_t j = 0; j < seq_len; j++) {
            float w = row[j];
            row_sum += w;
            float d = (float)(j > pos ? j - pos : pos - j);
            dist_sum += w * d;
        }
        if (row_sum > 1e-8f) {
            locality_sum += 1.0f - (dist_sum / row_sum) / (float)seq_len;
            locality_n++;
        }
    }
    out[0] = locality_n > 0 ? locality_sum / (float)locality_n : 0.5f;

    /* Sparsity: 1 - entropy / log(n) */
    float total = 0.0f;
    int64_t total_n = n_pos * seq_len;
    for (int64_t i = 0; i < total_n; i++) total += influence[i];
    float entropy = 0.0f;
    if (total > 1e-8f) {
        float inv_total = 1.0f / total;
        for (int64_t i = 0; i < total_n; i++) {
            float p = influence[i] * inv_total;
            if (p > 1e-10f) entropy -= p * logf(p);
        }
    }
    float max_entropy = logf((float)(total_n > 1 ? total_n : 2));
    out[1] = max_entropy > 1e-8f ? 1.0f - entropy / max_entropy : 0.5f;

    /* Symmetry: 1 - ||upper - lower^T|| / ||upper|| */
    int64_t sq = n_pos < seq_len ? n_pos : seq_len;
    float sym_diff_sq = 0.0f, upper_sq = 0.0f;
    if (sq >= 2) {
        for (int64_t i = 0; i < sq; i++) {
            for (int64_t j = i + 1; j < sq; j++) {
                float u = influence[i * seq_len + j];
                float l = influence[j * seq_len + i];
                float d = u - l;
                sym_diff_sq += d * d;
                upper_sq += u * u;
            }
        }
    }
    out[2] = upper_sq > 1e-8f ? 1.0f - sqrtf(sym_diff_sq) / sqrtf(upper_sq) : 0.5f;

    /* Hierarchy: var(coarse) / var(fine) using pooling factor 4 */
    float fine_mean = 0.0f, fine_var = 0.0f;
    for (int64_t i = 0; i < total_n; i++) fine_mean += influence[i];
    fine_mean /= (float)(total_n > 0 ? total_n : 1);
    for (int64_t i = 0; i < total_n; i++) {
        float d = influence[i] - fine_mean;
        fine_var += d * d;
    }
    fine_var /= (float)(total_n > 0 ? total_n : 1);

    int64_t pool = 4;
    int64_t coarse_cols = seq_len / pool;
    if (coarse_cols < 1) coarse_cols = 1;
    int64_t coarse_n = n_pos * coarse_cols;

    /* Stack-allocate coarse buffer for typical sizes, heap for large */
    float coarse_stack[256];
    float *coarse = coarse_n <= 256 ? coarse_stack : (float *)malloc((size_t)coarse_n * sizeof(float));

    for (int64_t i = 0; i < n_pos; i++) {
        const float *row = influence + i * seq_len;
        for (int64_t c = 0; c < coarse_cols; c++) {
            float s = 0.0f;
            int64_t start = c * pool;
            int64_t end = start + pool;
            if (end > seq_len) end = seq_len;
            for (int64_t j = start; j < end; j++) s += row[j];
            coarse[i * coarse_cols + c] = s / (float)(end - start);
        }
    }

    float coarse_mean = 0.0f, coarse_var = 0.0f;
    for (int64_t i = 0; i < coarse_n; i++) coarse_mean += coarse[i];
    coarse_mean /= (float)(coarse_n > 0 ? coarse_n : 1);
    for (int64_t i = 0; i < coarse_n; i++) {
        float d = coarse[i] - coarse_mean;
        coarse_var += d * d;
    }
    coarse_var /= (float)(coarse_n > 0 ? coarse_n : 1);

    if (coarse != coarse_stack) free(coarse);
    out[3] = fine_var > 1e-10f ? fminf(1.0f, coarse_var / fine_var) : 0.5f;
}


/**
 * Sensitivity metrics from a Jacobian sensitivity matrix.
 *
 * sens: [n_pos, seq_len] — per-position sensitivity norms
 * out: [3] — spectral_norm, uniformity, effective_rank
 */
void aria_sensitivity_metrics_f32(
    const float *sens, float *out, int64_t n_pos, int64_t seq_len
) {
    /* Spectral norm: Frobenius norm of the matrix */
    float frob_sq = 0.0f;
    int64_t total = n_pos * seq_len;
    for (int64_t i = 0; i < total; i++) frob_sq += sens[i] * sens[i];
    out[0] = sqrtf(frob_sq);

    /* Per-position sensitivity (sum over positions dim) */
    float *per_pos = NULL;
    float per_pos_stack[1024];
    if (seq_len <= 1024) {
        per_pos = per_pos_stack;
    } else {
        per_pos = (float *)malloc((size_t)seq_len * sizeof(float));
    }

    for (int64_t j = 0; j < seq_len; j++) per_pos[j] = 0.0f;
    for (int64_t i = 0; i < n_pos; i++) {
        const float *row = sens + i * seq_len;
        for (int64_t j = 0; j < seq_len; j++) per_pos[j] += row[j];
    }

    float pos_total = 0.0f;
    for (int64_t j = 0; j < seq_len; j++) pos_total += per_pos[j];

    if (pos_total <= 1e-8f) {
        out[1] = 0.0f;
        out[2] = 0.0f;
        if (per_pos != per_pos_stack) free(per_pos);
        return;
    }

    /* Uniformity: normalized entropy of per-position sensitivity */
    float entropy = 0.0f;
    float inv_total = 1.0f / pos_total;
    for (int64_t j = 0; j < seq_len; j++) {
        float p = per_pos[j] * inv_total;
        if (p > 1e-10f) entropy -= p * logf(p);
    }
    float max_ent = logf((float)(seq_len > 1 ? seq_len : 2));
    out[1] = max_ent > 1e-8f ? entropy / max_ent : 0.0f;

    /* Effective rank: exp(entropy) */
    out[2] = expf(entropy);

    if (per_pos != per_pos_stack) free(per_pos);
}

#ifdef __cplusplus
}
#endif
