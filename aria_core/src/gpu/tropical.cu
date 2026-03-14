#include <cuda_runtime.h>
#include <cstdint>
#include <cmath>
#include <cfloat>

__device__ inline float stable_sigmoid(float x) {
    if (x >= 0.0f) {
        float z = expf(-x);
        return 1.0f / (1.0f + z);
    }
    float z = expf(x);
    return z / (1.0f + z);
}

__global__ void cuda_tropical_matmul_batched_kernel(const float* A, const float* B, float* C, int64_t batch, int64_t M, int64_t K, int64_t N) {
    int64_t b = blockIdx.z;
    int64_t i = blockIdx.y * blockDim.y + threadIdx.y;
    int64_t j = blockIdx.x * blockDim.x + threadIdx.x;

    if (b < batch && i < M && j < N) {
        float min_val = FLT_MAX;
        for (int64_t k = 0; k < K; ++k) {
            float val = A[b * M * K + i * K + k] + B[b * K * N + k * N + j];
            if (val < min_val) {
                min_val = val;
            }
        }
        C[b * M * N + i * N + j] = min_val;
    }
}

void launch_cuda_tropical_matmul_batched_f32(const float* A, const float* B, float* C, int64_t batch, int64_t M, int64_t K, int64_t N) {
    dim3 threads(16, 16, 1);
    dim3 blocks((N + 15) / 16, (M + 15) / 16, batch);
    cuda_tropical_matmul_batched_kernel<<<blocks, threads>>>(A, B, C, batch, M, K, N);
}

__global__ void cuda_tropical_matmul_kernel(const float* A, const float* B, float* C, int64_t M, int64_t K, int64_t N) {
    int64_t i = blockIdx.y * blockDim.y + threadIdx.y;
    int64_t j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < M && j < N) {
        float min_val = FLT_MAX;
        for (int64_t k = 0; k < K; ++k) {
            float val = A[i * K + k] + B[k * N + j];
            if (val < min_val) {
                min_val = val;
            }
        }
        C[i * N + j] = min_val;
    }
}

void launch_cuda_tropical_matmul_f32(const float* A, const float* B, float* C, int64_t M, int64_t K, int64_t N) {
    dim3 threads(16, 16, 1);
    dim3 blocks((N + 15) / 16, (M + 15) / 16, 1);
    cuda_tropical_matmul_kernel<<<blocks, threads>>>(A, B, C, M, K, N);
}

__global__ void cuda_tropical_add_kernel(const float* a, const float* b, float* y, int64_t n) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        y[i] = fminf(a[i], b[i]);
    }
}

void launch_cuda_tropical_add_f32(const float* a, const float* b, float* y, int64_t n) {
    int64_t threads = 256;
    int64_t blocks = (n + threads - 1) / threads;
    cuda_tropical_add_kernel<<<blocks, threads>>>(a, b, y, n);
}

__global__ void cuda_tropical_center_kernel(const float* x, float* y, int64_t batch, int64_t seq, int64_t dim) {
    int64_t b = blockIdx.y;
    int64_t d = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= batch || d >= dim) return;
    float running_min = INFINITY;
    for (int64_t s = 0; s < seq; ++s) {
        int64_t idx = (b * seq + s) * dim + d;
        float value = x[idx];
        running_min = fminf(running_min, value);
        y[idx] = value - running_min;
    }
}

void launch_cuda_tropical_center_f32(const float* x, float* y, int64_t batch, int64_t seq, int64_t dim) {
    dim3 threads(256);
    dim3 blocks((dim + threads.x - 1) / threads.x, batch, 1);
    cuda_tropical_center_kernel<<<blocks, threads>>>(x, y, batch, seq, dim);
}

__global__ void cuda_tropical_attention_kernel(const float* x, float* y, int64_t batch, int64_t seq, int64_t dim, float temperature) {
    int64_t b = blockIdx.z;
    int64_t i = blockIdx.y * blockDim.y + threadIdx.y;
    int64_t d = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= batch || i >= seq || d >= dim) return;

    float inv_temp = 1.0f / fmaxf(temperature, 0.1f);
    float max_logit = -INFINITY;
    for (int64_t j = 0; j <= i; ++j) {
            float best = INFINITY;
        for (int64_t k = 0; k < dim; ++k) {
            float value = x[(b * seq + i) * dim + k] + x[(b * seq + j) * dim + k];
            best = fminf(best, value);
        }
        max_logit = fmaxf(max_logit, -best * inv_temp);
    }

    float denom = 0.0f;
    for (int64_t j = 0; j <= i; ++j) {
        float best = INFINITY;
        for (int64_t k = 0; k < dim; ++k) {
            float value = x[(b * seq + i) * dim + k] + x[(b * seq + j) * dim + k];
            best = fminf(best, value);
        }
        denom += expf((-best * inv_temp) - max_logit);
    }
    denom = fmaxf(denom, 1e-12f);

    float out = 0.0f;
    for (int64_t j = 0; j <= i; ++j) {
        float best = INFINITY;
        for (int64_t k = 0; k < dim; ++k) {
            float value = x[(b * seq + i) * dim + k] + x[(b * seq + j) * dim + k];
            best = fminf(best, value);
        }
        float weight = expf((-best * inv_temp) - max_logit) / denom;
        out += weight * x[(b * seq + j) * dim + d];
    }
    y[(b * seq + i) * dim + d] = out;
}

void launch_cuda_tropical_attention_f32(const float* x, float* y, int64_t batch, int64_t seq, int64_t dim, float temperature) {
    dim3 threads(16, 16, 1);
    dim3 blocks((dim + threads.x - 1) / threads.x, (seq + threads.y - 1) / threads.y, batch);
    cuda_tropical_attention_kernel<<<blocks, threads>>>(x, y, batch, seq, dim, temperature);
}

__global__ void cuda_tropical_gate_kernel(const float* x, float* y, int64_t batch, int64_t seq, int64_t dim, float temperature) {
    int64_t b = blockIdx.z;
    int64_t i = blockIdx.y * blockDim.y + threadIdx.y;
    int64_t d = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= batch || i >= seq || d >= dim) return;

    float inv_temp = 1.0f / fmaxf(temperature, 0.1f);
    float max_logit = -INFINITY;
    for (int64_t j = 0; j <= i; ++j) {
            float best = INFINITY;
        for (int64_t k = 0; k < dim; ++k) {
            float value = x[(b * seq + i) * dim + k] + x[(b * seq + j) * dim + k];
            best = fminf(best, value);
        }
        max_logit = fmaxf(max_logit, -best * inv_temp);
    }

    float denom = 0.0f;
    for (int64_t j = 0; j <= i; ++j) {
        float best = INFINITY;
        for (int64_t k = 0; k < dim; ++k) {
            float value = x[(b * seq + i) * dim + k] + x[(b * seq + j) * dim + k];
            best = fminf(best, value);
        }
        denom += expf((-best * inv_temp) - max_logit);
    }
    denom = fmaxf(denom, 1e-12f);

    float gated = 0.0f;
    for (int64_t j = 0; j <= i; ++j) {
        float best = INFINITY;
        for (int64_t k = 0; k < dim; ++k) {
            float value = x[(b * seq + i) * dim + k] + x[(b * seq + j) * dim + k];
            best = fminf(best, value);
        }
        float weight = expf((-best * inv_temp) - max_logit) / denom;
        gated += weight * x[(b * seq + j) * dim + d];
    }
    y[(b * seq + i) * dim + d] = x[(b * seq + i) * dim + d] * stable_sigmoid(gated);
}

void launch_cuda_tropical_gate_f32(const float* x, float* y, int64_t batch, int64_t seq, int64_t dim, float temperature) {
    dim3 threads(16, 16, 1);
    dim3 blocks((dim + threads.x - 1) / threads.x, (seq + threads.y - 1) / threads.y, batch);
    cuda_tropical_gate_kernel<<<blocks, threads>>>(x, y, batch, seq, dim, temperature);
}
