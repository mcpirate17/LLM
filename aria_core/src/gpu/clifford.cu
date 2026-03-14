#include <cuda_runtime.h>
#include <cstdint>
#include <cmath>

__device__ inline void gp_cl30_single(const float* a, const float* b, float* y) {
    y[0] = a[0]*b[0] + a[1]*b[1] + a[2]*b[2] + a[3]*b[3] - a[4]*b[4] - a[5]*b[5] - a[6]*b[6] - a[7]*b[7];
    y[1] = a[0]*b[1] + a[1]*b[0] - a[2]*b[4] + a[3]*b[6] + a[4]*b[2] - a[5]*b[7] - a[6]*b[3] + a[7]*b[5];
    y[2] = a[0]*b[2] + a[1]*b[4] + a[2]*b[0] - a[3]*b[5] - a[4]*b[1] + a[5]*b[3] - a[6]*b[7] + a[7]*b[6];
    y[3] = a[0]*b[3] - a[1]*b[6] + a[2]*b[5] + a[3]*b[0] + a[4]*b[7] - a[5]*b[2] + a[6]*b[1] + a[7]*b[4];
    y[4] = a[0]*b[4] + a[1]*b[2] - a[2]*b[1] + a[3]*b[7] + a[4]*b[0] - a[5]*b[6] + a[6]*b[5] + a[7]*b[3];
    y[5] = a[0]*b[5] + a[1]*b[7] + a[2]*b[3] - a[3]*b[2] + a[4]*b[6] + a[5]*b[0] - a[6]*b[4] + a[7]*b[1];
    y[6] = a[0]*b[6] - a[1]*b[3] + a[2]*b[7] + a[3]*b[1] - a[4]*b[5] + a[5]*b[4] + a[6]*b[0] + a[7]*b[2];
    y[7] = a[0]*b[7] + a[1]*b[5] + a[2]*b[6] + a[3]*b[4] + a[4]*b[3] + a[5]*b[1] + a[6]*b[2] + a[7]*b[0];
}

__global__ void cuda_clifford_geometric_product_cl30_kernel(const float* a, const float* b, float* y, int64_t num_elements) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t i = idx * 8;
    if (idx < num_elements) {
        // Scalar
        y[i+0] = a[i+0]*b[i+0] + a[i+1]*b[i+1] + a[i+2]*b[i+2] + a[i+3]*b[i+3] - a[i+4]*b[i+4] - a[i+5]*b[i+5] - a[i+6]*b[i+6] - a[i+7]*b[i+7];
        // Vectors
        y[i+1] = a[i+0]*b[i+1] + a[i+1]*b[i+0] - a[i+2]*b[i+4] + a[i+3]*b[i+6] + a[i+4]*b[i+2] - a[i+5]*b[i+7] - a[i+6]*b[i+3] + a[i+7]*b[i+5];
        y[i+2] = a[i+0]*b[i+2] + a[i+1]*b[i+4] + a[i+2]*b[i+0] - a[i+3]*b[i+5] - a[i+4]*b[i+1] + a[i+5]*b[i+3] - a[i+6]*b[i+7] + a[i+7]*b[i+6];
        y[i+3] = a[i+0]*b[i+3] - a[i+1]*b[i+6] + a[i+2]*b[i+5] + a[i+3]*b[i+0] + a[i+4]*b[i+7] - a[i+5]*b[i+2] + a[i+6]*b[i+1] + a[i+7]*b[i+4];
        // Bivectors
        y[i+4] = a[i+0]*b[i+4] + a[i+1]*b[i+2] - a[i+2]*b[i+1] + a[i+3]*b[i+7] + a[i+4]*b[i+0] - a[i+5]*b[i+6] + a[i+6]*b[i+5] + a[i+7]*b[i+3];
        y[i+5] = a[i+0]*b[i+5] + a[i+1]*b[i+7] + a[i+2]*b[i+3] - a[i+3]*b[i+2] + a[i+4]*b[i+6] + a[i+5]*b[i+0] - a[i+6]*b[i+4] + a[i+7]*b[i+1];
        y[i+6] = a[i+0]*b[i+6] - a[i+1]*b[i+3] + a[i+2]*b[i+7] + a[i+3]*b[i+1] - a[i+4]*b[i+5] + a[i+5]*b[i+4] + a[i+6]*b[i+0] + a[i+7]*b[i+2];
        // Trivector
        y[i+7] = a[i+0]*b[i+7] + a[i+1]*b[i+5] + a[i+2]*b[i+6] + a[i+3]*b[i+4] + a[i+4]*b[i+3] + a[i+5]*b[i+1] + a[i+6]*b[i+2] + a[i+7]*b[i+0];
    }
}

void launch_cuda_clifford_geometric_product_cl30_f32(const float* a, const float* b, float* y, int64_t n) {
    int64_t num_elements = n / 8;
    int64_t threads = 256;
    int64_t blocks = (num_elements + threads - 1) / threads;
    cuda_clifford_geometric_product_cl30_kernel<<<blocks, threads>>>(a, b, y, num_elements);
}

__global__ void cuda_clifford_attention_kernel(const float* x, float* y, int64_t batch, int64_t seq, int64_t dim) {
    int64_t b = blockIdx.z;
    int64_t i = blockIdx.y * blockDim.y + threadIdx.y;
    int64_t d = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= batch || i >= seq || d >= dim) return;

    int64_t n_mv = dim / 8;
    float max_score = -INFINITY;
    for (int64_t j = 0; j <= i; ++j) {
        float total_mag = 0.0f;
        for (int64_t m = 0; m < n_mv; ++m) {
            float tmp[8];
            gp_cl30_single(x + (b * seq + i) * dim + m * 8, x + (b * seq + j) * dim + m * 8, tmp);
            float mag2 = 0.0f;
            for (int k = 0; k < 8; ++k) mag2 += tmp[k] * tmp[k];
            total_mag += sqrtf(mag2);
        }
        max_score = fmaxf(max_score, total_mag);
    }

    float scale = sqrtf(fmaxf(static_cast<float>(dim), 1.0f));
    float denom = 0.0f;
    for (int64_t j = 0; j <= i; ++j) {
        float total_mag = 0.0f;
        for (int64_t m = 0; m < n_mv; ++m) {
            float tmp[8];
            gp_cl30_single(x + (b * seq + i) * dim + m * 8, x + (b * seq + j) * dim + m * 8, tmp);
            float mag2 = 0.0f;
            for (int k = 0; k < 8; ++k) mag2 += tmp[k] * tmp[k];
            total_mag += sqrtf(mag2);
        }
        denom += expf((total_mag - max_score) / scale);
    }
    denom = fmaxf(denom, 1e-7f);

    float out = 0.0f;
    for (int64_t j = 0; j <= i; ++j) {
        float total_mag = 0.0f;
        for (int64_t m = 0; m < n_mv; ++m) {
            float tmp[8];
            gp_cl30_single(x + (b * seq + i) * dim + m * 8, x + (b * seq + j) * dim + m * 8, tmp);
            float mag2 = 0.0f;
            for (int k = 0; k < 8; ++k) mag2 += tmp[k] * tmp[k];
            total_mag += sqrtf(mag2);
        }
        float weight = expf((total_mag - max_score) / scale) / denom;
        out += weight * x[(b * seq + j) * dim + d];
    }
    y[(b * seq + i) * dim + d] = out;
}

void launch_cuda_clifford_attention_f32(const float* x, float* y, int64_t batch, int64_t seq, int64_t dim) {
    dim3 threads(16, 16, 1);
    dim3 blocks((dim + threads.x - 1) / threads.x, (seq + threads.y - 1) / threads.y, batch);
    cuda_clifford_attention_kernel<<<blocks, threads>>>(x, y, batch, seq, dim);
}
