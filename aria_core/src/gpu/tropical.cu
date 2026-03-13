#include <cuda_runtime.h>
#include <cstdint>
#include <cmath>
#include <cfloat>

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
