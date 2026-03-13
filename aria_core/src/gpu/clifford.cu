#include <cuda_runtime.h>
#include <cstdint>

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
