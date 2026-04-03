/**
 * bind_common.h — Shared macros, includes, and helpers for pybind11 bindings.
 */
#pragma once

#include <torch/extension.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <unordered_set>
#include "kernels.h"
#include "clifford.h"
#include "hyperbolic.h"
#include "graph_validator.h"
#include "shape_inference.h"
#include "smoke_test.h"

#define CHECK_CPU(x) TORCH_CHECK(!x.is_cuda(), #x " must be a CPU tensor")
#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_F32(x) TORCH_CHECK(x.dtype() == torch::kFloat32, #x " must be float32")
#define CHECK_F16(x) TORCH_CHECK(x.dtype() == torch::kFloat16, #x " must be float16")
#define CHECK_INPUT(x) do { CHECK_CPU(x); CHECK_CONTIGUOUS(x); CHECK_F32(x); } while(0)
#define CHECK_INPUT_ANY(x) do { CHECK_CONTIGUOUS(x); CHECK_F32(x); } while(0)
#define CHECK_INPUT_F16(x) do { CHECK_CPU(x); CHECK_CONTIGUOUS(x); CHECK_F16(x); } while(0)
#define CHECK_I64(x) TORCH_CHECK(x.dtype() == torch::kInt64, #x " must be int64")

// CUDA kernel forward declarations
void launch_cuda_tropical_matmul_f32(const float* A, const float* B, float* C, int64_t M, int64_t K, int64_t N);
void launch_cuda_tropical_matmul_batched_f32(const float* A, const float* B, float* C, int64_t batch, int64_t M, int64_t K, int64_t N);
void launch_cuda_tropical_add_f32(const float* a, const float* b, float* y, int64_t n);
void launch_cuda_tropical_center_f32(const float* x, float* y, int64_t batch, int64_t seq, int64_t dim);
void launch_cuda_tropical_attention_f32(const float* x, float* y, int64_t batch, int64_t seq, int64_t dim, float temperature);
void launch_cuda_tropical_gate_f32(const float* x, float* y, int64_t batch, int64_t seq, int64_t dim, float temperature);
void launch_cuda_clifford_geometric_product_cl30_f32(const float* a, const float* b, float* y, int64_t n);
void launch_cuda_clifford_attention_f32(const float* x, float* y, int64_t batch, int64_t seq, int64_t dim);

namespace py = pybind11;

inline const uint16_t *half_ptr_const(torch::Tensor x) {
    return reinterpret_cast<const uint16_t *>(x.data_ptr<at::Half>());
}

inline uint16_t *half_ptr(torch::Tensor x) {
    return reinterpret_cast<uint16_t *>(x.data_ptr<at::Half>());
}

// Sub-file registration functions
void bind_kernels(py::module_ &m);
void bind_ops(py::module_ &m);
void bind_graph(py::module_ &m);
