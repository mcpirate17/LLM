"""Decisive CPU-vs-CUDA benchmark for the surprise-memory scan forward.

The native surprise-memory recurrence is an ``M x M`` associative-memory matrix
updated sequentially over ``L`` timesteps; each step is O(M^2) parallel work but
the steps are strictly serial, and the only batch-level parallelism is over ``B``.
This script ports the *plain* forward scan (``forward_one`` in
``native_surprise_memory.cpp``) to a CUDA block-per-sequence kernel, validates it
against the existing C++ CPU extension, and times both.

Decision rule (user directive): if CUDA is not faster than CPU at the run's shape,
the CUDA path is worthless for this model -> delete it, keep CPU.

Run:  python -m research.tools.bench_surprise_scan_cuda
"""

from __future__ import annotations

import time

import torch
from torch.utils.cpp_extension import load_inline

from component_fab.generator.native_surprise_memory import _native_ext

_CUDA_SRC = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

// One block per batch element. blockDim.x == M*M, one thread per memory entry
// (i, j). Memory + surprise live in shared memory and persist across timesteps;
// q/k/v/forget for the current step are staged into shared each iteration.
template <typename scalar_t>
__global__ void surprise_forward_kernel(
    const scalar_t* __restrict__ q,
    const scalar_t* __restrict__ k,
    const scalar_t* __restrict__ v,
    const scalar_t* __restrict__ write,
    const scalar_t* __restrict__ forget,
    const scalar_t momentum,
    scalar_t* __restrict__ y,
    int64_t L,
    int64_t M) {
  extern __shared__ float sh[];
  float* mem = sh;                 // M*M
  float* sur = mem + M * M;        // M*M
  float* qv = sur + M * M;         // M
  float* kv = qv + M;              // M
  float* vv = kv + M;              // M
  float* fv = vv + M;              // M
  float* pred = fv + M;            // M
  float* err = pred + M;           // M

  const int64_t b = blockIdx.x;
  const int tid = threadIdx.x;
  const int i = tid / M;
  const int j = tid % M;
  const float scale = rsqrtf((float)M);

  mem[tid] = 0.0f;
  sur[tid] = 0.0f;
  __syncthreads();

  const scalar_t* qb = q + b * L * M;
  const scalar_t* kb = k + b * L * M;
  const scalar_t* vb = v + b * L * M;
  const scalar_t* fb = forget + b * L * M;
  const scalar_t* wb = write + b * L;
  scalar_t* yb = y + b * L * M;

  for (int64_t t = 0; t < L; ++t) {
    if (tid < M) {
      qv[tid] = (float)qb[t * M + tid];
      kv[tid] = (float)kb[t * M + tid];
      vv[tid] = (float)vb[t * M + tid];
      fv[tid] = (float)fb[t * M + tid];
    }
    __syncthreads();

    // Column reductions: one thread per column j (the i==0 row of threads).
    if (i == 0) {
      float q_sum = 0.0f, k_sum = 0.0f;
      for (int ii = 0; ii < M; ++ii) {
        float m_ij = mem[ii * M + j];
        q_sum += qv[ii] * m_ij;
        k_sum += kv[ii] * m_ij;
      }
      yb[t * M + j] = (scalar_t)q_sum;
      pred[j] = k_sum;
      err[j] = vv[j] - k_sum;
    }
    __syncthreads();

    const float w = (float)wb[t];
    const float decay = 1.0f - fv[i];
    const float delta = kv[i] * err[j] * scale;
    const float s_new = momentum * sur[tid] + w * delta;
    sur[tid] = s_new;
    mem[tid] = decay * mem[tid] + s_new;
    __syncthreads();
  }
}

torch::Tensor surprise_forward_cuda(
    torch::Tensor q, torch::Tensor k, torch::Tensor v,
    torch::Tensor write, torch::Tensor forget, double momentum) {
  const int64_t B = q.size(0), L = q.size(1), M = q.size(2);
  auto y = torch::empty_like(q);
  const int threads = (int)(M * M);
  const int shared = (int)((2 * M * M + 6 * M) * sizeof(float));
  surprise_forward_kernel<float><<<(int)B, threads, shared>>>(
      q.data_ptr<float>(), k.data_ptr<float>(), v.data_ptr<float>(),
      write.data_ptr<float>(), forget.data_ptr<float>(), (float)momentum,
      y.data_ptr<float>(), L, M);
  return y;
}
"""

_CPP_DECL = "torch::Tensor surprise_forward_cuda(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, double);"


def _build():
    return load_inline(
        name="surprise_scan_cuda_bench",
        cpp_sources=[_CPP_DECL],
        cuda_sources=[_CUDA_SRC],
        functions=["surprise_forward_cuda"],
        verbose=False,
    )


def _inputs(B: int, L: int, M: int, device: str):
    g = torch.Generator(device="cpu").manual_seed(0)
    q = torch.randn(B, L, M, generator=g)
    k = torch.randn(B, L, M, generator=g)
    v = torch.randn(B, L, M, generator=g)
    write = torch.sigmoid(torch.randn(B, L, generator=g))
    forget = torch.sigmoid(torch.randn(B, L, M, generator=g))
    return [t.to(device) for t in (q, k, v, write, forget)]


def _time(fn, *, iters: int, cuda: bool) -> float:
    fn()  # warm
    if cuda:
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    if cuda:
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters


def main() -> None:
    ext = _build()
    cpu_ext = _native_ext()
    momentum = 0.5
    M = 32
    L = 256
    iters = 20

    # numerical match at the run's shape
    q, k, v, w, f = _inputs(8, L, M, "cpu")
    y_cpu = cpu_ext.forward(q, k, v, w, f, torch.tensor(momentum))[0]
    qg, kg, vg, wg, fg = (t.cuda() for t in (q, k, v, w, f))
    y_cuda = ext.surprise_forward_cuda(qg, kg, vg, wg, fg, momentum).cpu()
    max_abs = (y_cpu - y_cuda).abs().max().item()
    rel = max_abs / (y_cpu.abs().max().item() + 1e-9)
    print(
        f"numerical: max_abs_diff={max_abs:.3e} rel={rel:.3e}  -> {'MATCH' if rel < 1e-3 else 'MISMATCH'}"
    )

    print(f"\n{'B':>5} {'CPU ms':>10} {'CUDA ms':>10} {'speedup':>9}")
    for B in (16, 64, 256, 1024):
        qc, kc, vc, wc, fc = _inputs(B, L, M, "cpu")
        mom_t = torch.tensor(momentum)
        cpu_ms = (
            _time(
                lambda: cpu_ext.forward(qc, kc, vc, wc, fc, mom_t),
                iters=iters,
                cuda=False,
            )
            * 1e3
        )
        qg, kg, vg, wg, fg = (t.cuda() for t in (qc, kc, vc, wc, fc))
        cuda_ms = (
            _time(
                lambda: ext.surprise_forward_cuda(qg, kg, vg, wg, fg, momentum),
                iters=iters,
                cuda=True,
            )
            * 1e3
        )
        print(f"{B:>5} {cpu_ms:>10.3f} {cuda_ms:>10.3f} {cpu_ms / cuda_ms:>8.2f}x")

    print("\nNote: run shape is B=16. CUDA must beat CPU at B=16 to be worth keeping.")


if __name__ == "__main__":
    main()
