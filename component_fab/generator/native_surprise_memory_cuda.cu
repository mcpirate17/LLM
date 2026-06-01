// CUDA port of the surprise-memory scan (plain + adaptive-semiring variants).
//
// One CUDA block per sequence (batch element); blockDim = M*M threads, one
// thread per memory-matrix entry (i, j). The M x M memory + surprise state live
// in shared memory and persist across the sequential timestep loop. Math mirrors
// the CPU reference in native_surprise_memory.cpp exactly (validated against it).
//
// Only the two ops the bi-lane uses are ported: the plain delta-rule scan
// (lane_b) and the adaptive tempered-semiring scan (lane_a). The lane blend is
// a trivial gated sum done in PyTorch on-device.

#include <torch/extension.h>
#include <cuda_runtime.h>

#define MAXM 32  // M*M <= 1024 threads per block

// ----------------------------- plain forward ------------------------------
__global__ void plain_forward_kernel(
    const float* __restrict__ q, const float* __restrict__ k,
    const float* __restrict__ v, const float* __restrict__ write,
    const float* __restrict__ forget, const float momentum,
    float* __restrict__ y, float* __restrict__ mem_prev,
    float* __restrict__ surprise_prev, int64_t L, int M) {
  extern __shared__ float sh[];
  float* mem = sh;            // M*M
  float* sur = mem + M * M;   // M*M
  float* qv = sur + M * M;    // M
  float* kv = qv + M;
  float* vv = kv + M;
  float* fv = vv + M;
  float* pred = fv + M;
  float* err = pred + M;

  const int64_t b = blockIdx.x;
  const int tid = threadIdx.x;
  const int i = tid / M, j = tid % M;
  const float scale = rsqrtf((float)M);

  mem[tid] = 0.0f;
  sur[tid] = 0.0f;
  __syncthreads();

  const float* qb = q + b * L * M;
  const float* kb = k + b * L * M;
  const float* vb = v + b * L * M;
  const float* fb = forget + b * L * M;
  const float* wb = write + b * L;
  float* yb = y + b * L * M;
  float* memb = mem_prev + b * L * M * M;
  float* surb = surprise_prev + b * L * M * M;

  for (int64_t t = 0; t < L; ++t) {
    if (tid < M) {
      qv[tid] = qb[t * M + tid];
      kv[tid] = kb[t * M + tid];
      vv[tid] = vb[t * M + tid];
      fv[tid] = fb[t * M + tid];
    }
    memb[t * M * M + tid] = mem[tid];
    surb[t * M * M + tid] = sur[tid];
    __syncthreads();

    if (i == 0) {
      float qs = 0.0f, ks = 0.0f;
      for (int ii = 0; ii < M; ++ii) {
        float m = mem[ii * M + j];
        qs += qv[ii] * m;
        ks += kv[ii] * m;
      }
      yb[t * M + j] = qs;
      pred[j] = ks;
      err[j] = vv[j] - ks;
    }
    __syncthreads();

    const float w = wb[t];
    const float decay = 1.0f - fv[i];
    const float delta = kv[i] * err[j] * scale;
    const float s_new = momentum * sur[tid] + w * delta;
    sur[tid] = s_new;
    mem[tid] = decay * mem[tid] + s_new;
    __syncthreads();
  }
}

// ----------------------------- plain backward -----------------------------
__global__ void plain_backward_kernel(
    const float* __restrict__ q, const float* __restrict__ k,
    const float* __restrict__ v, const float* __restrict__ write,
    const float* __restrict__ forget, const float momentum,
    const float* __restrict__ grad_y, const float* __restrict__ mem_prev,
    const float* __restrict__ surprise_prev, float* __restrict__ grad_q,
    float* __restrict__ grad_k, float* __restrict__ grad_v,
    float* __restrict__ grad_write, float* __restrict__ grad_forget,
    float* __restrict__ grad_momentum, int64_t L, int M) {
  extern __shared__ float sh[];
  float* mem = sh;                  // M*M  (mem_prev[t])
  float* sur_t = mem + M * M;       // M*M  (surprise_prev[t])
  float* gmem = sur_t + M * M;      // M*M  carried grad_memory
  float* gmemp = gmem + M * M;      // M*M  grad_memory_prev (this t)
  float* gsur = gmemp + M * M;      // M*M  carried grad_surprise
  float* gsurp = gsur + M * M;      // M*M  grad_surprise_prev (this t)
  float* qv = gsurp + M * M;        // M
  float* kv = qv + M;
  float* vv = kv + M;
  float* fv = vv + M;
  float* gyv = fv + M;
  float* err = gyv + M;
  float* gq = err + M;
  float* gk = gq + M;
  float* gf = gk + M;
  float* gerr = gf + M;
  float* gpred = gerr + M;
  float* sc = gpred + M;            // 2 scalars: [0]=grad_w, [1]=grad_mom(block)

  const int64_t b = blockIdx.x;
  const int tid = threadIdx.x;
  const int i = tid / M, j = tid % M;
  const float scale = rsqrtf((float)M);

  gmem[tid] = 0.0f;
  gsur[tid] = 0.0f;
  if (tid == 0) sc[1] = 0.0f;  // block-local grad_momentum
  __syncthreads();

  const float* qb = q + b * L * M;
  const float* kb = k + b * L * M;
  const float* vb = v + b * L * M;
  const float* fb = forget + b * L * M;
  const float* wb = write + b * L;
  const float* gyb = grad_y + b * L * M;
  const float* memb = mem_prev + b * L * M * M;
  const float* surb = surprise_prev + b * L * M * M;
  float* gqb = grad_q + b * L * M;
  float* gkb = grad_k + b * L * M;
  float* gvb = grad_v + b * L * M;
  float* gfb = grad_forget + b * L * M;

  for (int64_t t = L - 1; t >= 0; --t) {
    mem[tid] = memb[t * M * M + tid];
    sur_t[tid] = surb[t * M * M + tid];
    gmemp[tid] = 0.0f;
    gsurp[tid] = 0.0f;
    if (tid < M) {
      qv[tid] = qb[t * M + tid];
      kv[tid] = kb[t * M + tid];
      vv[tid] = vb[t * M + tid];
      fv[tid] = fb[t * M + tid];
      gyv[tid] = gyb[t * M + tid];
      gq[tid] = 0.0f; gk[tid] = 0.0f; gf[tid] = 0.0f;
      gerr[tid] = 0.0f; gpred[tid] = 0.0f;
    }
    if (tid == 0) sc[0] = 0.0f;  // grad_w for this t
    __syncthreads();

    // pred = k @ mem_prev ; err = v - pred
    if (i == 0) {
      float ks = 0.0f;
      for (int ii = 0; ii < M; ++ii) ks += kv[ii] * mem[ii * M + j];
      err[j] = vv[j] - ks;
    }
    __syncthreads();

    // y = q @ mem_prev  -> gq[i] += gy[j]*mem[i,j]; gmemp[i,j] += gy[j]*q[i]
    atomicAdd(&gq[i], gyv[j] * mem[tid]);
    gmemp[tid] += gyv[j] * qv[i];

    // decay: gf[i] -= gmem*mem ; gmemp += gmem*decay ; gsur += gmem
    const float decay = 1.0f - fv[i];
    const float gm = gmem[tid];
    atomicAdd(&gf[i], -gm * mem[tid]);
    gmemp[tid] += gm * decay;
    gsur[tid] += gm;
    __syncthreads();

    // surprise_new = momentum*surprise_prev + w*delta
    const float w = wb[t];
    const float gs = gsur[tid];
    const float delta = kv[i] * err[j] * scale;
    atomicAdd(&sc[1], gs * sur_t[tid]);   // grad_momentum
    gsurp[tid] += gs * momentum;
    atomicAdd(&sc[0], gs * delta);        // grad_w
    const float gdelta = gs * w;
    atomicAdd(&gk[i], gdelta * err[j] * scale);
    atomicAdd(&gerr[j], gdelta * kv[i] * scale);
    __syncthreads();

    // err = v - pred
    if (tid < M) {
      gvb[t * M + tid] += gerr[tid];
      gpred[tid] = -gerr[tid];
    }
    __syncthreads();

    // pred = k @ mem_prev -> gk[i] += gpred[j]*mem[i,j]; gmemp += gpred[j]*k[i]
    atomicAdd(&gk[i], gpred[j] * mem[tid]);
    gmemp[tid] += gpred[j] * kv[i];
    __syncthreads();

    if (tid < M) {
      gqb[t * M + tid] += gq[tid];
      gkb[t * M + tid] += gk[tid];
      gfb[t * M + tid] += gf[tid];
    }
    if (tid == 0) grad_write[b * L + t] += sc[0];
    // carry: gmem <- gmemp, gsur <- gsurp
    gmem[tid] = gmemp[tid];
    gsur[tid] = gsurp[tid];
    __syncthreads();
  }
  if (tid == 0) atomicAdd(grad_momentum, sc[1]);
}

// --------------------- adaptive-semiring device helpers --------------------
__device__ __forceinline__ int adaptive_steps(
    float lvl, float lo, float hi, int maxs) {
  if (maxs <= 0 || lvl < lo) return 0;
  if (maxs == 1 || hi <= lo || lvl >= hi) return maxs < 1 ? 1 : maxs;
  float ratio = (lvl - lo) / (hi - lo);
  int s = 1 + (int)floorf(ratio * (float)(maxs - 1));
  s = s < 1 ? 1 : s;
  return s > maxs ? maxs : s;
}
__device__ __forceinline__ float bal(float raw, float balance) {
  return raw / (1.0f + balance * fabsf(raw));
}

// Column-j semiring read over shared mem[]+addr[]: out[j] = (lse - logM)/beta.
__device__ __forceinline__ float semiring_read_col(
    const float* mem, const float* addr, float beta, int j, int M, float logM) {
  float mx = -1e30f;
  for (int ii = 0; ii < M; ++ii) {
    float z = beta * (mem[ii * M + j] + addr[ii]);
    mx = z > mx ? z : mx;
  }
  float se = 0.0f;
  for (int ii = 0; ii < M; ++ii) se += __expf(beta * (mem[ii * M + j] + addr[ii]) - mx);
  return (mx + __logf(se) - logM) / beta;
}

// --------------------------- adaptive forward -----------------------------
__global__ void adaptive_forward_kernel(
    const float* __restrict__ q, const float* __restrict__ k,
    const float* __restrict__ v, const float* __restrict__ write,
    const float* __restrict__ forget, const float momentum, const float beta,
    const float balance, const float lo, const float hi, const int maxs,
    float* __restrict__ y, float* __restrict__ mem_prev,
    float* __restrict__ surprise_prev, int64_t* __restrict__ depth_counts,
    int64_t L, int M) {
  extern __shared__ float sh[];
  float* mem = sh;
  float* sur = mem + M * M;
  float* qv = sur + M * M;
  float* kv = qv + M;
  float* vv = kv + M;
  float* fv = vv + M;
  float* err = fv + M;
  float* red = err + M;  // reduction scratch [M*M] for mean_abs

  const int64_t b = blockIdx.x;
  const int tid = threadIdx.x;
  const int i = tid / M, j = tid % M;
  const float scale = rsqrtf((float)M);
  const float logM = __logf((float)M);

  mem[tid] = 0.0f; sur[tid] = 0.0f;
  __syncthreads();

  const float* qb = q + b * L * M;
  const float* kb = k + b * L * M;
  const float* vb = v + b * L * M;
  const float* fb = forget + b * L * M;
  const float* wb = write + b * L;
  float* yb = y + b * L * M;
  float* memb = mem_prev + b * L * M * M;
  float* surb = surprise_prev + b * L * M * M;

  for (int64_t t = 0; t < L; ++t) {
    if (tid < M) {
      qv[tid] = qb[t * M + tid]; kv[tid] = kb[t * M + tid];
      vv[tid] = vb[t * M + tid]; fv[tid] = fb[t * M + tid];
    }
    memb[t * M * M + tid] = mem[tid];
    surb[t * M * M + tid] = sur[tid];
    __syncthreads();

    if (i == 0) {
      yb[t * M + j] = semiring_read_col(mem, qv, beta, j, M, logM);
      err[j] = vv[j] - semiring_read_col(mem, kv, beta, j, M, logM);
    }
    __syncthreads();

    const float w = wb[t];
    const float delta = kv[i] * err[j] * scale;
    const float raw0 = momentum * sur[tid] + w * delta;
    red[tid] = fabsf(raw0);
    __syncthreads();
    // reduce red -> mean_abs (thread 0), broadcast via shared
    if (tid == 0) {
      float s = 0.0f;
      for (int n = 0; n < M * M; ++n) s += red[n];
      red[0] = s / (float)(M * M);
    }
    __syncthreads();
    const int steps = adaptive_steps(red[0], lo, hi, maxs);
    if (tid == 0) depth_counts[b * L + t] = steps;
    const int applied = steps < 1 ? 1 : steps;

    const float decay = 1.0f - fv[i];
    float s = bal(raw0, balance);
    for (int r = 1; r < applied; ++r) s = bal(momentum * s + w * delta, balance);
    sur[tid] = s;
    mem[tid] = decay * mem[tid] + (steps > 0 ? s : 0.0f);
    __syncthreads();
  }
}

// --------------------------- adaptive backward ----------------------------
__global__ void __launch_bounds__(1024) adaptive_backward_kernel(
    const float* __restrict__ q, const float* __restrict__ k,
    const float* __restrict__ v, const float* __restrict__ write,
    const float* __restrict__ forget, const float momentum, const float beta,
    const float balance, const float lo, const float hi, const int maxs,
    const float* __restrict__ grad_y, const float* __restrict__ mem_prev,
    const float* __restrict__ surprise_prev, float* __restrict__ grad_q,
    float* __restrict__ grad_k, float* __restrict__ grad_v,
    float* __restrict__ grad_write, float* __restrict__ grad_forget,
    float* __restrict__ grad_momentum, float* __restrict__ grad_beta,
    float* __restrict__ grad_balance, int64_t L, int M) {
  extern __shared__ float sh[];
  float* mem = sh;                  // M*M
  float* sur_t = mem + M * M;       // M*M
  float* gmem = sur_t + M * M;      // M*M
  float* gmemp = gmem + M * M;      // M*M
  float* gsur = gmemp + M * M;      // M*M
  float* gsurp = gsur + M * M;      // M*M
  float* qv = gsurp + M * M;        // M
  float* kv = qv + M;
  float* vv = kv + M;
  float* fv = vv + M;
  float* gyv = fv + M;
  float* err = gyv + M;
  float* gq = err + M;
  float* gk = gq + M;
  float* gf = gk + M;
  float* gerr = gf + M;
  float* gpred = gerr + M;
  float* red = gpred + M;           // M*M reduction scratch
  float* sc = red + M * M;          // [0]=gw,[1]=gmom,[2]=gbeta,[3]=gbal (block)

  const int64_t b = blockIdx.x;
  const int tid = threadIdx.x;
  const int i = tid / M, j = tid % M;
  const float scale = rsqrtf((float)M);
  const float logM = __logf((float)M);

  gmem[tid] = 0.0f; gsur[tid] = 0.0f;
  if (tid < 4) sc[tid] = 0.0f;
  __syncthreads();

  const float* qb = q + b * L * M;
  const float* kb = k + b * L * M;
  const float* vb = v + b * L * M;
  const float* fb = forget + b * L * M;
  const float* wb = write + b * L;
  const float* gyb = grad_y + b * L * M;
  const float* memb = mem_prev + b * L * M * M;
  const float* surb = surprise_prev + b * L * M * M;
  float* gqb = grad_q + b * L * M;
  float* gkb = grad_k + b * L * M;
  float* gvb = grad_v + b * L * M;
  float* gfb = grad_forget + b * L * M;

  for (int64_t t = L - 1; t >= 0; --t) {
    mem[tid] = memb[t * M * M + tid];
    sur_t[tid] = surb[t * M * M + tid];
    gmemp[tid] = 0.0f; gsurp[tid] = 0.0f;
    if (tid < M) {
      qv[tid] = qb[t * M + tid]; kv[tid] = kb[t * M + tid];
      vv[tid] = vb[t * M + tid]; fv[tid] = fb[t * M + tid];
      gyv[tid] = gyb[t * M + tid];
      gq[tid] = 0.0f; gk[tid] = 0.0f; gf[tid] = 0.0f;
      gerr[tid] = 0.0f; gpred[tid] = 0.0f;
    }
    if (tid == 0) { sc[0] = 0.0f; }
    __syncthreads();

    // pred(k), err, and mean_abs -> steps
    if (i == 0) err[j] = vv[j] - semiring_read_col(mem, kv, beta, j, M, logM);
    __syncthreads();
    const float w = wb[t];
    const float delta = kv[i] * err[j] * scale;
    red[tid] = fabsf(momentum * sur_t[tid] + w * delta);
    __syncthreads();
    if (tid == 0) {
      float s = 0.0f;
      for (int n = 0; n < M * M; ++n) s += red[n];
      red[0] = s / (float)(M * M);
    }
    __syncthreads();
    const int steps = adaptive_steps(red[0], lo, hi, maxs);
    const int applied = steps < 1 ? 1 : steps;

    // grad through y = semiring_read(mem, q): gq, gmemp, grad_beta
    if (i == 0) {
      float mx = -1e30f;
      for (int ii = 0; ii < M; ++ii) {
        float z = beta * (mem[ii * M + j] + qv[ii]);
        mx = z > mx ? z : mx;
      }
      float se = 0.0f;
      for (int ii = 0; ii < M; ++ii) se += __expf(beta * (mem[ii * M + j] + qv[ii]) - mx);
      float lse = mx + __logf(se);
      float exp_score = 0.0f;
      for (int ii = 0; ii < M; ++ii) {
        float wgt = __expf(beta * (mem[ii * M + j] + qv[ii]) - mx) / se;
        exp_score += wgt * (mem[ii * M + j] + qv[ii]);
      }
      float go = gyv[j];
      atomicAdd(&sc[2], go * (beta * exp_score - (lse - logM)) / (beta * beta));
      for (int ii = 0; ii < M; ++ii) {
        float wgt = __expf(beta * (mem[ii * M + j] + qv[ii]) - mx) / se;
        float g = go * wgt;
        atomicAdd(&gq[ii], g);
        gmemp[ii * M + j] += g;   // unique column j per i==0 thread
      }
    }
    __syncthreads();

    // decay
    const float decay = 1.0f - fv[i];
    const float gm = gmem[tid];
    atomicAdd(&gf[i], -gm * mem[tid]);
    gmemp[tid] += gm * decay;
    if (steps > 0) gsur[tid] += gm;
    __syncthreads();

    // surprise recursion backward (per i,j independent)
    {
      float raw_h[8];  // max_steps small (<=8, asserted in launcher)
      float s_h[8];
      raw_h[0] = momentum * sur_t[tid] + w * delta;
      s_h[0] = bal(raw_h[0], balance);
      for (int r = 1; r < applied; ++r) {
        raw_h[r] = momentum * s_h[r - 1] + w * delta;
        s_h[r] = bal(raw_h[r], balance);
      }
      float gs = gsur[tid];
      float grad_delta = 0.0f;
      for (int r = applied - 1; r >= 0; --r) {
        float raw = raw_h[r];
        float abs_raw = fabsf(raw);
        float denom = 1.0f + balance * abs_raw;
        float dsq = denom * denom;
        float g_raw = gs / dsq;
        atomicAdd(&sc[3], -gs * raw * abs_raw / dsq);  // grad_balance
        atomicAdd(&sc[0], g_raw * delta);              // grad_w
        grad_delta += g_raw * w;
        if (r == 0) {
          atomicAdd(&sc[1], g_raw * sur_t[tid]);       // grad_momentum
          gsurp[tid] += g_raw * momentum;
        } else {
          atomicAdd(&sc[1], g_raw * s_h[r - 1]);
          gs = g_raw * momentum;
        }
      }
      atomicAdd(&gk[i], grad_delta * err[j] * scale);
      atomicAdd(&gerr[j], grad_delta * kv[i] * scale);
    }
    __syncthreads();

    if (tid < M) { gvb[t * M + tid] += gerr[tid]; gpred[tid] = -gerr[tid]; }
    __syncthreads();

    // grad through pred = semiring_read(mem, k): gk, gmemp, grad_beta
    if (i == 0) {
      float mx = -1e30f;
      for (int ii = 0; ii < M; ++ii) {
        float z = beta * (mem[ii * M + j] + kv[ii]);
        mx = z > mx ? z : mx;
      }
      float se = 0.0f;
      for (int ii = 0; ii < M; ++ii) se += __expf(beta * (mem[ii * M + j] + kv[ii]) - mx);
      float lse = mx + __logf(se);
      float exp_score = 0.0f;
      for (int ii = 0; ii < M; ++ii) {
        float wgt = __expf(beta * (mem[ii * M + j] + kv[ii]) - mx) / se;
        exp_score += wgt * (mem[ii * M + j] + kv[ii]);
      }
      float go = gpred[j];
      atomicAdd(&sc[2], go * (beta * exp_score - (lse - logM)) / (beta * beta));
      for (int ii = 0; ii < M; ++ii) {
        float wgt = __expf(beta * (mem[ii * M + j] + kv[ii]) - mx) / se;
        float g = go * wgt;
        atomicAdd(&gk[ii], g);
        gmemp[ii * M + j] += g;
      }
    }
    __syncthreads();

    if (tid < M) {
      gqb[t * M + tid] += gq[tid];
      gkb[t * M + tid] += gk[tid];
      gfb[t * M + tid] += gf[tid];
    }
    if (tid == 0) grad_write[b * L + t] += sc[0];
    gmem[tid] = gmemp[tid];
    gsur[tid] = gsurp[tid];
    __syncthreads();
  }
  if (tid == 0) {
    atomicAdd(grad_momentum, sc[1]);
    atomicAdd(grad_beta, sc[2]);
    atomicAdd(grad_balance, sc[3]);
  }
}

// ------------------------------- launchers --------------------------------
static void check_cuda(torch::Tensor t, const char* n) {
  TORCH_CHECK(t.is_cuda(), n, " must be CUDA");
  TORCH_CHECK(t.is_contiguous(), n, " must be contiguous");
  TORCH_CHECK(t.scalar_type() == torch::kFloat32, n, " must be float32");
}

std::vector<torch::Tensor> plain_forward(
    torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor write,
    torch::Tensor forget, double momentum) {
  check_cuda(q, "q");
  const int64_t B = q.size(0), L = q.size(1); const int M = (int)q.size(2);
  TORCH_CHECK(M <= MAXM, "M too large for CUDA scan");
  auto y = torch::empty_like(q);
  auto mem_prev = torch::empty({B, L, M, M}, q.options());
  auto sur_prev = torch::empty({B, L, M, M}, q.options());
  int shared = (2 * M * M + 6 * M) * sizeof(float);
  plain_forward_kernel<<<(int)B, M * M, shared>>>(
      q.data_ptr<float>(), k.data_ptr<float>(), v.data_ptr<float>(),
      write.data_ptr<float>(), forget.data_ptr<float>(), (float)momentum,
      y.data_ptr<float>(), mem_prev.data_ptr<float>(), sur_prev.data_ptr<float>(), L, M);
  return {y, mem_prev, sur_prev};
}

std::vector<torch::Tensor> plain_backward(
    torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor write,
    torch::Tensor forget, double momentum, torch::Tensor grad_y,
    torch::Tensor mem_prev, torch::Tensor sur_prev) {
  check_cuda(q, "q");
  const int64_t B = q.size(0), L = q.size(1); const int M = (int)q.size(2);
  auto gq = torch::zeros_like(q), gk = torch::zeros_like(q), gv = torch::zeros_like(q);
  auto gf = torch::zeros_like(q);
  auto gw = torch::zeros({B, L}, q.options());
  auto gmom = torch::zeros({1}, q.options());
  int shared = (6 * M * M + 11 * M + 2) * sizeof(float);
  plain_backward_kernel<<<(int)B, M * M, shared>>>(
      q.data_ptr<float>(), k.data_ptr<float>(), v.data_ptr<float>(),
      write.data_ptr<float>(), forget.data_ptr<float>(), (float)momentum,
      grad_y.data_ptr<float>(), mem_prev.data_ptr<float>(), sur_prev.data_ptr<float>(),
      gq.data_ptr<float>(), gk.data_ptr<float>(), gv.data_ptr<float>(),
      gw.data_ptr<float>(), gf.data_ptr<float>(), gmom.data_ptr<float>(), L, M);
  return {gq, gk, gv, gw, gf, gmom.sum()};
}

std::vector<torch::Tensor> adaptive_forward(
    torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor write,
    torch::Tensor forget, double momentum, double beta, double balance,
    double lo, double hi, int64_t maxs) {
  check_cuda(q, "q");
  const int64_t B = q.size(0), L = q.size(1); const int M = (int)q.size(2);
  TORCH_CHECK(M <= MAXM, "M too large for CUDA scan");
  TORCH_CHECK(maxs <= 8, "max_steps must be <= 8 for CUDA scan");
  auto y = torch::empty_like(q);
  auto mem_prev = torch::empty({B, L, M, M}, q.options());
  auto sur_prev = torch::empty({B, L, M, M}, q.options());
  auto depth = torch::empty({B, L}, q.options().dtype(torch::kInt64));
  int shared = (3 * M * M + 5 * M) * sizeof(float);
  adaptive_forward_kernel<<<(int)B, M * M, shared>>>(
      q.data_ptr<float>(), k.data_ptr<float>(), v.data_ptr<float>(),
      write.data_ptr<float>(), forget.data_ptr<float>(), (float)momentum,
      (float)beta, (float)balance, (float)lo, (float)hi, (int)maxs,
      y.data_ptr<float>(), mem_prev.data_ptr<float>(), sur_prev.data_ptr<float>(),
      depth.data_ptr<int64_t>(), L, M);
  return {y, mem_prev, sur_prev, depth};
}

std::vector<torch::Tensor> adaptive_backward(
    torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor write,
    torch::Tensor forget, double momentum, double beta, double balance,
    double lo, double hi, int64_t maxs, torch::Tensor grad_y,
    torch::Tensor mem_prev, torch::Tensor sur_prev) {
  check_cuda(q, "q");
  const int64_t B = q.size(0), L = q.size(1); const int M = (int)q.size(2);
  auto gq = torch::zeros_like(q), gk = torch::zeros_like(q), gv = torch::zeros_like(q);
  auto gf = torch::zeros_like(q);
  auto gw = torch::zeros({B, L}, q.options());
  auto gmom = torch::zeros({1}, q.options());
  auto gbeta = torch::zeros({1}, q.options());
  auto gbal = torch::zeros({1}, q.options());
  int shared = (7 * M * M + 11 * M + 4) * sizeof(float);
  adaptive_backward_kernel<<<(int)B, M * M, shared>>>(
      q.data_ptr<float>(), k.data_ptr<float>(), v.data_ptr<float>(),
      write.data_ptr<float>(), forget.data_ptr<float>(), (float)momentum,
      (float)beta, (float)balance, (float)lo, (float)hi, (int)maxs,
      grad_y.data_ptr<float>(), mem_prev.data_ptr<float>(), sur_prev.data_ptr<float>(),
      gq.data_ptr<float>(), gk.data_ptr<float>(), gv.data_ptr<float>(),
      gw.data_ptr<float>(), gf.data_ptr<float>(), gmom.data_ptr<float>(),
      gbeta.data_ptr<float>(), gbal.data_ptr<float>(), L, M);
  return {gq, gk, gv, gw, gf, gmom.sum(), gbeta.sum(), gbal.sum()};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("plain_forward", &plain_forward);
  m.def("plain_backward", &plain_backward);
  m.def("adaptive_forward", &adaptive_forward);
  m.def("adaptive_backward", &adaptive_backward);
}
