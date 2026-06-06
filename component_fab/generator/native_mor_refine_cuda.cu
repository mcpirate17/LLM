// CUDA forward+backward for the MoR-refine surprise-memory scan with an inline
// MLP halting router (Mixture-of-Recursions / PonderNet). DEVELOPED IN ISOLATION
// from native_surprise_memory_cuda.cu so a compile error here cannot break the
// existing (working) native adaptive path. Validated rel<2e-5 against the torch
// reference MoRRefineMLPLaneA._scan (forward + every input grad), then merged.
//
// One block per batch element; blockDim = M*M threads (one per memory entry).
// Refine recursion: err is RE-MEASURED via a semiring read against the
// progressively refined memory each inner step. Router features
// f = [mean|s_r|, mean|err_r|, r/R] -> Linear(3,H) -> GELU -> Linear(H,1) ->
// sigmoid halt; PonderNet soft commit p_r = rem*halt, rem *= (1-halt).
//
// Forward saves M_{t-1} and S_{t-1} per t; the backward recomputes the per-step
// surprise history s_hist[0..R] (in shared) and the halts, then backprops in
// reverse. R<=8, M<=32, H<=128.

#include <torch/extension.h>
#include <cuda_runtime.h>

#define MAXM 32
#define MAXH 128
#define MAXR 8

__device__ __forceinline__ float bal_f(float raw, float balance) {
  return raw / (1.0f + balance * fabsf(raw));
}
__device__ __forceinline__ float gelu_f(float x) {
  return 0.5f * x * (1.0f + erff(x * 0.70710678118654752440f));
}
__device__ __forceinline__ float gelu_grad_f(float x) {
  const float c = 0.70710678118654752440f;       // 1/sqrt(2)
  const float ic = 0.39894228040143267794f;       // 1/sqrt(2*pi)
  return 0.5f * (1.0f + erff(x * c)) + x * ic * __expf(-0.5f * x * x);
}
__device__ __forceinline__ float semiring_read_col(
    const float* mem, const float* addr, float beta, int j, int M, float logM) {
  float mx = -1e30f;
  for (int ii = 0; ii < M; ++ii) {
    float z = beta * (mem[ii * M + j] + addr[ii]);
    mx = z > mx ? z : mx;
  }
  float se = 0.0f;
  for (int ii = 0; ii < M; ++ii)
    se += __expf(beta * (mem[ii * M + j] + addr[ii]) - mx);
  return (mx + __logf(se) - logM) / beta;
}

// ============================== FORWARD ====================================
__global__ void __launch_bounds__(1024) mor_refine_forward_kernel(
    const float* __restrict__ q, const float* __restrict__ k,
    const float* __restrict__ v, const float* __restrict__ write,
    const float* __restrict__ forget, const float momentum, const float beta,
    const float balance, const int R, const float* __restrict__ W1,
    const float* __restrict__ b1, const float* __restrict__ W2, const float b2,
    const float a_coupling, const int H, float* __restrict__ y,
    float* __restrict__ depth_out,
    float* __restrict__ hist_out, float* __restrict__ mem_prev_out,
    float* __restrict__ sur_prev_out, int64_t L, int M) {
  extern __shared__ float sh[];
  float* mem = sh;
  float* sur = mem + M * M;
  float* mem_r = sur + M * M;
  float* s_r = mem_r + M * M;
  float* mem_acc = s_r + M * M;
  float* sur_acc = mem_acc + M * M;
  float* red = sur_acc + M * M;
  float* qv = red + M * M;
  float* kv = qv + M;
  float* vv = kv + M;
  float* fv = vv + M;
  float* err = fv + M;
  float* hbuf = err + M;
  float* sc = hbuf + H;  // [0]meanS [1]meanErr [2]rem [3]halt [4]logit [5]depth
  float* histR = sc + 6;  // [R] accumulated halting mass per depth (over tokens)

  const int64_t b = blockIdx.x;
  const int tid = threadIdx.x;
  const int i = tid / M, j = tid % M;
  const float scale = rsqrtf((float)M);
  const float logM = __logf((float)M);

  mem[tid] = 0.0f;
  sur[tid] = 0.0f;
  if (tid < R) histR[tid] = 0.0f;
  __syncthreads();

  const float* qb = q + b * L * M;
  const float* kb = k + b * L * M;
  const float* vb = v + b * L * M;
  const float* fb = forget + b * L * M;
  const float* wb = write + b * L;
  float* yb = y + b * L * M;
  float* memb = mem_prev_out + b * L * M * M;
  float* surb = sur_prev_out + b * L * M * M;

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

    if (i == 0) yb[t * M + j] = semiring_read_col(mem, qv, beta, j, M, logM);
    mem_r[tid] = mem[tid];
    s_r[tid] = sur[tid];
    mem_acc[tid] = 0.0f;
    sur_acc[tid] = 0.0f;
    if (tid == 0) { sc[2] = 1.0f; sc[5] = 0.0f; }
    __syncthreads();

    const float w = wb[t];
    const float decay = 1.0f - fv[i];
    for (int r = 1; r <= R; ++r) {
      if (i == 0) err[j] = vv[j] - semiring_read_col(mem_r, kv, beta, j, M, logM);
      __syncthreads();
      const float delta = kv[i] * err[j] * scale;
      const float s_new = bal_f(momentum * s_r[tid] + w * delta, balance);
      s_r[tid] = s_new;
      mem_r[tid] = decay * mem[tid] + s_new;
      red[tid] = fabsf(s_new);
      __syncthreads();
      if (tid == 0) {
        float ss = 0.0f;
        for (int n = 0; n < M * M; ++n) ss += red[n];
        sc[0] = ss / (float)(M * M);
        float es = 0.0f;
        for (int n = 0; n < M; ++n) es += fabsf(err[n]);
        sc[1] = es / (float)M;
      }
      __syncthreads();
      const float f0 = sc[0], f1 = sc[1], f2 = (float)r / (float)R;
      if (tid < H)
        hbuf[tid] = gelu_f(W1[tid * 3 + 0] * f0 + W1[tid * 3 + 1] * f1 +
                           W1[tid * 3 + 2] * f2 + b1[tid]);
      __syncthreads();
      if (tid == 0) {
        float lg = b2;
        for (int h = 0; h < H; ++h) lg += W2[h] * hbuf[h];
        lg -= a_coupling * sc[1];  // surprise floor: more |err| -> lower halt -> deeper
        sc[4] = lg;
        sc[3] = (r == R) ? 1.0f : (1.0f / (1.0f + __expf(-lg)));
      }
      __syncthreads();
      const float halt = sc[3], rem = sc[2], p_r = rem * halt;
      mem_acc[tid] += p_r * mem_r[tid];
      sur_acc[tid] += p_r * s_r[tid];
      // barrier: every thread must finish reading rem=sc[2] before thread 0
      // overwrites sc[2] below — without it this is a WAR hazard that lets a
      // thread occasionally read the updated rem, corrupting the commit -> NaN.
      __syncthreads();
      if (tid == 0) {
        sc[5] += p_r * (float)r;
        sc[2] = rem * (1.0f - halt);
        histR[r - 1] += p_r;  // mass at depth r, summed over tokens
      }
      __syncthreads();
    }
    mem[tid] = mem_acc[tid];
    sur[tid] = sur_acc[tid];
    if (tid == 0) depth_out[b * L + t] = sc[5];
    __syncthreads();
  }
  if (tid < R) hist_out[b * R + tid] = histR[tid];  // sum_t p_r; norm on host
}

// ============================== BACKWARD ===================================
// grad inputs: grad_y [B,L,M], grad wrt M_t/S_t are zero (state internal, only
// y is the module output). We still carry grad through the state recurrence.
__global__ void __launch_bounds__(1024) mor_refine_backward_kernel(
    const float* __restrict__ q, const float* __restrict__ k,
    const float* __restrict__ v, const float* __restrict__ write,
    const float* __restrict__ forget, const float momentum, const float beta,
    const float balance, const int R, const float* __restrict__ W1,
    const float* __restrict__ b1, const float* __restrict__ W2, const float b2,
    const float a_coupling, const int H, const float* __restrict__ grad_y,
    const float* __restrict__ grad_depth,
    const float* __restrict__ mem_prev, const float* __restrict__ sur_prev,
    float* __restrict__ gq, float* __restrict__ gk, float* __restrict__ gv,
    float* __restrict__ gwrite, float* __restrict__ gforget,
    float* __restrict__ gmom, float* __restrict__ gbeta,
    float* __restrict__ gbal, float* __restrict__ gW1, float* __restrict__ gb1,
    float* __restrict__ gW2, float* __restrict__ gb2,
    float* __restrict__ g_a_coupling, int64_t L, int M) {
  extern __shared__ float sh[];
  float* mem = sh;                       // M*M  M_{t-1}
  float* GM = mem + M * M;               // M*M  grad wrt M_t (carried)
  float* GS = GM + M * M;                // M*M  grad wrt S_t (carried)
  float* gmem_o = GS + M * M;            // M*M  grad to M_{t-1}
  float* gsur_o = gmem_o + M * M;        // M*M  grad to S_{t-1}
  float* gs_cur = gsur_o + M * M;        // M*M  grad wrt s_hist[r] (from r+1)
  float* gs_next = gs_cur + M * M;       // M*M  grad wrt s_hist[r-1] (building)
  float* s_hist = gs_next + M * M;       // (R+1)*M*M
  float* red = s_hist + (R + 1) * M * M; // M*M
  float* qv = red + M * M;               // M
  float* kv = qv + M;
  float* vv = kv + M;
  float* fv = vv + M;
  float* err = fv + M;
  float* gerr = err + M;
  float* hbuf = gerr + M;                // H
  float* halt_h = hbuf + H;              // R+1
  float* rem_h = halt_h + (R + 1);       // R+1  (rem_before[r] = rem[r-1])
  float* sc = rem_h + (R + 1);           // scratch scalars [8]

  const int64_t b = blockIdx.x;
  const int tid = threadIdx.x;
  const int i = tid / M, j = tid % M;
  const float scale = rsqrtf((float)M);
  const float logM = __logf((float)M);

  GM[tid] = 0.0f;
  GS[tid] = 0.0f;
  __syncthreads();

  const float* qb = q + b * L * M;
  const float* kb = k + b * L * M;
  const float* vb = v + b * L * M;
  const float* fb = forget + b * L * M;
  const float* wb = write + b * L;
  const float* gyb = grad_y + b * L * M;
  const float* gdb = grad_depth + b * L;  // ponder grad: d L / d depth_acc[t]
  const float* memb = mem_prev + b * L * M * M;
  const float* surb = sur_prev + b * L * M * M;
  float* gqb = gq + b * L * M;
  float* gkb = gk + b * L * M;
  float* gvb = gv + b * L * M;
  float* gfb = gforget + b * L * M;

  for (int64_t t = L - 1; t >= 0; --t) {
    mem[tid] = memb[t * M * M + tid];
    if (tid < M) {
      qv[tid] = qb[t * M + tid];
      kv[tid] = kb[t * M + tid];
      vv[tid] = vb[t * M + tid];
      fv[tid] = fb[t * M + tid];
    }
    s_hist[tid] = surb[t * M * M + tid];  // s_hist[0] = S_{t-1}
    gmem_o[tid] = 0.0f;
    gsur_o[tid] = 0.0f;
    __syncthreads();

    const float w = wb[t];
    const float decay = 1.0f - fv[i];

    // -------- phase 1: recompute s_hist[1..R], halt_h[r], rem_h[r] --------
    if (tid == 0) { halt_h[0] = 0.0f; rem_h[1] = 1.0f; }
    __syncthreads();
    for (int r = 1; r <= R; ++r) {
      const float* mem_in = (r == 1) ? mem : (s_hist + (r - 1) * M * M);
      // err uses mem_in directly at r==1 (M_{t-1}); for r>1 mem_in is mem_r^out
      // = decay*mem + s_hist[r-1], so reconstruct that into red first.
      if (r > 1) red[tid] = decay * mem[tid] + s_hist[(r - 1) * M * M + tid];
      __syncthreads();
      const float* mr = (r == 1) ? mem : red;
      if (i == 0) err[j] = vv[j] - semiring_read_col(mr, kv, beta, j, M, logM);
      __syncthreads();
      const float delta = kv[i] * err[j] * scale;
      const float s_new =
          bal_f(momentum * s_hist[(r - 1) * M * M + tid] + w * delta, balance);
      s_hist[r * M * M + tid] = s_new;
      red[tid] = fabsf(s_new);
      __syncthreads();
      if (tid == 0) {
        float ss = 0.0f;
        for (int n = 0; n < M * M; ++n) ss += red[n];
        sc[0] = ss / (float)(M * M);
        float es = 0.0f;
        for (int n = 0; n < M; ++n) es += fabsf(err[n]);
        sc[1] = es / (float)M;
      }
      __syncthreads();
      const float f0 = sc[0], f1 = sc[1], f2 = (float)r / (float)R;
      if (tid < H)
        hbuf[tid] = gelu_f(W1[tid * 3 + 0] * f0 + W1[tid * 3 + 1] * f1 +
                           W1[tid * 3 + 2] * f2 + b1[tid]);
      __syncthreads();
      if (tid == 0) {
        float lg = b2;
        for (int h = 0; h < H; ++h) lg += W2[h] * hbuf[h];
        lg -= a_coupling * sc[1];  // match forward surprise floor
        halt_h[r] = (r == R) ? 1.0f : (1.0f / (1.0f + __expf(-lg)));
        if (r < R) rem_h[r + 1] = rem_h[r] * (1.0f - halt_h[r]);
      }
      __syncthreads();
    }

    // grad wrt y_t = SR(M_{t-1}, q): accumulate into gq, gmem_o, gbeta
    {
      const float go = (i == 0) ? gyb[t * M + j] : 0.0f;
      if (i == 0) {
        float mx = -1e30f;
        for (int ii = 0; ii < M; ++ii) {
          float z = beta * (mem[ii * M + j] + qv[ii]);
          mx = z > mx ? z : mx;
        }
        float se = 0.0f;
        for (int ii = 0; ii < M; ++ii)
          se += __expf(beta * (mem[ii * M + j] + qv[ii]) - mx);
        float lse = mx + __logf(se);
        float exp_score = 0.0f;
        for (int ii = 0; ii < M; ++ii) {
          float wgt = __expf(beta * (mem[ii * M + j] + qv[ii]) - mx) / se;
          exp_score += wgt * (mem[ii * M + j] + qv[ii]);
        }
        atomicAdd(gbeta, go * (beta * exp_score - (lse - logM)) / (beta * beta));
        for (int ii = 0; ii < M; ++ii) {
          float wgt = __expf(beta * (mem[ii * M + j] + qv[ii]) - mx) / se;
          float g = go * wgt;
          atomicAdd(&gqb[t * M + ii], g);
          gmem_o[ii * M + j] += g;
        }
      }
    }
    __syncthreads();

    // -------- phase 2: backprop the recursion, r = R .. 1 --------
    // gs_cur = grad wrt s_hist[r] carried from r+1 (momentum + mem_in[r+1] read);
    // gs_next accumulates grad wrt s_hist[r-1] this iteration.
    gs_cur[tid] = 0.0f;
    if (tid == 0) sc[7] = 0.0f;  // g_rem carried (grad wrt rem_h[r+1])
    __syncthreads();
    for (int r = R; r >= 1; --r) {
      const bool r1 = (r == 1);
      gs_next[tid] = 0.0f;
      if (!r1) red[tid] = decay * mem[tid] + s_hist[(r - 1) * M * M + tid];
      __syncthreads();
      const float* mr_in = r1 ? mem : red;       // mem_in[r]
      if (i == 0) err[j] = vv[j] - semiring_read_col(mr_in, kv, beta, j, M, logM);
      __syncthreads();
      const float s_rm1 = s_hist[(r - 1) * M * M + tid];
      const float delta = kv[i] * err[j] * scale;
      const float raw = momentum * s_rm1 + w * delta;
      const float adenom = 1.0f + balance * fabsf(raw);
      const float s_r = s_hist[r * M * M + tid];  // = bal(raw)
      const float mem_out = decay * mem[tid] + s_r;

      // recompute features + MLP forward for grads
      red[tid] = fabsf(s_r);
      __syncthreads();
      if (tid == 0) {
        float ss = 0.0f;
        for (int n = 0; n < M * M; ++n) ss += red[n];
        sc[0] = ss / (float)(M * M);
        float es = 0.0f;
        for (int n = 0; n < M; ++n) es += fabsf(err[n]);
        sc[1] = es / (float)M;
      }
      __syncthreads();
      const float f0 = sc[0], f1 = sc[1], f2 = (float)r / (float)R;
      if (tid < H)
        hbuf[tid] = gelu_f(W1[tid * 3 + 0] * f0 + W1[tid * 3 + 1] * f1 +
                           W1[tid * 3 + 2] * f2 + b1[tid]);
      __syncthreads();

      const float halt = halt_h[r];
      const float rem_before = rem_h[r];
      const float p_r = rem_before * halt;

      // dp_r = sum_ij (GM*mem_out + GS*s_r)
      red[tid] = GM[tid] * mem_out + GS[tid] * s_r;
      __syncthreads();
      if (tid == 0) {
        float ss = 0.0f;
        for (int n = 0; n < M * M; ++n) ss += red[n];
        sc[2] = ss;  // dp_r
      }
      __syncthreads();
      // depth_acc = Σ_r p_r·r, so the ponder loss adds grad_depth·r to dp_r;
      // it flows to the router through the same halt/rem chain as the commit grad.
      const float dp_r = sc[2] + gdb[t] * (float)r;

      // halt / rem (PonderNet) backward -> g_logit  (r==R: halt forced 1)
      if (tid == 0) {
        float g_rem_next = sc[7];
        float g_halt = dp_r * rem_before;
        float g_rem_cur = dp_r * halt;
        if (r < R) {
          g_halt += g_rem_next * (-rem_before);
          g_rem_cur += g_rem_next * (1.0f - halt);
        }
        sc[7] = g_rem_cur;
        sc[3] = (r < R) ? (g_halt * halt * (1.0f - halt)) : 0.0f;  // g_logit
        sc[5] = 0.0f;  // grad wrt f0
        // f1 gets the surprise-floor path (lg -= a*f1 => d lg/d f1 has -a); the
        // MLP threads add their W1[*,1] contributions to sc[6] below.
        sc[6] = -a_coupling * sc[3];  // grad wrt f1 from the coupling term
        // d lg / d a_coupling = -f1, summed over all blocks/tokens/steps.
        if (r < R) atomicAdd(g_a_coupling, -sc[3] * f1);
      }
      __syncthreads();
      const float g_logit = sc[3];

      // MLP backward -> gW2,gb2,gW1,gb1, and gf0(sc5)/gf1(sc6)
      if (r < R && tid < H) {
        float hv = hbuf[tid];
        atomicAdd(&gW2[tid], g_logit * hv);
        float a = W1[tid * 3 + 0] * f0 + W1[tid * 3 + 1] * f1 +
                  W1[tid * 3 + 2] * f2 + b1[tid];
        float g_a = g_logit * W2[tid] * gelu_grad_f(a);
        atomicAdd(&gW1[tid * 3 + 0], g_a * f0);
        atomicAdd(&gW1[tid * 3 + 1], g_a * f1);
        atomicAdd(&gW1[tid * 3 + 2], g_a * f2);
        atomicAdd(&gb1[tid], g_a);
        atomicAdd(&sc[5], g_a * W1[tid * 3 + 0]);
        atomicAdd(&sc[6], g_a * W1[tid * 3 + 1]);
      }
      if (r < R && tid == 0) atomicAdd(gb2, g_logit);
      __syncthreads();
      const float gf0 = sc[5], gf1 = sc[6];

      // grad into mem_out = decay*mem + s_r  (commit p_r*GM)
      const float g_mem_out = p_r * GM[tid];
      gmem_o[tid] += decay * g_mem_out;
      atomicAdd(&gfb[t * M + i], -mem[tid] * g_mem_out);

      // total grad wrt s_r
      float g_s_r = p_r * GS[tid] + g_mem_out + gs_cur[tid] +
                    (s_r >= 0.0f ? 1.0f : -1.0f) / (float)(M * M) * gf0;

      // s_r = bal(raw)
      const float g_raw = g_s_r / (adenom * adenom);
      atomicAdd(gbal, -g_s_r * raw * fabsf(raw) / (adenom * adenom));
      atomicAdd(gmom, g_raw * s_rm1);
      gs_next[tid] += g_raw * momentum;          // grad wrt s_{r-1} (momentum)
      const float g_delta = g_raw * w;

      // zero gerr (len M) and grad_write scalar before accumulation
      if (tid < M) gerr[tid] = 0.0f;
      if (tid == 0) sc[4] = 0.0f;
      __syncthreads();
      atomicAdd(&gkb[t * M + i], g_delta * err[j] * scale);
      atomicAdd(&gerr[j], g_delta * kv[i] * scale);
      atomicAdd(&sc[4], g_raw * delta);
      __syncthreads();
      if (tid == 0) atomicAdd(&gwrite[b * L + t], sc[4]);
      // feature f1 = mean|err|
      if (i == 0) gerr[j] += (err[j] >= 0.0f ? 1.0f : -1.0f) / (float)M * gf1;
      __syncthreads();

      // recompute mem_in[r] into red (clobbered by the reductions above)
      if (!r1) red[tid] = decay * mem[tid] + s_hist[(r - 1) * M * M + tid];
      __syncthreads();
      const float* mr_in2 = r1 ? mem : red;

      // err = v - SR(mem_in[r], k): gv += gerr ; -gerr -> SR backward
      if (i == 0) {
        atomicAdd(&gvb[t * M + j], gerr[j]);
        const float gpred = -gerr[j];
        float mx = -1e30f;
        for (int ii = 0; ii < M; ++ii) {
          float z = beta * (mr_in2[ii * M + j] + kv[ii]);
          mx = z > mx ? z : mx;
        }
        float se = 0.0f;
        for (int ii = 0; ii < M; ++ii)
          se += __expf(beta * (mr_in2[ii * M + j] + kv[ii]) - mx);
        float lse = mx + __logf(se);
        float exp_score = 0.0f;
        for (int ii = 0; ii < M; ++ii) {
          float wgt = __expf(beta * (mr_in2[ii * M + j] + kv[ii]) - mx) / se;
          exp_score += wgt * (mr_in2[ii * M + j] + kv[ii]);
        }
        atomicAdd(gbeta, gpred * (beta * exp_score - (lse - logM)) / (beta * beta));
        for (int ii = 0; ii < M; ++ii) {
          float wgt = __expf(beta * (mr_in2[ii * M + j] + kv[ii]) - mx) / se;
          float g = gpred * wgt;
          atomicAdd(&gkb[t * M + ii], g);
          if (r1) {
            atomicAdd(&gmem_o[ii * M + j], g);          // mem_in[1] = M_{t-1}
          } else {
            // decay is per key-row ii here (NOT the thread-local i==0 decay)
            atomicAdd(&gmem_o[ii * M + j], (1.0f - fv[ii]) * g);
            atomicAdd(&gfb[t * M + ii], -mem[ii * M + j] * g);
            atomicAdd(&gs_next[ii * M + j], g);         // -> s_hist[r-1]
          }
        }
      }
      __syncthreads();

      // carry: gs_cur <- gs_next for next lower r
      gs_cur[tid] = gs_next[tid];
      __syncthreads();
    }
    // gs_cur now holds grad wrt s_hist[0] = S_{t-1}
    gsur_o[tid] += gs_cur[tid];
    __syncthreads();

    // carry state grads to t-1
    GM[tid] = gmem_o[tid];
    GS[tid] = gsur_o[tid];
    __syncthreads();
  }
}

// ------------------------------- launchers --------------------------------
static void check_cuda(torch::Tensor t, const char* n) {
  TORCH_CHECK(t.is_cuda(), n, " must be CUDA");
  TORCH_CHECK(t.is_contiguous(), n, " must be contiguous");
  TORCH_CHECK(t.scalar_type() == torch::kFloat32, n, " must be float32");
}

std::vector<torch::Tensor> mor_refine_forward(
    torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor write,
    torch::Tensor forget, double momentum, double beta, double balance,
    int64_t R, torch::Tensor W1, torch::Tensor b1, torch::Tensor W2,
    double b2, double a_coupling) {
  check_cuda(q, "q");
  const int64_t B = q.size(0), L = q.size(1);
  const int M = (int)q.size(2);
  const int H = (int)W1.size(0);
  TORCH_CHECK(M <= MAXM && H <= MAXH && R >= 1 && R <= MAXR, "dim out of range");
  auto y = torch::empty_like(q);
  auto depth = torch::zeros({B, L}, q.options());
  auto hist = torch::zeros({B, R}, q.options());
  auto mem_prev = torch::empty({B, L, M, M}, q.options());
  auto sur_prev = torch::empty({B, L, M, M}, q.options());
  int shared = (7 * M * M + 5 * M + H + 6 + R) * sizeof(float);
  mor_refine_forward_kernel<<<(int)B, M * M, shared>>>(
      q.data_ptr<float>(), k.data_ptr<float>(), v.data_ptr<float>(),
      write.data_ptr<float>(), forget.data_ptr<float>(), (float)momentum,
      (float)beta, (float)balance, (int)R, W1.data_ptr<float>(),
      b1.data_ptr<float>(), W2.data_ptr<float>(), (float)b2, (float)a_coupling, H,
      y.data_ptr<float>(), depth.data_ptr<float>(), hist.data_ptr<float>(),
      mem_prev.data_ptr<float>(), sur_prev.data_ptr<float>(), L, M);
  return {y, depth, hist, mem_prev, sur_prev};
}

std::vector<torch::Tensor> mor_refine_backward(
    torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor write,
    torch::Tensor forget, double momentum, double beta, double balance,
    int64_t R, torch::Tensor W1, torch::Tensor b1, torch::Tensor W2, double b2,
    double a_coupling, torch::Tensor grad_y, torch::Tensor grad_depth,
    torch::Tensor mem_prev, torch::Tensor sur_prev) {
  check_cuda(q, "q");
  const int64_t B = q.size(0), L = q.size(1);
  const int M = (int)q.size(2);
  const int H = (int)W1.size(0);
  auto gq = torch::zeros_like(q), gk = torch::zeros_like(q),
       gv = torch::zeros_like(q), gf = torch::zeros_like(q);
  auto gw = torch::zeros({B, L}, q.options());
  auto gmom = torch::zeros({1}, q.options());
  auto gbeta = torch::zeros({1}, q.options());
  auto gbal = torch::zeros({1}, q.options());
  auto gW1 = torch::zeros_like(W1);
  auto gb1 = torch::zeros_like(b1);
  auto gW2 = torch::zeros_like(W2);
  auto gb2 = torch::zeros({1}, q.options());
  auto g_a = torch::zeros({1}, q.options());
  // shared: 7 M*M working arrays + s_hist[(R+1)] + red  = (9+R) M*M, then the
  // M-length vectors (qv,kv,vv,fv,err,gerr), hbuf[H], halt_h/rem_h[R+1], sc[8].
  int shared = ((9 + R) * M * M + 6 * M + H + 2 * (R + 1) + 8) * sizeof(float);
  cudaFuncSetAttribute(mor_refine_backward_kernel,
                       cudaFuncAttributeMaxDynamicSharedMemorySize, shared);
  mor_refine_backward_kernel<<<(int)B, M * M, shared>>>(
      q.data_ptr<float>(), k.data_ptr<float>(), v.data_ptr<float>(),
      write.data_ptr<float>(), forget.data_ptr<float>(), (float)momentum,
      (float)beta, (float)balance, (int)R, W1.data_ptr<float>(),
      b1.data_ptr<float>(), W2.data_ptr<float>(), (float)b2, (float)a_coupling, H,
      grad_y.data_ptr<float>(), grad_depth.data_ptr<float>(),
      mem_prev.data_ptr<float>(),
      sur_prev.data_ptr<float>(), gq.data_ptr<float>(), gk.data_ptr<float>(),
      gv.data_ptr<float>(), gw.data_ptr<float>(), gf.data_ptr<float>(),
      gmom.data_ptr<float>(), gbeta.data_ptr<float>(), gbal.data_ptr<float>(),
      gW1.data_ptr<float>(), gb1.data_ptr<float>(), gW2.data_ptr<float>(),
      gb2.data_ptr<float>(), g_a.data_ptr<float>(), L, M);
  return {gq, gk, gv, gw, gf, gmom.sum(), gbeta.sum(), gbal.sum(),
          gW1, gb1, gW2, gb2.sum(), g_a.sum()};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("mor_refine_forward", &mor_refine_forward);
  m.def("mor_refine_backward", &mor_refine_backward);
}
