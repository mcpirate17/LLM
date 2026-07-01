// CPU forward+backward for the MoR-refine surprise-memory scan with the inline
// MLP halting router — the CPU port of native_mor_refine_cuda.cu (same math,
// same entry-point signatures), so the fab's CPU grading path stops running the
// per-token torch loop in MoRRefineMLPLaneA._scan. The CUDA kernel is float32
// block-parallel over M*M; this port is templated (float32/float64, so
// gradcheck runs in double) and parallelizes over the batch with ATen's
// threadpool. Kept self-contained: each JIT extension is version-hashed on its
// single listed source, so nothing is shared with the .cu by header.
//
// Refine recursion per token t (matching MoRRefineLaneA._scan + MLP router):
//   y_t          = SR(M_{t-1}, q_t)                       (read-before-write)
//   loop r=1..R:
//     err_r      = v_t - SR(mem_r, k_t)                   (re-measured)
//     s_r        = bal(momentum*s_{r-1} + w_t*(k⊗err_r)*scale, balance)
//     mem_r      = (1-forget)∘M_{t-1} + s_r
//     halt       = sigmoid(MLP([mean|s_r|, mean|err_r|, r/R]) - a*mean|err_r|)
//     p_r        = rem*halt;  commit mem_acc/sur_acc/depth;  rem *= (1-halt)
//   M_t = mem_acc, S_t = sur_acc
// SR is the tempered log-sum-exp semiring read ((lse - log M)/beta).

#include <torch/extension.h>
#include <ATen/Parallel.h>

#include <cmath>
#include <limits>
#include <vector>

#define MAXM 32
#define MAXH 512
#define MAXR 8

namespace {

template <typename scalar_t>
scalar_t bal_f(scalar_t raw, scalar_t balance) {
  return raw / (static_cast<scalar_t>(1) + balance * std::abs(raw));
}

template <typename scalar_t>
scalar_t gelu_f(scalar_t x) {
  return static_cast<scalar_t>(0.5) * x *
         (static_cast<scalar_t>(1) +
          std::erf(x * static_cast<scalar_t>(0.70710678118654752440)));
}

template <typename scalar_t>
scalar_t gelu_grad_f(scalar_t x) {
  const scalar_t c = static_cast<scalar_t>(0.70710678118654752440);   // 1/sqrt(2)
  const scalar_t ic = static_cast<scalar_t>(0.39894228040143267794);  // 1/sqrt(2*pi)
  return static_cast<scalar_t>(0.5) * (static_cast<scalar_t>(1) + std::erf(x * c)) +
         x * ic * std::exp(static_cast<scalar_t>(-0.5) * x * x);
}

template <typename scalar_t>
scalar_t semiring_read_col(
    const scalar_t* mem, const scalar_t* addr, scalar_t beta, int64_t j,
    int64_t M, scalar_t logM) {
  scalar_t mx = -std::numeric_limits<scalar_t>::infinity();
  for (int64_t ii = 0; ii < M; ++ii) {
    mx = std::max(mx, beta * (mem[ii * M + j] + addr[ii]));
  }
  scalar_t se = static_cast<scalar_t>(0);
  for (int64_t ii = 0; ii < M; ++ii) {
    se += std::exp(beta * (mem[ii * M + j] + addr[ii]) - mx);
  }
  return (mx + std::log(se) - logM) / beta;
}

// Backward of one semiring-read column with upstream gradient gpred; the
// caller supplies how memory gradients land (r==1 reads M_{t-1} directly,
// r>1 reads decay∘M_{t-1} + s_{r-1}).
template <typename scalar_t, typename MemGradFn>
void semiring_read_col_grad(
    const scalar_t* mem, const scalar_t* addr, scalar_t beta, int64_t j,
    int64_t M, scalar_t logM, scalar_t gpred, scalar_t* gaddr,
    scalar_t* gbeta, MemGradFn&& mem_grad) {
  scalar_t mx = -std::numeric_limits<scalar_t>::infinity();
  for (int64_t ii = 0; ii < M; ++ii) {
    mx = std::max(mx, beta * (mem[ii * M + j] + addr[ii]));
  }
  scalar_t se = static_cast<scalar_t>(0);
  for (int64_t ii = 0; ii < M; ++ii) {
    se += std::exp(beta * (mem[ii * M + j] + addr[ii]) - mx);
  }
  const scalar_t lse = mx + std::log(se);
  scalar_t exp_score = static_cast<scalar_t>(0);
  for (int64_t ii = 0; ii < M; ++ii) {
    const scalar_t wgt = std::exp(beta * (mem[ii * M + j] + addr[ii]) - mx) / se;
    exp_score += wgt * (mem[ii * M + j] + addr[ii]);
  }
  *gbeta += gpred * (beta * exp_score - (lse - logM)) / (beta * beta);
  for (int64_t ii = 0; ii < M; ++ii) {
    const scalar_t wgt = std::exp(beta * (mem[ii * M + j] + addr[ii]) - mx) / se;
    const scalar_t g = gpred * wgt;
    gaddr[ii] += g;
    mem_grad(ii, g);
  }
}

template <typename scalar_t>
void router_forward(
    const scalar_t* W1, const scalar_t* b1, const scalar_t* W2, scalar_t b2,
    scalar_t a_coupling, int64_t H, scalar_t f0, scalar_t f1, scalar_t f2,
    scalar_t* hbuf, scalar_t* logit) {
  scalar_t lg = b2;
  for (int64_t h = 0; h < H; ++h) {
    hbuf[h] = gelu_f(W1[h * 3 + 0] * f0 + W1[h * 3 + 1] * f1 +
                     W1[h * 3 + 2] * f2 + b1[h]);
    lg += W2[h] * hbuf[h];
  }
  lg -= a_coupling * f1;  // surprise floor: more |err| -> lower halt -> deeper
  *logit = lg;
}

template <typename scalar_t>
void mor_refine_forward_one(
    const scalar_t* q, const scalar_t* k, const scalar_t* v,
    const scalar_t* write, const scalar_t* forget, scalar_t momentum,
    scalar_t beta, scalar_t balance, int64_t R, const scalar_t* W1,
    const scalar_t* b1, const scalar_t* W2, scalar_t b2, scalar_t a_coupling,
    int64_t H, scalar_t* y, scalar_t* depth_out, scalar_t* hist_out,
    scalar_t* mem_prev_out, scalar_t* sur_prev_out, int64_t L, int64_t M) {
  const scalar_t scale =
      static_cast<scalar_t>(1.0 / std::sqrt(static_cast<double>(M)));
  const scalar_t logM = std::log(static_cast<scalar_t>(M));
  std::vector<scalar_t> mem(M * M, 0), sur(M * M, 0);
  std::vector<scalar_t> mem_r(M * M), s_r(M * M);
  std::vector<scalar_t> mem_acc(M * M), sur_acc(M * M);
  std::vector<scalar_t> err(M), hbuf(H);

  for (int64_t t = 0; t < L; ++t) {
    std::copy(mem.begin(), mem.end(), mem_prev_out + t * M * M);
    std::copy(sur.begin(), sur.end(), sur_prev_out + t * M * M);

    for (int64_t j = 0; j < M; ++j) {
      y[t * M + j] = semiring_read_col(mem.data(), q + t * M, beta, j, M, logM);
    }
    std::copy(mem.begin(), mem.end(), mem_r.begin());
    std::copy(sur.begin(), sur.end(), s_r.begin());
    std::fill(mem_acc.begin(), mem_acc.end(), static_cast<scalar_t>(0));
    std::fill(sur_acc.begin(), sur_acc.end(), static_cast<scalar_t>(0));
    scalar_t rem = static_cast<scalar_t>(1);
    scalar_t depth_acc = static_cast<scalar_t>(0);

    const scalar_t w = write[t];
    const scalar_t* f_t = forget + t * M;
    const scalar_t* k_t = k + t * M;
    const scalar_t* v_t = v + t * M;

    for (int64_t r = 1; r <= R; ++r) {
      for (int64_t j = 0; j < M; ++j) {
        err[j] = v_t[j] - semiring_read_col(mem_r.data(), k_t, beta, j, M, logM);
      }
      scalar_t abs_s_sum = static_cast<scalar_t>(0);
      for (int64_t i = 0; i < M; ++i) {
        const scalar_t decay = static_cast<scalar_t>(1) - f_t[i];
        for (int64_t j = 0; j < M; ++j) {
          const int64_t idx = i * M + j;
          const scalar_t delta = k_t[i] * err[j] * scale;
          const scalar_t s_new = bal_f(momentum * s_r[idx] + w * delta, balance);
          s_r[idx] = s_new;
          mem_r[idx] = decay * mem[idx] + s_new;
          abs_s_sum += std::abs(s_new);
        }
      }
      scalar_t abs_e_sum = static_cast<scalar_t>(0);
      for (int64_t j = 0; j < M; ++j) abs_e_sum += std::abs(err[j]);
      const scalar_t f0 = abs_s_sum / static_cast<scalar_t>(M * M);
      const scalar_t f1 = abs_e_sum / static_cast<scalar_t>(M);
      const scalar_t f2 = static_cast<scalar_t>(r) / static_cast<scalar_t>(R);
      scalar_t logit;
      router_forward(W1, b1, W2, b2, a_coupling, H, f0, f1, f2, hbuf.data(),
                     &logit);
      const scalar_t halt =
          (r == R) ? static_cast<scalar_t>(1)
                   : static_cast<scalar_t>(1) /
                         (static_cast<scalar_t>(1) + std::exp(-logit));
      const scalar_t p_r = rem * halt;
      for (int64_t n = 0; n < M * M; ++n) {
        mem_acc[n] += p_r * mem_r[n];
        sur_acc[n] += p_r * s_r[n];
      }
      depth_acc += p_r * static_cast<scalar_t>(r);
      hist_out[r - 1] += p_r;
      rem *= (static_cast<scalar_t>(1) - halt);
    }
    mem.swap(mem_acc);
    sur.swap(sur_acc);
    depth_out[t] = depth_acc;
  }
}

template <typename scalar_t>
void mor_refine_backward_one(
    const scalar_t* q, const scalar_t* k, const scalar_t* v,
    const scalar_t* write, const scalar_t* forget, scalar_t momentum,
    scalar_t beta, scalar_t balance, int64_t R, const scalar_t* W1,
    const scalar_t* b1, const scalar_t* W2, scalar_t b2, scalar_t a_coupling,
    int64_t H, const scalar_t* grad_y, const scalar_t* grad_depth,
    const scalar_t* mem_prev, const scalar_t* sur_prev, scalar_t* gq,
    scalar_t* gk, scalar_t* gv, scalar_t* gwrite, scalar_t* gforget,
    scalar_t* gmom, scalar_t* gbeta, scalar_t* gbal, scalar_t* gW1,
    scalar_t* gb1, scalar_t* gW2, scalar_t* gb2, scalar_t* g_a_coupling,
    int64_t L, int64_t M) {
  const scalar_t scale =
      static_cast<scalar_t>(1.0 / std::sqrt(static_cast<double>(M)));
  const scalar_t logM = std::log(static_cast<scalar_t>(M));
  std::vector<scalar_t> GM(M * M, 0), GS(M * M, 0);
  std::vector<scalar_t> gmem_o(M * M), gsur_o(M * M);
  std::vector<scalar_t> gs_cur(M * M), gs_next(M * M);
  std::vector<scalar_t> s_hist((R + 1) * M * M);
  std::vector<scalar_t> mem_in(M * M);
  std::vector<scalar_t> err(M), gerr(M), hbuf(H);
  std::vector<scalar_t> halt_h(R + 1), rem_h(R + 2), f0_h(R + 1), f1_h(R + 1);

  for (int64_t t = L - 1; t >= 0; --t) {
    const scalar_t* mem = mem_prev + t * M * M;
    const scalar_t* q_t = q + t * M;
    const scalar_t* k_t = k + t * M;
    const scalar_t* v_t = v + t * M;
    const scalar_t* f_t = forget + t * M;
    const scalar_t w = write[t];
    std::copy(sur_prev + t * M * M, sur_prev + (t + 1) * M * M,
              s_hist.begin());  // s_hist[0] = S_{t-1}
    std::fill(gmem_o.begin(), gmem_o.end(), static_cast<scalar_t>(0));
    std::fill(gsur_o.begin(), gsur_o.end(), static_cast<scalar_t>(0));

    // ---- phase 1: recompute s_hist[1..R], halts, remainders, features ----
    rem_h[1] = static_cast<scalar_t>(1);
    for (int64_t r = 1; r <= R; ++r) {
      const scalar_t* mr_in = mem;
      if (r > 1) {
        for (int64_t i = 0; i < M; ++i) {
          const scalar_t decay = static_cast<scalar_t>(1) - f_t[i];
          for (int64_t j = 0; j < M; ++j) {
            mem_in[i * M + j] =
                decay * mem[i * M + j] + s_hist[(r - 1) * M * M + i * M + j];
          }
        }
        mr_in = mem_in.data();
      }
      for (int64_t j = 0; j < M; ++j) {
        err[j] = v_t[j] - semiring_read_col(mr_in, k_t, beta, j, M, logM);
      }
      scalar_t abs_s_sum = static_cast<scalar_t>(0);
      for (int64_t i = 0; i < M; ++i) {
        for (int64_t j = 0; j < M; ++j) {
          const int64_t idx = i * M + j;
          const scalar_t delta = k_t[i] * err[j] * scale;
          const scalar_t s_new =
              bal_f(momentum * s_hist[(r - 1) * M * M + idx] + w * delta,
                    balance);
          s_hist[r * M * M + idx] = s_new;
          abs_s_sum += std::abs(s_new);
        }
      }
      scalar_t abs_e_sum = static_cast<scalar_t>(0);
      for (int64_t j = 0; j < M; ++j) abs_e_sum += std::abs(err[j]);
      f0_h[r] = abs_s_sum / static_cast<scalar_t>(M * M);
      f1_h[r] = abs_e_sum / static_cast<scalar_t>(M);
      scalar_t logit;
      router_forward(W1, b1, W2, b2, a_coupling, H, f0_h[r], f1_h[r],
                     static_cast<scalar_t>(r) / static_cast<scalar_t>(R),
                     hbuf.data(), &logit);
      halt_h[r] = (r == R) ? static_cast<scalar_t>(1)
                           : static_cast<scalar_t>(1) /
                                 (static_cast<scalar_t>(1) + std::exp(-logit));
      if (r < R) rem_h[r + 1] = rem_h[r] * (static_cast<scalar_t>(1) - halt_h[r]);
    }

    // grad wrt y_t = SR(M_{t-1}, q_t)
    for (int64_t j = 0; j < M; ++j) {
      const scalar_t go = grad_y[t * M + j];
      if (go == static_cast<scalar_t>(0)) continue;
      semiring_read_col_grad(
          mem, q_t, beta, j, M, logM, go, gq + t * M, gbeta,
          [&](int64_t ii, scalar_t g) { gmem_o[ii * M + j] += g; });
    }

    // ---- phase 2: backprop the recursion, r = R .. 1 ----
    std::fill(gs_cur.begin(), gs_cur.end(), static_cast<scalar_t>(0));
    scalar_t g_rem = static_cast<scalar_t>(0);
    for (int64_t r = R; r >= 1; --r) {
      const bool r1 = (r == 1);
      std::fill(gs_next.begin(), gs_next.end(), static_cast<scalar_t>(0));
      const scalar_t* mr_in = mem;
      if (!r1) {
        for (int64_t i = 0; i < M; ++i) {
          const scalar_t decay = static_cast<scalar_t>(1) - f_t[i];
          for (int64_t j = 0; j < M; ++j) {
            mem_in[i * M + j] =
                decay * mem[i * M + j] + s_hist[(r - 1) * M * M + i * M + j];
          }
        }
        mr_in = mem_in.data();
      }
      for (int64_t j = 0; j < M; ++j) {
        err[j] = v_t[j] - semiring_read_col(mr_in, k_t, beta, j, M, logM);
      }
      const scalar_t f0 = f0_h[r], f1 = f1_h[r];
      const scalar_t f2 = static_cast<scalar_t>(r) / static_cast<scalar_t>(R);
      const scalar_t halt = halt_h[r];
      const scalar_t rem_before = rem_h[r];
      const scalar_t p_r = rem_before * halt;

      // dp_r = Σ_ij (GM·mem_out + GS·s_r) + ponder grad through depth_acc
      scalar_t dp_r = static_cast<scalar_t>(0);
      for (int64_t i = 0; i < M; ++i) {
        const scalar_t decay = static_cast<scalar_t>(1) - f_t[i];
        for (int64_t j = 0; j < M; ++j) {
          const int64_t idx = i * M + j;
          const scalar_t s_r_v = s_hist[r * M * M + idx];
          dp_r += GM[idx] * (decay * mem[idx] + s_r_v) + GS[idx] * s_r_v;
        }
      }
      dp_r += grad_depth[t] * static_cast<scalar_t>(r);

      // PonderNet halt/remainder backward -> g_logit
      scalar_t g_halt = dp_r * rem_before;
      scalar_t g_rem_cur = dp_r * halt;
      if (r < R) {
        g_halt += g_rem * (-rem_before);
        g_rem_cur += g_rem * (static_cast<scalar_t>(1) - halt);
      }
      g_rem = g_rem_cur;
      const scalar_t g_logit =
          (r < R) ? g_halt * halt * (static_cast<scalar_t>(1) - halt)
                  : static_cast<scalar_t>(0);
      scalar_t gf0 = static_cast<scalar_t>(0);
      scalar_t gf1 = -a_coupling * g_logit;
      if (r < R) {
        *g_a_coupling += -g_logit * f1;
        *gb2 += g_logit;
        for (int64_t h = 0; h < H; ++h) {
          const scalar_t pre = W1[h * 3 + 0] * f0 + W1[h * 3 + 1] * f1 +
                               W1[h * 3 + 2] * f2 + b1[h];
          const scalar_t hv = gelu_f(pre);
          gW2[h] += g_logit * hv;
          const scalar_t g_pre = g_logit * W2[h] * gelu_grad_f(pre);
          gW1[h * 3 + 0] += g_pre * f0;
          gW1[h * 3 + 1] += g_pre * f1;
          gW1[h * 3 + 2] += g_pre * f2;
          gb1[h] += g_pre;
          gf0 += g_pre * W1[h * 3 + 0];
          gf1 += g_pre * W1[h * 3 + 1];
        }
      }

      // element-wise grads through commit, bal(), delta
      std::fill(gerr.begin(), gerr.end(), static_cast<scalar_t>(0));
      scalar_t g_write_acc = static_cast<scalar_t>(0);
      const scalar_t inv_mm = static_cast<scalar_t>(1) / static_cast<scalar_t>(M * M);
      for (int64_t i = 0; i < M; ++i) {
        const scalar_t decay = static_cast<scalar_t>(1) - f_t[i];
        scalar_t gf_acc = static_cast<scalar_t>(0);
        scalar_t gk_acc = static_cast<scalar_t>(0);
        for (int64_t j = 0; j < M; ++j) {
          const int64_t idx = i * M + j;
          const scalar_t s_rm1 = s_hist[(r - 1) * M * M + idx];
          const scalar_t delta = k_t[i] * err[j] * scale;
          const scalar_t raw = momentum * s_rm1 + w * delta;
          const scalar_t adenom = static_cast<scalar_t>(1) + balance * std::abs(raw);
          const scalar_t s_r_v = s_hist[r * M * M + idx];

          const scalar_t g_mem_out = p_r * GM[idx];
          gmem_o[idx] += decay * g_mem_out;
          gf_acc += -mem[idx] * g_mem_out;

          const scalar_t g_s_r =
              p_r * GS[idx] + g_mem_out + gs_cur[idx] +
              (s_r_v >= static_cast<scalar_t>(0) ? inv_mm : -inv_mm) * gf0;
          const scalar_t g_raw = g_s_r / (adenom * adenom);
          *gbal += -g_s_r * raw * std::abs(raw) / (adenom * adenom);
          *gmom += g_raw * s_rm1;
          gs_next[idx] += g_raw * momentum;
          const scalar_t g_delta = g_raw * w;
          gk_acc += g_delta * err[j] * scale;
          gerr[j] += g_delta * k_t[i] * scale;
          g_write_acc += g_raw * delta;
        }
        gforget[t * M + i] += gf_acc;
        gk[t * M + i] += gk_acc;
      }
      gwrite[t] += g_write_acc;

      // feature f1 = mean|err|
      const scalar_t inv_m = static_cast<scalar_t>(1) / static_cast<scalar_t>(M);
      for (int64_t j = 0; j < M; ++j) {
        gerr[j] +=
            (err[j] >= static_cast<scalar_t>(0) ? inv_m : -inv_m) * gf1;
      }

      // err = v - SR(mem_in[r], k_t)
      for (int64_t j = 0; j < M; ++j) {
        const scalar_t ge = gerr[j];
        if (ge == static_cast<scalar_t>(0)) continue;
        gv[t * M + j] += ge;
        semiring_read_col_grad(
            mr_in, k_t, beta, j, M, logM, -ge, gk + t * M, gbeta,
            [&](int64_t ii, scalar_t g) {
              if (r1) {
                gmem_o[ii * M + j] += g;
              } else {
                gmem_o[ii * M + j] += (static_cast<scalar_t>(1) - f_t[ii]) * g;
                gforget[t * M + ii] += -mem[ii * M + j] * g;
                gs_next[ii * M + j] += g;
              }
            });
      }

      gs_cur.swap(gs_next);
    }
    // gs_cur now holds grad wrt s_hist[0] = S_{t-1}
    for (int64_t n = 0; n < M * M; ++n) gsur_o[n] += gs_cur[n];

    GM.swap(gmem_o);
    GS.swap(gsur_o);
  }
}

void check_cpu(const torch::Tensor& t, const char* n) {
  TORCH_CHECK(t.device().is_cpu(), n, " must be CPU");
}

}  // namespace

std::vector<torch::Tensor> mor_refine_forward(
    torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor write,
    torch::Tensor forget, double momentum, double beta, double balance,
    int64_t R, torch::Tensor W1, torch::Tensor b1, torch::Tensor W2,
    double b2, double a_coupling) {
  check_cpu(q, "q");
  auto q_c = q.contiguous();
  auto k_c = k.contiguous();
  auto v_c = v.contiguous();
  auto write_c = write.contiguous();
  auto forget_c = forget.contiguous();
  auto W1_c = W1.contiguous();
  auto b1_c = b1.contiguous();
  auto W2_c = W2.contiguous();
  const int64_t B = q_c.size(0), L = q_c.size(1), M = q_c.size(2);
  const int64_t H = W1_c.size(0);
  TORCH_CHECK(M <= MAXM && H <= MAXH && R >= 1 && R <= MAXR, "dim out of range");
  auto y = torch::empty_like(q_c);
  auto depth = torch::zeros({B, L}, q_c.options());
  auto hist = torch::zeros({B, R}, q_c.options());
  auto mem_prev = torch::empty({B, L, M, M}, q_c.options());
  auto sur_prev = torch::empty({B, L, M, M}, q_c.options());

  AT_DISPATCH_FLOATING_TYPES(q_c.scalar_type(), "mor_refine_forward_cpu", [&] {
    at::parallel_for(0, B, 1, [&](int64_t begin, int64_t end) {
      for (int64_t b = begin; b < end; ++b) {
        mor_refine_forward_one<scalar_t>(
            q_c.data_ptr<scalar_t>() + b * L * M,
            k_c.data_ptr<scalar_t>() + b * L * M,
            v_c.data_ptr<scalar_t>() + b * L * M,
            write_c.data_ptr<scalar_t>() + b * L,
            forget_c.data_ptr<scalar_t>() + b * L * M,
            static_cast<scalar_t>(momentum), static_cast<scalar_t>(beta),
            static_cast<scalar_t>(balance), R, W1_c.data_ptr<scalar_t>(),
            b1_c.data_ptr<scalar_t>(), W2_c.data_ptr<scalar_t>(),
            static_cast<scalar_t>(b2), static_cast<scalar_t>(a_coupling), H,
            y.data_ptr<scalar_t>() + b * L * M,
            depth.data_ptr<scalar_t>() + b * L,
            hist.data_ptr<scalar_t>() + b * R,
            mem_prev.data_ptr<scalar_t>() + b * L * M * M,
            sur_prev.data_ptr<scalar_t>() + b * L * M * M, L, M);
      }
    });
  });
  return {y, depth, hist, mem_prev, sur_prev};
}

std::vector<torch::Tensor> mor_refine_backward(
    torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor write,
    torch::Tensor forget, double momentum, double beta, double balance,
    int64_t R, torch::Tensor W1, torch::Tensor b1, torch::Tensor W2, double b2,
    double a_coupling, torch::Tensor grad_y, torch::Tensor grad_depth,
    torch::Tensor mem_prev, torch::Tensor sur_prev) {
  check_cpu(q, "q");
  auto q_c = q.contiguous();
  auto k_c = k.contiguous();
  auto v_c = v.contiguous();
  auto write_c = write.contiguous();
  auto forget_c = forget.contiguous();
  auto W1_c = W1.contiguous();
  auto b1_c = b1.contiguous();
  auto W2_c = W2.contiguous();
  auto grad_y_c = grad_y.contiguous();
  auto grad_depth_c = grad_depth.contiguous();
  auto mem_c = mem_prev.contiguous();
  auto sur_c = sur_prev.contiguous();
  const int64_t B = q_c.size(0), L = q_c.size(1), M = q_c.size(2);
  const int64_t H = W1_c.size(0);
  auto gq = torch::zeros_like(q_c), gk = torch::zeros_like(q_c),
       gv = torch::zeros_like(q_c), gf = torch::zeros_like(q_c);
  auto gw = torch::zeros({B, L}, q_c.options());
  // Router/scalar grads are shared across the batch: accumulate per-b buffers
  // (the batch loop runs on ATen's threadpool) and reduce after.
  auto gmom_b = torch::zeros({B}, q_c.options());
  auto gbeta_b = torch::zeros({B}, q_c.options());
  auto gbal_b = torch::zeros({B}, q_c.options());
  auto gW1_b = torch::zeros({B, H, 3}, q_c.options());
  auto gb1_b = torch::zeros({B, H}, q_c.options());
  auto gW2_b = torch::zeros({B, H}, q_c.options());
  auto gb2_b = torch::zeros({B}, q_c.options());
  auto g_a_b = torch::zeros({B}, q_c.options());

  AT_DISPATCH_FLOATING_TYPES(q_c.scalar_type(), "mor_refine_backward_cpu", [&] {
    at::parallel_for(0, B, 1, [&](int64_t begin, int64_t end) {
      for (int64_t b = begin; b < end; ++b) {
        mor_refine_backward_one<scalar_t>(
            q_c.data_ptr<scalar_t>() + b * L * M,
            k_c.data_ptr<scalar_t>() + b * L * M,
            v_c.data_ptr<scalar_t>() + b * L * M,
            write_c.data_ptr<scalar_t>() + b * L,
            forget_c.data_ptr<scalar_t>() + b * L * M,
            static_cast<scalar_t>(momentum), static_cast<scalar_t>(beta),
            static_cast<scalar_t>(balance), R, W1_c.data_ptr<scalar_t>(),
            b1_c.data_ptr<scalar_t>(), W2_c.data_ptr<scalar_t>(),
            static_cast<scalar_t>(b2), static_cast<scalar_t>(a_coupling), H,
            grad_y_c.data_ptr<scalar_t>() + b * L * M,
            grad_depth_c.data_ptr<scalar_t>() + b * L,
            mem_c.data_ptr<scalar_t>() + b * L * M * M,
            sur_c.data_ptr<scalar_t>() + b * L * M * M,
            gq.data_ptr<scalar_t>() + b * L * M,
            gk.data_ptr<scalar_t>() + b * L * M,
            gv.data_ptr<scalar_t>() + b * L * M,
            gw.data_ptr<scalar_t>() + b * L,
            gf.data_ptr<scalar_t>() + b * L * M,
            gmom_b.data_ptr<scalar_t>() + b, gbeta_b.data_ptr<scalar_t>() + b,
            gbal_b.data_ptr<scalar_t>() + b,
            gW1_b.data_ptr<scalar_t>() + b * H * 3,
            gb1_b.data_ptr<scalar_t>() + b * H,
            gW2_b.data_ptr<scalar_t>() + b * H,
            gb2_b.data_ptr<scalar_t>() + b, g_a_b.data_ptr<scalar_t>() + b, L,
            M);
      }
    });
  });
  return {gq, gk, gv, gw, gf, gmom_b.sum(), gbeta_b.sum(), gbal_b.sum(),
          gW1_b.sum(0), gb1_b.sum(0), gW2_b.sum(0), gb2_b.sum(), g_a_b.sum()};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("mor_refine_forward", &mor_refine_forward,
        "CPU MoR refine scan forward (port of the CUDA kernel)");
  m.def("mor_refine_backward", &mor_refine_backward,
        "CPU MoR refine scan backward (port of the CUDA kernel)");
}
