// CPU-native POST-WRITE-READ surprise scans: exact ports of the pure-Python
// `_SurpriseMemoryBase._delta_step` loop in memory_primitives.py, used by
// TropicalSurpriseMemoryLane, SemiringSurpriseMemoryLane, and the per-level
// scans of PadicSurpriseMemoryLane.
//
// These differ from native_surprise_memory.cpp on BOTH axes that define that
// family's mechanism, so they cannot reuse its entry points:
//   * retrieval algebra: tropical max-plus / tempered log-sum-exp, NOT the
//     Euclidean dot read (which collapses induction for this family);
//   * read timing: the readout at step t reads M_t AFTER the delta write
//     (native_surprise_memory.cpp reads M_{t-1} before the write).
// The tempered-LSE read helpers mirror semiring_read/semiring_read_backward in
// native_surprise_memory.cpp; they are re-declared here because each JIT
// extension is version-hashed on its single listed source file, so a shared
// header would not trigger rebuilds when edited.
//
// Per step t (matching _delta_step + _read exactly):
//   pred  = read(M_{t-1}, k_t)
//   err   = v_t - pred
//   delta = k_t (outer) err * M^{-1/2}
//   S_t   = momentum * S_{t-1} + write_t * delta
//   M_t   = (1 - forget_t) (row-wise) * M_{t-1} + S_t
//   y_t   = read(M_t, q_t)            <-- post-write read

#include <torch/extension.h>
#include <ATen/Parallel.h>

#include <cmath>
#include <limits>
#include <vector>

namespace {

void check_inputs(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& write,
    const torch::Tensor& forget,
    const torch::Tensor& momentum) {
  TORCH_CHECK(q.device().is_cpu(), "q must be CPU");
  TORCH_CHECK(k.device().is_cpu(), "k must be CPU");
  TORCH_CHECK(v.device().is_cpu(), "v must be CPU");
  TORCH_CHECK(write.device().is_cpu(), "write must be CPU");
  TORCH_CHECK(forget.device().is_cpu(), "forget must be CPU");
  TORCH_CHECK(momentum.device().is_cpu(), "momentum must be CPU");
  TORCH_CHECK(q.dim() == 3, "q must be [B,L,M]");
  TORCH_CHECK(k.sizes() == q.sizes(), "k shape must match q");
  TORCH_CHECK(v.sizes() == q.sizes(), "v shape must match q");
  TORCH_CHECK(write.dim() == 2, "write must be [B,L]");
  TORCH_CHECK(forget.sizes() == q.sizes(), "forget shape must match q");
  TORCH_CHECK(write.size(0) == q.size(0) && write.size(1) == q.size(1),
              "write shape must be [B,L]");
  TORCH_CHECK(momentum.numel() == 1, "momentum must be scalar");
}

// read[j] = max_i (memory[i,j] + addr[i]); argmax saved for the backward.
template <typename scalar_t>
void tropical_read(
    const scalar_t* memory,
    const scalar_t* addr,
    scalar_t* out,
    int64_t* argmax,
    int64_t M) {
  for (int64_t j = 0; j < M; ++j) {
    scalar_t best = -std::numeric_limits<scalar_t>::infinity();
    int64_t best_i = 0;
    for (int64_t i = 0; i < M; ++i) {
      const scalar_t z = memory[i * M + j] + addr[i];
      if (z > best) {
        best = z;
        best_i = i;
      }
    }
    out[j] = best;
    argmax[j] = best_i;
  }
}

// read[j] = (logsumexp_i(beta*(memory[i,j]+addr[i])) - log M) / beta
template <typename scalar_t>
void lse_read(
    const scalar_t* memory,
    const scalar_t* addr,
    scalar_t beta,
    scalar_t* out,
    int64_t M) {
  const scalar_t log_m = std::log(static_cast<scalar_t>(M));
  for (int64_t j = 0; j < M; ++j) {
    scalar_t max_z = -std::numeric_limits<scalar_t>::infinity();
    for (int64_t i = 0; i < M; ++i) {
      max_z = std::max(max_z, beta * (memory[i * M + j] + addr[i]));
    }
    scalar_t sum_exp = static_cast<scalar_t>(0);
    for (int64_t i = 0; i < M; ++i) {
      sum_exp += std::exp(beta * (memory[i * M + j] + addr[i]) - max_z);
    }
    const scalar_t lse = max_z + std::log(sum_exp);
    out[j] = (lse - log_m) / beta;
  }
}

template <typename scalar_t>
void lse_read_grad(
    const scalar_t* memory,
    const scalar_t* addr,
    scalar_t beta,
    const scalar_t* grad_out,
    scalar_t* grad_memory,
    scalar_t* grad_addr,
    scalar_t* grad_beta,
    int64_t M) {
  const scalar_t log_m = std::log(static_cast<scalar_t>(M));
  std::vector<scalar_t> weights(M);
  for (int64_t j = 0; j < M; ++j) {
    scalar_t max_z = -std::numeric_limits<scalar_t>::infinity();
    for (int64_t i = 0; i < M; ++i) {
      max_z = std::max(max_z, beta * (memory[i * M + j] + addr[i]));
    }
    scalar_t sum_exp = static_cast<scalar_t>(0);
    for (int64_t i = 0; i < M; ++i) {
      const scalar_t e = std::exp(beta * (memory[i * M + j] + addr[i]) - max_z);
      weights[i] = e;
      sum_exp += e;
    }
    const scalar_t lse = max_z + std::log(sum_exp);
    scalar_t expected_score = static_cast<scalar_t>(0);
    for (int64_t i = 0; i < M; ++i) {
      weights[i] /= sum_exp;
      expected_score += weights[i] * (memory[i * M + j] + addr[i]);
    }
    const scalar_t go = grad_out[j];
    *grad_beta += go * (beta * expected_score - (lse - log_m)) / (beta * beta);
    for (int64_t i = 0; i < M; ++i) {
      const scalar_t g = go * weights[i];
      grad_addr[i] += g;
      grad_memory[i * M + j] += g;
    }
  }
}

// Recompute one delta step from the saved pre-step state. Fills err, s_new,
// m_new; pred/argmax buffers are the caller's (tropical only uses pred_argmax).
template <typename scalar_t>
void recompute_step(
    const scalar_t* k_t,
    const scalar_t* v_t,
    const scalar_t* f_t,
    scalar_t w,
    scalar_t momentum,
    scalar_t scale,
    const scalar_t* mem_t,
    const scalar_t* surprise_t,
    const scalar_t* pred,
    scalar_t* err,
    scalar_t* s_new,
    scalar_t* m_new,
    int64_t M) {
  for (int64_t j = 0; j < M; ++j) {
    err[j] = v_t[j] - pred[j];
  }
  for (int64_t i = 0; i < M; ++i) {
    const scalar_t decay = static_cast<scalar_t>(1) - f_t[i];
    for (int64_t j = 0; j < M; ++j) {
      const int64_t idx = i * M + j;
      const scalar_t delta = k_t[i] * err[j] * scale;
      const scalar_t s = momentum * surprise_t[idx] + w * delta;
      s_new[idx] = s;
      m_new[idx] = decay * mem_t[idx] + s;
    }
  }
}

// Shared backward core through the write/decay recurrence (everything except
// the two reads, which the tropical/semiring callers inject themselves).
// On entry grad_memory must already include the readout contribution dL/dM_t;
// on exit grad_memory_prev holds the recurrence part of dL/dM_{t-1} (the pred
// read contribution is added by the caller) and grad_err the delta-path
// gradient dL/derr.
template <typename scalar_t>
void backward_step_core(
    const scalar_t* k_t,
    const scalar_t* f_t,
    scalar_t w,
    scalar_t momentum,
    scalar_t scale,
    const scalar_t* mem_t,
    const scalar_t* surprise_t,
    const scalar_t* err,
    const scalar_t* grad_memory,
    const scalar_t* grad_surprise,
    scalar_t* grad_memory_prev,
    scalar_t* grad_surprise_prev,
    scalar_t* grad_err,
    scalar_t* gk_t,
    scalar_t* gf_t,
    scalar_t* grad_write_t,
    scalar_t* grad_momentum,
    int64_t M) {
  scalar_t grad_w = static_cast<scalar_t>(0);
  scalar_t local_grad_momentum = static_cast<scalar_t>(0);
  for (int64_t i = 0; i < M; ++i) {
    const scalar_t decay = static_cast<scalar_t>(1) - f_t[i];
    scalar_t grad_decay = static_cast<scalar_t>(0);
    scalar_t gk_acc = static_cast<scalar_t>(0);
    for (int64_t j = 0; j < M; ++j) {
      const int64_t idx = i * M + j;
      const scalar_t gm = grad_memory[idx];
      grad_decay += gm * mem_t[idx];
      grad_memory_prev[idx] += gm * decay;
      const scalar_t gs = grad_surprise[idx] + gm;  // S_t feeds M_t directly
      local_grad_momentum += gs * surprise_t[idx];
      grad_surprise_prev[idx] += gs * momentum;
      const scalar_t delta = k_t[i] * err[j] * scale;
      grad_w += gs * delta;
      const scalar_t gdelta = gs * w;
      gk_acc += gdelta * err[j] * scale;
      grad_err[j] += gdelta * k_t[i] * scale;
    }
    gf_t[i] -= grad_decay;
    gk_t[i] += gk_acc;
  }
  *grad_write_t += grad_w;
  *grad_momentum += local_grad_momentum;
}

template <typename scalar_t>
void tropical_forward_one(
    const scalar_t* q,
    const scalar_t* k,
    const scalar_t* v,
    const scalar_t* write,
    const scalar_t* forget,
    const scalar_t momentum,
    scalar_t* y,
    scalar_t* mem_prev,
    scalar_t* surprise_prev,
    int64_t L,
    int64_t M) {
  const scalar_t scale =
      static_cast<scalar_t>(1.0 / std::sqrt(static_cast<double>(M)));
  std::vector<scalar_t> memory(M * M, static_cast<scalar_t>(0));
  std::vector<scalar_t> surprise(M * M, static_cast<scalar_t>(0));
  std::vector<scalar_t> pred(M);
  std::vector<scalar_t> err(M);
  std::vector<scalar_t> m_new(M * M);
  std::vector<scalar_t> s_new(M * M);
  std::vector<int64_t> argmax(M);

  for (int64_t t = 0; t < L; ++t) {
    std::copy(memory.begin(), memory.end(), mem_prev + t * M * M);
    std::copy(surprise.begin(), surprise.end(), surprise_prev + t * M * M);

    tropical_read(memory.data(), k + t * M, pred.data(), argmax.data(), M);
    recompute_step(
        k + t * M, v + t * M, forget + t * M, write[t], momentum, scale,
        memory.data(), surprise.data(), pred.data(), err.data(),
        s_new.data(), m_new.data(), M);
    tropical_read(m_new.data(), q + t * M, y + t * M, argmax.data(), M);

    memory.swap(m_new);
    surprise.swap(s_new);
  }
}

template <typename scalar_t>
void tropical_backward_one(
    const scalar_t* q,
    const scalar_t* k,
    const scalar_t* v,
    const scalar_t* write,
    const scalar_t* forget,
    const scalar_t momentum,
    const scalar_t* grad_y,
    const scalar_t* mem_prev,
    const scalar_t* surprise_prev,
    scalar_t* grad_q,
    scalar_t* grad_k,
    scalar_t* grad_v,
    scalar_t* grad_write,
    scalar_t* grad_forget,
    scalar_t* grad_momentum,
    int64_t L,
    int64_t M) {
  const scalar_t scale =
      static_cast<scalar_t>(1.0 / std::sqrt(static_cast<double>(M)));
  std::vector<scalar_t> grad_memory(M * M, static_cast<scalar_t>(0));
  std::vector<scalar_t> grad_surprise(M * M, static_cast<scalar_t>(0));
  std::vector<scalar_t> grad_memory_prev(M * M);
  std::vector<scalar_t> grad_surprise_prev(M * M);
  std::vector<scalar_t> pred(M);
  std::vector<scalar_t> err(M);
  std::vector<scalar_t> s_new(M * M);
  std::vector<scalar_t> m_new(M * M);
  std::vector<scalar_t> grad_err(M);
  std::vector<int64_t> pred_argmax(M);
  std::vector<int64_t> read_argmax(M);
  std::vector<scalar_t> read_val(M);

  for (int64_t t = L - 1; t >= 0; --t) {
    const scalar_t* q_t = q + t * M;
    const scalar_t* k_t = k + t * M;
    const scalar_t* f_t = forget + t * M;
    const scalar_t* gy_t = grad_y + t * M;
    const scalar_t* mem_t = mem_prev + t * M * M;
    const scalar_t* surprise_t = surprise_prev + t * M * M;
    scalar_t* gq_t = grad_q + t * M;
    scalar_t* gk_t = grad_k + t * M;
    scalar_t* gv_t = grad_v + t * M;
    scalar_t* gf_t = grad_forget + t * M;

    tropical_read(mem_t, k_t, pred.data(), pred_argmax.data(), M);
    recompute_step(
        k_t, v + t * M, f_t, write[t], momentum, scale, mem_t, surprise_t,
        pred.data(), err.data(), s_new.data(), m_new.data(), M);
    tropical_read(m_new.data(), q_t, read_val.data(), read_argmax.data(), M);

    std::fill(grad_memory_prev.begin(), grad_memory_prev.end(),
              static_cast<scalar_t>(0));
    std::fill(grad_surprise_prev.begin(), grad_surprise_prev.end(),
              static_cast<scalar_t>(0));
    std::fill(grad_err.begin(), grad_err.end(), static_cast<scalar_t>(0));

    // y_t[j] = m_new[i*,j] + q_t[i*] with i* = read_argmax[j]
    for (int64_t j = 0; j < M; ++j) {
      const scalar_t gy = gy_t[j];
      grad_memory[read_argmax[j] * M + j] += gy;
      gq_t[read_argmax[j]] += gy;
    }

    backward_step_core(
        k_t, f_t, write[t], momentum, scale, mem_t, surprise_t, err.data(),
        grad_memory.data(), grad_surprise.data(), grad_memory_prev.data(),
        grad_surprise_prev.data(), grad_err.data(), gk_t, gf_t,
        grad_write + t, grad_momentum, M);

    // err = v - pred; pred[j] = mem_t[i*,j] + k_t[i*] with i* = pred_argmax[j]
    for (int64_t j = 0; j < M; ++j) {
      const scalar_t ge = grad_err[j];
      gv_t[j] += ge;
      grad_memory_prev[pred_argmax[j] * M + j] -= ge;
      gk_t[pred_argmax[j]] -= ge;
    }

    grad_memory.swap(grad_memory_prev);
    grad_surprise.swap(grad_surprise_prev);
  }
}

template <typename scalar_t>
void semiring_postread_forward_one(
    const scalar_t* q,
    const scalar_t* k,
    const scalar_t* v,
    const scalar_t* write,
    const scalar_t* forget,
    const scalar_t momentum,
    const scalar_t beta,
    scalar_t* y,
    scalar_t* mem_prev,
    scalar_t* surprise_prev,
    int64_t L,
    int64_t M) {
  const scalar_t scale =
      static_cast<scalar_t>(1.0 / std::sqrt(static_cast<double>(M)));
  std::vector<scalar_t> memory(M * M, static_cast<scalar_t>(0));
  std::vector<scalar_t> surprise(M * M, static_cast<scalar_t>(0));
  std::vector<scalar_t> pred(M);
  std::vector<scalar_t> err(M);
  std::vector<scalar_t> m_new(M * M);
  std::vector<scalar_t> s_new(M * M);

  for (int64_t t = 0; t < L; ++t) {
    std::copy(memory.begin(), memory.end(), mem_prev + t * M * M);
    std::copy(surprise.begin(), surprise.end(), surprise_prev + t * M * M);

    lse_read(memory.data(), k + t * M, beta, pred.data(), M);
    recompute_step(
        k + t * M, v + t * M, forget + t * M, write[t], momentum, scale,
        memory.data(), surprise.data(), pred.data(), err.data(),
        s_new.data(), m_new.data(), M);
    lse_read(m_new.data(), q + t * M, beta, y + t * M, M);

    memory.swap(m_new);
    surprise.swap(s_new);
  }
}

template <typename scalar_t>
void semiring_postread_backward_one(
    const scalar_t* q,
    const scalar_t* k,
    const scalar_t* v,
    const scalar_t* write,
    const scalar_t* forget,
    const scalar_t momentum,
    const scalar_t beta,
    const scalar_t* grad_y,
    const scalar_t* mem_prev,
    const scalar_t* surprise_prev,
    scalar_t* grad_q,
    scalar_t* grad_k,
    scalar_t* grad_v,
    scalar_t* grad_write,
    scalar_t* grad_forget,
    scalar_t* grad_momentum,
    scalar_t* grad_beta,
    int64_t L,
    int64_t M) {
  const scalar_t scale =
      static_cast<scalar_t>(1.0 / std::sqrt(static_cast<double>(M)));
  std::vector<scalar_t> grad_memory(M * M, static_cast<scalar_t>(0));
  std::vector<scalar_t> grad_surprise(M * M, static_cast<scalar_t>(0));
  std::vector<scalar_t> grad_memory_prev(M * M);
  std::vector<scalar_t> grad_surprise_prev(M * M);
  std::vector<scalar_t> pred(M);
  std::vector<scalar_t> err(M);
  std::vector<scalar_t> s_new(M * M);
  std::vector<scalar_t> m_new(M * M);
  std::vector<scalar_t> grad_err(M);
  std::vector<scalar_t> grad_pred(M);

  for (int64_t t = L - 1; t >= 0; --t) {
    const scalar_t* q_t = q + t * M;
    const scalar_t* k_t = k + t * M;
    const scalar_t* f_t = forget + t * M;
    const scalar_t* gy_t = grad_y + t * M;
    const scalar_t* mem_t = mem_prev + t * M * M;
    const scalar_t* surprise_t = surprise_prev + t * M * M;
    scalar_t* gq_t = grad_q + t * M;
    scalar_t* gk_t = grad_k + t * M;
    scalar_t* gv_t = grad_v + t * M;
    scalar_t* gf_t = grad_forget + t * M;

    lse_read(mem_t, k_t, beta, pred.data(), M);
    recompute_step(
        k_t, v + t * M, f_t, write[t], momentum, scale, mem_t, surprise_t,
        pred.data(), err.data(), s_new.data(), m_new.data(), M);

    std::fill(grad_memory_prev.begin(), grad_memory_prev.end(),
              static_cast<scalar_t>(0));
    std::fill(grad_surprise_prev.begin(), grad_surprise_prev.end(),
              static_cast<scalar_t>(0));
    std::fill(grad_err.begin(), grad_err.end(), static_cast<scalar_t>(0));
    std::fill(grad_pred.begin(), grad_pred.end(), static_cast<scalar_t>(0));

    // y_t = lse_read(M_t, q_t): inject dL/dM_t and dL/dq_t (+ beta)
    lse_read_grad(m_new.data(), q_t, beta, gy_t, grad_memory.data(), gq_t,
                  grad_beta, M);

    backward_step_core(
        k_t, f_t, write[t], momentum, scale, mem_t, surprise_t, err.data(),
        grad_memory.data(), grad_surprise.data(), grad_memory_prev.data(),
        grad_surprise_prev.data(), grad_err.data(), gk_t, gf_t,
        grad_write + t, grad_momentum, M);

    // err = v - pred; pred = lse_read(M_{t-1}, k_t)
    for (int64_t j = 0; j < M; ++j) {
      gv_t[j] += grad_err[j];
      grad_pred[j] = -grad_err[j];
    }
    lse_read_grad(mem_t, k_t, beta, grad_pred.data(),
                  grad_memory_prev.data(), gk_t, grad_beta, M);

    grad_memory.swap(grad_memory_prev);
    grad_surprise.swap(grad_surprise_prev);
  }
}

}  // namespace

std::vector<torch::Tensor> tropical_postread_forward(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor write,
    torch::Tensor forget,
    torch::Tensor momentum) {
  check_inputs(q, k, v, write, forget, momentum);
  auto q_c = q.contiguous();
  auto k_c = k.contiguous();
  auto v_c = v.contiguous();
  auto write_c = write.contiguous();
  auto forget_c = forget.contiguous();
  const auto B = q_c.size(0);
  const auto L = q_c.size(1);
  const auto M = q_c.size(2);
  auto y = torch::empty_like(q_c);
  auto mem_prev = torch::empty({B, L, M, M}, q_c.options());
  auto surprise_prev = torch::empty({B, L, M, M}, q_c.options());

  AT_DISPATCH_FLOATING_TYPES(q_c.scalar_type(), "tropical_postread_forward", [&] {
    const scalar_t mom = momentum.item<scalar_t>();
    at::parallel_for(0, B, 1, [&](int64_t begin, int64_t end) {
      for (int64_t b = begin; b < end; ++b) {
        tropical_forward_one<scalar_t>(
            q_c.data_ptr<scalar_t>() + b * L * M,
            k_c.data_ptr<scalar_t>() + b * L * M,
            v_c.data_ptr<scalar_t>() + b * L * M,
            write_c.data_ptr<scalar_t>() + b * L,
            forget_c.data_ptr<scalar_t>() + b * L * M,
            mom,
            y.data_ptr<scalar_t>() + b * L * M,
            mem_prev.data_ptr<scalar_t>() + b * L * M * M,
            surprise_prev.data_ptr<scalar_t>() + b * L * M * M,
            L,
            M);
      }
    });
  });

  return {y, mem_prev, surprise_prev};
}

std::vector<torch::Tensor> tropical_postread_backward(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor write,
    torch::Tensor forget,
    torch::Tensor momentum,
    torch::Tensor grad_y,
    torch::Tensor mem_prev,
    torch::Tensor surprise_prev) {
  check_inputs(q, k, v, write, forget, momentum);
  TORCH_CHECK(grad_y.sizes() == q.sizes(), "grad_y shape must match q");
  auto q_c = q.contiguous();
  auto k_c = k.contiguous();
  auto v_c = v.contiguous();
  auto write_c = write.contiguous();
  auto forget_c = forget.contiguous();
  auto grad_y_c = grad_y.contiguous();
  auto mem_c = mem_prev.contiguous();
  auto surprise_c = surprise_prev.contiguous();
  const auto B = q_c.size(0);
  const auto L = q_c.size(1);
  const auto M = q_c.size(2);
  auto grad_q = torch::zeros_like(q_c);
  auto grad_k = torch::zeros_like(k_c);
  auto grad_v = torch::zeros_like(v_c);
  auto grad_write = torch::zeros_like(write_c);
  auto grad_forget = torch::zeros_like(forget_c);
  auto grad_momentum_per_b = torch::zeros({B}, q_c.options());

  AT_DISPATCH_FLOATING_TYPES(q_c.scalar_type(), "tropical_postread_backward", [&] {
    const scalar_t mom = momentum.item<scalar_t>();
    at::parallel_for(0, B, 1, [&](int64_t begin, int64_t end) {
      for (int64_t b = begin; b < end; ++b) {
        tropical_backward_one<scalar_t>(
            q_c.data_ptr<scalar_t>() + b * L * M,
            k_c.data_ptr<scalar_t>() + b * L * M,
            v_c.data_ptr<scalar_t>() + b * L * M,
            write_c.data_ptr<scalar_t>() + b * L,
            forget_c.data_ptr<scalar_t>() + b * L * M,
            mom,
            grad_y_c.data_ptr<scalar_t>() + b * L * M,
            mem_c.data_ptr<scalar_t>() + b * L * M * M,
            surprise_c.data_ptr<scalar_t>() + b * L * M * M,
            grad_q.data_ptr<scalar_t>() + b * L * M,
            grad_k.data_ptr<scalar_t>() + b * L * M,
            grad_v.data_ptr<scalar_t>() + b * L * M,
            grad_write.data_ptr<scalar_t>() + b * L,
            grad_forget.data_ptr<scalar_t>() + b * L * M,
            grad_momentum_per_b.data_ptr<scalar_t>() + b,
            L,
            M);
      }
    });
  });

  auto grad_momentum = grad_momentum_per_b.sum().reshape_as(momentum);
  return {grad_q, grad_k, grad_v, grad_write, grad_forget, grad_momentum};
}

std::vector<torch::Tensor> semiring_postread_forward(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor write,
    torch::Tensor forget,
    torch::Tensor momentum,
    torch::Tensor beta) {
  check_inputs(q, k, v, write, forget, momentum);
  TORCH_CHECK(beta.device().is_cpu(), "beta must be CPU");
  TORCH_CHECK(beta.numel() == 1, "beta must be scalar");
  auto q_c = q.contiguous();
  auto k_c = k.contiguous();
  auto v_c = v.contiguous();
  auto write_c = write.contiguous();
  auto forget_c = forget.contiguous();
  const auto B = q_c.size(0);
  const auto L = q_c.size(1);
  const auto M = q_c.size(2);
  auto y = torch::empty_like(q_c);
  auto mem_prev = torch::empty({B, L, M, M}, q_c.options());
  auto surprise_prev = torch::empty({B, L, M, M}, q_c.options());

  AT_DISPATCH_FLOATING_TYPES(
      q_c.scalar_type(), "semiring_postread_forward", [&] {
        const scalar_t mom = momentum.item<scalar_t>();
        const scalar_t bta = beta.item<scalar_t>();
        at::parallel_for(0, B, 1, [&](int64_t begin, int64_t end) {
          for (int64_t b = begin; b < end; ++b) {
            semiring_postread_forward_one<scalar_t>(
                q_c.data_ptr<scalar_t>() + b * L * M,
                k_c.data_ptr<scalar_t>() + b * L * M,
                v_c.data_ptr<scalar_t>() + b * L * M,
                write_c.data_ptr<scalar_t>() + b * L,
                forget_c.data_ptr<scalar_t>() + b * L * M,
                mom,
                bta,
                y.data_ptr<scalar_t>() + b * L * M,
                mem_prev.data_ptr<scalar_t>() + b * L * M * M,
                surprise_prev.data_ptr<scalar_t>() + b * L * M * M,
                L,
                M);
          }
        });
      });

  return {y, mem_prev, surprise_prev};
}

std::vector<torch::Tensor> semiring_postread_backward(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor write,
    torch::Tensor forget,
    torch::Tensor momentum,
    torch::Tensor beta,
    torch::Tensor grad_y,
    torch::Tensor mem_prev,
    torch::Tensor surprise_prev) {
  check_inputs(q, k, v, write, forget, momentum);
  TORCH_CHECK(beta.device().is_cpu(), "beta must be CPU");
  TORCH_CHECK(beta.numel() == 1, "beta must be scalar");
  TORCH_CHECK(grad_y.sizes() == q.sizes(), "grad_y shape must match q");
  auto q_c = q.contiguous();
  auto k_c = k.contiguous();
  auto v_c = v.contiguous();
  auto write_c = write.contiguous();
  auto forget_c = forget.contiguous();
  auto grad_y_c = grad_y.contiguous();
  auto mem_c = mem_prev.contiguous();
  auto surprise_c = surprise_prev.contiguous();
  const auto B = q_c.size(0);
  const auto L = q_c.size(1);
  const auto M = q_c.size(2);
  auto grad_q = torch::zeros_like(q_c);
  auto grad_k = torch::zeros_like(k_c);
  auto grad_v = torch::zeros_like(v_c);
  auto grad_write = torch::zeros_like(write_c);
  auto grad_forget = torch::zeros_like(forget_c);
  auto grad_momentum_per_b = torch::zeros({B}, q_c.options());
  auto grad_beta_per_b = torch::zeros({B}, q_c.options());

  AT_DISPATCH_FLOATING_TYPES(
      q_c.scalar_type(), "semiring_postread_backward", [&] {
        const scalar_t mom = momentum.item<scalar_t>();
        const scalar_t bta = beta.item<scalar_t>();
        at::parallel_for(0, B, 1, [&](int64_t begin, int64_t end) {
          for (int64_t b = begin; b < end; ++b) {
            semiring_postread_backward_one<scalar_t>(
                q_c.data_ptr<scalar_t>() + b * L * M,
                k_c.data_ptr<scalar_t>() + b * L * M,
                v_c.data_ptr<scalar_t>() + b * L * M,
                write_c.data_ptr<scalar_t>() + b * L,
                forget_c.data_ptr<scalar_t>() + b * L * M,
                mom,
                bta,
                grad_y_c.data_ptr<scalar_t>() + b * L * M,
                mem_c.data_ptr<scalar_t>() + b * L * M * M,
                surprise_c.data_ptr<scalar_t>() + b * L * M * M,
                grad_q.data_ptr<scalar_t>() + b * L * M,
                grad_k.data_ptr<scalar_t>() + b * L * M,
                grad_v.data_ptr<scalar_t>() + b * L * M,
                grad_write.data_ptr<scalar_t>() + b * L,
                grad_forget.data_ptr<scalar_t>() + b * L * M,
                grad_momentum_per_b.data_ptr<scalar_t>() + b,
                grad_beta_per_b.data_ptr<scalar_t>() + b,
                L,
                M);
          }
        });
      });

  auto grad_momentum = grad_momentum_per_b.sum().reshape_as(momentum);
  auto grad_beta = grad_beta_per_b.sum().reshape_as(beta);
  return {grad_q, grad_k, grad_v, grad_write, grad_forget, grad_momentum,
          grad_beta};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("tropical_forward", &tropical_postread_forward,
        "Post-write-read tropical surprise scan forward");
  m.def("tropical_backward", &tropical_postread_backward,
        "Post-write-read tropical surprise scan backward");
  m.def("semiring_forward", &semiring_postread_forward,
        "Post-write-read tempered-semiring surprise scan forward");
  m.def("semiring_backward", &semiring_postread_backward,
        "Post-write-read tempered-semiring surprise scan backward");
}
