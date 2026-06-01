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
  TORCH_CHECK(q.is_contiguous(), "q must be contiguous");
  TORCH_CHECK(k.is_contiguous(), "k must be contiguous");
  TORCH_CHECK(v.is_contiguous(), "v must be contiguous");
  TORCH_CHECK(write.is_contiguous(), "write must be contiguous");
  TORCH_CHECK(forget.is_contiguous(), "forget must be contiguous");
  TORCH_CHECK(q.dim() == 3, "q must be [B,L,M]");
  TORCH_CHECK(k.sizes() == q.sizes(), "k shape must match q");
  TORCH_CHECK(v.sizes() == q.sizes(), "v shape must match q");
  TORCH_CHECK(write.dim() == 2, "write must be [B,L]");
  TORCH_CHECK(forget.sizes() == q.sizes(), "forget shape must match q");
  TORCH_CHECK(write.size(0) == q.size(0) && write.size(1) == q.size(1),
              "write shape must be [B,L]");
  TORCH_CHECK(momentum.numel() == 1, "momentum must be scalar");
}

template <typename scalar_t>
void forward_one(
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
  const scalar_t scale = static_cast<scalar_t>(1.0 / std::sqrt(static_cast<double>(M)));
  std::vector<scalar_t> memory(M * M, static_cast<scalar_t>(0));
  std::vector<scalar_t> surprise(M * M, static_cast<scalar_t>(0));
  std::vector<scalar_t> pred(M);
  std::vector<scalar_t> err(M);
  std::vector<scalar_t> new_surprise(M * M);

  for (int64_t t = 0; t < L; ++t) {
    const scalar_t* q_t = q + t * M;
    const scalar_t* k_t = k + t * M;
    const scalar_t* v_t = v + t * M;
    const scalar_t* f_t = forget + t * M;
    scalar_t* y_t = y + t * M;
    scalar_t* mem_save = mem_prev + t * M * M;
    scalar_t* surprise_save = surprise_prev + t * M * M;

    std::copy(memory.begin(), memory.end(), mem_save);
    std::copy(surprise.begin(), surprise.end(), surprise_save);

    for (int64_t j = 0; j < M; ++j) {
      scalar_t q_sum = static_cast<scalar_t>(0);
      scalar_t k_sum = static_cast<scalar_t>(0);
      for (int64_t i = 0; i < M; ++i) {
        const scalar_t m_ij = memory[i * M + j];
        q_sum += q_t[i] * m_ij;
        k_sum += k_t[i] * m_ij;
      }
      y_t[j] = q_sum;
      pred[j] = k_sum;
      err[j] = v_t[j] - k_sum;
    }

    const scalar_t w = write[t];
    for (int64_t i = 0; i < M; ++i) {
      const scalar_t decay = static_cast<scalar_t>(1) - f_t[i];
      for (int64_t j = 0; j < M; ++j) {
        const int64_t idx = i * M + j;
        const scalar_t delta = k_t[i] * err[j] * scale;
        const scalar_t s_new = momentum * surprise[idx] + w * delta;
        new_surprise[idx] = s_new;
        memory[idx] = decay * memory[idx] + s_new;
      }
    }
    surprise.swap(new_surprise);
  }
}

template <typename scalar_t>
void backward_one(
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
  const scalar_t scale = static_cast<scalar_t>(1.0 / std::sqrt(static_cast<double>(M)));
  std::vector<scalar_t> grad_memory(M * M, static_cast<scalar_t>(0));
  std::vector<scalar_t> grad_surprise(M * M, static_cast<scalar_t>(0));
  std::vector<scalar_t> grad_memory_prev(M * M);
  std::vector<scalar_t> grad_surprise_prev(M * M);
  std::vector<scalar_t> pred(M);
  std::vector<scalar_t> err(M);
  std::vector<scalar_t> grad_pred(M);
  std::vector<scalar_t> grad_err(M);

  scalar_t local_grad_momentum = static_cast<scalar_t>(0);

  for (int64_t t = L - 1; t >= 0; --t) {
    std::fill(grad_memory_prev.begin(), grad_memory_prev.end(), static_cast<scalar_t>(0));
    std::fill(grad_surprise_prev.begin(), grad_surprise_prev.end(), static_cast<scalar_t>(0));
    std::fill(grad_pred.begin(), grad_pred.end(), static_cast<scalar_t>(0));
    std::fill(grad_err.begin(), grad_err.end(), static_cast<scalar_t>(0));

    const scalar_t* q_t = q + t * M;
    const scalar_t* k_t = k + t * M;
    const scalar_t* v_t = v + t * M;
    const scalar_t* w_t = write + t;
    const scalar_t* f_t = forget + t * M;
    const scalar_t* gy_t = grad_y + t * M;
    const scalar_t* mem_t = mem_prev + t * M * M;
    const scalar_t* surprise_t = surprise_prev + t * M * M;
    scalar_t* gq_t = grad_q + t * M;
    scalar_t* gk_t = grad_k + t * M;
    scalar_t* gv_t = grad_v + t * M;
    scalar_t* gf_t = grad_forget + t * M;

    for (int64_t j = 0; j < M; ++j) {
      scalar_t p = static_cast<scalar_t>(0);
      for (int64_t i = 0; i < M; ++i) {
        p += k_t[i] * mem_t[i * M + j];
      }
      pred[j] = p;
      err[j] = v_t[j] - p;
    }

    // y_t = q_t @ memory_prev
    for (int64_t i = 0; i < M; ++i) {
      scalar_t acc_q = static_cast<scalar_t>(0);
      for (int64_t j = 0; j < M; ++j) {
        const scalar_t gy = gy_t[j];
        acc_q += gy * mem_t[i * M + j];
        grad_memory_prev[i * M + j] += gy * q_t[i];
      }
      gq_t[i] += acc_q;
    }

    // memory_new = (1 - forget_i) * memory_prev + surprise_new
    for (int64_t i = 0; i < M; ++i) {
      const scalar_t decay = static_cast<scalar_t>(1) - f_t[i];
      scalar_t grad_decay = static_cast<scalar_t>(0);
      for (int64_t j = 0; j < M; ++j) {
        const int64_t idx = i * M + j;
        const scalar_t gm = grad_memory[idx];
        grad_decay += gm * mem_t[idx];
        grad_memory_prev[idx] += gm * decay;
        grad_surprise[idx] += gm;
      }
      gf_t[i] -= grad_decay;
    }

    // surprise_new = momentum * surprise_prev + write * delta
    scalar_t grad_w = static_cast<scalar_t>(0);
    for (int64_t i = 0; i < M; ++i) {
      for (int64_t j = 0; j < M; ++j) {
        const int64_t idx = i * M + j;
        const scalar_t gs = grad_surprise[idx];
        const scalar_t delta = k_t[i] * err[j] * scale;
        local_grad_momentum += gs * surprise_t[idx];
        grad_surprise_prev[idx] += gs * momentum;
        grad_w += gs * delta;
        const scalar_t gdelta = gs * (*w_t);
        gk_t[i] += gdelta * err[j] * scale;
        grad_err[j] += gdelta * k_t[i] * scale;
      }
    }
    grad_write[t] += grad_w;

    // err = v - pred
    for (int64_t j = 0; j < M; ++j) {
      gv_t[j] += grad_err[j];
      grad_pred[j] -= grad_err[j];
    }

    // pred = k_t @ memory_prev
    for (int64_t i = 0; i < M; ++i) {
      scalar_t acc_k = static_cast<scalar_t>(0);
      for (int64_t j = 0; j < M; ++j) {
        const scalar_t gp = grad_pred[j];
        acc_k += gp * mem_t[i * M + j];
        grad_memory_prev[i * M + j] += gp * k_t[i];
      }
      gk_t[i] += acc_k;
    }

    grad_memory.swap(grad_memory_prev);
    grad_surprise.swap(grad_surprise_prev);
  }

  *grad_momentum += local_grad_momentum;
}

template <typename scalar_t>
void semiring_read(
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
void semiring_read_backward(
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
    const scalar_t grad_beta_j =
        (beta * expected_score - (lse - log_m)) / (beta * beta);
    *grad_beta += go * grad_beta_j;
    for (int64_t i = 0; i < M; ++i) {
      const scalar_t g = go * weights[i];
      grad_addr[i] += g;
      grad_memory[i * M + j] += g;
    }
  }
}

template <typename scalar_t>
void semiring_forward_one(
    const scalar_t* q,
    const scalar_t* k,
    const scalar_t* v,
    const scalar_t* write,
    const scalar_t* forget,
    const scalar_t momentum,
    const scalar_t beta,
    const scalar_t balance,
    scalar_t* y,
    scalar_t* mem_prev,
    scalar_t* surprise_prev,
    int64_t L,
    int64_t M) {
  const scalar_t scale = static_cast<scalar_t>(1.0 / std::sqrt(static_cast<double>(M)));
  std::vector<scalar_t> memory(M * M, static_cast<scalar_t>(0));
  std::vector<scalar_t> surprise(M * M, static_cast<scalar_t>(0));
  std::vector<scalar_t> pred(M);
  std::vector<scalar_t> err(M);
  std::vector<scalar_t> new_surprise(M * M);

  for (int64_t t = 0; t < L; ++t) {
    const scalar_t* q_t = q + t * M;
    const scalar_t* k_t = k + t * M;
    const scalar_t* v_t = v + t * M;
    const scalar_t* f_t = forget + t * M;
    scalar_t* y_t = y + t * M;
    scalar_t* mem_save = mem_prev + t * M * M;
    scalar_t* surprise_save = surprise_prev + t * M * M;

    std::copy(memory.begin(), memory.end(), mem_save);
    std::copy(surprise.begin(), surprise.end(), surprise_save);

    semiring_read(memory.data(), q_t, beta, y_t, M);
    semiring_read(memory.data(), k_t, beta, pred.data(), M);
    for (int64_t j = 0; j < M; ++j) {
      err[j] = v_t[j] - pred[j];
    }

    const scalar_t w = write[t];
    for (int64_t i = 0; i < M; ++i) {
      const scalar_t decay = static_cast<scalar_t>(1) - f_t[i];
      for (int64_t j = 0; j < M; ++j) {
        const int64_t idx = i * M + j;
        const scalar_t delta = k_t[i] * err[j] * scale;
        const scalar_t raw_surprise = momentum * surprise[idx] + w * delta;
        const scalar_t denom =
            static_cast<scalar_t>(1) + balance * std::abs(raw_surprise);
        const scalar_t s_new = raw_surprise / denom;
        new_surprise[idx] = s_new;
        memory[idx] = decay * memory[idx] + s_new;
      }
    }
    surprise.swap(new_surprise);
  }
}

template <typename scalar_t>
void semiring_backward_one(
    const scalar_t* q,
    const scalar_t* k,
    const scalar_t* v,
    const scalar_t* write,
    const scalar_t* forget,
    const scalar_t momentum,
    const scalar_t beta,
    const scalar_t balance,
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
    scalar_t* grad_balance,
    int64_t L,
    int64_t M) {
  const scalar_t scale = static_cast<scalar_t>(1.0 / std::sqrt(static_cast<double>(M)));
  std::vector<scalar_t> grad_memory(M * M, static_cast<scalar_t>(0));
  std::vector<scalar_t> grad_surprise(M * M, static_cast<scalar_t>(0));
  std::vector<scalar_t> grad_memory_prev(M * M);
  std::vector<scalar_t> grad_surprise_prev(M * M);
  std::vector<scalar_t> pred(M);
  std::vector<scalar_t> err(M);
  std::vector<scalar_t> grad_pred(M);
  std::vector<scalar_t> grad_err(M);

  scalar_t local_grad_momentum = static_cast<scalar_t>(0);
  scalar_t local_grad_beta = static_cast<scalar_t>(0);
  scalar_t local_grad_balance = static_cast<scalar_t>(0);

  for (int64_t t = L - 1; t >= 0; --t) {
    std::fill(grad_memory_prev.begin(), grad_memory_prev.end(), static_cast<scalar_t>(0));
    std::fill(grad_surprise_prev.begin(), grad_surprise_prev.end(), static_cast<scalar_t>(0));
    std::fill(grad_pred.begin(), grad_pred.end(), static_cast<scalar_t>(0));
    std::fill(grad_err.begin(), grad_err.end(), static_cast<scalar_t>(0));

    const scalar_t* q_t = q + t * M;
    const scalar_t* k_t = k + t * M;
    const scalar_t* v_t = v + t * M;
    const scalar_t* w_t = write + t;
    const scalar_t* f_t = forget + t * M;
    const scalar_t* gy_t = grad_y + t * M;
    const scalar_t* mem_t = mem_prev + t * M * M;
    const scalar_t* surprise_t = surprise_prev + t * M * M;
    scalar_t* gq_t = grad_q + t * M;
    scalar_t* gk_t = grad_k + t * M;
    scalar_t* gv_t = grad_v + t * M;
    scalar_t* gf_t = grad_forget + t * M;

    semiring_read(mem_t, k_t, beta, pred.data(), M);
    for (int64_t j = 0; j < M; ++j) {
      err[j] = v_t[j] - pred[j];
    }

    semiring_read_backward(
        mem_t, q_t, beta, gy_t, grad_memory_prev.data(), gq_t, &local_grad_beta, M);

    for (int64_t i = 0; i < M; ++i) {
      const scalar_t decay = static_cast<scalar_t>(1) - f_t[i];
      scalar_t grad_decay = static_cast<scalar_t>(0);
      for (int64_t j = 0; j < M; ++j) {
        const int64_t idx = i * M + j;
        const scalar_t gm = grad_memory[idx];
        grad_decay += gm * mem_t[idx];
        grad_memory_prev[idx] += gm * decay;
        grad_surprise[idx] += gm;
      }
      gf_t[i] -= grad_decay;
    }

    scalar_t grad_w = static_cast<scalar_t>(0);
    for (int64_t i = 0; i < M; ++i) {
      for (int64_t j = 0; j < M; ++j) {
        const int64_t idx = i * M + j;
        const scalar_t gs = grad_surprise[idx];
        const scalar_t delta = k_t[i] * err[j] * scale;
        const scalar_t raw_surprise = momentum * surprise_t[idx] + (*w_t) * delta;
        const scalar_t abs_raw = std::abs(raw_surprise);
        const scalar_t denom = static_cast<scalar_t>(1) + balance * abs_raw;
        const scalar_t denom_sq = denom * denom;
        const scalar_t g_raw = gs / denom_sq;
        local_grad_balance -= gs * raw_surprise * abs_raw / denom_sq;
        local_grad_momentum += g_raw * surprise_t[idx];
        grad_surprise_prev[idx] += g_raw * momentum;
        grad_w += g_raw * delta;
        const scalar_t gdelta = g_raw * (*w_t);
        gk_t[i] += gdelta * err[j] * scale;
        grad_err[j] += gdelta * k_t[i] * scale;
      }
    }
    grad_write[t] += grad_w;

    for (int64_t j = 0; j < M; ++j) {
      gv_t[j] += grad_err[j];
      grad_pred[j] -= grad_err[j];
    }

    semiring_read_backward(
        mem_t,
        k_t,
        beta,
        grad_pred.data(),
        grad_memory_prev.data(),
        gk_t,
        &local_grad_beta,
        M);

    grad_memory.swap(grad_memory_prev);
    grad_surprise.swap(grad_surprise_prev);
  }

  *grad_momentum += local_grad_momentum;
  *grad_beta += local_grad_beta;
  *grad_balance += local_grad_balance;
}

template <typename scalar_t>
int64_t adaptive_step_count(
    const scalar_t surprise_level,
    const scalar_t low_threshold,
    const scalar_t high_threshold,
    const int64_t max_steps) {
  if (max_steps <= 0 || surprise_level < low_threshold) {
    return 0;
  }
  if (max_steps == 1 || high_threshold <= low_threshold || surprise_level >= high_threshold) {
    return std::max<int64_t>(1, max_steps);
  }
  const scalar_t ratio = (surprise_level - low_threshold) / (high_threshold - low_threshold);
  const int64_t steps = 1 + static_cast<int64_t>(
      std::floor(ratio * static_cast<scalar_t>(max_steps - 1)));
  return std::min<int64_t>(std::max<int64_t>(1, steps), max_steps);
}

template <typename scalar_t>
scalar_t balanced_surprise(const scalar_t raw, const scalar_t balance) {
  return raw / (static_cast<scalar_t>(1) + balance * std::abs(raw));
}

template <typename scalar_t>
void adaptive_semiring_forward_one(
    const scalar_t* q,
    const scalar_t* k,
    const scalar_t* v,
    const scalar_t* write,
    const scalar_t* forget,
    const scalar_t momentum,
    const scalar_t beta,
    const scalar_t balance,
    const scalar_t low_threshold,
    const scalar_t high_threshold,
    const int64_t max_steps,
    scalar_t* y,
    scalar_t* mem_prev,
    scalar_t* surprise_prev,
    int64_t* depth_counts,
    int64_t L,
    int64_t M) {
  const scalar_t scale = static_cast<scalar_t>(1.0 / std::sqrt(static_cast<double>(M)));
  std::vector<scalar_t> memory(M * M, static_cast<scalar_t>(0));
  std::vector<scalar_t> surprise(M * M, static_cast<scalar_t>(0));
  std::vector<scalar_t> pred(M);
  std::vector<scalar_t> err(M);
  std::vector<scalar_t> raw(M * M);
  std::vector<scalar_t> new_surprise(M * M);

  for (int64_t t = 0; t < L; ++t) {
    const scalar_t* q_t = q + t * M;
    const scalar_t* k_t = k + t * M;
    const scalar_t* v_t = v + t * M;
    const scalar_t* f_t = forget + t * M;
    scalar_t* y_t = y + t * M;
    scalar_t* mem_save = mem_prev + t * M * M;
    scalar_t* surprise_save = surprise_prev + t * M * M;

    std::copy(memory.begin(), memory.end(), mem_save);
    std::copy(surprise.begin(), surprise.end(), surprise_save);

    semiring_read(memory.data(), q_t, beta, y_t, M);
    semiring_read(memory.data(), k_t, beta, pred.data(), M);
    for (int64_t j = 0; j < M; ++j) {
      err[j] = v_t[j] - pred[j];
    }

    const scalar_t w = write[t];
    scalar_t mean_abs = static_cast<scalar_t>(0);
    for (int64_t i = 0; i < M; ++i) {
      for (int64_t j = 0; j < M; ++j) {
        const int64_t idx = i * M + j;
        const scalar_t delta = k_t[i] * err[j] * scale;
        const scalar_t raw0 = momentum * surprise[idx] + w * delta;
        raw[idx] = raw0;
        mean_abs += std::abs(raw0);
      }
    }
    mean_abs /= static_cast<scalar_t>(M * M);
    const int64_t steps = adaptive_step_count(mean_abs, low_threshold, high_threshold, max_steps);
    depth_counts[t] = steps;
    const int64_t applied_steps = std::max<int64_t>(1, steps);

    for (int64_t i = 0; i < M; ++i) {
      const scalar_t decay = static_cast<scalar_t>(1) - f_t[i];
      for (int64_t j = 0; j < M; ++j) {
        const int64_t idx = i * M + j;
        const scalar_t delta = k_t[i] * err[j] * scale;
        scalar_t s = balanced_surprise(raw[idx], balance);
        for (int64_t r = 1; r < applied_steps; ++r) {
          s = balanced_surprise(momentum * s + w * delta, balance);
        }
        new_surprise[idx] = s;
        memory[idx] = decay * memory[idx] + (steps > 0 ? s : static_cast<scalar_t>(0));
      }
    }
    surprise.swap(new_surprise);
  }
}

template <typename scalar_t>
void adaptive_semiring_backward_one(
    const scalar_t* q,
    const scalar_t* k,
    const scalar_t* v,
    const scalar_t* write,
    const scalar_t* forget,
    const scalar_t momentum,
    const scalar_t beta,
    const scalar_t balance,
    const scalar_t low_threshold,
    const scalar_t high_threshold,
    const int64_t max_steps,
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
    scalar_t* grad_balance,
    int64_t L,
    int64_t M) {
  const scalar_t scale = static_cast<scalar_t>(1.0 / std::sqrt(static_cast<double>(M)));
  std::vector<scalar_t> grad_memory(M * M, static_cast<scalar_t>(0));
  std::vector<scalar_t> grad_surprise(M * M, static_cast<scalar_t>(0));
  std::vector<scalar_t> grad_memory_prev(M * M);
  std::vector<scalar_t> grad_surprise_prev(M * M);
  std::vector<scalar_t> pred(M);
  std::vector<scalar_t> err(M);
  std::vector<scalar_t> grad_pred(M);
  std::vector<scalar_t> grad_err(M);
  std::vector<scalar_t> raw_hist(std::max<int64_t>(1, max_steps));
  std::vector<scalar_t> s_hist(std::max<int64_t>(1, max_steps));

  scalar_t local_grad_momentum = static_cast<scalar_t>(0);
  scalar_t local_grad_beta = static_cast<scalar_t>(0);
  scalar_t local_grad_balance = static_cast<scalar_t>(0);

  for (int64_t t = L - 1; t >= 0; --t) {
    std::fill(grad_memory_prev.begin(), grad_memory_prev.end(), static_cast<scalar_t>(0));
    std::fill(grad_surprise_prev.begin(), grad_surprise_prev.end(), static_cast<scalar_t>(0));
    std::fill(grad_pred.begin(), grad_pred.end(), static_cast<scalar_t>(0));
    std::fill(grad_err.begin(), grad_err.end(), static_cast<scalar_t>(0));

    const scalar_t* q_t = q + t * M;
    const scalar_t* k_t = k + t * M;
    const scalar_t* v_t = v + t * M;
    const scalar_t* w_t = write + t;
    const scalar_t* f_t = forget + t * M;
    const scalar_t* gy_t = grad_y + t * M;
    const scalar_t* mem_t = mem_prev + t * M * M;
    const scalar_t* surprise_t = surprise_prev + t * M * M;
    scalar_t* gq_t = grad_q + t * M;
    scalar_t* gk_t = grad_k + t * M;
    scalar_t* gv_t = grad_v + t * M;
    scalar_t* gf_t = grad_forget + t * M;

    semiring_read(mem_t, k_t, beta, pred.data(), M);
    for (int64_t j = 0; j < M; ++j) {
      err[j] = v_t[j] - pred[j];
    }

    scalar_t mean_abs = static_cast<scalar_t>(0);
    for (int64_t i = 0; i < M; ++i) {
      for (int64_t j = 0; j < M; ++j) {
        const int64_t idx = i * M + j;
        const scalar_t delta = k_t[i] * err[j] * scale;
        mean_abs += std::abs(momentum * surprise_t[idx] + (*w_t) * delta);
      }
    }
    mean_abs /= static_cast<scalar_t>(M * M);
    const int64_t steps = adaptive_step_count(mean_abs, low_threshold, high_threshold, max_steps);
    const int64_t applied_steps = std::max<int64_t>(1, steps);

    semiring_read_backward(
        mem_t, q_t, beta, gy_t, grad_memory_prev.data(), gq_t, &local_grad_beta, M);

    for (int64_t i = 0; i < M; ++i) {
      const scalar_t decay = static_cast<scalar_t>(1) - f_t[i];
      scalar_t grad_decay = static_cast<scalar_t>(0);
      for (int64_t j = 0; j < M; ++j) {
        const int64_t idx = i * M + j;
        const scalar_t gm = grad_memory[idx];
        grad_decay += gm * mem_t[idx];
        grad_memory_prev[idx] += gm * decay;
        if (steps > 0) {
          grad_surprise[idx] += gm;
        }
      }
      gf_t[i] -= grad_decay;
    }

    scalar_t grad_w = static_cast<scalar_t>(0);
    for (int64_t i = 0; i < M; ++i) {
      for (int64_t j = 0; j < M; ++j) {
        const int64_t idx = i * M + j;
        const scalar_t delta = k_t[i] * err[j] * scale;
        raw_hist[0] = momentum * surprise_t[idx] + (*w_t) * delta;
        s_hist[0] = balanced_surprise(raw_hist[0], balance);
        for (int64_t r = 1; r < applied_steps; ++r) {
          raw_hist[r] = momentum * s_hist[r - 1] + (*w_t) * delta;
          s_hist[r] = balanced_surprise(raw_hist[r], balance);
        }

        scalar_t gs = grad_surprise[idx];
        scalar_t grad_delta = static_cast<scalar_t>(0);
        for (int64_t r = applied_steps - 1; r >= 0; --r) {
          const scalar_t raw = raw_hist[r];
          const scalar_t abs_raw = std::abs(raw);
          const scalar_t denom = static_cast<scalar_t>(1) + balance * abs_raw;
          const scalar_t denom_sq = denom * denom;
          const scalar_t g_raw = gs / denom_sq;
          local_grad_balance -= gs * raw * abs_raw / denom_sq;
          grad_w += g_raw * delta;
          grad_delta += g_raw * (*w_t);
          if (r == 0) {
            local_grad_momentum += g_raw * surprise_t[idx];
            grad_surprise_prev[idx] += g_raw * momentum;
          } else {
            local_grad_momentum += g_raw * s_hist[r - 1];
            gs = g_raw * momentum;
          }
        }
        gk_t[i] += grad_delta * err[j] * scale;
        grad_err[j] += grad_delta * k_t[i] * scale;
      }
    }
    grad_write[t] += grad_w;

    for (int64_t j = 0; j < M; ++j) {
      gv_t[j] += grad_err[j];
      grad_pred[j] -= grad_err[j];
    }

    semiring_read_backward(
        mem_t,
        k_t,
        beta,
        grad_pred.data(),
        grad_memory_prev.data(),
        gk_t,
        &local_grad_beta,
        M);

    grad_memory.swap(grad_memory_prev);
    grad_surprise.swap(grad_surprise_prev);
  }

  *grad_momentum += local_grad_momentum;
  *grad_beta += local_grad_beta;
  *grad_balance += local_grad_balance;
}

}  // namespace

std::vector<torch::Tensor> surprise_scan_forward(
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

  AT_DISPATCH_FLOATING_TYPES(q_c.scalar_type(), "surprise_scan_forward", [&] {
    const scalar_t mom = momentum.item<scalar_t>();
    at::parallel_for(0, B, 1, [&](int64_t begin, int64_t end) {
      for (int64_t b = begin; b < end; ++b) {
        forward_one<scalar_t>(
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

std::vector<torch::Tensor> surprise_scan_backward(
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

  AT_DISPATCH_FLOATING_TYPES(q_c.scalar_type(), "surprise_scan_backward", [&] {
    const scalar_t mom = momentum.item<scalar_t>();
    at::parallel_for(0, B, 1, [&](int64_t begin, int64_t end) {
      for (int64_t b = begin; b < end; ++b) {
        backward_one<scalar_t>(
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

std::vector<torch::Tensor> semiring_surprise_scan_forward(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor write,
    torch::Tensor forget,
    torch::Tensor momentum,
    torch::Tensor beta,
    torch::Tensor balance) {
  check_inputs(q, k, v, write, forget, momentum);
  TORCH_CHECK(beta.device().is_cpu(), "beta must be CPU");
  TORCH_CHECK(beta.numel() == 1, "beta must be scalar");
  TORCH_CHECK(balance.device().is_cpu(), "balance must be CPU");
  TORCH_CHECK(balance.numel() == 1, "balance must be scalar");
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

  AT_DISPATCH_FLOATING_TYPES(q_c.scalar_type(), "semiring_surprise_scan_forward", [&] {
    const scalar_t mom = momentum.item<scalar_t>();
    const scalar_t bta = beta.item<scalar_t>();
    const scalar_t bal = balance.item<scalar_t>();
    at::parallel_for(0, B, 1, [&](int64_t begin, int64_t end) {
      for (int64_t b = begin; b < end; ++b) {
        semiring_forward_one<scalar_t>(
            q_c.data_ptr<scalar_t>() + b * L * M,
            k_c.data_ptr<scalar_t>() + b * L * M,
            v_c.data_ptr<scalar_t>() + b * L * M,
            write_c.data_ptr<scalar_t>() + b * L,
            forget_c.data_ptr<scalar_t>() + b * L * M,
            mom,
            bta,
            bal,
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

std::vector<torch::Tensor> semiring_surprise_scan_backward(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor write,
    torch::Tensor forget,
    torch::Tensor momentum,
    torch::Tensor beta,
    torch::Tensor balance,
    torch::Tensor grad_y,
    torch::Tensor mem_prev,
    torch::Tensor surprise_prev) {
  check_inputs(q, k, v, write, forget, momentum);
  TORCH_CHECK(beta.device().is_cpu(), "beta must be CPU");
  TORCH_CHECK(beta.numel() == 1, "beta must be scalar");
  TORCH_CHECK(balance.device().is_cpu(), "balance must be CPU");
  TORCH_CHECK(balance.numel() == 1, "balance must be scalar");
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
  auto grad_balance_per_b = torch::zeros({B}, q_c.options());

  AT_DISPATCH_FLOATING_TYPES(q_c.scalar_type(), "semiring_surprise_scan_backward", [&] {
    const scalar_t mom = momentum.item<scalar_t>();
    const scalar_t bta = beta.item<scalar_t>();
    const scalar_t bal = balance.item<scalar_t>();
    at::parallel_for(0, B, 1, [&](int64_t begin, int64_t end) {
      for (int64_t b = begin; b < end; ++b) {
        semiring_backward_one<scalar_t>(
            q_c.data_ptr<scalar_t>() + b * L * M,
            k_c.data_ptr<scalar_t>() + b * L * M,
            v_c.data_ptr<scalar_t>() + b * L * M,
            write_c.data_ptr<scalar_t>() + b * L,
            forget_c.data_ptr<scalar_t>() + b * L * M,
            mom,
            bta,
            bal,
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
            grad_balance_per_b.data_ptr<scalar_t>() + b,
            L,
            M);
      }
    });
  });

  auto grad_momentum = grad_momentum_per_b.sum().reshape_as(momentum);
  auto grad_beta = grad_beta_per_b.sum().reshape_as(beta);
  auto grad_balance = grad_balance_per_b.sum().reshape_as(balance);
  return {
      grad_q,
      grad_k,
      grad_v,
      grad_write,
      grad_forget,
      grad_momentum,
      grad_beta,
      grad_balance};
}

std::vector<torch::Tensor> adaptive_semiring_surprise_scan_forward(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor write,
    torch::Tensor forget,
    torch::Tensor momentum,
    torch::Tensor beta,
    torch::Tensor balance,
    torch::Tensor low_threshold,
    torch::Tensor high_threshold,
    int64_t max_steps) {
  check_inputs(q, k, v, write, forget, momentum);
  TORCH_CHECK(beta.device().is_cpu() && beta.numel() == 1, "beta must be scalar CPU");
  TORCH_CHECK(balance.device().is_cpu() && balance.numel() == 1, "balance must be scalar CPU");
  TORCH_CHECK(low_threshold.device().is_cpu() && low_threshold.numel() == 1, "low_threshold must be scalar CPU");
  TORCH_CHECK(high_threshold.device().is_cpu() && high_threshold.numel() == 1, "high_threshold must be scalar CPU");
  TORCH_CHECK(max_steps >= 0 && max_steps <= 8, "max_steps must be between 0 and 8");
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
  auto depth_counts = torch::empty({B, L}, torch::dtype(torch::kInt64));

  AT_DISPATCH_FLOATING_TYPES(q_c.scalar_type(), "adaptive_semiring_surprise_scan_forward", [&] {
    const scalar_t mom = momentum.item<scalar_t>();
    const scalar_t bta = beta.item<scalar_t>();
    const scalar_t bal = balance.item<scalar_t>();
    const scalar_t low = low_threshold.item<scalar_t>();
    const scalar_t high = high_threshold.item<scalar_t>();
    at::parallel_for(0, B, 1, [&](int64_t begin, int64_t end) {
      for (int64_t b = begin; b < end; ++b) {
        adaptive_semiring_forward_one<scalar_t>(
            q_c.data_ptr<scalar_t>() + b * L * M,
            k_c.data_ptr<scalar_t>() + b * L * M,
            v_c.data_ptr<scalar_t>() + b * L * M,
            write_c.data_ptr<scalar_t>() + b * L,
            forget_c.data_ptr<scalar_t>() + b * L * M,
            mom,
            bta,
            bal,
            low,
            high,
            max_steps,
            y.data_ptr<scalar_t>() + b * L * M,
            mem_prev.data_ptr<scalar_t>() + b * L * M * M,
            surprise_prev.data_ptr<scalar_t>() + b * L * M * M,
            depth_counts.data_ptr<int64_t>() + b * L,
            L,
            M);
      }
    });
  });

  return {y, mem_prev, surprise_prev, depth_counts};
}

std::vector<torch::Tensor> adaptive_semiring_surprise_scan_backward(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor write,
    torch::Tensor forget,
    torch::Tensor momentum,
    torch::Tensor beta,
    torch::Tensor balance,
    torch::Tensor low_threshold,
    torch::Tensor high_threshold,
    int64_t max_steps,
    torch::Tensor grad_y,
    torch::Tensor mem_prev,
    torch::Tensor surprise_prev) {
  check_inputs(q, k, v, write, forget, momentum);
  TORCH_CHECK(beta.device().is_cpu() && beta.numel() == 1, "beta must be scalar CPU");
  TORCH_CHECK(balance.device().is_cpu() && balance.numel() == 1, "balance must be scalar CPU");
  TORCH_CHECK(low_threshold.device().is_cpu() && low_threshold.numel() == 1, "low_threshold must be scalar CPU");
  TORCH_CHECK(high_threshold.device().is_cpu() && high_threshold.numel() == 1, "high_threshold must be scalar CPU");
  TORCH_CHECK(max_steps >= 0 && max_steps <= 8, "max_steps must be between 0 and 8");
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
  auto grad_balance_per_b = torch::zeros({B}, q_c.options());

  AT_DISPATCH_FLOATING_TYPES(q_c.scalar_type(), "adaptive_semiring_surprise_scan_backward", [&] {
    const scalar_t mom = momentum.item<scalar_t>();
    const scalar_t bta = beta.item<scalar_t>();
    const scalar_t bal = balance.item<scalar_t>();
    const scalar_t low = low_threshold.item<scalar_t>();
    const scalar_t high = high_threshold.item<scalar_t>();
    at::parallel_for(0, B, 1, [&](int64_t begin, int64_t end) {
      for (int64_t b = begin; b < end; ++b) {
        adaptive_semiring_backward_one<scalar_t>(
            q_c.data_ptr<scalar_t>() + b * L * M,
            k_c.data_ptr<scalar_t>() + b * L * M,
            v_c.data_ptr<scalar_t>() + b * L * M,
            write_c.data_ptr<scalar_t>() + b * L,
            forget_c.data_ptr<scalar_t>() + b * L * M,
            mom,
            bta,
            bal,
            low,
            high,
            max_steps,
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
            grad_balance_per_b.data_ptr<scalar_t>() + b,
            L,
            M);
      }
    });
  });

  auto grad_momentum = grad_momentum_per_b.sum().reshape_as(momentum);
  auto grad_beta = grad_beta_per_b.sum().reshape_as(beta);
  auto grad_balance = grad_balance_per_b.sum().reshape_as(balance);
  return {
      grad_q,
      grad_k,
      grad_v,
      grad_write,
      grad_forget,
      grad_momentum,
      grad_beta,
      grad_balance,
      torch::zeros_like(low_threshold),
      torch::zeros_like(high_threshold)};
}

std::vector<torch::Tensor> two_lane_blend_forward(
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor logit) {
  TORCH_CHECK(a.device().is_cpu() && b.device().is_cpu() && logit.device().is_cpu(), "two-lane blend is CPU-only");
  TORCH_CHECK(a.sizes() == b.sizes(), "a and b must have the same shape");
  TORCH_CHECK(a.dim() == 3, "a and b must be [B, L, D]");
  TORCH_CHECK(logit.dim() == 3 && logit.size(0) == a.size(0) && logit.size(1) == a.size(1) && logit.size(2) == 1,
              "logit must be [B, L, 1]");
  auto a_c = a.contiguous();
  auto b_c = b.contiguous();
  auto logit_c = logit.contiguous();
  auto y = torch::empty_like(a_c);
  auto gate = torch::empty_like(logit_c);
  const auto B = a_c.size(0);
  const auto L = a_c.size(1);
  const auto D = a_c.size(2);
  const auto N = B * L;

  AT_DISPATCH_FLOATING_TYPES(a_c.scalar_type(), "two_lane_blend_forward", [&] {
    at::parallel_for(0, N, 1, [&](int64_t begin, int64_t end) {
      for (int64_t n = begin; n < end; ++n) {
        const scalar_t g = static_cast<scalar_t>(1) /
            (static_cast<scalar_t>(1) + std::exp(-logit_c.data_ptr<scalar_t>()[n]));
        gate.data_ptr<scalar_t>()[n] = g;
        const int64_t off = n * D;
        for (int64_t d = 0; d < D; ++d) {
          y.data_ptr<scalar_t>()[off + d] =
              g * a_c.data_ptr<scalar_t>()[off + d] +
              (static_cast<scalar_t>(1) - g) * b_c.data_ptr<scalar_t>()[off + d];
        }
      }
    });
  });
  return {y, gate};
}

std::vector<torch::Tensor> two_lane_blend_backward(
    torch::Tensor grad_y,
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor gate) {
  TORCH_CHECK(grad_y.sizes() == a.sizes() && a.sizes() == b.sizes(), "grad_y/a/b shape mismatch");
  TORCH_CHECK(gate.dim() == 3 && gate.size(0) == a.size(0) && gate.size(1) == a.size(1) && gate.size(2) == 1,
              "gate must be [B, L, 1]");
  auto gy_c = grad_y.contiguous();
  auto a_c = a.contiguous();
  auto b_c = b.contiguous();
  auto gate_c = gate.contiguous();
  auto grad_a = torch::empty_like(a_c);
  auto grad_b = torch::empty_like(b_c);
  auto grad_logit = torch::empty_like(gate_c);
  const auto B = a_c.size(0);
  const auto L = a_c.size(1);
  const auto D = a_c.size(2);
  const auto N = B * L;

  AT_DISPATCH_FLOATING_TYPES(a_c.scalar_type(), "two_lane_blend_backward", [&] {
    at::parallel_for(0, N, 1, [&](int64_t begin, int64_t end) {
      for (int64_t n = begin; n < end; ++n) {
        const scalar_t g = gate_c.data_ptr<scalar_t>()[n];
        scalar_t grad_gate = static_cast<scalar_t>(0);
        const int64_t off = n * D;
        for (int64_t d = 0; d < D; ++d) {
          const scalar_t gy = gy_c.data_ptr<scalar_t>()[off + d];
          grad_a.data_ptr<scalar_t>()[off + d] = gy * g;
          grad_b.data_ptr<scalar_t>()[off + d] = gy * (static_cast<scalar_t>(1) - g);
          grad_gate += gy * (a_c.data_ptr<scalar_t>()[off + d] - b_c.data_ptr<scalar_t>()[off + d]);
        }
        grad_logit.data_ptr<scalar_t>()[n] = grad_gate * g * (static_cast<scalar_t>(1) - g);
      }
    });
  });
  return {grad_a, grad_b, grad_logit};
}

std::vector<torch::Tensor> three_lane_blend_forward(
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor c,
    torch::Tensor logits) {
  TORCH_CHECK(a.device().is_cpu() && b.device().is_cpu() && c.device().is_cpu() && logits.device().is_cpu(), "three-lane blend is CPU-only");
  TORCH_CHECK(a.sizes() == b.sizes() && a.sizes() == c.sizes(), "lane shapes must match");
  TORCH_CHECK(a.dim() == 3, "lanes must be [B, L, D]");
  TORCH_CHECK(logits.dim() == 3 && logits.size(0) == a.size(0) && logits.size(1) == a.size(1) && logits.size(2) == 3,
              "logits must be [B, L, 3]");
  auto a_c = a.contiguous();
  auto b_c = b.contiguous();
  auto c_c = c.contiguous();
  auto logits_c = logits.contiguous();
  auto y = torch::empty_like(a_c);
  auto weights = torch::empty_like(logits_c);
  const auto B = a_c.size(0);
  const auto L = a_c.size(1);
  const auto D = a_c.size(2);
  const auto N = B * L;

  AT_DISPATCH_FLOATING_TYPES(a_c.scalar_type(), "three_lane_blend_forward", [&] {
    at::parallel_for(0, N, 1, [&](int64_t begin, int64_t end) {
      for (int64_t n = begin; n < end; ++n) {
        const scalar_t* z = logits_c.data_ptr<scalar_t>() + n * 3;
        const scalar_t max_z = std::max(z[0], std::max(z[1], z[2]));
        const scalar_t e0 = std::exp(z[0] - max_z);
        const scalar_t e1 = std::exp(z[1] - max_z);
        const scalar_t e2 = std::exp(z[2] - max_z);
        const scalar_t inv = static_cast<scalar_t>(1) / (e0 + e1 + e2);
        const scalar_t w0 = e0 * inv;
        const scalar_t w1 = e1 * inv;
        const scalar_t w2 = e2 * inv;
        weights.data_ptr<scalar_t>()[n * 3] = w0;
        weights.data_ptr<scalar_t>()[n * 3 + 1] = w1;
        weights.data_ptr<scalar_t>()[n * 3 + 2] = w2;
        const int64_t off = n * D;
        for (int64_t d = 0; d < D; ++d) {
          y.data_ptr<scalar_t>()[off + d] =
              w0 * a_c.data_ptr<scalar_t>()[off + d] +
              w1 * b_c.data_ptr<scalar_t>()[off + d] +
              w2 * c_c.data_ptr<scalar_t>()[off + d];
        }
      }
    });
  });
  return {y, weights};
}

std::vector<torch::Tensor> three_lane_blend_backward(
    torch::Tensor grad_y,
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor c,
    torch::Tensor y,
    torch::Tensor weights) {
  TORCH_CHECK(grad_y.sizes() == a.sizes() && a.sizes() == b.sizes() && a.sizes() == c.sizes(), "lane shape mismatch");
  auto gy_c = grad_y.contiguous();
  auto a_c = a.contiguous();
  auto b_c = b.contiguous();
  auto c_c = c.contiguous();
  auto y_c = y.contiguous();
  auto weights_c = weights.contiguous();
  auto grad_a = torch::empty_like(a_c);
  auto grad_b = torch::empty_like(b_c);
  auto grad_c = torch::empty_like(c_c);
  auto grad_logits = torch::empty_like(weights_c);
  const auto B = a_c.size(0);
  const auto L = a_c.size(1);
  const auto D = a_c.size(2);
  const auto N = B * L;

  AT_DISPATCH_FLOATING_TYPES(a_c.scalar_type(), "three_lane_blend_backward", [&] {
    at::parallel_for(0, N, 1, [&](int64_t begin, int64_t end) {
      for (int64_t n = begin; n < end; ++n) {
        const scalar_t w0 = weights_c.data_ptr<scalar_t>()[n * 3];
        const scalar_t w1 = weights_c.data_ptr<scalar_t>()[n * 3 + 1];
        const scalar_t w2 = weights_c.data_ptr<scalar_t>()[n * 3 + 2];
        scalar_t gl0 = static_cast<scalar_t>(0);
        scalar_t gl1 = static_cast<scalar_t>(0);
        scalar_t gl2 = static_cast<scalar_t>(0);
        const int64_t off = n * D;
        for (int64_t d = 0; d < D; ++d) {
          const scalar_t gy = gy_c.data_ptr<scalar_t>()[off + d];
          grad_a.data_ptr<scalar_t>()[off + d] = gy * w0;
          grad_b.data_ptr<scalar_t>()[off + d] = gy * w1;
          grad_c.data_ptr<scalar_t>()[off + d] = gy * w2;
          const scalar_t yd = y_c.data_ptr<scalar_t>()[off + d];
          gl0 += gy * (a_c.data_ptr<scalar_t>()[off + d] - yd);
          gl1 += gy * (b_c.data_ptr<scalar_t>()[off + d] - yd);
          gl2 += gy * (c_c.data_ptr<scalar_t>()[off + d] - yd);
        }
        grad_logits.data_ptr<scalar_t>()[n * 3] = w0 * gl0;
        grad_logits.data_ptr<scalar_t>()[n * 3 + 1] = w1 * gl1;
        grad_logits.data_ptr<scalar_t>()[n * 3 + 2] = w2 * gl2;
      }
    });
  });
  return {grad_a, grad_b, grad_c, grad_logits};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &surprise_scan_forward, "Read-before-write surprise scan forward");
  m.def("backward", &surprise_scan_backward, "Read-before-write surprise scan backward");
  m.def(
      "semiring_forward",
      &semiring_surprise_scan_forward,
      "Read-before-write semiring surprise scan forward");
  m.def(
      "semiring_backward",
      &semiring_surprise_scan_backward,
      "Read-before-write semiring surprise scan backward");
  m.def(
      "adaptive_semiring_forward",
      &adaptive_semiring_surprise_scan_forward,
      "Adaptive-depth read-before-write semiring surprise scan forward");
  m.def(
      "adaptive_semiring_backward",
      &adaptive_semiring_surprise_scan_backward,
      "Adaptive-depth read-before-write semiring surprise scan backward");
  m.def("two_lane_blend_forward", &two_lane_blend_forward, "Native two-lane gate forward");
  m.def("two_lane_blend_backward", &two_lane_blend_backward, "Native two-lane gate backward");
  m.def("three_lane_blend_forward", &three_lane_blend_forward, "Native three-lane gate forward");
  m.def("three_lane_blend_backward", &three_lane_blend_backward, "Native three-lane gate backward");
}
