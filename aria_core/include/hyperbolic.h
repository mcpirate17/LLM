#ifndef ARIA_HYPERBOLIC_H
#define ARIA_HYPERBOLIC_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Möbius addition in the Poincaré ball model.
 * y = ((1 + 2c<x,v> + c|v|^2)x + (1 - c|x|^2)v) / (1 + 2c<x,v> + c^2|x|^2|v|^2)
 */
void aria_hyperbolic_mobius_add_f32(const float *x, const float *v, float *y,
                                    int64_t batch, int64_t dim, float c);

/**
 * Hyperbolic distance in the Poincaré ball model.
 * d(x, y) = (2/sqrt(c)) * atanh(sqrt(c) * |mobius_add(-x, y)|)
 */
void aria_hyperbolic_distance_f32(const float *x, const float *y, float *out,
                                  int64_t batch, int64_t dim, float c);

/** Exponential map: tangent space (Euclidean) → Poincaré ball. */
void aria_exp_map_f32(const float *x, float *y, int64_t batch, int64_t dim, float c);

/** Logarithmic map: Poincaré ball → tangent space (Euclidean). */
void aria_log_map_f32(const float *x, float *y, int64_t batch, int64_t dim, float c);

/** Möbius addition (alias for poincare_add). */
void aria_poincare_add_f32(const float *x, const float *v, float *y,
                            int64_t batch, int64_t dim, float c);

/** Hyperbolic linear: log_map → matmul → exp_map fused. */
void aria_hyp_linear_f32(const float *x, const float *W, float *y,
                          int64_t batch, int64_t dim_in, int64_t dim_out, float c);

/** Hyperbolic layer norm: log_map → layer_norm → exp_map. */
void aria_hyperbolic_norm_f32(const float *x, const float *gamma, const float *beta,
                               float *y, int64_t batch, int64_t dim, float c, float eps);

/** Hyperbolic tangent nonlinearity. */
void aria_hyp_tangent_nonlinear_f32(const float *x, float *y, int64_t n, float c);

/** Backward pass for exp_map: VJP = scale * grad + coeff * dot(grad, v) * v. */
void aria_exp_map_backward_f32(const float *v, const float *grad_out, float *grad_in,
                                 int64_t batch, int64_t dim, float c);

/** Backward pass for log_map: VJP = scale * grad + coeff * dot(grad, x) * x. */
void aria_log_map_backward_f32(const float *x, const float *grad_out, float *grad_in,
                                 int64_t batch, int64_t dim, float c);

#ifdef __cplusplus
}
#endif

#endif /* ARIA_HYPERBOLIC_H */
