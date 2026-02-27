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

#ifdef __cplusplus
}
#endif

#endif /* ARIA_HYPERBOLIC_H */
