#ifndef ARIA_CLIFFORD_H
#define ARIA_CLIFFORD_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Geometric product of two multivectors in Cl(3,0).
 * a, b, y: [n_multivectors, 8]
 * Basis order: {1, e1, e2, e3, e12, e13, e23, e123}
 */
void aria_clifford_geometric_product_cl30_f32(const float *a, const float *b, float *y, int64_t n_multivectors);

/**
 * Rotor transform: y = rotor * x * ~rotor
 * x, rotor, y: [n_multivectors, 8]
 */
void aria_clifford_rotor_transform_cl30_f32(const float *x, const float *rotor, float *y, int64_t n_multivectors);

#ifdef __cplusplus
}
#endif

#endif /* ARIA_CLIFFORD_H */
