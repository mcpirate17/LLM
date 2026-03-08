/**
 * kernels.cpp — Master CPU kernel inclusion file.
 *
 * This file acts as the primary compilation unit for CPU kernels,
 * including optimized modular implementations while maintaining
 * a clean build structure.
 */

#include "kernels_common.h"

// ── Master Inclusion List (DRY Canonical Sources) ──────────────────

#include "unary.cpp"
#include "binary.cpp"
#include "linalg.cpp"
#include "norm.cpp"
#include "mixing.cpp"
#include "math_space.cpp"
#include "backward.cpp"
#include "io.cpp"
#include "adaptive_routing.cpp"
#include "routing.c"
#include "dispatch.c"
#include "binding_stubs.cpp"
#include "fp16.cpp"
