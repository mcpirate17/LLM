#ifndef ARIA_REGISTRY_H
#define ARIA_REGISTRY_H

#include "../include/kernel_abi.h"
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define ARIA_MAX_KERNELS 256

/* Initialize the registry with built-in kernels */
void aria_registry_init(void);

/* Register a kernel */
nk_status_t aria_registry_register(const char *op_name,
                                    nk_unary_f32_fn unary_fn,
                                    nk_binary_f32_fn binary_fn);

/* Look up a kernel by name */
int aria_registry_lookup_unary(const char *op_name, nk_unary_f32_fn *out);
int aria_registry_lookup_binary(const char *op_name, nk_binary_f32_fn *out);

/* Query capabilities */
int32_t aria_registry_count(void);
void aria_registry_list(const char **names, int32_t max_count, int32_t *out_count);
int32_t aria_registry_is_native(const char *op_name);

#ifdef __cplusplus
}
#endif

#endif /* ARIA_REGISTRY_H */
