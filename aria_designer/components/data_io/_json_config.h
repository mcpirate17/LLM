#ifndef ARIA_DATA_IO_JSON_CONFIG_H
#define ARIA_DATA_IO_JSON_CONFIG_H

/* Shared component-kernel scaffolding for data_io kernels: the TensorView /
 * ComponentConfig ABI structs and the minimal flat-JSON config readers.
 * file_loader / file_writer / binary_file_reader previously carried
 * identical copies of all of this. */

#include <ctype.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    float* data;
    int64_t* shape;
    int ndim;
    int dtype;
} TensorView;

typedef struct {
    const char* json_config;
} ComponentConfig;

/* Pointer just past `"key":` (whitespace skipped), or NULL when absent. */
static const char* json_find_value(const char* json, const char* key) {
    if (!json || !key) return NULL;
    char needle[128];
    snprintf(needle, sizeof(needle), "\"%s\"", key);
    const char* p = strstr(json, needle);
    if (!p) return NULL;
    p = strchr(p, ':');
    if (!p) return NULL;
    p++;
    while (*p && isspace((unsigned char)*p)) p++;
    return p;
}

static int json_get_string(const char* json, const char* key, char* out, size_t out_sz) {
    if (!out || out_sz == 0) return 0;
    const char* p = json_find_value(json, key);
    if (!p || *p != '"') return 0;
    p++;
    size_t i = 0;
    while (*p && *p != '"' && i + 1 < out_sz) out[i++] = *p++;
    out[i] = '\0';
    return i > 0;
}

static int json_get_bool(const char* json, const char* key, int def) {
    const char* p = json_find_value(json, key);
    if (!p) return def;
    if (strncmp(p, "true", 4) == 0) return 1;
    if (strncmp(p, "false", 5) == 0) return 0;
    return def;
}

static int json_get_int(const char* json, const char* key, int def) {
    const char* p = json_find_value(json, key);
    if (!p) return def;
    return (int)strtol(p, NULL, 10);
}

#endif /* ARIA_DATA_IO_JSON_CONFIG_H */
