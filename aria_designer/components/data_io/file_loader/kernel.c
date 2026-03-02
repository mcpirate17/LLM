#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <ctype.h>

typedef struct {
    float* data;
    int64_t* shape;
    int ndim;
    int dtype;
} TensorView;

typedef struct {
    const char* json_config;
} ComponentConfig;

static int json_get_string(const char* json, const char* key, char* out, size_t out_sz) {
    if (!json || !key || !out || out_sz == 0) return 0;
    char needle[128];
    snprintf(needle, sizeof(needle), "\"%s\"", key);
    const char* p = strstr(json, needle);
    if (!p) return 0;
    p = strchr(p, ':');
    if (!p) return 0;
    p++;
    while (*p && isspace((unsigned char)*p)) p++;
    if (*p != '"') return 0;
    p++;
    size_t i = 0;
    while (*p && *p != '"' && i + 1 < out_sz) {
        out[i++] = *p++;
    }
    out[i] = '\0';
    return i > 0;
}

static int json_get_bool(const char* json, const char* key, int def) {
    if (!json || !key) return def;
    char needle[128];
    snprintf(needle, sizeof(needle), "\"%s\"", key);
    const char* p = strstr(json, needle);
    if (!p) return def;
    p = strchr(p, ':');
    if (!p) return def;
    p++;
    while (*p && isspace((unsigned char)*p)) p++;
    if (strncmp(p, "true", 4) == 0) return 1;
    if (strncmp(p, "false", 5) == 0) return 0;
    return def;
}

int component_validate(const ComponentConfig* config, char* error_buf, int buf_size) {
    char path[1024] = {0};
    if (config && config->json_config) {
        json_get_string(config->json_config, "file_path", path, sizeof(path));
    }
    if (path[0] == '\0') {
        if (error_buf && buf_size > 0) {
            strncpy(error_buf, "file_path is required", (size_t)buf_size - 1);
            error_buf[buf_size - 1] = '\0';
        }
        return -1;
    }
    return 0;
}

int component_forward(const TensorView* inputs, int n_inputs,
                      TensorView* outputs, int n_outputs,
                      const ComponentConfig* config) {
    (void)inputs;
    (void)n_inputs;
    if (!outputs || n_outputs < 1) return -1;

    char path[1024] = "data.csv";
    int has_header = 1;
    if (config && config->json_config) {
        json_get_string(config->json_config, "file_path", path, sizeof(path));
        has_header = json_get_bool(config->json_config, "has_header", 1);
    }

    FILE* fp = fopen(path, "r");
    if (!fp) return -2;

    size_t cap = 1024;
    float* data = (float*)malloc(sizeof(float) * cap);
    if (!data) {
        fclose(fp);
        return -3;
    }

    int64_t rows = 0;
    int64_t cols = 0;
    int64_t n = 0;
    char line[8192];
    int skip_first = has_header;

    while (fgets(line, sizeof(line), fp)) {
        if (skip_first) {
            skip_first = 0;
            continue;
        }
        int64_t this_cols = 0;
        char* p = line;
        while (*p) {
            char* end = p;
            float v = strtof(p, &end);
            if (end == p) {
                while (*p && *p != ',' && *p != '\n' && *p != '\r') p++;
                if (*p == ',') p++;
                continue;
            }
            if (n >= (int64_t)cap) {
                cap *= 2;
                float* grown = (float*)realloc(data, sizeof(float) * cap);
                if (!grown) {
                    free(data);
                    fclose(fp);
                    return -4;
                }
                data = grown;
            }
            data[n++] = v;
            this_cols++;
            p = end;
            while (*p == ' ' || *p == '\t') p++;
            if (*p == ',') p++;
            while (*p == ' ' || *p == '\t') p++;
            if (*p == '\n' || *p == '\r') break;
        }
        if (this_cols > 0) {
            if (cols == 0) cols = this_cols;
            rows++;
        }
    }
    fclose(fp);

    if (rows == 0 || cols == 0) {
        free(data);
        return -5;
    }

    int64_t expected = rows * cols;
    if (n < expected) {
        free(data);
        return -6;
    }

    outputs[0].data = data;
    if (outputs[0].shape) {
        outputs[0].shape[0] = rows;
        outputs[0].shape[1] = cols;
    }
    outputs[0].ndim = 2;
    outputs[0].dtype = 0;
    return 0;
}

void component_cleanup(void) {
}
