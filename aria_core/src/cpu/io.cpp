#ifndef ARIA_KERNELS_COMMON_H
#include "kernels_common.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* ── IO ────────────────────────────────────────────────────────────── */

int aria_read_csv_f32(const char *filename, float *out_data, int64_t max_rows, int64_t max_cols, char delimiter) {
    FILE *fp = fopen(filename, "r");
    if (!fp) return -1;

    char line[4096];
    int64_t row = 0;
    int64_t col = 0;
    
    while (fgets(line, sizeof(line), fp) && row < max_rows) {
        char *ptr = line;
        col = 0;
        while (*ptr && col < max_cols) {
            float val = strtof(ptr, &ptr);
            out_data[row * max_cols + col] = val;
            col++;
            if (*ptr == delimiter) ptr++;
            else break; 
        }
        row++;
    }
    
    fclose(fp);
    return (int)row;
}

int aria_filter_f32(const float *data, float *out_data, int64_t rows, int64_t cols, int64_t col_idx, float val, int op) {
    int64_t out_row = 0;
    for (int64_t i = 0; i < rows; i++) {
        float v = data[i * cols + col_idx];
        int keep = 0;
        switch (op) {
            case 0: keep = v > val; break;  // >
            case 1: keep = v < val; break;  // <
            case 2: keep = v >= val; break; // >=
            case 3: keep = v <= val; break; // <=
            case 4: keep = fabsf(v - val) < 1e-6; break; // ==
            case 5: keep = fabsf(v - val) > 1e-6; break; // !=
        }
        if (keep) {
            memcpy(out_data + out_row * cols, data + i * cols, (size_t)cols * sizeof(float));
            out_row++;
        }
    }
    return (int)out_row;
}

int aria_file_loader_csv_f32(const char *filename, float *out_data,
                             int64_t max_rows, int64_t max_cols,
                             char delimiter, int has_header) {
    FILE *fp = fopen(filename, "r");
    if (!fp) return -1;

    char line[8192];
    int64_t row = 0;
    int skip_first = has_header ? 1 : 0;

    char delim_str[2] = {delimiter, '\0'};
    while (fgets(line, sizeof(line), fp) && row < max_rows) {
        if (skip_first) {
            skip_first = 0;
            continue;
        }

        int64_t col = 0;
        char *tok = strtok(line, delim_str);
        while (tok && col < max_cols) {
            char *end = tok;
            float val = strtof(tok, &end);
            if (end != tok) {
                out_data[row * max_cols + col] = val;
                col++;
            }
            tok = strtok(NULL, delim_str);
        }

        if (col > 0) row++;
    }

    fclose(fp);
    return (int)row;
}

int aria_binary_file_reader_f32(const char *filename, float *out_data,
                                int64_t max_elems, int64_t offset_bytes) {
    FILE *fp = fopen(filename, "rb");
    if (!fp) return -1;

    if (offset_bytes > 0) {
        if (fseek(fp, (long)offset_bytes, SEEK_SET) != 0) {
            fclose(fp);
            return -2;
        }
    }

    size_t n = fread(out_data, sizeof(float), (size_t)max_elems, fp);
    fclose(fp);
    return (int)n;
}

int aria_file_writer_txt_f32(const char *filename, const float *data,
                             int64_t n, int overwrite) {
    if (!overwrite) {
        FILE *chk = fopen(filename, "r");
        if (chk) {
            fclose(chk);
            return -1;
        }
    }

    FILE *fp = fopen(filename, "w");
    if (!fp) return -2;

    for (int64_t i = 0; i < n; i++) {
        fprintf(fp, "%g\n", (double)data[i]);
    }

    fclose(fp);
    return (int)n;
}

#ifdef __cplusplus
}
#endif
