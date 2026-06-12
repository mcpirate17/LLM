#include "../_json_config.h"
int component_validate(const ComponentConfig* config, char* error_buf, int buf_size) {
    char path[1024] = {0};
    if (config && config->json_config) {
        json_get_string(config->json_config, "output_path", path, sizeof(path));
    }
    if (path[0] == '\0') {
        if (error_buf && buf_size > 0) {
            strncpy(error_buf, "output_path is required", (size_t)buf_size - 1);
            error_buf[buf_size - 1] = '\0';
        }
        return -1;
    }
    return 0;
}

int component_forward(const TensorView* inputs, int n_inputs,
                      TensorView* outputs, int n_outputs,
                      const ComponentConfig* config) {
    if (!inputs || n_inputs < 1) return -1;

    char output_path[1024] = "output.txt";
    char format[64] = "auto";
    int overwrite = 0;
    int include_shape = 1;

    if (config && config->json_config) {
        json_get_string(config->json_config, "output_path", output_path, sizeof(output_path));
        json_get_string(config->json_config, "file_format", format, sizeof(format));
        overwrite = json_get_bool(config->json_config, "overwrite", 0);
        include_shape = json_get_bool(config->json_config, "include_shape", 1);
    }

    if (!overwrite) {
        FILE* chk = fopen(output_path, "r");
        if (chk) {
            fclose(chk);
            return -2;
        }
    }

    const TensorView* in = &inputs[0];
    if (!in->data) return -3;

    int64_t total = 1;
    for (int i = 0; i < in->ndim; i++) {
        total *= in->shape ? in->shape[i] : 1;
    }
    if (total <= 0) total = 0;

    const char* ext = strrchr(output_path, '.');
    if (strcmp(format, "auto") == 0 && ext) {
        if (strcmp(ext, ".csv") == 0) strncpy(format, "csv", sizeof(format) - 1);
        else strncpy(format, "txt", sizeof(format) - 1);
    }

    FILE* fp = fopen(output_path, "w");
    if (!fp) return -4;

    if (include_shape && in->shape) {
        fprintf(fp, "#shape");
        for (int i = 0; i < in->ndim; i++) fprintf(fp, "%s%lld", (i == 0 ? "," : ","), (long long)in->shape[i]);
        fprintf(fp, "\n");
    }

    if (strcmp(format, "csv") == 0) {
        int64_t cols = (in->ndim > 0 && in->shape) ? in->shape[in->ndim - 1] : 1;
        if (cols <= 0) cols = 1;
        for (int64_t i = 0; i < total; i++) {
            fprintf(fp, "%g", (double)in->data[i]);
            if ((i + 1) % cols == 0) fprintf(fp, "\n");
            else fprintf(fp, ",");
        }
    } else {
        for (int64_t i = 0; i < total; i++) {
            fprintf(fp, "%g\n", (double)in->data[i]);
        }
    }

    fclose(fp);

    if (outputs && n_outputs > 0 && outputs[0].data) {
        outputs[0].data[0] = 1.0f;
        if (outputs[0].shape) outputs[0].shape[0] = 1;
        outputs[0].ndim = 1;
        outputs[0].dtype = 0;
    }
    return 0;
}

void component_cleanup(void) {
}
