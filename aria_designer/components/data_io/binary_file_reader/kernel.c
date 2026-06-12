#include "../_json_config.h"
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

    char path[1024] = "data.bin";
    int offset = 0;
    if (config && config->json_config) {
        json_get_string(config->json_config, "file_path", path, sizeof(path));
        offset = json_get_int(config->json_config, "offset_bytes", 0);
    }

    FILE* fp = fopen(path, "rb");
    if (!fp) return -2;
    if (offset > 0) fseek(fp, offset, SEEK_SET);

    fseek(fp, 0, SEEK_END);
    long end_pos = ftell(fp);
    if (end_pos < 0) {
        fclose(fp);
        return -3;
    }
    long size = end_pos - offset;
    if (size <= 0) {
        fclose(fp);
        return -4;
    }
    fseek(fp, offset, SEEK_SET);

    int64_t n_f32 = (int64_t)(size / (long)sizeof(float));
    if (n_f32 <= 0) {
        fclose(fp);
        return -5;
    }

    float* data = (float*)malloc(sizeof(float) * (size_t)n_f32);
    if (!data) {
        fclose(fp);
        return -6;
    }

    size_t read_n = fread(data, sizeof(float), (size_t)n_f32, fp);
    fclose(fp);
    if ((int64_t)read_n != n_f32) {
        free(data);
        return -7;
    }

    outputs[0].data = data;
    if (outputs[0].shape) outputs[0].shape[0] = n_f32;
    outputs[0].ndim = 1;
    outputs[0].dtype = 0;
    return 0;
}

void component_cleanup(void) {
}
