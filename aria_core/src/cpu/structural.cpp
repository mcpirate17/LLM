extern "C" {

void aria_concat_f32(
    const float** inputs,
    const int64_t* sizes,
    int32_t n_inputs,
    float* output
) {
    int64_t offset = 0;
    for (int32_t i = 0; i < n_inputs; ++i) {
        const int64_t n = sizes[i];
        if (n <= 0) {
            continue;
        }
        memcpy(output + offset, inputs[i], (size_t)n * sizeof(float));
        offset += n;
    }
}

void aria_split_f32(
    const float* input,
    float** outputs,
    const int64_t* sizes,
    int32_t n_outputs
) {
    int64_t offset = 0;
    for (int32_t i = 0; i < n_outputs; ++i) {
        const int64_t n = sizes[i];
        if (n <= 0) {
            continue;
        }
        memcpy(outputs[i], input + offset, (size_t)n * sizeof(float));
        offset += n;
    }
}

}  // extern "C"
