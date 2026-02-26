import os
from cffi import FFI

ffibuilder = FFI()

# C header definitions for cffi
ffibuilder.cdef("""
    /* ── Graph Validator ─────────────────────────────────────────── */

    typedef struct {
        int32_t source;
        int32_t target;
        int32_t src_port;
        int32_t tgt_port;
    } AriaEdge;

    typedef struct {
        int32_t n_nodes;
        int32_t n_edges;
        AriaEdge edges[4096];
    } AriaGraph;

    typedef enum {
        ARIA_OK = 0,
        ARIA_ERR_TOO_MANY_NODES = -1,
        ARIA_ERR_TOO_MANY_EDGES = -2,
        ARIA_ERR_CYCLE_DETECTED  = -3,
        ARIA_ERR_DANGLING_EDGE   = -4,
        ARIA_ERR_DUPLICATE_EDGE  = -5,
        ARIA_ERR_SELF_LOOP       = -6,
        ARIA_ERR_NO_SOURCE       = -7,
        ARIA_ERR_DISCONNECTED    = -8,
    } AriaResult;

    typedef struct {
        AriaResult code;
        char       error[512];
        int32_t    topo_order[1024];
        int32_t    topo_len;
        int32_t    in_degree[1024];
        int32_t    out_degree[1024];
    } AriaValidationResult;

    AriaResult aria_validate_graph(const AriaGraph *graph, AriaValidationResult *result);

    /* ── Shape Inference ─────────────────────────────────────────── */

    typedef enum {
        SHAPE_IDENTITY = 0,
        SHAPE_BINARY_BROADCAST,
        SHAPE_REDUCE_LAST,
        SHAPE_REDUCE_SEQ,
        SHAPE_MATMUL,
        SHAPE_OUTER,
        SHAPE_TRANSPOSE_SD,
        SHAPE_SPLIT,
        SHAPE_CONCAT,
        SHAPE_LINEAR,
        SHAPE_RFFT,
        SHAPE_IRFFT,
        SHAPE_CUMULATIVE,
        SHAPE_SOFTMAX,
        SHAPE_CAUSAL_MASK,
        SHAPE_SCALE,
        SHAPE_BIAS,
        SHAPE_ROLL,
        SHAPE_GATHER,
        SHAPE_SCATTER,
        SHAPE_SORT,
        SHAPE_UNSORT,
        SHAPE_RULE_COUNT
    } ShapeRule;

    typedef struct {
        int32_t dims[8];
        int32_t ndim;
        int32_t valid;
    } TensorShape;

    typedef struct {
        TensorShape shape;
        int32_t     port_index;
    } PortShape;

    typedef struct {
        ShapeRule  rule;
        int32_t    n_inputs;
        int32_t    n_outputs;
        int32_t    split_n;
        int32_t    out_dim;
        int32_t    orig_seq_len;
        PortShape  input_shapes[8];
        PortShape  output_shapes[8];
    } NodeShapeSpec;

    typedef struct {
        int32_t       valid;
        char          error[512];
        NodeShapeSpec nodes[1024];
        int32_t       n_nodes;
    } ShapeInferenceResult;

    int aria_propagate_shapes(ShapeInferenceResult *result,
                              const int32_t *topo_order, int32_t topo_len,
                              const int32_t edges[][4], int32_t n_edges);

    /* ── Kernels ─────────────────────────────────────────────────── */

    void aria_relu_f32(const float *x, float *y, int64_t n);
    void aria_gelu_f32(const float *x, float *y, int64_t n);
    void aria_silu_f32(const float *x, float *y, int64_t n);
    void aria_sin_f32(const float *x, float *y, int64_t n);
    void aria_cos_f32(const float *x, float *y, int64_t n);
    void aria_add_f32(const float *a, const float *b, float *y, int64_t n);
    void aria_mul_f32(const float *a, const float *b, float *y, int64_t n);
    void aria_tropical_add_f32(const float *a, const float *b, float *y, int64_t n);
    void aria_matmul_f32(const float *A, const float *B, float *C,
                         int64_t M, int64_t K, int64_t N);
    void aria_tropical_matmul_f32(const float *A, const float *B, float *C,
                         int64_t M, int64_t K, int64_t N);
    void aria_linear_f32(const float *x, const float *W, const float *bias,
                         float *y, int64_t batch, int64_t dim_in, int64_t dim_out);
    int aria_read_csv_f32(const char *filename, float *out_data, int64_t max_rows, int64_t max_cols, char delimiter);
    int aria_filter_f32(const float *data, float *out_data, int64_t rows, int64_t cols, int64_t col_idx, float val, int op);
    int aria_file_loader_csv_f32(const char *filename, float *out_data,
                                 int64_t max_rows, int64_t max_cols,
                                 char delimiter, int has_header);
    int aria_binary_file_reader_f32(const char *filename, float *out_data,
                                    int64_t max_elems, int64_t offset_bytes);
    int aria_file_writer_txt_f32(const char *filename, const float *data,
                                 int64_t n, int overwrite);
    void aria_rmsnorm_f32(const float *x, const float *weight, float *y,
                          int64_t batch, int64_t dim, float eps);
    void aria_swiglu_f32(const float *x,
                         const float *W_gate, const float *W_up, const float *W_down,
                         const float *bias_gate, const float *bias_up, const float *bias_down,
                         float *y, float *tmp_gate, float *tmp_up,
                         int64_t batch, int64_t dim, int64_t hidden_dim);
    void aria_rwkv_channel_f32(const float *x,
                               const float *mix_k, const float *mix_r,
                               const float *W_k, const float *W_r, const float *W_v,
                               float *y, float *tmp_xk, float *tmp_xr, float *tmp_k,
                               int64_t batch, int64_t seq, int64_t dim, int64_t hidden_dim);
    void aria_tropical_center_f32(const float *x, float *y,
                          int64_t batch, int64_t seq, int64_t dim);
    void aria_hyp_distance_f32(const float *x, const float *y, float *out,
                          int64_t batch, int64_t seq, int64_t dim);
    void aria_padic_gate_f32(const float *x, float *y, int64_t n, float p);
    void aria_tropical_attention_f32(const float *x, float *y,
                          int64_t batch, int64_t seq, int64_t dim,
                          float temperature);
    void aria_tropical_gate_f32(const float *x, float *y,
                          int64_t batch, int64_t seq, int64_t dim,
                          float temperature);
""")

class AriaRuntime:
    _lib = None

    @classmethod
    def get_lib(cls):
        if cls._lib is None:
            lib_path = os.path.join(os.path.dirname(__file__), "lib", "libaria_runtime.so")
            cls._lib = ffibuilder.dlopen(lib_path)
        return cls._lib

aria_lib = AriaRuntime.get_lib()
ffi = ffibuilder
