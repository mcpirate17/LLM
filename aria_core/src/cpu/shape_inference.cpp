/**
 * shape_inference.c — Shape propagation through computation graphs.
 *
 * Each shape rule defines how output shapes derive from input shapes.
 * -1 in a dimension means "symbolic/unknown" (must match consistently).
 */
#include "shape_inference.h"
#include <stdio.h>
#include <string.h>

/* ── Helpers ───────────────────────────────────────────────────────── */

static void copy_shape(TensorShape *dst, const TensorShape *src) {
    memcpy(dst, src, sizeof(TensorShape));
}

static int shapes_compatible(const TensorShape *a, const TensorShape *b) {
    if (a->ndim != b->ndim) return 0;
    for (int i = 0; i < a->ndim; i++) {
        if (a->dims[i] == -1 || b->dims[i] == -1) continue; /* symbolic */
        if (a->dims[i] != b->dims[i]) return 0;
    }
    return 1;
}

static void broadcast_shape(TensorShape *out, const TensorShape *a, const TensorShape *b) {
    out->ndim = a->ndim > b->ndim ? a->ndim : b->ndim;
    out->valid = 1;
    for (int i = 0; i < out->ndim; i++) {
        int ai = i < a->ndim ? a->dims[a->ndim - 1 - i] : 1;
        int bi = i < b->ndim ? b->dims[b->ndim - 1 - i] : 1;
        if (ai == -1 || bi == -1) {
            out->dims[out->ndim - 1 - i] = -1;
        } else if (ai == bi) {
            out->dims[out->ndim - 1 - i] = ai;
        } else if (ai == 1) {
            out->dims[out->ndim - 1 - i] = bi;
        } else if (bi == 1) {
            out->dims[out->ndim - 1 - i] = ai;
        } else {
            out->valid = 0; /* incompatible */
            return;
        }
    }
}

/* ── Shape rule implementations ────────────────────────────────────── */

int aria_apply_shape_rule(NodeShapeSpec *spec) {
    TensorShape *in0 = &spec->input_shapes[0].shape;

    switch (spec->rule) {

    case SHAPE_IDENTITY:
    case SHAPE_CUMULATIVE:
    case SHAPE_SOFTMAX:
    case SHAPE_CAUSAL_MASK:
    case SHAPE_SCALE:
    case SHAPE_BIAS:
    case SHAPE_ROLL:
    case SHAPE_GATHER:
    case SHAPE_SCATTER:
    case SHAPE_UNSORT:
        /* Output = input */
        if (!in0->valid) return -1;
        copy_shape(&spec->output_shapes[0].shape, in0);
        spec->output_shapes[0].shape.valid = 1;
        break;

    case SHAPE_BINARY_BROADCAST:
    case SHAPE_OUTER: {
        if (!in0->valid || !spec->input_shapes[1].shape.valid) return -1;
        TensorShape *in1 = &spec->input_shapes[1].shape;
        broadcast_shape(&spec->output_shapes[0].shape, in0, in1);
        if (!spec->output_shapes[0].shape.valid) return -1;
        break;
    }

    case SHAPE_REDUCE_LAST:
        /* (B,S,D) -> (B,S,1) */
        if (!in0->valid || in0->ndim < 1) return -1;
        copy_shape(&spec->output_shapes[0].shape, in0);
        spec->output_shapes[0].shape.dims[in0->ndim - 1] = 1;
        spec->output_shapes[0].shape.valid = 1;
        break;

    case SHAPE_REDUCE_SEQ:
        /* (B,S,D) -> (B,1,D) */
        if (!in0->valid || in0->ndim < 2) return -1;
        copy_shape(&spec->output_shapes[0].shape, in0);
        spec->output_shapes[0].shape.dims[in0->ndim - 2] = 1;
        spec->output_shapes[0].shape.valid = 1;
        break;

    case SHAPE_MATMUL: {
        /* (B,S,D) x (B,D,K) -> (B,S,K) */
        if (!in0->valid || !spec->input_shapes[1].shape.valid) return -1;
        TensorShape *in1 = &spec->input_shapes[1].shape;
        if (in0->ndim < 2 || in1->ndim < 2) return -1;
        TensorShape *out = &spec->output_shapes[0].shape;
        out->ndim = in0->ndim;
        out->valid = 1;
        /* Copy batch dims from first input */
        for (int i = 0; i < in0->ndim - 2; i++) {
            out->dims[i] = in0->dims[i];
        }
        /* S from first input, K from second */
        out->dims[out->ndim - 2] = in0->dims[in0->ndim - 2];
        out->dims[out->ndim - 1] = in1->dims[in1->ndim - 1];
        break;
    }

    case SHAPE_TRANSPOSE_SD:
        /* (B,S,D) -> (B,D,S) */
        if (!in0->valid || in0->ndim < 2) return -1;
        copy_shape(&spec->output_shapes[0].shape, in0);
        {
            int n = in0->ndim;
            int32_t tmp = spec->output_shapes[0].shape.dims[n - 1];
            spec->output_shapes[0].shape.dims[n - 1] = spec->output_shapes[0].shape.dims[n - 2];
            spec->output_shapes[0].shape.dims[n - 2] = tmp;
        }
        spec->output_shapes[0].shape.valid = 1;
        break;

    case SHAPE_SPLIT: {
        /* (B,S,D) -> N x (B,S,D/N) */
        if (!in0->valid || in0->ndim < 1) return -1;
        int n_split = spec->split_n > 0 ? spec->split_n : 2;
        int last_dim = in0->dims[in0->ndim - 1];
        int split_dim = (last_dim > 0) ? last_dim / n_split : -1;

        for (int i = 0; i < spec->n_outputs && i < n_split; i++) {
            copy_shape(&spec->output_shapes[i].shape, in0);
            spec->output_shapes[i].shape.dims[in0->ndim - 1] = split_dim;
            spec->output_shapes[i].shape.valid = 1;
        }
        break;
    }

    case SHAPE_CONCAT: {
        /* N x (B,S,D_i) -> (B,S,sum(D_i)) */
        if (!in0->valid) return -1;
        copy_shape(&spec->output_shapes[0].shape, in0);
        int32_t total = 0;
        int all_known = 1;
        for (int i = 0; i < spec->n_inputs; i++) {
            TensorShape *inp = &spec->input_shapes[i].shape;
            if (!inp->valid) return -1;
            int d = inp->dims[inp->ndim - 1];
            if (d < 0) { all_known = 0; break; }
            total += d;
        }
        spec->output_shapes[0].shape.dims[in0->ndim - 1] = all_known ? total : -1;
        spec->output_shapes[0].shape.valid = 1;
        break;
    }

    case SHAPE_LINEAR:
        /* (B,S,D_in) -> (B,S,D_out) */
        if (!in0->valid || in0->ndim < 1) return -1;
        copy_shape(&spec->output_shapes[0].shape, in0);
        if (spec->out_dim > 0) {
            spec->output_shapes[0].shape.dims[in0->ndim - 1] = spec->out_dim;
        }
        /* else: same as input (D_out = D_in) */
        spec->output_shapes[0].shape.valid = 1;
        break;

    case SHAPE_RFFT:
        /* (B,S,D) -> (B,S/2+1,D) */
        if (!in0->valid || in0->ndim < 2) return -1;
        copy_shape(&spec->output_shapes[0].shape, in0);
        {
            int s = in0->dims[in0->ndim - 2];
            spec->output_shapes[0].shape.dims[in0->ndim - 2] = (s > 0) ? s / 2 + 1 : -1;
        }
        spec->output_shapes[0].shape.valid = 1;
        break;

    case SHAPE_IRFFT:
        /* (B,S/2+1,D) -> (B,S,D) — needs original seq length */
        if (!in0->valid || in0->ndim < 2) return -1;
        copy_shape(&spec->output_shapes[0].shape, in0);
        if (spec->orig_seq_len > 0) {
            spec->output_shapes[0].shape.dims[in0->ndim - 2] = spec->orig_seq_len;
        } else {
            int sh = in0->dims[in0->ndim - 2];
            spec->output_shapes[0].shape.dims[in0->ndim - 2] = (sh > 0) ? (sh - 1) * 2 : -1;
        }
        spec->output_shapes[0].shape.valid = 1;
        break;

    case SHAPE_SORT:
        /* Output: sorted tensor + indices (same shape for both) */
        if (!in0->valid) return -1;
        copy_shape(&spec->output_shapes[0].shape, in0);
        spec->output_shapes[0].shape.valid = 1;
        if (spec->n_outputs > 1) {
            copy_shape(&spec->output_shapes[1].shape, in0);
            spec->output_shapes[1].shape.valid = 1;
        }
        break;

    default:
        return -1;
    }

    return 0;
}

int aria_propagate_shapes(ShapeInferenceResult *result,
                          const int32_t *topo_order, int32_t topo_len,
                          const int32_t edges[][4], int32_t n_edges) {
    result->valid = 1;
    result->error[0] = '\0';

    /* Process nodes in topological order */
    for (int32_t ti = 0; ti < topo_len; ti++) {
        int32_t node_idx = topo_order[ti];
        NodeShapeSpec *spec = &result->nodes[node_idx];

        /* Copy input shapes from connected upstream output ports */
        for (int32_t e = 0; e < n_edges; e++) {
            int32_t src_node = edges[e][0];
            int32_t tgt_node = edges[e][1];
            int32_t src_port = edges[e][2];
            int32_t tgt_port = edges[e][3];

            if (tgt_node == node_idx) {
                NodeShapeSpec *src_spec = &result->nodes[src_node];
                if (src_port < src_spec->n_outputs && tgt_port < spec->n_inputs) {
                    copy_shape(&spec->input_shapes[tgt_port].shape,
                               &src_spec->output_shapes[src_port].shape);
                }
            }
        }

        /* Apply shape rule */
        if (aria_apply_shape_rule(spec) != 0) {
            result->valid = 0;
            snprintf(result->error, sizeof(result->error),
                     "Shape inference failed at node %d (rule=%d)", node_idx, spec->rule);
            return -1;
        }
    }

    return 0;
}
