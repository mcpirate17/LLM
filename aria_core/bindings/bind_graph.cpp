/**
 * bind_graph.cpp — GraphExecutor, graph validation, shape inference,
 *                  smoke test, fingerprint metrics, and canonical topo sort.
 */
#include "bind_common.h"
#include <unordered_map>
#include <queue>

// ── GraphExecutor Wrapper ──

struct ExecNode {
    int type;
    std::vector<int> inputs;
    std::vector<int> outputs;
    std::vector<float> params;
};

class PyGraphExecutor {
public:
    PyGraphExecutor(int32_t n_tensors) : tensors_(n_tensors) {}

    void set_tensor(int32_t index, torch::Tensor t) {
        if (index >= 0 && index < (int32_t)tensors_.size()) {
            tensors_[index] = t;
        }
    }

    torch::Tensor get_tensor(int32_t index) {
        if (index >= 0 && index < (int32_t)tensors_.size()) {
            return tensors_[index];
        }
        return torch::Tensor();
    }

    void bake(const py::list& py_nodes) {
        baked_nodes_.clear();
        for (auto item : py_nodes) {
            baked_nodes_.push_back(parse_node(item));
        }
    }

    void execute() {
        for (size_t i = 0; i < baked_nodes_.size(); ++i) {
            execute_node(baked_nodes_[i]);
        }
    }

    void execute_list(const py::list& py_nodes) {
        for (auto item : py_nodes) {
            execute_node(parse_node(item));
        }
    }

private:
    static ExecNode parse_node(const py::handle& item) {
        auto node_dict = item.cast<py::dict>();
        ExecNode node;
        node.type = node_dict["type"].cast<int>();
        node.inputs = node_dict["inputs"].cast<std::vector<int>>();
        node.outputs = node_dict["outputs"].cast<std::vector<int>>();
        if (node_dict.contains("params")) {
            node.params = node_dict["params"].cast<std::vector<float>>();
        }
        return node;
    }

    void execute_node(const ExecNode& node) {
        for (int in_idx : node.inputs) {
            if (in_idx < 0 || in_idx >= (int)tensors_.size()) return;
        }
        for (int out_idx : node.outputs) {
            if (out_idx < 0 || out_idx >= (int)tensors_.size()) return;
        }

        switch (node.type) {
            case 0: { // RELU
                auto& x = tensors_[node.inputs[0]];
                auto& y = tensors_[node.outputs[0]];
                aria_relu_f32(x.data_ptr<float>(), y.data_ptr<float>(), x.numel());
                break;
            }
            case 1: { // GELU
                auto& x = tensors_[node.inputs[0]];
                auto& y = tensors_[node.outputs[0]];
                aria_gelu_f32(x.data_ptr<float>(), y.data_ptr<float>(), x.numel());
                break;
            }
            case 2: { // SILU
                auto& x = tensors_[node.inputs[0]];
                auto& y = tensors_[node.outputs[0]];
                aria_silu_f32(x.data_ptr<float>(), y.data_ptr<float>(), x.numel());
                break;
            }
            case 3: { // ADD
                auto& a = tensors_[node.inputs[0]];
                auto& b = tensors_[node.inputs[1]];
                auto& y = tensors_[node.outputs[0]];
                aria_add_f32(a.data_ptr<float>(), b.data_ptr<float>(), y.data_ptr<float>(), a.numel());
                break;
            }
            case 4: { // MUL
                auto& a = tensors_[node.inputs[0]];
                auto& b = tensors_[node.inputs[1]];
                auto& y = tensors_[node.outputs[0]];
                aria_mul_f32(a.data_ptr<float>(), b.data_ptr<float>(), y.data_ptr<float>(), a.numel());
                break;
            }
            case 5: { // SUB
                auto& a = tensors_[node.inputs[0]];
                auto& b = tensors_[node.inputs[1]];
                auto& y = tensors_[node.outputs[0]];
                aria_sub_f32(a.data_ptr<float>(), b.data_ptr<float>(), y.data_ptr<float>(), a.numel());
                break;
            }
            case 6: { // RMSNORM
                if (node.inputs.size() < 2) return;
                auto& x = tensors_[node.inputs[0]];
                auto& w = tensors_[node.inputs[1]];
                auto& y = tensors_[node.outputs[0]];
                float eps = node.params.size() > 0 ? node.params[0] : 1e-6f;
                int64_t dim = x.size(-1);
                int64_t batch = x.numel() / dim;
                aria_rmsnorm_f32(x.data_ptr<float>(), w.data_ptr<float>(), y.data_ptr<float>(), batch, dim, eps);
                break;
            }
            case 7: { // LAYERNORM
                if (node.inputs.size() < 3) return;
                auto& x = tensors_[node.inputs[0]];
                auto& w = tensors_[node.inputs[1]];
                auto& b = tensors_[node.inputs[2]];
                auto& y = tensors_[node.outputs[0]];
                float eps = node.params.size() > 0 ? node.params[0] : 1e-6f;
                int64_t dim = x.size(-1);
                int64_t batch = x.numel() / dim;
                float* bias_ptr = (b.numel() > 0) ? b.data_ptr<float>() : nullptr;
                aria_layernorm_f32(x.data_ptr<float>(), w.data_ptr<float>(), bias_ptr, y.data_ptr<float>(), batch, dim, eps);
                break;
            }
            case 8: { // MATMUL
                auto& a = tensors_[node.inputs[0]];
                auto& b = tensors_[node.inputs[1]];
                auto& y = tensors_[node.outputs[0]];
                int64_t M = a.size(0), K = a.size(1), N = b.size(1);
                aria_matmul_f32(a.data_ptr<float>(), b.data_ptr<float>(), y.data_ptr<float>(), M, K, N);
                break;
            }
            case 9: { // LINEAR
                if (node.inputs.size() < 2) return;
                auto& x = tensors_[node.inputs[0]];
                auto& w = tensors_[node.inputs[1]];
                auto& y = tensors_[node.outputs[0]];
                float* bias_ptr = nullptr;
                if (node.inputs.size() >= 3) {
                    auto& b = tensors_[node.inputs[2]];
                    if (b.numel() > 0) bias_ptr = b.data_ptr<float>();
                }
                int64_t dim_in = x.size(-1);
                int64_t batch = x.numel() / dim_in;
                int64_t dim_out = w.size(0);
                aria_linear_f32(x.data_ptr<float>(), w.data_ptr<float>(), bias_ptr, y.data_ptr<float>(), batch, dim_in, dim_out);
                break;
            }
            case 10: { // SOFTMAX
                auto& x = tensors_[node.inputs[0]];
                auto& y = tensors_[node.outputs[0]];
                int64_t dim = x.size(-1);
                int64_t batch = x.numel() / dim;
                aria_softmax_f32(x.data_ptr<float>(), y.data_ptr<float>(), batch, dim);
                break;
            }
        }
    }

    std::vector<torch::Tensor> tensors_;
    std::vector<ExecNode> baked_nodes_;
};

// ═══ Graph validation & shape inference ═══

static py::dict validate_graph(int32_t n_nodes, std::vector<std::vector<int32_t>> edges, std::vector<int32_t> op_codes) {
    AriaGraph graph;
    memset(&graph, 0, sizeof(graph));
    graph.n_nodes = n_nodes;
    graph.n_edges = static_cast<int32_t>(edges.size());
    for (int32_t i = 0; i < graph.n_edges; i++) {
        graph.edges[i].source = edges[i][0];
        graph.edges[i].target = edges[i][1];
        graph.edges[i].src_port = edges[i].size() > 2 ? edges[i][2] : 0;
        graph.edges[i].tgt_port = edges[i].size() > 3 ? edges[i][3] : 0;
    }
    for (int32_t i = 0; i < n_nodes && i < (int32_t)op_codes.size(); i++) {
        graph.op_codes[i] = op_codes[i];
    }

    AriaValidationResult result;
    memset(&result, 0, sizeof(result));
    AriaResult rc = aria_validate_graph(&graph, &result);

    py::dict out;
    if (rc == ARIA_OK) {
        out["valid"] = true;
        std::vector<int32_t> topo(result.topo_order, result.topo_order + result.topo_len);
        std::vector<int32_t> in_deg(result.in_degree, result.in_degree + n_nodes);
        std::vector<int32_t> out_deg(result.out_degree, result.out_degree + n_nodes);
        out["topo_order"] = topo;
        out["in_degrees"] = in_deg;
        out["out_degrees"] = out_deg;
    } else {
        out["valid"] = false;
        out["error"] = std::string(result.error);
        out["code"] = static_cast<int>(rc);
    }
    return out;
}

static py::dict proactive_gating(int32_t n_nodes, std::vector<std::vector<int32_t>> edges, std::vector<int32_t> op_codes,
                          std::vector<int32_t> norm_opcodes, std::vector<int32_t> param_opcodes, std::vector<int32_t> linear_opcodes) {
    AriaGraph graph;
    memset(&graph, 0, sizeof(graph));
    graph.n_nodes = n_nodes;
    graph.n_edges = static_cast<int32_t>(edges.size());
    for (int32_t i = 0; i < graph.n_edges; i++) {
        graph.edges[i].source = edges[i][0];
        graph.edges[i].target = edges[i][1];
    }
    std::unordered_set<int32_t> norm_set(norm_opcodes.begin(), norm_opcodes.end());
    std::unordered_set<int32_t> param_set(param_opcodes.begin(), param_opcodes.end());
    std::unordered_set<int32_t> linear_set(linear_opcodes.begin(), linear_opcodes.end());
    for (int32_t i = 0; i < n_nodes && i < (int32_t)op_codes.size(); i++) {
        graph.op_codes[i] = op_codes[i];
        graph.is_norm[i] = norm_set.count(op_codes[i]) ? 1 : 0;
        graph.is_parameterized[i] = param_set.count(op_codes[i]) ? 1 : 0;
        graph.is_linear[i] = linear_set.count(op_codes[i]) ? 1 : 0;
    }

    AriaValidationResult val;
    aria_validate_graph(&graph, &val);

    AriaProactiveGatingResult res;
    aria_proactive_gating(&graph, &val, &res);

    py::dict out;
    out["passed"] = res.passed != 0;
    out["reason"] = std::string(res.reason);
    out["max_depth"] = res.max_depth;
    out["n_toxic_motifs"] = res.n_toxic_motifs;
    out["has_normalization_gap"] = res.has_normalization_gap != 0;
    return out;
}

static py::dict analyze_graph(int32_t n_nodes,
                       std::vector<std::vector<int32_t>> edges,
                       std::vector<int32_t> op_codes,
                       int32_t output_node,
                       int32_t input_node) {
    AriaGraph graph;
    memset(&graph, 0, sizeof(graph));
    graph.n_nodes = n_nodes;
    graph.n_edges = static_cast<int32_t>(edges.size());
    for (int32_t i = 0; i < graph.n_edges; i++) {
        graph.edges[i].source = edges[i][0];
        graph.edges[i].target = edges[i][1];
        graph.edges[i].src_port = edges[i].size() > 2 ? edges[i][2] : 0;
        graph.edges[i].tgt_port = edges[i].size() > 3 ? edges[i][3] : 0;
    }
    for (int32_t i = 0; i < n_nodes && i < (int32_t)op_codes.size(); i++) {
        graph.op_codes[i] = op_codes[i];
    }

    AriaGraphAnalysisResult result;
    memset(&result, 0, sizeof(result));
    AriaResult rc = aria_analyze_graph(&graph, output_node, input_node, &result);

    py::dict out;
    if (rc == ARIA_OK) {
        out["valid"] = true;
        std::vector<int32_t> reachable(
            result.reachable_nodes, result.reachable_nodes + result.reachable_len
        );
        out["reachable_nodes"] = reachable;
        out["max_depth"] = result.max_depth;
        out["has_input_path"] = result.has_input_path != 0;
    } else {
        out["valid"] = false;
        out["error"] = std::string(result.error);
        out["code"] = static_cast<int>(rc);
    }
    return out;
}

static py::dict propagate_shapes(
    std::vector<int32_t> topo_order,
    std::vector<std::vector<int32_t>> edges,
    std::vector<py::dict> node_rules
) {
    ShapeInferenceResult res;
    memset(&res, 0, sizeof(res));
    res.n_nodes = static_cast<int32_t>(node_rules.size());

    for (int32_t i = 0; i < res.n_nodes; i++) {
        auto& rule_data = node_rules[i];
        NodeShapeSpec& node = res.nodes[i];
        node.rule = static_cast<ShapeRule>(rule_data["rule"].cast<int>());
        node.n_inputs = rule_data.contains("n_inputs") ? rule_data["n_inputs"].cast<int32_t>() : 1;
        node.n_outputs = rule_data.contains("n_outputs") ? rule_data["n_outputs"].cast<int32_t>() : 1;
        node.split_n = rule_data.contains("split_n") ? rule_data["split_n"].cast<int32_t>() : 0;
        node.out_dim = rule_data.contains("out_dim") ? rule_data["out_dim"].cast<int32_t>() : -1;
        node.orig_seq_len = rule_data.contains("orig_seq_len") ? rule_data["orig_seq_len"].cast<int32_t>() : 0;

        if (rule_data.contains("input_shapes")) {
            auto shapes = rule_data["input_shapes"].cast<std::vector<py::object>>();
            for (size_t p = 0; p < shapes.size(); p++) {
                if (!shapes[p].is_none()) {
                    auto dims = shapes[p].cast<std::vector<int32_t>>();
                    node.input_shapes[p].shape.ndim = static_cast<int32_t>(dims.size());
                    for (size_t d = 0; d < dims.size(); d++)
                        node.input_shapes[p].shape.dims[d] = dims[d];
                    node.input_shapes[p].shape.valid = 1;
                }
            }
        }
    }

    int32_t (*c_edges)[4] = new int32_t[edges.size()][4];
    for (size_t i = 0; i < edges.size(); i++) {
        c_edges[i][0] = edges[i][0];
        c_edges[i][1] = edges[i][1];
        c_edges[i][2] = edges[i].size() > 2 ? edges[i][2] : 0;
        c_edges[i][3] = edges[i].size() > 3 ? edges[i][3] : 0;
    }

    int rc = aria_propagate_shapes(&res, topo_order.data(),
                                    static_cast<int32_t>(topo_order.size()),
                                    c_edges, static_cast<int32_t>(edges.size()));
    delete[] c_edges;

    py::dict out;
    if (rc == 0) {
        out["valid"] = true;
        py::list all_shapes;
        for (int32_t i = 0; i < res.n_nodes; i++) {
            py::list node_out;
            for (int32_t p = 0; p < res.nodes[i].n_outputs; p++) {
                auto& shape = res.nodes[i].output_shapes[p].shape;
                if (shape.valid) {
                    py::list dims;
                    for (int32_t d = 0; d < shape.ndim; d++)
                        dims.append(shape.dims[d]);
                    node_out.append(dims);
                } else {
                    node_out.append(py::none());
                }
            }
            all_shapes.append(node_out);
        }
        out["output_shapes"] = all_shapes;
    } else {
        out["valid"] = false;
        out["error"] = std::string(res.error);
    }
    return out;
}

static py::list canonical_topo_sort(
    int32_t n_nodes,
    std::vector<std::pair<int32_t, int32_t>> edges,
    std::vector<std::string> op_names,
    std::vector<std::string> config_strs,
    std::vector<std::vector<int32_t>> node_inputs
) {
    if (n_nodes == 0) return py::list();

    std::vector<int32_t> in_degree(n_nodes, 0);
    std::vector<std::vector<int32_t>> children(n_nodes);

    for (const auto& edge : edges) {
        if (edge.first >= 0 && edge.first < n_nodes && edge.second >= 0 && edge.second < n_nodes) {
            children[edge.first].push_back(edge.second);
            in_degree[edge.second]++;
        }
    }

    struct CanonicalKey {
        std::string op_name;
        std::vector<int32_t> input_ranks;
        std::string config_str;
        int32_t node_id;

        bool operator>(const CanonicalKey& other) const {
            if (op_name != other.op_name) return op_name > other.op_name;
            if (input_ranks != other.input_ranks) return input_ranks > other.input_ranks;
            if (config_str != other.config_str) return config_str > other.config_str;
            return node_id > other.node_id;
        }
    };

    std::priority_queue<CanonicalKey, std::vector<CanonicalKey>, std::greater<CanonicalKey>> ready;
    std::vector<int32_t> canonical_id_map(n_nodes, -1);
    std::vector<int32_t> order;
    order.reserve(n_nodes);

    auto push_node = [&](int32_t nid) {
        CanonicalKey key;
        key.node_id = nid;
        key.op_name = op_names[nid];
        key.config_str = config_strs[nid];
        for (int32_t iid : node_inputs[nid]) {
            key.input_ranks.push_back(canonical_id_map[iid]);
        }
        ready.push(std::move(key));
    };

    for (int32_t i = 0; i < n_nodes; i++) {
        if (in_degree[i] == 0) {
            push_node(i);
        }
    }

    while (!ready.empty()) {
        CanonicalKey key = std::move(ready.top());
        ready.pop();
        int32_t u = key.node_id;
        canonical_id_map[u] = static_cast<int32_t>(order.size());
        order.push_back(u);
        for (int32_t v : children[u]) {
            in_degree[v]--;
            if (in_degree[v] == 0) {
                push_node(v);
            }
        }
    }

    py::list res;
    for (int32_t nid : order) res.append(nid);
    return res;
}

// ═══ Fingerprint Metrics ═══

static torch::Tensor interaction_metrics_f32_py(torch::Tensor influence, torch::Tensor positions) {
    CHECK_INPUT(influence);
    TORCH_CHECK(positions.dtype() == torch::kInt64, "positions must be int64");
    CHECK_CPU(positions); CHECK_CONTIGUOUS(positions);
    TORCH_CHECK(influence.dim() == 2, "influence must be 2D [n_pos, seq_len]");
    int64_t n_pos = influence.size(0);
    int64_t seq_len = influence.size(1);
    auto out = torch::empty({4}, torch::kFloat32);
    aria_interaction_metrics_f32(
        influence.data_ptr<float>(), positions.data_ptr<int64_t>(),
        out.data_ptr<float>(), n_pos, seq_len
    );
    return out;
}

static torch::Tensor sensitivity_metrics_f32_py(torch::Tensor sens) {
    CHECK_INPUT(sens);
    TORCH_CHECK(sens.dim() == 2, "sens must be 2D [n_pos, seq_len]");
    int64_t n_pos = sens.size(0);
    int64_t seq_len = sens.size(1);
    auto out = torch::empty({3}, torch::kFloat32);
    aria_sensitivity_metrics_f32(
        sens.data_ptr<float>(), out.data_ptr<float>(), n_pos, seq_len
    );
    return out;
}

// ═══ Smoke Test ═══

static py::object graph_ir_field(const py::object &graph_ir, const char *name) {
    if (py::hasattr(graph_ir, name)) {
        return graph_ir.attr(name);
    }
    if (py::isinstance<py::dict>(graph_ir)) {
        py::dict d = graph_ir.cast<py::dict>();
        if (d.contains(name)) {
            return d[name];
        }
    }
    throw std::runtime_error(std::string("graph_ir missing field: ") + name);
}

static int32_t smoke_role_code(const py::object &role) {
    static const std::unordered_map<std::string, int32_t> kRoleCodes = {
        {"PROJECT", 0}, {"NORMALIZE", 1}, {"ACTIVATE", 2}, {"MIX", 3},
        {"ROUTE", 4}, {"GATE", 5}, {"POSITION", 6}, {"REDUCE", 7}, {"RESIDUAL", 8},
    };
    auto it = kRoleCodes.find(py::str(role.attr("name")).cast<std::string>());
    return it == kRoleCodes.end() ? 9 : it->second;
}

static py::dict smoke_test_graph_ir_py(py::object graph_ir, int32_t d_model, int32_t seq_len) {
    (void)d_model;
    (void)seq_len;
    auto op_codes_obj = graph_ir_field(graph_ir, "op_codes");
    auto input_indices_obj = graph_ir_field(graph_ir, "input_indices");
    int32_t output_node = graph_ir_field(graph_ir, "output_node_idx").cast<int32_t>();

    py::array_t<int32_t, py::array::c_style | py::array::forcecast> op_codes_arr(op_codes_obj);
    py::array_t<int32_t, py::array::c_style | py::array::forcecast> input_indices_arr(input_indices_obj);
    TORCH_CHECK(op_codes_arr.ndim() == 1, "graph_ir.op_codes must be 1D");
    TORCH_CHECK(input_indices_arr.ndim() == 2 && input_indices_arr.shape(1) >= 2,
                "graph_ir.input_indices must be [N, >=2]");

    int32_t n_nodes = static_cast<int32_t>(op_codes_arr.shape(0));
    auto op_codes = op_codes_arr.unchecked<1>();
    auto input_indices = input_indices_arr.unchecked<2>();

    py::module primitives = py::module_::import("research.synthesis.primitives");
    py::module op_roles_mod = py::module_::import("research.synthesis.op_roles");
    py::object reverse_opcode_map = primitives.attr("REVERSE_OPCODE_MAP");
    py::object primitive_registry = primitives.attr("PRIMITIVE_REGISTRY");
    py::object get_role = op_roles_mod.attr("get_role");

    std::vector<int32_t> edges(static_cast<size_t>(n_nodes) * 2, -1);
    std::vector<int32_t> op_roles(static_cast<size_t>(n_nodes), 9);
    std::vector<int32_t> has_params_flag(static_cast<size_t>(n_nodes), 0);
    std::vector<int32_t> preserves_grad(static_cast<size_t>(n_nodes), 1);

    for (int32_t i = 0; i < n_nodes; ++i) {
        edges[static_cast<size_t>(i) * 2] = input_indices(i, 0);
        edges[static_cast<size_t>(i) * 2 + 1] = input_indices(i, 1);
        int32_t opcode = op_codes(i);
        if (opcode == 0) {
            op_roles[i] = 10;
            continue;
        }
        py::object op_name_obj = reverse_opcode_map.attr("get")(opcode);
        if (op_name_obj.is_none()) continue;
        py::object prim = primitive_registry.attr("get")(op_name_obj);
        py::object role = get_role(op_name_obj);
        op_roles[i] = smoke_role_code(role);
        if (!prim.is_none()) {
            has_params_flag[i] = prim.attr("has_params").cast<bool>() ? 1 : 0;
            preserves_grad[i] = prim.attr("preserves_gradient").cast<bool>() ? 1 : 0;
        }
    }

    SmokeTestResult r = smoke_test_graph(
        n_nodes, edges.data(), op_roles.data(),
        has_params_flag.data(), preserves_grad.data(), output_node
    );
    py::dict result;
    result["ok"] = static_cast<bool>(r.ok);
    result["has_params"] = static_cast<bool>(r.has_params);
    result["grad_flows"] = static_cast<bool>(r.grad_flows);
    result["no_unsafe"] = static_cast<bool>(r.no_unsafe);
    result["no_nan"] = static_cast<bool>(r.no_unsafe);
    return result;
}

static py::dict smoke_test_graph_py(
    int32_t n_nodes,
    std::vector<int32_t> edges,
    std::vector<int32_t> op_roles,
    std::vector<int32_t> has_params_flag,
    std::vector<int32_t> preserves_grad,
    int32_t output_node
) {
    SmokeTestResult r = smoke_test_graph(
        n_nodes, edges.data(), op_roles.data(),
        has_params_flag.data(), preserves_grad.data(), output_node
    );
    py::dict result;
    result["ok"] = (bool)r.ok;
    result["has_params"] = (bool)r.has_params;
    result["grad_flows"] = (bool)r.grad_flows;
    result["no_unsafe"] = (bool)r.no_unsafe;
    return result;
}

// ═══ Registration ═══

void bind_graph(py::module_ &m) {
    m.def("canonical_topo_sort", &canonical_topo_sort, "Stable topological sort for graph fingerprinting");
    m.def("analyze_graph", &analyze_graph, "Native graph reachability/depth analysis");
    // Fingerprint metrics
    m.def("interaction_metrics_f32", &interaction_metrics_f32_py, "Interaction metrics from influence matrix");
    m.def("sensitivity_metrics_f32", &sensitivity_metrics_f32_py, "Sensitivity metrics from Jacobian matrix");
    // Smoke test
    m.def("smoke_test_graph", &smoke_test_graph_py, "Fast structural smoke test for computation graphs",
          py::arg("n_nodes"), py::arg("edges"), py::arg("op_roles"),
          py::arg("has_params_flag"), py::arg("preserves_grad"), py::arg("output_node"));
    m.def("smoke_test_graph", &smoke_test_graph_ir_py, "Fast structural smoke test for graph IR",
          py::arg("graph_ir"), py::arg("d_model"), py::arg("seq_len"));
    // Graph validation & shape inference
    m.def("validate_graph", &validate_graph, "Validate a DAG: cycle detection, topological sort",
          py::arg("n_nodes"), py::arg("edges"), py::arg("op_codes") = std::vector<int32_t>());
    m.def("proactive_gating", &proactive_gating, "Native proactive stability and toxicity gating",
          py::arg("n_nodes"), py::arg("edges"), py::arg("op_codes"),
          py::arg("norm_opcodes"), py::arg("param_opcodes"), py::arg("linear_opcodes"));
    m.def("propagate_shapes", &propagate_shapes, "Propagate tensor shapes through a graph",
          py::arg("topo_order"), py::arg("edges"), py::arg("node_rules"));
    // GraphExecutor class
    py::class_<PyGraphExecutor>(m, "GraphExecutor")
        .def(py::init<int32_t>())
        .def("set_tensor", &PyGraphExecutor::set_tensor)
        .def("get_tensor", &PyGraphExecutor::get_tensor)
        .def("bake", &PyGraphExecutor::bake)
        .def("execute", &PyGraphExecutor::execute)
        .def("execute_list", &PyGraphExecutor::execute_list);
}
