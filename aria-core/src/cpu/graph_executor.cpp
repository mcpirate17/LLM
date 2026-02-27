#include "graph_executor.h"
#include "kernels.h"
#include <torch/extension.h>
#include <vector>

class GraphExecutor {
public:
    GraphExecutor(int32_t n_tensors) : tensors_(n_tensors) {}

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

    void execute(const std::vector<AriaExecutableNode>& nodes) {
        for (const auto& node : nodes) {
            switch (node.type) {
                case ARIA_OP_RELU: {
                    auto x = tensors_[node.input_indices[0]];
                    auto y = tensors_[node.output_indices[0]];
                    aria_relu_f32(x.data_ptr<float>(), y.data_ptr<float>(), x.numel());
                    break;
                }
                case ARIA_OP_ADD: {
                    auto a = tensors_[node.input_indices[0]];
                    auto b = tensors_[node.input_indices[1]];
                    auto y = tensors_[node.output_indices[0]];
                    aria_add_f32(a.data_ptr<float>(), b.data_ptr<float>(), y.data_ptr<float>(), a.numel());
                    break;
                }
                case ARIA_OP_RMSNORM: {
                    auto x = tensors_[node.input_indices[0]];
                    auto w = tensors_[node.input_indices[1]];
                    auto y = tensors_[node.output_indices[0]];
                    float eps = node.params[0];
                    int64_t dim = x.size(-1);
                    int64_t batch = x.numel() / dim;
                    aria_rmsnorm_f32(x.data_ptr<float>(), w.data_ptr<float>(), y.data_ptr<float>(), batch, dim, eps);
                    break;
                }
                case ARIA_OP_MATMUL: {
                    auto a = tensors_[node.input_indices[0]];
                    auto b = tensors_[node.input_indices[1]];
                    auto y = tensors_[node.output_indices[0]];
                    int64_t M = a.size(0), K = a.size(1), N = b.size(1);
                    aria_matmul_f32(a.data_ptr<float>(), b.data_ptr<float>(), y.data_ptr<float>(), M, K, N);
                    break;
                }
                // Add more cases as needed
                default:
                    break;
            }
        }
    }

private:
    std::vector<torch::Tensor> tensors_;
};
