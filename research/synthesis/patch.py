with open('/home/tim/Projects/LLM/aria_core/bindings/bindings.cpp', 'r') as f:
    text = f.read()

text = text.replace(
    'DEFINE_BINARY_F32(tropical_add_f32, aria_tropical_add_f32)',
    '''torch::Tensor tropical_add_f32(torch::Tensor a, torch::Tensor b) {
    CHECK_INPUT_ANY(a); CHECK_INPUT_ANY(b);
    TORCH_CHECK(a.numel() == b.numel(), "a and b must have same numel");
    auto y = torch::empty_like(a);
    if (a.is_cuda()) {
        launch_cuda_tropical_add_f32(a.data_ptr<float>(), b.data_ptr<float>(), y.data_ptr<float>(), a.numel());
    } else {
        aria_tropical_add_f32(a.data_ptr<float>(), b.data_ptr<float>(), y.data_ptr<float>(), a.numel());
    }
    return y;
}'''
)

text = text.replace(
    '''torch::Tensor clifford_geometric_product_cl30_f32(torch::Tensor a, torch::Tensor b) { CHECK_INPUT(a); CHECK_INPUT(b); auto y = torch::empty_like(a); aria_clifford_geometric_product_cl30_f32(a.data_ptr<float>(), b.data_ptr<float>(), y.data_ptr<float>(), a.numel()/8); return y; }''',
    '''torch::Tensor clifford_geometric_product_cl30_f32(torch::Tensor a, torch::Tensor b) {
    CHECK_INPUT_ANY(a); CHECK_INPUT_ANY(b);
    auto y = torch::empty_like(a);
    if (a.is_cuda()) {
        launch_cuda_clifford_geometric_product_cl30_f32(a.data_ptr<float>(), b.data_ptr<float>(), y.data_ptr<float>(), a.numel());
    } else {
        aria_clifford_geometric_product_cl30_f32(a.data_ptr<float>(), b.data_ptr<float>(), y.data_ptr<float>(), a.numel()/8);
    }
    return y;
}'''
)

with open('/home/tim/Projects/LLM/aria_core/bindings/bindings.cpp', 'w') as f:
    f.write(text)

