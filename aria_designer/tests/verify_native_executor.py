import torch
import torch.nn as nn
import aria_core
from research.synthesis.graph import ComputationGraph
from aria_designer.runtime.native_executor import NativeGraphExecutor

def test_mlp_native():
    print("Building test MLP...")
    dim = 64
    g = ComputationGraph(model_dim=dim)
    in_id = g.add_input()
    
    # Single Linear
    out_id = g.add_op("linear_proj", [in_id], {"out_dim": dim})
    g.set_output(out_id)
    
    print("Compiling NativeGraphExecutor...")
    native_exec = NativeGraphExecutor(g)
    
    # Force specific weights for easier debug
    for nid, mod in native_exec.compiled_layer.ops.items():
        if hasattr(mod, "weight"):
            nn.init.constant_(mod.weight, 0.1)
            print(f"DEBUG weight pointer: {mod.weight.data_ptr():x}")
        if hasattr(mod, "bias") and mod.bias is not None:
            nn.init.constant_(mod.bias, 0.0)
            print(f"DEBUG bias pointer: {mod.bias.data_ptr():x}")
            
    native_exec._sync_parameters()
    
    x = torch.ones(1, 16, dim)
    
    print("Running Native execution...")
    # Run once to warm up
    y_native = native_exec(x)
    
    print("Running standard PyTorch execution for parity check...")
    y_torch = native_exec.compiled_layer(x).detach().cpu()
    
    # Ensure y_native is also on CPU
    y_native_cpu = y_native.detach().cpu()
    
    # Standard linear: y = x @ W.T + b
    # x is [1, 16, 64] of 1.0s.
    # W is [64, 64] of 0.1s.
    # x @ W.T should be [1, 16, 64] where each element is 64 * 0.1 = 6.4.
    
    expected_val = dim * 0.1
    print(f"DEBUG: Expected element value: {expected_val:.6f}")
    print(f"DEBUG: Native element [0,0,0]: {y_native_cpu[0,0,0]:.6f}")
    print(f"DEBUG: Native element [0,1,0]: {y_native_cpu[0,1,0]:.6f}")
    print(f"DEBUG: Torch  element [0,0,0]: {y_torch[0,0,0]:.6f}")
    print(f"DEBUG: Torch  element [0,1,0]: {y_torch[0,1,0]:.6f}")

    diff = (y_native_cpu - y_torch).abs().max().item()
    print(f"Max absolute difference: {diff:.8f}")
    
    if diff < 1e-5:
        print("✅ Parity check passed!")
    else:
        print("❌ Parity check failed!")

    # Benchmark
    import time
    n_iters = 100
    
    start = time.time()
    for _ in range(n_iters):
        _ = native_exec(x)
    native_time = (time.time() - start) / n_iters
    
    start = time.time()
    for _ in range(n_iters):
        _ = native_exec.compiled_layer(x)
    torch_time = (time.time() - start) / n_iters
    
    print(f"Native throughput: {1.0/native_time:.2f} iterations/sec")
    print(f"Torch throughput:  {1.0/torch_time:.2f} iterations/sec")
    print(f"Speedup: {torch_time/native_time:.2f}x")

if __name__ == "__main__":
    test_mlp_native()
