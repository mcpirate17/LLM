import torch
import torch.nn.functional as F
import numpy as np
from research.synthesis.kernels import triton_rmsnorm, fused_linear_gelu, triton_block_sparse_linear

def validate_rmsnorm_stability():
    print("\nValidating triton_rmsnorm numerical stability...")
    dim = 256
    x_fp32 = torch.randn(16, dim, device="cuda", dtype=torch.float32)
    w_fp32 = torch.ones(dim, device="cuda", dtype=torch.float32)
    
    # Reference FP64
    x_fp64 = x_fp32.to(torch.float64)
    w_fp64 = w_fp32.to(torch.float64)
    eps = 1e-6
    rms = torch.sqrt(torch.mean(x_fp64**2, dim=-1, keepdim=True) + eps)
    ref = (x_fp64 / rms) * w_fp64
    
    for dtype in [torch.float32, torch.float16, torch.bfloat16]:
        x = x_fp32.to(dtype)
        w = w_fp32.to(dtype)
        out = triton_rmsnorm(x, w, eps=eps)
        diff = torch.abs(out.to(torch.float64) - ref).max().item()
        print(f"  {dtype}: Max Diff vs FP64 Ref: {diff:.6f}")
        if dtype == torch.bfloat16:
            assert diff < 2e-2
        elif dtype == torch.float16:
            assert diff < 1e-2
        else:
            assert diff < 1e-5

def validate_fused_linear_gelu_stability():
    print("\nValidating fused_linear_gelu numerical stability...")
    M, N, K = 16, 128, 256
    x_fp32 = torch.randn(M, K, device="cuda", dtype=torch.float32)
    w_fp32 = torch.randn(N, K, device="cuda", dtype=torch.float32)
    b_fp32 = torch.randn(N, device="cuda", dtype=torch.float32)
    
    # Reference
    ref_linear = F.linear(x_fp32.to(torch.float64), w_fp32.to(torch.float64), b_fp32.to(torch.float64))
    ref = F.gelu(ref_linear, approximate='none')
    
    for dtype in [torch.float32, torch.float16, torch.bfloat16]:
        x = x_fp32.to(dtype)
        w = w_fp32.to(dtype)
        b = b_fp32.to(dtype)
        out = fused_linear_gelu(x, w, b)
        diff = torch.abs(out.to(torch.float64) - ref).max().item()
        print(f"  {dtype}: Max Diff vs FP64 Ref: {diff:.6f}")
        assert diff < 0.25

def validate_block_sparse_stability():
    print("\nValidating triton_block_sparse_linear numerical stability...")
    M, N, K = 32, 128, 256
    block_size = 16
    x_fp32 = torch.randn(M, K, device="cuda", dtype=torch.float32)
    w_fp32 = torch.randn(N, K, device="cuda", dtype=torch.float32)
    
    # Block aligned mask
    mask = torch.zeros(N, K, device="cuda", dtype=torch.float32)
    for i in range(0, N, block_size):
        for j in range(0, K, block_size):
            if (i+j) % 32 == 0:
                mask[i:i+block_size, j:j+block_size] = 1.0
                
    # Reference
    ref_w = w_fp32.to(torch.float64) * mask.to(torch.float64)
    ref = F.linear(x_fp32.to(torch.float64), ref_w)
    
    for dtype in [torch.float32, torch.float16, torch.bfloat16]:
        x = x_fp32.to(dtype)
        w = w_fp32.to(dtype)
        m = mask.to(dtype)
        out = triton_block_sparse_linear(x, w, m, block_size=block_size)
        diff = torch.abs(out.to(torch.float64) - ref).max().item()
        print(f"  {dtype}: Max Diff vs FP64 Ref: {diff:.6f}")
        # Triton kernels often have higher error in lower precision due to accumulation order
        assert diff < 0.25 if dtype != torch.float32 else diff < 1e-3

if __name__ == "__main__":
    if torch.cuda.is_available():
        try:
            validate_rmsnorm_stability()
            validate_fused_linear_gelu_stability()
            validate_block_sparse_stability()
            print("\nAll stability checks passed!")
        except Exception as e:
            print(f"\nStability check FAILED: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("CUDA not available, skipping stability checks.")
