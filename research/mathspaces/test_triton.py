import torch
from ..env import aria_core, HAS_ARIA_CORE as _HAS_ARIA_CORE
import triton
import triton.language as tl

@triton.jit
def tropical_matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bn, stride_bk,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr, 
    BLOCK_SIZE_N: tl.constexpr, 
    BLOCK_SIZE_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    a_ptrs = A_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B_ptr + (offs_n[None, :] * stride_bn + offs_k[:, None] * stride_bk)
    
    accumulator = tl.full((BLOCK_SIZE_M, BLOCK_SIZE_N), float('inf'), dtype=tl.float32)
    
    for k in range(0, K, BLOCK_SIZE_K):
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < K), other=float('inf'))
        b = tl.load(b_ptrs, mask=(offs_n[None, :] < N) & (offs_k[:, None] < K), other=float('inf'))
        
        # We need to broadcast
        # a: (BLOCK_M, BLOCK_K)
        # b: (BLOCK_N, BLOCK_K)
        # b_T: (BLOCK_K, BLOCK_N)
        
        # Triton 2 doesn't have expand_dims cleanly for this without view.
        # But we can just use 3D
        a_3d = tl.expand_dims(a, 2) # (BLOCK_M, BLOCK_K, 1)
        b_3d = tl.expand_dims(b, 0) # (1, BLOCK_N, BLOCK_K) ? wait, if we transpose b

        # Let's do it safely
        pass

