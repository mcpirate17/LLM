import torch
import triton
import triton.language as tl

@triton.jit
def _tropical_matmul_kernel(
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
        
        # a is (BLOCK_M, BLOCK_K)
        # b is (BLOCK_N, BLOCK_K)
        # We want min_k (A[m, k] + B[n, k])   (B is transposed)
        
        # Broadcast to (BLOCK_M, BLOCK_N, BLOCK_K)
        a_ex = tl.expand_dims(a, 1) # (BLOCK_M, 1, BLOCK_K)
        b_ex = tl.expand_dims(b, 0) # (1, BLOCK_N, BLOCK_K)
        
        val = a_ex + b_ex
        min_val = tl.min(val, axis=2) # (BLOCK_M, BLOCK_N)
        accumulator = tl.minimum(accumulator, min_val)
        
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
        offs_k += BLOCK_SIZE_K
        
    c_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    tl.store(c_ptrs, accumulator, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))

def triton_tropical_matmul(a: torch.Tensor, b: torch.Tensor):
    # a: (B, M, K)
    # b: (B, N, K)  or (B, K, N) - we assume B,K,N and transpose to B,N,K for easy loading
    assert a.ndim == 3 and b.ndim == 3
    # If b is (B, K, N), transpose it to (B, N, K)
    # Actually wait: The caller of tropical_matmul usually passes `a` (B, S, D) and `b` (B, S, D) or (B, D, S)
    # We will just reshape so it's a 2D batch loop
    
    B, M, K = a.shape
    if b.shape[1] == K:
        # b is (B, K, N), we want (B, N, K)
        N = b.shape[2]
        b_t = b.transpose(1, 2).contiguous()
    else:
        N = b.shape[1]
        b_t = b.contiguous() # already (B, N, K)
        
    a_c = a.contiguous()
    c = torch.empty((B, M, N), device=a.device, dtype=a.dtype)
    
    # We can handle the batch dimension by keeping B outside, or adding it to grid
    def grid(META):
        return (triton.cdiv(M, META['BLOCK_SIZE_M']), triton.cdiv(N, META['BLOCK_SIZE_N']), B)
    
    # Let's verify compilation
    print("compiling")
    
