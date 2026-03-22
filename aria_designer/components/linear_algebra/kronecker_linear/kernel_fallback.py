"""Python fallback kernel for kronecker_linear."""

import math

import torch


class ComponentHandler:
    """Kronecker-factored linear: W = A ⊗ B, params 2*D vs D²."""

    def validate_config(self, config):
        return []

    def build(self, config):
        return None

    def forward(self, inputs, config):
        x = inputs["x"]  # (B, S, D)
        B, S, D = x.shape
        p = int(math.isqrt(D))
        q = D // p
        if p * q != D:
            # Fallback: find closest factorization
            for p in range(int(math.isqrt(D)), 0, -1):
                if D % p == 0:
                    q = D // p
                    break

        A = _lazy_factor(p, p, x.device, x.dtype, seed=42)
        B_mat = _lazy_factor(q, q, x.device, x.dtype, seed=137)

        # y = (x.view(B,S,p,q) @ B.T).permute(0,1,3,2) @ A.T then reshape
        out = x.view(B, S, p, q) @ B_mat.T  # (B, S, p, q)
        out = out.permute(0, 1, 3, 2) @ A.T  # (B, S, q, p)
        return {"y": out.reshape(B, S, D)}


def _lazy_factor(rows, cols, device, dtype, seed=0):
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    w = torch.randn(rows, cols, generator=gen, dtype=dtype).to(device)
    w *= cols**-0.5
    return w
