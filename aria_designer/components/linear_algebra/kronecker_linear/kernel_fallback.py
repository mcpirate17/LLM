"""Python fallback kernel for kronecker_linear."""

import math

from aria_designer.components._weight_cache import cached_randn


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
            for p in range(int(math.isqrt(D)), 0, -1):
                if D % p == 0:
                    q = D // p
                    break

        A = cached_randn(
            p,
            p,
            seed=42,
            device=x.device,
            dtype=x.dtype,
            scale=p**-0.5,
        )
        B_mat = cached_randn(
            q,
            q,
            seed=137,
            device=x.device,
            dtype=x.dtype,
            scale=q**-0.5,
        )

        out = x.view(B, S, p, q) @ B_mat.T  # (B, S, p, q)
        out = out.permute(0, 1, 3, 2) @ A.T  # (B, S, q, p)
        return {"y": out.reshape(B, S, D)}
