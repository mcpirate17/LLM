"""Python fallback kernel for chebyshev_spectral_mix."""

import torch


class ComponentHandler:
    """Chebyshev spectral mixing: K polynomial terms, K*D params."""

    def validate_config(self, config):
        errors = []
        K = config.get("chebyshev_order", 6)
        if not isinstance(K, int) or K < 2 or K > 16:
            errors.append("chebyshev_order must be int in [2, 16]")
        return errors

    def build(self, config):
        return None

    def forward(self, inputs, config):
        x = inputs["x"]  # (B, S, D)
        K = max(2, min(config.get("chebyshev_order", 6), 16))
        D = x.shape[-1]

        # Normalize to [-1, 1] range per-feature
        x_norm = torch.tanh(x)

        # Chebyshev coefficients: K values per feature dimension
        coeffs = _lazy_coeffs(K, D, x.device, x.dtype)

        # Chebyshev recurrence: T_0 = 1, T_1 = x, T_k = 2x*T_{k-1} - T_{k-2}
        T_prev2 = torch.ones_like(x_norm)  # T_0
        T_prev1 = x_norm  # T_1

        output = coeffs[0] * T_prev2 + coeffs[1] * T_prev1

        for k in range(2, K):
            T_k = 2 * x_norm * T_prev1 - T_prev2
            output = output + coeffs[k] * T_k
            T_prev2 = T_prev1
            T_prev1 = T_k

        return {"y": output}


def _lazy_coeffs(K, D, device, dtype):
    """Generate K coefficient vectors of size D."""
    gen = torch.Generator(device="cpu")
    gen.manual_seed(K * 65537 + D)
    coeffs = torch.randn(K, D, generator=gen, dtype=dtype).to(device)
    # Scale so initial output ≈ input magnitude
    coeffs *= K**-0.5
    # Bias T_1 coefficient toward 1.0 (identity-like initialization)
    coeffs[1] += 1.0
    return coeffs
