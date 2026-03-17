import torch
import torch.nn as nn


class MockModule(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        for k, v in kwargs.items():
            setattr(self, k, v)


def execute_low_rank_proj(module: nn.Module, x: torch.Tensor) -> torch.Tensor:
    if not hasattr(module, "U") or not hasattr(module, "V"):
        return x
    # This matches research/mathspaces/compression.py implementation
    return x @ module.U @ module.V


def test_debug():
    B, S, D = 1, 1, 4
    rank = 2
    x = torch.arange(D).float().view(B, S, D)
    U = torch.arange(rank * D).float().view(rank, D)
    V = torch.arange(D * rank).float().view(D, rank)
    bias = torch.ones(D)

    # Python mathspace version uses U:(D,r), V:(r,D) and x @ U @ V
    module = MockModule(U=U.t(), V=V.t(), bias=bias)
    expected = execute_low_rank_proj(module, x) + bias

    # Our C++ logic:
    # 1. aria_linear(x, U) computes x @ U^T [B, r]
    # 2. aria_linear(tmp, V) computes tmp @ V^T [B, D]
    tmp = x.view(-1, D) @ U.t()
    actual = tmp @ V.t() + bias

    print(f"Expected (x @ U.t() @ V.t() + bias):\n{expected}")
    print(f"Actual (Manual C++ logic):\n{actual}")

    torch.testing.assert_close(actual, expected.view(-1, D))


if __name__ == "__main__":
    test_debug()
