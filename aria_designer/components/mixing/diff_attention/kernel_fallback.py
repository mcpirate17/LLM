"""Python fallback kernel for diff_attention."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ComponentHandler:
    """Differential attention: two softmax maps subtracted to cancel noise."""

    def __init__(self):
        self._module = None

    def validate_config(self, config):
        return []

    def build(self, config):
        return None

    def forward(self, inputs, config):
        x = inputs["x"]
        B, S, D = x.shape
        n_heads = max(1, D // 64)
        hd = D // n_heads

        if self._module is None or self._module["q"].in_features != D:
            self._module = {
                "q": nn.Linear(D, n_heads * 2 * hd, bias=False),
                "k": nn.Linear(D, n_heads * 2 * hd, bias=False),
                "v": nn.Linear(D, n_heads * hd, bias=False),
                "o": nn.Linear(n_heads * hd, D, bias=False),
                "lam": nn.Parameter(torch.tensor(0.5)),
            }
            for key in ("q", "k", "v", "o"):
                self._module[key].to(device=x.device, dtype=x.dtype)

        q = self._module["q"](x).reshape(B, S, n_heads, 2, hd).permute(0, 2, 3, 1, 4)
        k = self._module["k"](x).reshape(B, S, n_heads, 2, hd).permute(0, 2, 3, 1, 4)
        v = self._module["v"](x).reshape(B, S, n_heads, hd).transpose(1, 2)

        scale = hd ** -0.5
        mask = torch.triu(torch.ones(S, S, device=x.device, dtype=torch.bool), diagonal=1)
        a1 = (q[:, :, 0] @ k[:, :, 0].transpose(-2, -1)) * scale
        a2 = (q[:, :, 1] @ k[:, :, 1].transpose(-2, -1)) * scale
        a1.masked_fill_(mask, float("-inf"))
        a2.masked_fill_(mask, float("-inf"))
        diff = F.softmax(a1, dim=-1) - self._module["lam"].abs() * F.softmax(a2, dim=-1)
        out = (diff @ v).transpose(1, 2).reshape(B, S, -1)
        return {"y": self._module["o"](out)}
