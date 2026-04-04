import aria_core
import torch


def test_sensitivity_collect_matches_python_reference():
    base = torch.randn(1, 6, 4, dtype=torch.float32)
    embed = base.clone().requires_grad_(True)
    x = (embed * 0.5 + embed.roll(shifts=1, dims=1) * 0.25).contiguous()
    positions = torch.tensor([0, 2, 5], dtype=torch.int64)

    native = aria_core.sensitivity_collect_f32(x, embed, positions)

    ref_embed = base.clone().requires_grad_(True)
    ref_x = (ref_embed * 0.5 + ref_embed.roll(shifts=1, dims=1) * 0.25).contiguous()
    rows = []
    for idx, pos in enumerate(positions.tolist()):
        grad = torch.autograd.grad(
            ref_x.select(1, pos).sum(),
            ref_embed,
            retain_graph=idx + 1 < len(positions),
            create_graph=False,
            allow_unused=True,
        )[0]
        rows.append(grad.norm(dim=-1).squeeze(0))
    reference = torch.stack(rows, dim=0)

    assert native.shape == reference.shape
    assert torch.allclose(native, reference, atol=1e-6, rtol=1e-6)
