import torch

def _linear_cka(X, Y):
    X = X - X.mean()
    Y = Y - Y.mean()
    hsic_xy = (X * Y).sum()
    hsic_xx = (X * X).sum()
    hsic_yy = (Y * Y).sum()
    return (hsic_xy / torch.sqrt(hsic_xx * hsic_yy)).clamp(0, 1).item()

S = 64
positions = torch.arange(S).float()
dist = (positions.unsqueeze(0) - positions.unsqueeze(1)).abs()
ref_transformer = torch.exp(-dist / (S * 0.3))

sim1 = ref_transformer + torch.randn(S, S) * 0.1
sim2 = ref_transformer + torch.randn(S, S) * 0.1

print("Similarity of two noisy variants:", _linear_cka(sim1, sim2))

