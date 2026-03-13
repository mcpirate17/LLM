import torch
import torch.nn.functional as F

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
ref_ssm = torch.exp(-dist / (S * 0.15)) * (dist >= 0).float()
ref_ssm = ref_ssm.tril()
ref_conv = (dist <= 5).float()

D = 256
reps = torch.randn(S, D) + 1.0  # adding mean shift
norm = F.normalize(reps, dim=-1)
sim_random = torch.mm(norm, norm.t())

print("Random+Mean reps vs Transformer:", _linear_cka(sim_random, ref_transformer))
print("Random+Mean reps vs SSM:", _linear_cka(sim_random, ref_ssm))
print("Random+Mean reps vs Conv:", _linear_cka(sim_random, ref_conv))
