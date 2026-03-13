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

# Case 1: Random representations
D = 256
reps = torch.randn(S, D)
norm = F.normalize(reps, dim=-1)
sim_random = torch.mm(norm, norm.t())

print("Random reps vs Transformer:", _linear_cka(sim_random, ref_transformer))
print("Random reps vs SSM:", _linear_cka(sim_random, ref_ssm))
print("Random reps vs Conv:", _linear_cka(sim_random, ref_conv))

# Case 2: Very local representations (similar to Conv)
reps_local = torch.randn(S, D)
# smooth them over a window of 5
reps_local_smoothed = F.conv1d(reps_local.unsqueeze(0).transpose(1, 2), torch.ones(1, 1, 5)/5, padding=2).transpose(1, 2).squeeze(0)
norm_local = F.normalize(reps_local_smoothed, dim=-1)
sim_local = torch.mm(norm_local, norm_local.t())
print("Local reps vs Conv:", _linear_cka(sim_local, ref_conv))
print("Local reps vs Transformer:", _linear_cka(sim_local, ref_transformer))

# Novelty score: 1.0 - max(cka)
max_cka_random = max(_linear_cka(sim_random, ref_transformer), _linear_cka(sim_random, ref_ssm), _linear_cka(sim_random, ref_conv))
print("Random novelty:", 1.0 - max_cka_random)

max_cka_local = max(_linear_cka(sim_local, ref_transformer), _linear_cka(sim_local, ref_ssm), _linear_cka(sim_local, ref_conv))
print("Local novelty:", 1.0 - max_cka_local)
