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

# Simulate a "trained" LM representation 
# LM representations learn positional decay because closer tokens are more semantically related.
# Even a completely novel architecture will learn this decay.
D = 256
# Create representations that naturally decay in similarity over sequence dimension
reps = torch.randn(S, D)
# add exponential smoothing over sequence to simulate a token looking at past tokens
smoothed_reps = torch.zeros(S, D)
alpha = 0.8
smoothed_reps[0] = reps[0]
for i in range(1, S):
    smoothed_reps[i] = alpha * smoothed_reps[i-1] + (1 - alpha) * reps[i]

norm = F.normalize(smoothed_reps, dim=-1)
sim_trained = torch.mm(norm, norm.t())

print("Plausible Trained LM vs Transformer:", _linear_cka(sim_trained, ref_transformer))
print("Plausible Trained LM vs SSM:", _linear_cka(sim_trained, ref_ssm))
print("Plausible Trained LM vs Conv:", _linear_cka(sim_trained, ref_conv))
