import time
import torch
import torch.nn as nn
from eval.hellaswag_eval import screening_hellaswag_eval


class DummyModel(nn.Module):
    def forward(self, x):
        return torch.randn(x.size(0), x.size(1), 100)


device = "cuda" if torch.cuda.is_available() else "cpu"
model = DummyModel().to(device)
t0 = time.time()
res = screening_hellaswag_eval(model, 100, device, n_examples=50)
print("Time:", time.time() - t0, res)
