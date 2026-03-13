import torch
import torch.nn.functional as F
from research.eval.cka_references import get_default_store
from research.eval.fingerprint import _linear_cka

store = get_default_store()
refs = store.get_references()

ref_t = refs["transformer"].float()
ref_s = refs["ssm"].float()

def prep(flat):
    norm = F.normalize(flat, dim=-1)
    D = norm.shape[-1]
    S = 64
    sim = torch.mm(norm.reshape(-1, D), norm.reshape(-1, D).t())
    return sim[:S, :S]

sim_t = prep(ref_t)
sim_s = prep(ref_s)

print("Transformer vs SSM CKA:", _linear_cka(sim_t, sim_s))
