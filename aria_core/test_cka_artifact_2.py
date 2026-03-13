import torch
from research.eval.cka_references import get_default_store

store = get_default_store()
refs = store.get_references()

print([(k, v.shape, v.sum().item()) for k, v in refs.items()])
