import torch
from component_fab.generator.memory_primitives import UniversalMasterLane


def probe_causality():
    dim = 32
    model = UniversalMasterLane(dim=dim)
    model.eval()

    B, L = 1, 10
    x = torch.randn(B, L, dim, requires_grad=True)
    y = model(x)

    print("Checking causality...")
    # Perturb input at step 5
    # If causal, y[0..4] should have ZERO gradient with respect to x[5]
    for i in range(L):
        # Scalar loss from output at step i
        loss = y[0, i].sum()
        loss.backward(retain_graph=True)

        # Gradients of x[j] with respect to output y[i]
        grads = x.grad[0].norm(dim=-1)
        # If j > i, grad should be 0
        leaks = (grads[i + 1 :] > 0).any()

        print(f"y[{i}] depends on x[:{i + 1}] | Grads: {grads.tolist()}")
        if leaks:
            print(f"!!! FAILURE: y[{i}] leaked from future tokens!")

        x.grad.zero_()


if __name__ == "__main__":
    probe_causality()
