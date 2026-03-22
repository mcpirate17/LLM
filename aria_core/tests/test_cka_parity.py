import torch
import aria_core
import time
from research.eval.fingerprint import _linear_cka


def test_cka_parity():
    print("Testing Native CKA Parity...")

    n = 64
    n * n

    # 1. Random matrices
    X = torch.randn(n, n)
    Y = torch.randn(n, n)

    # 2. PyTorch Reference
    start = time.time()
    pytorch_res = _linear_cka(X, Y)
    py_time = time.time() - start

    # 3. Native aria_core
    # Note: _linear_cka will now automatically use aria_core if on CPU
    start = time.time()
    native_res = aria_core.linear_cka_f32(X.contiguous(), Y.contiguous())
    native_time = time.time() - start

    print(f"PyTorch CKA: {pytorch_res:.6f} ({py_time * 1000:.2f}ms)")
    print(f"Native CKA:  {native_res:.6f} ({native_time * 1000:.2f}ms)")

    diff = abs(pytorch_res - native_res)
    print(f"Difference:  {diff:.6e}")

    assert diff < 1e-5, "CKA parity check failed!"

    # 4. Identity test
    id_res = aria_core.linear_cka_f32(X, X)
    print(f"Self-similarity (Identity): {id_res:.6f}")
    assert abs(id_res - 1.0) < 1e-5

    print("CKA PARITY TEST PASSED")


if __name__ == "__main__":
    test_cka_parity()
