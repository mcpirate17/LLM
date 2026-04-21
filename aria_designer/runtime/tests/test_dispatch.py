import numpy as np
import torch

from aria_designer.runtime.dispatch import KernelDispatcher


def test_dispatch_validator():
    dispatcher = KernelDispatcher()
    # 0 -> 1 -> 2
    nodes = ["n0", "n1", "n2"]
    edges = [(0, 1, 0, 0), (1, 2, 0, 0)]

    res = dispatcher.validate_graph(nodes, edges)
    assert res["valid"] is True
    assert res["topo_order"] == [0, 1, 2]


def test_dispatch_cycle():
    dispatcher = KernelDispatcher()
    # 0 -> 1 -> 0
    nodes = ["n0", "n1"]
    edges = [(0, 1, 0, 0), (1, 0, 0, 0)]

    res = dispatcher.validate_graph(nodes, edges)
    assert res["valid"] is False
    assert "no source" in res["error"].lower() or "cycle" in res["error"].lower()


def test_dispatch_relu():
    dispatcher = KernelDispatcher()
    x = np.array([-1, 0, 1, 2], dtype=np.float32)
    y = dispatcher.relu(x)
    expected = np.array([0, 0, 1, 2], dtype=np.float32)
    np.testing.assert_allclose(y, expected)


def test_dispatch_matmul():
    dispatcher = KernelDispatcher()
    a = np.random.randn(16, 32).astype(np.float32)
    b = np.random.randn(32, 64).astype(np.float32)
    c = dispatcher.matmul(a, b)
    expected = a @ b
    np.testing.assert_allclose(c, expected, atol=1e-5)


def test_dispatch_relu_fallback():
    # Disable native and see if it still works using torch
    dispatcher = KernelDispatcher(use_native=False)
    x = np.array([-1, 0, 1, 2], dtype=np.float32)
    y = dispatcher.relu(x)
    expected = np.array([0, 0, 1, 2], dtype=np.float32)
    np.testing.assert_allclose(y, expected)


def test_dispatch_file_loader_csv_native_or_fallback(tmp_path):
    dispatcher = KernelDispatcher()
    csv_path = tmp_path / "sample.csv"
    csv_path.write_text("a,b,c\n1,2,3\n4,5,6\n", encoding="utf-8")

    arr = dispatcher.file_loader_csv(
        str(csv_path), _max_rows=8, max_cols=3, delimiter=",", has_header=True
    )
    assert arr.shape[0] == 2
    np.testing.assert_allclose(arr[0], np.array([1.0, 2.0, 3.0], dtype=np.float32))


def test_dispatch_binary_file_reader_native_or_fallback(tmp_path):
    dispatcher = KernelDispatcher()
    bin_path = tmp_path / "sample.bin"
    src = np.array([1.5, -2.0, 3.25], dtype=np.float32)
    src.tofile(bin_path)

    out = dispatcher.binary_file_reader(str(bin_path), max_elems=16, offset_bytes=0)
    np.testing.assert_allclose(out[:3], src)


def test_dispatch_file_writer_txt_native_or_fallback(tmp_path):
    dispatcher = KernelDispatcher()
    out_path = tmp_path / "out.txt"
    data = np.array([0.5, 1.5, 2.5], dtype=np.float32)

    written = dispatcher.file_writer_txt(str(out_path), data, overwrite=True)
    assert written == 3
    lines = out_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 3


def test_dispatch_rwkv_time_mixing_torch_matches_numpy_fallback():
    dispatcher = KernelDispatcher(use_native=False)
    x_np = np.random.randn(2, 4, 3).astype(np.float32)
    decay_np = np.random.randn(3).astype(np.float32)
    bonus_np = np.random.randn(3).astype(np.float32)
    wk_np = np.random.randn(3, 3).astype(np.float32)
    wv_np = np.random.randn(3, 3).astype(np.float32)
    wr_np = np.random.randn(3, 3).astype(np.float32)

    expected = dispatcher.rwkv_time_mixing(
        x_np, decay_np, bonus_np, wk_np, wv_np, wr_np
    )
    actual = dispatcher.rwkv_time_mixing(
        torch.from_numpy(x_np),
        torch.from_numpy(decay_np),
        torch.from_numpy(bonus_np),
        torch.from_numpy(wk_np),
        torch.from_numpy(wv_np),
        torch.from_numpy(wr_np),
    )

    assert isinstance(actual, torch.Tensor)
    np.testing.assert_allclose(actual.detach().cpu().numpy(), expected, atol=3e-5)


if __name__ == "__main__":
    # Run tests manually if pytest is not used
    test_dispatch_validator()
    test_dispatch_cycle()
    test_dispatch_relu()
    test_dispatch_matmul()
    test_dispatch_relu_fallback()
    print("All dispatch tests passed!")
