from pathlib import Path

import pytest
import yaml


COMPONENTS_ROOT = Path(__file__).resolve().parent.parent / "components" / "math_space"
MANIFEST_PATHS = sorted(COMPONENTS_ROOT.glob("*/manifest.yaml"))


def _load_manifest(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def test_math_space_manifest_count_baseline() -> None:
    assert len(MANIFEST_PATHS) == 32


@pytest.mark.parametrize("manifest_path", MANIFEST_PATHS, ids=lambda p: p.parent.name)
def test_math_space_native_first_contract(manifest_path: Path) -> None:
    manifest = _load_manifest(manifest_path)
    impl = manifest.get("implementation", {})

    native_kernel = impl.get("native")
    assert native_kernel
    if native_kernel != "kernel.c":
        assert (manifest_path.parent / native_kernel).exists()

    python_fallback = impl.get("python")
    if python_fallback:
        assert (manifest_path.parent / python_fallback).exists()


@pytest.mark.parametrize("manifest_path", MANIFEST_PATHS, ids=lambda p: p.parent.name)
def test_math_space_shape_determinism_numerics_contract(manifest_path: Path) -> None:
    manifest = _load_manifest(manifest_path)

    assert manifest["id"] == manifest_path.parent.name
    assert manifest["category"] == "math_space"

    for port in manifest.get("inputs", []):
        assert "dtype" in port
        if port["dtype"] == "tensor":
            assert "shape" in port
            assert isinstance(port["shape"], list)
            assert len(port["shape"]) > 0

    for port in manifest.get("outputs", []):
        assert "dtype" in port
        if port["dtype"] == "tensor":
            assert "shape" in port
            assert isinstance(port["shape"], list)
            assert len(port["shape"]) > 0

    assert isinstance(manifest.get("limits", {}).get("deterministic"), bool)
    assert isinstance(manifest.get("performance", {}).get("numerically_risky"), bool)


@pytest.mark.parametrize("manifest_path", MANIFEST_PATHS, ids=lambda p: p.parent.name)
def test_math_space_generated_manifest_contract(manifest_path: Path) -> None:
    manifest = _load_manifest(manifest_path)
    assert manifest["id"] == manifest_path.parent.name
    assert manifest["category"] == "math_space"
    assert manifest["version"] == "1.0.0"
    assert len(manifest["outputs"]) >= 1
    assert manifest["limits"]["deterministic"] is True
    assert "numerically_risky" in manifest["performance"]
