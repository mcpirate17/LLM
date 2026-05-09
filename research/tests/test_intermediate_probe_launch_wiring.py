from __future__ import annotations

import pytest

from research.scientist.api_routes._experiment_launch import parse_start_request


@pytest.mark.parametrize(
    "payload, run_ar, run_binding, changes",
    [
        ({"mode": "single"}, False, False, {}),
        (
            {"mode": "single", "enable_intermediate_probes": True},
            True,
            True,
            {"run_ar_intermediate": True, "run_binding_multislot": True},
        ),
        (
            {
                "mode": "single",
                "run_ar_intermediate": True,
                "run_binding_multislot": False,
            },
            True,
            False,
            {},
        ),
        (
            {
                "mode": "single",
                "enable_intermediate_probes": True,
                "run_ar_intermediate": True,
                "run_binding_multislot": False,
            },
            True,
            True,
            {"run_binding_multislot": True},
        ),
    ],
    ids=[
        "default-off",
        "bundle-enables-both",
        "direct-flags-passthrough",
        "bundle-fills-only-missing",
    ],
)
def test_intermediate_probe_launch_bundle(
    payload, run_ar, run_binding, changes
) -> None:
    start = parse_start_request(payload)

    assert start.config.run_ar_intermediate is run_ar
    assert start.config.run_binding_multislot is run_binding
    assert start.intermediate_probe_changes == changes
