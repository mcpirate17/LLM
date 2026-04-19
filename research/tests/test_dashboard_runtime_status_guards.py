from __future__ import annotations

from unittest.mock import MagicMock, patch

from research.scientist.api_routes import _helpers


def test_resolve_runner_status_does_not_start_projector_by_default():
    nb = MagicMock()
    runner = MagicMock()
    runner.is_running = False
    runner.progress.to_dict.return_value = {"status": "idle"}

    with (
        patch.object(
            _helpers,
            "get_registry_running_experiment_snapshot",
            return_value=None,
        ),
        patch.object(
            _helpers,
            "get_external_running_experiment_snapshot",
            return_value=None,
        ),
        patch.object(
            _helpers,
            "get_projected_running_experiment_snapshot",
        ) as projected,
    ):
        result = _helpers.resolve_runner_status(nb, runner)

    assert result["is_running"] is False
    projected.assert_not_called()
