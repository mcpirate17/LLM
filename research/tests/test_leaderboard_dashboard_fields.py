import pytest

from research.scientist import leaderboard_dashboard_fields as fields
from research.scientist.api_routes import leaderboard_bp
from research.scientist.notebook import notebook_leaderboard


pytestmark = pytest.mark.unit


def test_leaderboard_dashboard_field_groups_are_shared():
    assert leaderboard_bp.CHAMPION_DASHBOARD_FIELDS is fields.CHAMPION_DASHBOARD_FIELDS
    assert (
        leaderboard_bp.INTERMEDIATE_SCREEN_DASHBOARD_FIELDS
        is fields.INTERMEDIATE_SCREEN_DASHBOARD_FIELDS
    )
    assert (
        notebook_leaderboard._PROGRAM_RESULT_DASHBOARD_ALIAS_FIELDS
        is fields.PROGRAM_RESULT_DASHBOARD_ALIAS_FIELDS
    )


def test_leaderboard_dashboard_field_order_preserves_response_contract():
    assert fields.CHAMPION_DASHBOARD_FIELDS[:3] == (
        "champion_floor_protocol_version",
        "champion_steps_to_floor",
        "champion_floor_loss",
    )
    assert fields.CHAMPION_DASHBOARD_FIELDS[-3:] == (
        "ar_validation_rank_score",
        "ar_validation_status",
        "ar_validation_elapsed_ms",
    )
    assert fields.INTERMEDIATE_SCREEN_DASHBOARD_FIELDS[:3] == (
        "ar_intermediate_metric_version",
        "ar_intermediate_diagnostic_score",
        "ar_intermediate_held_pair_acc",
    )
    assert fields.INTERMEDIATE_SCREEN_DASHBOARD_FIELDS[-3:] == (
        "ar_curriculum_elapsed_ms",
        "ar_curriculum_status",
        "ar_curriculum_error",
    )
    assert fields.PROGRAM_RESULT_DASHBOARD_ALIAS_FIELDS == (
        *fields.CHAMPION_DASHBOARD_FIELDS,
        *fields.V2_INVESTIGATION_DASHBOARD_FIELDS,
        *fields.INTERMEDIATE_SCREEN_DASHBOARD_FIELDS,
    )
