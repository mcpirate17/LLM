from research.tools.audit_multiscale_rich_lane_router_phase5 import (
    _build_schedule_specs,
    _lr_at_progress,
)


def test_lr_schedule_warmup_then_decay():
    early = _lr_at_progress(
        0.05,
        peak_lr=2.6e-4,
        warmup_frac=0.2,
        hold_frac=0.5,
        end_lr_scale=0.3,
        decay_style="cosine",
    )
    mid = _lr_at_progress(
        0.3,
        peak_lr=2.6e-4,
        warmup_frac=0.2,
        hold_frac=0.5,
        end_lr_scale=0.3,
        decay_style="cosine",
    )
    late = _lr_at_progress(
        0.9,
        peak_lr=2.6e-4,
        warmup_frac=0.2,
        hold_frac=0.5,
        end_lr_scale=0.3,
        decay_style="cosine",
    )
    assert early < mid
    assert late < mid


def test_schedule_specs_cover_baseline_and_targeted_curricula():
    names = {spec.name for spec in _build_schedule_specs()}
    assert "baseline_fixed" in names
    assert "optimizer_refined" in names
    assert "delayed_recursion_ramp" in names
    assert "gentle_routing_curriculum" in names
