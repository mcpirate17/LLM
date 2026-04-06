from research.scientist.runner._helpers import (
    InflightState,
    _headroom_ratio_threshold,
    check_inflight_health,
    resolve_stage1_gate_metrics,
    stage1_learning_gate,
)


def test_headroom_ratio_threshold_relaxes_for_low_headroom_models():
    threshold = _headroom_ratio_threshold(12.0924, random_baseline=11.52)
    assert 0.98 < threshold <= 0.99


def test_check_inflight_health_allows_low_headroom_progress_at_quarter_mark():
    state = InflightState()
    quarter = 187
    total_steps = 750
    initial_loss = 12.0924
    loss_val = 11.8591

    for step in range(quarter):
        fail = check_inflight_health(
            step=step,
            loss_val=initial_loss,
            grad_norm=1.0,
            min_loss=initial_loss,
            initial_loss=initial_loss,
            total_steps=total_steps,
            state=state,
        )
        assert fail is None

    fail = check_inflight_health(
        step=quarter,
        loss_val=loss_val,
        grad_norm=1.0,
        min_loss=loss_val,
        initial_loss=initial_loss,
        total_steps=total_steps,
        state=state,
    )
    assert fail is None


def test_check_inflight_health_still_kills_clear_no_progress_runs():
    state = InflightState()
    quarter = 187
    total_steps = 750
    initial_loss = 40.0

    for step in range(quarter):
        fail = check_inflight_health(
            step=step,
            loss_val=initial_loss,
            grad_norm=1.0,
            min_loss=initial_loss,
            initial_loss=initial_loss,
            total_steps=total_steps,
            state=state,
        )
        assert fail is None

    fail = check_inflight_health(
        step=quarter,
        loss_val=39.5,
        grad_norm=1.0,
        min_loss=39.5,
        initial_loss=initial_loss,
        total_steps=total_steps,
        state=state,
    )
    assert fail is not None
    assert fail["error_type"] == "inflight_no_progress"


def test_stage1_gate_prefers_validation_loss_over_noisy_final_training_loss():
    gate_loss, gate_ratio, gate_source = resolve_stage1_gate_metrics(
        initial_loss=184.27166748046875,
        final_loss=69.03941345214844,
        validation_loss=9.062361717224121,
    )
    assert gate_source == "validation_loss"
    assert gate_loss == 9.062361717224121
    assert gate_ratio < 0.1

    passed, reason = stage1_learning_gate(
        final_loss=gate_loss,
        loss_ratio=gate_ratio,
        initial_loss=184.27166748046875,
        n_steps=1000,
        corpus_type="wikitext103",
        tokenizer="tiktoken",
        reference_losses={},
    )
    assert passed, reason
