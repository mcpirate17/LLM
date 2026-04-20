from __future__ import annotations

import json
import sqlite3

from research.tools.backpopulate_screening_metrics import (
    _apply_row_updates,
    _backpopulate_provenance_context,
    _dedupe_rows_by_fingerprint_keep_latest,
    _evaluate_row_payload,
    _fetch_rows,
    _interleave_rows_by_family,
    _needs_post_train,
    _needs_rapid,
    _print_backpopulate_summary,
    _recover_hellaswag_after_gate_failure,
    _select_updates,
)
from research.tools.backfill import store_probe_results
from research.scientist.notebook import LabNotebook
from research.scientist.runner import RunConfig


class _Row(dict):
    def keys(self):
        return super().keys()


_BACKPOPULATE_SCHEMA_DDL = """
CREATE TABLE experiments (
    experiment_id TEXT PRIMARY KEY,
    experiment_type TEXT,
    config_json TEXT,
    timestamp REAL
);
CREATE TABLE program_results (
    result_id TEXT PRIMARY KEY,
    experiment_id TEXT,
    graph_fingerprint TEXT,
    graph_json TEXT,
    stage0_passed INTEGER,
    stage05_passed INTEGER,
    stage1_passed INTEGER,
    n_train_steps INTEGER,
    train_budget_steps INTEGER,
    rapid_screening_passed INTEGER,
    rapid_screening_elapsed_ms REAL,
    rapid_screening_steps_completed INTEGER,
    rapid_screening_max_steps INTEGER,
    wikitext_perplexity REAL,
    hellaswag_acc REAL,
    induction_auc REAL,
    binding_auc REAL,
    binding_composite REAL,
    trust_label TEXT,
    comparability_label TEXT,
    data_provenance_json TEXT
);
"""

_PROGRAM_ROW_DEFAULTS = {
    "result_id": "row_1",
    "experiment_id": "exp_1",
    "graph_fingerprint": "fp1",
    "graph_json": '{"nodes":{}}',
    "stage0_passed": 1,
    "stage05_passed": 1,
    "stage1_passed": 1,
    "n_train_steps": 50,
    "train_budget_steps": 50,
    "rapid_screening_passed": 1,
    "rapid_screening_elapsed_ms": 1.0,
    "rapid_screening_steps_completed": 10,
    "rapid_screening_max_steps": 10,
    "wikitext_perplexity": 7.0,
    "hellaswag_acc": None,
    "induction_auc": 0.2,
    "binding_auc": 0.3,
    "binding_composite": 0.25,
    "trust_label": "candidate_grade",
    "comparability_label": "candidate_comparable",
    "data_provenance_json": json.dumps({"graph": {"graph_family": "dense"}}),
}

_PROGRAM_ROW_COLS = list(_PROGRAM_ROW_DEFAULTS.keys())


def _make_program_row(**overrides):
    row = {**_PROGRAM_ROW_DEFAULTS, **overrides}
    return tuple(row[c] for c in _PROGRAM_ROW_COLS)


_EVAL_PAYLOAD_DEFAULTS = {
    "result_id": "rid-test",
    "graph_fingerprint": "fp-test",
    "graph_json": '{"nodes":{}}',
    "config_json": "{}",
    "stage0_passed": 1,
    "stage05_passed": 1,
    "stage1_passed": 1,
    "n_train_steps": 50,
    "train_budget_steps": 50,
    "rapid_screening_passed": 1,
    "rapid_screening_elapsed_ms": 1.0,
    "rapid_screening_steps_completed": 10,
    "rapid_screening_max_steps": 10,
    "wikitext_perplexity": 7.0,
    "hellaswag_acc": 0.31,
    "induction_auc": 0.1,
    "binding_auc": 0.2,
    "binding_composite": 0.15,
    "trust_label": "candidate_grade",
    "comparability_label": "candidate_comparable",
    "data_provenance_json": "{}",
    "timestamp": 0.0,
    "experiment_id": "exp-test",
}


def _make_eval_payload(**overrides):
    return {**_EVAL_PAYLOAD_DEFAULTS, **overrides}


def test_needs_rapid_requires_stage05():
    row = _Row(
        stage0_passed=1,
        stage05_passed=0,
        rapid_screening_passed=None,
        rapid_screening_elapsed_ms=None,
        rapid_screening_steps_completed=None,
        rapid_screening_max_steps=None,
    )
    assert not _needs_rapid(row, force=False)


def test_needs_post_train_requires_train_steps():
    row = _Row(
        stage0_passed=1,
        stage05_passed=1,
        n_train_steps=None,
        wikitext_perplexity=None,
        hellaswag_acc=None,
        induction_auc=None,
        binding_auc=None,
        binding_composite=None,
    )
    assert not _needs_post_train(
        row,
        force=False,
        target_fields=("wikitext_perplexity",),
    )


def test_needs_post_train_allows_reference_fallback_budget():
    row = _Row(
        result_id="ref_gpt2_test",
        stage0_passed=1,
        stage05_passed=1,
        stage1_passed=1,
        n_train_steps=None,
        train_budget_steps=None,
        trust_label="reference",
        comparability_label="reference_comparable",
        binding_auc=None,
    )
    assert _needs_post_train(
        row,
        force=False,
        target_fields=("binding_auc",),
    )


def test_needs_post_train_binding_only_does_not_require_train_steps():
    row = _Row(
        stage0_passed=1,
        stage05_passed=1,
        stage1_passed=0,
        n_train_steps=None,
        train_budget_steps=None,
        trust_label="candidate_grade",
        comparability_label="candidate_comparable",
        binding_auc=None,
    )
    assert _needs_post_train(
        row,
        force=False,
        target_fields=("binding_auc",),
    )


def test_needs_post_train_compile_only_induction_does_not_require_train_steps():
    row = _Row(
        stage0_passed=1,
        stage05_passed=1,
        stage1_passed=0,
        n_train_steps=None,
        train_budget_steps=None,
        induction_auc=None,
    )
    assert _needs_post_train(
        row,
        force=False,
        target_fields=("induction_auc",),
    )


def test_needs_post_train_compile_only_hellaswag_does_not_require_train_steps():
    row = _Row(
        stage0_passed=1,
        stage05_passed=1,
        stage1_passed=0,
        n_train_steps=None,
        train_budget_steps=None,
        hellaswag_acc=None,
    )
    assert _needs_post_train(
        row,
        force=False,
        target_fields=("hellaswag_acc",),
    )


def test_select_updates_only_fills_missing_without_force():
    row = _Row(induction_auc=0.1, binding_auc=None, rapid_screening_elapsed_ms=None)
    updates = {
        "induction_auc": 0.2,
        "binding_auc": 0.3,
        "rapid_screening_elapsed_ms": 1812.0,
    }
    assert _select_updates(row, updates, force=False) == {
        "binding_auc": 0.3,
        "rapid_screening_elapsed_ms": 1812.0,
    }


def test_select_updates_overwrites_with_force():
    row = _Row(induction_auc=0.1, binding_auc=None)
    updates = {"induction_auc": 0.2, "binding_auc": 0.3}
    assert _select_updates(row, updates, force=True) == updates


def test_force_replays_existing_binding_metrics(monkeypatch):
    payload = _make_eval_payload(
        result_id="rid-force",
        graph_fingerprint="fp-force",
        experiment_id="exp-force",
    )

    called = {"compile_only": 0}

    def _fake_compile_only(*args, **kwargs):
        called["compile_only"] += 1
        return {"binding_auc": 0.35}

    monkeypatch.setattr(
        "research.tools.backpopulate_screening_metrics._run_compile_only_post_eval",
        _fake_compile_only,
    )

    result = _evaluate_row_payload(
        payload=payload,
        device="cpu",
        force=True,
        skip_rapid=True,
        skip_post_train=False,
        post_train_stability_runs=1,
        stability_wikitext_rel_tol=0.1,
        stability_hellaswag_abs_tol=0.05,
        stability_probe_abs_tol=0.01,
        allow_insufficient_learning_metrics=False,
        post_train_target="binding",
        selection_slice="trusted_candidates",
    )

    assert called["compile_only"] == 1
    assert result["post_needed"] == 1
    assert result["updates"]["binding_auc"] == 0.35


def test_binding_target_uses_binding_probe_only_path(monkeypatch):
    payload = _make_eval_payload(
        result_id="rid-binding",
        graph_fingerprint="fp-binding",
        experiment_id="exp-binding",
        n_train_steps=None,
        train_budget_steps=None,
        binding_auc=None,
        binding_composite=None,
    )
    called = {"compile_only": 0, "post_train": 0}

    def _fake_compile_only(*args, **kwargs):
        called["compile_only"] += 1
        return {"binding_auc": 0.42}

    def _fake_run_post_train(*args, **kwargs):
        called["post_train"] += 1
        return {"binding_auc": 0.99}

    monkeypatch.setattr(
        "research.tools.backpopulate_screening_metrics._run_compile_only_post_eval",
        _fake_compile_only,
    )
    monkeypatch.setattr(
        "research.tools.backpopulate_screening_metrics._run_post_train",
        _fake_run_post_train,
    )

    result = _evaluate_row_payload(
        payload=payload,
        device="cpu",
        force=False,
        skip_rapid=True,
        skip_post_train=False,
        post_train_stability_runs=1,
        stability_wikitext_rel_tol=0.1,
        stability_hellaswag_abs_tol=0.05,
        stability_probe_abs_tol=0.01,
        allow_insufficient_learning_metrics=False,
        post_train_target="binding",
        selection_slice="backfill",
    )

    assert called == {"compile_only": 1, "post_train": 0}
    assert result["post_needed"] == 1
    assert result["updates"]["binding_auc"] == 0.42


def test_all_target_uses_compile_only_path(monkeypatch):
    payload = {
        "result_id": "rid-all",
        "graph_fingerprint": "fp-all",
        "graph_json": '{"nodes":{}}',
        "config_json": "{}",
        "stage0_passed": 1,
        "stage05_passed": 1,
        "stage1_passed": 0,
        "n_train_steps": None,
        "train_budget_steps": None,
        "rapid_screening_passed": 1,
        "rapid_screening_elapsed_ms": 1.0,
        "rapid_screening_steps_completed": 10,
        "rapid_screening_max_steps": 10,
        "wikitext_perplexity": None,
        "hellaswag_acc": None,
        "induction_auc": None,
        "binding_auc": None,
        "binding_composite": None,
        "trust_label": "candidate_grade",
        "comparability_label": "candidate_comparable",
        "data_provenance_json": "{}",
        "timestamp": 0.0,
        "experiment_id": "exp-all",
    }
    called = {"compile_only": 0, "post_train": 0}

    def _fake_compile_only(*args, **kwargs):
        called["compile_only"] += 1
        return {
            "hellaswag_acc": 0.27,
            "induction_auc": 0.05,
            "binding_auc": 0.09,
            "binding_composite": 0.042,
        }

    def _fake_run_post_train(*args, **kwargs):
        called["post_train"] += 1
        return {"binding_auc": 0.99}

    monkeypatch.setattr(
        "research.tools.backpopulate_screening_metrics._run_compile_only_post_eval",
        _fake_compile_only,
    )
    monkeypatch.setattr(
        "research.tools.backpopulate_screening_metrics._run_post_train",
        _fake_run_post_train,
    )

    result = _evaluate_row_payload(
        payload=payload,
        device="cpu",
        force=False,
        skip_rapid=True,
        skip_post_train=False,
        post_train_stability_runs=1,
        stability_wikitext_rel_tol=0.1,
        stability_hellaswag_abs_tol=0.05,
        stability_probe_abs_tol=0.01,
        allow_insufficient_learning_metrics=False,
        post_train_target="all",
        selection_slice="backfill",
    )

    assert called == {"compile_only": 1, "post_train": 0}
    assert result["post_needed"] == 1
    assert result["updates"]["hellaswag_acc"] == 0.27
    assert result["updates"]["induction_auc"] == 0.05
    assert result["updates"]["binding_auc"] == 0.09


def test_recover_hellaswag_after_gate_failure_respects_skip(monkeypatch):
    called = {"count": 0}

    def _fake_eval(model, vocab_size, device):
        called["count"] += 1
        return {"hellaswag_acc": 0.31, "hellaswag_status": "ok", "hellaswag_total": 50}

    monkeypatch.setattr(
        "research.eval.hellaswag_eval.screening_hellaswag_eval",
        _fake_eval,
    )
    cfg = RunConfig(skip_screening_hellaswag=True)
    assert (
        _recover_hellaswag_after_gate_failure(model=object(), config=cfg, device="cuda")
        == {}
    )
    assert called["count"] == 0


def test_recover_hellaswag_after_gate_failure_returns_metrics(monkeypatch):
    def _fake_eval(model, vocab_size, device):
        assert vocab_size == 100277
        assert device == "cuda:0"
        return {"hellaswag_acc": 0.31, "hellaswag_status": "ok", "hellaswag_total": 50}

    monkeypatch.setattr(
        "research.eval.hellaswag_eval.screening_hellaswag_eval",
        _fake_eval,
    )
    cfg = RunConfig(vocab_size=100277, skip_screening_hellaswag=False)
    assert _recover_hellaswag_after_gate_failure(
        model=object(),
        config=cfg,
        device="cuda:0",
    ) == {
        "hellaswag_acc": 0.31,
        "hellaswag_status": "ok",
        "hellaswag_n_examples": 50,
    }


def test_backpopulate_provenance_context_shape():
    class _Args:
        audit_prefix = "s1_hellaswag_full_20260406b"
        audit_experiment_id = "exp123"
        audit_source_script = "run_s1_backpopulate"
        post_train_target = "hellaswag"
        allow_insufficient_learning_metrics = True
        post_train_stability_runs = 2
        worker_timeout_seconds = 600

    ctx = _backpopulate_provenance_context(_Args(), "cuda")
    assert ctx["kind"] == "screening_metric_backfill"
    assert ctx["prefix"] == "s1_hellaswag_full_20260406b"
    assert ctx["experiment_id"] == "exp123"
    assert ctx["source_script"] == "run_s1_backpopulate"
    assert ctx["post_train_target"] == "hellaswag"
    assert ctx["device"] == "cuda"


def test_interleave_rows_by_family_round_robin():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("CREATE TABLE rows (result_id TEXT, data_provenance_json TEXT)")
    payloads = [
        ("d1", {"graph": {"graph_family": "dense"}}),
        ("d2", {"graph": {"graph_family": "dense"}}),
        ("s1", {"graph": {"graph_family": "sparse"}}),
        ("r1", {"graph": {"graph_family": "routing"}}),
    ]
    conn.executemany(
        "INSERT INTO rows VALUES (?, ?)",
        [(rid, json.dumps(payload)) for rid, payload in payloads],
    )
    rows = conn.execute("SELECT * FROM rows ORDER BY result_id ASC").fetchall()
    interleaved = _interleave_rows_by_family(rows)
    assert [row["result_id"] for row in interleaved] == ["d1", "r1", "s1", "d2"]
    conn.close()


def test_fetch_rows_trusted_candidate_slice_balances_family_order():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript(_BACKPOPULATE_SCHEMA_DDL)
    conn.executemany(
        "INSERT INTO experiments VALUES (?, ?, ?, ?)",
        [
            ("exp_dense_1", "synthesis", "{}", 1.0),
            ("exp_dense_2", "synthesis", "{}", 2.0),
            ("exp_sparse_1", "synthesis", "{}", 3.0),
            ("exp_routing_1", "synthesis", "{}", 4.0),
        ],
    )
    rows = [
        _make_program_row(
            result_id="dense_1",
            experiment_id="exp_dense_1",
            graph_fingerprint="fp1",
            data_provenance_json=json.dumps({"graph": {"graph_family": "dense"}}),
        ),
        _make_program_row(
            result_id="dense_2",
            experiment_id="exp_dense_2",
            graph_fingerprint="fp2",
            data_provenance_json=json.dumps({"graph": {"graph_family": "dense"}}),
        ),
        _make_program_row(
            result_id="sparse_1",
            experiment_id="exp_sparse_1",
            graph_fingerprint="fp3",
            data_provenance_json=json.dumps({"graph": {"graph_family": "sparse"}}),
        ),
        _make_program_row(
            result_id="routing_1",
            experiment_id="exp_routing_1",
            graph_fingerprint="fp4",
            data_provenance_json=json.dumps({"graph": {"graph_family": "routing"}}),
        ),
    ]
    conn.executemany(
        """
        INSERT INTO program_results
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    selected = _fetch_rows(
        conn,
        result_ids=[],
        limit=3,
        force=False,
        selection_slice="trusted_candidates",
        balance_by_family=True,
        target_post_fields=("hellaswag_acc",),
    )
    assert [row["result_id"] for row in selected] == [
        "dense_1",
        "sparse_1",
        "routing_1",
    ]
    conn.close()


def test_fetch_rows_selects_reference_missing_binding_without_train_steps():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript(_BACKPOPULATE_SCHEMA_DDL)
    conn.execute(
        "INSERT INTO experiments VALUES (?, ?, ?, ?)",
        ("exp_ref", "backfill", "{}", 1.0),
    )
    conn.execute(
        f"INSERT INTO program_results VALUES ({', '.join('?' for _ in _PROGRAM_ROW_COLS)})",
        _make_program_row(
            result_id="ref_gpt2_test",
            experiment_id="exp_ref",
            graph_fingerprint="fp_ref",
            n_train_steps=None,
            train_budget_steps=None,
            hellaswag_acc=0.3,
            induction_auc=0.05,
            binding_auc=None,
            binding_composite=None,
            trust_label="reference",
            comparability_label="reference_comparable",
        ),
    )
    selected = _fetch_rows(
        conn,
        result_ids=[],
        limit=10,
        force=False,
        selection_slice="backfill",
        balance_by_family=False,
        target_post_fields=("binding_auc",),
    )
    assert [row["result_id"] for row in selected] == ["ref_gpt2_test"]
    conn.close()


def test_dedupe_rows_by_fingerprint_keeps_latest_row():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute(
        "CREATE TABLE rows (result_id TEXT, graph_fingerprint TEXT, timestamp REAL)"
    )
    conn.executemany(
        "INSERT INTO rows VALUES (?, ?, ?)",
        [
            ("older_fp1", "fp1", 1.0),
            ("latest_fp1", "fp1", 2.0),
            ("only_fp2", "fp2", 1.5),
        ],
    )
    rows = conn.execute(
        "SELECT * FROM rows ORDER BY timestamp DESC, result_id DESC"
    ).fetchall()
    deduped = _dedupe_rows_by_fingerprint_keep_latest(rows)
    assert [row["result_id"] for row in deduped] == ["latest_fp1", "only_fp2"]
    conn.close()


def test_fetch_rows_nonref_unique_fingerprints_dedupes_latest():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript(
        """
        CREATE TABLE experiments (
            experiment_id TEXT PRIMARY KEY,
            experiment_type TEXT,
            config_json TEXT,
            timestamp REAL
        );
        CREATE TABLE program_results (
            result_id TEXT PRIMARY KEY,
            experiment_id TEXT,
            graph_fingerprint TEXT,
            graph_json TEXT,
            stage0_passed INTEGER,
            stage05_passed INTEGER,
            stage1_passed INTEGER,
            n_train_steps INTEGER,
            train_budget_steps INTEGER,
            rapid_screening_passed INTEGER,
            rapid_screening_elapsed_ms REAL,
            rapid_screening_steps_completed INTEGER,
            rapid_screening_max_steps INTEGER,
            wikitext_perplexity REAL,
            hellaswag_acc REAL,
            induction_auc REAL,
            binding_auc REAL,
            binding_composite REAL,
            ar_auc REAL,
            blimp_overall_accuracy REAL,
            ncd_score REAL,
            trust_label TEXT,
            comparability_label TEXT,
            data_provenance_json TEXT,
            config_json TEXT,
            timestamp REAL
        );
        """
    )
    conn.executemany(
        "INSERT INTO experiments VALUES (?, ?, ?, ?)",
        [
            ("exp1", "synthesis", "{}", 1.0),
            ("exp2", "synthesis", "{}", 2.0),
            ("exp3", "synthesis", "{}", 3.0),
        ],
    )
    conn.executemany(
        """
        INSERT INTO program_results
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                "fp1_old",
                "exp1",
                "fp1",
                '{"nodes":{}}',
                1,
                1,
                1,
                50,
                50,
                1,
                1.0,
                10,
                10,
                7.0,
                None,
                0.2,
                0.3,
                0.25,
                None,
                None,
                None,
                "runtime_observation",
                "partial",
                json.dumps({"graph": {"graph_family": "dense"}}),
                "{}",
                1.0,
            ),
            (
                "fp1_new",
                "exp2",
                "fp1",
                '{"nodes":{}}',
                1,
                1,
                1,
                50,
                50,
                1,
                1.0,
                10,
                10,
                7.0,
                None,
                0.2,
                0.31,
                0.26,
                None,
                None,
                None,
                "candidate_grade",
                "candidate_comparable",
                json.dumps({"graph": {"graph_family": "dense"}}),
                "{}",
                2.0,
            ),
            (
                "fp2_only",
                "exp3",
                "fp2",
                '{"nodes":{}}',
                1,
                1,
                1,
                50,
                50,
                1,
                1.0,
                10,
                10,
                7.0,
                None,
                0.2,
                0.32,
                0.27,
                None,
                None,
                None,
                "exploratory",
                "partial",
                json.dumps({"graph": {"graph_family": "sparse"}}),
                "{}",
                3.0,
            ),
            (
                "ref_row",
                "exp3",
                "fp_ref",
                '{"nodes":{}}',
                1,
                1,
                1,
                50,
                50,
                1,
                1.0,
                10,
                10,
                7.0,
                None,
                0.2,
                0.32,
                0.27,
                None,
                None,
                None,
                "reference",
                "reference_comparable",
                json.dumps({"graph": {"graph_family": "reference"}}),
                "{}",
                4.0,
            ),
        ],
    )
    selected = _fetch_rows(
        conn,
        result_ids=[],
        limit=10,
        force=True,
        selection_slice="nonref_unique_fingerprints",
        balance_by_family=False,
        target_post_fields=("binding_auc",),
    )
    assert [row["result_id"] for row in selected] == ["fp2_only", "fp1_new"]
    conn.close()


def test_store_probe_results_appends_backpopulate_context(tmp_path):
    db_path = str(tmp_path / "backpopulate_ctx.db")
    nb = LabNotebook(db_path)
    exp_id = nb.start_experiment(
        "synthesis",
        {
            "data_mode": "corpus",
            "corpus_path": "/tmp/corpus.txt",
            "corpus_format": "txt",
            "corpus_text_key": "text",
            "corpus_train_fraction": 0.9,
            "corpus_val_fraction": 0.1,
            "corpus_max_chars": 1000,
            "tokenizer_mode": "tiktoken",
            "tiktoken_encoding": "cl100k_base",
        },
        "test",
    )
    rid = nb.record_program_result(
        experiment_id=exp_id,
        graph_fingerprint="fp-backpopulate",
        graph_json='{"nodes": {}}',
        model_source="graph_synthesis",
        stage0_passed=True,
        stage05_passed=True,
        stage1_passed=True,
        loss_ratio=0.5,
        wikitext_perplexity=7.3,
        screening_wikitext_metric_version="screening_wikitext_v1",
    )
    nb.flush_writes()
    store_probe_results(
        nb,
        rid,
        {"hellaswag_acc": 0.33},
        provenance_context={
            "kind": "screening_metric_backfill",
            "prefix": "s1_hellaswag_full_20260406b",
            "experiment_id": "exp123",
            "source_script": "run_s1_backpopulate",
            "post_train_target": "hellaswag",
            "device": "cuda",
            "updated_at": 1.0,
        },
    )
    row = nb.conn.execute(
        "SELECT hellaswag_acc, data_provenance_json FROM program_results WHERE result_id = ?",
        (rid,),
    ).fetchone()
    assert row["hellaswag_acc"] == 0.33
    payload = json.loads(row["data_provenance_json"])
    assert payload["last_metric_backfill"]["prefix"] == "s1_hellaswag_full_20260406b"
    assert payload["metric_backfills"][-1]["post_train_target"] == "hellaswag"
    nb.close()


def test_apply_row_updates_uses_short_lived_batch(monkeypatch):
    calls: list[tuple[str, dict, dict]] = []

    class _Batch:
        def __init__(self, parent):
            self.parent = parent

        def __enter__(self):
            self.parent.batch_entries += 1
            return None

        def __exit__(self, exc_type, exc, tb):
            self.parent.batch_exits += 1
            return False

    class _Notebook:
        def __init__(self):
            self.batch_entries = 0
            self.batch_exits = 0

        def batch(self):
            return _Batch(self)

    def _fake_store(nb, result_id, updates, write_leaderboard, provenance_context):
        calls.append((result_id, dict(updates), dict(provenance_context)))
        assert write_leaderboard is True
        assert nb.batch_entries == 1
        assert nb.batch_exits == 0

    monkeypatch.setattr(
        "research.tools.backpopulate_screening_metrics.store_probe_results",
        _fake_store,
    )
    nb = _Notebook()
    _apply_row_updates(
        nb,
        result_id="rid1",
        updates={"hellaswag_acc": 0.4},
        provenance_context={"kind": "screening_metric_backfill"},
    )
    assert calls == [
        ("rid1", {"hellaswag_acc": 0.4}, {"kind": "screening_metric_backfill"})
    ]
    assert nb.batch_entries == 1
    assert nb.batch_exits == 1


def test_print_backpopulate_summary_for_interrupt(capsys, tmp_path):
    report = tmp_path / "report.tsv"
    _print_backpopulate_summary(
        processed=7,
        total_rows=10,
        updated=3,
        updated_cuda=3,
        report_path=report,
        elapsed=12.5,
        interrupted=True,
    )
    out = capsys.readouterr().out
    assert "Interrupted after 7/10 rows" in out
    assert "updated 3 (cuda=3)" in out
    assert str(report) in out
