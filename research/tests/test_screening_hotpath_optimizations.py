from __future__ import annotations

import time
from types import SimpleNamespace

from research.orchestrator.executor import WorkerPoolOrchestrator
from research.scientist.runner.execution_experiment_phase3 import (
    _ExecutionExperimentPhase3Mixin,
)


class _DummyGraph:
    def __init__(self, fingerprint: str):
        self._fingerprint = fingerprint

    def fingerprint(self) -> str:
        return self._fingerprint


class _Rows:
    def __init__(self, rows):
        self._rows = rows

    def fetchall(self):
        return self._rows


class _Conn:
    def __init__(self, existing):
        self.existing = set(existing)
        self.sql = []

    def execute(self, sql, params=()):
        self.sql.append(sql)
        assert "WHERE graph_fingerprint IN" in sql
        return _Rows([(fp,) for fp in params if fp in self.existing])


class _Runner(_ExecutionExperimentPhase3Mixin):
    pass


def test_dedup_graph_candidates_queries_only_candidate_fingerprints():
    runner = _Runner()
    conn = _Conn(existing={"fp_existing"})
    nb = SimpleNamespace(conn=conn)
    graphs = [
        _DummyGraph("fp_existing"),
        _DummyGraph("fp_novel"),
        _DummyGraph("fp_novel"),
    ]
    results = {"funnel_counts": {}}

    kept, existing = runner._dedup_graph_candidates(
        nb=nb,
        graphs=graphs,
        grammar=object(),
        config=SimpleNamespace(model_source="generated"),
        exp_id="exp_dedup",
        results=results,
    )

    assert [g.fingerprint() for g in kept] == ["fp_novel"]
    assert existing == {"fp_existing", "fp_novel"}
    assert results["skipped_dedup"] == 2
    assert results["dedup_known_fingerprints"] == 1
    assert len(conn.sql) == 1


def test_orchestrator_preprocessor_uses_injected_compile_fn():
    compiled = object()
    compile_calls = []

    def compile_fn(layer_graphs, *, vocab_size, max_seq_len):
        compile_calls.append((layer_graphs, vocab_size, max_seq_len))
        return compiled

    def train_fn(model, config, seed, dev):
        return {
            "passed": model is compiled,
            "seed": seed,
            "device": str(dev),
        }

    orchestrator = WorkerPoolOrchestrator(
        train_fn=train_fn,
        num_workers=1,
        max_queue_size=2,
        devices=["cpu"],
        compile_fn=compile_fn,
    )
    try:
        assert orchestrator.preprocessors == []
        graph = _DummyGraph("fp")
        config = SimpleNamespace(n_layers=2, vocab_size=17, max_seq_len=9)
        orchestrator.submit(index=3, graph=graph, config=config, seed=11)
        assert orchestrator.preprocessors

        deadline = time.perf_counter() + 2.0
        results = []
        while time.perf_counter() < deadline:
            results = orchestrator.get_results()
            if results:
                break
            time.sleep(0.01)

        assert len(results) == 1
        assert results[0].s1_result["passed"] is True
        assert compile_calls == [([graph, graph], 17, 9)]
        assert orchestrator.get_telemetry()["preprocessing_avg_ms"] > 0.0
    finally:
        orchestrator.shutdown()


def test_orchestrator_skips_preprocessor_threads_for_precompiled_models():
    def train_fn(model, config, seed, dev):
        return {"passed": model == "compiled"}

    orchestrator = WorkerPoolOrchestrator(
        train_fn=train_fn,
        num_workers=1,
        max_queue_size=2,
        devices=["cpu"],
    )
    try:
        orchestrator.submit(
            index=1,
            graph=_DummyGraph("fp"),
            config=SimpleNamespace(n_layers=1, vocab_size=8, max_seq_len=4),
            seed=5,
            model="compiled",
        )

        deadline = time.perf_counter() + 2.0
        results = []
        while time.perf_counter() < deadline:
            results = orchestrator.get_results()
            if results:
                break
            time.sleep(0.01)

        assert len(results) == 1
        assert results[0].s1_result["passed"] is True
        assert orchestrator.preprocessors == []
        assert orchestrator.get_telemetry()["preprocessing_avg_ms"] == 0.0
    finally:
        orchestrator.shutdown()
