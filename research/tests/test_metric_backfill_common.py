import sqlite3

import torch


def test_sample_token_batch_and_train_step_are_shared() -> None:
    from research.tools._metric_backfill_common import (
        sample_token_batch,
        train_next_token_step,
    )

    tokens = torch.arange(64, dtype=torch.long)
    batch = sample_token_batch(tokens, 4, 8, torch.device("cpu"))
    assert batch.shape == (4, 9)

    model = torch.nn.Sequential(torch.nn.Embedding(64, 8), torch.nn.Linear(8, 64))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss = train_next_token_step(model, opt, batch)
    assert loss > 0.0


def test_update_graph_runs_columns_updates_fingerprint_rows() -> None:
    from research.tools._metric_backfill_common import update_graph_runs_columns

    conn = sqlite3.connect(":memory:")
    conn.execute(
        "CREATE TABLE graph_runs (graph_fingerprint TEXT, metric_a REAL, metric_b TEXT)"
    )
    conn.executemany(
        "INSERT INTO graph_runs VALUES (?, ?, ?)",
        [("fp1", None, None), ("fp1", None, None), ("fp2", None, None)],
    )

    rowcount = update_graph_runs_columns(
        conn,
        "fp1",
        {"metric_a": 1.5, "metric_b": "ok"},
        ("metric_a", "metric_b"),
    )

    assert rowcount == 2
    rows = conn.execute(
        "SELECT metric_a, metric_b FROM graph_runs WHERE graph_fingerprint = 'fp1'"
    ).fetchall()
    assert rows == [(1.5, "ok"), (1.5, "ok")]
