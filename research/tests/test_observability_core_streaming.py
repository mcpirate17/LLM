from __future__ import annotations

from research.scientist.api_routes import _observability_core as obs


class _FakeCursor:
    def __init__(self, rows):
        self._rows = rows

    def __iter__(self):
        return iter(self._rows)

    def fetchall(self):
        raise AssertionError("_load_program_rows should not call fetchall()")


class _FakeConn:
    def __init__(self, rows):
        self._rows = rows

    def execute(self, *_args, **_kwargs):
        return _FakeCursor(self._rows)


class _FakeNotebook:
    def __init__(self, rows):
        self.conn = _FakeConn(rows)


def test_load_program_rows_streams_and_normalizes_payload():
    nb = _FakeNotebook(
        [
            {
                "graph_json": '{"nodes":[{"op_name":"gelu"}]}',
                "stage0_passed": 1,
                "stage1_passed": 0,
                "loss_ratio": 0.2,
                "error_type": "shape_mismatch",
                "failure_op": "gelu",
                "failure_details_json": '{"failure_op":"gelu"}',
            },
            {
                "graph_json": "",
                "stage0_passed": 1,
                "stage1_passed": 1,
                "loss_ratio": None,
                "error_type": None,
                "failure_op": None,
                "failure_details_json": None,
            },
        ]
    )

    rows = obs._load_program_rows(nb, "all")

    assert rows == [
        {
            "graph_json": '{"nodes":[{"op_name":"gelu"}]}',
            "stage0_passed": True,
            "stage1_passed": False,
            "loss_ratio": 0.2,
            "error_type": "shape_mismatch",
            "failure_op": "gelu",
            "failure_details_json": '{"failure_op":"gelu"}',
        }
    ]
