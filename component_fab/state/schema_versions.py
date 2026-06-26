"""Version identifiers for component_fab persisted artifacts."""

from __future__ import annotations

from typing import Any

LEDGER_GRADE_SCHEMA_VERSION = "component_fab.ledger.grade.v2"
LEDGER_PROMOTION_SCHEMA_VERSION = "component_fab.ledger.promotion.v1"
RUN_REPORT_SCHEMA_VERSION = "component_fab.run_report.v1"
PROPOSAL_SPEC_SCHEMA_VERSION = "component_fab.proposal_spec.v1"

SCHEMA_VERSIONS: dict[str, str] = {
    "ledger_grade": LEDGER_GRADE_SCHEMA_VERSION,
    "ledger_promotion": LEDGER_PROMOTION_SCHEMA_VERSION,
    "run_report": RUN_REPORT_SCHEMA_VERSION,
    "proposal_spec": PROPOSAL_SPEC_SCHEMA_VERSION,
}


def with_schema_version(record: dict[str, Any], schema_version: str) -> dict[str, Any]:
    return {"schema_version": schema_version, **record}
