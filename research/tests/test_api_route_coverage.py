"""Meta-test: every registered /api/* route must have at least one test.

This is a living test that catches new routes added without tests.
"""

import os
import re

import pytest

pytestmark = pytest.mark.api


def test_all_api_routes_have_tests():
    """Every registered /api/* route must have at least one test."""
    from research.scientist.api import create_app

    app = create_app(notebook_path=":memory:")

    routes: set[str] = set()
    for rule in app.url_map.iter_rules():
        if rule.rule.startswith("/api/"):
            # Normalize: strip trailing slash, replace <param> with placeholder
            normalized = re.sub(r"<[^>]+>", "<param>", rule.rule.rstrip("/"))
            routes.add(normalized)

    # Scan test files for endpoint references
    test_dir = os.path.dirname(__file__)
    tested_fragments: set[str] = set()
    for fname in os.listdir(test_dir):
        if not fname.startswith("test_") or not fname.endswith(".py"):
            continue
        filepath = os.path.join(test_dir, fname)
        with open(filepath) as f:
            content = f.read()
        # Find all /api/... string literals in test files
        for match in re.finditer(r'["\'](/api/[^"\'?\s]+)', content):
            fragment = re.sub(r"<[^>]+>", "<param>", match.group(1).rstrip("/"))
            tested_fragments.add(fragment)

    untested: set[str] = set()
    for route in routes:
        # Check if any test references this route or a prefix of it
        if not any(
            route.startswith(frag) or frag.startswith(route)
            for frag in tested_fragments
        ):
            untested.add(route)

    # These are known untested — tracked for incremental coverage.
    # Remove from this set as tests are added.
    known_untested: set[str] = {
        "/api/actions",
        "/api/actions/<param>/approve",
        "/api/actions/<param>/dismiss",
        "/api/actions/<param>/undo",
        "/api/analytics/compression-opportunities",
        "/api/analytics/control-comparison",
        "/api/analytics/negative-results",
        "/api/analytics/regression-vs-baseline",
        "/api/analytics/strategy-backtest",
        "/api/aria/activity",
        "/api/aria/agent/spawn",
        "/api/aria/agent/status/<param>",
        "/api/aria/agent/status/<param>/summary",
        "/api/aria/autonomy",
        "/api/aria/diagnose",
        "/api/aria/tools",
        "/api/decision-packet/<param>",
        "/api/designer/commit",
        "/api/designer/compile",
        "/api/designer/components",
        "/api/designer/export/python",
        "/api/designer/import",
        "/api/designer/import/survivors",
        "/api/designer/list",
        "/api/designer/load/<param>",
        "/api/designer/run",
        "/api/designer/save",
        "/api/designer/templates",
        "/api/designer/validate",
        "/api/fingerprint/history",
        "/api/fingerprint/resolve",
        "/api/live-loss-curve",
        "/api/metrics/<param>",
        "/api/native-runner/telemetry",
        "/api/observability/monitor",
        "/api/observability/stream",
        "/api/recompute-failure-signatures",
        "/api/references",
        "/api/reproducibility-manifest/<param>",
        "/api/reproducibility-manifest/<param>/workflow",
        "/api/reset-op-stats",
        "/api/v1/<param>",
        "/api/v1/components",
        "/api/v1/import/survivors",
        "/api/v1/import/survivors/<param>",
        "/api/v1/workflows/compile",
        "/api/v1/workflows/evaluate/stream",
        "/api/v1/workflows/run",
        "/api/v1/workflows/validate",
        "/api/v1/workflows/<param>",
        "/api/worker/evaluate",
    }

    newly_untested = untested - known_untested
    if newly_untested:
        pytest.fail(
            f"{len(newly_untested)} API routes have no tests:\n"
            + "\n".join(f"  {r}" for r in sorted(newly_untested))
        )
