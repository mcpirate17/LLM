#!/usr/bin/env python3
"""
aria_audit.py — Aria Architecture Search Audit Script
======================================================
Run from the workspace root:
    python aria_audit.py [--root /home/tim/Projects/LLM] [--out audit_report]

Produces:
  audit_report.json   — machine-readable findings for multi-agent dispatch
  audit_report.md     — human-readable executive summary

Audit modules (each emits structured AuditFinding objects):
  1. TEMPLATE_STRUCTURE   — per-template sanity: norms, residuals, lanes
  2. SLOT_COMPLETENESS    — multi-lane templates with under-processed lanes
  3. MATH_OPS_RULES       — must_precede / must_follow violations
  4. COMPONENT_COVERAGE   — ops never/rarely used in any template or motif
  5. CATALOG_METADATA     — empty ML-feedback fields in catalog
  6. HISTORICAL_INSIGHT   — weight sources: are they DB-driven or hardcoded?
  7. SEARCH_DIVERSITY     — collision / dead-code traps, exploration gaps
  8. COMMON_SENSE         — missing obvious architectural patterns
  9. OP_ROLE_COVERAGE     — every primitive has an OpRole in op_roles.py
  10. CONTEXT_RULE_COVERAGE — every primitive has a ContextRule, forbidden pairs symmetric
  11. ROUTING_SYSTEM       — true routing ops integrated, aliases resolve, no stale names
  12. VALIDATOR_WIRING     — template_rules.py and graph_validator.py are called
  13. MOTIF_COMPOSITION    — no forbidden op sequences within motifs

Each finding carries:
  severity  : CRITICAL | HIGH | MEDIUM | LOW
  agent     : aria-architect | aria-scientist | aria-kernel | aria-bridge
  category  : module tag
  location  : file:line or template/op name
  finding   : what is wrong
  suggestion: concrete fix or investigation step
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sqlite3
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path

# ── Finding dataclass ─────────────────────────────────────────────────────────

SEVERITIES = ("CRITICAL", "HIGH", "MEDIUM", "LOW")
_SEVERITY_ORDER = {s: i for i, s in enumerate(SEVERITIES)}
AGENTS = ("aria-architect", "aria-scientist", "aria-kernel", "aria-bridge")


@dataclass
class AuditFinding:
    severity: str  # CRITICAL | HIGH | MEDIUM | LOW
    agent: str  # which Claude Code agent should fix this
    category: str  # audit module name
    location: str  # template name, op name, or file:line
    finding: str  # concise description of the problem
    suggestion: str  # what to do about it
    evidence: dict = field(default_factory=dict)  # supporting data

    def __post_init__(self):
        assert self.severity in SEVERITIES, f"Bad severity: {self.severity}"
        assert self.agent in AGENTS, f"Bad agent: {self.agent}"


# ── Collectors ────────────────────────────────────────────────────────────────

findings: list[AuditFinding] = []


def emit(severity, agent, category, location, finding, suggestion, **evidence):
    findings.append(
        AuditFinding(
            severity=severity,
            agent=agent,
            category=category,
            location=location,
            finding=finding,
            suggestion=suggestion,
            evidence=evidence,
        )
    )


# ── Path discovery ────────────────────────────────────────────────────────────


def discover_paths(root: Path) -> dict[str, Path]:
    paths = {
        "templates_py": root / "research/synthesis/templates.py",
        "motifs_py": root / "research/synthesis/motifs.py",
        "context_rules_py": root / "research/synthesis/context_rules.py",
        "primitives_py": root / "research/synthesis/primitives.py",
        "grammar_py": root / "research/synthesis/grammar.py",
        "op_roles_py": root / "research/synthesis/op_roles.py",
        "true_routing_py": root / "research/synthesis/true_routing_ops.py",
        "template_rules_py": root / "research/synthesis/template_rules.py",
        "graph_validator_py": root / "research/synthesis/graph_validator.py",
        "notebook_db": root / "research/lab_notebook.db",
        "component_catalog": root / "research/profiling/component_catalog.csv",
        "component_mapping": root / "aria_designer/runtime/component_mapping.yaml",
        "runner": root / "research/scientist/runner/__init__.py",
        "judgment": root / "research/scientist/judgment.py",
        "scheduler": root / "research/search/scheduler.py",
        "compiler_py": root / "research/synthesis/compiler.py",
    }
    # Also accept the uploaded catalog as fallback
    for key, p in list(paths.items()):
        if not p.exists():
            # try sibling dirs
            alt = root / p.name
            if alt.exists():
                paths[key] = alt
    return paths


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE 1 — TEMPLATE STRUCTURE
# Checks every template for: pre-norm, post-norm, residual, lane processing
# ═══════════════════════════════════════════════════════════════════════════════

# Structural signals we look for in template source
_NORM_CALLS = re.compile(
    r'MOTIF_CLASS_NORM|add_op\(\s*["\']rmsnorm|add_op\(\s*["\']layernorm'
)
_RESIDUAL_ADD = re.compile(r'add_op\(\s*["\']add["\'].*\binput_id\b')
_SPLIT3_CALL = re.compile(r'add_op\(\s*["\']split3')
_SPLIT2_CALL = re.compile(r'add_op\(\s*["\']split2')
_ROUTE_LANES = re.compile(r'add_op\(\s*["\']route_lanes')
_MOTIF_SLOT = re.compile(r"_pick_compatible_motif(?:_from_classes)?\(")
_FFN_SLOT = re.compile(r"_FFN_CLASSES|MOTIF_CLASS_FFN")
_MIXER_SLOT = re.compile(r"_MIXER_CLASSES|MOTIF_CLASS_ATTENTION|MOTIF_CLASS_SSM")
_NUM_RISKY_OPS = re.compile(
    r'add_op\(\s*["\'](div_safe|cumprod_safe|hyp_distance|cosine_similarity|'
    r"fixed_point_iter|rwkv_time_mixing|selective_scan|exp_map|log_map|"
    r'hyp_linear|hyperbolic_norm)["\']'
)
_RETURN_FALLBACK = re.compile(r"return tpl_residual_block")


def _extract_template_bodies(source: str) -> dict[str, str]:
    """Return {template_name: function_body_source} by splitting on def tpl_."""
    bodies: dict[str, str] = {}
    parts = re.split(r"(?=^def tpl_)", source, flags=re.MULTILINE)
    for part in parts:
        m = re.match(r"^def (tpl_\w+)", part)
        if m:
            fname = m.group(1)
            key = fname[4:]  # strip tpl_
            bodies[key] = part
    return bodies


def _extract_registry_weights(source: str) -> dict[str, float]:
    """Parse DEFAULT_TEMPLATE_WEIGHTS dict from source."""
    m = re.search(
        r"DEFAULT_TEMPLATE_WEIGHTS\s*:\s*Dict\[.*?\]\s*=\s*\{(.*?)\}", source, re.DOTALL
    )
    if not m:
        return {}
    weights: dict[str, float] = {}
    for line in m.group(1).splitlines():
        wm = re.match(r'\s*["\'](\w+)["\']\s*:\s*([\d.]+)', line)
        if wm:
            weights[wm.group(1)] = float(wm.group(2))
    return weights


def _extract_templates_dict(source: str) -> list[str]:
    """Return template names listed in TEMPLATES dict."""
    m = re.search(
        r"^TEMPLATES\s*:\s*Dict.*?=\s*\{(.*?)\}", source, re.DOTALL | re.MULTILINE
    )
    if not m:
        return []
    return re.findall(r'["\'](\w+)["\'](?=\s*:)', m.group(1))


def audit_template_structure(templates_source: str) -> None:
    bodies = _extract_template_bodies(templates_source)
    registry = _extract_templates_dict(templates_source)
    weights = _extract_registry_weights(templates_source)

    for name, body in bodies.items():
        # ── 1a. No normalization at all ──────────────────────────────────────
        has_norm = bool(_NORM_CALLS.search(body))
        if not has_norm:
            emit(
                "HIGH",
                "aria-architect",
                "TEMPLATE_STRUCTURE",
                f"tpl_{name}",
                f"Template '{name}' contains no normalization (no MOTIF_CLASS_NORM pick "
                f"and no direct rmsnorm/layernorm add_op). Unnormalized inputs cause "
                f"activation drift and training instability, especially in deep stacks.",
                "Add a MOTIF_CLASS_NORM slot at the start. At minimum: "
                "norm = _pick_compatible_motif(graph, input_id, rng, MOTIF_CLASS_NORM, weights); "
                "normed = _instantiate_motif(...) if norm else input_id",
                has_norm=False,
            )

        # ── 1b. No residual connection ────────────────────────────────────────
        has_residual = bool(_RESIDUAL_ADD.search(body))
        if not has_residual:
            emit(
                "HIGH",
                "aria-architect",
                "TEMPLATE_STRUCTURE",
                f"tpl_{name}",
                f"Template '{name}' has no skip/residual connection back to input_id. "
                f"Without residual, gradient flow degrades and the template cannot "
                f"participate in depth-stacked graphs without vanishing.",
                "Wrap output with: try: return graph.add_op('add', [input_id, processed]) "
                "except ValueError: return processed",
                has_residual=False,
            )

        # ── 1c. Numerically risky ops without preceding norm ──────────────────
        risky_match = _NUM_RISKY_OPS.search(body)
        if risky_match and not has_norm:
            emit(
                "CRITICAL",
                "aria-architect",
                "TEMPLATE_STRUCTURE",
                f"tpl_{name}",
                f"Template '{name}' calls numerically risky op "
                f"'{risky_match.group(1)}' but has no normalization. "
                f"This is a known source of NaN/Inf explosions (see padic_attention "
                f"memory explosion post-mortem).",
                f"Add rmsnorm/layernorm before '{risky_match.group(1)}'. "
                f"Check MATH_SPACE_RULES['must_precede'] for this op.",
                risky_op=risky_match.group(1),
            )

        # ── 1d. Template defined but missing from TEMPLATES registry ──────────
        if name not in registry and name not in ("tropical_center_block",):
            emit(
                "MEDIUM",
                "aria-architect",
                "TEMPLATE_STRUCTURE",
                f"tpl_{name}",
                f"Template function tpl_{name} is defined but NOT in the TEMPLATES "
                f"registry dict and NOT in DEFAULT_TEMPLATE_WEIGHTS. It is dead code "
                f"and will never be sampled.",
                f"Either delete tpl_{name} or add it to TEMPLATES and "
                f"DEFAULT_TEMPLATE_WEIGHTS with an appropriate weight (suggest 2.0).",
                in_registry=False,
            )

        # ── 1e. Template weight == 0 (effectively dead) ───────────────────────
        w = weights.get(name, -1.0)
        if w == 0.0:
            emit(
                "HIGH",
                "aria-architect",
                "TEMPLATE_STRUCTURE",
                f"tpl_{name}",
                f"Template '{name}' has weight 0.0 in DEFAULT_TEMPLATE_WEIGHTS. "
                f"It will never be sampled by weighted random selection.",
                "Remove it or set a positive weight. Even 0.5 keeps it reachable.",
                weight=0.0,
            )

    # ── 1f. Templates in registry but no function defined ─────────────────────
    defined_names = set(bodies.keys())
    for reg_name in registry:
        if reg_name not in defined_names:
            emit(
                "CRITICAL",
                "aria-architect",
                "TEMPLATE_STRUCTURE",
                f"tpl_{reg_name}",
                f"Template '{reg_name}' is listed in TEMPLATES registry but has no "
                f"corresponding tpl_{reg_name} function. Will crash at runtime.",
                f"Define tpl_{reg_name} or remove it from TEMPLATES dict.",
                has_function=False,
            )

    # ── 1g. Templates with hardcoded uniform weights ──────────────────────────
    # If ALL weights are round numbers (integers or .0/.5), they're hand-tuned
    all_hardcoded = all(
        (w * 2) == int(w * 2)  # multiples of 0.5
        for w in weights.values()
    )
    if all_hardcoded and len(weights) > 5:
        emit(
            "HIGH",
            "aria-scientist",
            "HISTORICAL_INSIGHT",
            "DEFAULT_TEMPLATE_WEIGHTS",
            f"All {len(weights)} template weights are round numbers "
            f"(hardcoded guesses). They are NOT derived from historical evaluation "
            f"results. The search is flying blind — it does not prefer templates that "
            f"historically produce lower-loss graphs.",
            "Implement a weight updater that reads avg_loss per template from "
            "lab_notebook.db and updates DEFAULT_TEMPLATE_WEIGHTS proportionally. "
            "Example: weight[t] ∝ exp(-k * mean_loss[t]). Run nightly or after "
            "each batch of 50 evaluations. See Module 6 for full spec.",
            hardcoded_count=len(weights),
        )


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE 2 — SLOT COMPLETENESS
# Three-lane, two-lane, and split templates must process ALL lanes
# ═══════════════════════════════════════════════════════════════════════════════


def audit_slot_completeness(templates_source: str) -> None:
    bodies = _extract_template_bodies(templates_source)

    for name, body in bodies.items():
        # ── 2a. tpl_three_way_split: part2 gets no processing ────────────────
        if "three_way_split" in name or "split3" in name:
            # part2 is passed directly to concat without any motif
            if re.search(r"concat.*part2|part2.*concat", body) and not re.search(
                r"m2\s*=|motif.*part2|process.*part2", body
            ):
                emit(
                    "HIGH",
                    "aria-architect",
                    "SLOT_COMPLETENESS",
                    f"tpl_{name}",
                    f"In tpl_{name}, part2 (the third lane) is passed directly into "
                    f"concat WITHOUT any motif processing. Lanes 0 and 1 each get a "
                    f"NORM motif, but lane 2 is raw — its gradient signal is weaker "
                    f"and the lanes are not symmetrically treated. This kills the "
                    f"research value of 3-lane exploration.",
                    "Add: m2 = _pick_compatible_motif_from_classes(graph, part2, rng, "
                    "_FFN_CLASSES, weights) — give lane 2 an FFN or channel mixer slot "
                    "so all three lanes carry independent learned transforms. "
                    "Lane 0: MIXER (sequence), Lane 1: FFN (channel), Lane 2: GATE or MoE.",
                    unprocessed_lane=2,
                )

        # ── 2b. tpl_three_way_split: all lanes only get NORM motif ───────────
        if "three_way_split" in name:
            norm_only = re.findall(r"_pick_compatible_motif\(.*?MOTIF_CLASS_NORM", body)
            if len(norm_only) >= 2 and not re.search(
                r"_FFN_CLASSES|_MIXER_CLASSES", body
            ):
                emit(
                    "HIGH",
                    "aria-architect",
                    "SLOT_COMPLETENESS",
                    f"tpl_{name}",
                    f"tpl_{name} assigns only NORM motifs to lanes. A 3-lane split "
                    f"exists to provide DIFFERENT processing per lane (semantic diversity). "
                    f"Assigning norm to both processed lanes wastes the architectural "
                    f"advantage — they become functionally identical.",
                    "Assign DISTINCT motif classes per lane: "
                    "Lane 0 → MOTIF_CLASS_ATTENTION or MOTIF_CLASS_SSM (sequence mixing), "
                    "Lane 1 → _FFN_CLASSES (channel mixing), "
                    "Lane 2 → MOTIF_CLASS_MOE or MOTIF_CLASS_GATE (sparse routing). "
                    "This is the architectural insight of mixture-of-experts.",
                    lane_classes_distinct=False,
                )

        # ── 2c. Parallel split: neither lane gets normalization ───────────────
        if "parallel_split" in name or "hybrid_parallel" in name:
            split_present = bool(_SPLIT2_CALL.search(body))
            per_lane_norm = bool(
                re.search(r"split_[ab].*?MOTIF_CLASS_NORM", body, re.DOTALL)
            )
            if split_present and not per_lane_norm:
                emit(
                    "MEDIUM",
                    "aria-architect",
                    "SLOT_COMPLETENESS",
                    f"tpl_{name}",
                    f"tpl_{name} splits into 2 lanes but neither lane gets its own "
                    f"normalization before its motif. The pre-split normalization is "
                    f"missing — each lane processes raw split tensors.",
                    "After each split_a / split_b, apply per-lane norm before the motif: "
                    "norm_a = _pick_compatible_motif(graph, split_a, rng, MOTIF_CLASS_NORM); "
                    "split_a_normed = _instantiate_motif(graph, split_a, norm_a, rng) if norm_a else split_a",
                    per_lane_norm=False,
                )

        # ── 2d. Route_lanes with no per-lane downstream motif ────────────────
        if _ROUTE_LANES.search(body):
            post_route_motifs = re.findall(r"_pick_compatible_motif", body)
            if len(post_route_motifs) < 2:
                emit(
                    "MEDIUM",
                    "aria-architect",
                    "SLOT_COMPLETENESS",
                    f"tpl_{name}",
                    f"tpl_{name} calls route_lanes but only picks {len(post_route_motifs)} "
                    f"motif(s) post-routing. Multi-lane routing without per-lane learned "
                    f"transforms degenerates to a bottleneck, not a routing system.",
                    "After route_lanes, pick a distinct motif for each lane's output. "
                    "At minimum: lane_fast→_FFN_CLASSES, lane_hard→_MIXER_CLASSES.",
                    post_route_motif_count=len(post_route_motifs),
                )

    # ── 2e. Global: how many templates have 0 motif slots at all ─────────────
    no_slot_templates = []
    for name, body in bodies.items():
        if not _MOTIF_SLOT.search(body):
            no_slot_templates.append(name)
    if no_slot_templates:
        emit(
            "MEDIUM",
            "aria-architect",
            "SLOT_COMPLETENESS",
            "templates_py",
            f"{len(no_slot_templates)} templates have NO motif slots — they use "
            f"hardcoded op sequences with no motif substitution. These produce "
            f"zero architectural variety and are research dead zones: "
            f"{', '.join(no_slot_templates[:8])}{'...' if len(no_slot_templates) > 8 else ''}",
            "Add at least one _pick_compatible_motif_from_classes slot to every template. "
            "A template with zero randomness contributes 1 graph to the search space "
            "instead of hundreds.",
            zero_slot_templates=no_slot_templates,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE 3 — MATH OPS RULES
# Validate must_precede / must_follow logic and identify missing rules
# ═══════════════════════════════════════════════════════════════════════════════


# Known rules from context (would ideally be read from context_rules.py / motifs.py)
def _parse_math_space_rules(motifs_source: str | None) -> dict[str, set[str]]:
    """Parse MATH_SPACE_RULES from motifs.py source to get ops with must_precede."""
    if not motifs_source:
        return {}
    rules: dict[str, set[str]] = {}
    # Find all top-level keys and their must_precede sets
    # Pattern: "op_name": { ... "must_precede": {set} ... }
    # Also match ops that have must_follow or must_follow_with (they still have rules)
    for m in re.finditer(
        r'"(\w+)":\s*\{[^}]*"must_precede":\s*\{([^}]*)\}',
        motifs_source,
    ):
        op = m.group(1)
        preds = set(re.findall(r'"(\w+)"', m.group(2)))
        rules[op] = preds
    # Also capture ops with only must_follow / must_follow_with (they have rules, just not must_precede)
    for m in re.finditer(r'"(\w+)":\s*\{[^}]*"must_follow', motifs_source):
        op = m.group(1)
        if op not in rules:
            rules[op] = set()  # has a rule entry, just no must_precede
    return rules


# Fallback hardcoded rules — only used if motifs.py can't be parsed
_FALLBACK_MUST_PRECEDE: dict[str, list[str]] = {
    "tropical_attention": ["rmsnorm", "layernorm"],
    "tropical_gate": ["rmsnorm", "layernorm", "tropical_attention"],
    "tropical_center": ["tropical_attention", "tropical_gate"],
    "hyp_linear": ["exp_map", "rmsnorm", "layernorm"],
    "hyp_distance": ["exp_map"],
    "exp_map": ["rmsnorm", "layernorm", "linear_proj"],
    "log_map": ["exp_map", "poincare_add", "hyp_linear"],
    "ultrametric_attention": ["rmsnorm", "layernorm"],
    "clifford_attention": ["rmsnorm", "layernorm"],
    "fixed_point_iter": ["rmsnorm", "layernorm"],
    "padic_expand": ["rmsnorm", "layernorm"],
    "padic_residual": ["rmsnorm", "layernorm"],
    "rwkv_time_mixing": ["rmsnorm", "layernorm"],
    "selective_scan": ["rmsnorm", "layernorm"],
    "state_space": ["rmsnorm", "layernorm"],
}

KNOWN_MUST_FOLLOW: dict[str, list[str]] = {
    "tropical_center": ["linear_proj"],
    "exp_map": ["hyp_linear", "poincare_add", "hyp_distance"],
    "grade_select": ["linear_proj"],
    "hyp_distance": ["linear_proj_up", "linear_proj"],
    "cosine_similarity": ["linear_proj_up", "linear_proj"],
    "entropy_score": ["mul", "linear_proj_up"],
    "split2": ["linear_proj", "rmsnorm", "layernorm"],
    "split3": ["linear_proj", "rmsnorm", "layernorm"],
    "cumprod_safe": ["mul"],  # result must gate a value, not go direct
    "token_merge": ["swiglu_mlp", "conv1d_seq", "gelu", "silu"],  # not state_space/SSM
}

# Ops that are FORBIDDEN after certain ops
FORBIDDEN_AFTER: dict[str, list[str]] = {
    "token_merge": [
        "state_space",
        "selective_scan",
        "rwkv_time_mixing",
        "linear_attention",
    ],
    "entropy_score": ["softmax_last", "softmax_attention"],  # already normalized
    "sigmoid": ["softmax_last"],  # redundant double-normalization
}

# Ops that MUST NOT appear adjacent (in sequence)
ADJACENT_FORBIDDEN: list[tuple[str, str]] = [
    ("rmsnorm", "layernorm"),  # two norms in a row is wasteful
    ("layernorm", "rmsnorm"),
    ("gelu", "silu"),  # two similar activations back-to-back
    ("silu", "gelu"),
    ("sigmoid", "sigmoid"),  # same op twice without projection
    ("relu", "relu"),
    ("tanh", "tanh"),
]


def audit_math_ops_rules(templates_source: str, motifs_source: str | None) -> None:
    bodies = _extract_template_bodies(templates_source)

    # Parse must_precede rules from actual motifs.py source (live, not hardcoded)
    parsed_rules = _parse_math_space_rules(motifs_source)
    must_precede: dict[str, list[str]] = {}
    if parsed_rules:
        must_precede = {op: list(preds) for op, preds in parsed_rules.items() if preds}
    else:
        must_precede = dict(_FALLBACK_MUST_PRECEDE)

    # ── 3a. Check templates for must_precede violations ──────────────────────
    for name, body in bodies.items():
        for risky_op, required_preds in must_precede.items():
            if f'"{risky_op}"' in body or f"'{risky_op}'" in body:
                # Check if any required predecessor appears before it in the template body
                # This is approximate (source order ≈ execution order for linear templates)
                risky_pos = body.find(f'"{risky_op}"')
                if risky_pos == -1:
                    risky_pos = body.find(f"'{risky_op}'")
                has_pred = any(
                    body.find(f'"{pred}"') < risky_pos
                    or body.find(f"'{pred}'") < risky_pos
                    or "MOTIF_CLASS_NORM" in body[:risky_pos]
                    for pred in required_preds
                )
                if not has_pred:
                    emit(
                        "HIGH",
                        "aria-architect",
                        "MATH_OPS_RULES",
                        f"tpl_{name}",
                        f"Template '{name}' uses '{risky_op}' but none of its required "
                        f"predecessors ({required_preds}) appear before it in the template. "
                        f"The auto-fix in _instantiate_motif (insert rmsnorm) only covers "
                        f"motif steps, NOT template-level add_op calls.",
                        f"Add explicit norm before '{risky_op}': "
                        f"normed = graph.add_op('rmsnorm', [input_node]). "
                        f"Do not rely on _instantiate_motif's auto-fix for template-level ops.",
                        risky_op=risky_op,
                        required_predecessors=required_preds,
                    )

    # ── 3b. Missing must_follow enforcement ───────────────────────────────────
    # After token_merge, SSM/recurrent ops should be blocked
    # Use quoted op names to avoid matching comments/docstrings
    for name, body in bodies.items():
        has_merge = (
            '"token_merge"' in body
            or "'token_merge'" in body
            or '"adjacent_token_merge"' in body
            or "'adjacent_token_merge'" in body
        )
        if has_merge:
            bad_ops = [
                op
                for op in FORBIDDEN_AFTER.get("token_merge", [])
                if f'"{op}"' in body or f"'{op}'" in body
            ]
            if bad_ops:
                emit(
                    "CRITICAL",
                    "aria-architect",
                    "MATH_OPS_RULES",
                    f"tpl_{name}",
                    f"Template '{name}' uses token_merge then allows forbidden ops: "
                    f"{bad_ops}. token_merge changes sequence length, which breaks "
                    f"SSM/recurrent ops that assume fixed sequence dimension. "
                    f"This is a 100% S0 failure rate pattern.",
                    f"Remove {bad_ops} from the motif classes available after token_merge. "
                    f"Only _FFN_CLASSES are safe post-merge (conv1d_seq, swiglu_mlp, gelu).",
                    forbidden_ops=bad_ops,
                )

    # ── 3c. Missing rules for newly added exotic ops ──────────────────────────
    # Only emit for ops that are actually missing from MATH_SPACE_RULES
    all_ruled_ops = set(parsed_rules.keys()) if parsed_rules else set()
    candidate_missing = [
        (
            "lif_neuron",
            "LIF requires clipped membrane potential — needs sigmoid or tanh before",
        ),
        (
            "spike_rate_code",
            "rate coding needs bounded input [0,1] — sigmoid/tanh required before",
        ),
        (
            "stdp_attention",
            "STDP uses causal temporal decay — causal_mask must precede",
        ),
        (
            "padic_gate",
            "p-adic valuation is undefined at zero — needs abs+clamp before",
        ),
        (
            "chebyshev_spectral_mix",
            "Chebyshev polynomials blow up for |x|>1 — needs tanh/norm",
        ),
        (
            "kronecker_linear",
            "Kronecker product amplifies variance — needs rmsnorm before",
        ),
        (
            "n_way_sparse_router",
            "N-way routing with no norm produces uniform routing collapse",
        ),
        (
            "integral_kernel",
            "Integral kernel sums unbounded — norm before and after required",
        ),
        ("basis_expansion", "Sinusoidal basis explodes without bounded input range"),
        (
            "rotor_transform",
            "Clifford rotor requires unit-norm input — multi_head_mix or norm",
        ),
    ]
    for op, reason in candidate_missing:
        if op not in all_ruled_ops:
            emit(
                "MEDIUM",
                "aria-architect",
                "MATH_OPS_RULES",
                f"op:{op}",
                f"Op '{op}' has no enforced must_precede rule in MATH_SPACE_RULES. {reason}",
                f"Add to MATH_SPACE_RULES in motifs.py: "
                f'"{op}": {{"must_precede": ["rmsnorm", "layernorm"], "reason": "{reason}"}}',
                op=op,
            )

    # ── 3d. Op sequencing: same op twice without projection ──────────────────
    for op_a, op_b in ADJACENT_FORBIDDEN:
        pattern = rf'["\']({op_a})["\'].*?["\']({op_b})["\']'
        for name, body in bodies.items():
            if re.search(pattern, body, re.DOTALL):
                emit(
                    "MEDIUM",
                    "aria-architect",
                    "MATH_OPS_RULES",
                    f"tpl_{name}",
                    f"Template '{name}' places '{op_a}' immediately before '{op_b}'. "
                    f"These ops are functionally redundant in sequence and waste parameters.",
                    f"Interpose a linear_proj or motif slot between '{op_a}' and '{op_b}', "
                    f"or remove one. Alternatively, add this pair to ADJACENT_FORBIDDEN in "
                    f"context_rules.py to prevent motifs from generating this pattern.",
                    op_a=op_a,
                    op_b=op_b,
                )

    # ── 3e. No whole-template normalization rules ─────────────────────────────
    # Only emit if template_rules.py doesn't exist (Module 12 handles wiring)
    try:
        from pathlib import Path as _P

        _tpl_rules_exists = (
            _P(__file__).parent / "research/synthesis/template_rules.py"
        ).exists()
    except Exception:
        _tpl_rules_exists = False
    if not _tpl_rules_exists:
        emit(
            "HIGH",
            "aria-scientist",
            "MATH_OPS_RULES",
            "context_rules.py",
            "There are no WHOLE-TEMPLATE structural rules — only per-op must_precede/follow. "
            "Missing template-level invariants: "
            "(1) Every template MUST start with a norm. "
            "(2) Every template MUST end with a residual add. "
            "(3) A 3-lane split MUST assign distinct motif classes per lane. "
            "(4) After a bottleneck (D→D/4), no parametric op with out_dim=D may appear until up-projection. "
            "Without these, graph synthesis is only locally valid, not globally architecturally sound.",
            "Create research/synthesis/template_rules.py with a validate_template_graph() function "
            "that checks these invariants POST-SYNTHESIS before a graph enters eval. "
            "Invalid graphs get rejected early instead of wasting eval compute.",
            missing_invariants=[
                "start_norm",
                "end_residual",
                "lane_diversity",
                "bottleneck_dim",
            ],
        )


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE 4 — COMPONENT COVERAGE
# Ops that are never / rarely used across all templates
# ═══════════════════════════════════════════════════════════════════════════════


def audit_component_coverage(
    templates_source: str,
    catalog_rows: list[dict],
    motifs_source: str | None = None,
) -> None:
    # Count op appearances in templates source + motifs source
    combined_source = templates_source
    if motifs_source:
        combined_source += "\n" + motifs_source

    op_counts: Counter = Counter()
    for row in catalog_rows:
        op = row.get("name", "")
        if not op:
            continue
        count = combined_source.count(f'"{op}"') + combined_source.count(f"'{op}'")
        op_counts[op] = count

    # ── 4a. Ops with zero template+motif coverage ────────────────────────────
    zero_primitives = []
    for row in catalog_rows:
        op = row.get("name", "")
        if (
            op
            and op_counts.get(op, 0) == 0
            and row.get("component_type") == "primitive"
        ):
            zero_primitives.append(op)

    if zero_primitives:
        emit(
            "HIGH",
            "aria-architect",
            "COMPONENT_COVERAGE",
            "templates.py",
            f"{len(zero_primitives)} primitive ops have ZERO template coverage — they "
            f"will never appear in any synthesized graph unless hand-picked. "
            f"Each uncovered op represents a dead research axis. "
            f"Uncovered: {', '.join(sorted(zero_primitives)[:15])}...",
            "For each uncovered primitive, create a minimal dedicated template "
            "(or motif) that forces its selection. Follow the Phase 3 pattern: "
            "norm → [missing_op] → [FFN motif] → residual. "
            "One template per op is sufficient to make it reachable.",
            zero_coverage_count=len(zero_primitives),
            ops=sorted(zero_primitives),
        )

    # ── 4b. Ops used only 1-2 times (underexplored) ──────────────────────────
    rare_ops = [(op, cnt) for op, cnt in op_counts.items() if 0 < cnt <= 2]
    if rare_ops:
        emit(
            "MEDIUM",
            "aria-architect",
            "COMPONENT_COVERAGE",
            "templates.py",
            f"{len(rare_ops)} ops appear only 1-2 times across all templates. "
            f"With only 1-2 template appearances, these ops get sampled at <0.1% rate "
            f"in large runs. They are effectively invisible to the search: "
            f"{', '.join(op for op, _ in sorted(rare_ops, key=lambda x: x[1])[:10])}",
            "Add these ops to more templates as optional motif slots OR increase "
            "their motif_lift in the catalog so weighted sampling selects them more often.",
            rare_ops=dict(rare_ops),
        )

    # ── 4c. Category-level coverage gaps ─────────────────────────────────────
    by_category: dict[str, list[str]] = defaultdict(list)
    for row in catalog_rows:
        cat = row.get("category", "unknown")
        op = row.get("name", "")
        if op and row.get("component_type") == "primitive":
            by_category[cat].append(op)

    for cat, ops in by_category.items():
        covered = [op for op in ops if op_counts.get(op, 0) > 0]
        pct = len(covered) / len(ops) * 100 if ops else 100
        if pct < 50 and len(ops) >= 3:
            emit(
                "HIGH",
                "aria-architect",
                "COMPONENT_COVERAGE",
                f"category:{cat}",
                f"Category '{cat}' has only {pct:.0f}% template coverage "
                f"({len(covered)}/{len(ops)} ops). "
                f"Uncovered: {', '.join(op for op in ops if op_counts.get(op, 0) == 0)}",
                f"Create or extend a template specifically for '{cat}' ops. "
                f"The math_space and functional categories appear severely underrepresented.",
                coverage_pct=pct,
                covered=covered,
                uncovered=[op for op in ops if op_counts.get(op, 0) == 0],
            )

    # ── 4d. Estimate search space size ───────────────────────────────────────
    n_templates = len(_extract_templates_dict(templates_source))
    n_primitives = sum(
        1 for r in catalog_rows if r.get("component_type") == "primitive"
    )
    # Very rough lower bound: templates × ops^(avg_slots) × slot_classes
    n_motif_slots_per_template_avg = 2  # conservative
    n_motif_options = max(n_primitives // 4, 10)
    search_space_estimate = n_templates * (
        n_motif_options**n_motif_slots_per_template_avg
    )

    emit(
        "LOW",
        "aria-scientist",
        "COMPONENT_COVERAGE",
        "search_space",
        f"Estimated search space: ~{search_space_estimate:,.0f} distinct graphs "
        f"({n_templates} templates × ~{n_motif_options}^{n_motif_slots_per_template_avg} motif combos). "
        f"With {n_primitives} primitives, this is a large but NOT infinite space. "
        f"Key bottleneck: sampling is biased heavily toward the top 10 templates "
        f"by weight, exploring <5% of actual space.",
        "Implement adaptive template cooling: reduce weight of templates with many "
        "evaluated graphs; boost underexplored templates. "
        "Target: each template gets ≥50 evaluations before weight adjustment.",
        estimated_space=search_space_estimate,
        n_templates=n_templates,
        n_primitives=n_primitives,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE 5 — CATALOG METADATA
# Empty ML-feedback fields in component_catalog.csv
# ═══════════════════════════════════════════════════════════════════════════════

ML_FEEDBACK_FIELDS = [
    "motif_class",
    "motif_support",
    "motif_lift",
    "motif_avg_loss",
    "template_weight",
    "paradigm",
]


def audit_catalog_metadata(
    catalog_rows: list[dict], motifs_source: str | None = None
) -> None:
    total = len(catalog_rows)
    empty_by_field: dict[str, list[str]] = defaultdict(list)

    for row in catalog_rows:
        op = row.get("name", "?")
        for field_name in ML_FEEDBACK_FIELDS:
            val = row.get(field_name, "")
            if val == "" or val is None:
                empty_by_field[field_name].append(op)

    for field_name, empty_ops in empty_by_field.items():
        pct = len(empty_ops) / total * 100
        if pct > 30:
            emit(
                "HIGH",
                "aria-scientist",
                "CATALOG_METADATA",
                f"component_catalog.csv:{field_name}",
                f"Field '{field_name}' is EMPTY for {len(empty_ops)}/{total} ops ({pct:.0f}%). "
                f"This field is a feedback signal for ML-driven template/motif weighting. "
                f"Empty = the search has NO historical learning for these ops.",
                f"Populate '{field_name}' by querying lab_notebook.db: "
                f"SELECT op_name, COUNT(*), AVG(val_loss) FROM op_usage_stats GROUP BY op_name. "
                f"Write results back to catalog with: "
                f"UPDATE component_catalog SET {field_name}=? WHERE name=?",
                empty_count=len(empty_ops),
                empty_pct=round(pct, 1),
            )

    # ── 5a. byte_safe field: false entries that might be used unsafely ────────
    byte_unsafe = [
        row["name"]
        for row in catalog_rows
        if row.get("byte_safe", "true").lower() == "false"
    ]
    if byte_unsafe:
        emit(
            "MEDIUM",
            "aria-kernel",
            "CATALOG_METADATA",
            "component_catalog.csv",
            f"{len(byte_unsafe)} ops are marked byte_safe=false: {byte_unsafe}. "
            f"These may have tensor layout assumptions that break in batched graphs. "
            f"Are they blocked from selection in byte-safe execution contexts?",
            "Verify that context_rules.py or the grammar enforces byte_safe=true "
            "when running in quantized or native execution modes.",
            byte_unsafe_ops=byte_unsafe,
        )

    # ── 5b. Numerically risky ops with no catalog flag ────────────────────────
    risky_by_catalog = [
        row["name"]
        for row in catalog_rows
        if row.get("numerically_risky", "false").lower() == "true"
    ]
    # Parse live rules from motifs.py instead of using hardcoded dict
    _live_rules = _parse_math_space_rules(motifs_source) if motifs_source else {}
    _ruled_ops = (
        set(_live_rules.keys()) if _live_rules else set(_FALLBACK_MUST_PRECEDE.keys())
    )
    risky_without_rule = [op for op in risky_by_catalog if op not in _ruled_ops]
    if risky_without_rule:
        emit(
            "HIGH",
            "aria-architect",
            "CATALOG_METADATA",
            "MATH_SPACE_RULES",
            f"{len(risky_without_rule)} ops are flagged numerically_risky in the catalog "
            f"but have NO entry in MATH_SPACE_RULES (must_precede): {risky_without_rule}. "
            f"The auto-insert of rmsnorm in _instantiate_motif only fires when "
            f"MATH_SPACE_RULES has the op — these fall through unprotected.",
            "Add each op to MATH_SPACE_RULES in motifs.py with "
            'must_precede: ["rmsnorm", "layernorm"]. '
            "Then the auto-fix in _instantiate_motif will guard them.",
            unguarded_risky_ops=risky_without_rule,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE 6 — HISTORICAL INSIGHT
# Is the search learning from past results, or flying blind?
# ═══════════════════════════════════════════════════════════════════════════════

# Things to look for in the codebase to indicate live ML feedback loops

FEEDBACK_SIGNALS = {
    "notebook_query_in_weights": {
        "patterns": [r"lab_notebook", r"notebook\.query", r"SELECT.*loss.*template"],
        "description": "Lab notebook queried to update template weights",
    },
    "bandit_or_ucb": {
        "patterns": [r"UCB|upper.confidence|bandit|thompson|EI|expected.improvement"],
        "description": "Bandit or Bayesian acquisition function for template selection",
    },
    "bayesian_optimization": {
        "patterns": [r"gaussian.process|GP\b|BoTorch|scikit.optimize|Optuna"],
        "description": "Bayesian optimization for hyperparameter/motif selection",
    },
    "ml_predictor": {
        "patterns": [r"predict.*loss|predict.*score|XGBoost|GBM|RandomForest.*predict"],
        "description": "ML model predicting graph performance before eval",
    },
    "op_statistics_tracked": {
        "patterns": [r"op_stats|motif_stats|per.op.*loss|op_usage"],
        "description": "Per-op/per-motif loss statistics tracked in notebook",
    },
    "warm_start_priors": {
        "patterns": [r"warm.start|prior.*weight|posterior.*weight|beta.prior"],
        "description": "Bayesian warm-start priors informed by historical data",
    },
    "multi_armed_bandit": {
        "patterns": [r"Epsilon.greedy|epsilon_greedy|softmax.*temperature|boltzmann"],
        "description": "Multi-armed bandit exploration policy",
    },
    "novelty_detection": {
        "patterns": [r"novelty|CKA|centered.kernel|structural.similarity|fingerprint"],
        "description": "Graph novelty detection to avoid re-evaluating known configs",
    },
}


def audit_historical_insight(paths: dict[str, Path]) -> None:
    # Collect all relevant source files
    source_blobs: dict[str, str] = {}
    for key, path in paths.items():
        if path.suffix in (".py",) and path.exists():
            try:
                source_blobs[str(path)] = path.read_text(errors="replace")
            except Exception:
                pass

    combined_source = "\n".join(source_blobs.values())

    # ── 6a. Check each feedback signal ───────────────────────────────────────
    for signal_key, signal_info in FEEDBACK_SIGNALS.items():
        patterns = signal_info["patterns"]
        found = any(re.search(pat, combined_source, re.IGNORECASE) for pat in patterns)
        if not found:
            emit(
                "HIGH",
                "aria-scientist",
                "HISTORICAL_INSIGHT",
                signal_key,
                f"No evidence of '{signal_info['description']}' found in any source file. "
                f"The Aria search is NOT using this feedback mechanism. "
                f"GPT-2/Mamba-beating architectures require intelligent search, not pure random walk.",
                _historical_insight_suggestion(signal_key),
                signal=signal_key,
                patterns_checked=patterns,
            )

    # ── 6b. Check if notebook DB is being used for weight updates ────────────
    db_path = paths.get("notebook_db")
    if db_path and db_path.exists():
        try:
            conn = sqlite3.connect(db_path)
            cur = conn.cursor()
            # Check for template-level stats tables
            tables = {
                row[0]
                for row in cur.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                )
            }
            conn.close()

            expected_stat_tables = {"template_stats", "op_stats", "motif_stats"}
            missing_tables = expected_stat_tables - tables
            if missing_tables:
                emit(
                    "HIGH",
                    "aria-scientist",
                    "HISTORICAL_INSIGHT",
                    "lab_notebook.db",
                    f"lab_notebook.db exists but is missing analytics tables: {missing_tables}. "
                    f"Without these, template/op/motif weight updates cannot be automated. "
                    f"Existing tables: {sorted(tables)}",
                    "Create missing tables: "
                    "CREATE TABLE template_stats (template TEXT, eval_count INT, "
                    "mean_loss REAL, min_loss REAL, std_loss REAL); "
                    "Populate after each batch with INSERT OR REPLACE. "
                    "Read in grammar.py / scheduler.py to update sampling weights.",
                    missing_tables=list(missing_tables),
                    existing_tables=sorted(tables),
                )
        except Exception as e:
            emit(
                "MEDIUM",
                "aria-scientist",
                "HISTORICAL_INSIGHT",
                "lab_notebook.db",
                f"Could not query lab_notebook.db: {e}",
                "Ensure lab_notebook.db is accessible and not locked by a running experiment.",
                error=str(e),
            )
    else:
        emit(
            "MEDIUM",
            "aria-scientist",
            "HISTORICAL_INSIGHT",
            "lab_notebook.db",
            "lab_notebook.db not found at expected path. Historical evaluation data "
            "is unavailable for this audit run.",
            "Run 'python -m research --mode=dashboard' to initialize the notebook, "
            "then re-run this audit.",
        )


def _historical_insight_suggestion(signal_key: str) -> str:
    suggestions = {
        "notebook_query_in_weights": (
            "In pick_template(), query lab_notebook.db for template success rates: "
            "SELECT template_name, AVG(val_loss) FROM experiments GROUP BY template_name. "
            "Weight = exp(-k * avg_loss). Cache result for 60 seconds to avoid DB hammering."
        ),
        "bandit_or_ucb": (
            "Implement UCB1 for template selection: "
            "score(t) = mean_reward(t) + C * sqrt(log(N) / n(t)), "
            "where N=total evaluations, n(t)=template eval count, C=exploration constant. "
            "Replace weighted random pick with argmax(score). "
            "File: research/search/scheduler.py"
        ),
        "bayesian_optimization": (
            "Use Optuna or BoTorch to treat template+motif as a categorical-continuous "
            "hyperparameter space. Define a study with val_loss as the objective. "
            "Each eval = one trial. BO will cluster exploration around promising regions. "
            "pip install optuna"
        ),
        "ml_predictor": (
            "Train a GBM (LightGBM) on graph fingerprints → val_loss. "
            "Features: op_histogram, motif_class_histogram, template_name, depth, n_params. "
            "Use it as a cheap pre-screener: only eval graphs predicted to beat current best - 2σ. "
            "This can cut wasted compute by 60-80%."
        ),
        "op_statistics_tracked": (
            "In ExperimentRunner, after each eval, insert: "
            "INSERT INTO op_stats(op, graph_id, val_loss) "
            "SELECT op_name, graph_id, val_loss FROM graph_nodes JOIN experiments. "
            "Aggregate weekly into: UPDATE component_catalog SET motif_avg_loss=..."
        ),
        "warm_start_priors": (
            "Initialize template weights from a Beta distribution fitted on historical "
            "success/failure counts: alpha=successes+1, beta=failures+1. "
            "Thompson sampling then naturally balances exploration/exploitation."
        ),
        "multi_armed_bandit": (
            "Add epsilon-greedy annealing to pick_template(): "
            "with prob epsilon, sample uniformly (explore); "
            "with prob 1-epsilon, sample by performance weights (exploit). "
            "Anneal epsilon from 0.3 → 0.05 over 1000 evaluations."
        ),
        "novelty_detection": (
            "Before eval, compute graph CKA fingerprint (already in research/eval/novelty.py?). "
            "Skip graphs with CKA similarity > 0.95 to any already-evaluated graph. "
            "This prevents wasting compute on structural duplicates."
        ),
    }
    return suggestions.get(
        signal_key, "Investigate and implement this feedback mechanism."
    )


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE 7 — SEARCH DIVERSITY
# Dead code traps, collision rates, exploration coverage
# ═══════════════════════════════════════════════════════════════════════════════


def audit_search_diversity(templates_source: str, paths: dict[str, Path]) -> None:
    bodies = _extract_template_bodies(templates_source)
    weights = _extract_registry_weights(templates_source)

    # ── 7a. Weight concentration — top-N templates dominate ──────────────────
    if weights:
        total_weight = sum(weights.values())
        sorted_by_weight = sorted(weights.items(), key=lambda x: -x[1])
        top5_weight = sum(w for _, w in sorted_by_weight[:5])
        top5_pct = top5_weight / total_weight * 100

        if top5_pct > 35:
            emit(
                "HIGH",
                "aria-scientist",
                "SEARCH_DIVERSITY",
                "DEFAULT_TEMPLATE_WEIGHTS",
                f"Top 5 templates by weight account for {top5_pct:.1f}% of all sampling. "
                f"Top 5: {[(n, w) for n, w in sorted_by_weight[:5]]}. "
                f"This means ~{100 - top5_pct:.0f}% of templates get <{100 / len(weights):.1f}% "
                f"of evaluations — severe exploration bias.",
                "Cap maximum template weight at 3x the median weight. "
                "Alternatively, implement a temperature schedule: "
                "start hot (uniform) for first 500 evals, cool gradually toward weight-greedy. "
                "Or use Boltzmann sampling: weight = exp(lift / T) where T decreases over time.",
                top5_pct=round(top5_pct, 1),
                top5=[(n, w) for n, w in sorted_by_weight[:5]],
            )

    # ── 7b. Templates with only 1 codepath (no randomness) ───────────────────
    fully_deterministic = []
    for name, body in bodies.items():
        has_rng = bool(re.search(r"\brng\.\w+\(|rng\.choice|rng\.random", body))
        has_motif_slot = bool(_MOTIF_SLOT.search(body))
        if not has_rng and not has_motif_slot:
            fully_deterministic.append(name)
    if fully_deterministic:
        emit(
            "MEDIUM",
            "aria-architect",
            "SEARCH_DIVERSITY",
            "templates.py",
            f"{len(fully_deterministic)} templates are FULLY DETERMINISTIC — they always "
            f"produce exactly the same graph regardless of the rng state. "
            f"Each contributes only 1 unique graph to the entire search: "
            f"{', '.join(fully_deterministic)}",
            "Add at least one motif slot or rng.choice() per template. "
            "Minimum: randomize activation function (silu/gelu/relu), "
            "or add an optional FFN motif with 50% probability.",
            deterministic_templates=fully_deterministic,
        )

    # ── 7c. Missing templates for key paradigms ───────────────────────────────
    paradigm_coverage = {
        "state_space_only": any(
            "state_space" in b and "attention" not in b for b in bodies.values()
        ),
        "pure_conv": any("conv" in n for n in bodies),
        "linear_only_no_attn": any(
            "linear" in n and "attention" not in n for n in bodies
        ),
        "cross_modal_fusion": False,  # Would need embeddings from different modalities
        "two_tower": any("tower" in n for n in bodies),
        "retrieval_augmented": any("retrieval" in n or "topk" in n for n in bodies),
        "continual_learning": False,
        "gradient_checkpoint": False,
        "stochastic_depth": False,
        "dropout_free": False,  # Most modern arches — is there a stochastic_depth?
    }
    missing_paradigms = [k for k, v in paradigm_coverage.items() if not v]
    if missing_paradigms:
        emit(
            "MEDIUM",
            "aria-architect",
            "SEARCH_DIVERSITY",
            "templates.py",
            f"The following architectural paradigms have NO template: {missing_paradigms}. "
            f"These are all common in high-performing modern language models. "
            f"Without templates for these paradigms, the search cannot discover "
            f"architectures in these design regions.",
            "Add one template per missing paradigm. Minimal effort: "
            "'stochastic_depth' = randomly skip blocks with prob p=0.1; "
            "'gradient_checkpoint' template tag (metadata only, for memory-efficient training); "
            "'pure_conv' = layernorm → conv1d_seq × 3 → channel mixer → residual.",
            missing_paradigms=missing_paradigms,
        )

    # ── 7d. No explicit exploration scheduler ────────────────────────────────
    scheduler_path = paths.get("scheduler")
    if not (scheduler_path and scheduler_path.exists()):
        emit(
            "HIGH",
            "aria-scientist",
            "SEARCH_DIVERSITY",
            "scheduler.py",
            "No exploration scheduler found at research/search/scheduler.py. "
            "Without a scheduler, the search uses static weights forever — "
            "it cannot adapt to discovered regions, cannot enforce minimum coverage "
            "per template, and cannot cool exploration as the leaderboard matures.",
            "Create research/search/scheduler.py with: "
            "class ExplorationScheduler: "
            "  def step(self, n_evals, template_counts) -> Dict[str, float]: "
            "    # Increase weight of under-sampled templates "
            "    # Decrease weight of over-sampled templates "
            "    # Apply temperature annealing "
            "    return updated_template_weights",
        )


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE 8 — COMMON SENSE
# What any ML engineer would expect to be present
# ═══════════════════════════════════════════════════════════════════════════════

COMMON_SENSE_CHECKS: list[tuple] = [
    # (severity, agent, location, finding, suggestion)
    (
        "HIGH",
        "aria-scientist",
        "templates.py / grammar.py",
        "No template enforces that EVERY graph ends with a final LayerNorm before "
        "the output head. GPT-2, LLaMA, Mamba all have a final norm. Without it, "
        "the logit scale is determined by the last block's variance — uncontrolled.",
        "Add post_norm: Optional[str] field to ComputationGraph. "
        "Grammar should always insert rmsnorm/layernorm between last template output "
        "and output_head. This alone is worth 5-15% perplexity improvement.",
    ),
    (
        "HIGH",
        "aria-scientist",
        "templates.py",
        "No template specializes for EARLY vs LATE layers. "
        "GPT-2 research shows attention is more important in late layers, "
        "FFN more important in early layers. All templates are depth-agnostic.",
        "Add layer_index parameter to template selection. "
        "For depth < N/3: prefer FFN/conv templates. "
        "For depth > 2N/3: prefer attention/SSM templates. "
        "Use graph.metadata['layer_depth'] if available.",
    ),
    (
        "HIGH",
        "aria-architect",
        "tpl_three_way_split",
        "A 3-lane split into (feature/3 each) then concat gives each lane "
        "only D/3 feature width. With D=256 → D/3=85 features per lane. "
        "None of the split lane motifs are designed for reduced-width inputs. "
        "This is architecturally unsound — the lanes are too narrow for complex mixing.",
        "Either: (a) split on the TOKEN axis not the FEATURE axis, keeping each "
        "lane at full D width, or (b) up-project each part to D before its motif "
        "and down-project after before concat. Option (a) is simpler.",
    ),
    (
        "MEDIUM",
        "aria-scientist",
        "grammar.py / templates.py",
        "No template generates a POSITIONAL ENCODING layer. Every serious LM "
        "architecture uses RoPE, ALiBi, or sinusoidal position encoding. "
        "rope_rotate exists in the catalog but is used in 0 templates.",
        "Add rope_rotate as an optional slot after the first embedding lookup "
        "in any mixing template. Template pattern: "
        "embed → rope_rotate → [mixer motif] → residual.",
    ),
    (
        "HIGH",
        "aria-scientist",
        "templates.py",
        "No template models the interaction between DEPTH and PARAMETER COUNT. "
        "A 3-template deep graph with param-heavy motifs in every slot may have "
        "3x the params of a shallower graph — but the eval doesn't normalize for "
        "parameter count. The search may be selecting big-param graphs simply "
        "because they have more capacity, not because the architecture is better.",
        "Add param_budget: Optional[int] to graph config. Reject graphs where "
        "total_params > budget * 1.2. Or normalize reported loss by parameter count: "
        "adjusted_score = val_loss * (max_params / graph_params)^0.3",
    ),
    (
        "HIGH",
        "aria-architect",
        "tpl_sequential",
        "tpl_sequential stacks 2-3 motifs with NO normalization and NO residual. "
        "A pure sequential stack without any of these will suffer gradient vanishing "
        "for any depth > 2. This template is a training instability factory.",
        "Wrap each motif in the sequential loop with: "
        "(1) pre-norm, (2) motif, (3) add(prev_output, motif_output). "
        "Make it a loop of mini residual blocks, not a raw chain.",
    ),
    (
        "HIGH",
        "aria-architect",
        "tpl_dense_cascade",
        "tpl_dense_cascade has 0 normalization ops. A DenseNet without normalization "
        "was tried in 2016-era architectures and universally failed to scale. "
        "The dense skip connections compound variance exponentially.",
        "Add layernorm before each motif in the dense cascade loop. "
        "Pattern: for each step: normed = norm(outputs[-1]); motif(normed); "
        "dense_add = add(outputs[0], motif_out); outputs.append(dense_add)",
    ),
    (
        "MEDIUM",
        "aria-scientist",
        "primitives.py / templates.py",
        "The search has no KV-cache compatibility budget. Architectures that "
        "perform well at training may be unusable at inference because they "
        "require O(S²) memory for each forward pass (no caching). "
        "The goal is to beat GPT-2 — GPT-2 has O(S) KV cache.",
        "Add a kv_cacheable: bool flag to ComputationGraph. Templates that use "
        "state_space, rwkv_time_mixing, or linear_attention are cacheable. "
        "Templates using softmax_attention are cacheable only with standard KV cache. "
        "Templates with adaptive_recursion or mixed_recursion_gate are NOT cacheable. "
        "Track this as a first-class metric in eval output.",
    ),
    (
        "HIGH",
        "aria-scientist",
        "eval / scoring",
        "The leaderboard ranks by val_loss alone. GPT-2 and Mamba are competitive "
        "on both perplexity AND throughput. A 5× improvement in perplexity at 10× "
        "slower inference is not a win. The scoring function needs a speed term.",
        "Add composite score = val_loss * (target_tokens_per_sec / actual_tps)^α "
        "where α controls the speed-accuracy tradeoff (suggest α=0.3). "
        "Eval must measure: (1) val_loss, (2) tokens/sec at batch_size=1, "
        "(3) peak VRAM, (4) total params. Surface all 4 in the leaderboard.",
    ),
    (
        "CRITICAL",
        "aria-scientist",
        "template design",
        "There are ZERO templates that combine positional encoding + attention + FFN "
        "in the canonical GPT-2/LLaMA pattern as a BASELINE REFERENCE. "
        "Without a canonical baseline, you cannot know whether discovered architectures "
        "are actually better than well-tuned vanilla transformers at the same param count.",
        "Add tpl_gpt2_reference: embed → rope_rotate → rmsnorm → softmax_attention "
        "→ add → rmsnorm → swiglu_mlp → add. Run this as a required baseline in "
        "every search batch. Its score is the floor — anything below it is a failure.",
    ),
    (
        "HIGH",
        "aria-architect",
        "templates.py",
        "No template explores WEIGHT SHARING across lanes or blocks. "
        "GPT-2 uses tied input/output embeddings. ALBERT uses layer weight sharing. "
        "tied_proj exists in the catalog but no template forces weight sharing "
        "as a design strategy.",
        "Create tpl_weight_shared_block: uses tied_proj for the main projection "
        "across multiple instantiations. Also try a 'shared_basis' template using "
        "shared_basis_proj to force all slots to project through a common basis.",
    ),
]


def audit_common_sense(
    templates_source: str = "",
    grammar_source: str = "",
    motifs_source: str = "",
) -> None:
    # Verify each static check against actual source before emitting
    for severity, agent, location, finding, suggestion in COMMON_SENSE_CHECKS:
        # Suppress findings that are now false positives
        skip = False

        if "final LayerNorm before the output head" in finding:
            # Check if grammar.py adds final norm before output
            if re.search(
                r"rmsnorm.*output|final.*norm|last_op.*not in.*rmsnorm", grammar_source
            ):
                skip = True

        elif (
            "ZERO templates that combine positional encoding + attention + FFN"
            in finding
        ):
            # Check if tpl_gpt2_reference or similar exists
            if "tpl_gpt2_reference" in templates_source:
                skip = True

        elif "tpl_sequential stacks 2-3 motifs with NO normalization" in finding:
            # Check if tpl_sequential now has pre-norm
            bodies = _extract_template_bodies(templates_source)
            seq_body = bodies.get("sequential", "")
            if _NORM_CALLS.search(seq_body):
                skip = True

        elif "tpl_dense_cascade has 0 normalization" in finding:
            bodies = _extract_template_bodies(templates_source)
            dc_body = bodies.get("dense_cascade", "")
            if _NORM_CALLS.search(dc_body):
                skip = True

        elif "No template generates a POSITIONAL ENCODING" in finding:
            # Check if rope_rotate appears in any template
            if "rope_rotate" in templates_source:
                skip = True

        elif "No template explores WEIGHT SHARING" in finding:
            if (
                "tied_proj" in templates_source
                or "shared_basis_proj" in templates_source
            ):
                skip = True

        elif "leaderboard ranks by val_loss alone" in finding:
            # Check if composite_score or tokens_per_sec exists in scientist code
            if (
                "composite_score" in grammar_source
                or "tokens_per_sec" in grammar_source
            ):
                skip = True
            # Also check by looking at scientist source (passed via grammar_source or motifs_source)
            # The scoring system has been overhauled — suppress if composite_score exists anywhere
            try:
                from pathlib import Path as _P

                _sci_dir = _P(__file__).parent / "research/scientist"
                if _sci_dir.exists():
                    for _f in _sci_dir.rglob("*.py"):
                        try:
                            if "composite_score" in _f.read_text(errors="replace"):
                                skip = True
                                break
                        except Exception:
                            pass
            except Exception:
                pass

        if not skip:
            emit(severity, agent, "COMMON_SENSE", location, finding, suggestion)

    # ── 8b. Dimension consistency check ──────────────────────────────────────
    # Suppress if graph_validator.py with validate_dim_flow exists
    try:
        from pathlib import Path as _P

        _gv = _P(__file__).parent / "research/synthesis/graph_validator.py"
        _has_dim_validator = _gv.exists() and "validate_dim_flow" in _gv.read_text(
            errors="replace"
        )
    except Exception:
        _has_dim_validator = False
    if not _has_dim_validator:
        emit(
            "HIGH",
            "aria-kernel",
            "COMMON_SENSE",
            "primitives.py",
            "No static dimension-flow validator exists. The current approach tries "
            "add_op and catches ValueError at synthesis time. This means bad graphs "
            "silently fall back to input_id with no diagnostic, producing graphs that "
            "look valid but have skip-only paths.",
            "Add validate_graph_shapes(graph) → list[str] that walks the DAG and "
            "reports every node whose output_shape.dim does not match what its consumer "
            "expects. Run BEFORE eval to catch these silently broken paths.",
        )


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE 9 — OP ROLE COVERAGE
# Every primitive op should have an OpRole in op_roles.py
# ═══════════════════════════════════════════════════════════════════════════════


def _parse_op_role_map(source: str) -> dict[str, str]:
    """Parse _OP_ROLE_MAP from op_roles.py source → {op_name: role_name}."""
    return dict(re.findall(r'"(\w+)":\s*OpRole\.(\w+)', source))


def audit_op_roles(op_roles_source: str, catalog_rows: list[dict]) -> None:
    role_map = _parse_op_role_map(op_roles_source)
    ops_with_roles = set(role_map.keys())

    # 9a. Primitives missing an OpRole
    missing = []
    for row in catalog_rows:
        if row.get("component_type") == "primitive":
            op = row.get("name", "")
            if op and op not in ops_with_roles:
                missing.append(op)
    if missing:
        emit(
            "HIGH",
            "aria-architect",
            "OP_ROLE_COVERAGE",
            "op_roles.py",
            f"{len(missing)} primitive ops have no OpRole assignment in _OP_ROLE_MAP. "
            f"The grammar uses OpRole for valid sequencing — unassigned ops bypass "
            f"role-based ordering checks. Missing: {', '.join(sorted(missing)[:15])}"
            f"{'...' if len(missing) > 15 else ''}",
            "Add each op to _OP_ROLE_MAP with the appropriate OpRole.",
            missing_ops=sorted(missing),
        )

    # 9b. Ops in role map but not in catalog (stale entries)
    catalog_ops = {row.get("name", "") for row in catalog_rows if row.get("name")}
    stale = [op for op in ops_with_roles if op not in catalog_ops]
    if stale:
        emit(
            "LOW",
            "aria-architect",
            "OP_ROLE_COVERAGE",
            "op_roles.py",
            f"{len(stale)} ops in _OP_ROLE_MAP are not in the component catalog: "
            f"{', '.join(sorted(stale)[:10])}. These may be deleted ops with stale entries.",
            "Remove stale entries from _OP_ROLE_MAP or add missing catalog rows.",
            stale_ops=sorted(stale),
        )

    # 9c. Suspicious role assignments (norm op tagged as ACTIVATE, etc.)
    norm_ops = {"rmsnorm", "layernorm", "groupnorm", "instancenorm", "batchnorm"}
    for op, role in role_map.items():
        if op in norm_ops and role != "NORMALIZE":
            emit(
                "MEDIUM",
                "aria-architect",
                "OP_ROLE_COVERAGE",
                f"op:{op}",
                f"Op '{op}' is a normalization op but has role {role} instead of NORMALIZE.",
                f"Change _OP_ROLE_MAP['{op}'] to OpRole.NORMALIZE.",
                op=op,
                current_role=role,
            )


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE 10 — CONTEXT RULE COVERAGE
# Every primitive should have a ContextRule; forbidden pairs should be symmetric
# ═══════════════════════════════════════════════════════════════════════════════


def _parse_context_rules(source: str) -> set[str]:
    """Parse ops that have ContextRule entries in CONTEXT_RULES dict."""
    # Match both dict literal and subscript assignment patterns
    ops = set(re.findall(r'"(\w+)":\s*ContextRule\(', source))
    ops |= set(re.findall(r'CONTEXT_RULES\["(\w+)"\]', source))
    return ops


def _parse_forbidden_pairs(source: str) -> list[tuple[str, str, str]]:
    """Extract (op, direction, forbidden_op) triples from CONTEXT_RULES."""
    pairs: list[tuple[str, str, str]] = []
    # Find each ContextRule entry and extract forbidden sets
    for m in re.finditer(
        r'"(\w+)":\s*ContextRule\([^)]*?'
        r"forbidden_predecessors\s*=\s*(?:frozenset\(\{([^}]*)\}|(_\w+_OPS[^,]*))",
        source,
        re.DOTALL,
    ):
        op = m.group(1)
        preds_str = m.group(2) or ""
        for pred in re.findall(r'"(\w+)"', preds_str):
            pairs.append((op, "predecessor", pred))
    for m in re.finditer(
        r'"(\w+)":\s*ContextRule\([^)]*?'
        r"forbidden_successors\s*=\s*(?:frozenset\(\{([^}]*)\}|(_\w+_OPS[^,]*))",
        source,
        re.DOTALL,
    ):
        op = m.group(1)
        succs_str = m.group(2) or ""
        for succ in re.findall(r'"(\w+)"', succs_str):
            pairs.append((op, "successor", succ))
    return pairs


def audit_context_rules(context_rules_source: str, catalog_rows: list[dict]) -> None:
    ruled_ops = _parse_context_rules(context_rules_source)

    # 10a. Primitives missing a ContextRule
    missing = []
    for row in catalog_rows:
        if row.get("component_type") == "primitive":
            op = row.get("name", "")
            if op and op not in ruled_ops:
                missing.append(op)
    if missing:
        emit(
            "MEDIUM",
            "aria-architect",
            "CONTEXT_RULE_COVERAGE",
            "context_rules.py",
            f"{len(missing)} primitive ops have no ContextRule entry. Without a rule, "
            f"these ops use default placement (GENERAL mode, no forbidden pairs). "
            f"Missing: {', '.join(sorted(missing)[:15])}"
            f"{'...' if len(missing) > 15 else ''}",
            "Add ContextRule entries for each op in CONTEXT_RULES dict.",
            missing_ops=sorted(missing),
        )

    # 10b. Symmetry check — if A forbids B as successor, B should forbid A as predecessor
    pairs = _parse_forbidden_pairs(context_rules_source)
    successor_forbids: dict[tuple[str, str], bool] = {}
    predecessor_forbids: dict[tuple[str, str], bool] = {}
    for op, direction, other in pairs:
        if direction == "successor":
            successor_forbids[(op, other)] = True
        elif direction == "predecessor":
            predecessor_forbids[(op, other)] = True

    asymmetric = []
    for op_a, op_b in successor_forbids:
        if (op_b, op_a) not in predecessor_forbids:
            asymmetric.append((op_a, op_b))
    if asymmetric:
        emit(
            "MEDIUM",
            "aria-architect",
            "CONTEXT_RULE_COVERAGE",
            "context_rules.py",
            f"{len(asymmetric)} asymmetric forbidden pairs found. "
            f"If A forbids B as successor, B should forbid A as predecessor for "
            f"consistency. Examples: {asymmetric[:5]}",
            "Add matching forbidden_predecessors entries for each asymmetric pair.",
            asymmetric_pairs=asymmetric[:10],
        )


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE 11 — ROUTING SYSTEM INTEGRITY
# True routing ops properly integrated, aliases resolve, no stale old names
# ═══════════════════════════════════════════════════════════════════════════════


def _parse_op_name_aliases(primitives_source: str) -> dict[str, str]:
    """Parse OP_NAME_ALIASES dict from primitives.py source."""
    m = re.search(r"OP_NAME_ALIASES\s*[:{].*?\{(.*?)\}", primitives_source, re.DOTALL)
    if not m:
        return {}
    return dict(re.findall(r'"(\w+)":\s*"(\w+)"', m.group(1)))


def audit_routing_system(
    true_routing_source: str | None,
    context_rules_source: str,
    templates_source: str,
    primitives_source: str,
) -> None:
    # 11a. True routing ops have ContextRules
    true_routing_ops = ["hetero_moe", "arch_router", "compute_budget_router"]
    ruled_ops = _parse_context_rules(context_rules_source)
    for op in true_routing_ops:
        if op not in ruled_ops:
            emit(
                "HIGH",
                "aria-architect",
                "ROUTING_SYSTEM",
                f"op:{op}",
                f"True routing op '{op}' has no ContextRule entry. These ops use "
                f"gather-scatter dispatch and need specific placement constraints.",
                f"Add ContextRule for '{op}' in context_rules.py.",
                op=op,
            )

    # 11b. True routing ops appear in at least one template
    for op in true_routing_ops:
        if f'"{op}"' not in templates_source and f"'{op}'" not in templates_source:
            emit(
                "MEDIUM",
                "aria-architect",
                "ROUTING_SYSTEM",
                f"op:{op}",
                f"True routing op '{op}' doesn't appear in any template. "
                f"It's defined in true_routing_ops.py but never used in synthesis.",
                f"Add '{op}' to a routing template or create a dedicated template.",
                op=op,
            )

    # 11c. OP_NAME_ALIASES all resolve (old name → new name both in registry)
    aliases = _parse_op_name_aliases(primitives_source)
    for old_name, new_name in aliases.items():
        # New name should appear in templates or primitives as a registered op
        if f'"{new_name}"' not in primitives_source:
            emit(
                "HIGH",
                "aria-architect",
                "ROUTING_SYSTEM",
                f"alias:{old_name}→{new_name}",
                f"Alias '{old_name}' → '{new_name}' but '{new_name}' is not found as a "
                f"registered primitive. The alias resolves to nothing.",
                f"Register '{new_name}' in PRIMITIVE_REGISTRY or remove the stale alias.",
                old_name=old_name,
                new_name=new_name,
            )

    # 11d. No template uses old (aliased) names directly
    for old_name in aliases:
        if f'"{old_name}"' in templates_source or f"'{old_name}'" in templates_source:
            emit(
                "MEDIUM",
                "aria-architect",
                "ROUTING_SYSTEM",
                f"old_name:{old_name}",
                f"Template source still references old op name '{old_name}' "
                f"(renamed to '{aliases[old_name]}'). Should use new name.",
                f"Replace '{old_name}' with '{aliases[old_name]}' in templates.",
                old_name=old_name,
                new_name=aliases[old_name],
            )


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE 12 — VALIDATOR WIRING
# template_rules.py and graph_validator.py are actually called
# ═══════════════════════════════════════════════════════════════════════════════


def audit_validator_wiring(
    grammar_source: str,
    compiler_source: str | None = None,
) -> None:
    # 12a. validate_template_graph is imported and called in grammar.py
    if "validate_template_graph" not in grammar_source:
        emit(
            "HIGH",
            "aria-scientist",
            "VALIDATOR_WIRING",
            "grammar.py",
            "template_rules.validate_template_graph() is not imported or called in "
            "grammar.py. Template-level invariants (final_norm, lane_diversity, "
            "bottleneck_dim) are defined but never enforced during synthesis.",
            "Add: from .template_rules import validate_template_graph "
            "and call it after graph construction in _build_graph().",
        )
    elif not re.search(r"validate_template_graph\(", grammar_source):
        emit(
            "MEDIUM",
            "aria-scientist",
            "VALIDATOR_WIRING",
            "grammar.py",
            "validate_template_graph is imported but never called in grammar.py.",
            "Call validate_template_graph(graph) after graph construction.",
        )

    # 12b. validate_dim_flow is imported and called in grammar.py
    if "validate_dim_flow" not in grammar_source:
        emit(
            "HIGH",
            "aria-scientist",
            "VALIDATOR_WIRING",
            "grammar.py",
            "graph_validator.validate_dim_flow() is not imported or called in grammar.py. "
            "Dimension-flow validation is defined but never enforced.",
            "Add: from .graph_validator import validate_dim_flow "
            "and call it before returning the graph.",
        )
    elif not re.search(r"validate_dim_flow\(", grammar_source):
        emit(
            "MEDIUM",
            "aria-scientist",
            "VALIDATOR_WIRING",
            "grammar.py",
            "validate_dim_flow is imported but never called in grammar.py.",
            "Call validate_dim_flow(graph) after graph construction.",
        )

    # 12c. check_param_budget is imported and called
    if not re.search(r"check_param_budget\(", grammar_source):
        emit(
            "MEDIUM",
            "aria-scientist",
            "VALIDATOR_WIRING",
            "grammar.py",
            "check_param_budget() is not called in grammar.py. "
            "Graphs can exceed param budget without rejection.",
            "Call check_param_budget(graph, max_params) during validation.",
        )

    # 12d. annotate_kv_cacheable is called somewhere (grammar or compiler)
    called_anywhere = "annotate_kv_cacheable(" in grammar_source
    if compiler_source:
        called_anywhere = called_anywhere or "annotate_kv_cacheable(" in compiler_source
    if not called_anywhere:
        emit(
            "MEDIUM",
            "aria-scientist",
            "VALIDATOR_WIRING",
            "compiler.py",
            "annotate_kv_cacheable() is never called. KV-cache compatibility "
            "is not being tracked for synthesized graphs.",
            "Call annotate_kv_cacheable(graph) in compiler.py after compilation.",
        )


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE 13 — MOTIF COMPOSITION RULES
# Verify motifs don't contain forbidden op sequences
# ═══════════════════════════════════════════════════════════════════════════════


def _parse_motif_steps(motifs_source: str) -> list[tuple[str, list[str]]]:
    """Parse motifs and their op sequences → [(motif_name, [op1, op2, ...])]."""
    motifs: list[tuple[str, list[str]]] = []
    # Find Motif(...) blocks with name and steps
    for m in re.finditer(
        r'Motif\(\s*name="(\w+)".*?steps=\(\s*(.*?)\s*\)',
        motifs_source,
        re.DOTALL,
    ):
        name = m.group(1)
        steps_str = m.group(2)
        ops = re.findall(r'MotifStep\("(\w+)"', steps_str)
        if ops:
            motifs.append((name, ops))
    return motifs


def audit_motif_composition(motifs_source: str) -> None:
    motifs = _parse_motif_steps(motifs_source)
    parsed_rules = _parse_math_space_rules(motifs_source)

    # Forbidden sequences within motifs
    _MERGE_OPS = {"token_merge", "adjacent_token_merge", "depth_token_mask"}
    _SSM_OPS = {"state_space", "selective_scan", "rwkv_time_mixing", "linear_attention"}
    _NORM_OPS = {"rmsnorm", "layernorm", "groupnorm", "instancenorm"}
    _ACTIVATE_OPS = {"gelu", "silu", "relu", "sigmoid", "tanh"}

    for name, ops in motifs:
        for i, op in enumerate(ops):
            # 13a. No SSM op after merge op within same motif
            if op in _MERGE_OPS:
                for j in range(i + 1, len(ops)):
                    if ops[j] in _SSM_OPS:
                        emit(
                            "CRITICAL",
                            "aria-architect",
                            "MOTIF_COMPOSITION",
                            f"motif:{name}",
                            f"Motif '{name}' has {op} followed by {ops[j]}. "
                            f"Merge ops change sequence length, breaking SSM/recurrent ops. "
                            f"100% S0 failure rate pattern.",
                            f"Remove '{ops[j]}' from motif '{name}' or move merge to end.",
                            motif=name,
                            merge_op=op,
                            ssm_op=ops[j],
                        )

            # 13b. No two adjacent NORMALIZE ops (wasteful)
            if op in _NORM_OPS and i + 1 < len(ops) and ops[i + 1] in _NORM_OPS:
                emit(
                    "MEDIUM",
                    "aria-architect",
                    "MOTIF_COMPOSITION",
                    f"motif:{name}",
                    f"Motif '{name}' has adjacent norm ops: {op} → {ops[i + 1]}. "
                    f"Double normalization is wasteful and can suppress gradient signal.",
                    f"Remove one of the adjacent norms in motif '{name}'.",
                    motif=name,
                )

            # 13c. ACTIVATE without preceding PROJECT or MIX
            if op in _ACTIVATE_OPS and i == 0:
                emit(
                    "LOW",
                    "aria-architect",
                    "MOTIF_COMPOSITION",
                    f"motif:{name}",
                    f"Motif '{name}' starts with activation '{op}' — no preceding "
                    f"projection or mixing op. Activation on raw input has no learned "
                    f"transform to activate.",
                    f"Add a linear_proj or mixing op before '{op}' in motif '{name}'.",
                    motif=name,
                    op=op,
                )

            # 13d. Op with MATH_SPACE_RULES must_precede — check within motif
            if op in parsed_rules and parsed_rules[op]:
                required_preds = parsed_rules[op]
                preceding_ops = set(ops[:i])
                has_valid_pred = bool(preceding_ops & required_preds) or bool(
                    preceding_ops & _NORM_OPS  # norms generally satisfy must_precede
                )
                if i > 0 and not has_valid_pred:
                    emit(
                        "MEDIUM",
                        "aria-architect",
                        "MOTIF_COMPOSITION",
                        f"motif:{name}",
                        f"Motif '{name}' uses '{op}' which requires predecessors "
                        f"{required_preds} but none appear before it in the motif steps. "
                        f"The grammar's auto-fix only covers _instantiate_motif, "
                        f"not the motif's internal step order.",
                        f"Reorder motif steps to include a valid predecessor before '{op}'.",
                        motif=name,
                        op=op,
                        required=list(required_preds),
                    )


# ═══════════════════════════════════════════════════════════════════════════════
# REPORT GENERATION
# ═══════════════════════════════════════════════════════════════════════════════


def generate_markdown_report(findings: list[AuditFinding], root: Path) -> str:
    from collections import Counter

    lines = []
    lines.append("# Aria Architecture Search — Audit Report\n")
    lines.append(f"**Workspace root:** `{root}`\n")

    # Summary table
    by_severity = Counter(f.severity for f in findings)
    by_agent = Counter(f.agent for f in findings)

    lines.append("## Executive Summary\n")
    lines.append(f"**Total findings:** {len(findings)}\n")
    lines.append("| Severity | Count |")
    lines.append("|----------|-------|")
    for sev in SEVERITIES:
        lines.append(f"| {sev} | {by_severity.get(sev, 0)} |")
    lines.append("")

    lines.append("| Agent | Count |")
    lines.append("|-------|-------|")
    for agent in AGENTS:
        lines.append(f"| {agent} | {by_agent.get(agent, 0)} |")
    lines.append("")

    # Key themes
    lines.append("## Key Themes\n")
    themes = [
        (
            "🧱 Structural",
            "Templates missing normalization, residuals, or lane diversity",
        ),
        (
            "📊 Historical Insight",
            "Weights are hardcoded — not learned from evaluation data",
        ),
        ("🔍 Coverage", "30%+ of primitives never appear in any template"),
        (
            "🧠 Common Sense",
            "Missing baselines, positional encoding, final norm, speed metrics",
        ),
        ("🔢 Math Rules", "Numerically risky ops used without required predecessors"),
    ]
    for icon, text in themes:
        lines.append(f"- **{icon}**: {text}")
    lines.append("")

    # Per-category findings
    lines.append("## Findings by Category\n")
    for category in sorted(set(f.category for f in findings)):
        cat_findings = [f for f in findings if f.category == category]
        lines.append(f"### {category} ({len(cat_findings)} findings)\n")
        for f in sorted(cat_findings, key=lambda x: _SEVERITY_ORDER[x.severity]):
            lines.append(f"#### [{f.severity}] `{f.location}` → *{f.agent}*\n")
            lines.append(f"**Finding:** {f.finding}\n")
            lines.append(f"**Suggestion:** {f.suggestion}\n")
            if f.evidence:
                ev_str = json.dumps(f.evidence, indent=2, default=str)
                lines.append(
                    f"<details><summary>Evidence</summary>\n\n```json\n{ev_str}\n```\n</details>\n"
                )

    # Agent dispatch table
    lines.append("## Agent Dispatch Tasklist\n")
    lines.append(
        "Use this section to distribute findings to local Claude Code agents.\n"
    )
    for agent in AGENTS:
        agent_findings = [f for f in findings if f.agent == agent]
        if not agent_findings:
            continue
        lines.append(f"### {agent} ({len(agent_findings)} tasks)\n")
        criticals = [f for f in agent_findings if f.severity == "CRITICAL"]
        highs = [f for f in agent_findings if f.severity == "HIGH"]
        if criticals:
            lines.append("**CRITICAL — Fix first:**")
            for f in criticals:
                lines.append(f"- `{f.location}`: {f.finding[:100]}...")
            lines.append("")
        if highs:
            lines.append("**HIGH — Fix this session:**")
            for f in highs:
                lines.append(f"- `{f.location}`: {f.finding[:100]}...")
            lines.append("")

    return "\n".join(lines)


def generate_agent_json(findings: list[AuditFinding]) -> dict:
    """JSON structure optimized for agent consumption."""
    by_agent: dict[str, dict] = {}
    for agent in AGENTS:
        agent_findings = sorted(
            [asdict(f) for f in findings if f.agent == agent],
            key=lambda x: _SEVERITY_ORDER[x["severity"]],
        )
        by_agent[agent] = {
            "total": len(agent_findings),
            "critical": [f for f in agent_findings if f["severity"] == "CRITICAL"],
            "high": [f for f in agent_findings if f["severity"] == "HIGH"],
            "medium": [f for f in agent_findings if f["severity"] == "MEDIUM"],
            "low": [f for f in agent_findings if f["severity"] == "LOW"],
        }
    return {
        "summary": {
            "total_findings": len(findings),
            "by_severity": {
                s: sum(1 for f in findings if f.severity == s) for s in SEVERITIES
            },
            "by_category": {
                c: sum(1 for f in findings if f.category == c)
                for c in sorted(set(f.category for f in findings))
            },
        },
        "by_agent": by_agent,
        "all_findings": [
            asdict(f)
            for f in sorted(findings, key=lambda x: _SEVERITY_ORDER[x.severity])
        ],
    }


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(description="Aria Architecture Search Audit")
    parser.add_argument(
        "--root", default="/home/tim/Projects/LLM", help="Workspace root directory"
    )
    parser.add_argument(
        "--out", default="audit_report", help="Output file base name (no extension)"
    )
    parser.add_argument(
        "--templates", default=None, help="Override path to templates.py"
    )
    parser.add_argument(
        "--catalog", default=None, help="Override path to component_catalog.csv"
    )
    args = parser.parse_args()

    root = Path(args.root)
    print(f"[aria_audit] Workspace root: {root}")

    paths = discover_paths(root)

    # Allow overrides
    if args.templates:
        paths["templates_py"] = Path(args.templates)
    if args.catalog:
        paths["component_catalog"] = Path(args.catalog)

    # ── Load templates source ─────────────────────────────────────────────────
    # Templates may be split across multiple files (_templates_core.py, etc.)
    tpl_path = paths["templates_py"]
    tpl_dir = tpl_path.parent
    templates_source = ""
    tpl_files_loaded = []

    # Collect all template source files: templates.py + _templates_*.py + _template_helpers.py
    candidate_files = [tpl_path]
    if tpl_dir.exists():
        for p in sorted(tpl_dir.glob("_templates_*.py")):
            candidate_files.append(p)
        helper = tpl_dir / "_template_helpers.py"
        if helper.exists():
            candidate_files.append(helper)

    for f in candidate_files:
        if f.exists():
            templates_source += f.read_text(errors="replace") + "\n"
            tpl_files_loaded.append(f.name)

    if not templates_source.strip():
        print("[ERROR] Cannot find any template files. Audit will be incomplete.")
    else:
        print(f"[aria_audit] template files: {', '.join(tpl_files_loaded)}")

    print(f"[aria_audit] templates total: {len(templates_source):,} chars")

    # ── Load motifs source ────────────────────────────────────────────────────
    motifs_path = paths.get("motifs_py")
    motifs_source = (
        motifs_path.read_text(errors="replace")
        if (motifs_path and motifs_path.exists())
        else None
    )

    # ── Load grammar source ───────────────────────────────────────────────────
    grammar_path = paths.get("grammar_py")
    grammar_source = (
        grammar_path.read_text(errors="replace")
        if (grammar_path and grammar_path.exists())
        else ""
    )

    # ── Load new module sources ───────────────────────────────────────────────
    def _load(key: str) -> str:
        p = paths.get(key)
        return p.read_text(errors="replace") if (p and p.exists()) else ""

    op_roles_source = _load("op_roles_py")
    context_rules_source = _load("context_rules_py")
    true_routing_source = _load("true_routing_py")
    primitives_source = _load("primitives_py")
    compiler_source = _load("compiler_py")

    # ── Load component catalog ────────────────────────────────────────────────
    cat_path = paths["component_catalog"]
    if not cat_path.exists():
        cat_path = Path("/mnt/user-data/uploads/component_catalog.csv")
    catalog_rows: list[dict] = []
    if cat_path.exists():
        with open(cat_path, newline="", encoding="utf-8") as fh:
            catalog_rows = list(csv.DictReader(fh))
        print(f"[aria_audit] catalog: {len(catalog_rows)} rows")
    else:
        print("[WARN] component_catalog.csv not found.")

    # ── Run audit modules ─────────────────────────────────────────────────────
    print("[aria_audit] Running Module 1: TEMPLATE_STRUCTURE")
    if templates_source:
        audit_template_structure(templates_source)

    print("[aria_audit] Running Module 2: SLOT_COMPLETENESS")
    if templates_source:
        audit_slot_completeness(templates_source)

    print("[aria_audit] Running Module 3: MATH_OPS_RULES")
    if templates_source:
        audit_math_ops_rules(templates_source, motifs_source)

    print("[aria_audit] Running Module 4: COMPONENT_COVERAGE")
    if templates_source and catalog_rows:
        audit_component_coverage(templates_source, catalog_rows, motifs_source)

    print("[aria_audit] Running Module 5: CATALOG_METADATA")
    if catalog_rows:
        audit_catalog_metadata(catalog_rows, motifs_source)

    print("[aria_audit] Running Module 6: HISTORICAL_INSIGHT")
    audit_historical_insight(paths)

    print("[aria_audit] Running Module 7: SEARCH_DIVERSITY")
    if templates_source:
        audit_search_diversity(templates_source, paths)

    print("[aria_audit] Running Module 8: COMMON_SENSE")
    audit_common_sense(templates_source, grammar_source, motifs_source or "")

    print("[aria_audit] Running Module 9: OP_ROLE_COVERAGE")
    if op_roles_source and catalog_rows:
        audit_op_roles(op_roles_source, catalog_rows)
    elif not op_roles_source:
        emit(
            "HIGH",
            "aria-architect",
            "OP_ROLE_COVERAGE",
            "op_roles.py",
            "op_roles.py not found — cannot verify OpRole coverage.",
            "Create op_roles.py.",
        )

    print("[aria_audit] Running Module 10: CONTEXT_RULE_COVERAGE")
    if context_rules_source and catalog_rows:
        audit_context_rules(context_rules_source, catalog_rows)
    elif not context_rules_source:
        emit(
            "HIGH",
            "aria-architect",
            "CONTEXT_RULE_COVERAGE",
            "context_rules.py",
            "context_rules.py not found — cannot verify ContextRule coverage.",
            "Create context_rules.py.",
        )

    print("[aria_audit] Running Module 11: ROUTING_SYSTEM")
    if primitives_source:
        audit_routing_system(
            true_routing_source,
            context_rules_source,
            templates_source,
            primitives_source,
        )

    print("[aria_audit] Running Module 12: VALIDATOR_WIRING")
    if grammar_source:
        audit_validator_wiring(grammar_source, compiler_source)

    print("[aria_audit] Running Module 13: MOTIF_COMPOSITION")
    if motifs_source:
        audit_motif_composition(motifs_source)

    # ── Sort and output ───────────────────────────────────────────────────────
    findings.sort(key=lambda f: _SEVERITY_ORDER[f.severity])
    print(f"\n[aria_audit] Total findings: {len(findings)}")
    for sev in SEVERITIES:
        n = sum(1 for f in findings if f.severity == sev)
        print(f"  {sev}: {n}")

    # Write JSON
    json_path = Path(args.out + ".json")
    with open(json_path, "w") as fh:
        json.dump(generate_agent_json(findings), fh, indent=2, default=str)
    print(f"[aria_audit] JSON report: {json_path}")

    # Write Markdown
    md_path = Path(args.out + ".md")
    with open(md_path, "w") as fh:
        fh.write(generate_markdown_report(findings, root))
    print(f"[aria_audit] Markdown report: {md_path}")

    # Quick agent dispatch to stdout
    print("\n" + "=" * 70)
    print("AGENT DISPATCH SUMMARY")
    print("=" * 70)
    for agent in AGENTS:
        agent_findings = [f for f in findings if f.agent == agent]
        crits = sum(1 for f in agent_findings if f.severity == "CRITICAL")
        highs = sum(1 for f in agent_findings if f.severity == "HIGH")
        print(
            f"  {agent}: {len(agent_findings)} tasks ({crits} CRITICAL, {highs} HIGH)"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
