# component_fab

`component_fab` is the research platform for proposing, building, grading, and promoting novel neural-network components. It is intentionally separate from the main `research/synthesis` runtime: the fab loop can explore candidate lanes and blocks, record evidence, and surface promoted components without mutating production synthesis behavior.

## Operating model

The platform loop is:

1. Enumerate proposal specs from anchors, ledger feedback, static variants, NAS candidates, and invention mechanisms.
2. Generate a runnable `torch.nn.Module` from each `ProposalSpec`.
3. Run cheap capability screens before expensive training probes.
4. Run solo validation: smoke, shape, finite gradients, category metrics, and property cross-checks.
5. Optionally run in-context and TinyLM binding probes.
6. Write append-only JSONL evidence to `component_fab/catalog/`.
7. Replay the ledger to decide promotion, rejection, repair, and future proposal selection.
8. Expose run state through reports and the visual explainer.

## Fast local commands

```bash
# Run the fast platform contract suite.
make test-component_fab-contracts

# Run the whole component_fab test tree.
make test-component_fab

# Inspect the visual explainer locally.
python -m component_fab.viz
```

## Slower research commands

```bash
# Autonomous proposal/grade/promote loop.
python -m component_fab.tools.run_autonomous --cycles 3 --emit-run-summary

# Invention-track loop.
python -m component_fab.tools.run_invention --max-specs 4

# TinyLM hard-binding probe for a specific ledger proposal.
python -m component_fab.tools.run_lm_probe --proposal-id <id>
```

Use explicit seeds, task limits, and report output paths for publishable runs. For promotion-capable autonomous runs, prefer paired evidence such as `--paired-seeds 3` or higher when runtime allows it.

## Artifact surfaces

- `component_fab/catalog/ledger.jsonl`: append-only grade and promotion evidence.
- `component_fab/catalog/proposals.jsonl`: solo scorecards and proposal snapshots.
- `component_fab/catalog/*_run_*.json`: timestamped run reports.
- `component_fab/catalog/nas_graphs/`: cached NAS graph payloads used by graph-backed specs.

Ledger records should carry explicit schema versions. New reports should include run provenance: git SHA, dirty-tree state, argv, schema versions, and policy config versions.

## Policy/config surfaces

Versioned policy snapshots live in `component_fab/configs/`:

- `quality_v1.yml`: queue-ranking weights and budget split.
- `measured_screen_v1.yml`: descriptor-screen thresholds.
- `invention_promotion_v1.yml`: invention scoring and promotion defaults.

Treat these files as run-critical configuration. If a threshold or weighting changes, create a new versioned config instead of silently editing historical semantics.

## Boundaries

`component_fab` may import measurement and compilation utilities from `research/`, but promotion from the fab ledger should remain evidence-gated and auditable. The fab should fail loud when a proposal cannot be dispatched; silent linear stand-ins are not acceptable because they produce plausible but meaningless grades.

## Test strategy

The minimum platform contract is: generate a tiny spec, grade it without the expensive in-context probe, write/replay ledger evidence, load policy configs, and build run provenance. Slow TinyLM or Wikitext-style probes belong behind explicit test markers or manual run commands, not the default fast contract path.
