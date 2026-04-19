# Tools Registry

Every maintained module under `research/tools` must have an entry here. If it
is active code and not registered, it is a deletion candidate on the next
cleanup pass.

## Rules

1. Register before merging.
2. One-off scripts do not belong here. Run them, verify them, then archive or delete them.
3. Shared support modules count as maintained surface too. If active scripts import them, register them.
4. Review quarterly. If a tool is not wired into active workflows, tests, or CI, remove it.

## Categories

- **pipeline**: Core pipeline tools that run regularly for exploration, registration, profiling, training, or backfill.
- **integrity**: Data/code integrity checks and reporting helpers.
- **guardrail**: Transition guardrails with a defined expiry condition.
- **infra**: Shared support infrastructure used by active tools.

## Support Modules

| Tool | Category | Description | Entry Point | Expiry |
|------|----------|-------------|-------------|--------|
| `_legacy_backfill_cli.py` | infra | Shared CLI and device-resolution plumbing for legacy backfill wrappers | imported | - |
| `_script_audit.py` | infra | Shared experiment-audit helpers for long-running scripts | imported | - |
| `_wikitext_batches.py` | infra | Shared streamed WikiText batching for force-training and screening | imported | - |
| `hive/bus_client.py` | infra | Multi-agent message bus client | imported | When multi-agent work ends |
| `hive/ollama_bridge.py` | infra | Ollama bridge for multi-agent coordination | imported | When multi-agent work ends |
| `hive/signal_broker.py` | infra | Signal broker for multi-agent coordination | imported | When multi-agent work ends |

## Active Tools

| Tool | Category | Description | Entry Point | Expiry |
|------|----------|-------------|-------------|--------|
| `attention_template_backfill.py` | pipeline | Managed attention-template backfill campaign runner | `python -m research.tools.attention_template_backfill` | - |
| `backfill.py` | pipeline | Unified probe backfill and rescoring entrypoint | `python -m research.tools.backfill` | - |
| `backfill_binding.py` | pipeline | Legacy binding backfill CLI wrapper over canonical backfill | `python -m research.tools.backfill_binding` | Remove after callers switch to `backfill.py` |
| `backfill_stats.py` | pipeline | Rebuild template, op, and motif stats from deduped corpus plus notebook state | `python -m research.tools.backfill_stats` | - |
| `backfill_templates.py` | pipeline | Targeted template backfill runner using the full screening pipeline | `python -m research.tools.backfill_templates` | - |
| `backpopulate_screening_metrics.py` | pipeline | In-place replay/backpopulate for missing screening and probe metrics | `python -m research.tools.backpopulate_screening_metrics` | - |
| `eval_templates.py` | pipeline | Extended template evaluation runner with training and probes | `python -m research.tools.eval_templates` | - |
| `exact_graph_replay.py` | pipeline | Replay exact stored graphs through the current pipeline for validation and debugging | `python -m research.tools.exact_graph_replay` | - |
| `explore_under_observed.py` | pipeline | Force-explore components with low observation counts | `python -m research.tools.explore_under_observed` | - |
| `export_cka_references.py` | pipeline | Generate CKA reference activation artifacts for novelty scoring | `python -m research.tools.export_cka_references` | - |
| `profile_component_scaffolds.py` | pipeline | Profile scaffold-family operator coverage and screening behavior | `python -m research.tools.profile_component_scaffolds` | - |
| `profile_screening_hotpaths.py` | pipeline | Profile screening runtime hot paths and aggregate timing evidence | `python -m research.tools.profile_screening_hotpaths` | - |
| `profile_templates.py` | pipeline | Profile template screening performance and failure modes | `python -m research.tools.profile_templates` | - |
| `recompute_screening_metrics.py` | pipeline | Recompute persisted screening metrics from canonical sources | `python -m research.tools.recompute_screening_metrics` | - |
| `register_references.py` | pipeline | Register reference architectures to the leaderboard | `python -m research --mode=register-references` | - |
| `rescore_all_v7.py` | pipeline | Rescore the full leaderboard through the canonical composite implementation | `python -m research.tools.rescore_all_v7` | Remove after callers switch to `backfill.py --probe rescore` |
| `run_binding_pilot.py` | pipeline | Run staged binding-pilot backpopulation campaigns | `python -m research.tools.run_binding_pilot` | - |
| `run_probe_backfill.py` | pipeline | Generic concurrent probe backfill runner for post-train targets | `python -m research.tools.run_probe_backfill` | - |
| `run_s1_backpopulate.py` | pipeline | Run guarded S1 backpopulate campaigns over selected cohorts | `python -m research.tools.run_s1_backpopulate` | - |
| `screen_template.py` | pipeline | Force-screen a single template through screening and investigation stages | `python -m research.tools.screen_template` | - |
| `targeted_backfill.py` | pipeline | Drive targeted backfill campaigns for selected result sets | `python -m research.tools.targeted_backfill` | - |
| `train_predictors.py` | pipeline | Train intelligence-layer predictors and emit runtime metrics reports | `python -m research.tools.train_predictors` | - |
| `train_template_wikitext.py` | pipeline | Force-train templates on WikiText with shared streamed batching | `python -m research.tools.train_template_wikitext` | - |
| `dry_language_guardrails.py` | integrity | Enforce DRY, canonical naming, and no Python hotspots in dispatch | `python -m research.tools.dry_language_guardrails` | - |
| `novelty_integrity_check.py` | integrity | Validate novelty pipeline consistency | imported by tests | - |
| `perf_summary.py` | integrity | Report recent performance profiling artifacts | `python -m research.tools.perf_summary` | - |
| `repair_leaderboard_tier_data.py` | integrity | Repair missing tier-specific leaderboard fields from `program_results` | imported by tests | - |
| `vulture_whitelist.py` | integrity | Vulture allowlist for intentionally dynamic or API-required symbols | referenced by tooling | - |
| `check_cutover_gate.py` | guardrail | Block legacy compiler usage during native-runner transition | CI check | When native-runner is sole path |
| `check_native_compile_callsites.py` | guardrail | Validate all compile callsites use the native path | CI check | When native-runner is sole path |
| `check_no_legacy_compile.py` | guardrail | Ensure no callsites use old compile paths | CI check | When native-runner is sole path |
| `check_no_legacy_execution_paths.py` | guardrail | Ensure no legacy execution paths remain | CI check | When native-runner is sole path |

## Adding a New Tool

```markdown
`<tool.py>` | category | One-line description | how to run it | expiry or `-`
```

If a script is a one-off backfill, repair, or migration:
- Do not register it.
- Run it, verify the result, then archive or delete it.
- Git history is the documentation.
