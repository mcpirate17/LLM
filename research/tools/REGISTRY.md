# Tools Registry

Every tool in this directory must have an entry here. If it's not in the
registry, it gets deleted on the next cleanup pass.

## Rules

1. **Register before merging.** Any new tool must add its entry to this file
   in the same PR that introduces it.
2. **One-off scripts don't belong here.** Backfills, data repairs, and
   migration scripts should be run and discarded. If the script has a specific
   DB state it targets, it's a one-off — commit, run, delete. Git history is
   the archive.
3. **Each tool must have a clear owner category.** If you can't classify it
   below, it probably shouldn't be a tool.
4. **Review quarterly.** If a tool hasn't been run in 3 months and isn't
   wired into CI, it's a deletion candidate.

## Categories

- **pipeline**: Core pipeline tools that run regularly (exploration, registration, validation)
- **integrity**: Data/code integrity checks (can be wired into CI)
- **guardrail**: Transition guardrails with a defined expiry condition
- **infra**: Supporting infrastructure (multi-agent coordination, profiling)

---

## Active Tools

| Tool | Category | Description | Entry Point | Expiry |
|------|----------|-------------|-------------|--------|
| `explore_under_observed.py` | pipeline | Force-explore components with low observation counts | `python -m research.tools.explore_under_observed` | - |
| `register_references.py` | pipeline | Register reference architectures (GPT-2, Mamba, RWKV, RAG) to leaderboard | `python -m research --mode=register-references` | - |
| `export_cka_references.py` | pipeline | Generate CKA reference activation artifacts for novelty scoring | `python -m research.tools.export_cka_references` | - |
| `novelty_integrity_check.py` | integrity | Validate novelty pipeline consistency (reference versions, CKA sources) | imported by tests | - |
| `repair_leaderboard_tier_data.py` | integrity | Repair missing tier-specific leaderboard fields from program_results | imported by tests | - |
| `check_association_integrity.py` | integrity | Validate experiment/fingerprint/lineage DB consistency | `python -m research.tools.check_association_integrity` | - |
| `dry_language_guardrails.py` | integrity | Enforce DRY, canonical naming, no Python hotspots in dispatch | `python -m research.tools.dry_language_guardrails` | - |
| `perf_summary.py` | integrity | Report recent performance profiling artifacts | `python -m research.tools.perf_summary` | - |
| `check_cutover_gate.py` | guardrail | Block legacy compiler usage during native-runner transition | CI check | When native-runner is sole path |
| `check_no_legacy_compile.py` | guardrail | Ensure no callsites use old compile paths | CI check | When native-runner is sole path |
| `check_no_legacy_execution_paths.py` | guardrail | Ensure no legacy execution paths remain | CI check | When native-runner is sole path |
| `check_native_compile_callsites.py` | guardrail | Validate all compile callsites use native path | CI check | When native-runner is sole path |
| `hive/bus_client.py` | infra | Multi-agent message bus client | imported | When multi-agent work ends |
| `hive/ollama_bridge.py` | infra | Ollama LLM bridge for multi-agent coordination | imported | When multi-agent work ends |
| `hive/signal_broker.py` | infra | Signal broker for multi-agent coordination | imported | When multi-agent work ends |

---

## Adding a New Tool

```markdown
| `my_tool.py` | category | One-line description | how to run it | expiry or `-` |
```

If your tool is a one-off (backfill, repair, migration):
- Do NOT add it here
- Run it, verify the result, then delete the file
- The git commit that introduced it is sufficient documentation
