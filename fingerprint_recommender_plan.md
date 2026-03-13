# Shared Judgment Engine Plan For Designer And Continuous Research

## Summary
Upgrade the system from heuristic component suggestions to a shared, evidence-backed judgment engine used by both the Designer and continuous research mode. The implementation will consume fingerprint, topology, lineage, and insight-interaction signals; replace the currently empty toxic-op path with usable failure penalties; and restructure scoring so the same research priors can guide human-facing assembly suggestions, automated candidate generation, mutation/refinement, screening, and promotion decisions.

## Implementation Changes
### 1. Expand the research recommendation payload
- Extend `/api/analytics/recommendation-signals` in `research/scientist/api_routes/analytics_bp.py` to export composition-focused signals, not just per-op priors.
- Add fingerprint-conditioned priors:
  - aggregate prior runs by coarse fingerprint buckets derived from existing evaluator metrics: locality band, sparsity band, CKA winner, hierarchy band, and novelty band.
  - emit per-bucket top operators and top operator-pairs with support counts, S1 rate, and average novelty.
- Add topology/composition priors:
  - successful operator-pair stats, successful 3-op motif stats, and lineage-conditioned successor stats from surviving graphs.
  - include minimum support and confidence fields on every aggregate so the suggester can ignore weak evidence.
- Replace the current toxic blocklist behavior:
  - keep existing hard-fail signatures, but also emit a softer failure-risk table for signatures seen at least 3 times with elevated fail rate.
  - export both `toxic_signatures_strict` and `failure_risk_signatures_soft`.
- Normalize insight interactions for consumption:
  - export only interaction rows with stable identifiers, support counts, and a signed reward/penalty interpretation.
  - include a compact map keyed by `insight_a|insight_b` for fast lookup in the suggester.

### 2. Persist or derive the missing fingerprint/topology aggregates
- Add notebook-side aggregation helpers in `research/scientist/notebook/notebook_analytics.py` or the existing analytics package, rather than embedding SQL-heavy logic in the Flask route.
- Build helpers for:
  - fingerprint bucket assignment from stored result metrics,
  - operator-pair success tables,
  - motif success tables,
  - lineage successor frequencies by parent fingerprint,
  - soft failure-risk signature extraction.
- Use existing `program_results`, `designer_run_lineage`, and insight tables as sources; do not require a schema migration unless a missing metric prevents stable bucketing.
- If a required fingerprint feature is not currently stored in `program_results`, use the already persisted evaluation-stage metrics first; only add storage fields if those metrics are missing in practice.

### 3. Refactor Designer suggestion scoring into composable signal modules
- In `aria_designer/api/app/suggestions.py`, split `_score_adjustment` into separate scoring functions:
  - baseline heuristic score,
  - per-op prior adjustment,
  - fingerprint-bucket adjustment,
  - composition/topology adjustment,
  - insight-interaction adjustment,
  - lineage adjustment,
  - failure-risk penalty.
- Pass richer context into the suggester:
  - current workflow fingerprint surrogate derived from present nodes,
  - recent parent fingerprint if present in workflow metadata,
  - active graph motifs and operator pairs,
  - matched insight tags from prompt and current graph state.
- Change the returned suggestion evidence so each suggestion includes the contributing signals and support counts, not just generic prose.
- Preserve existing endpoint shape for compatibility, but add optional structured evidence fields:
  - `signal_breakdown`,
  - `support`,
  - `matched_priors`,
  - `risk_flags`.

### 4. Add a shared judgment engine consumed by both Designer and research automation
- Introduce a shared scoring/policy layer in the research or shared service boundary, rather than duplicating ranking logic in the Designer route and the continuous research runner.
- The shared engine should expose two modes over the same underlying signals:
  - `recommend_components(context)` for Designer suggestion/ranking,
  - `score_candidate(candidate, context)` for continuous research generation, triage, and promotion.
- Standardize the decision inputs:
  - current fingerprint bucket,
  - active operator pairs and graph motifs,
  - parent lineage fingerprint,
  - matched insight tags,
  - recent failure signatures,
  - novelty and performance context,
  - support/confidence thresholds.
- Standardize the decision outputs:
  - total score,
  - per-signal breakdown,
  - evidence/support counts,
  - confidence,
  - risk flags,
  - recommended action such as `promote`, `mutate`, `hold`, `discard`, or `suggest`.
- Keep signal extraction and policy application separate so the same analytics payload can support different frontends without forking the evidence logic.

### 5. Apply the shared judgment engine to continuous research mode
- Wire the same fingerprint/topology/lineage/insight signals into the continuous research loop so autonomous runs make better architecture decisions, not just the Designer.
- Use the shared engine at the following decision points:
  - candidate generation: bias operator and motif selection toward historically successful combinations for the active fingerprint bucket,
  - mutation/refinement: choose next edits based on lineage-conditioned successor stats and recent failure risk,
  - screening prioritization: rank which generated candidates to evaluate first using support-weighted priors plus novelty constraints,
  - promotion/retention: use composition evidence and failure-risk penalties alongside existing performance metrics,
  - exploration control: deliberately sample lower-confidence but novel combinations when exploitation evidence is saturated.
- Add policy rules so continuous research mode does not overfit historical priors:
  - preserve an exploration budget for under-sampled combinations,
  - cap any single prior's influence when support is low or stale,
  - treat hard toxic signatures as block conditions and soft failure signatures as score penalties,
  - require novelty-aware tie-breaking when two candidates have similar empirical support.
- Store the shared judgment traces for each automated decision so later analytics can explain why a candidate was generated, advanced, or rejected.

### 6. Wire actual research signals everywhere the recommender is exposed
- Keep the main FastAPI route in `aria_designer/api/app/main.py` as the source of truth for `/api/v1/aria/suggest-components`.
- Bring `aria_designer/api/app/routers/aria.py` into parity with `main.py` so it does not pass an empty `research_signals={}` placeholder.
- Ensure the continuous research runner uses the same signal fetch and shared judgment layer rather than re-implementing local heuristics.
- Ensure cache TTL and timeout behavior remain unchanged unless signal size materially degrades latency; if it does, add payload trimming in the research API, not ad hoc filtering in the Designer route or runner.

## Multi-Agent Execution Split
### Agent 1: Research aggregates
- Own the research API and notebook analytics changes.
- Deliver the expanded `recommendation-signals` payload and helper functions.
- Define stable payload keys and support/confidence semantics.

### Agent 2: Shared judgment engine
- Own the shared policy/scoring module used by both Designer and continuous research mode.
- Integrate fingerprint, topology, insight, lineage, novelty, and risk scoring into a reusable decision layer.
- Preserve backward-compatible Designer response shape while exposing richer structured evidence internally.

### Agent 3: Designer integration
- Own `suggestions.py`, route parity, UI-facing evidence fields, and response-contract tests.
- Verify Designer entrypoints use the shared judgment engine and no placeholder payload survives.
- Validate latency and payload-size impact on suggestion calls.

### Agent 4: Continuous research integration
- Own runner-side candidate generation, screening, mutation, and promotion hooks that consume the shared judgment engine.
- Add decision-trace persistence so automated runs can be audited after the fact.
- Ensure exploration safeguards remain active and autonomous search does not collapse into pure exploitation.

Coordination rule: Agent 1 publishes the payload contract first; Agent 2 defines the shared scoring interface next; Agents 3 and 4 implement against those contracts and must not rename exported keys or score fields without coordination.

## Test Plan
- Research API tests:
  - `recommendation-signals` includes new payload sections with stable keys.
  - strict toxic signatures may be empty, but soft failure-risk signatures are populated for the current notebook when evidence exists.
  - low-support rows are filtered out correctly.
- Analytics helper tests:
  - fingerprint bucketing is deterministic.
  - pair/motif aggregation returns correct support and success rates on synthetic notebook fixtures.
  - lineage successor aggregation groups by parent fingerprint correctly.
- Designer suggester tests:
  - fingerprint-aware prompts and graphs change ranking when matching bucket priors exist.
  - operator-pair and motif priors raise relevant components above generic category suggestions.
  - insight interaction penalties/rewards affect scores only when support thresholds are met.
  - failure-risk penalties suppress risky suggestions even when per-op priors are positive.
  - existing heuristic-only behavior remains as fallback when research payload is unavailable.
- Shared judgment engine tests:
  - the same input context produces consistent signal breakdowns whether called from Designer or the runner.
  - low-support priors are downweighted or ignored according to the published support/confidence policy.
  - hard toxic signatures block recommendations while soft failure signatures only penalize them.
  - novelty caps and exploration rules prevent a single historical prior from dominating every decision.
- Continuous research mode tests:
  - candidate generation shifts toward supported pairs/motifs when evidence exists.
  - mutation selection prefers lineage-supported successors over random edits in controlled fixtures.
  - promotion decisions include shared judgment evidence alongside raw performance metrics.
  - exploration budget still yields under-sampled combinations even when strong priors exist.
  - decision traces are persisted and queryable for post-run analysis.
- Integration tests:
  - `/api/v1/aria/suggest-components` in FastAPI consumes non-empty research signals.
  - router parity test confirms `routers/aria.py` no longer sends `{}`.
  - continuous research mode consumes the same shared judgment interface rather than a duplicate heuristic path.
  - response still returns top 5 suggestions and remains JSON-compatible with current UI consumers.
- Performance checks:
  - suggestion endpoint stays within current interactive latency budget.
  - continuous research throughput does not regress materially after adding shared judgment evaluation.
  - research payload size remains bounded and cacheable.

## Assumptions And Defaults
- Default plan artifact path: `/home/tim/Projects/fingerprint_recommender_plan.md`.
- No UI redesign is included; this plan changes backend decision quality, evidence, and autonomous selection policy.
- No mandatory DB migration is planned unless exploration during implementation proves fingerprint metrics required for bucketing are not already persisted.
- Support defaults:
  - per-op priors: minimum 5 uses,
  - soft failure-risk signatures: minimum 3 uses,
  - topology priors: minimum 5 supporting graphs,
  - insight interactions: minimum 2 trials.
- When evidence is absent or weak, Designer falls back to current heuristic scoring and continuous research falls back to its existing exploration heuristics rather than suppressing search entirely.
- Shared judgment should remain advisory at first rollout behind a flag so autonomous policy changes can be validated before full enforcement.
