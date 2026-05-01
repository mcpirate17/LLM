# Champion Exhaustive Ablation Plan - 2026-04-29

## Target

Primary target:

- result id: `574271ca-f37`
- fingerprint: `6d746b2317f49d`
- current composite score: `456.0`
- tier: validation

Reason:

The first targeted ablation pass tested a small set of selected signals and all
six were refuted. In plain terms: the obvious tested parts were probably not why
this graph scored well. The champion may be winning because of another part of
the graph, a structural interaction, training artifact, metric quirk, or
unmeasured template/slot effect.

## Hard Guardrails

- Do not modify the parent graph.
- Do not edit or replace the leaderboard row.
- Do not delete existing experiment data.
- All tests must write child observations, causal evidence, and provenance.
- Every generated child must point back to the parent result id and fingerprint.
- Take local and Google Drive DB backups before any large new ablation campaign.

## Goal

Determine why `574271ca-f37` scores highly by systematically testing every
reasonable causal hypothesis about the graph.

The output should answer:

1. Which individual nodes are necessary?
2. Which edges are necessary?
3. Which slot motifs are necessary?
4. Which template-level structures are necessary?
5. Which op pairs are fake correlations?
6. Which components can be removed or simplified without hurting score?
7. Which ablations improve the model, implying the parent contains harmful or
   unnecessary structure?

## Ablation Levels

### Level 1 - Node Ablation

For every non-scaffold node:

- replace with identity when shape-compatible
- replace with noop/pass-through when valid
- replace with cheap linear projection when pass-through is invalid
- replace nonlinear activation with neutral/standard activation
- remove sparse/routing/modulation behavior where possible

Record:

- node id
- operation
- input/output shape
- replacement strategy
- child fingerprint
- loss ratio
- S1 pass/fail
- metric deltas
- compile/runtime failures

### Level 2 - Edge Ablation

For every edge or dependency:

- remove edge if graph remains valid
- replace source with nearest residual/input source
- bypass intermediate node
- test whether downstream node still matters

This catches cases where an op looks important only because of how it is wired.

### Level 3 - Slot-Motif Ablation

For every template slot assignment:

- remove the motif
- replace motif with baseline motif for the slot
- replace motif with strongest supported motif from causal evidence
- move motif to adjacent compatible slot

This is the highest-priority level because prior evidence says slot context is
more meaningful than raw op adjacency.

### Level 4 - Op-Pair and Local Subgraph Ablation

For every repeated or high-signal local pattern:

- remove first op only
- remove second op only
- replace both with baseline pair
- reverse or reroute if legal
- test pair in isolation if supported by the graph builder

Examples to explicitly test:

- `rmsnorm -> adjacent_token_merge`
- `adjacent_token_merge -> rmsnorm`
- `adjacent_token_merge -> add`
- `linear_proj -> matmul`
- `matmul -> add`
- any routing/sparse/attention bridge pairs

### Level 5 - Template Skeleton Ablation

Create children that preserve most components but alter the macro skeleton:

- same ops, different template
- same template, simplified slots
- same template, remove one block/lane
- same template, reduce routing depth
- same template, reduce sparse/merge branches

This tests whether the score comes from the whole architecture shape rather than
from identifiable local components.

### Level 6 - Metric and Training Robustness

For the parent and the best/worst ablation children:

- rerun with different seed
- rerun with same seed to measure variance
- rerun with slightly longer Stage 1
- rerun screening metrics independently
- compare induction, binding, BLiMP, HellaSwag, validation loss, and loss ratio

This tests whether the parent is a stable winner or a noisy outlier.

### Level 7 - Constructive Counterfactuals

Use the strongest children to ask constructive questions:

- If removing a part improves the model, build a cleaned parent.
- If replacing a part improves the model, build a repaired parent.
- If most parts are non-causal, reduce graph complexity.
- If one slot remains consistently necessary, promote that slot rule.

Constructive children are not replacements for the parent; they become new
candidate descendants with explicit lineage.

## Scoring

Each ablation should produce:

- effect size: child best loss ratio minus parent loss ratio
- outcome:
  - supported: child worsens after ablation
  - refuted_ablation_improved: child improves after ablation
  - inconclusive: effect too small or data too weak
- confidence:
  - increases with repeated child fingerprints
  - increases with repeated S1-passing children
  - decreases for compile/runtime failures
  - decreases for one-off evidence

Do not promote rules from `n=1` rows.

Minimum credible rule:

- at least 3 evidence rows
- at least 3 child fingerprints
- supported or refuted count clearly greater than the opposite direction

## Campaign Shape

Run in stages, not as one uncontrolled blast.

### Stage A - Inventory

Extract from the parent:

- full graph JSON
- node list
- edge list
- template slots
- op pairs
- motifs
- graph depth/width
- parameter count by component if available
- all currently associated evidence

Expected output:

- `research/runtime/champion_574271ca_ablation_inventory.json`

### Stage B - Cheap Exhaustive Screen

Generate every valid single-change child.

Prioritize:

- non-scaffold nodes
- slot motifs
- routing components
- sparse components
- merge/compression components
- local op pairs already suspected by evidence

Use cheap screening first. Do not scale every child.

Expected output:

- roughly tens to hundreds of child tests, depending graph size
- all children linked to parent

### Stage C - Replication of Strong Effects

For the largest positive and negative effects:

- rerun 3-5 variants
- use different seeds where possible
- require repeated child fingerprints or controlled equivalent variants

This separates real causal effects from one lucky child.

### Stage D - Deep Validation

Only promote children to deeper validation if:

- effect size is meaningful
- repeated evidence agrees
- no obvious metric artifact
- child is simpler or more interpretable, or improves at least one important
  generalization metric

### Stage E - Report

Write a final report:

- what the champion actually depends on
- what can be removed
- what improves it
- which rules should be promoted
- which rules should be suppressed
- whether the champion is stable or likely noisy

Expected output:

- `research/runtime/champion_574271ca_exhaustive_ablation_report.md`

## Stop Conditions

Stop early if:

- all high-priority components have repeated evidence
- evidence converges on a small set of causal structures
- most new tests are duplicate/inconclusive
- runtime exceeds expected ROI
- the parent appears unstable under rerun validation

Continue if:

- new children keep improving over parent
- a specific slot/template family emerges
- refutations expose removable complexity

## Engineering Requirements

Implement this as a dedicated champion-ablation campaign, not by abusing the
generic manual ablation wrapper.

Required improvements:

- campaign-level status row that does not masquerade as a normal experiment
- child-level progress visible in dashboard
- loss/metric summaries for child ablations
- no stale wrapper rows
- resumable target list
- per-stage logs
- credible/evidence-filtered UI tables
- provenance fields suitable for future ML training

## Immediate Next Step

Build `research/tools/champion_exhaustive_ablation.py` to:

1. Load parent `574271ca-f37`.
2. Create the inventory JSON.
3. Generate valid ablation plans for every node, edge, slot, motif, and op pair.
4. Run a bounded Stage B cheap screen.
5. Record all evidence with parent provenance.
6. Produce a first report.

Do not start a large exhaustive run until the inventory and planned child count
are reviewed.
