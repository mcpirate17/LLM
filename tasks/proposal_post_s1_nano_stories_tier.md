# Proposal — post-S1 / pre-investigation "nano-stories" scoring tier

**Proposed by:** Tim, 2026-05-02
**Status:** Approved in principle, not yet built
**Sequence:** Build after current full-DB S0.5 backfill completes (ETA ~7.5h from now)

## The gap this closes

Current pipeline:

| tier | what it tests | budget | signal? |
|---|---|---|---|
| S1 / screening | "did it learn language at all?" (wikitext PPL @ 750 steps) | full 50K vocab | noisy — many architectures stuck near baseline |
| controlled-lang S0.5/S1.0/Inv (just shipped) | "can it bind tokens?" — abstract integer IDs labeled noun/verb/adj | tiny, 40 train steps, 120-300 vocab | clean — but uses arbitrary IDs, no semantics |
| Investigation | iv2/bv2/erf/etc — full deep probes | 2500 steps + multi-second eval suite | expensive, only top candidates |

**Missing**: a probe that asks *"can the architecture learn real, short, coherent linguistic phrases?"* — at a budget every candidate can pay.

Wikitext PPL at 750 steps doesn't answer this cleanly: too much vocabulary, too much variance, no per-phrase semantic check. Controlled-lang uses abstract IDs (no real-word semantics). The new tier sits between.

## Spec

| dimension | value | rationale |
|---|---|---|
| Corpus source | TinyStories filtered to top-5K vocab, or curated synthetic | TinyStories is already a known-tractable distribution at nano scale |
| Vocab | **3,000 BPE tokens** (tiktoken cl100k_base) | sweet spot in Tim's 2-5K range; small enough to learn, big enough for real grammar |
| Phrase length | **5-15 tokens** | real short phrases |
| Train steps | **1,000** | 10× the controlled-lang probes; well below investigation's 2,500 |
| Per-fingerprint cost | **~30s** (10s train + 20s eval suite) | matches user's ask |
| Full-DB backfill cost | **~65 hours** for 7,800 rows | heavy but cleaner signal than wikitext PPL |

## Eval components (4-part suite per fingerprint)

| component | what it measures | scoring |
|---|---|---|
| **held-out PPL** | did it learn the corpus distribution? | lower better, anchor at cohort median |
| **cloze accuracy** | "the dog ___ the bone" → predicts "ate"/"chewed" | top-K accuracy on masked-target |
| **word-order acc** | model prefers `the cat sleeps` over `sleeps cat the` | log-prob comparison |
| **rare-pair association** | model assigns higher prob to seen-in-train pairs vs distractors | retrieval-style |

Final score = weighted sum, anchored at cohort medians.

## Why this could replace several current metrics

If calibrated cleanly, candidate to retire:
- **HellaSwag at screening** (current ρ=0.088, near-noise per audit)
- **BLiMP at screening** (current ρ=0.019, pure noise; already at 5pt floor)
- Possibly even **TinyStories PPL** (current ρ=0.408 — works, but this would be more interpretable)

Should NOT replace:
- **diagnostic_score** (ρ=0.473, abstract reasoning — different signal)
- **controlled-lang ladder** (architectural binding test — different signal)

## Proposed scoring placement

- Tier name: `nano_stories_post_s1`
- Position: after S1 (screening), before investigation tier
- Soft gate: scoring/ranking weight only, no kill-on-fail
- Initial weight: 10-15 pts (similar magnitude to S1.0 controlled-lang)
- Bonus structure (per-component):
  - PPL: 4 pts (S-curve vs cohort median)
  - cloze: 3 pts
  - order: 3 pts
  - rare-pair: 3 pts
  - Total: 13 pts

## Build sequence

1. **Finish current S0.5 backfill first** (~7.5h ETA) — locks existing data
2. **Corpus prep** (~2h):
   - Filter TinyStories to phrases using only the 3K most-common cl100k_base tokens
   - Generate held-out test set: cloze pairs, order pairs, rare-association pairs
   - Cache as JSON/numpy files at `research/eval/nano_stories/`
3. **Probe code** (~3h):
   - `research/eval/nano_stories_eval.py` — main probe, follows nano_blimp_eval pattern
   - State-dict snapshot/restore (no `copy.deepcopy` — same fix we applied elsewhere)
   - Returns NanoStoriesResult with all 4 component accuracies + PPL
4. **Calibration** (~1h GPU + analysis):
   - Test on top-15 leaderboard at 3 vocab sizes (2K, 3K, 5K)
   - Find config where ~50% of cohort saturates ppl, ~50% saturates cloze (no single-component dominance)
5. **Schema + scoring** (~30min):
   - Add 4-5 columns to program_results (ppl, cloze, order, rare_pair, status, version)
   - Add `_V15_CONFIG` (or amend v14) with anchors + weights
6. **Backfill**:
   - Top-200 first (~1.5h GPU) — confirm calibration at scale
   - Full DB (~65h GPU, run overnight×3)
7. **Rescore + dashboard surface**

## Risks / open

- **Vocab curation is real work.** Need balanced parts-of-speech, common-frequent, no rare/proper nouns. If too narrow, all models memorize; too broad, all fail.
- **Phrase distribution matters.** If phrases are too repetitive, ppl is too easy; too varied, too hard.
- **Need to define "rare-pair"** rigorously — my hypothesis: pairs that appear ≤3× in training but >0×.
- **Calibration burden** — probably need 5-6 tuning rounds (matches my nano_blimp experience).

## Key references

- Existing controlled-lang code as starting pattern: `research/eval/nano_blimp_eval.py`, `research/eval/synthetic_association_eval.py` (codex), `research/eval/controlled_lang_probe.py`
- Existing TinyStories eval: `research/eval/tinystories_eval.py` (just trains + measures PPL — too simple for our purposes but shows the corpus loading pattern)
- BPE tokenization: `research/eval/utils.py::tokenize_string` (cl100k_base default)
- Scoring audit context: `tasks/scoring_audit_2026-05-02.md`
- Controlled-lang ladder context: `tasks/session_scoring_overhaul_2026-05-02.md`
