# Tomorrow Task: Small LLM Probe Validation

Follow up on the champion-mode reliability issue by testing the graph family against published, externally recognizable small-LLM probes instead of relying on the current champion screening numbers.

Scope:

- Scale the candidate setup to roughly 150M-200M parameters before drawing conclusions.
- Compare against controlled GPT-2/Pythia-like and Mamba/SSM baselines under the same tokenizer, corpus, context length, optimizer, and training budget where feasible.
- Prioritize published or widely used probes/evals:
  - MQAR / Zoology-style associative recall.
  - BLiMP and BabyLM linguistic evaluations.
  - LM Evaluation Harness tasks such as LAMBADA, HellaSwag, PIQA, ARC, and Winogrande where size-appropriate.
  - RULER / needle-style tests only if the context length makes them meaningful.
  - TinyStories only if the data setup matches that regime.
- Audit the current champion probe path for CPU fallback, misleading aggregate scores, and mismatch with the DB replay path.
- Do not persist new metrics or experiment results to the lab notebook database unless explicitly approved.

Success criteria:

- Produce a reproducible dry-run script or command set for the selected published probes.
- Report graph-vs-baseline results with raw per-probe values, not only aggregate scores.
- Decide whether the induction behavior is genuinely competitive or an artifact of the current champion harness.

---

# Tomorrow TODO: 2026-05-09

## 1. Validate AR Validation Tooling Quality

- Run the new AR validation hot-path benchmark on CUDA, not just CPU:
  - `python -m research.tools.bench_ar_validation_hotpath --device cuda --batches 5000 --warmup-batches 500 --out research/runtime/ar_validation_fingerprint_sweep/ar_validation_hotpath_benchmark_cuda.json`
- Compare CPU vs CUDA benchmark artifacts and record whether batch generation is negligible relative to model forward/backward time.
- Confirm the read-only mmap warning is gone in a real AR validation/backfill run.
- Decide whether to keep `ar_validation_hotpath_benchmark_latest.json` as a checked artifact, rotate it, or move benchmarks under a dated runtime directory.

## 2. Finish AR Validation Backfill Design

- Turn the hand-built targeted chunk selection into a first-class tool mode instead of manually passing ad hoc `--result-id` lists.
- Add buckets for:
  - top validation candidates missing AR VAL
  - high AR Gate candidates missing AR VAL
  - high-scoring candidates with weak or missing binding/induction evidence
  - recently promoted or suspicious leaderboard movers
- Dry-run the selection and confirm it prints the exact rank/order before any CUDA work.
- Run a small targeted write batch first, then inspect DB fields and provenance.

## 3. Reassess Scoring After AR VAL Backfill

- Re-run scoring comparison after the new AR VAL rows are populated.
- Check whether AR Gate is acting only as a no-go gate and not over-ranking saturated candidates.
- Check whether AR VAL materially rank-orders validation candidates better than loss alone.
- Review breakthrough threshold movement before accepting any large leaderboard jumps.
- Specifically inspect cases where loss is weak but AR VAL is high, and cases where loss is strong but AR VAL is poor.

## 4. Probe Reliability And Naming Cleanup

- Audit remaining old names in code, UI labels, docs, and test fixtures:
  - `small_ar`
  - `nano_ar`
  - `controlled_lang`
  - `induction_v2`
  - `induction_v3`
- Centralize dashboard display aliases so labels like `AR VAL`, `Ind INTER`, and `Bind INTER` do not drift across components.
- Re-check dashboard build/tests after any alias cleanup.

## 5. Model-Maturity Question

- Inspect the completed 200K run for the current graph and compare:
  - PPL trajectory
  - AR Gate
  - AR VAL
  - intermediate/medium AR if available
  - induction and binding probes
- Decide whether probe inconsistency is mostly model immaturity or a weak probe design.
- If continuing the model, prefer a clearly stated objective: lower PPL, stronger AR VAL, or larger-model predictive evidence.

## 6. Medium AR / Intermediate Probe Work

- Pull in results from the other chat’s medium AR work.
- Decide whether medium AR is the missing screen between AR Gate and AR VAL.
- Run the same seed/config matrix on known references if artifacts are available:
  - GPT-2 small/reference checkpoint
  - f86a checkpoints
  - a failed/low learner
  - a high-loss but high-AR candidate
- Compare rank ordering across AR Gate, medium AR, AR VAL, and downstream larger-run strength.

## 7. External Probe Sanity Check

- Pick one published or recognizable external probe path to start:
  - MQAR/Zoology-style associative recall
  - LM Evaluation Harness small tasks
  - BLiMP/BabyLM subset
- Keep raw per-probe metrics separate from the internal composite score.
- Do not write external probe metrics into the DB until the result format and interpretation are settled.

## 8. Operational Cleanup

- Check for long-running or stale Python/CUDA jobs before starting new experiments.
- Keep runtime data, DBs, and logs intact.
- Clean only obvious local caches or generated scratch files, not research artifacts.
- Before any DB write, verify backup freshness.
