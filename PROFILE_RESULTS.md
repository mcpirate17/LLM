# Training Loop Profile Results

## Environment Summary

- Date: 2026-04-03
- Host: local workstation
- Python: 3.12.3
- PyTorch CUDA device: NVIDIA GeForce RTX 5090
- Training target profiled: `research.scientist.runner._micro_train` with the baseline transformer on CUDA
- Additional real-model target profiled: notebook result `ref_mamba_76ff10cd` on CUDA via `--result-id`
- Data mode profiled: `corpus` using `research/corpus/wikitext103_train.npy`
- Distributed / NCCL overlap: not exercised in this local single-GPU profile

## Tools Available vs Unavailable

- Available: `torch.profiler`, `cProfile`, `nsys`, `ncu`
- Not exercised: distributed HTA / NCCL analysis, because the profiled run was single-GPU and never entered a distributed path

## Exact Commands

```bash
./scripts/profile_train.sh --output-dir profiles/train_loop_audit --data-mode corpus --steps 8 --benchmark-repeats 1
```

```bash
./scripts/profile_train.sh --output-dir profiles/real_graph_harness \
  --result-id ref_mamba_76ff10cd --data-mode corpus --steps 12 \
  --benchmark-repeats 1 --dim 256 --layers 4 --vocab-size 8192 \
  --seq-len 128 --batch-size 8
```

```bash
nsys profile -o profiles/train_loop_audit/raw/nsys_clean --force-overwrite=true --sample=none --trace=cuda,nvtx,osrt \
  python -m research.profiling.train_loop --disable-torch-profiler \
  --output-dir profiles/train_loop_audit/nsys_clean_run --data-mode corpus --steps 6 --benchmark-repeats 1
```

```bash
ncu --set speedOfLight --target-processes all --kernel-name regex:flash_fwd_kernel --launch-count 1 \
  --export profiles/train_loop_audit/raw/ncu_flash python - <<'PY'
import torch
import torch.nn.functional as F
from research.eval.baseline import _BaselineTransformer

dev = torch.device("cuda")
model = _BaselineTransformer(8192, 128, n_layers=2).to(dev)
opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
input_ids = torch.randint(0, 8192, (8, 128), device=dev)
for _ in range(3):
    opt.zero_grad(set_to_none=True)
    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        logits = model(input_ids)
        loss = F.cross_entropy(
            logits[:, :-1].reshape(-1, logits.shape[-1]),
            input_ids[:, 1:].reshape(-1),
        )
    loss.backward()
    opt.step()
torch.cuda.synchronize()
PY
```

## Top Bottlenecks Found

1. Baseline attention block was doing avoidable work every forward pass.
   Evidence: the baseline layer recreated a causal mask each call and asked `nn.MultiheadAttention` for attention weights it immediately discarded. An isolated warm CUDA step microbenchmark on `B=8, S=128, d=128, L=2` went from `1.653 ms` before the change to `1.455 ms` after the change, a `12.0%` step-time reduction.

2. Naive first-run timings overstated input stalls because they mixed cold-start work into steady-state training.
   Evidence: an unfixed pre-audit corpus run showed a one-time `~318 ms` data stall and starvation hit on the first active step, but the guarded active-window profile shows steady-state input work is only `0.1067 ms/step`, including `0.0192 ms` average H2D copy and `0.0400 ms` average native gather.

3. Kernel launch overhead is now more important than data loading for this local setup.
   Evidence from `torch.profiler`: `cudaLaunchKernel` accounted for `3.0737 ms` self CPU time over the 8-step active window and was the top CPU-side entry. Evidence from `nsys`: `cudaLaunchKernel` was `72.7%` of total CUDA API time.

4. The steady-state input path is not worth a broad refactor yet.
   Evidence from targeted timers and `cProfile`: `sample_batch()` took `0.688 ms` total across 25 microbench iterations, while per-step active-window timing showed `data_sampling_ms avg=0.1067`.

5. Checkpoint save overhead is measurable but not on the hot path for the profiled short run.
   Evidence: the checkpoint microbench was `3.6968 ms` per `save_phase()` call.

## Evidence

- `torch.profiler` trace and summary:
  - `profiles/train_loop_audit/torch_profiler/20260403-180435/torch_trace.json`
  - `profiles/train_loop_audit/torch_profiler/20260403-180435/profile_summary.json`
- `nsys` system timeline:
  - `profiles/train_loop_audit/raw/nsys_clean.nsys-rep`
  - `profiles/train_loop_audit/raw/nsys_clean.sqlite`
- `ncu` kernel capture:
  - `profiles/train_loop_audit/raw/ncu_flash.ncu-rep`
- Run summary:
  - `profiles/train_loop_audit/summary.json`

Key measured active-window timings from `profile_summary.json`:

- `data_sampling_ms`: `0.1067 ms/step`
- `forward_pass_ms`: `0.8641 ms/step`
- `backward_pass_ms`: `1.9444 ms/step`
- `optimizer_step_ms`: `0.1777 ms/step`
- CUDA memory allocated: `55.452 MB` steady-state
- CUDA memory reserved: `218.0 MB` steady-state

Key `nsys` observations:

- `cudaLaunchKernel`: `114,397,467 ns` total, `72.7%` of CUDA API time
- Host-to-device memcpy: `1,067,032 ns` total across 93 ops
- `flash_fwd_kernel`: present, but not dominant in the short run (`89,727 ns` total in the `nsys` summary)
- `triu_tril_kernel`: still appears in the timeline, but after caching it is no longer a dominant cost

Key `ncu` observation:

- The short kernel-level capture reached `flash_fwd_kernel`, confirming the attention forward path is using the flash attention kernel on this stack. No slow custom kernel emerged that justified kernel-specific tuning before loop-level fixes.

## Fixes Applied

1. Cached the baseline transformer causal mask and set `need_weights=False` in `MultiheadAttention`.
   Files:
   - `research/eval/baseline.py`

2. Added a guarded training-loop profiling integration that writes raw JSON summaries and Chrome/Perfetto-compatible traces under `profiles/`.
   Files:
   - `research/training/profiling.py`
   - `research/scientist/runner/execution_training.py`
   - `research/scientist/runner/_types.py`

3. Added targeted input-pipeline timers for start-index sampling, native gather, pinning, and H2D copy.
   Files:
   - `research/training/data_pipeline.py`

4. Added a repeatable runnable entrypoint for local profiling and optional Nsight commands.
   Files:
   - `research/profiling/train_loop.py`
   - `scripts/profile_train.sh`

5. Fixed a real correctness bug in `_micro_train`.
   Detail: post-training binding-probe logic could access an undefined `graph` for runs without `graph_json`. The code now handles that path safely.

## Before / After Metrics

Isolated warm training-step microbenchmark on CUDA (`B=8, S=128, d=128, L=2, vocab=8192`, bf16 autocast, AdamW):

| Metric | Before | After | Delta |
| --- | ---: | ---: | ---: |
| Step time | 1.653 ms | 1.455 ms | -12.0% |
| Throughput proxy | 1.00x | 1.136x | +13.6% |

Current unprofiled warmed corpus run through `_micro_train` (`stage1_steps=24`, same model family, `B=4`, `S=128`):

| Metric | After |
| --- | ---: |
| Avg step time | 1.763 ms |
| Throughput | 270,579 tok/s |
| Train steps completed | 24 |

Current profiled active-window corpus run (`stage1_steps=8`, `B=8`, `S=128`):

| Metric | After |
| --- | ---: |
| Avg step time | 3.259 ms |
| Throughput | 284,090 tok/s |

Note: the profiled run is intentionally slower than the unprofiled run because `torch.profiler` adds overhead.

## Next Recommended Optimizations

- Do not refactor the corpus dataloader yet. The active-window evidence says it is not the bottleneck.
- If you want more speed after this, focus on reducing launch count or fusing more work in the model path. That is where both `torch.profiler` and `nsys` point.
- If your real local workload is not the baseline transformer, run the same script against that exact model path next. The profiling harness is now in place for that.
- For deeper kernel tuning, rerun `ncu` against a larger or more representative batch/sequence size. The short local run was enough to identify the kernel family, but not to justify hand-tuning the flash kernel path.

## Second Pass

- Measured `torch.compile` on the actual short `_micro_train` corpus path and did not enable it by default.
  Evidence:
  - 12-step short run: eager `4.79 ms/step`, compile `6.49 ms/step`
  - 80-step run: eager `2.69 ms/step`, compile `2.90 ms/step`
  - Conclusion: compile overhead and/or graph-break costs are not amortized well enough for this local short-run setup.

- Measured a custom SDPA-only baseline block and confirmed that lower-launch attention can help, but did not ship it because it would be a broader baseline-model rewrite than the profiler justified for this pass.
  Evidence:
  - Isolated warm step: current cached-MHA baseline `2.057 ms`, custom SDPA block `1.291 ms`
  - Conclusion: the next real ROI is still launch-count reduction in the model path, but not via a speculative broad rewrite in this pass.

- Removed the starvation detector’s forced CUDA synchronization.
  File:
  - `research/scientist/perf.py`
  Rationale:
  - The old implementation used CUDA events plus `torch.cuda.synchronize()` just to measure input wait. That is an avoidable sync in a performance diagnostic helper.
  Measured effect on the real warmed corpus loop (`stage1_steps=80`, `B=8`, `S=128`, 3 fixed-seed runs):
  - Before: `1.969 ms/step`, `481k tok/s`
  - After: `1.644 ms/step`, `586k tok/s`
  - Delta: `16.5%` lower step time

## Third Pass

- The real graph-backed model path was not kernel-bound first. It was paying an extreme host-array bridge penalty inside the native-dispatch wrappers on CUDA.
  Evidence from the pre-fix real-model profile (`ref_mamba_76ff10cd`, 12 steps, same seed/config):
  - `forward_pass_ms`: `539.67 ms/step`
  - `backward_pass_ms`: `687.54 ms/step`
  - `torch.profiler` top ops were custom autograd `_FnBackward` / `_Fn`
  - `Memcpy HtoD`: `6.03 ms` across 676 calls
  - `Memcpy DtoH`: `5.99 ms` across 724 calls
  - `cudaStreamSynchronize`: `14.19 ms` self CPU across 1449 calls
  - Root cause in code: the native bridge was converting CUDA tensors to CPU NumPy arrays and back for per-op and bound-subgraph dispatch.

- Applied fix: skip host-array native dispatch for non-CPU tensors and fall back to the existing pure PyTorch graph executor on GPU.
  Files:
  - `research/scientist/native/tensor_bridge.py`
  - `research/scientist/native/autograd.py`
  - `research/synthesis/native_bound_graph.py`
  - `research/profiling/train_loop.py`
  - `research/tests/test_native_forward_wrapper.py`
  - `research/tests/test_subgraph_dispatch.py`
  - `research/tests/test_synthesis_native_hotpaths.py`

- Measured effect on the exact same real-model profiled run (`ref_mamba_76ff10cd`, 12 steps, same seed/config):

| Metric | Before | After | Delta |
| --- | ---: | ---: | ---: |
| Avg step time | 1229.057 ms | 12.471 ms | -99.0% |
| Throughput | 821 tok/s | 46,473 tok/s | +55.6x |
| Forward pass | 539.671 ms/step | 2.504 ms/step | -99.5% |
| Backward pass | 687.535 ms/step | 8.519 ms/step | -98.8% |

- Evidence from the post-fix real-model profile:
  - `profiles/real_graph_after/torch_profiler/20260403-182344/profile_summary.json`
  - `profiles/real_graph_after/torch_profiler/20260403-182344/torch_trace.json`
  - `profiles/real_graph_harness/SUMMARY.md`
  - `profiles/real_graph_harness/summary.json`

- What remains after the fix:
  - The real model is now on the normal GPU execution path, and the next limiter is launch count rather than host transfers.
  - Post-fix `torch.profiler` still shows `cudaLaunchKernel` as the top CPU-side entry (`41.10 ms` across 6727 calls in the profiled run).
  - Steady-state real-model timings after the fix are approximately `2.50 ms` forward, `8.52 ms` backward, `0.49 ms` optimizer, and `0.44 ms` data sampling per step.
