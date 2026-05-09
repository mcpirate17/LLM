"""Audit local Ollama models against text-form reference probes.

This tool is intentionally separate from the PyTorch probe stack. Ollama GGUF
models are useful as mature reference models for probe stability, but they are
not directly trainable by the existing graph-model probe code. The audit uses
deterministic generation to check whether a local reference model can solve the
same families of tasks: associative recall, induction copy, entity binding,
binding multislot, and BLiMP-like grammar choice.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import shutil
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, median
from typing import Any


from research.defaults import RUNTIME_DIR_ABS

DEFAULT_MODEL = "qwen3.5:0.8b"
DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434"
DEFAULT_OUT_ROOT = RUNTIME_DIR_ABS / "ollama_reference_probe_audit"
DEFAULT_MODEL_COPY_DIR = RUNTIME_DIR_ABS / "ollama_models"


@dataclass(frozen=True, slots=True)
class ProbeItem:
    task: str
    seed: int
    item_index: int
    prompt: str
    expected: tuple[str, ...]
    candidates: tuple[str, ...]
    score_mode: str = "single"


@dataclass(slots=True)
class ProbeResult:
    model: str
    task: str
    seed: int
    item_index: int
    expected: str
    response: str
    correct: float
    slot_acc: float
    latency_ms: float
    prompt_eval_count: int | None
    eval_count: int | None
    total_duration_ms: float | None
    done_reason: str | None
    thinking_tail: str


def _slug_model_name(model: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", model).strip("-")


def _now_stamp() -> str:
    return time.strftime("%Y%m%dT%H%M%S", time.gmtime())


def _run_ollama_show(model: str, *extra: str) -> str:
    proc = subprocess.run(
        ["ollama", "show", model, *extra],
        check=True,
        text=True,
        capture_output=True,
    )
    return proc.stdout


def _parse_blob_path(modelfile_text: str) -> Path | None:
    for line in modelfile_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("FROM "):
            raw = stripped.removeprefix("FROM ").strip()
            if raw.startswith("/") and Path(raw).exists():
                return Path(raw)
    return None


def copy_ollama_blob(model: str, dest_dir: Path) -> dict[str, Any]:
    modelfile = _run_ollama_show(model, "--modelfile")
    blob_path = _parse_blob_path(modelfile)
    if blob_path is None:
        return {
            "copied": False,
            "source_path": None,
            "dest_path": None,
            "error": "could not locate local blob path from ollama modelfile",
            "modelfile": modelfile,
        }

    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"{_slug_model_name(model)}.gguf"
    source_size = blob_path.stat().st_size
    if dest.exists() and dest.stat().st_size == source_size:
        copied = False
    else:
        shutil.copy2(blob_path, dest)
        copied = True
    return {
        "copied": copied,
        "source_path": str(blob_path),
        "source_size_bytes": source_size,
        "dest_path": str(dest),
        "dest_size_bytes": dest.stat().st_size,
        "modelfile": modelfile,
    }


def save_ollama_metadata(model: str, out_dir: Path) -> dict[str, Any]:
    metadata: dict[str, Any] = {"model": model}
    try:
        modelfile = _run_ollama_show(model, "--modelfile")
        metadata["modelfile_path"] = str(out_dir / "modelfile.txt")
        (out_dir / "modelfile.txt").write_text(modelfile, encoding="utf-8")
    except (OSError, subprocess.CalledProcessError) as exc:
        metadata["modelfile_error"] = str(exc)
    try:
        verbose = _run_ollama_show(model, "--verbose")
        metadata["verbose_path"] = str(out_dir / "ollama_show_verbose.txt")
        (out_dir / "ollama_show_verbose.txt").write_text(verbose, encoding="utf-8")
        metadata.update(_parse_verbose_summary(verbose))
    except (OSError, subprocess.CalledProcessError) as exc:
        metadata["verbose_error"] = str(exc)
    return metadata


def _parse_verbose_summary(text: str) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("architecture"):
            summary["architecture"] = stripped.split(maxsplit=1)[-1]
        elif stripped.startswith("parameters"):
            summary["parameters"] = stripped.split(maxsplit=1)[-1]
        elif stripped.startswith("quantization"):
            summary["quantization"] = stripped.split(maxsplit=1)[-1]
        elif stripped.startswith("context length"):
            summary["context_length"] = stripped.split(maxsplit=2)[-1]
        elif stripped.startswith("embedding length"):
            summary["embedding_length"] = stripped.split(maxsplit=2)[-1]
    return summary


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def _token_present(response: str, token: str) -> bool:
    return bool(
        re.search(
            rf"(?<![a-z0-9]){re.escape(_norm(token))}(?![a-z0-9])", _norm(response)
        )
    )


def score_response(item: ProbeItem, response: str) -> tuple[float, float]:
    if item.score_mode == "slots":
        hits = sum(
            1 for expected in item.expected if _token_present(response, expected)
        )
        slot_acc = hits / max(1, len(item.expected))
        return float(slot_acc == 1.0), slot_acc

    first_candidate = None
    response_norm = _norm(response)
    for candidate in item.candidates:
        candidate_norm = _norm(candidate)
        match = re.search(
            rf"(?<![a-z0-9]){re.escape(candidate_norm)}(?![a-z0-9])", response_norm
        )
        if match is not None:
            if first_candidate is None or match.start() < first_candidate[0]:
                first_candidate = (match.start(), candidate)
    if first_candidate is not None:
        correct = float(_norm(first_candidate[1]) == _norm(item.expected[0]))
        return correct, correct
    correct = float(_token_present(response, item.expected[0]))
    return correct, correct


def _rng(seed: int, item_index: int, offset: int) -> random.Random:
    return random.Random(seed * 100_003 + item_index * 997 + offset)


PSEUDO_KEYS = (
    "dax",
    "mivo",
    "sorek",
    "luma",
    "narin",
    "tavo",
    "bek",
    "rindle",
    "zoma",
    "pelu",
    "kesh",
    "vondo",
    "haska",
    "nubo",
    "tirn",
    "calo",
)
PSEUDO_VALUES = (
    "amber",
    "cobalt",
    "ivory",
    "saffron",
    "violet",
    "silver",
    "crimson",
    "olive",
    "teal",
    "indigo",
    "pearl",
    "bronze",
    "coral",
    "ochre",
    "white",
    "black",
)
NAMES = ("Ada", "Ben", "Cora", "Dina", "Eli", "Faye", "Gus", "Hana")
OBJECTS = ("mug", "key", "coin", "map", "ring", "book", "shell", "lamp")


def build_associative_recall(seed: int, item_index: int) -> ProbeItem:
    rng = _rng(seed, item_index, 11)
    pairs = list(zip(rng.sample(PSEUDO_KEYS, 7), rng.sample(PSEUDO_VALUES, 7)))
    query_key, answer = rng.choice(pairs)
    table = "\n".join(f"{k} -> {v}" for k, v in pairs)
    prompt = (
        "Use only this lookup table.\n"
        f"{table}\n"
        f"Question: what value is mapped from {query_key}?\n"
        "Answer with only the value word."
    )
    return ProbeItem(
        task="associative_recall",
        seed=seed,
        item_index=item_index,
        prompt=prompt,
        expected=(answer,),
        candidates=tuple(v for _, v in pairs),
    )


def build_induction_copy(seed: int, item_index: int) -> ProbeItem:
    rng = _rng(seed, item_index, 23)
    markers = rng.sample(PSEUDO_KEYS, 6)
    followers = rng.sample(PSEUDO_VALUES, 6)
    pairs = list(zip(markers, followers))
    query_marker, answer = rng.choice(pairs)
    fragments = [f"{marker} {follower}" for marker, follower in pairs]
    rng.shuffle(fragments)
    prompt = (
        "In the sequence, each marker is followed by its continuation. "
        "Copy the continuation that followed the repeated marker.\n"
        f"Sequence: {' ; '.join(fragments)} ; {query_marker}\n"
        f"What word comes after {query_marker}?\n"
        "Answer with only one word."
    )
    return ProbeItem(
        task="induction_copy",
        seed=seed,
        item_index=item_index,
        prompt=prompt,
        expected=(answer,),
        candidates=tuple(followers),
    )


def build_entity_binding(seed: int, item_index: int) -> ProbeItem:
    rng = _rng(seed, item_index, 37)
    names = rng.sample(NAMES, 5)
    colors = rng.sample(PSEUDO_VALUES, 5)
    objects = rng.sample(OBJECTS, 5)
    facts = [
        f"{name} carries the {color} {obj}"
        for name, color, obj in zip(names, colors, objects)
    ]
    query_i = rng.randrange(len(names))
    prompt = (
        "Read the facts and answer the binding question.\n"
        f"Facts: {'. '.join(facts)}.\n"
        f"Question: what color is the {objects[query_i]} that {names[query_i]} carries?\n"
        "Answer with only the color word."
    )
    return ProbeItem(
        task="entity_binding",
        seed=seed,
        item_index=item_index,
        prompt=prompt,
        expected=(colors[query_i],),
        candidates=tuple(colors),
    )


def build_binding_multislot(seed: int, item_index: int) -> ProbeItem:
    rng = _rng(seed, item_index, 41)
    names = rng.sample(NAMES, 4)
    colors = rng.sample(PSEUDO_VALUES, 4)
    objects = rng.sample(OBJECTS, 4)
    facts = [
        f"{name} has color {color} and object {obj}"
        for name, color, obj in zip(names, colors, objects)
    ]
    slots = [
        (f"{names[0]} color", colors[0]),
        (f"{names[1]} object", objects[1]),
        (f"{names[2]} color", colors[2]),
        (f"{names[3]} object", objects[3]),
    ]
    prompt = (
        "Fill every requested blank from the facts.\n"
        f"Facts: {'. '.join(facts)}.\n"
        "Return compact JSON with keys 1,2,3,4 and only the answer words.\n"
        "Requests: "
        + "; ".join(f"{i + 1}: {label}" for i, (label, _answer) in enumerate(slots))
        + "."
    )
    return ProbeItem(
        task="binding_multislot",
        seed=seed,
        item_index=item_index,
        prompt=prompt,
        expected=tuple(answer for _label, answer in slots),
        candidates=tuple(colors + objects),
        score_mode="slots",
    )


def build_blimp_choice(seed: int, item_index: int) -> ProbeItem:
    rng = _rng(seed, item_index, 53)
    cases = [
        ("The keys are on the table.", "The keys is on the table."),
        (
            "The child near the doors is smiling.",
            "The child near the doors are smiling.",
        ),
        ("Those books have fallen.", "Those books has fallen."),
        ("Each of the pilots was ready.", "Each of the pilots were ready."),
        (
            "The woman who saw the dancers was laughing.",
            "The woman who saw the dancers were laughing.",
        ),
        ("These maps show the route.", "These maps shows the route."),
    ]
    good, bad = rng.choice(cases)
    if rng.random() < 0.5:
        option_a, option_b, answer = good, bad, "A"
    else:
        option_a, option_b, answer = bad, good, "B"
    prompt = (
        "Choose the grammatical sentence.\n"
        f"A: {option_a}\n"
        f"B: {option_b}\n"
        "Answer with only A or B."
    )
    return ProbeItem(
        task="blimp_choice",
        seed=seed,
        item_index=item_index,
        prompt=prompt,
        expected=(answer,),
        candidates=("A", "B"),
    )


BUILDERS = (
    build_associative_recall,
    build_induction_copy,
    build_entity_binding,
    build_binding_multislot,
    build_blimp_choice,
)


def build_items(seeds: list[int], samples_per_task: int) -> list[ProbeItem]:
    items: list[ProbeItem] = []
    for seed in seeds:
        for item_index in range(samples_per_task):
            for builder in BUILDERS:
                items.append(builder(seed, item_index))
    return items


def query_ollama(
    *,
    base_url: str,
    model: str,
    prompt: str,
    seed: int,
    timeout_s: float,
    num_predict: int,
) -> dict[str, Any]:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0,
            "top_k": 1,
            "top_p": 0.1,
            "seed": int(seed),
            "num_predict": int(num_predict),
        },
    }
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/api/generate",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        return json.loads(response.read().decode("utf-8"))


def list_local_ollama_models(base_url: str, timeout_s: float) -> list[dict[str, Any]]:
    request = urllib.request.Request(f"{base_url.rstrip('/')}/api/tags", method="GET")
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        payload = json.loads(response.read().decode("utf-8"))
    models = payload.get("models") or []
    if not isinstance(models, list):
        return []
    return [model for model in models if isinstance(model, dict) and model.get("name")]


def run_audit(
    *,
    model: str,
    base_url: str,
    seeds: list[int],
    samples_per_task: int,
    timeout_s: float,
    num_predict: int,
) -> list[ProbeResult]:
    results: list[ProbeResult] = []
    for item in build_items(seeds, samples_per_task):
        start = time.perf_counter()
        try:
            payload = query_ollama(
                base_url=base_url,
                model=model,
                prompt=item.prompt,
                seed=item.seed,
                timeout_s=timeout_s,
                num_predict=num_predict,
            )
            response = str(payload.get("response") or "")
            thinking = str(payload.get("thinking") or "")
            done_reason = str(payload.get("done_reason") or "") or None
            prompt_eval_count = _maybe_int(payload.get("prompt_eval_count"))
            eval_count = _maybe_int(payload.get("eval_count"))
            total_duration_ms = _duration_ns_to_ms(payload.get("total_duration"))
        except (OSError, urllib.error.URLError, TimeoutError) as exc:
            response = f"ERROR: {exc}"
            thinking = ""
            done_reason = "error"
            prompt_eval_count = None
            eval_count = None
            total_duration_ms = None
        latency_ms = (time.perf_counter() - start) * 1000.0
        correct, slot_acc = score_response(item, response)
        results.append(
            ProbeResult(
                model=model,
                task=item.task,
                seed=item.seed,
                item_index=item.item_index,
                expected=json.dumps(item.expected),
                response=response.strip(),
                correct=correct,
                slot_acc=slot_acc,
                latency_ms=latency_ms,
                prompt_eval_count=prompt_eval_count,
                eval_count=eval_count,
                total_duration_ms=total_duration_ms,
                done_reason=done_reason,
                thinking_tail=thinking[-500:],
            )
        )
    return results


def _maybe_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _duration_ns_to_ms(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value) / 1_000_000.0
    except (TypeError, ValueError):
        return None


def write_results(
    results: list[ProbeResult], out_dir: Path, metadata: dict[str, Any]
) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [asdict(result) for result in results]
    results_csv = out_dir / "probe_results.csv"
    with results_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)

    summary_rows = summarize_results(results)
    summary_csv = out_dir / "probe_summary.csv"
    with summary_csv.open("w", encoding="utf-8", newline="") as fh:
        fieldnames = [
            "task",
            "n",
            "mean_correct",
            "mean_slot_acc",
            "median_latency_ms",
            "mean_prompt_eval_count",
            "mean_eval_count",
        ]
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    model_task_summary_csv = out_dir / "model_task_summary.csv"
    with model_task_summary_csv.open("w", encoding="utf-8", newline="") as fh:
        fieldnames = [
            "model",
            "task",
            "n",
            "mean_correct",
            "mean_slot_acc",
            "median_latency_ms",
        ]
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summarize_results_by_model(results))

    summary_json = out_dir / "summary.json"
    summary_json.write_text(
        json.dumps(
            {
                "metadata": metadata,
                "summary": summary_rows,
                "model_task_summary": summarize_results_by_model(results),
                "n_results": len(results),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "results_csv": results_csv,
        "summary_csv": summary_csv,
        "model_task_summary_csv": model_task_summary_csv,
        "summary_json": summary_json,
    }


def summarize_results(results: list[ProbeResult]) -> list[dict[str, Any]]:
    by_task: dict[str, list[ProbeResult]] = {}
    for result in results:
        by_task.setdefault(result.task, []).append(result)

    rows: list[dict[str, Any]] = []
    for task in sorted(by_task):
        task_results = by_task[task]
        prompt_counts = [
            result.prompt_eval_count
            for result in task_results
            if result.prompt_eval_count is not None
        ]
        eval_counts = [
            result.eval_count
            for result in task_results
            if result.eval_count is not None
        ]
        rows.append(
            {
                "task": task,
                "n": len(task_results),
                "mean_correct": round(
                    mean(result.correct for result in task_results), 6
                ),
                "mean_slot_acc": round(
                    mean(result.slot_acc for result in task_results), 6
                ),
                "median_latency_ms": round(
                    median(result.latency_ms for result in task_results), 3
                ),
                "mean_prompt_eval_count": round(mean(prompt_counts), 3)
                if prompt_counts
                else "",
                "mean_eval_count": round(mean(eval_counts), 3) if eval_counts else "",
            }
        )
    return rows


def summarize_results_by_model(results: list[ProbeResult]) -> list[dict[str, Any]]:
    by_key: dict[tuple[str, str], list[ProbeResult]] = {}
    for result in results:
        by_key.setdefault((result.model, result.task), []).append(result)

    rows: list[dict[str, Any]] = []
    for model, task in sorted(by_key):
        task_results = by_key[(model, task)]
        rows.append(
            {
                "model": model,
                "task": task,
                "n": len(task_results),
                "mean_correct": round(
                    mean(result.correct for result in task_results), 6
                ),
                "mean_slot_acc": round(
                    mean(result.slot_acc for result in task_results), 6
                ),
                "median_latency_ms": round(
                    median(result.latency_ms for result in task_results), 3
                ),
            }
        )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", default=None, help="Single model alias; kept for compatibility."
    )
    parser.add_argument(
        "--models", nargs="+", default=None, help="One or more Ollama model aliases."
    )
    parser.add_argument(
        "--all-local-models",
        action="store_true",
        help="Audit every model returned by Ollama /api/tags.",
    )
    parser.add_argument(
        "--max-model-size-gb",
        type=float,
        default=None,
        help="When --all-local-models is set, skip local models larger than this many GB.",
    )
    parser.add_argument(
        "--ollama-url", default=os.environ.get("OLLAMA_URL", DEFAULT_OLLAMA_URL)
    )
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--copy-dir", type=Path, default=DEFAULT_MODEL_COPY_DIR)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--samples-per-task", type=int, default=6)
    parser.add_argument("--timeout-s", type=float, default=120.0)
    parser.add_argument("--num-predict", type=int, default=48)
    parser.add_argument("--skip-copy", action="store_true")
    return parser.parse_args()


def resolve_models(args: argparse.Namespace) -> list[str]:
    explicit = args.models or ([args.model] if args.model else None)
    if explicit:
        return list(dict.fromkeys(str(model) for model in explicit))
    if not args.all_local_models:
        return [DEFAULT_MODEL]

    local = list_local_ollama_models(
        args.ollama_url, timeout_s=min(float(args.timeout_s), 15.0)
    )
    resolved: list[str] = []
    for item in local:
        size = item.get("size")
        if args.max_model_size_gb is not None and size is not None:
            try:
                if float(size) > float(args.max_model_size_gb) * 1_000_000_000:
                    continue
            except (TypeError, ValueError):
                pass
        resolved.append(str(item["name"]))
    return sorted(dict.fromkeys(resolved))


def main() -> int:
    args = parse_args()
    models = resolve_models(args)
    if not models:
        raise SystemExit("no Ollama models selected")

    run_stamp = _now_stamp()
    root_out_dir = args.out_root / f"ollama_reference_{run_stamp}"
    root_out_dir.mkdir(parents=True, exist_ok=True)

    combined_results: list[ProbeResult] = []
    combined_metadata: dict[str, Any] = {"models": {}}

    for model in models:
        out_dir = root_out_dir / _slug_model_name(model)
        out_dir.mkdir(parents=True, exist_ok=True)

        metadata = save_ollama_metadata(model, out_dir)
        if not args.skip_copy:
            try:
                metadata["model_copy"] = copy_ollama_blob(model, args.copy_dir)
            except (OSError, subprocess.CalledProcessError) as exc:
                metadata["model_copy"] = {"copied": False, "error": str(exc)}

        results = run_audit(
            model=model,
            base_url=args.ollama_url,
            seeds=list(args.seeds),
            samples_per_task=int(args.samples_per_task),
            timeout_s=float(args.timeout_s),
            num_predict=int(args.num_predict),
        )
        paths = write_results(results, out_dir, metadata)
        combined_results.extend(results)
        combined_metadata["models"][model] = metadata

        print(f"model {model}")
        print(f"wrote {paths['results_csv']}")
        print(f"wrote {paths['summary_csv']}")
        print(f"wrote {paths['model_task_summary_csv']}")
        print(f"wrote {paths['summary_json']}")
        for row in summarize_results(results):
            print(
                f"{row['task']}: n={row['n']} correct={row['mean_correct']:.3f} "
                f"slot={row['mean_slot_acc']:.3f} median_ms={row['median_latency_ms']:.1f}"
            )

    combined_paths = write_results(combined_results, root_out_dir, combined_metadata)
    print(f"combined results: {combined_paths['results_csv']}")
    print(f"combined summary: {combined_paths['summary_csv']}")
    print(f"combined model/task summary: {combined_paths['model_task_summary_csv']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
