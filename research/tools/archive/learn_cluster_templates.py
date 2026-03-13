#!/usr/bin/env python3
"""Learn compact failure/success templates via PCA + PLS-lite clustering.

Outputs:
  - research/runtime/learning/cluster_templates.json
  - research/runtime/learning/cluster_summaries.json
  - research/runtime/learning/cluster_suggestions.json

Local-first LLM interpretation with remote fallback if configured.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


def _load_llm_config(db_path: str) -> Dict:
    cfg_path = Path(db_path).parent / "llm_config.json"
    if not cfg_path.exists():
        return {}
    try:
        return json.loads(cfg_path.read_text())
    except Exception:
        return {}


def _get_llm_backend(cfg: Dict):
    try:
        from research.scientist.llm.backend import create_backend_from_config
    except Exception:
        return None, None

    local = cfg.get("local") or {}
    remote = cfg.get("remote") or {}

    def _mk(blk: Dict):
        backend = blk.get("backend") or ""
        model = blk.get("model") or ""
        host = blk.get("host") or ""
        if not backend:
            return None
        return create_backend_from_config(backend, model=model, host=host)

    lb = _mk(local)
    if lb and lb.is_available():
        return lb, "local"
    rb = _mk(remote)
    if rb and rb.is_available():
        return rb, "remote"
    return None, None


def _get_llm_backend_from_env():
    try:
        from research.scientist.llm.backend import create_backend
    except Exception:
        return None, None
    try:
        b = create_backend(is_analyst=True)
        if b and b.is_available():
            return b, "env"
    except Exception:
        pass
    return None, None


def _get_free_vram_gb() -> float:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=2,
        )
        vals = []
        for line in out.strip().splitlines():
            line = line.strip()
            if line:
                vals.append(float(line) / 1024.0)
        return max(vals) if vals else 0.0
    except Exception:
        return 0.0


def _ollama_list_models() -> List[Dict]:
    try:
        import requests
        host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        r = requests.get(f"{host}/api/tags", timeout=3)
        if r.status_code != 200:
            return []
        return r.json().get("models", []) or []
    except Exception:
        return []


def _pick_local_model(prefer: List[str]) -> str:
    models = _ollama_list_models()
    if not models:
        return ""
    # Always honor explicit preference order if available
    names = [m.get("name", "") for m in models]
    for p in prefer:
        if p in names:
            return p
    # Otherwise pick largest model that fits in free VRAM (80% safety margin)
    free_gb = _get_free_vram_gb()
    if free_gb <= 0:
        return ""
    cap = free_gb * 0.8
    candidates = []
    for m in models:
        name = m.get("name", "")
        size_gb = (m.get("size", 0) or 0) / (1024**3)
        if size_gb <= cap:
            candidates.append((name, size_gb))
    if not candidates:
        return ""
    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates[0][0]


PREFERRED_LOCAL_MODELS = [
    "gemma2:2b",
    "qwen2.5-coder:7b-instruct",
    "qwen2.5-coder:3b",
    "hf.co/microsoft/Phi-3-mini-4k-instruct-gguf:latest",
    "hf.co/MaziyarPanahi/phi-4-GGUF:Q8_0",
    "hf.co/unsloth/GLM-4.7-Flash-REAP-23B-A3B-GGUF:Q6_K_XL",
]


def _parse_graph_ops(graph_json: str) -> Tuple[List[str], List[str]]:
    try:
        data = json.loads(graph_json)
    except Exception:
        return [], []
    nodes = data.get("nodes") or {}
    ops = []
    pairs = []
    # nodes is dict id -> node
    for nid, node in nodes.items():
        if not isinstance(node, dict):
            continue
        op = str(node.get("op_name") or "").strip()
        if op and op != "input":
            ops.append(op)
    for nid, node in nodes.items():
        if not isinstance(node, dict):
            continue
        op = str(node.get("op_name") or "").strip()
        if not op or op == "input":
            continue
        for inp in node.get("input_ids") or []:
            parent = nodes.get(str(inp)) or nodes.get(inp)
            if not isinstance(parent, dict):
                continue
            pop = str(parent.get("op_name") or "").strip()
            if pop and pop != "input":
                pairs.append(f"{pop}->{op}")
    return ops, pairs


def _build_feature_space(rows, max_ops=64, max_pairs=128):
    op_counts = Counter()
    pair_counts = Counter()
    for r in rows:
        ops, pairs = _parse_graph_ops(r["graph_json"])
        op_counts.update(ops)
        pair_counts.update(pairs)
    top_ops = [op for op, _ in op_counts.most_common(max_ops)]
    top_pairs = [p for p, _ in pair_counts.most_common(max_pairs)]
    return top_ops, top_pairs


def _vectorize(rows, top_ops, top_pairs):
    op_index = {op: i for i, op in enumerate(top_ops)}
    pair_index = {p: i for i, p in enumerate(top_pairs)}
    n = len(rows)
    m = len(top_ops) + len(top_pairs) + 4
    X = [[0.0] * m for _ in range(n)]
    y = [0.0] * n
    for i, r in enumerate(rows):
        ops, pairs = _parse_graph_ops(r["graph_json"])
        for op in ops:
            idx = op_index.get(op)
            if idx is not None:
                X[i][idx] += 1.0
        offset = len(top_ops)
        for p in pairs:
            idx = pair_index.get(p)
            if idx is not None:
                X[i][offset + idx] += 1.0
        # flags
        f0 = 1.0 if r.get("routing_mode") else 0.0
        f1 = 1.0 if r.get("compression_ratio") is not None else 0.0
        f2 = 1.0 if r.get("depth_savings_ratio") is not None else 0.0
        f3 = 1.0 if r.get("recursion_savings_ratio") is not None else 0.0
        X[i][offset + len(top_pairs) + 0] = f0
        X[i][offset + len(top_pairs) + 1] = f1
        X[i][offset + len(top_pairs) + 2] = f2
        X[i][offset + len(top_pairs) + 3] = f3

        y[i] = float(r.get("stage1_passed") or 0)
    return X, y


def _center_scale(X):
    # center only
    n = len(X)
    m = len(X[0]) if n else 0
    means = [0.0] * m
    for row in X:
        for j, v in enumerate(row):
            means[j] += v
    means = [v / max(1, n) for v in means]
    Xc = [[row[j] - means[j] for j in range(m)] for row in X]
    return Xc, means


def _pca(X, k=8):
    import numpy as np
    Xn = np.array(X, dtype=float)
    U, S, Vt = np.linalg.svd(Xn, full_matrices=False)
    comps = Vt[:k]
    scores = (U[:, :k] * S[:k])
    return scores, comps, S


def _kmeans(X, k=6, iters=20, seed=42):
    import numpy as np
    rng = random.Random(seed)
    Xn = np.array(X, dtype=float)
    n = Xn.shape[0]
    if n == 0:
        return [], []
    # init with random points
    centers = Xn[rng.sample(range(n), min(k, n))]
    for _ in range(iters):
        # assign
        dists = ((Xn[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        labels = dists.argmin(axis=1)
        # update
        new_centers = []
        for i in range(len(centers)):
            pts = Xn[labels == i]
            if len(pts) == 0:
                new_centers.append(centers[i])
            else:
                new_centers.append(pts.mean(axis=0))
        centers = np.array(new_centers)
    return labels.tolist(), centers.tolist()


def _pls1_nipals(X, y, n_components=4):
    import numpy as np
    Xn = np.array(X, dtype=float)
    yv = np.array(y, dtype=float).reshape(-1, 1)
    # center
    Xn = Xn - Xn.mean(axis=0, keepdims=True)
    yv = yv - yv.mean(axis=0, keepdims=True)
    W = []
    P = []
    Q = []
    T = []
    for _ in range(n_components):
        # NIPALS
        w = Xn.T @ yv
        w_norm = np.linalg.norm(w)
        if w_norm == 0:
            break
        w = w / w_norm
        t = Xn @ w
        q = (yv.T @ t) / (t.T @ t)
        p = (Xn.T @ t) / (t.T @ t)
        Xn = Xn - t @ p.T
        yv = yv - t * q
        W.append(w.flatten())
        P.append(p.flatten())
        q_scalar = float(q.squeeze()) if hasattr(q, "shape") else float(q)
        Q.append(q_scalar)
        T.append(t.flatten())
    return {
        "W": W,
        "P": P,
        "Q": Q,
    }


def learn_templates(db_path: str, limit: int, n_components: int, n_clusters: int, s1_only: bool = False, pls_target: str = "stage1"):
    import sqlite3
    from research.synthesis.primitives import PRIMITIVE_REGISTRY

    valid_ops = set(PRIMITIVE_REGISTRY.keys())

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    # Default to stage0_passed to exclude compilation failures and deprecated-op models
    query = """SELECT result_id, graph_json, stage1_passed, loss_ratio,
                  routing_mode, compression_ratio,
                  depth_savings_ratio, recursion_savings_ratio
           FROM program_results
           WHERE graph_json IS NOT NULL AND graph_json != ''
             AND stage0_passed = 1"""
    if s1_only:
        query += " AND stage1_passed = 1"
    query += " ORDER BY timestamp DESC LIMIT ?"
    rows = cur.execute(query, (limit,)).fetchall()
    rows = [dict(r) for r in rows]
    conn.close()

    # Filter out results containing deprecated/removed ops
    clean_rows = []
    for r in rows:
        ops, _ = _parse_graph_ops(r["graph_json"])
        if all(op in valid_ops for op in ops):
            clean_rows.append(r)
    skipped = len(rows) - len(clean_rows)
    if skipped:
        print(f"  Filtered out {skipped} results with deprecated ops")
    rows = clean_rows

    top_ops, top_pairs = _build_feature_space(rows)
    X, y = _vectorize(rows, top_ops, top_pairs)
    if pls_target == "loss_ratio":
        y = [float(r.get("loss_ratio") or 1.0) for r in rows]
    Xc, means = _center_scale(X)
    scores, comps, svals = _pca(Xc, k=n_components)
    labels, centers = _kmeans(scores, k=n_clusters)
    pls = _pls1_nipals(Xc, y, n_components=min(4, n_components))

    # Feature names
    feat_names = top_ops + top_pairs + [
        "flag_routing", "flag_compression", "flag_adaptive_depth", "flag_adaptive_recursion"
    ]

    # Cluster templates: top features by centroid magnitude
    templates = []
    if labels:
        import numpy as np
        Xn = np.array(Xc, dtype=float)
        for cid in sorted(set(labels)):
            idx = [i for i, l in enumerate(labels) if l == cid]
            if not idx:
                continue
            centroid = Xn[idx].mean(axis=0)
            top_idx = list(reversed(sorted(range(len(centroid)), key=lambda i: abs(centroid[i]))))[:12]
            top_feats = [{"feature": feat_names[i], "weight": float(centroid[i])} for i in top_idx]
            pass_rate = sum(1 for i in idx if rows[i].get("stage1_passed")) / max(1, len(idx))
            templates.append({
                "cluster_id": int(cid),
                "size": len(idx),
                "stage1_pass_rate": round(pass_rate, 3),
                "top_features": top_feats,
            })

    # PLS feature loadings (use first component weights)
    pls_load = pls["W"][0] if pls.get("W") else []
    pls_rank = []
    if pls_load is not None and len(pls_load) > 0:
        pls_rank = list(reversed(sorted(range(len(pls_load)), key=lambda i: abs(pls_load[i]))))[:20]
    pls_top = [{"feature": feat_names[i], "weight": float(pls_load[i])} for i in pls_rank]

    out_dir = Path("research/runtime/learning")
    out_dir.mkdir(parents=True, exist_ok=True)
    templates_path = out_dir / "cluster_templates.json"
    templates_payload = {
        "generated_at": time.time(),
        "n_rows": len(rows),
        "n_components": n_components,
        "n_clusters": n_clusters,
        "op_names": top_ops,
        "op_pairs": top_pairs,
        "templates": templates,
        "pls_top_features": pls_top,
    }
    templates_path.write_text(json.dumps(templates_payload, indent=2))

    return templates_payload


def summarize_with_llm(templates_payload: Dict, db_path: str) -> Dict:
    # Force local-only summaries via Ollama (no remote usage).
    try:
        from research.scientist.llm.backend import create_backend_from_config
    except Exception:
        create_backend_from_config = None

    if create_backend_from_config is None:
        return {
            "source": "heuristic",
            "summary": "No local Ollama backend available. Use templates and PLS top features to adjust op weights.",
        }

    prompt = (
        "You are Aria's local learning summarizer. Return ONLY compact JSON with fields:\n"
        "- summary: <= 280 chars\n"
        "- op_weight_suggestions: map of op_name -> multiplier (0.6 to 1.4)\n"
        "- avoid_patterns: list of op pairs like \"A->B\"\n"
        "- promote_patterns: list of op pairs like \"A->B\"\n"
        "No prose outside JSON.\n\n"
        + json.dumps(templates_payload, indent=2)
    )
    host = os.environ.get("OLLAMA_HOST", "")

    def _run_model(model: str) -> Dict:
        backend = create_backend_from_config("ollama", model=model, host=host)
        if not backend or not backend.is_available():
            return {"ok": False, "error": "backend_unavailable"}
        try:
            resp = backend.generate(
                prompt,
                system="Return compact JSON only. Do not include markdown.",
                max_tokens=220,
                temperature=0.1,
            )
            text = (resp.text or "").strip()
            if len(text) > 1400:
                text = text[:1400]
            return {
                "ok": True,
                "source": f"ollama-{model}",
                "model": getattr(resp, "model", None),
                "tokens_used": int(getattr(resp, "tokens_used", 0) or 0),
                "summary": text,
            }
        except Exception as e:
            return {"ok": False, "error": str(e)}
        finally:
            # Stop larger models after use to free VRAM
            if model and model != "gemma2:2b":
                try:
                    subprocess.run(["ollama", "stop", model], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                except Exception:
                    pass

    # Try preferred model if available; if invalid JSON, try bigger models within VRAM.
    picked = _pick_local_model(PREFERRED_LOCAL_MODELS)
    tried = []
    candidates = [picked] if picked else []
    for m in PREFERRED_LOCAL_MODELS:
        if m not in candidates:
            candidates.append(m)

    last_result = None
    for model in candidates:
        if not model or model in tried:
            continue
        tried.append(model)
        res = _run_model(model)
        last_result = res
        if not res.get("ok"):
            continue
        # accept; JSON validation happens downstream
        return res

    return {
        "source": "heuristic",
        "summary": f"No local Ollama backend available. Last error: {last_result.get('error') if last_result else 'unknown'}",
    }


def main():
    parser = argparse.ArgumentParser(description="Learn PCA/PLS cluster templates from program_results")
    parser.add_argument("--db", default="research/lab_notebook.db")
    parser.add_argument("--limit", type=int, default=1500)
    parser.add_argument("--components", type=int, default=8)
    parser.add_argument("--clusters", type=int, default=6)
    parser.add_argument("--s1-only", action="store_true", help="Use only S1 survivors")
    parser.add_argument("--pls-target", choices=["stage1", "loss_ratio"], default="stage1")
    args = parser.parse_args()

    payload = learn_templates(args.db, args.limit, args.components, args.clusters, s1_only=args.s1_only, pls_target=args.pls_target)
    summary = summarize_with_llm(payload, args.db)

    out_dir = Path("research/runtime/learning")
    out_dir.mkdir(parents=True, exist_ok=True)
    summaries_path = out_dir / "cluster_summaries.json"
    suggestions_path = out_dir / "cluster_suggestions.json"
    summaries_path.write_text(json.dumps(summary, indent=2))

    # Extract lightweight suggestions if LLM returns JSON
    suggestions = {}
    def _strip_fences(text: str) -> str:
        t = (text or "").strip()
        if t.startswith("```"):
            # remove leading fence line
            parts = t.split("\n", 1)
            if len(parts) == 2:
                t = parts[1]
        if t.endswith("```"):
            t = t[: t.rfind("```")].rstrip()
        return t

    def _extract_json(text: str) -> Dict:
        import re
        t = _strip_fences(text)
        m = re.search(r"\{.+\}", t, flags=re.S)
        if not m:
            return {}
        try:
            return json.loads(m.group(0))
        except Exception:
            return {}

    def _heuristic_suggestions(templates_payload: Dict) -> Dict:
        op_weights = {}
        avoid = []
        promote = []
        for t in templates_payload.get("templates", []):
            pass_rate = float(t.get("stage1_pass_rate") or 0.0)
            top_feats = t.get("top_features") or []
            for feat in top_feats:
                name = feat.get("feature", "")
                w = float(feat.get("weight") or 0.0)
                if "->" in name:
                    if pass_rate < 0.25 and w > 0:
                        avoid.append(name)
                    elif pass_rate > 0.6 and w > 0:
                        promote.append(name)
                else:
                    if pass_rate < 0.25 and w > 0:
                        op_weights[name] = min(op_weights.get(name, 1.0), 0.8)
                    elif pass_rate > 0.6 and w > 0:
                        op_weights[name] = max(op_weights.get(name, 1.0), 1.1)
        # cap sizes
        avoid = avoid[:20]
        promote = promote[:20]
        return {
            "summary": "Heuristic cluster guidance (LLM JSON not available).",
            "op_weight_suggestions": op_weights,
            "avoid_patterns": avoid,
            "promote_patterns": promote,
        }

    def _is_valid_suggestions(s: Dict, payload: Dict) -> bool:
        if not isinstance(s, dict):
            return False
        # Must have expected keys
        for k in ("summary", "op_weight_suggestions", "avoid_patterns", "promote_patterns"):
            if k not in s:
                return False
        if not isinstance(s.get("op_weight_suggestions"), dict):
            return False
        if not isinstance(s.get("avoid_patterns"), list) or not isinstance(s.get("promote_patterns"), list):
            return False
        # Reject placeholder content
        placeholders = {"A->B", "B->A", "0.6 to 1.4"}
        if any(p in placeholders for p in s.get("avoid_patterns", [])):
            return False
        if any(p in placeholders for p in s.get("promote_patterns", [])):
            return False
        if any(k in placeholders for k in s.get("op_weight_suggestions", {}).keys()):
            return False
        # Validate against known ops/pairs from payload
        known_ops = set(payload.get("op_names") or [])
        known_pairs = set(payload.get("op_pairs") or [])
        if known_ops:
            for op in s.get("op_weight_suggestions", {}).keys():
                if op not in known_ops:
                    return False
        if known_pairs:
            for pat in s.get("avoid_patterns", []) + s.get("promote_patterns", []):
                if pat not in known_pairs:
                    return False
        # Must include at least one actionable item
        if not s.get("op_weight_suggestions") and not s.get("avoid_patterns") and not s.get("promote_patterns"):
            return False
        return True

    def _retry_with_larger_model(payload: Dict) -> Dict:
        try:
            from research.scientist.llm.backend import create_backend_from_config
        except Exception:
            return {}
        host = os.environ.get("OLLAMA_HOST", "")
        retry_prompt = (
            "Return ONLY JSON. No code fences. No prose. "
            "Use real op names/pairs from the payload. Do not use placeholders like A->B. "
            "Fields: summary (<=280 chars), op_weight_suggestions, avoid_patterns, promote_patterns.\n\n"
            + json.dumps(payload, indent=2)
        )
        # Prefer larger models after gemma2:2b
        for model in PREFERRED_LOCAL_MODELS[1:]:
            backend = create_backend_from_config("ollama", model=model, host=host)
            if not backend or not backend.is_available():
                continue
            try:
                resp = backend.generate(
                    retry_prompt,
                    system="Return compact JSON only. Do not include markdown.",
                    max_tokens=220,
                    temperature=0.1,
                )
                suggestions = _extract_json(resp.text or "")
                if _is_valid_suggestions(suggestions, payload):
                    return suggestions
            except Exception:
                continue
        return {}

    suggestions = _extract_json(summary.get("summary", ""))
    if not suggestions or not _is_valid_suggestions(suggestions, payload):
        # Retry with larger local models if LLM returned placeholders/invalid JSON
        if summary.get("source") and summary.get("source") != "heuristic":
            suggestions = _retry_with_larger_model(payload)
        if not suggestions or not _is_valid_suggestions(suggestions, payload):
            suggestions = _heuristic_suggestions(payload)
    suggestions_path.write_text(json.dumps(suggestions, indent=2))

    print("Learned cluster templates")
    print(f"  templates: research/runtime/learning/cluster_templates.json")
    print(f"  summaries: research/runtime/learning/cluster_summaries.json")
    print(f"  suggestions: research/runtime/learning/cluster_suggestions.json")


if __name__ == "__main__":
    main()
