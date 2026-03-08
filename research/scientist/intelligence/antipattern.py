"""
Anti-Pattern Extraction and Semantic Failure Analysis.

Identifies structural patterns (op-sequences) that correlate with failures
or poor performance, and generates plain-english 'Anti-Pattern Insights'
via LLM to help the synthesizer avoid these pitfalls.
"""

from __future__ import annotations

import json
import logging
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)

def _get_all_paths(nodes: dict, current_node_id: str | int, current_path: tuple, max_depth: int = 5) -> List[tuple]:
    """Recursively extract all unique paths (op-name sequences) of length up to max_depth."""
    if len(current_path) >= max_depth:
        return [current_path]
    
    # In this graph schema, nodes have 'input_ids' (the IDs of nodes feeding INTO them)
    # To traverse paths, we need the NEXT nodes (the nodes this one feeds into).
    # Since we only have 'input_ids', we have two choices:
    # 1. Reverse the graph to get 'output_ids'
    # 2. Extract paths from output back to input (reversed paths).
    # Let's use reversed paths (output to input) because failures often originate
    # from a sink that doesn't propagate gradients well.
    
    paths = [current_path]
    node_data = nodes.get(str(current_node_id))
    if not node_data:
        return paths
    
    input_ids = node_data.get("input_ids", [])
    for input_id in input_ids:
        in_node = nodes.get(str(input_id))
        if not in_node:
            continue
        op_name = in_node.get("op_name", "unknown")
        # Don't follow paths through 'input'
        if op_name == "input":
            paths.append(current_path + (op_name,))
            continue
            
        new_paths = _get_all_paths(nodes, input_id, current_path + (op_name,), max_depth)
        paths.extend(new_paths)
        
    return paths

def extract_patterns_from_graph(graph_json: str, max_depth: int = 3) -> Set[tuple]:
    """Extract unique N-gram op-sequences from the graph."""
    try:
        data = json.loads(graph_json)
        nodes = data.get("nodes", {})
        output_id = data.get("output_node_id")
        if output_id is None:
            # Try to find nodes with no consumers (approximate output)
            return set()
            
        # Start from output and work backwards
        out_node = nodes.get(str(output_id))
        if not out_node:
            return set()
            
        all_paths = _get_all_paths(nodes, output_id, (out_node.get("op_name", "output"),), max_depth)
        # Normalize: strip 'output' and 'input' if they are at the ends
        normalized = set()
        for p in all_paths:
            if len(p) < 2: continue
            normalized.add(p)
            
        return normalized
    except Exception as e:
        logger.warning("Failed to extract patterns from graph: %s", e)
        return set()

def analyze_antipatterns(nb, min_count: int = 5, min_fail_ratio: float = 0.8) -> List[dict]:
    """
    Finds op-sequences that are significantly more likely to fail than succeed.
    Returns a list of identified anti-pattern dictionaries.
    """
    logger.info("Analyzing structural anti-patterns in recent experiments...")
    
    # Query all results with graph_json
    try:
        rows = nb.conn.execute(
            """
            SELECT graph_json, stage0_passed, stage1_passed, error_type
            FROM program_results
            WHERE graph_json IS NOT NULL
            ORDER BY timestamp DESC
            LIMIT 2000
            """
        ).fetchall()
    except Exception as e:
        logger.error("Failed to query program_results for anti-pattern extraction: %s", e)
        return []

    if not rows:
        return []

    pattern_fails = Counter()
    pattern_total = Counter()
    pattern_errors = defaultdict(Counter)

    for row in rows:
        passed = (row["stage0_passed"] == 1)
        patterns = extract_patterns_from_graph(row["graph_json"])
        
        for p in patterns:
            pattern_total[p] += 1
            if not passed:
                pattern_fails[p] += 1
                if row["error_type"]:
                    pattern_errors[p][row["error_type"]] += 1

    antipatterns = []
    for p, total in pattern_total.items():
        if total < min_count:
            continue
            
        fail_count = pattern_fails[p]
        fail_ratio = fail_count / total
        
        if fail_ratio >= min_fail_ratio:
            # Common error for this pattern
            top_error = "unknown"
            if pattern_errors[p]:
                top_error = pattern_errors[p].most_common(1)[0][0]
                
            antipatterns.append({
                "sequence": p,
                "fail_ratio": fail_ratio,
                "total_occurrences": total,
                "fail_count": fail_count,
                "primary_error": top_error,
                "human_readable": " -> ".join(reversed(p)) # paths are output-to-input
            })

    # Sort by significance (count * ratio)
    antipatterns.sort(key=lambda x: x["fail_count"], reverse=True)
    return antipatterns[:15]

def summarize_antipatterns_with_llm(nb, antipatterns: List[dict]) -> List[dict]:
    """Calls LLM to synthesize semantic insights from raw anti-patterns."""
    if not antipatterns:
        return []

    from ..llm.ollama import OllamaBackend
    llm = OllamaBackend()
    
    if not llm.is_available():
        logger.debug("Ollama not available, skipping semantic anti-pattern summarization")
        return []

    results = []
    
    # Batch them for the LLM
    data_text = "\n".join([
        f"- {a['human_readable']} (Failed {a['fail_count']}/{a['total_occurrences']} times, primary error: {a['primary_error']})"
        for a in antipatterns
    ])
    
    prompt = (
        """You are a deep learning stability expert. Below are structural patterns extracted from computation graphs that consistently cause failures (Stage 0 errors) or bad loss ratios.\n\nPATTERNS (source -> sink):\n{data_text}\n\nTASK:\nIdentify the TOP 3 semantic 'Failure Modes' represented here. For each, provide:\n1. A TITLE for the anti-pattern (e.g., 'Activation Cascade Without Mixing')\n2. A 2-sentence EXPLANATION of why this pattern fails theoretically.\n"""
    )

    try:
        resp = llm.generate(prompt, max_tokens=1000, temperature=0.1)
        # Attempt to parse JSON from response
        text = resp.text.strip()
        # Find JSON boundaries if LLM added chatter
        start = text.find('[')
        end = text.rfind(']') + 1
        if start >= 0 and end > 0:
            json_str = text[start:end]
            summaries = json.loads(json_str)
            
            for s in summaries:
                # Store in insights table
                nb.conn.execute(
                    """
                    INSERT INTO insights (insight_id, timestamp, category, content, confidence, status, insight_type)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        f"antipattern_{hash(s['title']) % 1000000}_{int(time.time())}",
                        time.time(),
                        "failure_mode",
                        f"Title: {s['title']}\nExplanation: {s['explanation']}\nSuggestion: {s['suggestion']}",
                        0.85,
                        "active",
                        "antipattern"
                    )
                )
                results.append(s)
            nb.conn.commit()
    except Exception as e:
        logger.warning("Failed to summarize anti-patterns via LLM: %s", e)

    return results

import time # needed for the inline timestamp in INSERT
