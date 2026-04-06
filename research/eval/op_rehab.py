"""
Op Rehabilitation — Test ops in isolation to distinguish intrinsically broken
ops from ops that merely got bad stats due to poor placement.

Principle: if an op compiles and runs forward on its own, it should never be
hard-excluded. Its failures came from incompatible combinations, not the op itself.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def test_op_in_isolation(
    op_name: str,
    model_dim: int = 128,
    vocab_size: int = 256,
    max_seq_len: int = 32,
    batch_size: int = 1,
    seq_len: int = 16,
    timeout_seconds: int = 5,
    device: str = "cpu",
) -> Dict[str, Any]:
    """Build a minimal graph containing a single op and test compile + forward.

    Returns dict with keys: op_name, compile_passed, forward_passed, error_message.
    """
    from ..synthesis.primitives import get_primitive, PRIMITIVE_REGISTRY
    from ..synthesis.graph import ComputationGraph
    from ..synthesis.compiler import compile_model
    from ..eval.sandbox import safe_eval

    result = {
        "op_name": op_name,
        "compile_passed": False,
        "forward_passed": False,
        "error_message": None,
    }

    if op_name not in PRIMITIVE_REGISTRY:
        result["error_message"] = f"Unknown op: {op_name}"
        return result

    prim = get_primitive(op_name)

    try:
        g = ComputationGraph(model_dim)
        inp = g.add_input()

        if prim.n_inputs == 1:
            op_id = g.add_op(op_name, [inp])
        elif prim.n_inputs == 2:
            # For binary ops, feed the same input to both ports
            op_id = g.add_op(op_name, [inp, inp])
        else:
            # n_inputs >= 3: replicate input
            op_id = g.add_op(op_name, [inp] * prim.n_inputs)

        node = g.nodes[op_id]
        if node.output_shape.dim != model_dim or not node.output_shape.is_standard:
            result["error_message"] = (
                f"Output shape mismatch: dim={node.output_shape.dim}, "
                f"is_standard={node.output_shape.is_standard}"
            )
            return result
        g.set_output(op_id)

    except Exception as e:
        result["error_message"] = f"Graph build error: {e}"
        return result

    # Compile
    try:
        model = compile_model([g], vocab_size=vocab_size, max_seq_len=max_seq_len)
        result["compile_passed"] = True
    except Exception as e:
        result["error_message"] = f"Compile error: {e}"
        return result

    # Forward pass via safe_eval
    # For rehabilitation we only care that the op doesn't crash/NaN.
    # zero_grad is expected for single-op graphs with no learnable context.
    try:
        sr = safe_eval(
            model,
            batch_size=batch_size,
            seq_len=seq_len,
            vocab_size=vocab_size,
            device=device,
            timeout_seconds=timeout_seconds,
            run_stability_probe=False,
        )
        benign_failures = {"zero_grad"}
        if sr.passed or sr.error_type in benign_failures:
            result["forward_passed"] = True
        else:
            result["error_message"] = f"safe_eval failed: {sr.error_type}: {sr.error}"
    except Exception as e:
        result["error_message"] = f"Forward error: {e}"

    return result


def rehabilitate_ops(
    notebook: Any,
    ops_to_test: Optional[List[str]] = None,
    model_dim: int = 128,
    device: str = "cpu",
    max_age_hours: float = 24.0,
) -> List[str]:
    """Test ops with 0% S1 rate in isolation to see if they're intrinsically broken.

    Returns list of op names that passed rehabilitation (compile + forward OK).
    """
    # Determine which ops to test
    if ops_to_test is None:
        rows = notebook.conn.execute(
            """SELECT op_name, n_used, n_stage1_passed
               FROM op_success_rates
               WHERE n_stage1_passed = 0 AND n_used >= 5"""
        ).fetchall()
        ops_to_test = [r[0] for r in rows]

    if not ops_to_test:
        return []

    # Skip recently tested ops
    cache = notebook.get_op_rehabilitation_cache(max_age_hours=max_age_hours)
    ops_to_test = [op for op in ops_to_test if op not in cache]

    if not ops_to_test:
        logger.info("All candidate ops already tested within %dh", max_age_hours)
        return []

    rehabilitated = []
    for op_name in ops_to_test:
        try:
            result = test_op_in_isolation(op_name, model_dim=model_dim, device=device)
            notebook.save_op_rehabilitation_result(
                op_name=op_name,
                compile_passed=result["compile_passed"],
                forward_passed=result["forward_passed"],
                error_message=result["error_message"],
                model_dim=model_dim,
            )
            if result["compile_passed"] and result["forward_passed"]:
                rehabilitated.append(op_name)
                logger.info("Op '%s' rehabilitated — works in isolation", op_name)
            else:
                logger.info(
                    "Op '%s' failed rehab: %s", op_name, result["error_message"]
                )
        except Exception as e:
            logger.warning("Rehab test for '%s' raised exception: %s", op_name, e)

    if rehabilitated:
        logger.info(
            "Rehabilitated %d/%d ops: %s",
            len(rehabilitated),
            len(ops_to_test),
            ", ".join(rehabilitated),
        )

    return rehabilitated
