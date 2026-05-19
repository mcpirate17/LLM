"""Shared post-investigation behavioral fingerprint completion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class FingerprintCompletionResult:
    attempted: bool
    completed: bool
    source: dict[str, Any]
    error_count: int = 0


def complete_post_investigation_fingerprint(
    *,
    source: dict[str, Any],
    source_result_id: str,
    best_inv_model: Any,
    config: Any,
    device: Any,
    notebook: Any,
    logger: Any,
    catch_exceptions: tuple[type[BaseException], ...] = (
        RuntimeError,
        ValueError,
        TypeError,
    ),
) -> FingerprintCompletionResult:
    """Complete and persist a candidate fingerprint after investigation training."""
    fp_dict = source.get("_behavioral_fingerprint")
    if best_inv_model is None or fp_dict is None:
        return FingerprintCompletionResult(False, False, source)

    from ...eval.fingerprint import BehavioralFingerprint
    from ...eval.fingerprint_runtime import complete_fingerprint_post_investigation

    allowed_fields = {
        f.name for f in BehavioralFingerprint.__dataclass_fields__.values()
    }
    fp = BehavioralFingerprint(
        **{k: v for k, v in fp_dict.items() if k in allowed_fields}
    )
    if fp.fingerprint_completed_post_investigation:
        return FingerprintCompletionResult(True, True, source)

    error_count = 0
    for attempt in range(2):
        try:
            fp = complete_fingerprint_post_investigation(
                fp,
                best_inv_model,
                seq_len=min(64, config.max_seq_len),
                model_dim=config.model_dim,
                vocab_size=config.vocab_size,
                device=str(device),
            )
            if fp.fingerprint_completed_post_investigation:
                fp_dict_updated = fp.to_dict()
                source["_behavioral_fingerprint"] = fp_dict_updated
                source["novelty_confidence"] = (
                    0.9
                    if fp.quality == "full"
                    else 0.4 + (fp.analyses_succeeded * 0.1)
                    if fp.quality == "partial"
                    else 0.3
                )
                source.update(
                    notebook._behavioral_fingerprint_program_fields(
                        fp_dict_updated,
                        novelty_confidence=source["novelty_confidence"],
                    )
                )
                notebook.sync_behavioral_fingerprint_result(
                    result_id=source_result_id,
                    fp_payload=fp_dict_updated,
                    novelty_confidence=source["novelty_confidence"],
                )
                logger.info(
                    "post_investigation_fingerprint_completed: "
                    "result_id=%s novelty_score=%.4f "
                    "novelty_valid=%s cka_source=%s attempt=%d",
                    source_result_id[:12],
                    fp.novelty_score,
                    fp.novelty_valid_for_promotion,
                    fp.cka_source,
                    attempt + 1,
                )
                return FingerprintCompletionResult(True, True, source, error_count)
        except catch_exceptions as exc:
            error_count += 1
            logger.error(
                "post_investigation_fingerprint_failed: "
                "result_id=%s attempt=%d error=%s",
                source_result_id[:12],
                attempt + 1,
                str(exc),
            )

    return FingerprintCompletionResult(True, False, source, error_count)
