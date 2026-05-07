"""Champion-mode floor metrics from training loss curves.

The helpers in this module are intentionally pure: callers provide in-memory
``(step, loss)`` data and receive serializable metric dictionaries. They do not
read or write notebook tables, runtime artifacts, or checkpoints.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import median, pstdev
from typing import Any, Iterable, Mapping, NamedTuple


CHAMPION_FLOOR_PROTOCOL_VERSION = "champion_floor_v1_500step_median"
CHAMPION_PLATEAU_WINDOW_STEPS = 500
_ABS_PLATEAU_DELTA = 0.02
_REL_PLATEAU_DELTA = 0.005
_FLOOR_BAND = 0.03


class ChampionFloorMetrics(NamedTuple):
    champion_steps_to_floor: int | None
    champion_floor_loss: float | None
    champion_floor_ppl: float | None
    champion_floor_loss_std: float | None
    champion_plateau_detected_step: int | None
    champion_plateau_window: int
    champion_floor_protocol_version: str

    def to_dict(self) -> dict[str, Any]:
        return self._asdict()


@dataclass(frozen=True)
class ChampionGpt2Baseline:
    layers: int
    protocol_version: str
    result_id: str
    steps: int
    champion_steps_to_floor: int | None
    champion_floor_loss: float | None
    champion_floor_ppl: float | None
    champion_floor_loss_std: float | None
    champion_plateau_detected_step: int | None
    champion_plateau_window: int
    wikitext_perplexity: float | None = None
    final_loss: float | None = None
    min_loss: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "champion_baseline_layers": self.layers,
            "champion_baseline_protocol_version": self.protocol_version,
            "champion_baseline_result_id": self.result_id,
            "champion_baseline_steps": self.steps,
            "champion_steps_to_floor": self.champion_steps_to_floor,
            "champion_floor_loss": self.champion_floor_loss,
            "champion_floor_ppl": self.champion_floor_ppl,
            "champion_floor_loss_std": self.champion_floor_loss_std,
            "champion_plateau_detected_step": self.champion_plateau_detected_step,
            "champion_plateau_window": self.champion_plateau_window,
            "wikitext_perplexity": self.wikitext_perplexity,
            "final_loss": self.final_loss,
            "min_loss": self.min_loss,
        }


def _finite_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _curve_point(point: Any) -> tuple[int, float] | None:
    if isinstance(point, Mapping):
        step = point.get("step")
        loss = point.get("loss")
    else:
        try:
            step, loss = point[0], point[1]
        except (TypeError, IndexError):
            return None
    step_f = _finite_float(step)
    loss_f = _finite_float(loss)
    if step_f is None or loss_f is None:
        return None
    return int(step_f), loss_f


def _normalized_curve(
    curve: Iterable[Any],
) -> list[tuple[int, float]]:
    by_step: dict[int, float] = {}
    for item in curve:
        point = _curve_point(item)
        if point is None:
            continue
        step, loss = point
        if step < 0:
            continue
        by_step[step] = loss
    return sorted(by_step.items())


def _rolling_median(
    values: list[float],
    *,
    plateau_window: int,
    step_interval: float,
) -> list[float]:
    effective_samples = max(1.0, float(plateau_window) / max(float(step_interval), 1.0))
    span = max(3, min(101, int(effective_samples // 10) * 2 + 1))
    half_span = span // 2
    smoothed = []
    for idx in range(len(values)):
        start = max(0, idx - half_span)
        end = min(len(values), idx + half_span + 1)
        smoothed.append(float(median(values[start:end])))
    return smoothed


def _plateau_delta_threshold(start_loss: float) -> float:
    return max(_ABS_PLATEAU_DELTA, abs(start_loss) * _REL_PLATEAU_DELTA)


def _first_index_at_or_after_step(
    points: list[tuple[int, float]],
    target_step: int,
) -> int | None:
    for idx, (step, _loss) in enumerate(points):
        if step >= target_step:
            return idx
    return None


def _median_step_interval(points: list[tuple[int, float]]) -> float:
    deltas = [
        float(points[idx][0] - points[idx - 1][0])
        for idx in range(1, len(points))
        if points[idx][0] > points[idx - 1][0]
    ]
    return float(median(deltas)) if deltas else 1.0


def extract_champion_floor_metrics(
    curve: Iterable[Any],
    *,
    plateau_window: int = CHAMPION_PLATEAU_WINDOW_STEPS,
    protocol_version: str = CHAMPION_FLOOR_PROTOCOL_VERSION,
) -> ChampionFloorMetrics:
    """Extract champion floor metrics from ``(step, loss)`` curve data.

    Plateau detection compares rolling-median-smoothed loss at ``step`` with the
    smoothed loss at least ``plateau_window`` steps earlier. Once a plateau is
    found, the floor entry is back-counted to the first prior step whose
    smoothed loss is inside the detected plateau band's loss range.
    """

    window = int(plateau_window)
    empty = ChampionFloorMetrics(
        champion_steps_to_floor=None,
        champion_floor_loss=None,
        champion_floor_ppl=None,
        champion_floor_loss_std=None,
        champion_plateau_detected_step=None,
        champion_plateau_window=window,
        champion_floor_protocol_version=protocol_version,
    )
    if window <= 0:
        raise ValueError("plateau_window must be positive")

    points = _normalized_curve(curve)
    if len(points) < 2 or points[-1][0] - points[0][0] < window:
        return empty

    raw_losses = [loss for _step, loss in points]
    smoothed = _rolling_median(
        raw_losses,
        plateau_window=window,
        step_interval=_median_step_interval(points),
    )
    tail_start_idx = _first_index_at_or_after_step(points, points[-1][0] - window)
    if tail_start_idx is None:
        return empty
    tail_smoothed = smoothed[tail_start_idx:]
    tail_raw = raw_losses[tail_start_idx:]
    floor_loss = sum(tail_smoothed) / len(tail_smoothed)
    floor_loss_std = pstdev(tail_raw) if len(tail_raw) > 1 else 0.0
    floor_band = max(_FLOOR_BAND, floor_loss_std)
    floor_ceiling = floor_loss + floor_band

    detected_idx: int | None = None
    start_idx_for_detection: int | None = None
    start_idx = 0
    for idx, (step, _loss) in enumerate(points):
        target_step = step - window
        if points[start_idx][0] > target_step:
            continue
        while start_idx + 1 < len(points) and points[start_idx + 1][0] <= target_step:
            start_idx += 1
        if smoothed[idx] > floor_ceiling or smoothed[start_idx] > floor_ceiling:
            continue
        improvement = smoothed[start_idx] - smoothed[idx]
        if improvement <= _plateau_delta_threshold(smoothed[start_idx]):
            detected_idx = idx
            start_idx_for_detection = start_idx
            break

    if detected_idx is None or start_idx_for_detection is None:
        return empty

    entry_idx = start_idx_for_detection
    for idx in range(0, detected_idx + 1):
        if smoothed[idx] <= floor_ceiling:
            entry_idx = idx
            break

    return ChampionFloorMetrics(
        champion_steps_to_floor=int(points[entry_idx][0]),
        champion_floor_loss=float(floor_loss),
        champion_floor_ppl=float(math.exp(floor_loss)),
        champion_floor_loss_std=float(floor_loss_std),
        champion_plateau_detected_step=int(points[detected_idx][0]),
        champion_plateau_window=window,
        champion_floor_protocol_version=protocol_version,
    )


GPT2_CHAMPION_BASELINES: dict[tuple[int, str], ChampionGpt2Baseline] = {
    (
        4,
        CHAMPION_FLOOR_PROTOCOL_VERSION,
    ): ChampionGpt2Baseline(
        layers=4,
        protocol_version=CHAMPION_FLOOR_PROTOCOL_VERSION,
        result_id="gpt2cal490d5",
        steps=40_000,
        champion_steps_to_floor=11_742,
        champion_floor_loss=5.138596884504763,
        champion_floor_ppl=170.47640234983245,
        champion_floor_loss_std=0.2468890209700839,
        champion_plateau_detected_step=12_946,
        champion_plateau_window=CHAMPION_PLATEAU_WINDOW_STEPS,
        wikitext_perplexity=225.09,
        final_loss=5.4482808113098145,
        min_loss=4.057538032531738,
    ),
    (
        6,
        CHAMPION_FLOOR_PROTOCOL_VERSION,
    ): ChampionGpt2Baseline(
        layers=6,
        protocol_version=CHAMPION_FLOOR_PROTOCOL_VERSION,
        result_id="gpt2cal87a29",
        steps=40_000,
        champion_steps_to_floor=12_593,
        champion_floor_loss=5.050924430112401,
        champion_floor_ppl=156.16676303746402,
        champion_floor_loss_std=0.2618200647512958,
        champion_plateau_detected_step=13_404,
        champion_plateau_window=CHAMPION_PLATEAU_WINDOW_STEPS,
        wikitext_perplexity=204.45,
        final_loss=5.054659843444824,
        min_loss=3.9020092487335205,
    ),
}


def lookup_gpt2_champion_baseline(
    layers: int,
    *,
    protocol_version: str = CHAMPION_FLOOR_PROTOCOL_VERSION,
    default_layers: int | None = None,
) -> ChampionGpt2Baseline:
    """Return the neutral GPT-2 champion baseline for ``layers``.

    ``default_layers`` is an explicit escape hatch for callers that want a
    known fallback rather than a hard error when a layer count has not been
    calibrated yet.
    """

    key = (int(layers), str(protocol_version))
    baseline = GPT2_CHAMPION_BASELINES.get(key)
    if baseline is not None:
        return baseline
    if default_layers is not None:
        fallback = GPT2_CHAMPION_BASELINES.get(
            (int(default_layers), str(protocol_version))
        )
        if fallback is not None:
            return fallback
    supported = sorted(
        layer for layer, proto in GPT2_CHAMPION_BASELINES if proto == protocol_version
    )
    raise KeyError(
        f"No GPT-2 champion baseline for layers={int(layers)} "
        f"protocol={protocol_version!r}; supported layers={supported}"
    )


__all__ = [
    "CHAMPION_FLOOR_PROTOCOL_VERSION",
    "CHAMPION_PLATEAU_WINDOW_STEPS",
    "ChampionFloorMetrics",
    "ChampionGpt2Baseline",
    "GPT2_CHAMPION_BASELINES",
    "extract_champion_floor_metrics",
    "lookup_gpt2_champion_baseline",
]
