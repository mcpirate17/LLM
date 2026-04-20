from __future__ import annotations

from typing import Any

from ..runtime_events import publish_lifecycle_event


class _LifecycleMixin:
    """Shared runner lifecycle helpers."""

    __slots__ = ()

    def _publish_terminal_event(
        self,
        *,
        producer: str,
        event_type: str,
        exp_id: str,
        payload: dict[str, Any],
    ) -> None:
        publish_lifecycle_event(
            notebook_path=self.notebook_path,
            event_type=event_type,
            producer=producer,
            run_id=exp_id,
            payload=payload,
        )

    def _complete_experiment_compat(
        self,
        *,
        nb: Any,
        experiment_id: str,
        results: dict[str, Any],
        aria_summary: str,
        insights: Any = None,
        llm_analysis: str | None = None,
    ) -> None:
        aria = getattr(self, "aria", None)
        aria_mood = getattr(getattr(aria, "state", None), "mood", "contemplative")
        getattr(nb, "complete_experiment")(
            experiment_id=experiment_id,
            results=results,
            aria_summary=aria_summary,
            aria_mood=aria_mood,
            insights=insights,
            llm_analysis=llm_analysis,
        )

    def _fail_experiment_compat(
        self,
        *,
        nb: Any,
        experiment_id: str,
        error: str,
        results: dict[str, Any] | None = None,
    ) -> None:
        kwargs: dict[str, Any] = {}
        if results is not None:
            kwargs["results"] = results
        getattr(nb, "fail_experiment")(experiment_id, error, **kwargs)

    def _log_learning_event_compat(self, nb: Any, *args: Any, **kwargs: Any) -> None:
        getattr(nb, "log_learning_event")(*args, **kwargs)
