"""Background knowledge distiller daemon.

Periodically runs statistical analysis on historical experiment data,
calls a local LLM (via Ollama) to generate narrative synthesis, and
stores the result as an ExperimentDigest for use by the rule-based
fallback and LLM context injection.
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Optional

from .analyzer import run_full_analysis
from .digest import ExperimentDigest

logger = logging.getLogger(__name__)


class KnowledgeDistiller:
    """Background thread that distills experiment history into structured knowledge.

    Usage:
        distiller = KnowledgeDistiller(db_path)
        distiller.start()
        ...
        distiller.notify_cycle_complete()   # after each experiment cycle
        digest = distiller.get_digest()     # thread-safe read
        ...
        distiller.stop()
    """

    def __init__(
        self,
        db_path: str | Path,
        distill_interval_cycles: int = 3,
        local_model: str = "gemma2:2b",
    ):
        self._db_path = str(db_path)
        self._interval = distill_interval_cycles
        self._local_model = local_model

        self._digest: Optional[ExperimentDigest] = None
        self._lock = threading.Lock()
        self._cycle_event = threading.Event()
        self._stop_event = threading.Event()
        self._cycles_since_distill = 0
        self._thread: Optional[threading.Thread] = None

    def start(self):
        """Start the background distiller thread."""
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run_loop, name="KnowledgeDistiller", daemon=True,
        )
        self._thread.start()
        logger.info("KnowledgeDistiller started (interval=%d cycles, model=%s)",
                     self._interval, self._local_model)

    def stop(self):
        """Stop the background thread."""
        self._stop_event.set()
        self._cycle_event.set()  # unblock if waiting
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5.0)
        logger.info("KnowledgeDistiller stopped")

    def notify_cycle_complete(self):
        """Called by the runner after each experiment cycle."""
        self._cycles_since_distill += 1
        if self._cycles_since_distill >= self._interval:
            self._cycle_event.set()

    def get_digest(self) -> Optional[ExperimentDigest]:
        """Thread-safe read of the latest digest."""
        with self._lock:
            return self._digest

    def set_digest(self, digest: ExperimentDigest):
        """Thread-safe write (used for recovery from DB)."""
        with self._lock:
            self._digest = digest

    def _run_loop(self):
        """Main loop: wait for cycle events, run distillation."""
        while not self._stop_event.is_set():
            # Wait for enough cycles or stop
            self._cycle_event.wait(timeout=300)  # 5-min max idle
            if self._stop_event.is_set():
                break
            self._cycle_event.clear()

            if self._cycles_since_distill < self._interval:
                continue

            self._cycles_since_distill = 0
            try:
                self._distill()
            except Exception as e:
                logger.warning("Distillation failed: %s", e, exc_info=True)

    def _distill(self):
        """Run full analysis, call local LLM, store digest."""
        t0 = time.time()
        logger.info("Starting knowledge distillation...")

        # Open a dedicated notebook connection for analysis
        try:
            from ..notebook import LabNotebook
            nb = LabNotebook(self._db_path)
        except Exception as e:
            logger.warning("Failed to open notebook for distillation: %s", e)
            return

        try:
            stats = run_full_analysis(nb)

            # Build stats summary for LLM
            summary_text = self._build_stats_summary(stats)

            # Generate narrative via local LLM
            narrative, recommendations = self._generate_narrative(summary_text)

            digest = ExperimentDigest(
                timestamp=time.time(),
                cycle_number=0,  # will be set by caller
                n_experiments_analyzed=stats.get("n_experiments_analyzed", 0),
                n_curves_analyzed=stats.get("n_curves_analyzed", 0),
                convergence_profiles=stats.get("convergence_profiles", []),
                architecture_families=stats.get("architecture_families", []),
                config_effects=stats.get("config_effects", []),
                op_synergies=stats.get("op_synergies", []),
                hypothesis_outcomes=stats.get("hypothesis_outcomes", []),
                narrative=narrative,
                recommendations=recommendations,
            )

            # Store in DB
            try:
                nb.store_digest(digest.to_dict())
            except Exception as e:
                logger.warning("Failed to persist digest: %s", e)

            # Update thread-safe cache
            with self._lock:
                self._digest = digest

            elapsed = time.time() - t0
            logger.info("Knowledge distillation complete in %.1fs: %s",
                        elapsed, digest.summary_stats())
        finally:
            nb.close()

    def _build_stats_summary(self, stats: dict) -> str:
        """Build a compact text summary of analysis results for the LLM."""
        lines = []

        lines.append(f"Experiments analyzed: {stats.get('n_experiments_analyzed', 0)}")
        lines.append(f"Training curves analyzed: {stats.get('n_curves_analyzed', 0)}")

        # Convergence profiles
        profiles = stats.get("convergence_profiles", [])
        if profiles:
            lines.append("\nTraining Curve Profiles:")
            for p in profiles:
                lines.append(
                    f"  {p.category}: {p.count} curves, "
                    f"S1 rate={p.s1_pass_rate:.0%}, "
                    f"avg final loss={p.avg_final_loss:.4f}, "
                    f"monotonicity={p.avg_monotonicity:.2f}"
                )

        # Architecture families
        families = stats.get("architecture_families", [])
        if families:
            lines.append(f"\nArchitecture Families ({len(families)} clusters):")
            for f in families[:5]:
                lines.append(
                    f"  Family {f.family_id}: {f.n_members} members, "
                    f"ops=[{', '.join(f.representative_ops[:5])}], "
                    f"novelty={f.avg_novelty:.3f}, loss={f.avg_loss_ratio:.4f}"
                )

        # Config effects
        sig_effects = [e for e in stats.get("config_effects", []) if e.p_value < 0.05]
        if sig_effects:
            lines.append(f"\nSignificant Config Effects (p<0.05):")
            for e in sig_effects[:8]:
                lines.append(
                    f"  {e.param_name} -> {e.target}: "
                    f"rho={e.rho:+.3f}, p={e.p_value:.4f} ({e.direction})"
                )

        # Synergies
        synergies = stats.get("op_synergies", [])
        syn = [s for s in synergies if s.label == "synergistic"]
        anti = [s for s in synergies if s.label == "anti_synergistic"]
        if syn:
            lines.append(f"\nSynergistic Op Pairs ({len(syn)} found):")
            for s in syn[:5]:
                lines.append(f"  {s.op_a} + {s.op_b}: lift={s.lift:.2f}x ({s.co_occurrences} co-occurrences)")
        if anti:
            lines.append(f"\nAnti-Synergistic Pairs ({len(anti)} found):")
            for s in anti[:5]:
                lines.append(f"  {s.op_a} + {s.op_b}: lift={s.lift:.2f}x")

        # Hypothesis outcomes
        outcomes = stats.get("hypothesis_outcomes", [])
        if outcomes:
            confirmed = sum(1 for h in outcomes if h.outcome == "confirmed")
            refuted = sum(1 for h in outcomes if h.outcome == "refuted")
            lines.append(f"\nHypothesis Closure: {confirmed} confirmed, {refuted} refuted, "
                         f"{len(outcomes) - confirmed - refuted} inconclusive")

        return "\n".join(lines)

    def _generate_narrative(self, stats_summary: str) -> tuple[str, list[str]]:
        """Call local LLM to synthesize narrative + recommendations.

        Returns (narrative, list_of_recommendations).
        Falls back to empty strings if LLM unavailable.
        """
        try:
            from ..llm.ollama import OllamaBackend
            llm = OllamaBackend()
            llm.model = self._local_model
            llm.keep_alive = 0  # unload after use
        except Exception:
            logger.debug("Could not create Ollama backend for distillation")
            return ("", [])

        if not llm.is_available():
            logger.debug("Ollama not available, skipping narrative generation")
            return ("", [])

        prompt = (
            "You are a neural architecture research analyst. "
            "Synthesize the following experimental statistics into:\n"
            "1. A 150-word narrative summary of the most important findings\n"
            "2. Exactly 5 strategic recommendations for the next experiments, "
            "each on its own line starting with '- '\n\n"
            "Focus on actionable insights. Be specific about which ops, "
            "config values, and architecture patterns to try or avoid.\n\n"
            f"DATA:\n{stats_summary}\n\n"
            "NARRATIVE:\n"
        )

        try:
            resp = llm.generate(prompt, max_tokens=500, temperature=0.3)
            text = resp.text.strip()
        except Exception as e:
            logger.warning("LLM narrative generation failed: %s", e)
            return ("", [])

        # Parse narrative and recommendations
        lines = text.split("\n")
        narrative_lines = []
        recommendations = []
        in_recs = False

        for line in lines:
            stripped = line.strip()
            if stripped.startswith("- ") or stripped.startswith("* "):
                in_recs = True
                recommendations.append(stripped[2:].strip())
            elif in_recs and stripped:
                # Additional recommendation line
                if recommendations:
                    recommendations[-1] += " " + stripped
            else:
                if not in_recs:
                    narrative_lines.append(line)

        narrative = "\n".join(narrative_lines).strip()
        return (narrative, recommendations[:5])
