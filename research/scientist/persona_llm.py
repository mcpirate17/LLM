import logging

logger = logging.getLogger(__name__)


def _init_llm_backend(is_analyst: bool):
    """Create an LLM backend, returning None on failure.

    Shared by _get_llm (primary) and _get_analyst_llm (analyst).
    """
    try:
        from .llm import create_backend

        backend = create_backend(is_analyst=is_analyst)
        if backend:
            label = "Analyst" if is_analyst else "Primary"
            logger.info(
                f"Aria {label} LLM backend: {backend.name} ({getattr(backend, 'model', 'default')})"
            )
        return backend
    except Exception as e:
        label = "Analyst" if is_analyst else "Primary"
        logger.debug(f"{label} LLM backend init failed: {e}")
        return None


class _PersonaLLMMixin:
    def _get_llm(self):
        """Lazy-init primary LLM backend (only try once)."""
        if not self._llm_initialized:
            self._llm_initialized = True
            self._llm = _init_llm_backend(is_analyst=False)
        return self._llm

    def _track_cost(self, resp):
        """Accumulate token usage and estimated cost from an LLM response."""
        if resp and resp.tokens_used:
            self._total_tokens += resp.tokens_used
            backend_name = getattr(self._llm, "name", "")
            rate = self._COST_PER_TOKEN.get(backend_name)
            if rate is None:
                rate = self._COST_PER_TOKEN["anthropic"]
                if (
                    backend_name
                    and backend_name not in self._unknown_cost_backends_warned
                ):
                    logger.warning(
                        "Unknown LLM backend '%s' for cost estimation; using anthropic default rate.",
                        backend_name,
                    )
                    self._unknown_cost_backends_warned.add(backend_name)
            self._total_cost += resp.tokens_used * rate

    def configure_llm(
        self, backend_name: str, api_key: str = "", model: str = "", host: str = ""
    ) -> bool:
        """Configure (or reconfigure) the LLM backend at runtime.

        Returns True if the backend was created successfully.
        """
        from .llm import create_backend_from_config

        try:
            new_backend = create_backend_from_config(
                backend_name, api_key=api_key, model=model, host=host
            )
            if new_backend and new_backend.is_available():
                self._llm = new_backend
                self._llm_initialized = True
                logger.info(f"Aria LLM reconfigured: {new_backend.name}")
                return True
            elif new_backend:
                # Backend created but not reachable — still set it
                # (might become available later, e.g. Ollama starting up)
                self._llm = new_backend
                self._llm_initialized = True
                logger.warning(
                    f"Aria LLM set to {new_backend.name} but not currently reachable"
                )
                return True
        except Exception as e:
            logger.warning(f"LLM reconfiguration failed: {e}")
        return False
