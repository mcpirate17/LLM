import os

class Settings:
    LINEAGE_SYNC_ENABLED: bool = os.environ.get("ARIA_LINEAGE_SYNC_ENABLED", "0") != "0"
    LINEAGE_SYNC_BASE: str = os.environ.get("ARIA_RESEARCH_API_BASE", "http://127.0.0.1:5000")
    LINEAGE_SYNC_TIMEOUT: float = float(os.environ.get("ARIA_LINEAGE_SYNC_TIMEOUT", "3"))
    RECOMMENDER_USE_RESEARCH_SIGNALS: bool = os.environ.get("ARIA_RECOMMENDER_USE_RESEARCH_SIGNALS", "1") != "0"
    RECOMMENDER_SIGNALS_TIMEOUT: float = float(os.environ.get("ARIA_RECOMMENDER_SIGNALS_TIMEOUT", "0.8"))
    RECOMMENDER_SIGNALS_TTL_S: float = float(os.environ.get("ARIA_RECOMMENDER_SIGNALS_TTL_S", "45"))

settings = Settings()
