import os

class Settings:
    LINEAGE_SYNC_ENABLED: bool = os.environ.get("ARIA_LINEAGE_SYNC_ENABLED", "0") != "0"
    LINEAGE_SYNC_BASE: str = os.environ.get("ARIA_RESEARCH_API_BASE", "http://127.0.0.1:5000")
    LINEAGE_SYNC_TIMEOUT: float = float(os.environ.get("ARIA_LINEAGE_SYNC_TIMEOUT", "3"))

settings = Settings()
