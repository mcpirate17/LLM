from __future__ import annotations

import threading
from collections import defaultdict

API_HEALTH_COUNTERS: dict[str, int] = defaultdict(int)
API_HEALTH_LOCK = threading.Lock()
