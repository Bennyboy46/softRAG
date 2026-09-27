import threading
from typing import Any, Dict, Optional


class InMemoryCache:
    """Simple in-memory cache with repository-scoped keys for deterministic lookups."""

    def __init__(self):
        self._store: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.RLock()

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return None
            return entry.get("value")

    def set(self, key: str, value: Any, ttl_seconds: Optional[float] = None) -> None:
        with self._lock:
            self._store[key] = {"value": value, "ttl_seconds": ttl_seconds}

    def invalidate(self, key: str) -> None:
        with self._lock:
            self._store.pop(key, None)

    def clear(self) -> None:
        with self._lock:
            self._store.clear()


def make_cache_key(*parts: Any) -> str:
    return "::".join(str(part) for part in parts)


cache = InMemoryCache()
