"""In-memory stub for Redis/Valkey.

Drop-in replacement for aioredis when you don't have a real Redis instance.
Stores everything in a plain Python dict — data is lost on restart.
Switch back to real Redis by setting USE_MEMORY_STORE=false in .env
"""
from __future__ import annotations

import asyncio
import time
from typing import Any, Optional


class MemoryStore:
    """Mimics the aioredis.Redis interface we actually use."""

    def __init__(self):
        self._store: dict[str, tuple[Any, Optional[float]]] = {}  # key → (value, expires_at)
        self._lock = asyncio.Lock()

    def _is_expired(self, key: str) -> bool:
        if key not in self._store:
            return True
        _, expires_at = self._store[key]
        if expires_at is None:
            return False
        return time.monotonic() > expires_at

    async def ping(self) -> bool:
        return True

    async def get(self, key: str) -> Optional[str]:
        if self._is_expired(key):
            self._store.pop(key, None)
            return None
        value, _ = self._store[key]
        return value

    async def set(self, key: str, value: str, ex: Optional[int] = None) -> bool:
        expires_at = time.monotonic() + ex if ex else None
        self._store[key] = (value, expires_at)
        return True

    async def delete(self, *keys: str) -> int:
        count = 0
        for key in keys:
            if key in self._store:
                del self._store[key]
                count += 1
        return count


# Singleton
_memory_store: Optional[MemoryStore] = None


def get_memory_store() -> MemoryStore:
    global _memory_store
    if _memory_store is None:
        _memory_store = MemoryStore()
    return _memory_store