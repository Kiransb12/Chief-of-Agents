import json
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional
import redis
from app.config import REDIS_HOST, REDIS_PORT, REDIS_PASSWORD, SESSION_STORE_BACKEND

logger = logging.getLogger(__name__)


class BaseSessionStore(ABC):

    @abstractmethod
    def get(self, session_id: str) -> List[Dict[str, str]]:
        """Retrieve message list for the session."""
        pass

    @abstractmethod
    def save(self, session_id: str, turns: List[Dict[str, str]]) -> None:
        """Save message list for the session."""
        pass

    @abstractmethod
    def clear(self, session_id: str) -> None:
        """Clear message history for the session."""
        pass


class InMemorySessionStore(BaseSessionStore):

    def __init__(self):
        self._store: Dict[str, List[Dict[str, str]]] = {}

    def get(self, session_id: str) -> List[Dict[str, str]]:
        return self._store.get(session_id, [])

    def save(self, session_id: str, turns: List[Dict[str, str]]) -> None:
        self._store[session_id] = turns

    def clear(self, session_id: str) -> None:
        if session_id in self._store:
            del self._store[session_id]


class RedisSessionStore(BaseSessionStore):

    def __init__(self):
        self._fallback = InMemorySessionStore()
        self._client = None
        self._use_fallback = False
        try:
            self._client = redis.Redis(
                host=REDIS_HOST,
                port=REDIS_PORT,
                password=REDIS_PASSWORD,
                decode_responses=True,
                socket_timeout=2.0,
            )
            # Test connection
            self._client.ping()
            logger.info("Successfully connected to Redis session store.")
        except Exception as e:
            logger.warning(
                f"Failed to connect to Redis: {e}. Falling back to In-Memory session store."
            )
            self._use_fallback = True

    def get(self, session_id: str) -> List[Dict[str, str]]:
        if self._use_fallback:
            return self._fallback.get(session_id)
        try:
            val = self._client.get(f"session:{session_id}")
            if val:
                return json.loads(val)
            return []
        except Exception as e:
            logger.error(f"Redis get error: {e}. Falling back to memory.")
            return self._fallback.get(session_id)

    def save(self, session_id: str, turns: List[Dict[str, str]]) -> None:
        if self._use_fallback:
            self._fallback.save(session_id, turns)
            return
        try:
            # Expire sessions after 24 hours of inactivity
            self._client.setex(f"session:{session_id}", 86400, json.dumps(turns))
        except Exception as e:
            logger.error(f"Redis save error: {e}. Falling back to memory.")
            self._fallback.save(session_id, turns)

    def clear(self, session_id: str) -> None:
        if self._use_fallback:
            self._fallback.clear(session_id)
            return
        try:
            self._client.delete(f"session:{session_id}")
        except Exception as e:
            logger.error(f"Redis delete error: {e}. Falling back to memory.")
            self._fallback.clear(session_id)


# Factory function to get store
def get_session_store() -> BaseSessionStore:
    if SESSION_STORE_BACKEND.lower() == "redis":
        return RedisSessionStore()
    return InMemorySessionStore()
