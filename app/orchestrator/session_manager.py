import time
import asyncio
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional

from app.orchestrator.state import get_or_create_session as get_conversation_state
from app.orchestrator.memory import consolidate_session

logger = logging.getLogger(__name__)


@dataclass
class SessionState:
    session_id: str
    user_id: str = "default_user"
    is_interrupted: bool = False
    active_tasks: List[asyncio.Task] = field(default_factory=list)
    start_time: float = field(default_factory=time.time)
    metrics: Dict[str, Any] = field(
        default_factory=lambda: {
            "tool_call_count": 0,
            "tool_duration_ms": 0.0,
            "total_latency_ms": 0.0,
            "errors": [],
        }
    )

    def cancel_tasks(self):
        """Cancel all pending asynchronous tasks for this session to free resources."""
        for task in self.active_tasks:
            if not task.done():
                task.cancel()
                logger.info(
                    f"[SessionState] Cancelled pending task: {task.get_name()}"
                )
        self.active_tasks.clear()

    def add_task(self, task: asyncio.Task):
        self.active_tasks.append(task)
        # Prune completed tasks
        self.active_tasks = [t for t in self.active_tasks if not t.done()]


class SessionManager:

    def __init__(self):
        self._sessions: Dict[str, SessionState] = {}

    def get(self, session_id: str) -> SessionState:
        if session_id not in self._sessions:
            self._sessions[session_id] = SessionState(session_id=session_id)
            logger.info(
                f"[SessionManager] Initialized isolated session: {session_id}"
            )
        return self._sessions[session_id]

    def remove(self, session_id: str) -> None:
        if session_id in self._sessions:
            session = self._sessions[session_id]
            session.cancel_tasks()
            del self._sessions[session_id]
            logger.info(
                f"[SessionManager] Cleaned up session resources: {session_id}"
            )

    def add_turn(self, session_id: str, role: str, content: str) -> None:
        state = get_conversation_state(session_id)
        state.add_turn(role, content)

    def get_recent_turns(
        self, session_id: str, n: int = 10
    ) -> List[Dict[str, str]]:
        state = get_conversation_state(session_id)
        return state.recent_turns(n)

    def consolidate(self, session_id: str) -> dict:
        """Invokes consolidation job, saves semantic facts/episodic summary, and flushes turns."""
        state = get_conversation_state(session_id)
        turns = state.recent_turns(n=100)

        # Run memory consolidation LLM pipeline
        res = consolidate_session(session_id, turns)

        # Clear working chat turns
        state.clear()

        # Log session metrics
        session = self.get(session_id)
        elapsed = (time.time() - session.start_time) * 1000
        logger.info(
            f"[SessionManager] Consolidation complete for {session_id}: "
            f"elapsed_ms={elapsed:.2f} | "
            f"tools_run={session.metrics['tool_call_count']} | "
            f"tools_duration_ms={session.metrics['tool_duration_ms']:.2f} | "
            f"errors={len(session.metrics['errors'])}"
        )

        self.remove(session_id)
        return res


# Global Session Manager Instance
session_manager = SessionManager()
