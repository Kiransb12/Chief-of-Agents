"""
Working memory for the current session.
"""
from typing import Dict, List
from app.orchestrator.session_store import get_session_store

# Initialize session store globally
_store = get_session_store()


class ConversationState:

    def __init__(self, session_id: str):
        self.session_id = session_id

    def add_turn(self, role: str, content: str) -> None:
        turns = _store.get(self.session_id)
        turns.append({"role": role, "content": content})
        _store.save(self.session_id, turns)

    def recent_turns(self, n: int = 10) -> List[Dict[str, str]]:
        turns = _store.get(self.session_id)
        return turns[-n:]

    def clear(self) -> None:
        _store.clear(self.session_id)


def get_or_create_session(session_id: str) -> ConversationState:
    # Return a state wrapper. Retrieval is performed dynamically from the store.
    return ConversationState(session_id=session_id)
