"""
Dedicated Tool Authorization Layer and History Sanitization Module.
Authoritative source of truth for tool availability, history sanitization, and intent-based tool trimming.
"""
import logging
from typing import Dict, List, Tuple, Any, Set, Optional

logger = logging.getLogger(__name__)

# Map router intents to minimum required tool sets
INTENT_TOOL_MAP: Dict[str, Set[str]] = {
    "check_weather": {"get_live_weather"},
    "search_web": {"search_web"},
    "open_browser_search": {
        "open_browser_and_search",
        "scroll_webpage",
        "search_in_page",
        "go_to_main_page",
        "open_new_tab",
    },
    "scroll_webpage": {"scroll_webpage"},
    "search_in_page": {"search_in_page"},
    "go_to_main_page": {"go_to_main_page"},
    "open_new_tab": {"open_new_tab"},
    "manage_calendar": {"get_calendar_events", "create_calendar_event"},
    "whatsapp_send_message": {"whatsapp_send_message"},
    "search_knowledge": {"retrieve_rag_context"},  # Memory / personal info
    "general_chat": set(),      # Direct LLM response, 0 tools
    "greet": set(),             # Reflex path, 0 tools
}


def sanitize_conversation_history(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Sanitizes conversation history before sending to LLM.
    Strips out:
    - Assistant messages containing tool_calls or function calls from previous turns.
    - Role 'tool' / 'function' / ToolMessage objects.
    Preserves:
    - System messages
    - User / Human messages
    - Assistant messages containing plain text responses only.
    """
    sanitized = []
    for msg in messages:
        role = msg.get("role")
        if role in ("system", "user"):
            sanitized.append(msg)
        elif role == "assistant":
            tool_calls = msg.get("tool_calls")
            content = msg.get("content")
            # Preserve assistant text response only if no tool calls are attached
            if not tool_calls and content and content.strip():
                sanitized.append({"role": "assistant", "content": content})
        # 'tool' and 'function' roles are excluded completely
    return sanitized


def get_tools_for_intent(intent: str, all_tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Dynamically filters all registered tools down to the minimum required tool set for the routed intent.
    If intent requires no tools (e.g. general_chat, search_knowledge), returns an empty list [].
    """
    required_names = INTENT_TOOL_MAP.get(intent)
    if required_names is None:
        # Default fallback for unmapped custom intents: expose all tools
        return all_tools
    
    if not required_names:
        return []

    return [
        t for t in all_tools
        if "function" in t and t["function"].get("name") in required_names
    ]


class ToolAuthorizationLayer:
    """
    Dedicated Authorization Layer between Planner/Recovery and Executor.
    Maintains the authoritative allow-list for the current request context.
    """

    def __init__(self, allowed_tools: List[Dict[str, Any]]):
        self.allowed_tools = allowed_tools
        self.allowed_tool_names: Set[str] = {
            t["function"]["name"]
            for t in allowed_tools
            if isinstance(t, dict) and "function" in t and "name" in t["function"]
        }

    def is_authorized(self, tool_name: str) -> bool:
        """Returns True if tool_name is present in the allow-list for the current request."""
        return tool_name in self.allowed_tool_names

    def authorize_call(self, tool_name: str, call_id: str = "call_0") -> bool:
        """
        Validates tool_name against allowed tools and logs structured output for rejection.
        """
        if self.is_authorized(tool_name):
            logger.info(f"[ToolAuthorizer] AUTHORIZED tool '{tool_name}' (call_id={call_id})")
            return True
        else:
            logger.warning(
                f"[ToolAuthorizer] REJECTED hallucinated tool '{tool_name}' (call_id={call_id}). "
                f"Allowed tools for current request: {sorted(list(self.allowed_tool_names)) or 'NONE'}"
            )
            return False

    def authorize_and_filter(
        self, requested_calls: List[Tuple[str, str, dict]]
    ) -> Tuple[List[Tuple[str, str, dict]], List[str]]:
        """
        Validates a batch of requested tool calls (id, name, args).
        Returns:
            (authorized_calls, rejected_tool_names)
        """
        authorized = []
        rejected = []
        for call_id, name, args in requested_calls:
            if self.authorize_call(name, call_id):
                authorized.append((call_id, name, args))
            else:
                rejected.append(name)
        return authorized, rejected
