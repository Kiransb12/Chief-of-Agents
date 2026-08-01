import time
import logging
import asyncio
from dataclasses import dataclass, field
from typing import List, Dict, Callable, Any

from app.orchestrator.memory import load_semantic_memory
from app.rag.retriever import retrieve as rag_retrieve
from app.orchestrator.tools import (
    get_calendar_events as local_get_calendar_events,
    create_calendar_event as local_create_calendar_event,
    search_web as local_search_web,
    get_live_weather as local_get_live_weather,
    open_browser_and_search as local_open_browser_and_search,
    scroll_webpage as local_scroll_webpage,
    search_in_page as local_search_in_page,
    whatsapp_send_message as local_whatsapp_send_message,
    go_to_main_page as local_go_to_main_page,
    open_new_tab as local_open_new_tab,
)




logger = logging.getLogger(__name__)


@dataclass
class ToolContext:
    session_id: str
    user_id: str = "default_user"
    request_metadata: dict = field(default_factory=dict)


def make_response(
    status: str,
    data: dict = None,
    error: str = None,
    execution_time_ms: float = 0.0,
) -> dict:
    return {
        "status": status,
        "data": data or {},
        "metadata": {
            "execution_time_ms": execution_time_ms,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "error": error,
    }


class ToolRegistry:

    def __init__(self):
        self.registry: Dict[str, Callable] = {}
        self.schemas: List[dict] = []

    def register(self, name: str, description: str, parameters: dict = None):
        def decorator(func: Callable):
            self.registry[name] = func
            self.schemas.append(
                {
                    "name": name,
                    "description": description,
                    "parameters": parameters
                    or {"type": "OBJECT", "properties": {}},
                }
            )
            return func

        return decorator


# Global Registry Instance
tool_registry = ToolRegistry()


# Helper for word overlap deduplication (70% threshold check)
def _get_word_overlap_ratio(s1: str, s2: str) -> float:
    w1 = set(s1.lower().split())
    w2 = set(s2.lower().split())
    if not w1 or not w2:
        return 0.0
    return len(w1.intersection(w2)) / min(len(w1), len(w2))


# ----------------------------------------------------
# TOOL REGISTRATIONS
# ----------------------------------------------------


@tool_registry.register(
    name="retrieve_rag_context",
    description="Retrieve relevant personal background context and active memories matching the query.",
    parameters={
        "type": "OBJECT",
        "properties": {
            "query": {
                "type": "STRING",
                "description": "The search term to query personal info or documents.",
            }
        },
        "required": ["query"],
    },
)
def retrieve_rag_context(context: ToolContext, query: str) -> str:
    # 1. Fetch top RAG documents from vector store (Chroma)
    raw_docs = []
    try:
        raw_results = rag_retrieve(query, k=3)
        raw_docs = [r["text"] for r in raw_results]
    except Exception as e:
        logger.error(f"[Tool: retrieve_rag_context] Chroma retrieval error: {e}")

    # 2. Fetch semantic memories and rank by query keyword overlap
    facts = load_semantic_memory()
    query_words = set(query.lower().split())
    fact_scores = []
    for fact in facts:
        fact_words = set(fact.lower().split())
        overlap = len(query_words.intersection(fact_words))
        if overlap > 0:
            fact_scores.append((overlap, fact))

    fact_scores.sort(key=lambda x: x[0], reverse=True)
    top_facts = [fact for score, fact in fact_scores[:3]]

    # 3. Merge & Deduplicate (discard document if word overlap ratio > 0.70 with a semantic fact)
    filtered_docs = []
    for doc in raw_docs:
        is_duplicate = False
        for fact in top_facts:
            if _get_word_overlap_ratio(doc, fact) > 0.70:
                is_duplicate = True
                break
        if not is_duplicate:
            filtered_docs.append(doc)

    # 4. Compile context (Facts listed first, then RAG docs)
    results = []
    if top_facts:
        results.append("Relevant Semantic Memory Facts:")
        results.extend(f"- {f}" for f in top_facts)
    if filtered_docs:
        results.append("Relevant Document context:")
        results.extend(f"- {d}" for d in filtered_docs)

    if not results:
        return "No relevant personal documents or active memories found for this query."

    return "\n".join(results)


@tool_registry.register(
    name="get_calendar_events",
    description="Retrieve the list of scheduled events on the user's calendar.",
    parameters={"type": "OBJECT", "properties": {}},
)
def get_calendar_events(context: ToolContext) -> str:
    return local_get_calendar_events()


@tool_registry.register(
    name="create_calendar_event",
    description="Schedule a new calendar event.",
    parameters={
        "type": "OBJECT",
        "properties": {
            "title": {
                "type": "STRING",
                "description": "Title/description of the event.",
            },
            "date_time": {
                "type": "STRING",
                "description": "Date and time of the event (e.g. tomorrow at 4 PM).",
            },
            "duration_minutes": {
                "type": "INTEGER",
                "description": "Duration in minutes. Defaults to 30.",
            },
        },
        "required": ["title", "date_time"],
    },
)
def create_calendar_event(
    context: ToolContext,
    title: str,
    date_time: str,
    duration_minutes: int = 30,
) -> str:
    return local_create_calendar_event(
        title=title, date_time=date_time, duration_minutes=duration_minutes
    )


@tool_registry.register(
    name="search_web",
    description="Search the web for news or info and return snippets.",
    parameters={
        "type": "OBJECT",
        "properties": {
            "query": {
                "type": "STRING",
                "description": "Web search query terms.",
            }
        },
        "required": ["query"],
    },
)
def search_web(context: ToolContext, query: str) -> str:
    return local_search_web(query)


@tool_registry.register(
    name="open_browser_and_search",
    description="Open the default web browser and search for a query or open a URL.",
    parameters={
        "type": "OBJECT",
        "properties": {
            "query": {
                "type": "STRING",
                "description": "The search query or URL to open in the browser.",
            }
        },
        "required": ["query"],
    },
)
def open_browser_and_search(context: ToolContext, query: str) -> str:
    return local_open_browser_and_search(query)


@tool_registry.register(
    name="scroll_webpage",
    description="Scroll the active webpage window up or down.",
    parameters={
        "type": "OBJECT",
        "properties": {
            "direction": {
                "type": "STRING",
                "description": "The direction to scroll: 'up' or 'down'.",
            },
            "amount": {
                "type": "INTEGER",
                "description": "The number of scroll increments (default 5).",
            }
        },
        "required": ["direction"],
    },
)
def scroll_webpage(context: ToolContext, direction: str, amount: int = 5) -> str:
    return local_scroll_webpage(direction, amount)


@tool_registry.register(
    name="search_in_page",
    description="Perform a text search (Ctrl+F) on the active webpage for the given query.",
    parameters={
        "type": "OBJECT",
        "properties": {
            "query": {
                "type": "STRING",
                "description": "The text to search for on the page.",
            }
        },
        "required": ["query"],
    },
)
def search_in_page(context: ToolContext, query: str) -> str:
    return local_search_in_page(query)


@tool_registry.register(
    name="whatsapp_send_message",
    description="Open WhatsApp Web to send a message to a phone number or a contact name.",
    parameters={
        "type": "OBJECT",
        "properties": {
            "phone": {
                "type": "STRING",
                "description": "The phone number (digits only, e.g., '+919999999999') or the contact name (e.g., 'Lohit') on WhatsApp.",
            },
            "message": {
                "type": "STRING",
                "description": "The message text to send.",
            }
        },
        "required": ["phone", "message"],
    },
)
def whatsapp_send_message(context: ToolContext, phone: str, message: str) -> str:
    return local_whatsapp_send_message(phone, message)



@tool_registry.register(
    name="go_to_main_page",
    description="Open the default web browser and navigate back to the agent's main dashboard page.",
    parameters={"type": "OBJECT", "properties": {}},
)
def go_to_main_page(context: ToolContext) -> str:
    return local_go_to_main_page()


@tool_registry.register(
    name="open_new_tab",
    description="Open a new tab in the default web browser with the given URL.",
    parameters={
        "type": "OBJECT",
        "properties": {
            "url": {
                "type": "STRING",
                "description": "The URL to open in the new tab (default is Google).",
            }
        },
    },
)
def open_new_tab(context: ToolContext, url: str = "https://www.google.com") -> str:
    return local_open_new_tab(url)





@tool_registry.register(
    name="get_live_weather",
    description="Get the live weather forecast for a given city.",
    parameters={
        "type": "OBJECT",
        "properties": {
            "location": {
                "type": "STRING",
                "description": "City or location name.",
            }
        },
        "required": ["location"],
    },
)
def get_live_weather(context: ToolContext, location: str) -> str:
    return local_get_live_weather(location)


# ----------------------------------------------------
# EXECUTOR INTERFACE
# ----------------------------------------------------


async def execute_tool(name: str, args: dict, context: ToolContext) -> dict:
    """Invokes the local registered tool, sanitizing and validating inputs."""
    start_time = time.time()
    if name not in tool_registry.registry:
        elapsed = (time.time() - start_time) * 1000
        return make_response(
            "error",
            error=f"Tool '{name}' is not registered.",
            execution_time_ms=elapsed,
        )

    # 1. Input Sanitization & Argument Validation
    try:
        if name == "retrieve_rag_context":
            if "query" not in args or not isinstance(args["query"], str):
                raise ValueError("Missing or invalid string argument: 'query'")
            if len(args["query"].strip()) == 0:
                raise ValueError("Query string cannot be empty.")

        elif name == "create_calendar_event":
            if "title" not in args or not isinstance(args["title"], str):
                raise ValueError("Missing or invalid string argument: 'title'")
            if "date_time" not in args or not isinstance(
                args["date_time"], str
            ):
                raise ValueError(
                    "Missing or invalid string argument: 'date_time'"
                )
            if "duration_minutes" in args and not isinstance(
                args["duration_minutes"], (int, float)
            ):
                raise ValueError(
                    "Invalid numerical argument: 'duration_minutes'"
                )

        elif name == "search_web":
            if "query" not in args or not isinstance(args["query"], str):
                raise ValueError("Missing or invalid string argument: 'query'")

        elif name == "open_browser_and_search":
            if "query" not in args or not isinstance(args["query"], str):
                raise ValueError("Missing or invalid string argument: 'query'")

        elif name == "scroll_webpage":
            if "direction" not in args or not isinstance(args["direction"], str):
                raise ValueError("Missing or invalid string argument: 'direction'")
            dir_lower = args["direction"].lower().strip()
            if "down" in dir_lower or "dwon" in dir_lower:
                args["direction"] = "down"
            elif "up" in dir_lower:
                args["direction"] = "up"
            else:
                args["direction"] = "down" # Fallback default
            if "amount" in args and not isinstance(args["amount"], (int, float)):
                raise ValueError("Invalid numerical argument: 'amount'")


        elif name == "search_in_page":
            if "query" not in args or not isinstance(args["query"], str):
                raise ValueError("Missing or invalid string argument: 'query'")

        elif name == "whatsapp_send_message":
            if "phone" not in args or not isinstance(args["phone"], str):
                raise ValueError("Missing or invalid string argument: 'phone'")
            if "message" not in args or not isinstance(args["message"], str):
                raise ValueError("Missing or invalid string argument: 'message'")

        elif name == "go_to_main_page":
            pass

        elif name == "open_new_tab":
            if "url" in args and not isinstance(args["url"], str):
                raise ValueError("Invalid string argument: 'url'")



        elif name == "get_live_weather":
            if "location" not in args or not isinstance(args["location"], str):
                raise ValueError(
                    "Missing or invalid string argument: 'location'"
                )

        # 2. Run Tool Invocation (runs synchronous functions safely in executor pool)
        func = tool_registry.registry[name]
        if asyncio.iscoroutinefunction(func):
            res_data = await func(context, **args)
        else:
            res_data = await asyncio.to_thread(func, context, **args)

        elapsed = (time.time() - start_time) * 1000
        return make_response(
            "success", data={"result": res_data}, execution_time_ms=elapsed
        )

    except Exception as e:
        elapsed = (time.time() - start_time) * 1000
        logger.error(f"[ToolExecutor] Error executing '{name}': {e}")
        return make_response("error", error=str(e), execution_time_ms=elapsed)
