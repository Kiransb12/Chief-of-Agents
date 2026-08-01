"""
The routing layer: a cheap, fast model call using Groq.
"""
import json
import logging
import asyncio
from typing import Dict, List

from groq import AsyncGroq

from app.config import GROQ_API_KEY, ROUTER_MODEL

logger = logging.getLogger(__name__)

_client = None


def _get_client() -> AsyncGroq:
    global _client
    if _client is None:
        _client = AsyncGroq(api_key=GROQ_API_KEY)
    return _client


ROUTER_SYSTEM_PROMPT = """You are an intent classifier for a personal AI assistant.
Output ONLY one route in raw JSON format.

Possible routes:
- REFLEX: Greeting, Thanks, Bye, Small talk (e.g. "Hi", "Hello", "Good morning", "Thanks", "Bye").
- MEMORY: Questions about previous conversations, user preferences, or stored memories (e.g. "What are my preferences?", "What do you remember?", "Who am I?", "What did I tell you?", "Summarize our chats.").
- TOOL_AGENT: Requires external information or actions (e.g. Weather, News, Stocks, Search, Email, Calendar, Files, Database, Browser actions, WhatsApp).
- DIRECT_LLM: Everything answerable from internal knowledge or general concepts (e.g. "Explain transformers", "Explain AI Engineering", "How does TCP work", "What is Kubernetes", "Difference between CNN and RNN").

You MUST respond in raw JSON format. The JSON schema must contain the following keys:
{
  "intent": "greet" | "search_web" | "manage_calendar" | "check_weather" | "search_knowledge" | "open_browser_search" | "scroll_webpage" | "search_in_page" | "whatsapp_send_message" | "go_to_main_page" | "open_new_tab" | "general_chat",
  "is_reflex": boolean,
  "reflex_response": "string response if is_reflex is true, otherwise empty",
  "needs_rag": boolean,
  "route": "single_agent" | "direct_answer",
  "reasoning": "one sentence explanation"
}

ROUTER CLASSIFICATION RULES:
1. REFLEX: If intent is 'greet' or small talk, set `is_reflex` = true, `route` = 'direct_answer', `needs_rag` = false, and output a friendly response in `reflex_response`.
2. MEMORY: If the user asks about preferences, stored facts, or past conversation summaries, set `is_reflex` = false, `needs_rag` = true, `intent` = 'search_knowledge', `route` = 'direct_answer'.
3. TOOL_AGENT: If external information (live weather, web search, browser operations, calendar, WhatsApp) is strictly required, set `is_reflex` = false, `needs_rag` = false, `route` = 'single_agent'.
4. DIRECT_LLM: For general knowledge, coding, science, technology explanations (e.g. AI Engineering, transformers, TCP, Kubernetes, CNN vs RNN), set `is_reflex` = false, `needs_rag` = false, `intent` = 'general_chat', `route` = 'direct_answer'. Do NOT route conceptual/general knowledge questions to TOOL_AGENT.
"""


async def _generate_content_with_retry(
    client: AsyncGroq, model, messages, response_format=None, max_retries=5, initial_delay=2.0
):
    delay = initial_delay
    for attempt in range(max_retries):
        try:
            return await client.chat.completions.create(
                model=model,
                messages=messages,
                response_format=response_format,
                temperature=0.0,
            )
        except Exception as e:
            if any(
                err in str(e)
                for err in [
                    "429",
                    "RESOURCE_EXHAUSTED",
                    "503",
                    "UNAVAILABLE",
                    "Rate limit",
                ]
            ):
                logger.warning(
                    f"Router rate/demand limit hit. Retrying in {delay}s... (Attempt {attempt+1}/{max_retries})"
                )
                await asyncio.sleep(delay)
                delay *= 2
            else:
                raise e
    return await client.chat.completions.create(
        model=model,
        messages=messages,
        response_format=response_format,
        temperature=0.0,
    )


async def route(user_message: str, recent_turns: List[Dict[str, str]]) -> Dict:
    client = _get_client()

    messages = [{"role": "system", "content": ROUTER_SYSTEM_PROMPT}]
    for turn in recent_turns:
        role = "user" if turn["role"] == "user" else "assistant"
        messages.append({"role": role, "content": turn["content"]})
    if not (recent_turns and recent_turns[-1].get("role") == "user" and recent_turns[-1].get("content") == user_message):
        messages.append({"role": "user", "content": user_message})

    try:
        response = await _generate_content_with_retry(
            client=client,
            model=ROUTER_MODEL,
            messages=messages,
            response_format={"type": "json_object"},
        )

        decision_text = response.choices[0].message.content.strip()
        decision = json.loads(decision_text)

        # Extract tokens
        usage = {}
        if response.usage:
            usage = {
                "input_tokens": response.usage.prompt_tokens,
                "output_tokens": response.usage.completion_tokens,
            }
        decision["usage"] = usage
        return decision

    except Exception as e:
        logger.error(f"Router call failed: {e}")
        return {
            "intent": "general_chat",
            "is_reflex": False,
            "reflex_response": "",
            "needs_rag": True,
            "route": "direct_answer",
            "reasoning": f"router_failed: {str(e)}",
            "usage": {"input_tokens": 0, "output_tokens": 0},
        }
