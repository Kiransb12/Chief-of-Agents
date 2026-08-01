"""
Unit tests for ToolAuthorizationLayer, History Sanitization, and Intent-based Tool Trimming.
"""
import pytest
from app.orchestrator.tool_authorizer import (
    sanitize_conversation_history,
    get_tools_for_intent,
    ToolAuthorizationLayer,
)

ALL_TEST_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_live_weather",
            "description": "Get current weather conditions.",
            "parameters": {"type": "object", "properties": {"location": {"type": "string"}}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "Search the web.",
            "parameters": {"type": "object", "properties": {"query": {"type": "string"}}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_calendar_events",
            "description": "Get calendar events.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]


def test_sanitize_conversation_history():
    """Verify tool_calls and role='tool' messages are stripped out while user and plain assistant responses are kept."""
    raw_history = [
        {"role": "user", "content": "What's the weather?"},
        {
            "role": "assistant",
            "content": "Checking weather...",
            "tool_calls": [{"id": "call_1", "function": {"name": "get_live_weather"}}],
        },
        {"role": "tool", "tool_call_id": "call_1", "name": "get_live_weather", "content": "25°C Sunny"},
        {"role": "assistant", "content": "The weather is 25°C and sunny."},
        {"role": "user", "content": "Explain AI engineering."},
    ]

    sanitized = sanitize_conversation_history(raw_history)

    # Should keep user turns and the plain text assistant response, dropping tool_calls & tool messages
    assert len(sanitized) == 3
    assert sanitized[0] == {"role": "user", "content": "What's the weather?"}
    assert sanitized[1] == {"role": "assistant", "content": "The weather is 25°C and sunny."}
    assert sanitized[2] == {"role": "user", "content": "Explain AI engineering."}


def test_get_tools_for_intent_trimming():
    """Verify dynamic intent-based tool set selection."""
    # 1. Weather intent -> get_live_weather only
    weather_tools = get_tools_for_intent("check_weather", ALL_TEST_TOOLS)
    assert len(weather_tools) == 1
    assert weather_tools[0]["function"]["name"] == "get_live_weather"

    # 2. Search intent -> search_web only
    search_tools = get_tools_for_intent("search_web", ALL_TEST_TOOLS)
    assert len(search_tools) == 1
    assert search_tools[0]["function"]["name"] == "search_web"

    # 3. General knowledge / DIRECT_LLM -> 0 tools
    general_tools = get_tools_for_intent("general_chat", ALL_TEST_TOOLS)
    assert len(general_tools) == 0

    # 4. Memory / search_knowledge -> 0 tools
    memory_tools = get_tools_for_intent("search_knowledge", ALL_TEST_TOOLS)
    assert len(memory_tools) == 0


def test_tool_authorization_layer():
    """Verify ToolAuthorizationLayer authorizes allowed tools and rejects unauthorized/hallucinated tools."""
    allowed_tools = [ALL_TEST_TOOLS[0]]  # Only get_live_weather is allowed
    authorizer = ToolAuthorizationLayer(allowed_tools)

    assert authorizer.is_authorized("get_live_weather") is True
    assert authorizer.is_authorized("search_web") is False
    assert authorizer.is_authorized("fake_hallucinated_tool") is False

    raw_calls = [
        ("call_1", "get_live_weather", {"location": "Chennai"}),
        ("call_2", "search_web", {"query": "Google news"}),
        ("call_3", "fake_hallucinated_tool", {}),
    ]

    authorized, rejected = authorizer.authorize_and_filter(raw_calls)

    assert len(authorized) == 1
    assert authorized[0][1] == "get_live_weather"
    assert rejected == ["search_web", "fake_hallucinated_tool"]
