import pytest
import asyncio
from unittest.mock import AsyncMock, patch

from app.orchestrator.openai_client import get_openai_tools, _to_openai_json_schema
from app.orchestrator.voice_bridge import async_run_voice_bridge


def test_openai_schema_conversion():
    """Verify Gemini-style uppercase types are converted to OpenAI lowercase types."""
    gemini_schema = {
        "type": "OBJECT",
        "properties": {
            "query": {"type": "STRING", "description": "Search term"},
            "count": {"type": "INTEGER", "description": "Result count"}
        },
        "required": ["query"]
    }
    openai_schema = _to_openai_json_schema(gemini_schema)
    assert openai_schema["type"] == "object"
    assert openai_schema["properties"]["query"]["type"] == "string"
    assert openai_schema["properties"]["count"]["type"] == "integer"


def test_get_openai_tools():
    """Verify tool registry tools are formatted for OpenAI Realtime API."""
    tools = get_openai_tools()
    assert isinstance(tools, list)
    assert len(tools) > 0
    first_tool = tools[0]
    assert "type" in first_tool
    assert first_tool["type"] == "function"
    assert "name" in first_tool
    assert "description" in first_tool
    assert "parameters" in first_tool


@pytest.mark.asyncio
async def test_voice_bridge_fallback_when_no_openai_key():
    """Verify voice bridge automatically falls back to Gemini when OpenAI key is unconfigured."""
    mock_ws = AsyncMock()
    with patch("app.orchestrator.voice_bridge.DEEPGRAM_API_KEY", ""), \
         patch("app.orchestrator.voice_bridge.CARTESIA_API_KEY", ""), \
         patch("app.orchestrator.voice_bridge.OPENAI_API_KEY", ""), \
         patch("app.orchestrator.voice_bridge.GEMINI_API_KEY", "mock_gemini_key"), \
         patch("app.orchestrator.voice_bridge.async_run_gemini_live_bridge", new_callable=AsyncMock) as mock_gemini:
        
        await async_run_voice_bridge(mock_ws, "test-session-123")
        mock_gemini.assert_called_once_with(mock_ws, "test-session-123")


@pytest.mark.asyncio
async def test_voice_bridge_fallback_on_openai_error():
    """Verify voice bridge falls back to Gemini when primary OpenAI connection fails."""
    mock_ws = AsyncMock()
    with patch("app.orchestrator.voice_bridge.DEEPGRAM_API_KEY", ""), \
         patch("app.orchestrator.voice_bridge.CARTESIA_API_KEY", ""), \
         patch("app.orchestrator.voice_bridge.OPENAI_API_KEY", "mock_openai_key"), \
         patch("app.orchestrator.voice_bridge.GEMINI_API_KEY", "mock_gemini_key"), \
         patch("app.orchestrator.voice_bridge.async_run_openai_realtime_bridge", side_effect=RuntimeError("OpenAI connection failed")), \
         patch("app.orchestrator.voice_bridge.async_run_gemini_live_bridge", new_callable=AsyncMock) as mock_gemini:
        
        await async_run_voice_bridge(mock_ws, "test-session-456")
        mock_gemini.assert_called_once_with(mock_ws, "test-session-456")
