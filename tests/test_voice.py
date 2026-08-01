import pytest
import asyncio
import time
from typing import Generator

from app.orchestrator.tool_executor import (
    tool_registry,
    execute_tool,
    ToolContext,
    make_response,
    _get_word_overlap_ratio,
)
from app.orchestrator.session_manager import session_manager, SessionState


def test_tool_response_schema():
    """Verify tool execution returns the standardized response model."""
    res = make_response("success", data={"result": "test"}, execution_time_ms=10.0)
    assert "status" in res
    assert "data" in res
    assert "metadata" in res
    assert "error" in res
    assert res["status"] == "success"
    assert res["data"] == {"result": "test"}
    assert res["metadata"]["execution_time_ms"] == 10.0


def test_overlap_deduplication():
    """Assert that word overlap ratio correctly identifies duplicate strings."""
    s1 = "User prefers window seats on long international flights."
    s2 = "User prefers window seats on long flights."
    s3 = "User likes eating apples in the morning."

    assert _get_word_overlap_ratio(s1, s2) > 0.70
    assert _get_word_overlap_ratio(s1, s3) < 0.20


@pytest.mark.asyncio
async def test_tool_concurrency_and_timeouts():
    """Verify tool execution correctly intercepts and cancels hanging tools after 8s."""
    
    # Register a mock hanging tool
    @tool_registry.register(
        name="test_hanging_tool",
        description="A mock tool that sleeps for a long time.",
        parameters={"type": "OBJECT", "properties": {}}
    )
    async def test_hanging_tool(context: ToolContext) -> str:
        await asyncio.sleep(20.0)
        return "finished"

    context = ToolContext(session_id="test-session")
    session = session_manager.get("test-session")

    # Wrap tool execution in a wait_for wrapper mimicking our Gemini client loop
    t0 = time.time()
    try:
        from app.orchestrator.gemini_client import _run_and_format_tool
        res = await _run_and_format_tool(
            "test_hanging_tool", {}, "call_id_1", context, session
        )
        assert res["response"]["output"]["status"] == "error"
        assert "timed out" in res["response"]["output"]["error"]
    except Exception as e:
        pytest.fail(f"Execution threw unexpected exception: {e}")

    elapsed = time.time() - t0
    # Confirm it returned in ~8 seconds, not 20 seconds
    assert elapsed < 15.0


def test_session_isolation():
    """Assert that SessionStates are isolated and don't leak metrics between sessions."""
    s1 = session_manager.get("session-alpha")
    s2 = session_manager.get("session-beta")

    s1.metrics["tool_call_count"] = 5
    s2.metrics["tool_call_count"] = 2

    assert s1.metrics["tool_call_count"] == 5
    assert s2.metrics["tool_call_count"] == 2

    session_manager.remove("session-alpha")
    session_manager.remove("session-beta")


@pytest.mark.asyncio
async def test_async_consolidation():
    """Verify that consolidation runs in a background thread and updates/cleans state properly."""
    from unittest.mock import patch
    
    session_id = "test-consolidation-session"
    session = session_manager.get(session_id)
    session_manager.add_turn(session_id, "user", "Hello")
    session_manager.add_turn(session_id, "assistant", "Hi there")
    
    with patch("app.orchestrator.session_manager.consolidate_session") as mock_consolidate:
        mock_consolidate.return_value = {"summary": "Mocked summary", "facts_updated": 0}
        
        # Call session_manager.consolidate asynchronously in a thread
        res = await asyncio.to_thread(session_manager.consolidate, session_id)
        
        # Verify consolidate_session was called and session was removed
        mock_consolidate.assert_called_once()
        assert res["summary"] == "Mocked summary"
        assert session_id not in session_manager._sessions
