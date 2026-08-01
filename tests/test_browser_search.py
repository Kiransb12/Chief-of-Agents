import pytest
from unittest.mock import patch
from app.orchestrator.tool_executor import execute_tool, ToolContext

@pytest.mark.asyncio
async def test_open_browser_and_search_query():
    context = ToolContext(session_id="test-session")
    with patch("app.orchestrator.tools.webbrowser.open") as mock_open:
        res = await execute_tool("open_browser_and_search", {"query": "python tutorials"}, context)
        assert res["status"] == "success"
        assert "Successfully opened default browser to:" in res["data"]["result"]
        mock_open.assert_called_once_with("https://www.google.com/search?q=python%20tutorials")

@pytest.mark.asyncio
async def test_open_browser_and_search_url():
    context = ToolContext(session_id="test-session")
    with patch("app.orchestrator.tools.webbrowser.open") as mock_open:
        res = await execute_tool("open_browser_and_search", {"query": "https://github.com"}, context)
        assert res["status"] == "success"
        assert "Successfully opened default browser to:" in res["data"]["result"]
        mock_open.assert_called_once_with("https://github.com")

@pytest.mark.asyncio
async def test_open_browser_and_search_domain():
    context = ToolContext(session_id="test-session")
    with patch("app.orchestrator.tools.webbrowser.open") as mock_open:
        res = await execute_tool("open_browser_and_search", {"query": "google.com"}, context)
        assert res["status"] == "success"
        assert "Successfully opened default browser to:" in res["data"]["result"]
        mock_open.assert_called_once_with("https://google.com")

@pytest.mark.asyncio
async def test_open_browser_and_search_validation():
    context = ToolContext(session_id="test-session")
    res = await execute_tool("open_browser_and_search", {}, context)
    assert res["status"] == "error"
    assert "Missing or invalid string argument" in res["error"]
