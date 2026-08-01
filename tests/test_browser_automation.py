import pytest
from unittest.mock import patch, MagicMock
from app.orchestrator.tool_executor import execute_tool, ToolContext

@pytest.mark.asyncio
async def test_scroll_webpage_down():
    context = ToolContext(session_id="test-session")
    with patch("pyautogui.scroll") as mock_scroll:
        res = await execute_tool("scroll_webpage", {"direction": "down", "amount": 3}, context)
        assert res["status"] == "success"
        mock_scroll.assert_called_once_with(-360)

@pytest.mark.asyncio
async def test_scroll_webpage_up():
    context = ToolContext(session_id="test-session")
    with patch("pyautogui.scroll") as mock_scroll:
        res = await execute_tool("scroll_webpage", {"direction": "up", "amount": 5}, context)
        assert res["status"] == "success"
        mock_scroll.assert_called_once_with(600)

@pytest.mark.asyncio
async def test_scroll_webpage_typo():
    context = ToolContext(session_id="test-session")
    with patch("pyautogui.scroll") as mock_scroll:
        res = await execute_tool("scroll_webpage", {"direction": "dwon", "amount": 2}, context)
        assert res["status"] == "success"
        mock_scroll.assert_called_once_with(-240)

@pytest.mark.asyncio
async def test_search_in_page():
    context = ToolContext(session_id="test-session")
    with patch("pyautogui.hotkey") as mock_hotkey, \
         patch("pyautogui.write") as mock_write, \
         patch("pyautogui.press") as mock_press:
        
        res = await execute_tool("search_in_page", {"query": "hello world"}, context)
        assert res["status"] == "success"
        mock_hotkey.assert_called_once_with('ctrl', 'f')
        mock_write.assert_called_once_with("hello world")
        mock_press.assert_called_once_with('enter')

@pytest.mark.asyncio
async def test_go_to_main_page():
    context = ToolContext(session_id="test-session")
    with patch("app.orchestrator.tools.webbrowser.open") as mock_open:
        res = await execute_tool("go_to_main_page", {}, context)
        assert res["status"] == "success"
        mock_open.assert_called_once_with("http://127.0.0.1:8000/")

@pytest.mark.asyncio
async def test_open_new_tab():
    context = ToolContext(session_id="test-session")
    with patch("app.orchestrator.tools.webbrowser.open_new_tab") as mock_open:
        res = await execute_tool("open_new_tab", {"url": "https://github.com"}, context)
        assert res["status"] == "success"
        mock_open.assert_called_once_with("https://github.com")

@pytest.mark.asyncio
async def test_whatsapp_send_message_phone():
    context = ToolContext(session_id="test-session")
    with patch("app.orchestrator.tools.webbrowser.open") as mock_open:
        with patch("app.orchestrator.tools.threading.Thread") as mock_thread:
            res = await execute_tool("whatsapp_send_message", {"phone": "+91 98765 43210", "message": "Hello!"}, context)
            assert res["status"] == "success"
            assert "Opening WhatsApp Web to send message to phone: +919876543210" in res["data"]["result"]
            assert mock_thread.called

@pytest.mark.asyncio
async def test_whatsapp_send_message_contact():
    context = ToolContext(session_id="test-session")
    with patch("app.orchestrator.tools.webbrowser.open") as mock_open:
        with patch("app.orchestrator.tools.threading.Thread") as mock_thread:
            res = await execute_tool("whatsapp_send_message", {"phone": "Nidhish Mangaluru", "message": "hello!"}, context)
            assert res["status"] == "success"
            assert "Opening WhatsApp Web to search and send message to contact: 'Nidhish Mangaluru'" in res["data"]["result"]
            assert mock_thread.called
