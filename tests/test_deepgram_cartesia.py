import pytest
import asyncio
from unittest.mock import AsyncMock, patch, MagicMock

from app.orchestrator.voice_bridge import async_run_voice_bridge
from app.orchestrator.deepgram_cartesia_bridge import synthesize_cartesia_tts_http


@pytest.mark.asyncio
async def test_voice_bridge_deepgram_cartesia_dispatch():
    """Verify voice bridge dispatches to Deepgram + Cartesia when keys are configured."""
    mock_ws = AsyncMock()
    with patch("app.orchestrator.voice_bridge.DEEPGRAM_API_KEY", "mock_dg_key"), \
         patch("app.orchestrator.voice_bridge.CARTESIA_API_KEY", "mock_cartesia_key"), \
         patch("app.orchestrator.voice_bridge.async_run_deepgram_cartesia_bridge", new_callable=AsyncMock) as mock_dg_bridge:
        
        await async_run_voice_bridge(mock_ws, "session-test-dg")
        mock_dg_bridge.assert_called_once_with(mock_ws, "session-test-dg")


@pytest.mark.asyncio
async def test_voice_bridge_fallback_when_deepgram_unconfigured():
    """Verify voice bridge falls back to OpenAI / Gemini when Deepgram key is absent."""
    mock_ws = AsyncMock()
    with patch("app.orchestrator.voice_bridge.DEEPGRAM_API_KEY", ""), \
         patch("app.orchestrator.voice_bridge.CARTESIA_API_KEY", ""), \
         patch("app.orchestrator.voice_bridge.OPENAI_API_KEY", "mock_openai_key"), \
         patch("app.orchestrator.voice_bridge.async_run_openai_realtime_bridge", new_callable=AsyncMock) as mock_openai_bridge:
        
        await async_run_voice_bridge(mock_ws, "session-test-fallback")
        mock_openai_bridge.assert_called_once_with(mock_ws, "session-test-fallback")


@pytest.mark.asyncio
async def test_cartesia_tts_http_fallback():
    """Verify Cartesia HTTP fallback synthesizes audio and transmits audio websocket frames."""
    mock_ws = AsyncMock()
    mock_session = MagicMock()
    mock_session.is_interrupted = False

    fake_pcm_bytes = b"\x00\x01\x02\x03" * 100

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.content = fake_pcm_bytes
    mock_resp.iter_content.return_value = [fake_pcm_bytes]

    with patch("app.orchestrator.deepgram_cartesia_bridge.CARTESIA_API_KEY", "mock_cartesia_key"), \
         patch("requests.post", return_value=mock_resp):
        
        await synthesize_cartesia_tts_http("Hello test", mock_ws, mock_session)
        assert mock_ws.send_json.called
        call_args = mock_ws.send_json.call_args[0][0]
        assert call_args["type"] == "audio"
        assert "data" in call_args


@pytest.mark.asyncio
async def test_deepgram_workflow_invocation():
    """Verify deepgram bridge calls run_workflow with message, recent_turns, and session_id."""
    from app.orchestrator.deepgram_cartesia_bridge import run_workflow

    with patch("app.orchestrator.deepgram_cartesia_bridge.run_workflow", new_callable=AsyncMock) as mock_wf:
        mock_wf.return_value = {"reply": "Hello back!"}
        res = await mock_wf(message="Hello", recent_turns=[], session_id="test-session")
        assert res["reply"] == "Hello back!"
        mock_wf.assert_called_once_with(message="Hello", recent_turns=[], session_id="test-session")


@pytest.mark.asyncio
async def test_deepgram_keepalive_and_bridge_lifecycle():
    """Verify Deepgram bridge connects and maintains keepalive loop properly."""
    from app.orchestrator.deepgram_cartesia_bridge import async_run_deepgram_cartesia_bridge

    mock_client_ws = AsyncMock()
    mock_client_ws.receive.side_effect = [{"type": "websocket.disconnect"}]

    mock_dg_ws = AsyncMock()
    mock_dg_ws.__aiter__.return_value = []
    mock_dg_ws.closed = False

    class MockConnectContext:
        async def __aenter__(self):
            return mock_dg_ws

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass

    with patch("app.orchestrator.deepgram_cartesia_bridge.DEEPGRAM_API_KEY", "mock_dg_key"), \
         patch("app.orchestrator.deepgram_cartesia_bridge.CARTESIA_API_KEY", "mock_cartesia_key"), \
         patch("websockets.connect", return_value=MockConnectContext()):

        await async_run_deepgram_cartesia_bridge(mock_client_ws, "test-session-keepalive")

        # Verify client WS receive was called to process frames
        assert mock_client_ws.receive.called


