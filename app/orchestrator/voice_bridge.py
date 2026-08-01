"""
Voice Bridge Dispatcher with Multi-Provider Support.

Attempts to run primary voice model (Deepgram STT + Cartesia TTS Pipeline).
If unconfigured or an error occurs during connection setup, falls back to
OpenAI GPT-4o Realtime or Gemini Multimodal Live API.
"""
import logging
from fastapi import WebSocket

from app.config import DEEPGRAM_API_KEY, CARTESIA_API_KEY, OPENAI_API_KEY, GEMINI_API_KEY
from app.orchestrator.deepgram_cartesia_bridge import async_run_deepgram_cartesia_bridge
from app.orchestrator.openai_client import async_run_openai_realtime_bridge
from app.orchestrator.gemini_client import async_run_gemini_live_bridge

logger = logging.getLogger(__name__)


async def async_run_voice_bridge(
    client_ws: WebSocket, session_id: str
) -> None:
    """Dispatches live WebSocket session prioritizing Deepgram STT + Cartesia TTS."""
    deepgram_cartesia_available = bool(
        DEEPGRAM_API_KEY and DEEPGRAM_API_KEY.strip() and
        CARTESIA_API_KEY and CARTESIA_API_KEY.strip()
    )

    if deepgram_cartesia_available:
        try:
            logger.info(f"[VoiceBridge] Initiating session {session_id} using primary provider: Deepgram STT + Cartesia TTS")
            await async_run_deepgram_cartesia_bridge(client_ws, session_id)
            return
        except Exception as e:
            logger.warning(
                f"[VoiceBridge] Primary provider (Deepgram STT + Cartesia TTS) failed for session {session_id}: {e}. "
                "Initiating fallback to OpenAI Realtime / Gemini Live API..."
            )

    openai_available = bool(OPENAI_API_KEY and OPENAI_API_KEY.strip())
    if openai_available:
        try:
            logger.info(f"[VoiceBridge] Initiating session {session_id} using fallback provider: OpenAI GPT-4o Realtime")
            await async_run_openai_realtime_bridge(client_ws, session_id)
            return
        except Exception as e:
            logger.warning(
                f"[VoiceBridge] Provider OpenAI GPT-4o Realtime failed for session {session_id}: {e}. "
                "Initiating automatic fallback to Gemini Multimodal Live API..."
            )

    # Fallback to Gemini Live API
    if not GEMINI_API_KEY:
        logger.error("[VoiceBridge] All voice model providers are unconfigured or failed!")
        await client_ws.send_json({
            "type": "error",
            "message": "No available live voice model provider configured (Deepgram/Cartesia, OpenAI, and Gemini all failed)."
        })
        return

    logger.info(f"[VoiceBridge] Initiating session {session_id} using fallback provider: Gemini Live API")
    try:
        await async_run_gemini_live_bridge(client_ws, session_id)
    except Exception as e:
        logger.error(f"[VoiceBridge] Fallback provider (Gemini Live API) also failed for session {session_id}: {e}")
        await client_ws.send_json({
            "type": "error",
            "message": f"Voice bridge error: {e}"
        })

