"""
WebRTC Dual-DataChannel Voice Bridge for Chief of Agents.

Manages real-time WebRTC data channel communication using aiortc:
- media_channel (Data Channel 1): Transports raw audio bytes/base64 PCM and artifact payloads.
- live_updates_channel (Data Channel 2): Dedicated SSE-style event stream for real-time UI updates
  (captions, thinking states, interruptions, backend tool progress).
"""
import json
import base64
import logging
import asyncio
import time
from typing import Dict, Any, Optional, List
import requests
import websockets

from app.config import (
    DEEPGRAM_API_KEY,
    DEEPGRAM_MODEL,
    DEEPGRAM_STT_URI,
    DEEPGRAM_ENDPOINTING,
    DEEPGRAM_SMART_FORMAT,
    DEEPGRAM_TTS_MODEL,
    DEEPGRAM_TTS_URI,
    CARTESIA_API_KEY,
    CARTESIA_MODEL_ID,
    CARTESIA_VOICE_ID,
    CARTESIA_VERSION,
    CARTESIA_TTS_URI,
)
from app.orchestrator.session_manager import session_manager, SessionState
from app.orchestrator.workflow import run_workflow

logger = logging.getLogger(__name__)


def send_media_payload(media_channel, payload: dict) -> None:
    """Helper to send JSON or binary payload on media_channel."""
    if media_channel and media_channel.readyState == "open":
        try:
            media_channel.send(json.dumps(payload))
        except Exception as e:
            logger.warning(f"[WebRTCBridge] Failed to send media payload: {e}")


def send_live_update(updates_channel, event: dict) -> None:
    """Helper to stream real-time SSE-style event updates on live_updates_channel."""
    if updates_channel and updates_channel.readyState == "open":
        try:
            updates_channel.send(json.dumps(event))
        except Exception as e:
            logger.warning(f"[WebRTCBridge] Failed to send live update: {e}")


async def synthesize_cartesia_tts_webrtc(
    text: str, media_channel, updates_channel, session: SessionState
) -> None:
    """Streams synthesized speech audio chunks from Cartesia TTS to media_channel."""
    if not CARTESIA_API_KEY:
        logger.error("[CartesiaTTS] CARTESIA_API_KEY is not configured.")
        return

    headers = {
        "X-API-Key": CARTESIA_API_KEY,
        "Cartesia-Version": CARTESIA_VERSION,
    }
    url = f"{CARTESIA_TTS_URI}?api_key={CARTESIA_API_KEY}&cartesia_version={CARTESIA_VERSION}"
    context_id = f"ctx-{int(time.time() * 1000)}"
    tts_payload = {
        "model_id": CARTESIA_MODEL_ID,
        "transcript": text,
        "voice": {
            "mode": "id",
            "id": CARTESIA_VOICE_ID,
        },
        "output_format": {
            "container": "raw",
            "encoding": "pcm_s16le",
            "sample_rate": 24000,
        },
        "generation_config": {
            "speed": 1,
            "volume": 1,
        },
        "context_id": context_id,
    }

    try:
        try:
            connect_ctx = websockets.connect(url, additional_headers=headers, open_timeout=4.0)
        except TypeError:
            connect_ctx = websockets.connect(url, extra_headers=headers, open_timeout=4.0)

        async with connect_ctx as cartesia_ws:
            await cartesia_ws.send(json.dumps(tts_payload))
            
            async for raw_msg in cartesia_ws:
                if session.is_interrupted:
                    logger.info("[CartesiaTTS] WebRTC Synthesis interrupted by user.")
                    break
                
                try:
                    data = json.loads(raw_msg)
                    is_chunk = (data.get("status") == "chunk" or data.get("type") == "chunk") and "data" in data
                    is_done = (data.get("status") == "done" or data.get("type") == "done" or data.get("done") is True)

                    if is_chunk:
                        audio_b64 = data["data"]
                        send_media_payload(media_channel, {
                            "type": "audio",
                            "data": audio_b64
                        })
                    elif is_done:
                        logger.info(f"[CartesiaTTS] Synthesis complete for context_id={context_id}")
                        break
                    elif "error" in data:
                        logger.error(f"[CartesiaTTS] Cartesia error: {data['error']}")
                        break
                except Exception as ex:
                    logger.warning(f"[CartesiaTTS] Error processing TTS chunk: {ex}")
                    break

    except Exception as e:
        logger.warning(f"[CartesiaTTS] WebRTC WebSocket streaming failed ({e}), falling back to HTTP synthesis...")
        await synthesize_cartesia_tts_http_webrtc(text, media_channel, session)


async def synthesize_cartesia_tts_http_webrtc(
    text: str, media_channel, session: SessionState
) -> None:
    """HTTP streaming fallback for Cartesia TTS over WebRTC media_channel."""
    if not CARTESIA_API_KEY:
        return

    url = "https://api.cartesia.ai/tts/bytes"
    headers = {
        "X-API-Key": CARTESIA_API_KEY,
        "Cartesia-Version": CARTESIA_VERSION,
        "Content-Type": "application/json",
    }
    payload = {
        "model_id": CARTESIA_MODEL_ID,
        "transcript": text,
        "voice": {"mode": "id", "id": CARTESIA_VOICE_ID},
        "output_format": {"container": "raw", "encoding": "pcm_s16le", "sample_rate": 24000},
        "generation_config": {"speed": 1, "volume": 1},
    }

    try:
        def fetch_stream():
            return requests.post(url, headers=headers, json=payload, stream=True, timeout=10)

        response = await asyncio.to_thread(fetch_stream)
        if response.status_code == 200:
            for chunk in response.iter_content(chunk_size=4096):
                if session.is_interrupted:
                    break
                if chunk:
                    b64_chunk = base64.b64encode(chunk).decode("utf-8")
                    send_media_payload(media_channel, {"type": "audio", "data": b64_chunk})
    except Exception as e:
        logger.error(f"[CartesiaTTS] WebRTC HTTP synthesis failed: {e}")


async def synthesize_tts_webrtc(
    text: str, media_channel, updates_channel, session: SessionState
) -> None:
    """Dispatches TTS synthesis over WebRTC data channels."""
    if CARTESIA_API_KEY:
        try:
            await synthesize_cartesia_tts_webrtc(text, media_channel, updates_channel, session)
            return
        except Exception as e:
            logger.warning(f"[TTSDispatcher] WebRTC Cartesia failed ({e})...")


async def async_run_webrtc_bridge(
    media_channel, updates_channel, session_id: str
) -> None:
    """Manages Deepgram STT + Orchestrator + Cartesia TTS over WebRTC data channels."""
    if not DEEPGRAM_API_KEY:
        raise ValueError("DEEPGRAM_API_KEY environment variable is not configured.")

    session: SessionState = session_manager.get(session_id)
    deepgram_url = (
        f"{DEEPGRAM_STT_URI}?encoding=linear16&sample_rate=16000&channels=1"
        f"&model={DEEPGRAM_MODEL}&language=en&smart_format={str(DEEPGRAM_SMART_FORMAT).lower()}"
        f"&interim_results=true&endpointing={DEEPGRAM_ENDPOINTING}"
    )

    headers = {"Authorization": f"Token {DEEPGRAM_API_KEY}"}

    # 1. Initialize audio queue & attach media_channel listener IMMEDIATELY
    # so early microphone audio chunks sent during STT connection handshake are not dropped
    audio_queue: asyncio.Queue = asyncio.Queue()
    utterance_chunks: List[str] = []

    @media_channel.on("message")
    def on_media_message(message):
        if isinstance(message, (bytes, bytearray, memoryview)):
            audio_queue.put_nowait(bytes(message))
        elif isinstance(message, str):
            try:
                payload = json.loads(message)
                ptype = payload.get("type")
                if ptype == "interrupt":
                    logger.info(f"[WebRTCBridge] Interruption signal received for session {session_id}")
                    session.is_interrupted = True
                    utterance_chunks.clear()
                    send_live_update(updates_channel, {"type": "interrupted"})
                elif ptype == "audio" and "data" in payload:
                    audio_bytes = base64.b64decode(payload["data"])
                    audio_queue.put_nowait(audio_bytes)
            except json.JSONDecodeError:
                pass

    logger.info(f"[WebRTCBridge] Connecting to Deepgram STT for session {session_id}")

    try:
        try:
            connect_ctx = websockets.connect(
                deepgram_url,
                additional_headers=headers,
                ping_interval=10,
                ping_timeout=10,
            )
        except TypeError:
            connect_ctx = websockets.connect(
                deepgram_url,
                extra_headers=headers,
                ping_interval=10,
                ping_timeout=10,
            )

        async with connect_ctx as dg_ws:

            # Task 0: Keepalive Ping for Deepgram STT
            async def send_keepalive():
                logger.info(f"[WebRTCBridge] Started keepalive task for session {session_id}")
                while True:
                    try:
                        await dg_ws.send(json.dumps({"type": "KeepAlive"}))
                        await asyncio.sleep(2.0)
                    except (asyncio.CancelledError, websockets.ConnectionClosed):
                        break
                    except Exception as ex:
                        logger.warning(f"[WebRTCBridge] Keepalive ping failed: {ex}")
                        await asyncio.sleep(1.0)

            keepalive_task = asyncio.create_task(send_keepalive())
            session.add_task(keepalive_task)

            utterance_chunks.clear()

            async def forward_audio_queue_to_deepgram():
                try:
                    while True:
                        audio_bytes = await audio_queue.get()
                        await dg_ws.send(audio_bytes)
                        audio_queue.task_done()
                except (asyncio.CancelledError, Exception) as e:
                    logger.info(f"[WebRTCBridge] Audio forward loop ended: {e}")

            # Task 2: Deepgram STT -> LLM Orchestrator -> Cartesia TTS -> live_updates_channel & media_channel
            async def forward_deepgram_to_client():
                active_tts_task: Optional[asyncio.Task] = None

                try:
                    async for raw_resp in dg_ws:
                        try:
                            res = json.loads(raw_resp)
                        except json.JSONDecodeError:
                            continue

                        if res.get("type") == "Results":
                            channel = res.get("channel", {})
                            alternatives = channel.get("alternatives", [])
                            if not alternatives:
                                continue

                            transcript = alternatives[0].get("transcript", "").strip()
                            is_final = res.get("is_final", False)
                            speech_final = res.get("speech_final", False)

                            if transcript:
                                logger.info(f"[WebRTCBridge] STT output: '{transcript}' (is_final={is_final}, speech_final={speech_final})")
                                if is_final or speech_final:
                                    utterance_chunks.append(transcript)
                                
                                # Stream interim/final user caption to client HUD in real-time
                                send_live_update(updates_channel, {
                                    "type": "caption",
                                    "role": "user",
                                    "text": transcript
                                })

                            if speech_final and utterance_chunks:
                                full_transcript = " ".join(chunk.strip() for chunk in utterance_chunks if chunk.strip()).strip()
                                utterance_chunks.clear()

                                if full_transcript:
                                    logger.info(f"[WebRTCBridge] Final Utterance STT Transcript: '{full_transcript}'")
                                    session.is_interrupted = False

                                    # Send thinking SSE live update
                                    send_live_update(updates_channel, {"type": "thinking"})

                                    if active_tts_task and not active_tts_task.done():
                                        active_tts_task.cancel()

                                    try:
                                        async def handle_progress(status_text: str):
                                            logger.info(f"[WebRTCBridge] Backend progress update: '{status_text}'")
                                            if not session.is_interrupted:
                                                send_live_update(updates_channel, {
                                                    "type": "caption",
                                                    "role": "assistant",
                                                    "text": status_text
                                                })
                                                await synthesize_tts_webrtc(status_text, media_channel, updates_channel, session)

                                        async def handle_tool_event(event: dict):
                                            logger.info(f"[WebRTCBridge] Dispatching tool event to UI: {event.get('type')} - {event.get('name', event.get('call_id'))}")
                                            if not session.is_interrupted:
                                                send_live_update(updates_channel, event)

                                        session_manager.add_turn(session_id, "user", full_transcript)

                                        wf_res = await run_workflow(
                                            message=full_transcript,
                                            recent_turns=session_manager.get_recent_turns(session_id),
                                            session_id=session_id,
                                            on_progress=handle_progress,
                                            on_tool_event=handle_tool_event,
                                        )

                                        reply = wf_res.get("reply", "")
                                        session_manager.add_turn(session_id, "assistant", reply)

                                        # Stream assistant final caption SSE live update
                                        send_live_update(updates_channel, {
                                            "type": "caption",
                                            "role": "assistant",
                                            "text": reply
                                        })

                                        if reply and not session.is_interrupted:
                                            active_tts_task = asyncio.create_task(
                                                synthesize_tts_webrtc(reply, media_channel, updates_channel, session)
                                            )

                                    except Exception as wf_err:
                                        logger.error(f"[WebRTCBridge] Workflow execution failed: {wf_err}")
                                        send_live_update(updates_channel, {
                                            "type": "error",
                                            "message": f"Orchestrator error: {wf_err}"
                                        })

                except asyncio.CancelledError:
                    logger.info(f"[WebRTCBridge] Deepgram receiver cancelled for session {session_id}")
                except Exception as e:
                    logger.error(f"[WebRTCBridge] Deepgram processing loop ended: {e}")

            try:
                await asyncio.gather(
                    forward_audio_queue_to_deepgram(),
                    forward_deepgram_to_client(),
                    return_exceptions=True
                )
            finally:
                keepalive_task.cancel()

    except Exception as e:
        logger.error(f"[WebRTCBridge] Connection setup error: {e}")
        raise
