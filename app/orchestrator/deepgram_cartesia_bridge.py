"""
Deepgram STT + Cartesia TTS Voice Bridge Client.

Handles streaming speech recognition via Deepgram (wss://api.deepgram.com/v1/listen),
orchestrator LLM workflow routing, and ultra-fast speech synthesis via Cartesia TTS
(wss://api.cartesia.ai/tts/websocket or HTTP bytes).
"""
import json
import base64
import logging
import asyncio
import time
from typing import Dict, Any, Optional
from fastapi import WebSocket, WebSocketDisconnect
import websockets
import requests

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


async def synthesize_cartesia_tts_ws(
    text: str, client_ws: WebSocket, session: SessionState
) -> None:
    """Streams synthesized speech audio chunks from Cartesia TTS WebSocket back to client."""
    if not CARTESIA_API_KEY:
        logger.error("[CartesiaTTS] CARTESIA_API_KEY is not configured.")
        return

    headers = {
        "X-API-Key": CARTESIA_API_KEY,
        "Cartesia-Version": CARTESIA_VERSION,
    }
    
    # WebSocket connection URL for Cartesia
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
                    logger.info("[CartesiaTTS] Synthesis interrupted by user.")
                    break
                
                try:
                    data = json.loads(raw_msg)
                    is_chunk = (data.get("status") == "chunk" or data.get("type") == "chunk") and "data" in data
                    is_done = (data.get("status") == "done" or data.get("type") == "done" or data.get("done") is True)

                    if is_chunk:
                        # Base64 PCM audio chunk
                        audio_b64 = data["data"]
                        logger.info(f"[CartesiaTTS] Sending audio chunk ({len(audio_b64)} b64 bytes) to client.")
                        try:
                            await client_ws.send_json({
                                "type": "audio",
                                "data": audio_b64
                            })
                        except (RuntimeError, WebSocketDisconnect):
                            logger.info("[CartesiaTTS] Client disconnected during streaming.")
                            break
                    elif is_done:
                        logger.info(f"[CartesiaTTS] Synthesis complete for context_id={context_id}")
                        break
                    elif "error" in data:
                        logger.error(f"[CartesiaTTS] Cartesia error: {data['error']}")
                        break

                except (RuntimeError, WebSocketDisconnect):
                    break
                except Exception as ex:
                    logger.warning(f"[CartesiaTTS] Error processing TTS chunk: {ex}")
                    break

    except Exception as e:
        logger.warning(f"[CartesiaTTS] WebSocket streaming failed ({e}), falling back to HTTP synthesis...")
        await synthesize_cartesia_tts_http(text, client_ws, session)


async def synthesize_cartesia_tts_http(
    text: str, client_ws: WebSocket, session: SessionState
) -> None:
    """HTTP streaming fallback for Cartesia TTS synthesis."""
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
                    try:
                        await client_ws.send_json({
                            "type": "audio",
                            "data": b64_chunk
                        })
                    except (RuntimeError, WebSocketDisconnect):
                        break
                    await asyncio.sleep(0.01)
        else:
            logger.error(f"[CartesiaTTS] HTTP status {response.status_code}: {response.text}")
    except Exception as e:
        logger.error(f"[CartesiaTTS] HTTP synthesis failed: {e}")



async def synthesize_deepgram_tts(
    text: str, client_ws: WebSocket, session: SessionState
) -> None:
    """Synthesizes speech audio using Deepgram Aura TTS API (e.g. aura-2-neptune-en)."""
    if not DEEPGRAM_API_KEY:
        logger.error("[DeepgramTTS] DEEPGRAM_API_KEY is not configured.")
        return

    url = f"{DEEPGRAM_TTS_URI}?model={DEEPGRAM_TTS_MODEL}&encoding=linear16&sample_rate=24000"
    headers = {
        "Authorization": f"Token {DEEPGRAM_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {"text": text}

    try:
        def fetch_audio():
            return requests.post(url, headers=headers, json=payload, timeout=10)

        response = await asyncio.to_thread(fetch_audio)
        if response.status_code == 200:
            audio_bytes = response.content
            chunk_size = 4096
            for i in range(0, len(audio_bytes), chunk_size):
                if session.is_interrupted:
                    break
                chunk = audio_bytes[i : i + chunk_size]
                b64_chunk = base64.b64encode(chunk).decode("utf-8")
                try:
                    await client_ws.send_json({
                        "type": "audio",
                        "data": b64_chunk
                    })
                except (RuntimeError, WebSocketDisconnect):
                    break
                await asyncio.sleep(0.02)

        else:
            logger.error(f"[DeepgramTTS] HTTP status {response.status_code}: {response.text}")
    except Exception as e:
        logger.error(f"[DeepgramTTS] Deepgram TTS synthesis failed: {e}")


async def synthesize_tts(
    text: str, client_ws: WebSocket, session: SessionState
) -> None:
    """Dispatches TTS synthesis to Cartesia TTS or Deepgram Aura TTS fallback."""
    if CARTESIA_API_KEY:
        try:
            await synthesize_cartesia_tts_ws(text, client_ws, session)
            return
        except Exception as e:
            logger.warning(f"[TTSDispatcher] Cartesia TTS failed ({e}), falling back to Deepgram Aura TTS...")
    
    await synthesize_deepgram_tts(text, client_ws, session)



async def async_run_deepgram_cartesia_bridge(
    client_ws: WebSocket, session_id: str
) -> None:
    """Manages the lifecycle of a Deepgram STT + Cartesia TTS voice session over WebSockets."""
    if not DEEPGRAM_API_KEY:
        raise ValueError("DEEPGRAM_API_KEY environment variable is not configured.")
    if not CARTESIA_API_KEY:
        raise ValueError("CARTESIA_API_KEY environment variable is not configured.")

    session: SessionState = session_manager.get(session_id)
    deepgram_url = (
        f"{DEEPGRAM_STT_URI}?encoding=linear16&sample_rate=16000&channels=1"
        f"&model={DEEPGRAM_MODEL}&language=en&smart_format={str(DEEPGRAM_SMART_FORMAT).lower()}"
        f"&interim_results=true&endpointing={DEEPGRAM_ENDPOINTING}"
    )


    headers = {
        "Authorization": f"Token {DEEPGRAM_API_KEY}",
    }

    logger.info(
        f"[DeepgramCartesiaBridge] Connecting to Deepgram STT: model={DEEPGRAM_MODEL}, session={session_id}"
    )

    try:
        try:
            connect_ctx = websockets.connect(deepgram_url, additional_headers=headers)
        except TypeError:
            connect_ctx = websockets.connect(deepgram_url, extra_headers=headers)

        async with connect_ctx as dg_ws:

            # Task 0: Keepalive task for Deepgram STT to prevent 1011 timeout during silence
            async def send_keepalive():
                logger.warning(f"[DeepgramBridge] Started keepalive task for session {session_id}")
                while True:
                    try:
                        await dg_ws.send(json.dumps({"type": "KeepAlive"}))
                        logger.warning(f"[DeepgramBridge] Sent KeepAlive ping for session {session_id}")
                        await asyncio.sleep(2.0)
                    except (asyncio.CancelledError, websockets.ConnectionClosed):
                        logger.warning(f"[DeepgramBridge] Keepalive task closed for session {session_id}")
                        break
                    except Exception as ex:
                        logger.warning(f"[DeepgramBridge] Keepalive ping failed for session {session_id}: {ex}")
                        await asyncio.sleep(1.0)

            keepalive_task = asyncio.create_task(send_keepalive())

            utterance_chunks: List[str] = []
            utterance_chunks.clear()  # (a) Session initialization clear

            # Task 1: Client -> Deepgram STT
            async def forward_client_to_deepgram():
                try:
                    while True:
                        msg = await client_ws.receive()
                        if msg.get("type") == "websocket.disconnect":
                            utterance_chunks.clear()  # (c) Disconnect clear
                            break
                        if "bytes" in msg and msg["bytes"]:
                            # Binary PCM 16kHz audio from browser
                            audio_bytes = msg["bytes"]
                            await dg_ws.send(audio_bytes)
                        elif "text" in msg and msg["text"]:
                            try:
                                payload = json.loads(msg["text"])
                                ptype = payload.get("type")
                                if ptype == "interrupt":
                                    logger.info(f"[DeepgramBridge] Interruption signal received for session {session_id}")
                                    session.is_interrupted = True
                                    utterance_chunks.clear()  # (c) Interruption clear
                                    await client_ws.send_json({"type": "interrupted"})
                                elif ptype == "audio" and "data" in payload:
                                    # Base64 encoded audio
                                    audio_bytes = base64.b64decode(payload["data"])
                                    await dg_ws.send(audio_bytes)
                            except json.JSONDecodeError:
                                pass
                except (WebSocketDisconnect, RuntimeError, asyncio.CancelledError):
                    logger.info(f"[DeepgramBridge] Client WS disconnected for session {session_id}")
                    utterance_chunks.clear()  # (c) Disconnect clear
                except Exception as e:
                    logger.warning(f"[DeepgramBridge] Client forward loop ended: {e}")

            # Task 2: Deepgram STT -> LLM Orchestrator -> Cartesia TTS -> Client
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

                            # Accept transcript if is_final OR speech_final
                            if transcript and (is_final or speech_final):
                                utterance_chunks.append(transcript)
                                # Send interim user caption
                                await client_ws.send_json({
                                    "type": "caption",
                                    "role": "user",
                                    "text": transcript
                                })

                            if speech_final and utterance_chunks:
                                # Join buffered chunks for full user utterance with clean single-space separation
                                full_transcript = " ".join(chunk.strip() for chunk in utterance_chunks if chunk.strip()).strip()
                                utterance_chunks.clear()  # (b) Dispatch clear

                                logger.info(f"[DeepgramBridge] Utterance STT Transcript: '{full_transcript}'")
                                
                                # Reset interruption state & set visualizer thinking
                                session.is_interrupted = False
                                await client_ws.send_json({"type": "thinking"})

                                # If previous TTS is still playing, cancel it
                                if active_tts_task and not active_tts_task.done():
                                    active_tts_task.cancel()

                                # Execute Orchestrator LLM Workflow with backend progress acknowledgments
                                try:
                                    async def handle_progress(status_text: str):
                                        logger.info(f"[DeepgramBridge] Backend progress acknowledgment: '{status_text}'")
                                        if not session.is_interrupted:
                                            try:
                                                await client_ws.send_json({
                                                    "type": "caption",
                                                    "role": "assistant",
                                                    "text": status_text
                                                })
                                                await synthesize_tts(status_text, client_ws, session)
                                            except (RuntimeError, WebSocketDisconnect):
                                                pass

                                    # Save user turn to session history before workflow execution
                                    session_manager.add_turn(session_id, "user", full_transcript)

                                    wf_res = await run_workflow(
                                        message=full_transcript,
                                        recent_turns=session_manager.get_recent_turns(session_id),
                                        session_id=session_id,
                                        on_progress=handle_progress
                                    )

                                    reply = wf_res.get("reply", "")

                                    # Save assistant turn to session history
                                    session_manager.add_turn(session_id, "assistant", reply)

                                    # Send assistant caption
                                    await client_ws.send_json({
                                        "type": "caption",
                                        "role": "assistant",
                                        "text": reply
                                    })

                                    # Synthesize and stream speech via Cartesia or Deepgram Aura TTS
                                    if reply and not session.is_interrupted:
                                        active_tts_task = asyncio.create_task(
                                            synthesize_tts(reply, client_ws, session)
                                        )

                                except Exception as wf_err:
                                    logger.error(f"[DeepgramBridge] Workflow execution failed: {wf_err}")
                                    await client_ws.send_json({
                                        "type": "error",
                                        "message": f"Orchestrator error: {wf_err}"
                                    })

                except asyncio.CancelledError:
                    logger.info(f"[DeepgramBridge] Deepgram receiver cancelled for session {session_id}")
                except Exception as e:
                    logger.error(f"[DeepgramBridge] Deepgram processing loop ended: {e}")

            try:
                # Run client reader and Deepgram listener concurrently
                await asyncio.gather(
                    forward_client_to_deepgram(),
                    forward_deepgram_to_client(),
                    return_exceptions=True
                )
            finally:
                keepalive_task.cancel()



    except Exception as e:
        logger.error(f"[DeepgramCartesiaBridge] Error establishing connection: {e}")
        raise
