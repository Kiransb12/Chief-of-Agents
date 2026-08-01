import json
import base64
import logging
import asyncio
import time
from typing import Dict, Any
from fastapi import WebSocket, WebSocketDisconnect
import websockets

from app.config import (
    GEMINI_API_KEY,
    GEMINI_LIVE_MODEL,
    GEMINI_LIVE_VOICE,
    GEMINI_LIVE_URI,
)
from app.orchestrator.tool_executor import tool_registry, execute_tool, ToolContext, make_response
from app.orchestrator.session_manager import session_manager, SessionState
from app.orchestrator.memory import load_semantic_memory

logger = logging.getLogger(__name__)


async def async_run_gemini_live_bridge(
    client_ws: WebSocket, session_id: str
) -> None:
    """Manages the lifecycle of a Gemini Live session over WebSockets.

    Coordinates raw audio data forwarding, tool executions, and caption streams.
    """
    session: SessionState = session_manager.get(session_id)
    gemini_uri = f"{GEMINI_LIVE_URI}?key={GEMINI_API_KEY}"

    # Load semantic memories to inject directly so agent knows user profile instantly
    facts = load_semantic_memory()
    facts_str = "\n".join(f"- {f}" for f in facts) if facts else "No semantic facts recorded yet."

    # 1. Base Setup Prompt Instruction
    system_prompt = (
        "You are a helpful, professional personal AI assistant.\n"
        "Here are the active, summarized semantic memory facts about the user that you must know:\n"
        f"{facts_str}\n\n"
        "You have access to the user's personal documents, notes, and detailed info through the 'retrieve_rag_context' tool.\n"
        "CRITICAL: If the user asks about detailed personal files, document details, or project tasks, "
        "you MUST call 'retrieve_rag_context' first before answering. Be concise, conversational, and direct in your speech."
    )

    logger.info(
        f"[GeminiClient] Initiating WebSocket connection to Gemini Live: {GEMINI_LIVE_MODEL}"
    )

    try:
        async with websockets.connect(gemini_uri) as gemini_ws:
            # 2. Send initial Setup Frame
            setup_frame = {
                "setup": {
                    "model": GEMINI_LIVE_MODEL,
                    "generationConfig": {
                        "responseModalities": ["AUDIO"],
                        "speechConfig": {
                            "voiceConfig": {
                                "prebuiltVoiceConfig": {
                                    "voiceName": GEMINI_LIVE_VOICE
                                }
                            }
                        },
                    },
                    "systemInstruction": {
                        "parts": [{"text": system_prompt}]
                    },
                    "tools": [{"functionDeclarations": tool_registry.schemas}],
                    "inputAudioTranscription": {},
                    "realtimeInputConfig": {
                        "automaticActivityDetection": {
                            "disabled": False
                        }
                    }
                }
            }
            await gemini_ws.send(json.dumps(setup_frame))
            logger.info("[GeminiClient] Setup frame sent successfully.")

            # Define bridging sub-tasks
            async def forward_client_to_gemini():
                """Forwards client microphone PCM inputs to Gemini Live."""
                try:
                    while True:
                        data = await client_ws.receive()
                        if "bytes" in data:
                            # Audio chunk (PCM mono 16kHz)
                            raw_pcm = data["bytes"]
                            b64_chunk = base64.b64encode(raw_pcm).decode(
                                "utf-8"
                            )
                            media_frame = {
                                "realtimeInput": {
                                    "mediaChunks": [
                                        {
                                            "mimeType": "audio/pcm;rate=16000",
                                            "data": b64_chunk,
                                        }
                                    ]
                                }
                            }
                            await gemini_ws.send(json.dumps(media_frame))

                        elif "text" in data:
                            text_payload = json.loads(data["text"])
                            # Check for user-driven commands (e.g. manual interruption)
                            if text_payload.get("type") == "interrupt":
                                logger.info(
                                    "[GeminiClient] Client requested interruption. Flushing audio."
                                )
                                session.is_interrupted = True
                            elif text_payload.get("type") == "text":
                                user_message = text_payload.get("message")
                                logger.info(
                                    f"[GeminiClient] Client sent text input: {user_message}"
                                )
                                # Interrupt any active agent playback before sending text query
                                session.is_interrupted = True
                                await client_ws.send_json({"type": "interrupted"})
                                
                                # Send text content turn to Gemini Live WebSocket
                                gemini_text_frame = {
                                    "clientContent": {
                                        "turns": [
                                            {
                                                "role": "user",
                                                "parts": [{"text": user_message}]
                                            }
                                        ],
                                        "turnComplete": True
                                    }
                                }
                                await gemini_ws.send(json.dumps(gemini_text_frame))

                except WebSocketDisconnect:
                    logger.info("[GeminiClient] Client WebSocket disconnected.")
                except Exception as e:
                    logger.error(
                        f"[GeminiClient] Error in client-to-gemini loop: {e}"
                    )
                    session.metrics["errors"].append(f"ClientLoop: {e}")

            async def forward_gemini_to_client():
                """Forwards Gemini response streams (audio/text) and handles tool calls."""
                # Buffer for accumulating word-by-word user speech transcription.
                # Gemini sends inputTranscription in small chunks (often single words);
                # we stream each chunk to the browser for live display but only save
                # the full concatenated utterance as one session turn when the model's
                # turn completes — so memory consolidation sees coherent sentences,
                # not word salad.
                user_transcript_buffer = []

                try:
                    async for raw_message in gemini_ws:
                        msg = json.loads(raw_message)

                        # Handle server audio and text output
                        if "serverContent" in msg:
                            content = msg["serverContent"]

                            # Accumulate user speech transcription chunks
                            if "inputTranscription" in content:
                                user_transcription = content["inputTranscription"]
                                user_txt = user_transcription.get("text", "")
                                if user_txt:
                                    user_transcript_buffer.append(user_txt)
                                    # Stream each chunk to browser for live captions
                                    await client_ws.send_json(
                                        {
                                            "type": "caption",
                                            "role": "user",
                                            "text": user_txt,
                                        }
                                    )

                            # Check for interruptions
                            if content.get("interrupted"):
                                logger.info(
                                    "[GeminiClient] Gemini Live signal: Interrupted."
                                )
                                await client_ws.send_json(
                                    {"type": "interrupted"}
                                )
                                session.is_interrupted = True
                                continue

                            # Flush user transcript buffer and reset interruption flag
                            # when the model's turn is complete
                            if content.get("turnComplete"):
                                logger.info(
                                    "[GeminiClient] Turn complete. Resetting interruption flag."
                                )
                                session.is_interrupted = False

                                # Flush buffered user transcription as one turn
                                if user_transcript_buffer:
                                    full_utterance = " ".join(user_transcript_buffer).strip()
                                    if full_utterance:
                                        logger.info(f"[GeminiClient] Flushing user transcript: {full_utterance}")
                                        session_manager.add_turn(
                                            session_id, "user", full_utterance
                                        )
                                    user_transcript_buffer.clear()

                            if session.is_interrupted:
                                continue

                            # Audio inline data
                            model_turn = content.get("modelTurn", {})
                            parts = model_turn.get("parts", [])
                            for part in parts:
                                # Captions / Text output
                                if "text" in part:
                                    if part.get("thought"):
                                        continue
                                    txt = part["text"]
                                    # Filter out internal chain-of-thought reasoning text blocks
                                    if txt.strip().startswith("**") or "I need to access" in txt or "The plan is to" in txt:
                                        continue
                                    # Forward to browser for live captions
                                    await client_ws.send_json(
                                        {
                                            "type": "caption",
                                            "role": "assistant",
                                            "text": txt,
                                        }
                                    )
                                    session_manager.add_turn(
                                        session_id, "assistant", txt
                                    )

                                # Audio chunk (PCM mono 24kHz)
                                if "inlineData" in part:
                                    inline = part["inlineData"]
                                    b64_audio = inline.get("data")
                                    if b64_audio:
                                        await client_ws.send_json(
                                            {
                                                "type": "audio",
                                                "data": b64_audio,
                                            }
                                        )

                        # Handle Tool Calls concurrently
                        elif "toolCall" in msg:
                            tool_call = msg["toolCall"]
                            function_calls = tool_call.get("functionCalls", [])

                            # Send thinking signal to the browser visualizer
                            await client_ws.send_json({"type": "thinking"})

                            # Build tasks to execute tools in parallel
                            tasks = []
                            for fc in function_calls:
                                name = fc.get("name")
                                args = fc.get("args", {})
                                call_id = fc.get("id")
                                logger.info(
                                    f"[GeminiClient] Gemini tool call requested: {name}({args})"
                                )

                                # Context injection
                                context = ToolContext(session_id=session_id)

                                async def run_and_notify(n=name, a=args, c_id=call_id, ctx=context, s=session):
                                    try:
                                        await client_ws.send_json({
                                            "type": "tool_start",
                                            "name": n,
                                            "args": a,
                                            "call_id": c_id
                                        })
                                    except Exception as err:
                                        logger.error(f"Error sending tool_start event: {err}")

                                    res = await _run_and_format_tool(n, a, c_id, ctx, s)

                                    try:
                                        await client_ws.send_json({
                                            "type": "tool_complete",
                                            "name": n,
                                            "args": a,
                                            "call_id": c_id,
                                            "output": res["response"]["output"]
                                        })
                                    except Exception as err:
                                        logger.error(f"Error sending tool_complete event: {err}")
                                    return res

                                tasks.append(run_and_notify())

                            # Concurrently execute tool calls using gather
                            tool_responses = await asyncio.gather(*tasks)

                            # Send tool responses back to Gemini
                            response_frame = {
                                "toolResponse": {
                                    "functionResponses": tool_responses
                                }
                            }
                            await gemini_ws.send(json.dumps(response_frame))

                except websockets.exceptions.ConnectionClosed:
                    logger.info(
                        "[GeminiClient] Gemini Live connection closed."
                    )
                except Exception as e:
                    logger.error(
                        f"[GeminiClient] Error in gemini-to-client loop: {e}"
                    )
                    session.metrics["errors"].append(f"GeminiLoop: {e}")

            # Run loops concurrently
            await asyncio.gather(
                forward_client_to_gemini(), forward_gemini_to_client()
            )

    except Exception as e:
        logger.error(f"[GeminiClient] Failed to execute Live session bridge: {e}")
        session.metrics["errors"].append(f"BridgeSetup: {e}")
        await client_ws.send_json({"type": "error", "message": str(e)})


async def _run_and_format_tool(
    name: str,
    args: dict,
    call_id: str,
    context: ToolContext,
    session: SessionState,
) -> dict:
    """Executes a tool and wraps its response in the standardized schemas."""
    # Track metrics
    session.metrics["tool_call_count"] += 1
    t0 = time.time()

    # Wrap in wait_for timeout (8 seconds)
    try:
        res = await asyncio.wait_for(
            execute_tool(name, args, context), timeout=8.0
        )
    except asyncio.TimeoutError:
        logger.warning(f"[GeminiClient] Tool execution timed out: {name}")
        res = make_response("error", error="Execution timed out after 8.0s")

    elapsed_ms = (time.time() - t0) * 1000
    session.metrics["tool_duration_ms"] += elapsed_ms

    # Format return dictionary matching Gemini's ToolResponse frame expectancies
    return {
        "id": call_id,
        "name": name,
        "response": {
            "output": {
                "status": res["status"],
                "result": res["data"].get("result", "")
                if res["status"] == "success"
                else "",
                "error": res["error"],
            }
        },
    }
