"""
OpenAI GPT-4o Realtime Voice Client & WebSocket Bridge.

Manages bidirectional audio streaming, real-time Whisper transcriptions,
and tool calls using the OpenAI Realtime API (gpt-4o-realtime-preview-2024-12-17).
"""
import json
import base64
import logging
import asyncio
import time
from typing import Dict, Any, List
from fastapi import WebSocket, WebSocketDisconnect
import websockets

from app.config import (
    OPENAI_API_KEY,
    OPENAI_REALTIME_MODEL,
    OPENAI_REALTIME_VOICE,
    OPENAI_REALTIME_URI,
)
from app.orchestrator.tool_executor import tool_registry, execute_tool, ToolContext, make_response
from app.orchestrator.session_manager import session_manager, SessionState
from app.orchestrator.memory import load_semantic_memory

logger = logging.getLogger(__name__)


def _to_openai_json_schema(schema: Any) -> Any:
    """Recursively converts uppercase JSON schema types (Gemini style) to lowercase (OpenAI style)."""
    if isinstance(schema, dict):
        out = {}
        for k, v in schema.items():
            if k == "type" and isinstance(v, str):
                out[k] = v.lower()
            else:
                out[k] = _to_openai_json_schema(v)
        return out
    if isinstance(schema, list):
        return [_to_openai_json_schema(v) for v in schema]
    return schema


def get_openai_tools() -> List[dict]:
    """Formats local tool registry schemas into OpenAI Realtime function definitions."""
    tools = []
    for s in tool_registry.schemas:
        tools.append({
            "type": "function",
            "name": s["name"],
            "description": s.get("description", ""),
            "parameters": _to_openai_json_schema(s.get("parameters", {"type": "OBJECT", "properties": {}}))
        })
    return tools


async def async_run_openai_realtime_bridge(
    client_ws: WebSocket, session_id: str
) -> None:
    """Manages the lifecycle of an OpenAI GPT-4o Realtime session over WebSockets."""
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY environment variable is not configured.")

    session: SessionState = session_manager.get(session_id)
    openai_url = f"{OPENAI_REALTIME_URI}?model={OPENAI_REALTIME_MODEL}"

    # Load semantic memories to inject into system instruction
    facts = load_semantic_memory()
    facts_str = "\n".join(f"- {f}" for f in facts) if facts else "No semantic facts recorded yet."

    system_prompt = (
        "You are a helpful, professional personal AI assistant.\n"
        "Here are the active, summarized semantic memory facts about the user that you must know:\n"
        f"{facts_str}\n\n"
        "You have access to the user's personal documents, notes, and detailed info through the 'retrieve_rag_context' tool.\n"
        "CRITICAL INSTRUCTIONS: Speak immediately, concisely, and punchily. Avoid fillers or unnecessary preamble. "
        "If the user asks about detailed personal files, document details, or project tasks, call 'retrieve_rag_context' first before answering."
    )

    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
    }

    logger.info(
        f"[OpenAIClient] Connecting to OpenAI Realtime API: model={OPENAI_REALTIME_MODEL}, voice={OPENAI_REALTIME_VOICE}"
    )

    try:
        # Support both additional_headers and extra_headers for websockets library compatibility
        try:
            connect_ctx = websockets.connect(openai_url, additional_headers=headers)
        except TypeError:
            connect_ctx = websockets.connect(openai_url, extra_headers=headers)

        async with connect_ctx as openai_ws:
            # 1. Send session.update configuration frame
            session_update_frame = {
                "type": "session.update",
                "session": {
                    "modalities": ["audio", "text"],
                    "instructions": system_prompt,
                    "voice": OPENAI_REALTIME_VOICE,
                    "input_audio_format": "pcm16",
                    "output_audio_format": "pcm16",
                    "temperature": 0.6,
                    "input_audio_transcription": {
                        "model": "whisper-1"
                    },
                    "turn_detection": {
                        "type": "server_vad",
                        "threshold": 0.5,
                        "prefix_padding_ms": 200,
                        "silence_duration_ms": 250
                    },
                    "tools": get_openai_tools()
                }
            }
            await openai_ws.send(json.dumps(session_update_frame))
            logger.info("[OpenAIClient] Session update frame sent successfully.")

            # Define sub-tasks
            async def forward_client_to_openai():
                """Forwards client microphone PCM input and control messages to OpenAI Realtime."""
                try:
                    while True:
                        data = await client_ws.receive()
                        if "bytes" in data:
                            raw_pcm = data["bytes"]
                            b64_chunk = base64.b64encode(raw_pcm).decode("utf-8")
                            audio_frame = {
                                "type": "input_audio_buffer.append",
                                "audio": b64_chunk
                            }
                            await openai_ws.send(json.dumps(audio_frame))

                        elif "text" in data:
                            text_payload = json.loads(data["text"])
                            msg_type = text_payload.get("type")
                            
                            if msg_type == "interrupt":
                                logger.info("[OpenAIClient] Interruption signal received from client.")
                                session.is_interrupted = True
                                await openai_ws.send(json.dumps({"type": "input_audio_buffer.clear"}))
                                await openai_ws.send(json.dumps({"type": "response.cancel"}))
                                await client_ws.send_json({"type": "interrupted"})

                            elif msg_type == "text":
                                user_msg = text_payload.get("message", "")
                                logger.info(f"[OpenAIClient] User sent text message: {user_msg}")
                                session.is_interrupted = True
                                await openai_ws.send(json.dumps({"type": "response.cancel"}))
                                await client_ws.send_json({"type": "interrupted"})

                                # Create conversation item & trigger response
                                item_frame = {
                                    "type": "conversation.item.create",
                                    "item": {
                                        "type": "message",
                                        "role": "user",
                                        "content": [{"type": "input_text", "text": user_msg}]
                                    }
                                }
                                await openai_ws.send(json.dumps(item_frame))
                                await openai_ws.send(json.dumps({"type": "response.create"}))

                except WebSocketDisconnect:
                    logger.info("[OpenAIClient] Client WebSocket disconnected.")
                except Exception as e:
                    logger.error(f"[OpenAIClient] Error in client-to-openai loop: {e}")
                    session.metrics["errors"].append(f"ClientLoop: {e}")

            async def forward_openai_to_client():
                """Processes OpenAI response events (audio, transcriptions, tool calls)."""
                user_transcript_buffer = []

                try:
                    async for raw_message in openai_ws:
                        event = json.loads(raw_message)
                        event_type = event.get("type")

                        # 1. Incoming Audio Output Delta (PCM 16-bit 24kHz)
                        if event_type == "response.audio.delta":
                            b64_audio = event.get("delta")
                            if b64_audio and not session.is_interrupted:
                                await client_ws.send_json({
                                    "type": "audio",
                                    "data": b64_audio
                                })

                        # 2. User Input Audio Transcription (Whisper)
                        elif event_type == "conversation.item.input_audio_transcription.completed":
                            transcript = event.get("transcript", "").strip()
                            if transcript:
                                user_transcript_buffer.append(transcript)
                                await client_ws.send_json({
                                    "type": "caption",
                                    "role": "user",
                                    "text": transcript
                                })

                        # 3. Model Output Spoken Text Transcript Delta
                        elif event_type == "response.audio_transcript.delta":
                            txt_delta = event.get("delta", "")
                            if txt_delta:
                                await client_ws.send_json({
                                    "type": "caption",
                                    "role": "assistant",
                                    "text": txt_delta
                                })

                        # 4. Turn complete / response done
                        elif event_type == "response.done":
                            session.is_interrupted = False
                            if user_transcript_buffer:
                                full_utterance = " ".join(user_transcript_buffer).strip()
                                if full_utterance:
                                    logger.info(f"[OpenAIClient] Flushing user transcript: {full_utterance}")
                                    session_manager.add_turn(session_id, "user", full_utterance)
                                user_transcript_buffer.clear()

                        # 5. Tool / Function Call Execution
                        elif event_type == "response.function_call_arguments.done":
                            call_id = event.get("call_id") or event.get("item_id", "call_default")
                            name = event.get("name")
                            arg_str = event.get("arguments", "{}")

                            try:
                                args = json.loads(arg_str) if arg_str else {}
                            except Exception:
                                args = {}

                            logger.info(f"[OpenAIClient] Tool call requested: {name}({args}) [call_id={call_id}]")
                            await client_ws.send_json({"type": "thinking"})
                            await client_ws.send_json({
                                "type": "tool_start",
                                "name": name,
                                "args": args,
                                "call_id": call_id
                            })

                            context = ToolContext(session_id=session_id)
                            res = await _run_and_format_tool(name, args, call_id, context, session)

                            await client_ws.send_json({
                                "type": "tool_complete",
                                "name": name,
                                "args": args,
                                "call_id": call_id,
                                "output": res["response"]["output"]
                            })

                            # Submit tool result back to OpenAI
                            tool_output_frame = {
                                "type": "conversation.item.create",
                                "item": {
                                    "type": "function_call_output",
                                    "call_id": call_id,
                                    "output": json.dumps(res["response"]["output"])
                                }
                            }
                            await openai_ws.send(json.dumps(tool_output_frame))
                            # Trigger response generation with tool output
                            await openai_ws.send(json.dumps({"type": "response.create"}))

                except websockets.exceptions.ConnectionClosed:
                    logger.info("[OpenAIClient] OpenAI Realtime connection closed.")
                except Exception as e:
                    logger.error(f"[OpenAIClient] Error in openai-to-client loop: {e}")
                    session.metrics["errors"].append(f"OpenAILoop: {e}")

            # Run bridging loops concurrently
            await asyncio.gather(
                forward_client_to_openai(), forward_openai_to_client()
            )

    except Exception as e:
        logger.error(f"[OpenAIClient] Failed to execute Realtime session bridge: {e}")
        session.metrics["errors"].append(f"OpenAIBridgeSetup: {e}")
        raise e


async def _run_and_format_tool(
    name: str,
    args: dict,
    call_id: str,
    context: ToolContext,
    session: SessionState,
) -> dict:
    """Executes a tool with timeout tracking and formats response matching standard schema."""
    session.metrics["tool_call_count"] += 1
    t0 = time.time()

    try:
        res = await asyncio.wait_for(
            execute_tool(name, args, context), timeout=8.0
        )
    except asyncio.TimeoutError:
        logger.warning(f"[OpenAIClient] Tool execution timed out: {name}")
        res = make_response("error", error="Execution timed out after 8.0s")

    elapsed_ms = (time.time() - t0) * 1000
    session.metrics["tool_duration_ms"] += elapsed_ms

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
