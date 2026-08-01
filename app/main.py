"""
Phase 4: text-only orchestrator + Gemini Live voice integration.

Run:
    uvicorn app.main:app --reload
"""
import os
import logging

# Configure console logging across all modules in the project
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    force=True,
)

logger = logging.getLogger(__name__)

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.config import (
    DEEPGRAM_API_KEY,
    CARTESIA_API_KEY,
    GEMINI_API_KEY,
    OPENAI_API_KEY,
    GROQ_API_KEY,
    CHROMA_PERSIST_DIR,
)
from app.orchestrator.state import get_or_create_session
from app.orchestrator.workflow import run_workflow
from app.orchestrator.memory import consolidate_session
from app.orchestrator.session_manager import session_manager
from app.orchestrator.voice_bridge import async_run_voice_bridge

app = FastAPI(title="Chief of Staff — Voice Agent (Deepgram STT & Cartesia TTS)")


# ----------------------------------------------------
# CONFIGURATION VALIDATION ON STARTUP
# ----------------------------------------------------
import asyncio


def silence_aioice_exceptions(loop, context):
    """Silences aioice background timer retries on closed WebRTC sockets."""
    exception = context.get("exception")
    msg = str(context.get("message", ""))
    err_str = str(exception) if exception else ""
    if (
        "sendto" in err_str
        or "call_exception_handler" in err_str
        or "sendto" in msg
        or "InvalidStateError" in err_str
        or "TransactionTimeout" in err_str
        or isinstance(exception, (asyncio.InvalidStateError, AttributeError))
    ):
        return
    loop.default_exception_handler(context)


@app.on_event("startup")
def setup_webrtc_exception_handler():
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.get_event_loop()
    loop.set_exception_handler(silence_aioice_exceptions)


@app.on_event("startup")
def validate_startup_configuration():
    logger.info("[Startup] Validating system configurations...")
    if DEEPGRAM_API_KEY and CARTESIA_API_KEY:
        logger.info("[Startup] Primary voice provider: Deepgram STT + Cartesia TTS configured.")
    elif OPENAI_API_KEY:
        logger.info("[Startup] Fallback voice provider: OpenAI GPT-4o Realtime API configured.")
    elif GEMINI_API_KEY:
        logger.info("[Startup] Fallback voice provider: Gemini Live API configured.")
    else:
        logger.warning("[Startup] No live voice providers configured in .env.")
    if not GROQ_API_KEY:
        logger.error("CRITICAL CONFIG ERROR: 'GROQ_API_KEY' is not set.")
        raise RuntimeError("Missing required environment variable: GROQ_API_KEY")


    # Validate Chroma directory
    if not os.path.exists(CHROMA_PERSIST_DIR):
        logger.warning(
            f"Chroma persist directory '{CHROMA_PERSIST_DIR}' does not exist. "
            "It will be created dynamically on first write."
        )
    logger.info("[Startup] All required configuration checks passed successfully.")


@app.on_event("startup")
async def prewarm_chroma_embedding_model():
    """Eagerly load Chroma's embedding model so the first RAG query doesn't
    pay the cold-start cost (model download + load) inside the voice bridge's
    8-second tool timeout window."""
    def _do_prewarm():
        import chromadb
        from app.config import COLLECTION_NAME
        client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
        collection = client.get_or_create_collection(COLLECTION_NAME)
        collection.query(query_texts=["warmup"], n_results=1)

    try:
        logger.info("[Startup] Pre-warming Chroma embedding model...")
        await asyncio.to_thread(_do_prewarm)
        logger.info("[Startup] Chroma embedding model loaded successfully.")
    except Exception as e:
        logger.warning(f"[Startup] Chroma pre-warm failed (non-fatal): {e}")


async def _async_safe_consolidate(session_id: str) -> None:
    """Safely runs session memory consolidation on a worker thread using asyncio.to_thread."""
    try:
        await asyncio.to_thread(session_manager.consolidate, session_id)
    except Exception as ex:
        logger.error(f"[SessionCleanup] Failed to consolidate session {session_id}: {ex}")
        session_manager.remove(session_id)


# ----------------------------------------------------
# REST API ENDPOINTS
# ----------------------------------------------------
class ChatRequest(BaseModel):
    session_id: str
    message: str


class ChatResponse(BaseModel):
    reply: str
    used_rag: bool
    route: str
    sources: list
    latency_seconds: float
    total_cost_usd: float
    router_input_tokens: int
    router_output_tokens: int
    reasoning_input_tokens: int
    reasoning_output_tokens: int


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    state = get_or_create_session(req.session_id)

    # Execute workflow nodes
    res = await run_workflow(req.message, state.recent_turns(), session_id=req.session_id)

    # Record history
    state.add_turn("user", req.message)
    state.add_turn("assistant", res["reply"])

    return ChatResponse(
        reply=res["reply"],
        used_rag=res["used_rag"],
        route=res["route"],
        sources=res["sources"],
        latency_seconds=res["latency_seconds"],
        total_cost_usd=res["total_cost_usd"],
        router_input_tokens=res["router_input_tokens"],
        router_output_tokens=res["router_output_tokens"],
        reasoning_input_tokens=res["reasoning_input_tokens"],
        reasoning_output_tokens=res["reasoning_output_tokens"],
    )


class ConsolidateRequest(BaseModel):
    session_id: str


class ConsolidateResponse(BaseModel):
    summary: str
    facts_updated: int


@app.post("/session/consolidate", response_model=ConsolidateResponse)
def consolidate(req: ConsolidateRequest) -> ConsolidateResponse:
    logger.info(f"REST Consolidating session {req.session_id}")
    state = get_or_create_session(req.session_id)
    turns = state.recent_turns(n=100)
    res = consolidate_session(req.session_id, turns)
    state.clear()
    return ConsolidateResponse(
        summary=res["summary"],
        facts_updated=res["facts_updated"]
    )


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


from aiortc import RTCPeerConnection, RTCSessionDescription, RTCConfiguration
from app.orchestrator.webrtc_bridge import async_run_webrtc_bridge


# ----------------------------------------------------
# WEBRTC SIGNALING & DUAL DATACHANNEL ENDPOINTS
# ----------------------------------------------------
class WebRTCOfferRequest(BaseModel):
    sdp: str
    type: str
    session_id: str = "default-live-session"


class WebRTCAnswerResponse(BaseModel):
    sdp: str
    type: str
    session_id: str


pcs = set()


@app.post("/webrtc/offer", response_model=WebRTCAnswerResponse)
async def webrtc_offer(req: WebRTCOfferRequest) -> WebRTCAnswerResponse:
    logger.info(f"[WebRTC] Offer received for session ID: {req.session_id}")
    offer = RTCSessionDescription(sdp=req.sdp, type=req.type)
    config = RTCConfiguration(iceServers=[])
    pc = RTCPeerConnection(configuration=config)
    pcs.add(pc)

    channels = {}

    @pc.on("datachannel")
    def on_datachannel(channel):
        logger.info(f"[WebRTC] Data channel created: {channel.label}")
        channels[channel.label] = channel
        _check_and_start_bridge()

    def _check_and_start_bridge():
        if "media_channel" in channels and "live_updates" in channels:
            media_ch = channels["media_channel"]
            updates_ch = channels["live_updates"]
            logger.info(f"[WebRTC] Dual data channels ready for session {req.session_id}. Starting WebRTC bridge...")
            asyncio.create_task(async_run_webrtc_bridge(media_ch, updates_ch, req.session_id))

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        logger.info(f"[WebRTC] Connection state changed: {pc.connectionState}")
        if pc.connectionState in ["failed", "closed"]:
            await pc.close()
            pcs.discard(pc)
            asyncio.create_task(_async_safe_consolidate(req.session_id))

    await pc.setRemoteDescription(offer)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return WebRTCAnswerResponse(
        sdp=pc.localDescription.sdp,
        type=pc.localDescription.type,
        session_id=req.session_id,
    )


# ----------------------------------------------------
# LIVE AUDIO WEBSOCKET & VIEW ROUTING
# ----------------------------------------------------
@app.websocket("/ws/live")
async def websocket_live_endpoint(websocket: WebSocket):
    await websocket.accept()
    session_id = websocket.query_params.get("session_id", "default-live-session")
    logger.info(f"[WebSocket] Connected. Assigned session ID: {session_id}")

    try:
        await async_run_voice_bridge(websocket, session_id)
    except Exception as e:
        logger.error(f"[WebSocket] Bridge error on session {session_id}: {e}")
    finally:
        logger.info(f"[WebSocket] Session {session_id} closed. Triggering consolidation...")
        asyncio.create_task(_async_safe_consolidate(session_id))


app.mount("/visualizer", StaticFiles(directory="visualizer"), name="visualizer")

@app.get("/", response_class=HTMLResponse)
def get_voice_portal() -> HTMLResponse:
    import os
    tpl_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "visualizer/index.html")
    if os.path.exists(tpl_path):
        with open(tpl_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(
        content="<h1>Chief of Staff Portal</h1><p>Visualizer index.html not found.</p>"
    )
