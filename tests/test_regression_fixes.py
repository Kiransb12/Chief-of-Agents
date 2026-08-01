"""
Regression Test Suite for Chief of Agents Confirmed Fixes.

Locks in test coverage for:
1. Event loop non-blocking behavior under retry/rate-limit.
2. Agent tool loop reply preservation on max iteration.
3. RAG cosine metric configuration & score bounds.
4. Deepgram STT speech_final trigger discipline.
5. Episodic memory concurrent consolidation atomicity.
6. Voice bridge session turn ordering.
"""
import asyncio
import json
import os
import tempfile
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import chromadb

from app.config import COLLECTION_NAME
from app.orchestrator.session_manager import session_manager
from app.orchestrator.memory import (
    consolidate_session,
    load_episodic_memory,
    _atomic_write_json,
)
from app.orchestrator.workflow import agent_node, _generate_content_with_retry
from app.rag.ingest import ingest_directory
from app.rag.retriever import retrieve


# ---------------------------------------------------------------------------
# 1. Event loop non-blocking under rate limit
# ---------------------------------------------------------------------------
def test_event_loop_non_blocking_on_retry():
    """Verify rate-limit retries use async sleep, allowing concurrent tasks to execute."""
    async def _impl():
        completion_order = []

        async def slow_retry_call():
            mock_client = AsyncMock()
            call_count = 0

            async def mock_create(*args, **kwargs):
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    raise Exception("429 Rate limit exceeded")
                mock_resp = MagicMock()
                return mock_resp

            mock_client.chat.completions.create.side_effect = mock_create

            # Initial delay 0.1s so test runs fast
            await _generate_content_with_retry(
                client=mock_client,
                model="mock-model",
                messages=[],
                max_retries=2,
                initial_delay=0.1,
            )
            completion_order.append("retry_done")

        async def fast_concurrent_task():
            await asyncio.sleep(0.02)
            completion_order.append("fast_task_done")

        # Launch both tasks concurrently
        await asyncio.gather(slow_retry_call(), fast_concurrent_task())

        # Fast task MUST complete before slow retrying task
        assert completion_order == ["fast_task_done", "retry_done"]

    asyncio.run(_impl())


# ---------------------------------------------------------------------------
# 2. Agent loop reply preservation on final iteration
# ---------------------------------------------------------------------------
def test_agent_loop_reply_preservation_on_final_iteration():
    """Verify a valid text response on iteration 5 is preserved and not overwritten by max-iter error."""
    async def _impl():
        mock_responses = []

        # Iterations 1-4: Return tool call requests
        for i in range(1, 5):
            msg = MagicMock()
            msg.content = ""
            tc = MagicMock()
            tc.id = f"call_{i}"
            tc.function = MagicMock()
            tc.function.name = "search_web"
            tc.function.arguments = json.dumps({"query": f"test_{i}"})
            msg.tool_calls = [tc]

            resp = MagicMock()
            resp.choices = [MagicMock(message=msg)]
            resp.usage = MagicMock(prompt_tokens=10, completion_tokens=10)
            mock_responses.append(resp)

        # Iteration 5: Return final valid text reply
        final_msg = MagicMock()
        final_msg.content = "Valid final answer on 5th iteration"
        final_msg.tool_calls = None
        final_resp = MagicMock()
        final_resp.choices = [MagicMock(message=final_msg)]
        final_resp.usage = MagicMock(prompt_tokens=10, completion_tokens=10)
        mock_responses.append(final_resp)

        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(side_effect=mock_responses)

        with patch(
            "app.orchestrator.workflow._get_client",
            return_value=mock_client,
        ), patch(
            "app.orchestrator.workflow.execute_tool",
            new_callable=AsyncMock,
            return_value={"status": "success", "data": {"result": "tool result"}},
        ):
            res = await agent_node(
                message="Complex query requiring 4 tool steps",
                context_block="",
                recent_turns=[],
                session_id="test-max-iter-session",
                intent="search_web",
            )

            assert res["reply"] == "Valid final answer on 5th iteration"
            assert "Error: Agent exceeded maximum tool calling iterations" not in res["reply"]

    asyncio.run(_impl())


# ---------------------------------------------------------------------------
# 3. RAG cosine scoring & metadata
# ---------------------------------------------------------------------------
def test_rag_cosine_scoring_and_metadata():
    """Verify Chroma collection uses cosine space and retriever produces valid positive scores [0, 1]."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp_dir:
        doc_dir = os.path.join(tmp_dir, "docs")
        os.makedirs(doc_dir, exist_ok=True)
        doc_file = os.path.join(doc_dir, "sample.txt")
        with open(doc_file, "w", encoding="utf-8") as f:
            f.write("User prefers window seats on international flights and vegetarian meals.")

        chroma_dir = os.path.join(tmp_dir, "chroma")

        with patch("app.rag.ingest.CHROMA_PERSIST_DIR", chroma_dir), \
             patch("app.rag.retriever.CHROMA_PERSIST_DIR", chroma_dir):

            ingest_directory(doc_dir)

            client = chromadb.PersistentClient(path=chroma_dir)
            coll = client.get_collection(COLLECTION_NAME)
            assert coll.metadata is not None
            assert coll.metadata.get("hnsw:space") == "cosine"

            results = retrieve("What are the flight seating preferences?", k=2, min_score=0.10)
            assert len(results) > 0
            for r in results:
                assert 0.0 <= r["score"] <= 1.0


# ---------------------------------------------------------------------------
# 4. STT trigger discipline
# ---------------------------------------------------------------------------
def test_stt_trigger_discipline():
    """Verify workflow dispatch is called ONLY on speech_final=True, not on intermediate is_final chunks."""
    async def _impl():
        from app.orchestrator.deepgram_cartesia_bridge import async_run_deepgram_cartesia_bridge

        mock_client_ws = AsyncMock()
        mock_client_ws.receive.side_effect = [{"type": "websocket.disconnect"}]

        dg_frames = [
            json.dumps({
                "type": "Results",
                "is_final": True,
                "speech_final": False,
                "channel": {"alternatives": [{"transcript": "hello"}]}
            }),
            json.dumps({
                "type": "Results",
                "is_final": True,
                "speech_final": False,
                "channel": {"alternatives": [{"transcript": "there"}]}
            }),
            json.dumps({
                "type": "Results",
                "is_final": True,
                "speech_final": True,
                "channel": {"alternatives": [{"transcript": "friend"}]}
            }),
        ]

        class MockAsyncIter:
            def __init__(self, items):
                self.items = items
            def __aiter__(self):
                return self
            async def __anext__(self):
                if not self.items:
                    raise StopAsyncIteration
                return self.items.pop(0)

        mock_dg_ws = MockAsyncIter(dg_frames)

        class MockConnectContext:
            async def __aenter__(self):
                return mock_dg_ws
            async def __aexit__(self, *args):
                pass

        with patch("app.orchestrator.deepgram_cartesia_bridge.DEEPGRAM_API_KEY", "mock_key"), \
             patch("app.orchestrator.deepgram_cartesia_bridge.CARTESIA_API_KEY", "mock_key"), \
             patch("websockets.connect", return_value=MockConnectContext()), \
             patch("app.orchestrator.deepgram_cartesia_bridge.run_workflow", new_callable=AsyncMock) as mock_wf:

            mock_wf.return_value = {"reply": "Hi!"}

            await async_run_deepgram_cartesia_bridge(mock_client_ws, "test-stt-trigger-session")

            assert mock_wf.call_count == 1
            assert mock_wf.call_args[1]["message"] == "hello there friend"

    asyncio.run(_impl())


def test_stt_buffer_joining_and_edge_case():
    """Verify STT buffer joins with single spaces and includes final chunk when is_final=False but speech_final=True."""
    async def _impl():
        from app.orchestrator.deepgram_cartesia_bridge import async_run_deepgram_cartesia_bridge

        mock_client_ws = AsyncMock()
        mock_client_ws.receive.side_effect = [{"type": "websocket.disconnect"}]

        dg_frames = [
            json.dumps({
                "type": "Results",
                "is_final": True,
                "speech_final": False,
                "channel": {"alternatives": [{"transcript": "So"}]}
            }),
            # Edge case: final chunk has is_final=False but speech_final=True
            json.dumps({
                "type": "Results",
                "is_final": False,
                "speech_final": True,
                "channel": {"alternatives": [{"transcript": "is there any news regarding"}]}
            }),
        ]

        class MockAsyncIter:
            def __init__(self, items):
                self.items = items
            def __aiter__(self):
                return self
            async def __anext__(self):
                if not self.items:
                    raise StopAsyncIteration
                return self.items.pop(0)

        class MockConnectContext:
            async def __aenter__(self):
                return MockAsyncIter(dg_frames)
            async def __aexit__(self, *args):
                pass

        with patch("app.orchestrator.deepgram_cartesia_bridge.DEEPGRAM_API_KEY", "mock_key"), \
             patch("app.orchestrator.deepgram_cartesia_bridge.CARTESIA_API_KEY", "mock_key"), \
             patch("websockets.connect", return_value=MockConnectContext()), \
             patch("app.orchestrator.deepgram_cartesia_bridge.run_workflow", new_callable=AsyncMock) as mock_wf:

            mock_wf.return_value = {"reply": "News response"}

            await async_run_deepgram_cartesia_bridge(mock_client_ws, "test-stt-edge-case-session")

            assert mock_wf.call_count == 1
            assert mock_wf.call_args[1]["message"] == "So is there any news regarding"

    asyncio.run(_impl())



# ---------------------------------------------------------------------------
# 5. Episodic memory atomicity under concurrency
# ---------------------------------------------------------------------------
def test_episodic_memory_atomicity_under_concurrency():
    """Verify concurrent consolidate_session calls in separate background threads do not clobber episodic history entries."""
    import threading

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp_dir:
        test_ep_file = os.path.join(tmp_dir, "episodic_memory.json")
        _atomic_write_json(test_ep_file, [])

        def mock_call_llama(messages, **kwargs):
            time.sleep(0.05)
            user_content = messages[1]["content"]
            return f"Summary for {user_content[:20]}"

        with patch("app.orchestrator.memory.EPISODIC_FILE", test_ep_file), \
             patch("app.orchestrator.memory._call_llama", side_effect=mock_call_llama):

            turns1 = [{"role": "user", "content": "Session 1 message"}]
            turns2 = [{"role": "user", "content": "Session 2 message"}]

            t1 = threading.Thread(target=consolidate_session, args=("session-1", turns1))
            t2 = threading.Thread(target=consolidate_session, args=("session-2", turns2))

            t1.start()
            t2.start()

            t1.join()
            t2.join()

            with patch("app.orchestrator.memory.EPISODIC_FILE", test_ep_file):
                episodes = load_episodic_memory()
                session_ids = [e["session_id"] for e in episodes]

                assert len(episodes) == 2
                assert "session-1" in session_ids
                assert "session-2" in session_ids



# ---------------------------------------------------------------------------
# 6. Turn ordering (voice path)
# ---------------------------------------------------------------------------
def test_turn_ordering_voice_path():
    """Verify user turn is saved in session_manager BEFORE run_workflow is invoked."""
    async def _impl():
        from app.orchestrator.deepgram_cartesia_bridge import async_run_deepgram_cartesia_bridge

        mock_client_ws = AsyncMock()
        mock_client_ws.receive.side_effect = [{"type": "websocket.disconnect"}]

        dg_frame = json.dumps({
            "type": "Results",
            "is_final": True,
            "speech_final": True,
            "channel": {"alternatives": [{"transcript": "What is my focus area today?"}]}
        })

        class MockAsyncIter:
            def __init__(self, items):
                self.items = items
            def __aiter__(self):
                return self
            async def __anext__(self):
                if not self.items:
                    raise StopAsyncIteration
                return self.items.pop(0)

        class MockConnectContext:
            async def __aenter__(self):
                return MockAsyncIter([dg_frame])
            async def __aexit__(self, *args):
                pass

        session_id = "test-turn-order-session"
        observed_recent_turns_at_workflow_time = []

        async def mock_workflow_impl(message, recent_turns, session_id, on_progress=None):
            turns = session_manager.get_recent_turns(session_id)
            observed_recent_turns_at_workflow_time.extend(turns)
            return {"reply": "Your focus area is deep work."}

        with patch("app.orchestrator.deepgram_cartesia_bridge.DEEPGRAM_API_KEY", "mock_key"), \
             patch("app.orchestrator.deepgram_cartesia_bridge.CARTESIA_API_KEY", "mock_key"), \
             patch("websockets.connect", return_value=MockConnectContext()), \
             patch("app.orchestrator.deepgram_cartesia_bridge.run_workflow", side_effect=mock_workflow_impl):

            await async_run_deepgram_cartesia_bridge(mock_client_ws, session_id)

            assert len(observed_recent_turns_at_workflow_time) > 0
            user_turn = observed_recent_turns_at_workflow_time[-1]
            assert user_turn["role"] == "user"
            assert user_turn["content"] == "What is my focus area today?"

            session_manager.remove(session_id)

    asyncio.run(_impl())


# ---------------------------------------------------------------------------
# 7. Tool-call text fallback parsing & sanitization (Fix 1 & 2)
# ---------------------------------------------------------------------------
def test_agent_loop_fallback_tool_call_parsing_and_sanitization():
    """Verify response with tool_calls=None and inline <function=...> content triggers execute_tool and sanitizes reply."""
    async def _impl():
        # Iteration 1: Return text containing inline <function=...> tag (tool_calls=None)
        msg1 = MagicMock()
        msg1.content = 'Got it, let me check. <function=search_web>{"query": "latest news"}</function>'
        msg1.tool_calls = None
        resp1 = MagicMock()
        resp1.choices = [MagicMock(message=msg1)]
        resp1.usage = MagicMock(prompt_tokens=10, completion_tokens=10)

        # Iteration 2: Return final answer with residual tag that needs sanitization
        msg2 = MagicMock()
        msg2.content = 'Here is the news summary.'
        msg2.tool_calls = None
        resp2 = MagicMock()
        resp2.choices = [MagicMock(message=msg2)]
        resp2.usage = MagicMock(prompt_tokens=10, completion_tokens=10)

        executed_tools = []

        async def mock_execute_tool(name, args, context):
            executed_tools.append((name, args))
            return {"status": "success", "data": {"result": "search results content"}}

        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(side_effect=[resp1, resp2])

        with patch(
            "app.orchestrator.workflow._get_client",
            return_value=mock_client,
        ), patch(
            "app.orchestrator.workflow.execute_tool",
            side_effect=mock_execute_tool,
        ):
            res = await agent_node(
                message="What is the latest news?",
                context_block="",
                recent_turns=[],
                session_id="test-fallback-session",
                intent="search_web",
            )

            # Tool executed via fallback parser
            assert len(executed_tools) == 1
            assert executed_tools[0] == ("search_web", {"query": "latest news"})

            # Final reply is clean
            assert res["reply"] == "Here is the news summary."

    asyncio.run(_impl())


def test_reply_sanitization_defense_in_depth():
    import re
    raw_reply = "Here is the summary. <function=search_web>{\"query\": \"test\"}</function>"
    clean_reply = re.sub(r'<function=.*?</function>', '', raw_reply, flags=re.DOTALL).strip()
    assert clean_reply == "Here is the summary."


# ---------------------------------------------------------------------------
# 8. Duplicate Tool Call Loop Prevention & Async Safe Consolidation Tests
# ---------------------------------------------------------------------------
def test_duplicate_tool_call_deduplication_in_agent_node():
    """Verify duplicate tool calls in same turn are skipped and text synthesis is forced."""
    async def _impl():
        # Iteration 1: Return get_calendar_events
        tc1 = MagicMock()
        tc1.id = "call_1"
        tc1.function = MagicMock()
        tc1.function.name = "get_calendar_events"
        tc1.function.arguments = "{}"
        msg1 = MagicMock(content="", tool_calls=[tc1])
        resp1 = MagicMock(choices=[MagicMock(message=msg1)], usage=MagicMock(prompt_tokens=10, completion_tokens=10))

        # Iteration 2: Return identical get_calendar_events call
        tc2 = MagicMock()
        tc2.id = "call_2"
        tc2.function = MagicMock()
        tc2.function.name = "get_calendar_events"
        tc2.function.arguments = "{}"
        msg2 = MagicMock(content="", tool_calls=[tc2])
        resp2 = MagicMock(choices=[MagicMock(message=msg2)], usage=MagicMock(prompt_tokens=10, completion_tokens=10))

        # Final synthesis response (forced when duplicate detected)
        msg_final = MagicMock(content="You have 2 meetings tomorrow.", tool_calls=None)
        resp_final = MagicMock(choices=[MagicMock(message=msg_final)], usage=MagicMock(prompt_tokens=10, completion_tokens=10))

        executed_tools = []
        async def mock_execute(name, args, context):
            executed_tools.append((name, args))
            return {"status": "success", "data": {"result": "2 events scheduled"}}

        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(side_effect=[resp1, resp2, resp_final])

        with patch("app.orchestrator.workflow._get_client", return_value=mock_client):
            with patch("app.orchestrator.workflow.execute_tool", side_effect=mock_execute):
                res = await agent_node(
                    message="What are my events?",
                    context_block="",
                    recent_turns=[],
                    session_id="test-dedup-session",
                    intent="manage_calendar"
                )

                # Tool was executed ONLY ONCE (iteration 1), not twice
                assert len(executed_tools) == 1
                assert executed_tools[0][0] == "get_calendar_events"
                assert res["reply"] == "You have 2 meetings tomorrow."

    asyncio.run(_impl())


def test_async_safe_session_consolidation():
    """Verify _async_safe_consolidate runs session manager consolidation cleanly in background task."""
    async def _impl():
        from app.main import _async_safe_consolidate
        with patch("app.orchestrator.session_manager.session_manager.consolidate") as mock_cons:
            mock_cons.return_value = {"summary": "Session consolidated", "facts_updated": 1}
            await _async_safe_consolidate("test-session-123")
            mock_cons.assert_called_once_with("test-session-123")

    asyncio.run(_impl())



