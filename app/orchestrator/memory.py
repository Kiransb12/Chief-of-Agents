import os
import json
import time
import logging
import threading
from typing import List, Dict, Tuple
from groq import Groq
from app.config import GROQ_API_KEY, REASONING_MODEL

logger = logging.getLogger(__name__)

SEMANTIC_FILE = "./data/semantic_memory.json"
EPISODIC_FILE = "./data/episodic_memory.json"
PROCEDURAL_FILE = "./data/procedural_memory.json"

# Text sessions and voice sessions can both trigger consolidation and both
# write to these same JSON files. A single process-wide lock plus an atomic
# write (temp file + os.replace) prevents two concurrent consolidations from
# interleaving writes and corrupting the store. This is a single-process
# safeguard, not a multi-process one — at real scale this data belongs in
# SQLite/Postgres with row-level writes, not flat JSON files.
_MEMORY_LOCK = threading.Lock()


def _atomic_write_json(path: str, data) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp_path, path)  # atomic on POSIX and Windows


def _get_client() -> Groq:
    return Groq(api_key=GROQ_API_KEY)


# ----------------------------------------------------
# STORAGE LOADERS / WRITERS
# ----------------------------------------------------


def load_semantic_memory() -> List[str]:
    if not os.path.exists(SEMANTIC_FILE):
        os.makedirs(os.path.dirname(SEMANTIC_FILE), exist_ok=True)
        with open(SEMANTIC_FILE, "w", encoding="utf-8") as f:
            json.dump([], f)
        return []
    try:
        with open(SEMANTIC_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def save_semantic_memory(facts: List[str]) -> None:
    with _MEMORY_LOCK:
        _atomic_write_json(SEMANTIC_FILE, facts)


def load_episodic_memory() -> List[Dict]:
    if not os.path.exists(EPISODIC_FILE):
        os.makedirs(os.path.dirname(EPISODIC_FILE), exist_ok=True)
        with open(EPISODIC_FILE, "w", encoding="utf-8") as f:
            json.dump([], f)
        return []
    try:
        with open(EPISODIC_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def save_episodic_memory(episodes: List[Dict]) -> None:
    with _MEMORY_LOCK:
        _atomic_write_json(EPISODIC_FILE, episodes)


def load_procedural_memory() -> List[str]:
    if not os.path.exists(PROCEDURAL_FILE):
        os.makedirs(os.path.dirname(PROCEDURAL_FILE), exist_ok=True)
        with open(PROCEDURAL_FILE, "w", encoding="utf-8") as f:
            json.dump([], f)
        return []
    try:
        with open(PROCEDURAL_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


# ----------------------------------------------------
# CONSOLIDATION SYSTEM PROMPTS
# ----------------------------------------------------

SEMANTIC_CONSOLIDATE_PROMPT = """You are a memory consolidation engine for a personal AI assistant.
Your task is to merge new chat session transcripts with the user's existing Semantic Memory (durable facts) and output a single, updated list of all active facts.

Input:
1. Existing facts list (JSON).
2. The latest chat session transcript.

Instructions:
- Identify any new durable facts, long-term preferences, or rules the user shared in the transcript (e.g. travel habits, vegetarian diet, manager names, contact relationships).
- DO NOT extract transient one-off action requests or commands (e.g. do NOT extract "User wants to know weather", "User wants to open browser", "User wants to send message").
- Update any existing facts if the user has changed their mind or clarified details.
- Remove duplicates and keep facts concise (1-sentence per fact).
- Output the complete, updated list of ALL facts as a JSON array of strings.
- Example output: ["User prefers aisle seats on short flights", "User hates spicy food"].
- Do not output any conversational text, only the JSON list of facts.
"""


EPISODIC_SUMMARY_PROMPT = """You are a memory consolidation engine.
Given the chat session transcript below, write a concise 1-2 sentence summary of what occurred in this session.
Focus on topics discussed, actions taken, or items scheduled. Keep it brief.
Do not output any introductory or conversational text, only the 1-2 sentence summary.
"""


# Helper for Groq model call
def _call_llama(messages, response_format=None) -> str:
    client = _get_client()
    try:
        response = client.chat.completions.create(
            model=REASONING_MODEL,
            messages=messages,
            response_format=response_format,
            temperature=0.0,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Memory Llama call failed: {e}")
        return ""


# ----------------------------------------------------
# MAIN CONSOLIDATION RUNNER
# ----------------------------------------------------


def consolidate_session(session_id: str, turns: List[Dict[str, str]]) -> Dict:
    """Run background consolidation for a session transcript.

    Extracts new semantic facts, compiles episodic summary, and writes to store.
    """
    if not turns:
        logger.info(f"No turns to consolidate for session {session_id}")
        return {"summary": "Empty session.", "facts_updated": 0}

    # 1. Format transcript
    transcript_lines = []
    for turn in turns:
        role = "User" if turn["role"] == "user" else "Assistant"
        transcript_lines.append(f"{role}: {turn['content']}")
    transcript = "\n".join(transcript_lines)

    # 2. Compile Episodic summary
    logger.info(f"[Consolidation] Compiling episodic summary for {session_id}...")
    summary = _call_llama(
        [
            {"role": "system", "content": EPISODIC_SUMMARY_PROMPT},
            {"role": "user", "content": f"Transcript:\n{transcript}"},
        ]
    )
    if not summary:
        summary = "Session completed successfully."

    # Save to episodic history
    with _MEMORY_LOCK:
        episodes = load_episodic_memory()
        episodes.append(
            {
                "session_id": session_id,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "summary": summary,
            }
        )
        _atomic_write_json(EPISODIC_FILE, episodes)

    # 3. Consolidate Semantic Memory
    # The full load -> merge -> save span is held under the lock (not just
    # the final write) so that two sessions consolidating at nearly the same
    # time can't both read the same "existing_facts" snapshot and then have
    # the second save silently clobber the first session's updates. This
    # does mean concurrent consolidations queue up rather than run in
    # parallel, which is the right tradeoff for a background, low-frequency
    # operation writing to a single shared file.
    logger.info(
        f"[Consolidation] Updating semantic memory facts list for {session_id}..."
    )
    facts_updated = 0
    with _MEMORY_LOCK:
        existing_facts = load_semantic_memory()

        user_payload = {
            "existing_facts": existing_facts,
            "new_transcript": transcript,
        }

        raw_response = _call_llama(
            [
                {"role": "system", "content": SEMANTIC_CONSOLIDATE_PROMPT},
                {"role": "user", "content": json.dumps(user_payload)},
            ],
            response_format={"type": "json_object"},
        )

        if raw_response:
            try:
                # Parse Llama output (expects JSON array of strings inside a dict or raw array)
                parsed = json.loads(raw_response)
                if isinstance(parsed, dict) and "facts" in parsed:
                    updated_facts = parsed["facts"]
                elif isinstance(parsed, dict) and "existing_facts" in parsed:
                    updated_facts = parsed["existing_facts"]
                elif isinstance(parsed, list):
                    updated_facts = parsed
                else:
                    # Sometime model wraps inside a single top-level key
                    keys = list(parsed.keys())
                    if len(keys) == 1 and isinstance(parsed[keys[0]], list):
                        updated_facts = parsed[keys[0]]
                    else:
                        updated_facts = list(parsed.values())[0]

                if isinstance(updated_facts, list):
                    _atomic_write_json(SEMANTIC_FILE, updated_facts)
                    facts_updated = len(updated_facts)
                    logger.info(
                        f"[Consolidation] Saved {facts_updated} updated facts to semantic store."
                    )
                else:
                    logger.warning(
                        f"Consolidation facts response was not a list: {parsed}"
                    )
            except Exception as e:
                logger.error(f"Failed to parse consolidated facts JSON: {e}")

    return {"summary": summary, "facts_updated": facts_updated}
