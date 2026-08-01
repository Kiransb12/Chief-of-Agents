# Chief of Staff Agent — Phase 3: Multi-Layer Memory

Phase 3 introduces a multi-layer memory architecture to ensure the agent retains context across conversations. It implements a session-based working memory that compiles into semantic preferences and episodic summaries during session teardowns.

---

## 1. Technical Stack
- **Durable Layers**: Local JSON database files (`semantic_memory.json`, `episodic_memory.json`)
- **Semantic Filters**: Vector space overlap filters (cosine similarity checks)
- **Framework**: Python 3.12, FastAPI websocket handlers

---

## 2. Memory Architecture & Hierarchy

The agent maintains three distinct memory pools:
1. **Working Memory**: In-process runtime turn dictionary (reset on session restart).
2. **Episodic Memory**: Long-term summaries of past conversation sessions, logged chronologically.
3. **Semantic Memory**: Durable database of parsed facts, travel options, names, and preferences.

```
+--------------------------------------------------------+
|                   Working Memory                       |
|           (Live turn context in-session)               |
+---------------------------+----------------------------+
                            |
                            v [Post-Session Consolidation]
+---------------------------+----------------------------+
|            Episodic Summary Generation                 |
|    (Claude summarizes the session's overall themes)    |
+---------------------------+----------------------------+
                            |
                            +-------------------+
                            |                   |
                            v                   v
                +-------------------+   +-------------------+
                |  Episodic Memory  |   |  Semantic Memory  |
                |   (Chronological  |   |   (Facts list,    |
                |     summaries)    |   |   deduplicated)   |
                +-------------------+   +-------------------+
```

---

## 3. Deduplication and Context Assembly
To prevent context clutter and cost scaling, the memory manager filters duplicates:
- **Cosine Overlap Threshold**: If a newly extracted fact has `> 70%` similarity with an existing semantic fact or RAG document chunk, the duplicate is ignored or updated in place.
- **Priority Stack**: When prompting the reasoning model, memory facts are ordered as follows:
  1. *Semantic memory user preferences* (highest priority context).
  2. *Document RAG context segments*.
  3. *Episodic history summaries*.

---

## 4. Post-Session Consolidation Worker
When a session WebSocket connection closes or terminates:
1. The **`SessionManager`** catches the shutdown signal.
2. Triggers a background thread calling `consolidate_session`.
3. Claude evaluates the session's turn array:
   - Compiles a short episodic summary.
   - Extracts new facts (e.g. *"User is interested in news about Dell"*).
4. Commits updates to `semantic_memory.json` and `episodic_memory.json`.
