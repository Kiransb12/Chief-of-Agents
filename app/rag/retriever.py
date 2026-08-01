import chromadb

from app.config import CHROMA_PERSIST_DIR, COLLECTION_NAME


MIN_RELEVANCE_SCORE = 0.20  # below this, a chunk is noise, not signal


def retrieve(query: str, k: int = 5, min_score: float = MIN_RELEVANCE_SCORE) -> list:
    """Semantic retrieval over the personal knowledge base.

    Phase 1: pure vector search only. Next step (see README): add BM25
    keyword search alongside this, merge the two result sets (hybrid
    search), then rerank before returning the top-k. Pure vector search
    alone tends to miss exact-match queries (names, IDs, dates).

    A relevance floor (min_score) is applied so that when the corpus has
    no good match for the query, Chroma's "closest available" results
    aren't silently injected into the prompt as if they were relevant.
    """
    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    collection = client.get_or_create_collection(COLLECTION_NAME, metadata={"hnsw:space": "cosine"})

    if collection.count() == 0:
        return []

    results = collection.query(query_texts=[query], n_results=k)

    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    retrieved = []
    for doc, meta, dist in zip(docs, metas, distances):
        # Convert Chroma L2 distance dist into Cosine Similarity:
        # L2 = sqrt(2 * (1 - cos_sim)) => cos_sim = 1 - (dist^2) / 2
        score = 1.0 - (dist ** 2) / 2.0
        if score < min_score:
            continue
        retrieved.append(
            {
                "text": doc,
                "source": meta.get("source", "unknown"),
                "score": score,
            }
        )
    return retrieved

