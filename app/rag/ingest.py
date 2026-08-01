"""
Chunk and embed personal documents into the Chroma vector store.

Uses Chroma's bundled local embedding model, so no embedding API key
is required. For better retrieval quality later, swap in a dedicated
embedding model (e.g. Voyage AI's voyage-3, which Anthropic recommends
for RAG) — see README "Next steps".
"""
import glob
import os

import chromadb

from app.config import CHROMA_PERSIST_DIR, COLLECTION_NAME


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 150) -> list:
    """Simple sliding-window chunker."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start += chunk_size - overlap
    return chunks



def ingest_directory(path: str) -> None:
    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    collection = client.get_or_create_collection(COLLECTION_NAME, metadata={"hnsw:space": "cosine"})

    files = glob.glob(os.path.join(path, "**/*.txt"), recursive=True)
    if not files:
        print(f"No .txt files found under {path}")
        return

    doc_id = 0
    for filepath in files:
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()

        for chunk in chunk_text(text):
            collection.add(
                ids=[f"doc_{doc_id}"],
                documents=[chunk],
                metadatas=[{"source": os.path.basename(filepath)}],
            )
            doc_id += 1

    print(f"Ingested {doc_id} chunks from {len(files)} files into '{COLLECTION_NAME}'")
