"""
rag/vector_store.py
===================
FAISS-based persistent vector store for the RAG knowledge base.

Replaces ChromaDB which is broken on Python 3.14 (Pydantic V1 incompatibility).
FAISS has no Pydantic dependency — zero compatibility issues.

Persistence: index + metadata saved as two files in data/chroma/:
  - faiss_index.bin   : the FAISS index
  - faiss_meta.json   : chunk ids, texts, and metadata

Responsibilities:
  - Build and persist the FAISS index from document chunks
  - Query the index with a natural language question
  - Return top-k relevant chunks with similarity scores
"""

import json
import logging
from pathlib import Path

import faiss
import numpy as np
import requests

from config import OLLAMA_BASE_URL, OLLAMA_EMBED_MODEL, RAG_DB_PATH
from rag.documents import get_all_chunks

log = logging.getLogger(__name__)

INDEX_PATH = Path(RAG_DB_PATH) / "faiss_index.bin"
META_PATH  = Path(RAG_DB_PATH) / "faiss_meta.json"


# ---------------------------------------------------------------------------
# Ollama embeddings
# ---------------------------------------------------------------------------

class OllamaEmbedder:
    """
    Calls Ollama's /api/embeddings endpoint to produce dense vector embeddings.
    Uses nomic-embed-text — a strong 768-dim local embedding model.
    """

    def __init__(
        self,
        model: str = OLLAMA_EMBED_MODEL,
        base_url: str = OLLAMA_BASE_URL,
    ):
        self.model = model
        self.embed_url = base_url.replace("/v1", "") + "/api/embeddings"

    def embed(self, texts: list[str]) -> np.ndarray:
        """
        Embed a list of texts. Returns float32 numpy array (n_texts, dim).
        """
        vectors = []
        for text in texts:
            try:
                resp = requests.post(
                    self.embed_url,
                    json={"model": self.model, "prompt": text},
                    timeout=30,
                )
                resp.raise_for_status()
                vectors.append(resp.json()["embedding"])
            except Exception as e:
                raise RuntimeError(f"Ollama embedding failed: {e}") from e

        arr = np.array(vectors, dtype=np.float32)
        # L2-normalise for cosine similarity via inner product
        faiss.normalize_L2(arr)
        return arr

    def embed_one(self, text: str) -> np.ndarray:
        """Embed a single text. Returns shape (1, dim)."""
        return self.embed([text])


# ---------------------------------------------------------------------------
# VectorStore — wraps FAISS index + metadata
# ---------------------------------------------------------------------------

class VectorStore:
    """
    Thin wrapper around a FAISS IndexFlatIP (inner product = cosine after normalisation).
    Stores chunk metadata alongside the index for result reconstruction.
    """

    def __init__(self, embedder: OllamaEmbedder):
        self.embedder = embedder
        self.index: faiss.IndexFlatIP | None = None
        self.chunks: list[dict] = []   # parallel list to index vectors

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(self, chunks: list[dict]) -> None:
        """Embed all chunks and build the FAISS index."""
        log.info(f"Embedding {len(chunks)} chunks with {self.embedder.model}...")
        texts = [c["text"] for c in chunks]
        vectors = self.embedder.embed(texts)

        dim = vectors.shape[1]
        self.index = faiss.IndexFlatIP(dim)   # inner product on L2-normalised = cosine
        self.index.add(vectors)
        self.chunks = chunks
        log.info(f"FAISS index built: {self.index.ntotal} vectors, dim={dim}")

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self) -> None:
        """Save index and metadata to disk."""
        if self.index is None:
            raise RuntimeError("Cannot save — index not built yet.")
        faiss.write_index(self.index, str(INDEX_PATH))
        with open(META_PATH, "w", encoding="utf-8") as f:
            json.dump(self.chunks, f, indent=2, ensure_ascii=False)
        log.info(f"Vector store saved to {RAG_DB_PATH}")

    def load(self) -> bool:
        """
        Load index and metadata from disk.
        Returns True if successful, False if files don't exist.
        """
        if not INDEX_PATH.exists() or not META_PATH.exists():
            return False
        self.index = faiss.read_index(str(INDEX_PATH))
        with open(META_PATH, encoding="utf-8") as f:
            self.chunks = json.load(f)
        log.info(f"Vector store loaded: {self.index.ntotal} vectors")
        return True

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def query(self, question: str, n_results: int = 3) -> list[dict]:
        """
        Retrieve top-k most relevant chunks for a question.

        Returns:
            List of dicts: {id, text, metadata, score}
            Score is cosine similarity (0-1, higher = more relevant).
        """
        if self.index is None or self.index.ntotal == 0:
            raise RuntimeError("Vector store is empty — run build_vector_store() first.")

        query_vec = self.embedder.embed_one(question)
        k = min(n_results, self.index.ntotal)
        scores, indices = self.index.search(query_vec, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            chunk = self.chunks[idx].copy()
            chunk["score"] = float(score)
            # Convert score to distance (1 - cosine_sim) to match previous interface
            chunk["distance"] = float(1.0 - score)
            results.append(chunk)

        return results


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_vector_store(force_rebuild: bool = False) -> VectorStore:
    """
    Build (or load) the FAISS vector store.

    Args:
        force_rebuild: If True, re-embed all chunks even if saved index exists.

    Returns:
        Loaded VectorStore ready for querying.
    """
    embedder = OllamaEmbedder()
    store = VectorStore(embedder)

    if not force_rebuild and store.load():
        log.info("Vector store loaded from disk.")
        return store

    log.info("Building vector store from scratch...")
    store.build(get_all_chunks())
    store.save()
    return store


def query_knowledge_base(
    question: str,
    store: VectorStore,
    n_results: int = 3,
) -> list[dict]:
    """
    Retrieve top-k relevant chunks. Thin wrapper for consistent interface.
    Returns list of dicts with keys: id, text, metadata, distance.
    """
    return store.query(question, n_results=n_results)