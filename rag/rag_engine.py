"""
from rag.vector_store import VectorStore
rag/rag_engine.py
=================
Top-level RAG query interface.

Takes a natural language question → retrieves relevant chunks →
passes to LLM with context → returns grounded plain text answer.
"""

import logging


from openai import OpenAI

from config import OLLAMA_BASE_URL, OLLAMA_MODEL
from rag.vector_store import query_knowledge_base, VectorStore

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# LLM client (same Ollama setup as the query engine)
# ---------------------------------------------------------------------------

client = OpenAI(
    base_url=OLLAMA_BASE_URL,
    api_key="ollama",
)

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are ChargeGPT, an expert assistant on EV charging infrastructure and the ACN-Data dataset.

Answer the user's question using ONLY the provided context passages.
If the context does not contain enough information to answer, say so clearly — do not invent facts.
Keep answers concise, accurate, and grounded in the context.
Do not mention that you are using context passages or a knowledge base — just answer naturally."""

USER_PROMPT_TEMPLATE = """Context:
{context}

Question: {question}

Answer:"""

# Similarity distance threshold — chunks above this are too dissimilar to be useful
DISTANCE_THRESHOLD = 0.6


# ---------------------------------------------------------------------------
# RAG response dataclass
# ---------------------------------------------------------------------------

class RAGResponse:
    def __init__(
        self,
        answer: str,
        chunks_used: list[dict],
        success: bool = True,
        error: str | None = None,
    ):
        self.answer = answer
        self.chunks_used = chunks_used
        self.success = success
        self.error = error

    def __repr__(self):
        return f"RAGResponse(chunks={len(self.chunks_used)}): {self.answer[:80]}..."


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_rag_query(
    question: str,
    store: VectorStore,
    n_results: int = 3,
) -> RAGResponse:
    """
    Answer a conceptual/contextual question using RAG.

    Pipeline:
      1. Retrieve top-k relevant chunks from ChromaDB
      2. Filter out chunks that are too dissimilar
      3. Build context string
      4. Call LLM with context + question
      5. Return grounded answer

    Never raises — errors are caught and returned in RAGResponse.
    """
    # Step 1: Retrieve
    try:
        chunks = query_knowledge_base(question, store, n_results=n_results)
    except Exception as e:
        log.error(f"Vector store query failed: {e}")
        return RAGResponse(
            answer="I'm having trouble accessing the knowledge base right now. Please try again.",
            chunks_used=[],
            success=False,
            error=str(e),
        )

    # Step 2: Filter by relevance threshold
    relevant_chunks = [c for c in chunks if c["distance"] < DISTANCE_THRESHOLD]

    if not relevant_chunks:
        log.info(f"No relevant chunks found for: {question!r} (best distance: {chunks[0]['distance']:.3f})")
        return RAGResponse(
            answer=(
                "I don't have specific documentation to answer that question. "
                "If it's a question about the charging data statistics (e.g. peak hours, "
                "average energy), try asking it directly and I'll query the database."
            ),
            chunks_used=[],
            success=False,
        )

    log.info(f"Retrieved {len(relevant_chunks)} relevant chunks (distances: "
             f"{[round(c['distance'], 3) for c in relevant_chunks]})")

    # Step 3: Build context
    context = "\n\n---\n\n".join(c["text"] for c in relevant_chunks)

    # Step 4: LLM call
    try:
        response = client.chat.completions.create(
            model=OLLAMA_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": USER_PROMPT_TEMPLATE.format(
                    context=context,
                    question=question,
                )},
            ],
            temperature=0.1,  # slight creativity for natural answers, but mostly grounded
            max_tokens=400,
        )
        answer = response.choices[0].message.content or ""
    except Exception as e:
        log.error(f"LLM call failed: {e}")
        return RAGResponse(
            answer="I'm having trouble generating a response right now. Please try again.",
            chunks_used=relevant_chunks,
            success=False,
            error=str(e),
        )

    log.info(f"RAG answer: {answer[:100]}...")
    return RAGResponse(answer=answer.strip(), chunks_used=relevant_chunks, success=True)