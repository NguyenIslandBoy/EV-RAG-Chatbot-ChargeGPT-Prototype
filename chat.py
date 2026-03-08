"""
chat.py
=======
ChargeGPT — top-level chat interface.

Wires together:
  Router       → classify question (sql / rag / both)
  Query Engine → Text-to-SQL → DuckDB → grounded numerical answer
  RAG Engine   → vector retrieval → LLM → grounded contextual answer

Usage (interactive):
    python chat.py

Usage (single query):
    python chat.py --query "What are the peak charging hours?"

Usage (with SQL debug output):
    python chat.py --debug
"""

import argparse
import logging

import config  # noqa: F401 — triggers logging setup
from db import get_connection
from engine.query_engine import run_query
from engine.router import ROUTE_BOTH, ROUTE_RAG, ROUTE_SQL, classify_query
from rag.rag_engine import run_rag_query
from rag.vector_store import build_vector_store

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Response merger for "both" route
# ---------------------------------------------------------------------------

def merge_responses(sql_answer: str, rag_answer: str) -> str:
    """
    Combine a data answer and a contextual answer into one coherent response.
    Simple concatenation with clear section labels.
    """
    return (
        f"**Data answer:**\n{sql_answer}\n\n"
        f"**Context:**\n{rag_answer}"
    )


# ---------------------------------------------------------------------------
# Core ask function
# ---------------------------------------------------------------------------

def ask(
    question: str,
    db_con,
    vector_store,
    debug: bool = False,
) -> str:
    """
    Process a single question end-to-end.

    Returns the final plain-text answer string.
    """
    # Step 1: Route
    route = classify_query(question)
    log.info(f"Routed to: {route}")

    # Step 2: Execute
    if route == ROUTE_SQL:
        response = run_query(question, db_con)
        if debug and response.sql:
            print(f"\n  [SQL] {response.sql}\n")
        return response.answer

    elif route == ROUTE_RAG:
        response = run_rag_query(question, vector_store)
        return response.answer

    elif route == ROUTE_BOTH:
        sql_response = run_query(question, db_con)
        rag_response = run_rag_query(question, vector_store)

        if debug and sql_response.sql:
            print(f"\n  [SQL] {sql_response.sql}\n")

        # If either subsystem failed, return whichever succeeded
        if not sql_response.success:
            return rag_response.answer
        if not rag_response.success:
            return sql_response.answer

        return merge_responses(sql_response.answer, rag_response.answer)

    # Should never reach here (router defaults to sql)
    return "I couldn't determine how to handle that question. Please try rephrasing."


# ---------------------------------------------------------------------------
# Interactive loop
# ---------------------------------------------------------------------------

def run_interactive(db_con, vector_store, debug: bool = False):
    print("\n" + "=" * 55)
    print("  ChargeGPT — EV Charging Data Assistant")
    print("  Type 'exit' or 'quit' to stop")
    print("=" * 55 + "\n")

    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not question:
            continue
        if question.lower() in {"exit", "quit", "q"}:
            print("Goodbye.")
            break

        answer = ask(question, db_con, vector_store, debug=debug)
        print(f"\nChargeGPT: {answer}\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="ChargeGPT — EV Charging Data Assistant")
    parser.add_argument("--query",  type=str, help="Single question (non-interactive mode)")
    parser.add_argument("--debug",  action="store_true", help="Show generated SQL alongside answers")
    args = parser.parse_args()

    # Initialise shared resources
    log.info("Loading database...")
    db_con = get_connection()

    log.info("Loading vector store...")
    vector_store = build_vector_store()

    if args.query:
        # Single query mode
        answer = ask(args.query, db_con, vector_store, debug=args.debug)
        print(f"\n{answer}\n")
    else:
        # Interactive mode
        run_interactive(db_con, vector_store, debug=args.debug)

    db_con.close()


if __name__ == "__main__":
    main()