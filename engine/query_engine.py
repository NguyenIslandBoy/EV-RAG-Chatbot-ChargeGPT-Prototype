"""
engine/query_engine.py
======================
Top-level interface for the Structured Query Engine.

Orchestrates: natural language question → SQL → DuckDB → plain text answer.
This is the single entry point called by the chatbot layer.
"""

import logging

import duckdb

from engine.sql_executor import execute_sql, format_result
from engine.sql_generator import generate_sql

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Response dataclass
# ---------------------------------------------------------------------------

class QueryResponse:
    """Container for a structured query result."""

    def __init__(
        self,
        answer: str,
        sql: str | None = None,
        success: bool = True,
        error: str | None = None,
    ):
        self.answer = answer
        self.sql = sql
        self.success = success
        self.error = error

    def __repr__(self):
        status = "OK" if self.success else "ERROR"
        return f"QueryResponse({status}): {self.answer[:80]}..."


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_query(question: str, db_con: duckdb.DuckDBPyConnection) -> QueryResponse:
    """
    Execute a natural language question against the EV charging database.

    Pipeline:
      1. Generate SQL from question (LLM)
      2. Execute SQL against DuckDB
      3. Format result as plain text

    Returns a QueryResponse with .answer (always set) and .sql (for debugging).
    Never raises — all errors are caught and returned as QueryResponse with
    success=False so the caller can decide how to handle them.
    """
    log.info(f"Query: {question!r}")

    # Step 1: Generate SQL
    try:
        sql = generate_sql(question)
    except RuntimeError as e:
        log.error(f"SQL generation failed: {e}")
        return QueryResponse(
            answer="I'm having trouble connecting to the language model right now. Please try again.",
            success=False,
            error=str(e),
        )
    except ValueError as e:
        log.error(f"SQL safety check failed: {e}")
        return QueryResponse(
            answer="I can only answer read-only analytical questions about the charging data.",
            success=False,
            error=str(e),
        )

    # Step 2: Handle unsupported queries
    if sql is None:
        return QueryResponse(
            answer=(
                "I can't answer that from the available data. The dataset does not contain "
                "information about carbon intensity, grid emissions, electricity prices, "
                "weather, or vehicle specifications. "
                "I can help with session counts, energy consumption, peak hours, "
                "utilization rates, seasonal patterns, and site comparisons."
            ),
            success=False,
        )

    log.info(f"Generated SQL: {sql}")

    # Step 3: Execute — with one self-correction retry on syntax error
    result_df = execute_sql(sql, db_con)

    if result_df is None:
        log.info("First SQL attempt failed — attempting self-correction retry")
        try:
            from engine.sql_generator import generate_sql_with_error_context
            sql_retry = generate_sql_with_error_context(question, sql)
            if sql_retry and sql_retry != sql:
                log.info(f"Retry SQL: {sql_retry}")
                result_df = execute_sql(sql_retry, db_con)
                if result_df is not None:
                    sql = sql_retry  # use corrected SQL in response
        except Exception as retry_err:
            log.warning(f"Retry attempt failed: {retry_err}")

    if result_df is None:
        return QueryResponse(
            answer=(
                "I generated a query but it failed to execute. "
                "This may be due to an ambiguous question — try rephrasing it."
            ),
            sql=sql,
            success=False,
            error="DuckDB execution error",
        )

    # Step 4: Format
    answer = format_result(result_df, question)
    log.info(f"Answer: {answer[:100]}...")

    return QueryResponse(answer=answer, sql=sql, success=True)