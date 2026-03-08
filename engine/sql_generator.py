"""
engine/sql_generator.py
=======================
Generates DuckDB SQL from a natural language query using an OpenAI LLM.

Responsibilities:
  - Build the system + user prompt
  - Call OpenAI chat completion
  - Extract and clean the SQL from the response
  - Basic SQL safety validation (read-only enforcement)
"""

import logging
import re

from openai import OpenAI

from config import OLLAMA_BASE_URL, OLLAMA_MODEL
from engine.schema_context import get_schema_context

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# OpenAI client
# ---------------------------------------------------------------------------

client = OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama")

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a DuckDB SQL expert assistant for an EV charging analytics system.

Your job is to convert natural language questions into a single valid DuckDB SQL query.

RULES:
1. Return ONLY the SQL query — no explanation, no markdown, no backticks, no preamble.
2. Always use sessions_features (not raw sessions) for analytical queries unless explicitly asked for raw data.
3. Use site_lookup to resolve site names when the user mentions 'Caltech', 'JPL', or 'Office 1'.
4. Only generate SELECT statements. Never generate INSERT, UPDATE, DELETE, DROP, or DDL.
5. Use ROUND(value, 2) for all floating point results.
6. Always include a LIMIT clause — default to LIMIT 100 unless the query is a single aggregation.
7. For time-of-day queries, use hour_of_day from sessions_features or hourly_utilization.
8. For seasonal queries, use the seasonal_summary view.
9. "Peak hours" or "busiest hours" always means highest session_count, never kWh or charge rate unless explicitly asked.
10. If the question cannot be answered from the available schema, return exactly: UNSUPPORTED_QUERY

RETURN UNSUPPORTED_QUERY FOR:
- Carbon intensity, grid emissions, CO2, carbon footprint (no grid carbon data in schema)
- Electricity prices or costs (no pricing data in schema)
- Weather or temperature (no weather data in schema)
- Vehicle make, model, or battery capacity (not in schema)
- Predictions or forecasts about future behaviour
- Any question requiring data columns that do not exist in the schema

SCHEMA:
{schema}
"""

USER_PROMPT_TEMPLATE = """Question: {question}

SQL:"""


# ---------------------------------------------------------------------------
# SQL safety check
# ---------------------------------------------------------------------------

FORBIDDEN_KEYWORDS = {"insert", "update", "delete", "drop", "alter", "create",
                       "truncate", "replace", "merge", "exec", "execute"}

def is_safe_sql(sql: str) -> bool:
    """Reject any SQL containing write or DDL keywords."""
    tokens = set(re.findall(r"\b\w+\b", sql.lower()))
    violations = tokens & FORBIDDEN_KEYWORDS
    if violations:
        log.warning(f"Unsafe SQL rejected — forbidden keywords: {violations}")
        return False
    return True


# ---------------------------------------------------------------------------
# SQL extraction
# ---------------------------------------------------------------------------

def extract_sql(raw_response: str) -> str:
    """
    Clean up the LLM response to extract just the SQL.
    Handles markdown fences and fixes common phi3.5 output corruption patterns.

    Known phi3.5 bugs this corrects:
    1. Semicolon before LIMIT:   "FROM t;\nLIMIT 100"         -> "FROM t LIMIT 100"
    2. Garbage after semicolon:  "SELECT ...;\nWITH ROUND..."  -> first stmt only
    3. GROUP BY corruption:      "GROUP BY hour_of extr"       -> "GROUP BY hour_of_day"
    4. Prose after valid SQL:    "SELECT ...\n\nNote: This..." -> truncated at Note
    5. Incomplete CASE in ORDER: missing END before LIMIT      -> END inserted
    """
    # Step 1: Extract from markdown fence if present
    fenced = re.search(r"```(?:sql)?\s*(.*?)```", raw_response, re.DOTALL | re.IGNORECASE)
    sql = fenced.group(1).strip() if fenced else raw_response.strip()

    # Step 2: Take only first SELECT statement (handles ";\nWITH ROUND(...)" corruption)
    first_stmt = re.match(r"(SELECT\s.+?);\s*(?:WITH|ROUND|FROM|SELECT|\n\n)", sql, re.DOTALL | re.IGNORECASE)
    if first_stmt:
        sql = first_stmt.group(1).strip()

    # Step 3: Fix semicolon immediately before LIMIT
    sql = re.sub(r";\s*(?=LIMIT\s+\d+)", " ", sql, flags=re.IGNORECASE)

    # Step 4: Fix GROUP BY token corruption: "GROUP BY hour_of extr" -> "GROUP BY hour_of_day"
    sql = re.sub(r"GROUP\s+BY\s+hour_of\s+\w+", "GROUP BY hour_of_day", sql, flags=re.IGNORECASE)

    # Step 5: Truncate prose commentary after valid SQL
    prose = re.search(r"\n\s*(?:Note:|This query|To compare|The query|Explanation:)", sql, re.IGNORECASE)
    if prose:
        sql = sql[:prose.start()].strip()

    # Step 6: Fix incomplete CASE in ORDER BY — insert END before LIMIT if missing
    sql = re.sub(
        r"(CASE\s+\w+\s+WHEN\s+.+?THEN\s+\d+)\s*\n(\s*LIMIT)",
        r"\1 END\n\2",
        sql,
        flags=re.DOTALL | re.IGNORECASE,
    )

    sql = sql.rstrip("; \n")
    return sql


# ---------------------------------------------------------------------------
# Main generator
# ---------------------------------------------------------------------------

def generate_sql(question: str) -> str | None:
    """
    Convert a natural language question to a DuckDB SQL query.

    Returns:
        SQL string if successful
        None if the query is unsupported or generation fails
    
    Raises:
        RuntimeError on API failure
    """
    system = SYSTEM_PROMPT.format(schema=get_schema_context())
    user = USER_PROMPT_TEMPLATE.format(question=question)

    log.debug(f"Generating SQL for: {question!r}")

    try:
        response = client.chat.completions.create(
            model=OLLAMA_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
            temperature=0,        # deterministic — we want consistent SQL
            max_tokens=500,
        )
    except Exception as e:
        raise RuntimeError(f"OpenAI API call failed: {e}") from e

    raw = response.choices[0].message.content or ""
    sql = extract_sql(raw)

    log.debug(f"Generated SQL: {sql}")

    if sql.strip().upper() == "UNSUPPORTED_QUERY":
        log.info(f"Query marked unsupported by LLM: {question!r}")
        return None

    if not is_safe_sql(sql):
        raise ValueError(f"LLM generated unsafe SQL for query: {question!r}")

    return sql


def generate_sql_with_error_context(question: str, failed_sql: str) -> str | None:
    """
    Retry SQL generation with the failed SQL and its error context.
    Called when the first attempt produces a syntax error.
    """
    system = SYSTEM_PROMPT.format(schema=get_schema_context())
    user = f"""Question: {question}

Your previous SQL attempt failed with a syntax error:

FAILED SQL:
{failed_sql}

Common errors to avoid:
- Never write "GROUP BY hour_of extr" — always write "GROUP BY hour_of_day"
- Never append text after semicolons — return ONE complete SQL statement only
- Always close CASE statements with END before LIMIT
- Use the exact alias name from SELECT in ORDER BY

Please write a corrected SQL query:

SQL:"""

    try:
        response = client.chat.completions.create(
            model=OLLAMA_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0,
            max_tokens=500,
        )
    except Exception as e:
        raise RuntimeError(f"Retry SQL generation failed: {e}") from e

    raw = response.choices[0].message.content or ""
    sql = extract_sql(raw)

    if sql.strip().upper() == "UNSUPPORTED_QUERY":
        return None
    if not is_safe_sql(sql):
        return None

    return sql