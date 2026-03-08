"""
engine/sql_executor.py
======================
Executes generated SQL against DuckDB and formats the result as plain text.

Responsibilities:
  - Safe SQL execution with error handling
  - Result → plain text answer formatting
  - Handles both single-value and multi-row results
"""

import logging

import duckdb
import pandas as pd

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------

def execute_sql(sql: str, db_con: duckdb.DuckDBPyConnection) -> pd.DataFrame | None:
    """
    Execute a SQL query against DuckDB.

    Returns:
        DataFrame of results, or None on execution error.
    """
    try:
        result = db_con.execute(sql).fetchdf()
        log.debug(f"Query returned {len(result)} rows, {len(result.columns)} columns")
        return result
    except duckdb.Error as e:
        log.error(f"DuckDB execution error: {e}\nSQL: {sql}")
        return None


# ---------------------------------------------------------------------------
# Result formatting
# ---------------------------------------------------------------------------

def format_result(df: pd.DataFrame, question: str) -> str:
    """
    Convert a query result DataFrame into a plain text answer.

    Formatting rules:
      - Single value  → inline answer sentence
      - Single row    → key-value pairs
      - Multiple rows → compact table (max 20 rows shown)
    """
    if df is None or df.empty:
        return "No data found for that query. The dataset may not contain matching records."

    rows, cols = df.shape

    # --- Single scalar result (e.g. COUNT(*), AVG(...)) ---
    if rows == 1 and cols == 1:
        value = df.iloc[0, 0]
        col_name = df.columns[0]
        formatted_value = _format_value(value)
        return f"{formatted_value}"

    # --- Single row, multiple columns ---
    if rows == 1:
        parts = []
        for col in df.columns:
            parts.append(f"{_humanise_col(col)}: {_format_value(df.iloc[0][col])}")
        return "\n".join(parts)

    # --- Multiple rows ---
    # Cap display at 20 rows to keep responses readable
    display_df = df.head(20)
    truncated = rows > 20

    # Humanise column names
    display_df = display_df.copy()
    display_df.columns = [_humanise_col(c) for c in display_df.columns]

    # Format numeric columns
    for col in display_df.select_dtypes(include="number").columns:
        display_df[col] = display_df[col].apply(_format_value)

    table = display_df.to_string(index=False)
    note = f"\n(Showing 20 of {rows} rows)" if truncated else ""

    return f"{table}{note}"


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _format_value(value) -> str:
    """Format a single cell value cleanly. Handles Python and numpy numeric types."""
    if pd.isna(value):
        return "N/A"
    # Use pd.api.types to handle numpy int64/float64 alongside Python native types
    if pd.api.types.is_bool_dtype(type(value)):
        return str(value)
    if pd.api.types.is_float_dtype(type(value)) or isinstance(value, float):
        if value == int(value):
            return f"{int(value):,}"
        return f"{value:,.2f}"
    if pd.api.types.is_integer_dtype(type(value)) or isinstance(value, int):
        return f"{int(value):,}"
    return str(value)


def _humanise_col(col: str) -> str:
    """
    Convert snake_case / camelCase column names to readable labels.
    e.g. 'avg_kwh_per_session' → 'Avg Kwh Per Session'
         'siteID' → 'Site ID'
    """
    # Insert space before uppercase letters (camelCase)
    spaced = re.sub(r"([A-Z])", r" \1", col)
    # Replace underscores with spaces
    spaced = spaced.replace("_", " ")
    # Title case and strip
    return spaced.strip().title()


# Need re for _humanise_col
import re  # noqa: E402 — imported here to keep it close to usage