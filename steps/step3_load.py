"""
steps/step3_load.py
===================
Load parsed session DataFrames into DuckDB.

Responsibilities:
  - Idempotent insert (skip sessions already in DB by _id)
  - Returns insert/skip counts for logging
"""

import logging

import duckdb
import pandas as pd

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Insert
# ---------------------------------------------------------------------------

def load_sessions(
    df: pd.DataFrame,
    db_con: duckdb.DuckDBPyConnection,
) -> tuple[int, int]:
    """
    Insert new sessions into the sessions table.
    Skips any rows whose _id already exists (idempotent — safe to re-run).

    Returns:
        (inserted_count, skipped_count)
    """
    if df.empty:
        return 0, 0

    # Find which _ids already exist
    ids = df["_id"].dropna().tolist()
    if not ids:
        return 0, 0

    placeholders = ",".join(f"'{i}'" for i in ids)
    existing_ids = set(
        db_con.execute(
            f"SELECT _id FROM sessions WHERE _id IN ({placeholders})"
        ).fetchdf()["_id"].tolist()
    )

    new_rows = df[~df["_id"].isin(existing_ids)].copy()
    skipped = len(df) - len(new_rows)

    if not new_rows.empty:
        cols = ", ".join(new_rows.columns.tolist())
        db_con.register("_load_batch", new_rows)
        db_con.execute(f"INSERT INTO sessions ({cols}) SELECT {cols} FROM _load_batch")
        db_con.unregister("_load_batch")

    return len(new_rows), skipped