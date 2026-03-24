"""
tests/conftest.py
=================
Shared pytest fixtures for ChargeGPT test suite.

All fixtures avoid real LLM calls — Ollama is mocked throughout.
The mock_db fixture uses an in-memory DuckDB instance with a minimal
schema that mirrors the real sessions table structure.
"""

import sys
from pathlib import Path
from unittest.mock import patch

import duckdb
import pytest

# Ensure project root is on path regardless of where pytest is invoked from
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def mock_db():
    """
    In-memory DuckDB connection with minimal schema for testing.
    Mirrors the real sessions table — enough for query_engine tests.
    """
    con = duckdb.connect(":memory:")
    con.execute("""
        CREATE TABLE sessions (
            session_id     VARCHAR,
            site_name      VARCHAR,
            energy_kwh     DOUBLE,
            duration_hours DOUBLE,
            start_time     TIMESTAMP
        )
    """)
    con.execute("""
        INSERT INTO sessions VALUES
        ('s001', 'caltech', 15.2, 2.5, '2021-06-01 08:00:00'),
        ('s002', 'jpl',     22.1, 3.1, '2021-06-01 09:00:00'),
        ('s003', 'caltech',  8.4, 1.2, '2021-06-02 14:00:00')
    """)
    # Also create sessions_features view (used by most real queries)
    con.execute("""
        CREATE VIEW sessions_features AS
        SELECT
            session_id,
            site_name,
            energy_kwh     AS kWhDelivered,
            duration_hours,
            start_time     AS connectionTime
        FROM sessions
    """)
    yield con
    con.close()


@pytest.fixture
def mock_generate_sql_success():
    """Mocks generate_sql to return a valid SELECT query."""
    with patch("engine.query_engine.generate_sql") as mock:
        mock.return_value = "SELECT COUNT(*) AS total FROM sessions"
        yield mock


@pytest.fixture
def mock_generate_sql_none():
    """Simulates an unsupported query — LLM returns None (UNSUPPORTED_QUERY)."""
    with patch("engine.query_engine.generate_sql") as mock:
        mock.return_value = None
        yield mock


@pytest.fixture
def mock_generate_sql_runtime_error():
    """Simulates Ollama being unavailable — connection refused."""
    with patch("engine.query_engine.generate_sql") as mock:
        mock.side_effect = RuntimeError("Ollama connection refused")
        yield mock


@pytest.fixture
def mock_generate_sql_bad_syntax():
    """Simulates phi3.5 producing syntactically broken SQL."""
    with patch("engine.query_engine.generate_sql") as mock:
        mock.return_value = "SELECT COUNT(*) FROM sessions GROUP BY hour_of extr"
        yield mock