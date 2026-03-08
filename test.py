"""
test.py
=======
Smoke tests — verify all pipeline components work before running the full pull.

Run:
    python test.py

Expected: all tests pass. Fix any failures before running main.py.
"""

import sys
import traceback
from datetime import datetime
from unittest.mock import patch

# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

PASS = "...Passed"
FAIL = "...Failed"
results = []


def run_test(name: str, fn):
    try:
        fn()
        print(f"  {PASS}  {name}")
        results.append((name, True))
    except Exception as e:
        print(f"  {FAIL}  {name}")
        traceback.print_exc()
        results.append((name, False))


# ---------------------------------------------------------------------------
# Test 1: Config loads and token is present
# ---------------------------------------------------------------------------

def test_config():
    from config import ACN_BASE_URL, ACN_SITES, ACN_TOKEN, DB_PATH
    assert ACN_TOKEN, "ACN_TOKEN is empty — check your .env file"
    assert ACN_BASE_URL.startswith("https://"), "BASE_URL looks wrong"
    assert len(ACN_SITES) == 3, "Expected 3 sites"
    assert str(DB_PATH).endswith(".duckdb"), "DB_PATH should be a .duckdb file"


# ---------------------------------------------------------------------------
# Test 2: DB connection and schema creation
# ---------------------------------------------------------------------------

def test_db():
    import duckdb
    from db import get_connection

    # Use in-memory DB for the test
    with patch("db.DB_PATH", ":memory:"):
        con = duckdb.connect(":memory:")
        from db import SCHEMA_SQL
        con.execute(SCHEMA_SQL)
        tables = con.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_type='BASE TABLE'"
        ).fetchdf()["table_name"].tolist()
        assert "sessions" in tables, "sessions table not created"
        con.close()


# ---------------------------------------------------------------------------
# Test 3: Datetime parsing
# ---------------------------------------------------------------------------

def test_parse_dt():
    from steps.step2_transform import parse_dt
    result = parse_dt("Wed, 01 May 2019 12:00:00 GMT")
    assert isinstance(result, datetime), "Expected datetime object"
    assert result.year == 2019
    assert result.month == 5
    assert result.day == 1

    assert parse_dt(None) is None
    assert parse_dt("") is None
    assert parse_dt("not-a-date") is None


# ---------------------------------------------------------------------------
# Test 4: Session parsing
# ---------------------------------------------------------------------------

def test_parse_session():
    from steps.step2_transform import parse_session

    raw = {
        "_id": "test_001",
        "sessionID": "S001",
        "siteID": "caltech",
        "stationID": "CA-001",
        "spaceID": "SP-01",
        "clusterID": "CL-01",
        "userID": "U001",
        "connectionTime": "Wed, 01 May 2019 08:00:00 GMT",
        "disconnectTime": "Wed, 01 May 2019 10:00:00 GMT",
        "doneChargingTime": "Wed, 01 May 2019 09:30:00 GMT",
        "kWhDelivered": 12.5,
        "timezone": "America/Los_Angeles",
        "userInputs": [
            {
                "kWhRequested": 15.0,
                "milesRequested": 50.0,
                "minutesAvailable": 120.0,
                "requestedDeparture": "Wed, 01 May 2019 10:00:00 GMT",
                "WhPerMile": 300.0,
                "paymentRequired": False,
                "modifiedAt": "Wed, 01 May 2019 07:55:00 GMT",
            }
        ],
    }

    row = parse_session(raw)
    assert row["_id"] == "test_001"
    assert row["siteID"] == "caltech"
    assert row["kWhDelivered"] == 12.5
    assert isinstance(row["connectionTime"], datetime)
    assert row["kWhRequested"] == 15.0
    assert row["paymentRequired"] is False

    # Missing userInputs — should return None fields, not crash
    raw_no_inputs = {**raw, "userInputs": []}
    row_no_inputs = parse_session(raw_no_inputs)
    assert row_no_inputs["kWhRequested"] is None


# ---------------------------------------------------------------------------
# Test 5: Load (idempotency)
# ---------------------------------------------------------------------------

def test_load_idempotency():
    import duckdb
    import pandas as pd
    from db import SCHEMA_SQL
    from steps.step3_load import load_sessions

    con = duckdb.connect(":memory:")
    con.execute(SCHEMA_SQL)

    df = pd.DataFrame([{
        "_id": "dup_test",
        "sessionID": None, "siteID": "caltech", "stationID": None,
        "spaceID": None, "clusterID": None, "userID": None,
        "connectionTime": datetime(2019, 5, 1, 8, 0),
        "disconnectTime": datetime(2019, 5, 1, 10, 0),
        "doneChargingTime": None, "kWhDelivered": 10.0,
        "timezone": "America/Los_Angeles", "kWhRequested": None,
        "milesRequested": None, "minutesAvailable": None,
        "requestedDeparture": None, "WhPerMile": None, "paymentRequired": None,
    }])

    inserted1, skipped1 = load_sessions(df, con)
    assert inserted1 == 1 and skipped1 == 0, "First insert should succeed"

    inserted2, skipped2 = load_sessions(df, con)
    assert inserted2 == 0 and skipped2 == 1, "Duplicate should be skipped"

    con.close()


# ---------------------------------------------------------------------------
# Test 6: Feature view creation
# ---------------------------------------------------------------------------

def test_feature_views():
    import duckdb
    import pandas as pd
    from db import SCHEMA_SQL
    from steps.step4_features import build_feature_views

    con = duckdb.connect(":memory:")
    con.execute(SCHEMA_SQL)

    # Insert a minimal valid session
    con.execute("""
        INSERT INTO sessions (_id, siteID, stationID, userID, connectionTime,
                              disconnectTime, kWhDelivered, timezone)
        VALUES ('v001', 'caltech', 'CA-001', 'U001',
                TIMESTAMP '2019-05-01 08:00:00',
                TIMESTAMP '2019-05-01 10:00:00',
                12.5, 'America/Los_Angeles')
    """)

    build_feature_views(con)

    views = con.execute(
        "SELECT table_name FROM information_schema.tables WHERE table_type='VIEW'"
    ).fetchdf()["table_name"].tolist()

    expected = [
        "sessions_features",
        "hourly_utilization",
        "seasonal_summary",
        "daily_evse_utilization",
    ]
    for v in expected:
        assert v in views, f"View '{v}' not created"

    # Verify sessions_features returns the row with derived columns
    result = con.execute("SELECT * FROM sessions_features").fetchdf()
    assert len(result) == 1
    assert "hour_of_day" in result.columns
    assert "season" in result.columns
    assert "duration_hours" in result.columns

    con.close()


# ---------------------------------------------------------------------------
# Test 7: Schema context is non-empty and contains key identifiers
# ---------------------------------------------------------------------------

def test_schema_context():
    from engine.schema_context import get_schema_context
    schema = get_schema_context()
    assert len(schema) > 100, "Schema context is too short"
    for expected in ["sessions_features", "hourly_utilization", "siteID", "kWhDelivered", "0001", "0002"]:
        assert expected in schema, f"Schema context missing expected term: {expected!r}"


# ---------------------------------------------------------------------------
# Test 8: SQL safety check blocks write operations
# ---------------------------------------------------------------------------

def test_sql_safety():
    from engine.sql_generator import is_safe_sql
    assert is_safe_sql("SELECT * FROM sessions_features LIMIT 10") is True
    assert is_safe_sql("DROP TABLE sessions") is False
    assert is_safe_sql("INSERT INTO sessions VALUES (1)") is False
    assert is_safe_sql("SELECT * FROM sessions; DELETE FROM sessions") is False


# ---------------------------------------------------------------------------
# Test 9: SQL extraction strips markdown fences
# ---------------------------------------------------------------------------

def test_sql_extraction():
    from engine.sql_generator import extract_sql
    # Plain SQL — returned as-is
    plain = "SELECT COUNT(*) FROM sessions_features"
    assert extract_sql(plain) == plain

    # Wrapped in markdown fence
    fenced = "```sql\nSELECT COUNT(*) FROM sessions_features\n```"
    assert extract_sql(fenced) == "SELECT COUNT(*) FROM sessions_features"

    # Fence without language tag
    fenced2 = "```\nSELECT 1\n```"
    assert extract_sql(fenced2) == "SELECT 1"


# ---------------------------------------------------------------------------
# Test 10: Result formatting
# ---------------------------------------------------------------------------

def test_result_formatting():
    import pandas as pd
    from engine.sql_executor import format_result

    # Single scalar
    df_scalar = pd.DataFrame({"count_star()": [12345]})
    result = format_result(df_scalar, "how many sessions?")
    assert "12,345" in result

    # Single row multi-column
    df_row = pd.DataFrame({"siteID": ["0002"], "avg_kwh": [12.567]})
    result = format_result(df_row, "avg kwh at caltech?")
    assert "12.57" in result

    # Empty result
    df_empty = pd.DataFrame()
    result = format_result(df_empty, "anything")
    assert "No data" in result

    # Multi-row
    df_multi = pd.DataFrame({
        "hour_of_day": list(range(5)),
        "session_count": [10, 20, 30, 40, 50]
    })
    result = format_result(df_multi, "sessions by hour?")
    assert "50" in result

# ---------------------------------------------------------------------------
# Tests 11-12: RAG Knowledge Base
# ---------------------------------------------------------------------------

def test_rag_documents():
    from rag.documents import get_all_chunks, get_chunks_by_topic
    chunks = get_all_chunks()
    assert len(chunks) >= 10, "Expected at least 10 document chunks"
    for chunk in chunks:
        assert "id" in chunk, "Chunk missing 'id'"
        assert "text" in chunk, "Chunk missing 'text'"
        assert "metadata" in chunk, "Chunk missing 'metadata'"
        assert len(chunk["text"]) > 50, f"Chunk {chunk['id']} text too short"
    # Topic filter works
    limitations = get_chunks_by_topic("dataset_limitations")
    assert len(limitations) >= 1, "Expected at least one limitations chunk"


def test_rag_embedding_function():
    """Test OllamaEmbedder instantiation and embedding call (mocked HTTP)."""
    from unittest.mock import MagicMock, patch
    from rag.vector_store import OllamaEmbedder
    embedder = OllamaEmbedder()
    assert embedder.model == "nomic-embed-text:latest"
    assert "api/embeddings" in embedder.embed_url

    # Mock the HTTP call — no Ollama required for this test
    mock_response = MagicMock()
    mock_response.json.return_value = {"embedding": [0.1, 0.2, 0.3]}
    mock_response.raise_for_status = MagicMock()
    with patch("requests.post", return_value=mock_response):
        result = embedder.embed(["test text"])
    assert result.shape == (1, 3)  # numpy array (1 text, 3-dim vector)

# ---------------------------------------------------------------------------
# Tests 13-14: Router
# ---------------------------------------------------------------------------

def test_router_imports():
    """Router module imports and constants are correct."""
    from engine.router import ROUTE_BOTH, ROUTE_RAG, ROUTE_SQL, VALID_ROUTES
    assert ROUTE_SQL  == "sql"
    assert ROUTE_RAG  == "rag"
    assert ROUTE_BOTH == "both"
    assert VALID_ROUTES == {"sql", "rag", "both"}


def test_router_classify(monkeypatch=None):
    """Router returns correct labels and falls back to sql on bad output."""
    from unittest.mock import MagicMock, patch
    from engine.router import classify_query

    def make_mock_response(text):
        mock = MagicMock()
        mock.choices[0].message.content = text
        return mock

    # sql routing
    with patch("engine.router.client") as mock_client:
        mock_client.chat.completions.create.return_value = make_mock_response("sql")
        assert classify_query("How many sessions at Caltech?") == "sql"

    # rag routing
    with patch("engine.router.client") as mock_client:
        mock_client.chat.completions.create.return_value = make_mock_response("rag")
        assert classify_query("What is ACN-Data?") == "rag"

    # both routing
    with patch("engine.router.client") as mock_client:
        mock_client.chat.completions.create.return_value = make_mock_response("both")
        assert classify_query("What are peak hours and why?") == "both"

    # fallback on garbage output
    with patch("engine.router.client") as mock_client:
        mock_client.chat.completions.create.return_value = make_mock_response("I don't know")
        assert classify_query("anything") == "sql"

    # fallback on exception
    with patch("engine.router.client") as mock_client:
        mock_client.chat.completions.create.side_effect = Exception("connection error")
        assert classify_query("anything") == "sql"

# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("\nChargeGPT - Smoke Tests")
    print("=" * 40)

    print("\n[ETL Pipeline]")
    run_test("Config loads correctly",     test_config)
    run_test("DB schema creation",         test_db)
    run_test("Datetime parsing",           test_parse_dt)
    run_test("Session parsing",            test_parse_session)
    run_test("Load idempotency",           test_load_idempotency)
    run_test("Feature view creation",      test_feature_views)

    print("\n[Structured Query Engine]")
    run_test("Schema context",             test_schema_context)
    run_test("SQL safety check",           test_sql_safety)
    run_test("SQL extraction",             test_sql_extraction)
    run_test("Result formatting",          test_result_formatting)

    print("\n[RAG Knowledge Base]")
    run_test("RAG documents",              test_rag_documents)
    run_test("RAG embedding function",     test_rag_embedding_function)

    print("\n[Router]")
    run_test("Router imports",             test_router_imports)
    run_test("Router classify",            test_router_classify)

    print("\n" + "=" * 40)
    passed = sum(1 for _, ok in results if ok)
    total  = len(results)
    print(f"Results: {passed}/{total} passed")

    if passed < total:
        print("\nFix failing tests before proceeding.")
        sys.exit(1)
    else:
        print("\nAll tests passed.")