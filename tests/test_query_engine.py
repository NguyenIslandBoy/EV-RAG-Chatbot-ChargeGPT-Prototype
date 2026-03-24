"""
tests/test_query_engine.py
==========================
Tests for engine/query_engine.py — the core pipeline.

All LLM calls are mocked via conftest fixtures.
No Ollama connection required to run these tests.
"""

import pytest
from engine.query_engine import run_query, QueryResponse


class TestQueryResponseDataclass:
    def test_success_defaults(self):
        r = QueryResponse(answer="test answer")
        assert r.success is True
        assert r.error is None
        assert r.sql is None

    def test_failure_state(self):
        r = QueryResponse(answer="error", success=False, error="something went wrong")
        assert r.success is False
        assert "something" in r.error

    def test_repr_shows_ok_on_success(self):
        r = QueryResponse(answer="hello world")
        assert "OK" in repr(r)

    def test_repr_shows_error_on_failure(self):
        r = QueryResponse(answer="oops", success=False)
        assert "ERROR" in repr(r)


class TestRunQueryHappyPath:
    def test_returns_query_response_type(self, mock_db, mock_generate_sql_success):
        result = run_query("How many sessions are there?", mock_db)
        assert isinstance(result, QueryResponse)

    def test_success_true_on_valid_query(self, mock_db, mock_generate_sql_success):
        result = run_query("How many sessions?", mock_db)
        assert result.success is True

    def test_sql_stored_in_response(self, mock_db, mock_generate_sql_success):
        result = run_query("How many sessions?", mock_db)
        assert result.sql is not None
        assert "SELECT" in result.sql.upper()

    def test_answer_is_non_empty_string(self, mock_db, mock_generate_sql_success):
        result = run_query("How many sessions?", mock_db)
        assert isinstance(result.answer, str)
        assert len(result.answer) > 0

    def test_answer_contains_count_result(self, mock_db, mock_generate_sql_success):
        """The query returns COUNT(*) = 3 from mock_db — answer should contain 3."""
        result = run_query("How many sessions?", mock_db)
        assert "3" in result.answer


class TestRunQueryErrorHandling:
    def test_ollama_connection_error_returns_graceful_response(
        self, mock_db, mock_generate_sql_runtime_error
    ):
        result = run_query("How many sessions?", mock_db)
        assert result.success is False
        assert result.error is not None
        assert isinstance(result.answer, str)

    def test_unsupported_query_returns_success_false(
        self, mock_db, mock_generate_sql_none
    ):
        result = run_query("What is the carbon intensity?", mock_db)
        assert result.success is False
        assert result.sql is None

    def test_unsupported_query_answer_explains_limitation(
        self, mock_db, mock_generate_sql_none
    ):
        """Refusal message should be informative, not silent."""
        result = run_query("What is the carbon intensity?", mock_db)
        assert len(result.answer) > 20

    def test_unsupported_query_mentions_data_scope(
        self, mock_db, mock_generate_sql_none
    ):
        """Refusal message should hint at what CAN be answered."""
        result = run_query("What is the carbon intensity?", mock_db)
        answer_lower = result.answer.lower()
        # Should mention the limitation or suggest alternatives
        assert any(word in answer_lower for word in [
            "can't", "cannot", "available", "carbon", "session", "energy"
        ])

    def test_run_query_never_raises(self, mock_db, mock_generate_sql_runtime_error):
        """
        Core contract: run_query must never propagate exceptions.
        All errors are returned as QueryResponse with success=False.
        """
        try:
            run_query("anything", mock_db)
        except Exception as e:
            pytest.fail(f"run_query raised an exception when it must not: {e}")

    def test_run_query_never_raises_on_bad_sql(
        self, mock_db, mock_generate_sql_bad_syntax
    ):
        """Bad SQL from LLM should be caught, not propagated."""
        try:
            result = run_query("What is the peak hour?", mock_db)
            # Either succeeds via retry or returns graceful failure
            assert isinstance(result, QueryResponse)
        except Exception as e:
            pytest.fail(f"run_query raised on bad SQL: {e}")