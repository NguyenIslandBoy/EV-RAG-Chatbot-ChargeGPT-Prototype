"""
tests/test_scoring.py
=====================
Tests for eval/evaluate.py scoring functions.

These are pure functions with no LLM calls — fast to run,
no fixtures needed beyond imports.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from eval.evaluate import (
    extract_first_number,
    score_categorical,
    score_keywords,
    score_numerical,
    score_out_of_scope,
)


class TestExtractFirstNumber:
    def test_plain_integer(self):
        assert extract_first_number("There are 42 sessions") == 42.0

    def test_comma_formatted_number(self):
        assert extract_first_number("Total: 66,745 kWh") == 66745.0

    def test_decimal(self):
        assert extract_first_number("Average: 15.7 kWh") == 15.7

    def test_no_number_returns_none(self):
        assert extract_first_number("No numbers here") is None

    def test_negative_number(self):
        assert extract_first_number("Change: -3.2%") == -3.2

    def test_first_number_wins(self):
        """When multiple numbers present, first one is extracted."""
        assert extract_first_number("66446 sessions across 115 stations") == 66446.0


class TestScoreNumerical:
    def test_exact_match(self):
        result = score_numerical("The answer is 100", "100", tolerance=0.05)
        assert result["correct"] is True

    def test_within_tolerance(self):
        result = score_numerical("Approximately 98 sessions", "100", tolerance=0.05)
        assert result["correct"] is True

    def test_outside_tolerance(self):
        result = score_numerical("About 80 sessions", "100", tolerance=0.05)
        assert result["correct"] is False

    def test_no_number_in_response(self):
        result = score_numerical("I don't know", "100", tolerance=0.05)
        assert result["correct"] is False
        assert result["predicted"] is None

    def test_zero_ground_truth(self):
        result = score_numerical("The answer is 0", "0", tolerance=0.05)
        assert result["correct"] is True

    def test_relative_error_reported(self):
        result = score_numerical("About 90", "100", tolerance=0.05)
        assert "relative_error" in result
        assert abs(result["relative_error"] - 0.1) < 0.001

    def test_comma_in_response_handled(self):
        """Comma-formatted numbers like 66,446 should parse correctly."""
        result = score_numerical("Total: 66,446 sessions", "66446", tolerance=0.0)
        assert result["correct"] is True


class TestScoreCategorical:
    def test_exact_match(self):
        result = score_categorical("The site is caltech", "0002")
        assert result["correct"] is True  # alias: caltech → 0002

    def test_alias_match_office1(self):
        result = score_categorical("Site 0019 had the highest usage", "0019")
        assert result["correct"] is True

    def test_alias_match_office_spelled_out(self):
        result = score_categorical("Office 1 delivered the most energy", "0019")
        assert result["correct"] is True

    def test_alias_match_tuesday(self):
        result = score_categorical("Tuesday has the most sessions", "2")
        assert result["correct"] is True

    def test_alias_match_fall(self):
        result = score_categorical("Autumn is the busiest season", "autumn")
        assert result["correct"] is True

    def test_no_match(self):
        result = score_categorical("I don't know the site", "0019")
        assert result["correct"] is False

    def test_case_insensitive(self):
        result = score_categorical("CALTECH has the highest count", "0002")
        assert result["correct"] is True


class TestScoreKeywords:
    def test_all_keywords_present(self):
        result = score_keywords(
            "The peak hour is 9am with high utilization and demand",
            keywords=["peak", "utilization"],
            min_matched=2,
        )
        assert result["correct"] is True

    def test_below_threshold(self):
        result = score_keywords(
            "The charging session completed",
            keywords=["peak", "utilization", "demand"],
            min_matched=2,
        )
        assert result["correct"] is False

    def test_case_insensitive(self):
        result = score_keywords(
            "PEAK hours show HIGH UTILIZATION",
            keywords=["peak", "utilization"],
            min_matched=2,
        )
        assert result["correct"] is True

    def test_min_matched_one(self):
        result = score_keywords(
            "The dataset covers workplace charging",
            keywords=["workplace", "residential", "public"],
            min_matched=1,
        )
        assert result["correct"] is True

    def test_matched_keywords_reported(self):
        result = score_keywords(
            "workplace charging at caltech and jpl",
            keywords=["workplace", "caltech", "residential"],
            min_matched=1,
        )
        assert "workplace" in result["matched_keywords"]
        assert "residential" not in result["matched_keywords"]


class TestScoreOutOfScope:
    def test_refusal_is_correct(self):
        result = score_out_of_scope(
            "I can't answer that. The dataset does not contain information about carbon intensity.",
            keywords=[],
            min_matched=0,
        )
        assert result["correct"] is True
        assert result["hallucinated"] is False

    def test_query_engine_unsupported_message_is_correct(self):
        """The actual UNSUPPORTED_QUERY response from query_engine.py should score correctly."""
        response = (
            "I can't answer that from the available data. The dataset does not contain "
            "information about carbon intensity, grid emissions, electricity prices, "
            "weather, or vehicle specifications."
        )
        result = score_out_of_scope(response, keywords=[], min_matched=0)
        assert result["correct"] is True
        assert result["hallucinated"] is False

    def test_hallucinated_large_number_is_incorrect(self):
        # Hallucination = large fabricated number with no refusal signal
        # correct=True/False depends on keyword scorer; hallucinated flag is the key metric
        result = score_out_of_scope(
            "The average charging rate is 987654 kW across all stations.",
            keywords=["not available", "cannot answer"],
            min_matched=1,
        )
        assert result["hallucinated"] is True
        assert result["has_refusal_signal"] is False
        assert result["correct"] is False

    def test_year_not_treated_as_hallucination(self):
        """4-digit numbers (years) must not trigger the hallucination flag."""
        result = score_out_of_scope(
            "Data from 2021 shows no emissions information available.",
            keywords=[],
            min_matched=0,
        )
        assert result["hallucinated"] is False

    def test_five_digit_number_triggers_hallucination(self):
        """5+ digit numbers without refusal signal = hallucination."""
        result = score_out_of_scope(
            "The emissions are 45000 tonnes of CO2 per year.",
            keywords=[],
            min_matched=0,
        )
        assert result["hallucinated"] is True

    def test_keyword_match_also_scores_correct(self):
        result = score_out_of_scope(
            "Carbon data is not available in this dataset.",
            keywords=["carbon", "not available"],
            min_matched=1,
        )
        assert result["correct"] is True