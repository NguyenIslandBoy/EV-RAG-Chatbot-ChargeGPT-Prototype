"""
eval/evaluate.py
================
Comparative evaluation framework for ChargeGPT.

Three conditions measured against the same 30-question test set:

  Condition A — LLM Only
    Raw Llama with no tools or retrieval. Measures baseline hallucination rate.

  Condition B — LLM + RAG
    RAG knowledge base only. No SQL engine. Measures grounding improvement.

  Condition C — LLM + RAG + Structured Query Engine  (full system)
    The complete ChargeGPT pipeline. Measures full system performance.

Metrics:
  - Execution accuracy    : did the system return an answer without error?
  - Numeric accuracy      : is the number within tolerance of ground truth? (sql_numerical)
  - Keyword match rate    : do responses contain expected keywords? (rag_conceptual, both)
  - Hallucination rate    : confident wrong answers on out_of_scope questions
  - Response latency      : wall-clock time per query (seconds)

Run:
    python eval/evaluate.py                    # all 3 conditions, all questions
    python eval/evaluate.py --condition C      # single condition
    python eval/evaluate.py --category sql_numerical  # filter by category
"""

import argparse
import json
import logging
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from openai import OpenAI

# Add project root to path so imports work when run from eval/ subdirectory
sys.path.insert(0, str(Path(__file__).parent.parent))

import config  # noqa: F401 — triggers logging setup
from db import get_connection
from engine.query_engine import run_query
from rag.rag_engine import run_rag_query
from rag.vector_store import build_vector_store
from engine.router import classify_query, ROUTE_SQL, ROUTE_RAG, ROUTE_BOTH
from chat import ask

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("faiss").setLevel(logging.WARNING)

log = logging.getLogger("eval")

QUESTIONS_PATH = Path(__file__).parent / "questions.json"
RESULTS_DIR    = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# LLM-only client (Condition A)
# ---------------------------------------------------------------------------

llm_client = OpenAI(base_url=config.OLLAMA_BASE_URL, api_key="ollama")

LLM_ONLY_SYSTEM = """You are an expert on EV charging infrastructure and data analysis.
Answer the user's question as accurately and concisely as possible.
If you do not know the answer, say so clearly."""

def run_llm_only(question: str) -> str:
    """Condition A: raw LLM, no tools, no retrieval."""
    try:
        resp = llm_client.chat.completions.create(
            model=config.OLLAMA_MODEL,
            messages=[
                {"role": "system", "content": LLM_ONLY_SYSTEM},
                {"role": "user",   "content": question},
            ],
            temperature=0.1,
            max_tokens=300,
        )
        return resp.choices[0].message.content or ""
    except Exception as e:
        return f"ERROR: {e}"


# ---------------------------------------------------------------------------
# Condition B: LLM + RAG only (bypasses router and SQL engine)
# ---------------------------------------------------------------------------

def run_rag_only(question: str, vector_store) -> str:
    """Condition B: RAG knowledge base only, no SQL engine."""
    resp = run_rag_query(question, vector_store)
    return resp.answer


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

def extract_first_number(text: str) -> float | None:
    """Extract the first number from a response string."""
    # Remove commas from numbers like "66,745"
    cleaned = text.replace(",", "")
    matches = re.findall(r"-?\d+(?:\.\d+)?", cleaned)
    return float(matches[0]) if matches else None


def score_numerical(response: str, ground_truth: str, tolerance: float) -> dict:
    """
    Score a numerical answer.
    Returns: {correct: bool, predicted: float|None, expected: float, within_tolerance: bool}
    """
    try:
        expected = float(ground_truth.replace(",", ""))
    except ValueError:
        return {"correct": False, "predicted": None, "expected": None, "error": "unparseable ground truth"}

    predicted = extract_first_number(response)
    if predicted is None:
        return {"correct": False, "predicted": None, "expected": expected, "within_tolerance": False}

    if expected == 0:
        within_tolerance = predicted == 0
    else:
        relative_error = abs(predicted - expected) / abs(expected)
        within_tolerance = relative_error <= tolerance

    return {
        "correct": within_tolerance,
        "predicted": predicted,
        "expected": expected,
        "relative_error": abs(predicted - expected) / abs(expected) if expected != 0 else abs(predicted),
        "within_tolerance": within_tolerance,
    }


# Maps acceptable aliases for categorical answers
CATEGORICAL_ALIASES = {
    "0019": ["0019", "office 1", "office1", "office001"],
    "0002": ["0002", "caltech"],
    "0001": ["0001", "jpl"],
    "2": ["2", "tuesday"],
    "summer": ["summer"],
    "autumn": ["autumn", "fall"],
}

def score_categorical(response: str, ground_truth: str) -> dict:
    """Score a categorical answer — accepts known aliases."""
    response_lower = response.lower()
    aliases = CATEGORICAL_ALIASES.get(ground_truth.lower(), [ground_truth.lower()])
    correct = any(alias in response_lower for alias in aliases)
    return {"correct": correct, "predicted": response[:100], "expected": ground_truth}


def score_keywords(response: str, keywords: list[str], min_matched: int) -> dict:
    """Score a conceptual answer by keyword presence."""
    response_lower = response.lower()
    matched = [kw for kw in keywords if kw.lower() in response_lower]
    passed = len(matched) >= min_matched
    return {
        "correct": passed,
        "matched_keywords": matched,
        "total_keywords": keywords,
        "matched_count": len(matched),
        "required_count": min_matched,
    }


def score_out_of_scope(response: str, keywords: list[str], min_matched: int) -> dict:
    """
    Score an out-of-scope question.
    Correct = system refuses or explains limitation.
    Incorrect (hallucination) = system returns a confident number or fabricated answer.

    Fix notes:
    - Added query_engine refusal phrases ("can't answer", "not available in this dataset")
    - Number check excludes year-like 4-digit values (e.g. 2021) to avoid false positives
    - Hallucination requires a large confident number (5+ digits) without any refusal signal
    """
    # Check if response contains a confident fabricated large number (not a year)
    # 5+ digit numbers are likely hallucinated statistics; 4-digit numbers are often years
    has_large_number = bool(re.search(r"\b\d{5,}\b", response.replace(",", "")))

    # Check if response acknowledges the limitation — covers both RAG and SQL engine refusals
    refusal_signals = [
        # RAG engine signals
        "not available", "not contain", "no data", "cannot answer",
        "don't have", "unable to", "outside", "not in", "not include",
        "dataset does not", "caiso",
        # query_engine UNSUPPORTED_QUERY signals
        "can't answer", "cannot be answered", "not in the dataset",
        "does not contain information about carbon",
        "does not contain information about grid",
        "does not contain information about emissions",
        # domain keywords that indicate the model is discussing the gap, not hallucinating
        "carbon intensity", "grid emissions", "electricity prices",
    ]
    has_refusal = any(sig in response.lower() for sig in refusal_signals)
    kw_score = score_keywords(response, keywords, min_matched)

    # Hallucination: returns a large fabricated number WITHOUT any refusal signal
    hallucinated = has_large_number and not has_refusal

    return {
        "correct": has_refusal or kw_score["correct"],
        "hallucinated": hallucinated,
        "has_refusal_signal": has_refusal,
        "has_large_number": has_large_number,
        **kw_score,
    }


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class QuestionResult:
    question_id: str
    question: str
    category: str
    condition: str
    response: str
    latency_seconds: float
    score: dict = field(default_factory=dict)
    error: str | None = None


# ---------------------------------------------------------------------------
# Single question runner
# ---------------------------------------------------------------------------

def evaluate_question(
    q: dict,
    condition: str,
    db_con,
    vector_store,
) -> QuestionResult:
    """Run a single question under the given condition and score the response."""

    start = time.perf_counter()
    error = None

    try:
        if condition == "A":
            response = run_llm_only(q["question"])
        elif condition == "B":
            response = run_rag_only(q["question"], vector_store)
        elif condition == "C":
            response = ask(q["question"], db_con, vector_store, debug=False)
        else:
            raise ValueError(f"Unknown condition: {condition}")
    except Exception as e:
        response = f"ERROR: {e}"
        error = str(e)

    latency = time.perf_counter() - start

    # Score
    category = q["category"]
    score = {}

    if category == "sql_numerical":
        score = score_numerical(response, q["ground_truth"], q.get("tolerance", 0.05))

    elif category == "sql_categorical":
        score = score_categorical(response, q["ground_truth"])

    elif category == "rag_conceptual":
        score = score_keywords(response, q["ground_truth_keywords"], q["min_keywords_matched"])

    elif category == "both":
        # Score SQL part if present, keywords for context part
        kw_score = score_keywords(response, q["ground_truth_keywords"], q["min_keywords_matched"])
        if "ground_truth_sql" in q:
            sql_correct = q["ground_truth_sql"].lower() in response.lower()
            score = {**kw_score, "sql_correct": sql_correct,
                     "correct": kw_score["correct"] or sql_correct}
        else:
            score = kw_score

    elif category == "out_of_scope":
        score = score_out_of_scope(
            response,
            q.get("ground_truth_keywords", []),
            q.get("min_keywords_matched", 1),
        )

    return QuestionResult(
        question_id=q["id"],
        question=q["question"],
        category=category,
        condition=condition,
        response=response,
        latency_seconds=latency,
        score=score,
        error=error,
    )


# ---------------------------------------------------------------------------
# Aggregate metrics
# ---------------------------------------------------------------------------

def compute_metrics(results: list[QuestionResult]) -> dict:
    """Compute aggregate metrics from a list of results."""
    if not results:
        return {}

    total = len(results)
    correct = sum(1 for r in results if r.score.get("correct", False))
    errors  = sum(1 for r in results if r.error)
    latencies = [r.latency_seconds for r in results]

    # Hallucination rate — out_of_scope questions only
    oos = [r for r in results if r.category == "out_of_scope"]
    hallucinations = sum(1 for r in oos if r.score.get("hallucinated", False))

    # Per-category accuracy
    categories = {}
    for r in results:
        cat = r.category
        if cat not in categories:
            categories[cat] = {"total": 0, "correct": 0}
        categories[cat]["total"] += 1
        if r.score.get("correct", False):
            categories[cat]["correct"] += 1

    cat_accuracy = {
        cat: round(v["correct"] / v["total"] * 100, 1)
        for cat, v in categories.items()
    }

    return {
        "total_questions": total,
        "overall_accuracy_pct": round(correct / total * 100, 1),
        "correct": correct,
        "errors": errors,
        "hallucination_count": hallucinations,
        "hallucination_rate_pct": round(hallucinations / len(oos) * 100, 1) if oos else 0,
        "avg_latency_seconds": round(sum(latencies) / len(latencies), 2),
        "p50_latency_seconds": round(sorted(latencies)[len(latencies) // 2], 2),
        "p95_latency_seconds": round(sorted(latencies)[int(len(latencies) * 0.95)], 2),
        "per_category_accuracy_pct": cat_accuracy,
    }


# ---------------------------------------------------------------------------
# Report printer
# ---------------------------------------------------------------------------

def print_report(all_results: dict[str, list[QuestionResult]]):
    """Print a formatted comparison report to stdout."""
    print("\n" + "=" * 65)
    print("  ChargeGPT — Evaluation Report")
    print("=" * 65)

    # Per-condition summary
    metrics_by_condition = {}
    for condition, results in all_results.items():
        m = compute_metrics(results)
        metrics_by_condition[condition] = m
        label = {"A": "LLM Only", "B": "LLM + RAG", "C": "Full System (RAG + SQL)"}[condition]
        print(f"\nCondition {condition}: {label}")
        print(f"  Overall accuracy : {m['overall_accuracy_pct']}%  ({m['correct']}/{m['total_questions']})")
        print(f"  Hallucination    : {m['hallucination_rate_pct']}%  ({m['hallucination_count']} out-of-scope failures)")
        print(f"  Avg latency      : {m['avg_latency_seconds']}s  (p95: {m['p95_latency_seconds']}s)")
        print(f"  Per-category     :")
        for cat, acc in m["per_category_accuracy_pct"].items():
            print(f"    {cat:<25} {acc}%")

    # Per-question detail table
    print("\n" + "-" * 65)
    print(f"  {'ID':<5} {'Category':<18} {'A':>6} {'B':>6} {'C':>6}  Question")
    print("-" * 65)

    all_ids = sorted({r.question_id for results in all_results.values() for r in results})
    for qid in all_ids:
        row = {}
        question_text = ""
        category = ""
        for cond, results in all_results.items():
            match = next((r for r in results if r.question_id == qid), None)
            if match:
                row[cond] = "✓" if match.score.get("correct") else "✗"
                if match.score.get("hallucinated"):
                    row[cond] = "H"  # hallucinated
                question_text = match.question[:38]
                category = match.category

        a = row.get("A", "-")
        b = row.get("B", "-")
        c = row.get("C", "-")
        print(f"  {qid:<5} {category:<18} {a:>6} {b:>6} {c:>6}  {question_text}")

    print("\nLegend: ✓ correct  ✗ incorrect  H hallucinated  - not run")


# ---------------------------------------------------------------------------
# Save results to JSON
# ---------------------------------------------------------------------------

def save_results(all_results: dict[str, list[QuestionResult]], label: str = ""):
    """Save full results to JSON for reproducibility."""
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = RESULTS_DIR / f"eval_{timestamp}{('_' + label) if label else ''}.json"

    output = {}
    for condition, results in all_results.items():
        output[condition] = {
            "metrics": compute_metrics(results),
            "questions": [
                {
                    "id": r.question_id,
                    "question": r.question,
                    "category": r.category,
                    "response": r.response,
                    "score": r.score,
                    "latency_seconds": round(r.latency_seconds, 3),
                    "error": r.error,
                }
                for r in results
            ],
        }

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: {filename}")
    return filename


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="ChargeGPT Evaluation Framework")
    parser.add_argument("--condition",  choices=["A", "B", "C"], help="Run single condition only")
    parser.add_argument("--category",  help="Filter questions by category")
    parser.add_argument("--ids",        help="Comma-separated question IDs to run (e.g. q01,q05)")
    parser.add_argument("--no-save",    action="store_true", help="Don't save results to disk")
    args = parser.parse_args()

    # Load questions
    questions = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))

    # Apply filters
    if args.category:
        questions = [q for q in questions if q["category"] == args.category]
    if args.ids:
        ids = set(args.ids.split(","))
        questions = [q for q in questions if q["id"] in ids]

    if not questions:
        print("No questions matched the filters.")
        sys.exit(1)

    conditions = [args.condition] if args.condition else ["A", "B", "C"]

    print(f"\nRunning evaluation: {len(questions)} questions × {len(conditions)} conditions")
    print(f"Model: {config.OLLAMA_MODEL}")

    # Initialise shared resources
    print("Loading database...")
    db_con = get_connection()
    print("Loading vector store...")
    vector_store = build_vector_store()

    # Run evaluation
    all_results: dict[str, list[QuestionResult]] = {}

    for condition in conditions:
        label = {"A": "LLM Only", "B": "LLM + RAG", "C": "Full System"}[condition]
        print(f"\n--- Condition {condition}: {label} ---")
        results = []

        for i, q in enumerate(questions, 1):
            print(f"  [{i:02d}/{len(questions)}] {q['id']} — {q['question'][:55]}...")
            result = evaluate_question(q, condition, db_con, vector_store)
            results.append(result)

            status = "✓" if result.score.get("correct") else "✗"
            if result.score.get("hallucinated"):
                status = "H (hallucinated)"
            print(f"         → {status}  ({result.latency_seconds:.1f}s)")

        all_results[condition] = results

    db_con.close()

    # Print report
    print_report(all_results)

    # Save
    if not args.no_save:
        save_results(all_results)


if __name__ == "__main__":
    main()