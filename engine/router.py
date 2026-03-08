"""
engine/router.py
================
Query router — classifies incoming questions to the correct subsystem.

Routes to:
  - "sql" : numerical/analytical questions → structured query engine (DuckDB)
  - "rag" : conceptual/background questions → RAG knowledge base
  - "both": questions needing data + context → both subsystems, answers merged

Design: LLM-based zero-shot classifier with a strict output format.
One fast low-token call before the main query. Falls back to "sql" on
any ambiguity since the structured engine is more reliable and grounded.
"""

import logging
import re

from openai import OpenAI

from config import OLLAMA_BASE_URL, OLLAMA_MODEL

log = logging.getLogger(__name__)

client = OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama")

# ---------------------------------------------------------------------------
# Route labels
# ---------------------------------------------------------------------------

ROUTE_SQL  = "sql"
ROUTE_RAG  = "rag"
ROUTE_BOTH = "both"

VALID_ROUTES = {ROUTE_SQL, ROUTE_RAG, ROUTE_BOTH}

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

ROUTER_SYSTEM_PROMPT = """You are a query classifier for an EV charging data assistant.

Classify the user's question into exactly one of three categories:

  sql  - Questions requiring numerical data, statistics, aggregations, counts,
         averages, patterns, comparisons, or trends FROM THE DATASET.
         Examples:
           "What are the peak charging hours?"
           "How many sessions were recorded at Caltech?"
           "What is the average kWh delivered per session?"
           "How does utilization vary by season?"
           "Which site has more sessions on weekends?"
           "When is the best time to charge to minimise waiting?"

  rag  - Questions about concepts, definitions, methodology, background,
         limitations, or the ACN-Data paper itself.
         Examples:
           "What is ACN-Data?"
           "What does utilization rate mean?"
           "What is the Adaptive Charging Network?"
           "What are the limitations of this dataset?"
           "Who collected this data and how?"
           "What is driver laxity?"

  both - Questions that explicitly ask for BOTH a number/statistic AND an explanation.
         Only use this when the question contains both "what is the number" AND "why".
         Examples:
           "What are peak charging hours and why does that pattern occur?"
           "What is the average session duration and what does that tell us?"
         NOT both:
           "What is ACN-Data and where was it collected?" → rag (no number needed)
           "What is the average kWh per session?" → sql (number only, no explanation)

Rules:
- Reply with ONLY the single word: sql, rag, or both
- No punctuation, no explanation, no other text
- "What is X" questions about concepts, datasets, or methodology are ALWAYS rag
- When uncertain, default to: sql"""

# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------

def classify_query(question: str) -> str:
    """
    Classify a question into sql, rag, or both.

    Returns one of: ROUTE_SQL, ROUTE_RAG, ROUTE_BOTH
    Falls back to ROUTE_SQL on any error or ambiguous response.
    """
    log.debug(f"Classifying query: {question!r}")

    try:
        response = client.chat.completions.create(
            model=OLLAMA_MODEL,
            messages=[
                {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
                {"role": "user",   "content": question},
            ],
            temperature=0,
            max_tokens=5,   # we only need one word
        )
        raw = response.choices[0].message.content or ""
        label = raw.strip().lower()

        # Strip any punctuation the model may have added
        label = re.sub(r"[^a-z]", "", label)

        if label not in VALID_ROUTES:
            log.warning(f"Router returned unexpected label {label!r} — defaulting to sql")
            return ROUTE_SQL

        log.info(f"Route: {label!r} for query: {question!r}")
        return label

    except Exception as e:
        log.error(f"Router failed: {e} — defaulting to sql")
        return ROUTE_SQL