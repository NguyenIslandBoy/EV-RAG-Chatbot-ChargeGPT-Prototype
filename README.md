# ⚡ ChargeGPT — EV Charging Analytics Assistant

A conversational AI assistant for querying and understanding EV charging infrastructure data. A personal portfolio project demonstrating production-quality data engineering and LLM integration.

ChargeGPT combines a **Text-to-SQL engine**, a **RAG knowledge base**, and an **LLM query router** to answer both quantitative and conceptual questions about real-world EV charging sessions — grounded entirely in data, with no hallucination on out-of-scope queries.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-red)
![DuckDB](https://img.shields.io/badge/DuckDB-analytical--DB-yellow)
![FAISS](https://img.shields.io/badge/FAISS-vector--store-orange)
![Ollama](https://img.shields.io/badge/Ollama-local--LLM-green)

---

## What it does

Ask ChargeGPT a question in plain English. The router classifies it and sends it to the right subsystem:

| Question type | Route | Example |
|---|---|---|
| Quantitative / data | **SQL** | *"How many sessions occurred in summer?"* |
| Conceptual / background | **RAG** | *"What is driver laxity in EV charging?"* |
| Mixed | **BOTH** | *"What are the peak hours and why?"* |
| Out of scope | **Refuse** | *"What is the carbon intensity of EV charging?"* |

The Streamlit UI shows the route badge, generated SQL, and response latency for every query — making the grounding mechanism fully transparent.

---

## Architecture

```
User question
      │
      ▼
┌─────────────┐
│   Router    │  LLM classifier → sql / rag / both
└──────┬──────┘
       │
  ┌────┴─────────────────────────┐
  │                              │
  ▼                              ▼
┌──────────────────┐    ┌────────────────────┐
│  SQL Engine      │    │   RAG Engine       │
│                  │    │                    │
│  NL → SQL (LLM)  │    │  FAISS retrieval   │
│  DuckDB execute  │    │  nomic-embed-text  │
│  66,446 sessions │    │  15 paper chunks   │
└────────┬─────────┘    └────────┬───────────┘
         │                       │
         └──────────┬────────────┘
                    ▼
             Merged answer
                    │
                    ▼
           Streamlit web UI
```

**Stack:** Python · DuckDB · FAISS · Ollama (phi3.5 / llama3.1:8b) · nomic-embed-text · Streamlit

---

## Dataset

**ACN-Data** — a publicly available dataset of real-world workplace EV charging sessions collected by Caltech's Adaptive Charging Network.

- **66,446 sessions** after quality filtering
- **3 sites:** Caltech (0002), JPL (0001), Office 1 (0019)
- **115 unique charging stations**
- **787,322 kWh** total energy delivered
- Date range: 2018–2021

> Data is not included in this repository. You must obtain your own API token from [ev.caltech.edu](https://ev.caltech.edu/) and run the ETL pipeline to build the local database.

---

## Evaluation results

Three-condition comparative evaluation across 30 questions:

| Condition | Description | Accuracy | Hallucination rate | Avg latency |
|---|---|---|---|---|
| **A** | LLM only (no tools) | 13.3% | 0% | 12s |
| **B** | LLM + RAG only | 33.3% | 0% | 29s |
| **C** | Full system (RAG + SQL) | **90.0%** | **0%** | 114s |

Per-category breakdown for Condition C:

| Category | Accuracy |
|---|---|
| sql_numerical | 82.4% |
| sql_categorical | 100% |
| rag_conceptual | 71.4% |
| both (hybrid) | 100% |
| out_of_scope | 100% |

Key finding: the hybrid `both` category achieves 100% — questions requiring data *and* contextual explanation are exactly the use case that justifies the architecture. The full system's 0% hallucination rate on out-of-scope queries (carbon intensity, grid emissions) demonstrates effective scope enforcement.

---

## Project structure

```
EV_ChargeGPT_Project/
├── streamlit_app.py          # Streamlit web UI
├── chat.py                   # Core ask() function & CLI interface
├── main.py                   # ETL pipeline entrypoint
├── config.py                 # Settings & environment variables
├── db.py                     # DuckDB connection helpers
├── requirements.txt
├── .env.example              # Environment variable template
│
├── engine/
│   ├── router.py             # LLM query classifier (sql/rag/both)
│   ├── query_engine.py       # NL → SQL → DuckDB → answer
│   ├── sql_generator.py      # Text-to-SQL with phi3.5
│   ├── sql_executor.py       # DuckDB execution & formatting
│   └── schema_context.py     # Schema description injected into prompts
│
├── rag/
│   ├── documents.py          # 15 chunked passages from ACN paper
│   ├── vector_store.py       # FAISS index (nomic-embed-text embeddings)
│   └── rag_engine.py         # Retrieval + LLM synthesis
│
├── steps/
│   ├── step1_extract.py      # ACN API pull with pagination
│   ├── step2_transform.py    # Parse & normalise sessions
│   ├── step3_load.py         # Idempotent DuckDB insert
│   └── step4_features.py     # Analytical views (hourly, seasonal, EVSE)
│
├── eval/
│   ├── evaluate.py           # 3-condition evaluation framework
│   └── questions.json        # 30 questions with verified ground truth
│
└── .streamlit/
    └── config.toml           # Streamlit theme settings
```

---

## Setup

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.ai/) installed and running locally
- An ACN-Data API token from [ev.caltech.edu](https://ev.caltech.edu/)

### 1. Clone and install

```bash
git clone https://github.com/NguyenIslandBoy/EV-RAG-Chatbot-ChargeGPT-Prototype.git
cd EV-RAG-Chatbot-ChargeGPT-Prototype

pip install -r requirements.txt
```

### 2. Pull required Ollama models

```bash
ollama pull phi3.5
ollama pull nomic-embed-text
```

### 3. Configure environment
Add .env and add your API token and LLM model information

---

## Running

### Option A — Full pipeline (build database from scratch, then launch UI)

Run this once to fetch data from the ACN API and build the local DuckDB:

```bash
# Start Ollama in a separate terminal
ollama serve

# Build the database (~10–20 min depending on connection)
python main.py

# Launch the web UI
streamlit run streamlit_app.py
```

### Option B — UI only (if you already have the database)

```bash
ollama serve          # separate terminal
streamlit run streamlit_app.py
```

### Option C — CLI chat

```bash
python chat.py                          # interactive mode
python chat.py --query "How many sessions at JPL?"   # single query
python chat.py --debug                  # show generated SQL
```

---

## Evaluation

Run the full 3-condition evaluation (~3 hours on a local machine with phi3.5):

```bash
python eval/evaluate.py
```

Run a single condition or subset:

```bash
python eval/evaluate.py --condition C
python eval/evaluate.py --condition C --ids q01,q05,q16 --no-save
```

Results are saved to `eval/results/eval_TIMESTAMP.json`.

---

## Limitations

- **Local model latency** — phi3.5 takes ~2 minutes per query on CPU. Replacing with an API model (e.g. GPT-4o-mini at ~$0.001/query) would reduce latency to ~3s and improve SQL generation accuracy.
- **RAG corpus size** — 15 chunks from a single paper. Adding more ACN-related literature would improve conceptual question coverage.
- **US dataset only** — ACN-Data is from Caltech and JPL in California. Generalisability to UK or European charging behaviour is limited.
- **Workplace charging bias** — the dataset captures workplace sessions only; residential and public charging patterns are not represented.

---

## Tech stack

| Component | Technology |
|---|---|
| Language model | phi3.5 (3.8B) via Ollama |
| Embeddings | nomic-embed-text via Ollama |
| Vector store | FAISS (faiss-cpu) |
| Analytical database | DuckDB |
| Data source | ACN-Data REST API |
| Web UI | Streamlit |
| Data processing | Python, Pandas |

---

## References

- Lee, Z. J., Li, T., & Low, S. H. (2019). *ACN-Data: Analysis and Applications of an Open EV Charging Dataset*. ACM e-Energy.
- ACN-Data API: [ev.caltech.edu](https://ev.caltech.edu/)
