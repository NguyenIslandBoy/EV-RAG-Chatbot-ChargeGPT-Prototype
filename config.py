"""
config.py
=========
Central configuration — constants, paths, and environment variable loading.
All other modules import from here; nothing is hardcoded elsewhere.
"""

import logging
import os
from pathlib import Path

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Load .env
# ---------------------------------------------------------------------------

load_dotenv()

# ---------------------------------------------------------------------------
# ACN-Data API
# ---------------------------------------------------------------------------

ACN_TOKEN: str = os.getenv("token", "")
if not ACN_TOKEN:
    raise EnvironmentError(
        "ACN-Data API token not found. "
        "Add 'token=YOUR_TOKEN' to your .env file."
    )

ACN_BASE_URL: str = "https://ev.caltech.edu/api/v1"
ACN_SITES: list[str] = ["caltech", "jpl", "office001"]

# ---------------------------------------------------------------------------
# Ollama (local LLM)
# ---------------------------------------------------------------------------

OLLAMA_BASE_URL: str   = os.getenv("ollama_base_url",   "http://localhost:11434/v1")
# llama3.1:8b
# OLLAMA_MODEL: str      = os.getenv("ollama_model",      "llama3.1:8b" )
# phi3.5
OLLAMA_MODEL: str      = os.getenv("ollama_model",      "phi3.5" )
OLLAMA_EMBED_MODEL: str = os.getenv("ollama_embed_model", "nomic-embed-text:latest")

# ---------------------------------------------------------------------------
# HTTP behaviour
# ---------------------------------------------------------------------------

REQUEST_DELAY_SECONDS: float = 0.5
MAX_RETRIES: int = 3
RETRY_BACKOFF_SECONDS: int = 5

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
LOGS_DIR = BASE_DIR / "logs"
RAG_DIR  = BASE_DIR / "data" / "faiss"

DB_PATH         = DATA_DIR / "acn_data.duckdb"
CHECKPOINT_PATH = DATA_DIR / "acn_checkpoint.json"
RAG_DB_PATH     = RAG_DIR

# Ensure directories exist
DATA_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)
RAG_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s - %(message)s"

logging.basicConfig(
    level=logging.INFO,
    format=LOG_FORMAT,
    handlers=[
        logging.FileHandler(LOGS_DIR / "pipeline.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)