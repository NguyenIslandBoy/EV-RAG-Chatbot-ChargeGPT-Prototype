"""
steps/step1_extract.py
======================
ACN-Data API client.

Responsibilities:
  - HTTP session setup (token-based Basic Auth)
  - Paginated session fetching with retry logic
  - Checkpoint save/load for resumable pulls
"""

import json
import logging
import time
from pathlib import Path

import requests

from config import (
    ACN_BASE_URL,
    ACN_TOKEN,
    CHECKPOINT_PATH,
    MAX_RETRIES,
    REQUEST_DELAY_SECONDS,
    RETRY_BACKOFF_SECONDS,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# HTTP session
# ---------------------------------------------------------------------------

def make_http_session(token: str = ACN_TOKEN) -> requests.Session:
    """
    Build an authenticated requests.Session.
    ACN-Data uses HTTP Basic Auth: token as username, blank password.
    """
    session = requests.Session()
    session.auth = (token, "")
    session.headers.update({"Accept": "application/json"})
    return session


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def load_checkpoint() -> dict:
    """Load saved pagination state. Returns empty dict if no checkpoint exists."""
    if Path(CHECKPOINT_PATH).exists():
        with open(CHECKPOINT_PATH) as f:
            return json.load(f)
    return {}


def save_checkpoint(checkpoint: dict) -> None:
    with open(CHECKPOINT_PATH, "w") as f:
        json.dump(checkpoint, f, indent=2)


# ---------------------------------------------------------------------------
# Single-page fetch with retry
# ---------------------------------------------------------------------------

def fetch_page(http_session: requests.Session, url: str) -> dict:
    """
    Fetch one API page. Retries on transient errors with exponential-ish backoff.
    Raises immediately on auth failure (401) — no point retrying bad credentials.
    """
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = http_session.get(url, timeout=30)
            resp.raise_for_status()
            return resp.json()

        except requests.exceptions.HTTPError as e:
            status = resp.status_code

            if status == 401:
                raise RuntimeError(
                    "Authentication failed. "
                    "Ensure 'token=YOUR_TOKEN' is set in .env. "
                    "The token is used as the HTTP Basic Auth username."
                ) from e

            if status == 429:
                wait = RETRY_BACKOFF_SECONDS * attempt
                log.warning(f"Rate limited (429). Waiting {wait}s — attempt {attempt}/{MAX_RETRIES}")
                time.sleep(wait)
                continue

            log.error(f"HTTP {status} on attempt {attempt}/{MAX_RETRIES}: {e}")
            if attempt == MAX_RETRIES:
                raise
            time.sleep(RETRY_BACKOFF_SECONDS)

        except requests.exceptions.RequestException as e:
            log.error(f"Request error on attempt {attempt}/{MAX_RETRIES}: {e}")
            if attempt == MAX_RETRIES:
                raise
            time.sleep(RETRY_BACKOFF_SECONDS * attempt)

    return {}


# ---------------------------------------------------------------------------
# Full site generator — yields pages one at a time
# ---------------------------------------------------------------------------

def iter_site_pages(
    site: str,
    http_session: requests.Session,
    checkpoint: dict,
    resume: bool = False,
):
    """
    Generator that yields (page_data, next_url | None) for each page of a site.
    Handles pagination by following _links.next until exhausted.
    Updates checkpoint dict in-place after each page.
    """
    start_url = (
        checkpoint.get(site)
        if resume and site in checkpoint
        else f"{ACN_BASE_URL}/sessions/{site}?sort=-connectionTime"
    )

    log.info(
        f"[{site}] {'Resuming from checkpoint' if resume and site in checkpoint else 'Starting fresh pull'}"
    )

    current_data = fetch_page(http_session, start_url)
    total = current_data.get("_meta", {}).get("total", 0)
    log.info(f"[{site}] Total sessions reported by API: {total:,}")

    while True:
        items = current_data.get("_items", [])
        if not items:
            break

        # Resolve next URL before yielding (so caller can checkpoint after insert)
        next_link = current_data.get("_links", {}).get("next", {})
        next_href = next_link.get("href") if isinstance(next_link, dict) else None
        next_url = f"{ACN_BASE_URL}/{next_href}" if next_href else None

        yield current_data, next_url, total

        if next_url:
            checkpoint[site] = next_url
            save_checkpoint(checkpoint)
            time.sleep(REQUEST_DELAY_SECONDS)
            current_data = fetch_page(http_session, next_url)
        else:
            # Site complete — remove from checkpoint
            checkpoint.pop(site, None)
            save_checkpoint(checkpoint)
            break