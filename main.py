"""
main.py
=======
ChargeGPT — ACN-Data Ingestion Pipeline
========================================
Orchestrates the full ETL pipeline:

  Step 1 — Extract   : Pull paginated sessions from ACN-Data API
  Step 2 — Transform : Parse and normalise raw records
  Step 3 — Load      : Insert into DuckDB (idempotent)
  Step 4 — Features  : Build analytical views

Usage:
    python main.py                          # pull all sites
    python main.py --sites caltech jpl      # specific sites only
    python main.py --resume                 # resume interrupted pull
    python main.py --views-only             # rebuild views without re-pulling

Requirements:
    pip install requests duckdb pandas tqdm python-dotenv

.env file:
    token=YOUR_ACN_DATA_TOKEN
"""

import argparse
import logging

from tqdm import tqdm

import config  # noqa: F401 — runs logging setup and .env load on import
from config import ACN_SITES
from db import get_connection
from steps.step1_extract import (
    iter_site_pages,
    load_checkpoint,
    make_http_session,
)
from steps.step2_transform import parse_sessions_batch
from steps.step3_load import load_sessions
from steps.step4_features import build_feature_views

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Summary report
# ---------------------------------------------------------------------------

def print_summary(db_con) -> None:
    print("\n" + "=" * 55)
    print("INGESTION SUMMARY")
    print("=" * 55)

    total = db_con.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
    print(f"Total sessions in DB : {total:,}")

    by_site = db_con.execute(
        "SELECT siteID, COUNT(*) AS sessions "
        "FROM sessions GROUP BY siteID ORDER BY sessions DESC"
    ).fetchdf()
    print(f"\nSessions by site:\n{by_site.to_string(index=False)}")

    dates = db_con.execute(
        "SELECT MIN(connectionTime), MAX(connectionTime) FROM sessions"
    ).fetchone()
    print(f"\nDate range : {dates[0]}  →  {dates[1]}")

    kwh = db_con.execute(
        "SELECT ROUND(AVG(kWhDelivered), 2), "
        "       ROUND(MIN(kWhDelivered), 2), "
        "       ROUND(MAX(kWhDelivered), 2) "
        "FROM sessions WHERE kWhDelivered > 0"
    ).fetchone()
    print(f"kWh delivered  avg: {kwh[0]}  min: {kwh[1]}  max: {kwh[2]}")
    print("=" * 55 + "\n")


# ---------------------------------------------------------------------------
# Per-site ingestion
# ---------------------------------------------------------------------------

def run_site(site: str, http_session, db_con, checkpoint: dict, resume: bool) -> None:
    total_inserted = 0
    total_skipped = 0
    total_api = None

    with tqdm(desc=site, unit="sessions", dynamic_ncols=True) as pbar:
        for page_data, next_url, total_api in iter_site_pages(
            site, http_session, checkpoint, resume
        ):
            if pbar.total is None and total_api:
                pbar.total = total_api
                pbar.refresh()

            raw_items = page_data.get("_items", [])
            df = parse_sessions_batch(raw_items)
            inserted, skipped = load_sessions(df, db_con)

            total_inserted += inserted
            total_skipped += skipped
            pbar.update(len(raw_items))

    log.info(
        f"[{site}] Complete — "
        f"inserted: {total_inserted:,}  skipped (duplicate): {total_skipped:,}"
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="ChargeGPT — ingest ACN-Data EV charging sessions into DuckDB"
    )
    parser.add_argument(
        "--sites", nargs="+", default=ACN_SITES, choices=ACN_SITES,
        help="Sites to ingest (default: all)"
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from last saved checkpoint"
    )
    parser.add_argument(
        "--views-only", action="store_true",
        help="Skip ingestion — only rebuild analytical views"
    )
    args = parser.parse_args()

    db_con = get_connection()

    if not args.views_only:
        http_session = make_http_session()
        checkpoint = load_checkpoint() if args.resume else {}

        for site in args.sites:
            try:
                run_site(site, http_session, db_con, checkpoint, args.resume)
            except Exception as e:
                log.error(
                    f"[{site}] Pipeline failed: {e}. "
                    f"Checkpoint saved — rerun with --resume to continue."
                )
                raise

    build_feature_views(db_con)
    print_summary(db_con)
    db_con.close()
    log.info("Pipeline complete.")


if __name__ == "__main__":
    main()