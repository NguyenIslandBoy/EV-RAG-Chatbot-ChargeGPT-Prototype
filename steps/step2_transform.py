"""
steps/step2_transform.py
========================
Parsing and normalisation of raw ACN-Data API session records.

Responsibilities:
  - RFC 1123 datetime parsing
  - userInputs flattening (pick most recent entry)
  - Raw dict → flat row dict ready for DB insert
  - DataFrame-level cleaning and validation
"""

import logging
from datetime import datetime

import pandas as pd

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Site ID normalisation
# ---------------------------------------------------------------------------

# NOTES: After using normal raw.get() in parse_session, it got 2 values: "0002" and "2" -> have to normalise "0002" and "2" into "0002" (caltech)
SITE_ID_NORMALISATION = {
    "1":  "0001",   # JPL
    "2":  "0002",   # Caltech (old format, Oct 2018 only)
    "19": "0019",   # Office 1
}

def normalise_site_id(raw_id: str | None) -> str | None:
    if not raw_id:
        return raw_id
    return SITE_ID_NORMALISATION.get(raw_id, raw_id)

# ---------------------------------------------------------------------------
# Datetime parsing
# ---------------------------------------------------------------------------

def parse_dt(value) -> datetime | None:
    """
    Parse ACN-Data RFC 1123 datetime string to Python datetime.
    Returns None on missing or malformed values.
    Example input: "Wed, 01 May 2019 12:00:00 GMT"
    All timestamps are UTC per ACN-Data documentation.
    """
    if not value:
        return None
    try:
        return datetime.strptime(value, "%a, %d %b %Y %H:%M:%S %Z")
    except (ValueError, TypeError):
        log.debug(f"Could not parse datetime: {value!r}")
        return None


# ---------------------------------------------------------------------------
# userInputs flattening
# ---------------------------------------------------------------------------

def extract_user_inputs(user_inputs: list) -> dict:
    """
    Flatten the userInputs list into a single dict.
    ACN-Data may have multiple entries if the user updated their inputs mid-session.
    We take the most recently modified entry.
    """
    empty = {
        "kWhRequested":       None,
        "milesRequested":     None,
        "minutesAvailable":   None,
        "requestedDeparture": None,
        "WhPerMile":          None,
        "paymentRequired":    None,
    }

    if not user_inputs:
        return empty

    try:
        latest = sorted(
            user_inputs,
            key=lambda x: parse_dt(x.get("modifiedAt")) or datetime.min,
            reverse=True,
        )[0]
    except Exception:
        latest = user_inputs[-1]

    return {
        "kWhRequested":       latest.get("kWhRequested"),
        "milesRequested":     latest.get("milesRequested"),
        "minutesAvailable":   latest.get("minutesAvailable"),
        "requestedDeparture": parse_dt(latest.get("requestedDeparture")),
        "WhPerMile":          latest.get("WhPerMile"),
        "paymentRequired":    latest.get("paymentRequired"),
    }


# ---------------------------------------------------------------------------
# Single record parser
# ---------------------------------------------------------------------------

def parse_session(raw: dict) -> dict:
    """Convert one raw API session dict into a flat row dict."""
    ui = extract_user_inputs(raw.get("userInputs") or [])
    return {
        "_id":               raw.get("_id"),
        "sessionID":         raw.get("sessionID"),
        "siteID":            normalise_site_id(raw.get("siteID")), # use normalise_site_id() instead
        "stationID":         raw.get("stationID"),
        "spaceID":           raw.get("spaceID"),
        "clusterID":         raw.get("clusterID"),
        "userID":            raw.get("userID"),
        "connectionTime":    parse_dt(raw.get("connectionTime")),
        "disconnectTime":    parse_dt(raw.get("disconnectTime")),
        "doneChargingTime":  parse_dt(raw.get("doneChargingTime")),
        "kWhDelivered":      raw.get("kWhDelivered"),
        "timezone":          raw.get("timezone"),
        **ui,
    }


# ---------------------------------------------------------------------------
# Batch parse + validation
# ---------------------------------------------------------------------------

def parse_sessions_batch(raw_items: list[dict]) -> pd.DataFrame:
    """
    Parse a list of raw session dicts into a validated DataFrame.
    Logs a warning if any records are dropped during validation.
    """
    rows = [parse_session(item) for item in raw_items]
    df = pd.DataFrame(rows)

    before = len(df)

    # Drop records with no primary key (shouldn't happen but defensive)
    df = df.dropna(subset=["_id"])

    dropped = before - len(df)
    if dropped > 0:
        log.warning(f"Dropped {dropped} records missing _id during parsing")

    return df