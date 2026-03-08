"""
steps/step4_features.py
=======================
Feature engineering and analytical view creation.

Creates DuckDB VIEWs on top of the raw sessions table.
Views are cheap — they're computed at query time, so no storage overhead.

Views created:
  - sessions_features   : cleaned sessions + temporal + derived metrics
  - hourly_utilization  : peak hour profiles by site and day type
  - seasonal_summary    : season × year aggregations
  - daily_evse_utilization : per-charger daily throughput
"""

import logging

import duckdb

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# View definitions
# ---------------------------------------------------------------------------

SESSIONS_FEATURES_VIEW = """
CREATE OR REPLACE VIEW sessions_features AS
SELECT
    *,
    -- Temporal breakdowns
    EXTRACT(hour  FROM connectionTime)  AS hour_of_day,
    EXTRACT(dow   FROM connectionTime)  AS day_of_week,   -- 0 = Sunday, 6 = Saturday
    EXTRACT(month FROM connectionTime)  AS month,
    EXTRACT(year  FROM connectionTime)  AS year,

    -- Season (Northern Hemisphere)
    CASE
        WHEN EXTRACT(month FROM connectionTime) IN (12, 1, 2) THEN 'winter'
        WHEN EXTRACT(month FROM connectionTime) IN (3, 4, 5)  THEN 'spring'
        WHEN EXTRACT(month FROM connectionTime) IN (6, 7, 8)  THEN 'summer'
        ELSE 'autumn'
    END AS season,

    -- Weekday / weekend
    CASE
        WHEN EXTRACT(dow FROM connectionTime) IN (0, 6) THEN 'weekend'
        ELSE 'weekday'
    END AS day_type,

    -- Session duration in hours (connection → disconnection)
    EPOCH(disconnectTime - connectionTime) / 3600.0 AS duration_hours,

    -- Idle time: connected but not actively charging
    CASE
        WHEN doneChargingTime IS NOT NULL
        THEN EPOCH(disconnectTime - doneChargingTime) / 3600.0
    END AS idle_hours,

    -- Average charging rate (kW)
    CASE
        WHEN EPOCH(disconnectTime - connectionTime) > 0
        THEN kWhDelivered / (EPOCH(disconnectTime - connectionTime) / 3600.0)
    END AS avg_charge_rate_kw

FROM sessions
WHERE
    -- Data quality filters
    kWhDelivered > 0              -- exclude zero-energy sessions
    AND kWhDelivered < 200        -- exclude implausible values
    AND EPOCH(disconnectTime - connectionTime) / 3600.0 > (1.0 / 60)  -- min 1 minute
    AND EPOCH(disconnectTime - connectionTime) / 3600.0 < 24           -- max 24 hours
    AND connectionTime  IS NOT NULL
    AND disconnectTime  IS NOT NULL;
"""

HOURLY_UTILIZATION_VIEW = """
CREATE OR REPLACE VIEW hourly_utilization AS
SELECT
    siteID,
    day_type,
    hour_of_day,
    COUNT(*)                    AS session_count,
    AVG(kWhDelivered)           AS avg_kwh,
    AVG(duration_hours)         AS avg_duration_hours,
    AVG(avg_charge_rate_kw)     AS avg_charge_rate_kw
FROM sessions_features
GROUP BY siteID, day_type, hour_of_day
ORDER BY siteID, day_type, hour_of_day;
"""

SEASONAL_SUMMARY_VIEW = """
CREATE OR REPLACE VIEW seasonal_summary AS
SELECT
    siteID,
    season,
    year,
    COUNT(*)                    AS session_count,
    SUM(kWhDelivered)           AS total_kwh,
    AVG(kWhDelivered)           AS avg_kwh_per_session,
    AVG(duration_hours)         AS avg_duration_hours,
    COUNT(DISTINCT userID)      AS unique_users
FROM sessions_features
GROUP BY siteID, season, year
ORDER BY siteID, year, season;
"""

DAILY_EVSE_UTILIZATION_VIEW = """
CREATE OR REPLACE VIEW daily_evse_utilization AS
SELECT
    siteID,
    stationID,
    connectionTime::DATE        AS date,
    COUNT(*)                    AS sessions_per_day,
    SUM(kWhDelivered)           AS kwh_per_day,
    SUM(duration_hours)         AS occupied_hours
FROM sessions_features
GROUP BY siteID, stationID, date
ORDER BY siteID, stationID, date;
"""


# ---------------------------------------------------------------------------
# Build all views
# ---------------------------------------------------------------------------

def build_feature_views(db_con: duckdb.DuckDBPyConnection) -> None:
    """Create or replace all analytical views. Safe to re-run at any time."""
    views = {
        "sessions_features":      SESSIONS_FEATURES_VIEW,
        "hourly_utilization":     HOURLY_UTILIZATION_VIEW,
        "seasonal_summary":       SEASONAL_SUMMARY_VIEW,
        "daily_evse_utilization": DAILY_EVSE_UTILIZATION_VIEW,
    }

    log.info("Building feature views...")
    for name, sql in views.items():
        db_con.execute(sql)
        log.info(f"  [OK] {name}")

    log.info("All views ready.")