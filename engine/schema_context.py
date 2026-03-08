"""
engine/schema_context.py
========================
Provides the schema context string injected into every Text-to-SQL prompt.

Keeping this in one place means any schema change (new view, renamed column)
is fixed once here and reflected everywhere automatically.
"""

# ---------------------------------------------------------------------------
# Schema context
# ---------------------------------------------------------------------------
# Written as explicit DDL + notes rather than auto-introspected,
# because the LLM needs semantic hints (what columns mean) not just types.

SCHEMA_CONTEXT = """
You have access to a DuckDB database with the following tables and views.
All timestamps are UTC. Site IDs: '0001' = JPL, '0002' = Caltech, '0019' = Office 1.

--- TABLE: sessions ---
Raw charging session records from ACN-Data (Caltech EV charging network).
One row = one charging session.

  _id                VARCHAR        -- unique session record ID (primary key)
  sessionID          VARCHAR        -- human-readable session identifier
  siteID             VARCHAR        -- site code: '0001'=JPL, '0002'=Caltech, '0019'=Office1
  stationID          VARCHAR        -- EVSE (charger) identifier
  spaceID            VARCHAR        -- parking space identifier
  clusterID          VARCHAR        -- sub-group of EVSEs within a site
  userID             VARCHAR        -- anonymised user identifier (NULL if unclaimed)
  connectionTime     TIMESTAMP      -- when the EV plugged in (UTC)
  disconnectTime     TIMESTAMP      -- when the EV unplugged (UTC)
  doneChargingTime   TIMESTAMP      -- last non-zero current draw (NULL if not recorded)
  kWhDelivered       DOUBLE         -- energy delivered in kilowatt-hours
  timezone           VARCHAR        -- site local timezone (e.g. 'America/Los_Angeles')
  kWhRequested       DOUBLE         -- energy requested by user (may be NULL)
  milesRequested     DOUBLE         -- miles requested by user (may be NULL)
  minutesAvailable   DOUBLE         -- session length estimated by user (may be NULL)
  requestedDeparture TIMESTAMP      -- user's estimated departure time (may be NULL)
  WhPerMile          DOUBLE         -- vehicle efficiency in Wh/mile (may be NULL)
  paymentRequired    BOOLEAN        -- whether user paid for the session

--- VIEW: sessions_features ---
Cleaned sessions with derived temporal and energy features.
Filters out: kWhDelivered=0, kWhDelivered>=200, duration<1min, duration>=24hrs.
USE THIS VIEW (not raw sessions) for any analytical query.

  (all columns from sessions, plus:)
  hour_of_day        DOUBLE         -- 0–23, extracted from connectionTime
  day_of_week        DOUBLE         -- 0=Sunday, 6=Saturday
  month              DOUBLE         -- 1–12
  year               DOUBLE         -- e.g. 2019, 2020, 2021
  season             VARCHAR        -- 'winter','spring','summer','autumn'
  day_type           VARCHAR        -- 'weekday' or 'weekend'
  duration_hours     DOUBLE         -- session duration (connectionTime → disconnectTime)
  idle_hours         DOUBLE         -- time connected but not charging (NULL if doneChargingTime missing)
  avg_charge_rate_kw DOUBLE         -- average charging rate in kW

--- VIEW: hourly_utilization ---
Session counts and averages grouped by site, day type, and hour of day.
USE FOR: peak hour queries, time-of-use patterns.

  siteID             VARCHAR
  day_type           VARCHAR        -- 'weekday' or 'weekend'
  hour_of_day        DOUBLE         -- 0–23
  session_count      BIGINT
  avg_kwh            DOUBLE
  avg_duration_hours DOUBLE
  avg_charge_rate_kw DOUBLE

--- VIEW: seasonal_summary ---
Aggregated metrics by site, season, and year.
USE FOR: seasonal variation queries.

  siteID             VARCHAR
  season             VARCHAR        -- 'winter','spring','summer','autumn'
  year               DOUBLE
  session_count      BIGINT
  total_kwh          DOUBLE
  avg_kwh_per_session DOUBLE
  avg_duration_hours DOUBLE
  unique_users       BIGINT

--- VIEW: daily_evse_utilization ---
Per-charger daily throughput.
USE FOR: utilization rate queries, charger-level analysis.

  siteID             VARCHAR
  stationID          VARCHAR
  date               DATE
  sessions_per_day   BIGINT
  kwh_per_day        DOUBLE
  occupied_hours     DOUBLE

--- TABLE: site_lookup ---
Maps siteID codes to human-readable names.

  site_code          VARCHAR        -- '0001', '0002', '0019'
  site_name          VARCHAR        -- LOWERCASE: 'jpl', 'caltech', 'office001'
  site_label         VARCHAR        -- full descriptive name
  evse_count         INTEGER        -- number of chargers at site (54 Caltech, 52 JPL, 9 Office1)

--- CRITICAL QUERY RULES ---

1. PEAK HOUR QUERIES (q05, q28 pattern):
   To find the busiest hour, always query sessions_features directly — NOT hourly_utilization.
   Use GROUP BY hour_of_day, not GROUP BY hour_of extr or any other variant.
   Correct pattern:
     SELECT CAST(hour_of_day AS INTEGER) AS hour, COUNT(*) AS n
     FROM sessions_features
     WHERE day_type = 'weekday'           -- add site filter if needed
     GROUP BY hour_of_day
     ORDER BY n DESC
     LIMIT 1

2. SUMMER SESSION COUNT (q27 pattern):
   seasonal_summary has ONE ROW PER SITE/YEAR COMBINATION — COUNT(*) on it returns 10,
   not the number of sessions. Always use sessions_features for row-level counts:
     SELECT COUNT(*) FROM sessions_features WHERE season = 'summer'

3. SESSIONS PER EVSE (q30 pattern):
   daily_evse_utilization.sessions_per_day is per charger per day.
   Filter by siteID directly — do NOT join site_lookup for site filtering:
     SELECT AVG(sessions_per_day)
     FROM daily_evse_utilization
     WHERE siteID = '0002'

4. SITE NAME MATCHING:
   site_lookup.site_name values are LOWERCASE: 'caltech', 'jpl', 'office001'.
   Never use 'Caltech' or 'JPL' in WHERE clauses on site_lookup.
   Prefer filtering on sessions_features.siteID directly using codes '0001','0002','0019'.

5. ALIAS CONSISTENCY IN ORDER BY:
   Always use the alias defined in SELECT, not a modified version.
   Example: SELECT AVG(x) AS avg_kwh ... ORDER BY avg_kwh DESC  (not avg_kwh_per_session)

6. CASE STATEMENT IN ORDER BY must have an END clause before LIMIT:
   ORDER BY siteID, CASE day_type WHEN 'weekday' THEN 1 WHEN 'weekend' THEN 2 END
   (not truncated before END)

7. YEAR COLUMN TYPE: year in sessions_features is DOUBLE (e.g. 2019.0).
   Use: WHERE year = 2021   (not EXTRACT(YEAR FROM year))
"""


def get_schema_context() -> str:
    return SCHEMA_CONTEXT.strip()