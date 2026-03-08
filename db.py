"""
db.py
=====
DuckDB connection management and schema initialisation.
"""

import logging

import duckdb

from config import DB_PATH

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS sessions (
    _id                 VARCHAR PRIMARY KEY,
    sessionID           VARCHAR,
    siteID              VARCHAR,
    stationID           VARCHAR,
    spaceID             VARCHAR,
    clusterID           VARCHAR,
    userID              VARCHAR,
    connectionTime      TIMESTAMP,
    disconnectTime      TIMESTAMP,
    doneChargingTime    TIMESTAMP,
    kWhDelivered        DOUBLE,
    timezone            VARCHAR,
    -- Flattened from userInputs (latest entry)
    kWhRequested        DOUBLE,
    milesRequested      DOUBLE,
    minutesAvailable    DOUBLE,
    requestedDeparture  TIMESTAMP,
    WhPerMile           DOUBLE,
    paymentRequired     BOOLEAN,
    -- Audit
    ingested_at         TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""


# ---------------------------------------------------------------------------
# Connection
# ---------------------------------------------------------------------------

def get_connection(db_path=DB_PATH) -> duckdb.DuckDBPyConnection:
    """Open (or create) the DuckDB database and ensure schema exists."""
    con = duckdb.connect(str(db_path))
    con.execute(SCHEMA_SQL)
    log.info(f"Database ready at {db_path}")
    return con