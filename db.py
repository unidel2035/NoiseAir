import sqlite3
import os
from datetime import datetime

DB_PATH = os.environ.get("NOISEAIR_DB", os.path.join(os.path.dirname(__file__), "noiseair.db"))


def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    with get_conn() as conn:
        conn.executescript("""
            PRAGMA journal_mode=WAL;

            CREATE TABLE IF NOT EXISTS events (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                ts_start        TEXT NOT NULL,
                ts_end          TEXT NOT NULL,
                duration_sec    REAL NOT NULL,
                peak_db         REAL NOT NULL,
                avg_db          REAL NOT NULL,
                avg_confidence  REAL NOT NULL,
                icao            TEXT,
                callsign        TEXT,
                altitude_ft     INTEGER,
                distance_km     REAL,
                created_at      TEXT NOT NULL DEFAULT (datetime('now'))
            );

            CREATE INDEX IF NOT EXISTS idx_events_ts_start ON events(ts_start);
            CREATE INDEX IF NOT EXISTS idx_events_icao ON events(icao);
        """)


def insert_event(event: dict) -> int:
    with get_conn() as conn:
        cur = conn.execute("""
            INSERT INTO events
                (ts_start, ts_end, duration_sec, peak_db, avg_db, avg_confidence,
                 icao, callsign, altitude_ft, distance_km)
            VALUES
                (:ts_start, :ts_end, :duration_sec, :peak_db, :avg_db, :avg_confidence,
                 :icao, :callsign, :altitude_ft, :distance_km)
        """, event)
        return cur.lastrowid


def get_events(date_from: str = None, date_to: str = None,
               icao: str = None, limit: int = 200) -> list:
    query = "SELECT * FROM events WHERE 1=1"
    params = []

    if date_from:
        query += " AND ts_start >= ?"
        params.append(date_from)
    if date_to:
        query += " AND ts_start <= ?"
        params.append(date_to)
    if icao:
        query += " AND icao = ?"
        params.append(icao)

    query += " ORDER BY ts_start DESC LIMIT ?"
    params.append(limit)

    with get_conn() as conn:
        rows = conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]


def get_stats() -> dict:
    with get_conn() as conn:
        row = conn.execute("""
            SELECT
                COUNT(*)            AS total_events,
                ROUND(AVG(duration_sec), 1) AS avg_duration,
                ROUND(MAX(peak_db), 1)      AS max_peak_db,
                ROUND(AVG(peak_db), 1)      AS avg_peak_db,
                COUNT(DISTINCT icao)        AS unique_aircraft,
                DATE(MAX(ts_start))         AS last_event_date
            FROM events
        """).fetchone()
        return dict(row)
