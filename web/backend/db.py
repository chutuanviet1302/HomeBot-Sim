"""SQLite store of the HomeBot-Sim results (web/homebot.db): experiments, runs, replays, jobs."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DB_PATH = ROOT / "web" / "homebot.db"

# columns shared by all result CSVs (after renaming); everything else goes to runs.extra (JSON)
RUN_COLUMNS = ["experiment", "seed", "policy", "success", "failed_at", "time_s", "person_contacts",
               "person_contacts_robot_moving", "min_person_dist", "furniture_contacts", "arm_contacts",
               "arm_stops", "fell", "basin_err_mm", "sink_offset", "perception_err_mm"]

SCHEMA = """
CREATE TABLE IF NOT EXISTS experiments (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT,
    source TEXT,
    sort INTEGER DEFAULT 0
);
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment TEXT NOT NULL REFERENCES experiments(id),
    seed INTEGER,
    policy TEXT,
    success INTEGER,
    failed_at TEXT,
    time_s REAL,
    person_contacts INTEGER,
    person_contacts_robot_moving INTEGER,
    min_person_dist REAL,
    furniture_contacts INTEGER,
    arm_contacts INTEGER,
    arm_stops INTEGER,
    fell INTEGER,
    basin_err_mm REAL,
    sink_offset REAL,
    perception_err_mm REAL,
    extra TEXT
);
CREATE INDEX IF NOT EXISTS runs_exp ON runs(experiment, policy);
CREATE TABLE IF NOT EXISTS replays (
    key TEXT PRIMARY KEY,
    level INTEGER,
    seed INTEGER,
    variant TEXT,
    npz TEXT,
    npz_mtime REAL,
    json_path TEXT,
    result TEXT,
    duration_s REAL
);
CREATE TABLE IF NOT EXISTS jobs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    level INTEGER,
    seed INTEGER,
    sink_range REAL,
    person TEXT,
    policy TEXT,
    status TEXT,
    created REAL,
    started REAL,
    finished REAL,
    log_path TEXT,
    replay_key TEXT,
    result TEXT
);
"""


def connect(path: Path = DB_PATH) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path, check_same_thread=False, timeout=30)
    con.row_factory = sqlite3.Row
    con.executescript(SCHEMA)
    return con


def run_row(r: sqlite3.Row) -> dict:
    d = dict(r)
    d["extra"] = json.loads(d["extra"] or "{}")
    for k in ("success", "fell"):
        if d.get(k) is not None:
            d[k] = bool(d[k])
    return d
