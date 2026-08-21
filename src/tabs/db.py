import sqlite3
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS sources (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    feed_url TEXT NOT NULL UNIQUE,
    category TEXT NOT NULL,
    institutional_tier INTEGER NOT NULL,
    earned_tier INTEGER NOT NULL,
    consecutive_failures INTEGER NOT NULL DEFAULT 0,
    last_successful_fetch_at TEXT
);

CREATE TABLE IF NOT EXISTS articles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_id INTEGER NOT NULL REFERENCES sources(id),
    url TEXT NOT NULL,
    title TEXT,
    full_text TEXT,
    content_hash TEXT NOT NULL,
    published_at TEXT,
    retrieved_at TEXT NOT NULL,
    previous_version_id INTEGER REFERENCES articles(id)
);

CREATE TABLE IF NOT EXISTS claims (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    article_id INTEGER NOT NULL REFERENCES articles(id),
    source_id INTEGER NOT NULL REFERENCES sources(id),
    claim_text TEXT NOT NULL,
    supporting_excerpt TEXT NOT NULL,
    claim_type TEXT NOT NULL,
    category TEXT NOT NULL,
    sub_tags TEXT,
    status TEXT NOT NULL DEFAULT 'unverified',
    confidence_score REAL,
    llm_certainty REAL,
    corroboration_count INTEGER NOT NULL DEFAULT 0,
    story_cluster_id INTEGER REFERENCES story_clusters(id),
    author TEXT,
    published_at TEXT,
    retrieved_at TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS perspectives (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    article_id INTEGER NOT NULL REFERENCES articles(id),
    source_id INTEGER NOT NULL REFERENCES sources(id),
    perspective_text TEXT NOT NULL,
    supporting_excerpt TEXT NOT NULL,
    category TEXT NOT NULL,
    sub_tags TEXT,
    author TEXT,
    published_at TEXT,
    retrieved_at TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS conflicts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    claim_a_id INTEGER NOT NULL REFERENCES claims(id),
    claim_b_id INTEGER NOT NULL REFERENCES claims(id),
    resolution TEXT NOT NULL,
    winning_claim_id INTEGER REFERENCES claims(id),
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS story_clusters (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    category TEXT NOT NULL,
    summary TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS anomaly_flags (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    article_id INTEGER NOT NULL REFERENCES articles(id),
    reason TEXT NOT NULL,
    created_at TEXT NOT NULL,
    reviewed INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS run_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_started_at TEXT NOT NULL,
    run_finished_at TEXT,
    source_id INTEGER REFERENCES sources(id),
    status TEXT NOT NULL,
    message TEXT
);
"""


def get_connection(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)
    conn.commit()
