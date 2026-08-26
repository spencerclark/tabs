from tabs.db import SCHEMA, SCHEMA_VERSION, get_connection, init_db

# The pre-Phase-2b claims schema: identical to current SCHEMA's claims table except the
# `embedding TEXT` column line is removed, to simulate a database created before the
# embedding column existed.
PRE_PHASE_2B_CLAIMS_SCHEMA = """
CREATE TABLE claims (
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
"""

EXPECTED_TABLES = {
    "sources", "articles", "claims", "perspectives",
    "conflicts", "story_clusters", "anomaly_flags", "run_log",
}


def test_init_db_creates_all_tables(tmp_path):
    conn = get_connection(tmp_path / "test.db")
    init_db(conn)

    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()
    table_names = {row["name"] for row in rows}

    assert EXPECTED_TABLES.issubset(table_names)
    conn.close()


def test_init_db_creates_the_articles_url_index(tmp_path):
    conn = get_connection(tmp_path / "test.db")
    init_db(conn)

    rows = conn.execute("SELECT name FROM sqlite_master WHERE type='index'").fetchall()

    assert "idx_articles_url" in {row["name"] for row in rows}
    conn.close()


def test_init_db_stamps_the_schema_version(tmp_path):
    conn = get_connection(tmp_path / "test.db")
    init_db(conn)

    assert conn.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
    conn.close()


def test_init_db_is_idempotent(tmp_path):
    conn = get_connection(tmp_path / "test.db")
    init_db(conn)
    init_db(conn)  # must not raise

    conn.close()


def test_init_db_adds_an_embedding_column_to_claims(tmp_path):
    conn = get_connection(tmp_path / "test.db")
    init_db(conn)

    columns = {row["name"] for row in conn.execute("PRAGMA table_info(claims)").fetchall()}

    assert "embedding" in columns
    conn.close()


def test_init_db_backfills_the_embedding_column_on_a_pre_phase_2b_database(tmp_path):
    conn = get_connection(tmp_path / "test.db")

    # Build a pre-Phase-2b database: full current schema for everything except claims,
    # which uses the schema as it existed before the embedding column was added.
    conn.executescript(SCHEMA)
    conn.execute("DROP TABLE claims")
    conn.executescript(PRE_PHASE_2B_CLAIMS_SCHEMA)
    conn.execute(f"PRAGMA user_version = 1")
    conn.commit()

    columns_before = {row["name"] for row in conn.execute("PRAGMA table_info(claims)").fetchall()}
    assert "embedding" not in columns_before

    conn.execute(
        "INSERT INTO sources (id, name, feed_url, category, institutional_tier, earned_tier) "
        "VALUES (1, 'Test Source', 'https://example.com/feed', 'appsec', 2, 2)"
    )
    conn.execute(
        "INSERT INTO articles (id, source_id, url, content_hash, retrieved_at) "
        "VALUES (1, 1, 'https://example.com/a', 'hash123', '2026-01-01T00:00:00+00:00')"
    )
    conn.execute(
        "INSERT INTO claims "
        "(id, article_id, source_id, claim_text, supporting_excerpt, claim_type, category, "
        " retrieved_at, created_at) "
        "VALUES (1, 1, 1, 'A pre-existing claim', 'excerpt text', 'factual', 'appsec', "
        " '2026-01-01T00:00:00+00:00', '2026-01-01T00:00:00+00:00')"
    )
    conn.commit()

    init_db(conn)

    columns_after = {row["name"] for row in conn.execute("PRAGMA table_info(claims)").fetchall()}
    assert "embedding" in columns_after

    row = conn.execute("SELECT claim_text FROM claims WHERE id = 1").fetchone()
    assert row is not None
    assert row["claim_text"] == "A pre-existing claim"

    assert conn.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
    conn.close()


def test_init_db_creates_the_claims_category_retrieved_at_index(tmp_path):
    conn = get_connection(tmp_path / "test.db")
    init_db(conn)

    rows = conn.execute("SELECT name FROM sqlite_master WHERE type='index'").fetchall()

    assert "idx_claims_category_retrieved_at" in {row["name"] for row in rows}
    conn.close()
