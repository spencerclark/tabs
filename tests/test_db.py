from tabs.db import SCHEMA_VERSION, get_connection, init_db

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


def test_init_db_creates_the_claims_category_retrieved_at_index(tmp_path):
    conn = get_connection(tmp_path / "test.db")
    init_db(conn)

    rows = conn.execute("SELECT name FROM sqlite_master WHERE type='index'").fetchall()

    assert "idx_claims_category_retrieved_at" in {row["name"] for row in rows}
    conn.close()
