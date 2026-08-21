from tabs.db import get_connection, init_db

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


def test_init_db_is_idempotent(tmp_path):
    conn = get_connection(tmp_path / "test.db")
    init_db(conn)
    init_db(conn)  # must not raise

    conn.close()
