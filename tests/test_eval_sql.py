"""Phase 12.3.7 - the SQL check earns its keep only where execution does not.

The point of these tests is the module's own premise: **SQLite's DDL accepts almost everything a
generator can get wrong**, so a checker that merely executes the script and reports no exception
is close to worthless. Each test below therefore pins one thing that executes cleanly and is
still wrong, plus the postcondition that nothing which passes should have.
"""

from __future__ import annotations

import sqlite3

import pytest

from src.eval import sql


def test_the_premise_holds_sqlite_really_does_accept_a_dangling_foreign_key():
    """If this ever fails, the module's reason to exist has changed and it should be re-argued."""
    con = sqlite3.connect(":memory:")
    con.execute("PRAGMA foreign_keys = ON")
    con.execute("CREATE TABLE a(id INTEGER PRIMARY KEY, b_id INTEGER REFERENCES nosuch(id))")
    assert con.execute("PRAGMA foreign_keys").fetchone()[0] == 1
    assert con.execute("PRAGMA foreign_key_check").fetchall() == []
    con.close()


def test_the_premise_holds_for_invented_type_names():
    con = sqlite3.connect(":memory:")
    con.execute("CREATE TABLE t(x TEXTT, y BANANA)")
    assert [r[2] for r in con.execute("PRAGMA table_info(t)")] == ["TEXTT", "BANANA"]
    con.close()


def test_reference_schema_passes_and_reports_its_structure():
    result = sql.check(sql.REFERENCE_DDL)
    assert result["ok"] is True
    assert result["kind"] == sql.KIND_OK
    assert result["foreign_keys"] == 8
    assert len(result["tables"]) == 8


@pytest.mark.parametrize(
    ("name", "ddl", "kind"),
    [
        (
            "dangling parent table",
            "CREATE TABLE a(id INTEGER PRIMARY KEY, b INTEGER REFERENCES nosuch(id));",
            sql.KIND_FK,
        ),
        (
            "non-unique parent column",
            "CREATE TABLE p(a INT);\nCREATE TABLE c(b INT REFERENCES p(a));",
            sql.KIND_FK,
        ),
        ("empty schema", "-- just a comment\n", sql.KIND_EMPTY),
        ("duplicate table", "CREATE TABLE t(x INT);\nCREATE TABLE t(y INT);", sql.KIND_EXEC),
        ("syntax error", "CREATE TABL x(", sql.KIND_EXEC),
    ],
)
def test_bad_ddl_is_rejected_with_the_right_kind(name, ddl, kind):
    result = sql.check(ddl)
    assert result["ok"] is False, name
    assert result["kind"] == kind, f"{name}: {result['detail']}"


def test_invented_type_names_are_reported_even_when_another_check_names_the_kind():
    """`kind` is the first failing category; the invented types must still appear in `errors`."""
    result = sql.check("CREATE TABLE t(x TEXTT, y BANANA);")
    assert result["ok"] is False
    joined = " ".join(result["errors"])
    assert "TEXTT" in joined and "BANANA" in joined


def test_strict_types_can_be_switched_off():
    strict = sql.check("CREATE TABLE t(id INTEGER PRIMARY KEY, x BANANA);")
    relaxed = sql.check("CREATE TABLE t(id INTEGER PRIMARY KEY, x BANANA);", strict_types=False)
    assert strict["ok"] is False
    assert relaxed["ok"] is True


def test_an_empty_schema_is_a_failure_not_a_pass():
    """The likeliest way for a broken generator to score 100% is to emit nothing at all."""
    for ddl in ("", "   ", "-- nothing\n", "\n\n"):
        assert sql.check(ddl)["kind"] == sql.KIND_EMPTY


# ------------------------------------------------------------------------------------------
# statement splitting - the bug this module shipped with
# ------------------------------------------------------------------------------------------


def test_two_statements_on_one_line_are_split():
    """A line-at-a-time splitter handed both to `execute` and SQLite refused the pair.

    That surfaced as `sql.execution_error`, i.e. a schema rejected for the checker's bug rather
    than its own, which is the worst failure mode a checker has.
    """
    assert sql.split_statements("CREATE TABLE p(a INT); CREATE TABLE c(b INT);") == [
        "CREATE TABLE p(a INT);",
        "CREATE TABLE c(b INT);",
    ]


def test_a_trigger_body_is_not_torn_at_its_inner_semicolon():
    ddl = "CREATE TRIGGER t AFTER INSERT ON a BEGIN UPDATE b SET x=1; END;"
    assert sql.split_statements(ddl) == [ddl]


def test_valid_schema_written_on_one_line_is_accepted():
    result = sql.check(
        "CREATE TABLE p(a INTEGER PRIMARY KEY); "
        "CREATE TABLE c(id INTEGER PRIMARY KEY, b INTEGER REFERENCES p(a));"
    )
    assert result["ok"] is True, result["detail"]


# ------------------------------------------------------------------------------------------
# cardinality
# ------------------------------------------------------------------------------------------


def test_one_to_one_is_distinguished_from_one_to_many():
    result = sql.check(
        "CREATE TABLE person(id INTEGER PRIMARY KEY);\n"
        "CREATE TABLE profile(id INTEGER PRIMARY KEY, "
        "person_id INTEGER UNIQUE NOT NULL REFERENCES person(id));\n"
        "CREATE TABLE note(id INTEGER PRIMARY KEY, "
        "person_id INTEGER NOT NULL REFERENCES person(id));"
    )
    assert result["ok"] is True, result["detail"]
    kinds = {(r["child"], r["cardinality"]) for r in result["relationships"]}
    assert ("profile", "one-to-one") in kinds
    assert ("note", "one-to-many") in kinds


def test_a_junction_table_is_recognised_as_many_to_many():
    result = sql.check(sql.REFERENCE_DDL)
    assert {"book_author", "book_tag"} <= {j["table"] for j in result["junction_tables"]}


def test_a_nullable_foreign_key_is_optional_participation_not_an_error():
    result = sql.check(
        "CREATE TABLE p(id INTEGER PRIMARY KEY);\n"
        "CREATE TABLE c(id INTEGER PRIMARY KEY, p_id INTEGER REFERENCES p(id));"
    )
    assert result["ok"] is True, result["detail"]
    assert any(r["optional"] for r in result["relationships"])


# ------------------------------------------------------------------------------------------
# postgres, and refusing rather than pretending
# ------------------------------------------------------------------------------------------


def test_postgres_is_refused_explicitly_rather_than_silently_downgraded():
    """A Postgres pass that was really a SQLite pass is worse than no number at all."""
    available = sql.postgres_available()
    result = sql.check(sql.REFERENCE_DDL, dialect="postgres")
    if available["available"]:
        assert result["kind"] != sql.KIND_UNAVAILABLE
    else:
        assert result["ok"] is False
        assert result["kind"] == sql.KIND_UNAVAILABLE
        assert result["available"] is False


# ------------------------------------------------------------------------------------------
# batch, isolation and cost
# ------------------------------------------------------------------------------------------


def test_check_many_matches_check_one_by_one():
    ddls = [sql.REFERENCE_DDL, "CREATE TABL bad(", "-- empty"]
    assert [r["kind"] for r in sql.check_many(ddls)] == [sql.check(d)["kind"] for d in ddls]


def test_each_call_gets_a_fresh_database():
    first = "CREATE TABLE only_in_first(id INTEGER PRIMARY KEY);"
    sql.check(first)
    second = sql.check(
        "CREATE TABLE c(id INTEGER PRIMARY KEY, " "x INTEGER REFERENCES only_in_first(id));"
    )
    assert second["ok"] is False
    assert second["kind"] == sql.KIND_FK


def test_the_repo_is_not_touched(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    sql.check(sql.REFERENCE_DDL)
    assert list(tmp_path.iterdir()) == []


def test_cost_stays_in_the_band_the_docstring_claims():
    """0.41 ms mean per call is quoted; a 20x regression should fail rather than be absorbed."""
    import time

    started = time.perf_counter()
    for _ in range(100):
        sql.check(sql.REFERENCE_DDL)
    per_call_ms = (time.perf_counter() - started) * 1000.0 / 100
    assert per_call_ms < 8.0, f"{per_call_ms:.2f} ms per call"
