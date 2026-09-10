"""Phase 12.3.7 - executing generated DDL, and checking the two things execution does not.

    from src.eval import sql
    sql.check(ddl)                       # -> {"ok": ..., "kind": ..., "detail": ..., ...}
    sql.check_many([ddl, ...])           # one connection per script, same result dicts

The plan's line is *execute DDL on SQLite/Postgres, verify FKs and cardinality*, and the
interesting half is `verify`. Execution alone is close to worthless as a gate here, because
**SQLite's DDL accepts almost everything a generator can get wrong.** That is measured, not
assumed - the three probes below all succeed on a fresh in-memory connection:

    CREATE TABLE a(id INTEGER PRIMARY KEY,
                   b_id INTEGER REFERENCES nosuch(id));   -- parent table does not exist
    PRAGMA foreign_keys;                                  -- -> 1, enforcement IS on
    PRAGMA foreign_key_check;                             -- -> [] , nothing to report
    CREATE TABLE t(x TEXTT, y BANANA);                    -- -> accepted, both columns created

So a checker that executes the script and reports the absence of an exception passes a schema
whose foreign keys point at nothing. Everything of value in this module is the static pass that
runs *after* execution succeeds.

## Why `PRAGMA foreign_keys = ON` is necessary but not sufficient

The pragma is off by default and this module turns it on, because without it any INSERT-bearing
script sails past its own violations. But the pragma governs **DML**: SQLite resolves a foreign
key's parent lazily, at the first row touched, which is why the dangling reference above only
raises `no such table: main.nosuch` on the INSERT and why `foreign_key_check` - which walks
*rows* - reports nothing for an empty schema. Generated DDL is empty by construction. Turning the
pragma on is therefore correct and load-bearing for scripts that seed rows, and a no-op for the
common case; the dangling-parent detection below is what actually catches those, by resolving
every `PRAGMA foreign_key_list` entry against `PRAGMA table_info` itself.

## The three verifications

    foreign keys   every FK's parent table exists; every referenced parent column exists; and the
                   parent column set is UNIQUE or the primary key. That last one is a real error,
                   not a nicety - SQLite raises `foreign key mismatch` at DML time against a
                   non-unique parent, and Postgres rejects it at DDL time. An FK with no explicit
                   column list resolves against the parent's primary key, and is an error when the
                   parent has none.
    cardinality    each FK is classified from the *child* side: a UNIQUE (or PK) child column
                   makes it one-to-one, otherwise one-to-many; and a table whose primary key is
                   exactly the union of two or more FK columns is reported as a junction table,
                   which is how a many-to-many appears in a relational schema. Nullability of the
                   child column is reported as optional/mandatory participation.
    declared types under `strict_types` (default on) each declared type name is matched against the
                   SQL92-plus-SQLite set. This exists only because of the `TEXTT`/`BANANA` probe
                   above: SQLite stores an unknown type name verbatim and gives the column BLOB
                   affinity, so a hallucinated type is invisible to every other check in this
                   file. Postgres would reject it. Columns with no declared type at all are legal
                   SQLite and are not flagged.

An empty schema - a script that parses and runs but creates no table - is a failure, not a pass.
It is the single most likely way for a broken generator to score 100%.

## What was rejected

    `executescript`         it wraps the batch in one implicit transaction, and `PRAGMA
                            foreign_keys` is documented as a no-op inside a transaction, so the
                            pragma this module depends on would have been silently ignored.
                            Statements are split on `sqlite3.complete_statement` and executed one
                            at a time, which also buys per-statement error attribution
                            (`error_statement_index`) instead of one opaque message per script.
    a SQL parser dependency `sqlglot`/`sqlparse` would give a dialect-agnostic AST, but the whole
                            point of the row is that the database is the authority on whether the
                            DDL executes. The schema is read back out of SQLite's own pragmas,
                            so what is verified is what the engine actually built.
    treating a nullable FK  an optional relationship is a legitimate modelling choice, and the IR
    as an error             this evaluates cannot distinguish it from an omission. Reported in
                            `relationships[*].optional`, never counted as an error.

## Postgres

**Not available in this environment, and this module does not pretend otherwise.** Probed at
authoring time: no `psycopg`, no `psycopg2`, and no `psql`/`postgres`/`pg_ctl` on PATH.
`postgres_available()` re-probes at call time. `check(ddl, dialect="postgres")` therefore returns
`ok=False` with `kind="sql.unavailable"` and `available=False` - an explicit refusal rather than a
silent fall back to SQLite, because a Postgres pass that was really a SQLite pass is worse than no
number at all. Everything reported by this module is a **SQLite** result. The gap that leaves is
named: Postgres rejects unknown types and non-unique FK parents at DDL time, and enforces NOT NULL
and CHECK on the DDL path, so the static passes here stand in for a stricter engine.

## Measured

Timings are from `python -m src.eval.sql --benchmark`, CPU only, on the eight-table reference
schema in `REFERENCE_DDL` (8 tables, 8 foreign keys, 2 junction tables):

    per call, mean over 1,000 calls          0.41 ms   (median 0.39, max 1.00)
    of which connection setup                0.02 ms
    empty-schema script                      0.03 ms

so a sweep over 10,000 candidates costs about 4 s single-threaded. Nothing is written to disk;
the connection is `:memory:` and is closed in a `finally`.

Detection was proved against `BAD_DDL`, eleven scripts each broken in exactly one way, and
**all eleven are rejected** (`tests/test_eval_sql.py`), while `REFERENCE_DDL` - 8 tables, 8
resolving foreign keys, 2 junction tables - passes. The split is the point of the module:

    8 of the 11 execute without raising a single error   and are caught only by the static pass
      dangling parent table, dangling parent column, non-unique FK parent, parent with no
      primary key, `TEXTT`, `BANANA`, a table with no key at all, and the empty schema
    3 of the 11 are caught by execution itself
      duplicate table, duplicate column, syntax error

One thing found while measuring, and worth stating because it contradicts the tidy story above:
`PRAGMA foreign_key_check` does **not** silently return `[]` for every unresolvable parent - on
some of them it raises `OperationalError: foreign key mismatch` while resolving the parent key,
with zero rows in the table. So it is a partial schema check, not purely a row walk. It is
caught rather than allowed to crash, and it is not relied on: its message names only the two
tables, so it cannot distinguish a missing table from a missing column from a non-unique parent.
The static pass produces the specific message and the pragma result is a fallback.
"""

from __future__ import annotations

import re
import shutil
import sqlite3
import time
from typing import Any

__all__ = [
    "check",
    "check_many",
    "postgres_available",
    "split_statements",
    "KNOWN_TYPES",
    "REFERENCE_DDL",
    "BAD_DDL",
]

KIND_OK = "sql.ok"
KIND_EXEC = "sql.execution_error"
KIND_EMPTY = "sql.empty_schema"
KIND_FK = "sql.foreign_key_error"
KIND_CARD = "sql.cardinality_error"
KIND_TYPE = "sql.type_error"
KIND_UNAVAILABLE = "sql.unavailable"

# SQL92 plus the SQLite affinity names plus the handful Postgres adds that a generator
# reasonably emits. Parameters (VARCHAR(40), NUMERIC(10,2)) are stripped before lookup.
KNOWN_TYPES = frozenset(
    {
        "INT",
        "INTEGER",
        "TINYINT",
        "SMALLINT",
        "MEDIUMINT",
        "BIGINT",
        "INT2",
        "INT4",
        "INT8",
        "SERIAL",
        "BIGSERIAL",
        "SMALLSERIAL",
        "UNSIGNED BIG INT",
        "CHARACTER",
        "CHAR",
        "VARCHAR",
        "VARYING CHARACTER",
        "NCHAR",
        "NATIVE CHARACTER",
        "NVARCHAR",
        "TEXT",
        "CLOB",
        "BLOB",
        "BYTEA",
        "REAL",
        "DOUBLE",
        "DOUBLE PRECISION",
        "FLOAT",
        "NUMERIC",
        "DECIMAL",
        "MONEY",
        "BOOLEAN",
        "BOOL",
        "DATE",
        "DATETIME",
        "TIME",
        "TIMESTAMP",
        "TIMESTAMPTZ",
        "TIMESTAMP WITH TIME ZONE",
        "TIMESTAMP WITHOUT TIME ZONE",
        "INTERVAL",
        "UUID",
        "JSON",
        "JSONB",
        "XML",
        "CHARACTER VARYING",
    }
)

_TYPE_PARAMS = re.compile(r"\s*\([^)]*\)\s*$")
_WS = re.compile(r"\s+")


def _normalise_type(declared: str) -> str:
    return _WS.sub(" ", _TYPE_PARAMS.sub("", declared).strip()).upper()


def postgres_available() -> dict[str, Any]:
    """Probe for a usable Postgres. Returns ``{"available": bool, "driver": .., "psql": ..}``.

    Deliberately does not attempt a connection: an unreachable server and an absent driver are
    both "no" for this row's purposes, and a connection attempt would block the sweep.
    """
    driver = None
    for name in ("psycopg", "psycopg2"):
        try:
            __import__(name)
        except ImportError:
            continue
        driver = name
        break
    psql = shutil.which("psql")
    return {"available": driver is not None, "driver": driver, "psql": psql}


def split_statements(ddl: str) -> list[str]:
    """Split a script into complete SQL statements using SQLite's own parser.

    ``sqlite3.complete_statement`` understands ``BEGIN ... END`` trigger bodies, so a trigger
    containing semicolons is not torn in half the way a naive ``split(";")`` tears it.

    **Scanning is per terminator, not per line.** A line-at-a-time version of this function
    shipped first and was wrong on a single line holding two statements: the whole line becomes
    one buffer, ``complete_statement`` agrees it is complete, and the pair is handed to
    ``execute`` as one string, which raises ``You can only execute one statement at a time``.
    Generated DDL is exactly the kind of input that arrives on one line, and the failure was
    reported as ``sql.execution_error`` - a schema rejected for the checker's bug rather than
    its own. Cutting at each ``;`` that leaves a complete statement fixes it and keeps the
    trigger-body behaviour, which is the only reason ``complete_statement`` is here at all.
    """
    statements: list[str] = []
    buffer = ""
    for character in ddl:
        buffer += character
        if character == ";" and sqlite3.complete_statement(buffer):
            if buffer.strip():
                statements.append(buffer.strip())
            buffer = ""
    if buffer.strip():
        statements.append(buffer.strip())
    return statements


def _result(ok: bool, kind: str, detail: str, **extra: Any) -> dict[str, Any]:
    out: dict[str, Any] = {"ok": ok, "kind": kind, "detail": detail}
    out.update(extra)
    return out


def _read_schema(con: sqlite3.Connection) -> dict[str, dict[str, Any]]:
    """Read the built schema back out of SQLite's pragmas - the engine is the authority."""
    rows = con.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
    ).fetchall()
    schema: dict[str, dict[str, Any]] = {}
    for (table,) in rows:
        info = con.execute(f'PRAGMA table_info("{table}")').fetchall()
        columns = {
            row[1]: {
                "type": row[2] or "",
                "not_null": bool(row[3]),
                "default": row[4],
                "pk_position": int(row[5]),
            }
            for row in info
        }
        unique_sets: list[tuple[str, ...]] = []
        for idx in con.execute(f'PRAGMA index_list("{table}")').fetchall():
            # index_list rows are (seq, name, unique, origin, partial)
            if not idx[2] or idx[4]:
                continue  # non-unique, or partial (which cannot back a foreign key)
            members = con.execute(f'PRAGMA index_info("{idx[1]}")').fetchall()
            unique_sets.append(tuple(m[2] for m in members))
        pk = tuple(
            name
            for name, meta in sorted(columns.items(), key=lambda kv: kv[1]["pk_position"])
            if meta["pk_position"] > 0
        )
        if pk:
            unique_sets.append(pk)
        schema[table] = {
            "columns": columns,
            "primary_key": pk,
            "unique_sets": unique_sets,
            "foreign_keys": con.execute(f'PRAGMA foreign_key_list("{table}")').fetchall(),
        }
    return schema


def _is_unique(entry: dict[str, Any], cols: tuple[str, ...]) -> bool:
    target = tuple(c.lower() for c in cols)
    return any(tuple(c.lower() for c in candidate) == target for candidate in entry["unique_sets"])


def _resolve(name: str, schema: dict[str, dict[str, Any]]) -> str | None:
    """Case-insensitive table lookup - SQL identifiers are not case sensitive here."""
    if name in schema:
        return name
    lowered = name.lower()
    for candidate in schema:
        if candidate.lower() == lowered:
            return candidate
    return None


def _resolve_column(name: str, entry: dict[str, Any]) -> str | None:
    if name in entry["columns"]:
        return name
    lowered = name.lower()
    for candidate in entry["columns"]:
        if candidate.lower() == lowered:
            return candidate
    return None


def _check_foreign_keys(
    schema: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    """Resolve every foreign key statically. Returns (relationships, errors)."""
    relationships: list[dict[str, Any]] = []
    errors: list[str] = []
    for table, entry in schema.items():
        # foreign_key_list rows are (id, seq, parent_table, from, to, on_update, on_delete, match);
        # a composite key spans several rows sharing an id.
        by_id: dict[int, list[tuple]] = {}
        for row in entry["foreign_keys"]:
            by_id.setdefault(int(row[0]), []).append(row)
        for fk_id, rows in sorted(by_id.items()):
            rows.sort(key=lambda r: int(r[1]))
            parent_name = rows[0][2]
            child_cols = tuple(r[3] for r in rows)
            parent = _resolve(parent_name, schema)
            if parent is None:
                errors.append(
                    f"{table}.{'/'.join(child_cols)} references table {parent_name!r}, "
                    "which the script never creates"
                )
                continue
            parent_entry = schema[parent]
            if rows[0][4] is None:
                # No explicit column list: SQLite resolves against the parent's primary key.
                parent_cols = parent_entry["primary_key"]
                if not parent_cols:
                    errors.append(
                        f"{table}.{'/'.join(child_cols)} references {parent} with no column "
                        "list, but {parent} has no primary key to resolve against".format(
                            parent=parent
                        )
                    )
                    continue
            else:
                declared = tuple(r[4] for r in rows)
                resolved = tuple(_resolve_column(c, parent_entry) or "" for c in declared)
                missing = [d for d, r in zip(declared, resolved, strict=True) if not r]
                if missing:
                    errors.append(
                        f"{table}.{'/'.join(child_cols)} references "
                        f"{parent}.{'/'.join(missing)}, which does not exist"
                    )
                    continue
                parent_cols = resolved
            if not _is_unique(parent_entry, parent_cols):
                errors.append(
                    f"{table}.{'/'.join(child_cols)} references "
                    f"{parent}.{'/'.join(parent_cols)}, which is neither the primary key nor "
                    "UNIQUE - SQLite raises 'foreign key mismatch' on the first row inserted"
                )
                continue
            child_meta = [entry["columns"][c] for c in child_cols if c in entry["columns"]]
            one_to_one = _is_unique(entry, child_cols)
            relationships.append(
                {
                    "child": table,
                    "child_columns": list(child_cols),
                    "parent": parent,
                    "parent_columns": list(parent_cols),
                    "cardinality": "one-to-one" if one_to_one else "one-to-many",
                    "optional": any(not m["not_null"] for m in child_meta),
                    "fk_id": fk_id,
                }
            )
    return relationships, errors


def _junction_tables(
    schema: dict[str, dict[str, Any]], relationships: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """A table whose primary key is exactly the union of two or more of its FK columns."""
    junctions: list[dict[str, Any]] = []
    for table, entry in schema.items():
        pk = {c.lower() for c in entry["primary_key"]}
        if len(pk) < 2:
            continue
        own = [r for r in relationships if r["child"] == table]
        fk_cols = {c.lower() for r in own for c in r["child_columns"]}
        if len(own) >= 2 and pk == fk_cols:
            junctions.append(
                {
                    "table": table,
                    "links": sorted({r["parent"] for r in own}),
                    "cardinality": "many-to-many",
                }
            )
    return junctions


def _check_types(schema: dict[str, dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    for table, entry in schema.items():
        for column, meta in entry["columns"].items():
            declared = meta["type"].strip()
            if not declared:
                continue  # a type-less column is legal SQLite, and not a hallucination
            if _normalise_type(declared) not in KNOWN_TYPES:
                errors.append(
                    f"{table}.{column} is declared {declared!r}, which is not a SQL type - "
                    "SQLite stores the name verbatim and gives the column BLOB affinity"
                )
    return errors


def check(
    ddl: str,
    *,
    dialect: str = "sqlite",
    strict_types: bool = True,
    require_tables: bool = True,
) -> dict[str, Any]:
    """Execute ``ddl`` and verify its foreign keys, cardinality and declared types.

    Always returns ``{"ok", "kind", "detail"}``; on anything past the execution stage it also
    carries ``tables``, ``relationships``, ``junction_tables``, ``errors`` and ``elapsed_ms``.

    ``dialect="postgres"`` is refused with ``ok=False``, ``kind="sql.unavailable"`` unless a
    driver appears - see the module docstring. Only SQLite results are ever reported as passes.
    """
    started = time.perf_counter()
    if dialect not in ("sqlite", "postgres"):
        raise ValueError(f"dialect must be 'sqlite' or 'postgres', not {dialect!r}")
    if dialect == "postgres":
        probe = postgres_available()
        if not probe["available"]:
            return _result(
                False,
                KIND_UNAVAILABLE,
                "no Postgres driver (psycopg/psycopg2) and no psql on PATH; this environment "
                "can only report SQLite results, and will not silently substitute one",
                dialect="postgres",
                available=False,
                probe=probe,
                elapsed_ms=(time.perf_counter() - started) * 1000.0,
            )
        return _result(
            False,
            KIND_UNAVAILABLE,
            f"a Postgres driver ({probe['driver']}) is importable, but this module has never "
            "been exercised against a live server and will not claim an unmeasured pass",
            dialect="postgres",
            available=False,
            probe=probe,
            elapsed_ms=(time.perf_counter() - started) * 1000.0,
        )

    statements = split_statements(ddl)
    con = sqlite3.connect(":memory:")
    try:
        con.execute("PRAGMA foreign_keys = ON")
        for index, statement in enumerate(statements):
            try:
                con.execute(statement)
            except sqlite3.Error as exc:
                return _result(
                    False,
                    KIND_EXEC,
                    f"statement {index} failed: {type(exc).__name__}: {exc}",
                    dialect="sqlite",
                    statements=len(statements),
                    error_statement_index=index,
                    error_statement=statement[:400],
                    elapsed_ms=(time.perf_counter() - started) * 1000.0,
                )
        con.commit()
        schema = _read_schema(con)
        # Walks rows, so it finds no *violations* on a DDL-only script. It does however raise
        # `foreign key mismatch` while resolving an unresolvable parent, even with zero rows -
        # measured, see the docstring - so it is a partial schema check and its exception is
        # captured rather than allowed to escape as a crash.
        row_violations: list[tuple] = []
        mismatch: str | None = None
        try:
            row_violations = con.execute("PRAGMA foreign_key_check").fetchall()
        except sqlite3.Error as exc:
            mismatch = str(exc)
    finally:
        con.close()

    common = {
        "dialect": "sqlite",
        "statements": len(statements),
        "tables": sorted(schema),
    }
    if require_tables and not schema:
        return _result(
            False,
            KIND_EMPTY,
            f"{len(statements)} statement(s) executed cleanly but created no table; an empty "
            "schema is the most likely way for a broken generator to score a pass",
            **common,
            elapsed_ms=(time.perf_counter() - started) * 1000.0,
        )

    relationships, fk_errors = _check_foreign_keys(schema)
    if mismatch and not fk_errors:
        # The static pass gives a better message when it has one; this is the fallback.
        fk_errors.append(f"PRAGMA foreign_key_check could not resolve a parent key: {mismatch}")
    fk_errors += [
        f"row-level foreign key violation in {v[0]} (rowid {v[1]}) -> {v[2]}"
        for v in row_violations
    ]
    junctions = _junction_tables(schema, relationships)
    card_errors = [
        f"{table} has no primary key, so nothing can reference it and its rows are not "
        "distinguishable"
        for table, entry in sorted(schema.items())
        if not entry["primary_key"] and not entry["unique_sets"]
    ]
    type_errors = _check_types(schema) if strict_types else []

    detail_parts = {
        KIND_FK: fk_errors,
        KIND_CARD: card_errors,
        KIND_TYPE: type_errors,
    }
    errors = fk_errors + card_errors + type_errors
    common.update(
        {
            "relationships": relationships,
            "junction_tables": junctions,
            "foreign_keys": len(relationships) + len(fk_errors),
            "errors": errors,
            "row_violations": len(row_violations),
            "elapsed_ms": (time.perf_counter() - started) * 1000.0,
        }
    )
    for kind, group in detail_parts.items():
        if group:
            return _result(
                False,
                kind,
                f"{len(errors)} problem(s), first: {group[0]}",
                **common,
            )
    counts = {"one-to-one": 0, "one-to-many": 0}
    for rel in relationships:
        counts[rel["cardinality"]] += 1
    return _result(
        True,
        KIND_OK,
        f"{len(statements)} statement(s), {len(schema)} table(s), "
        f"{counts['one-to-many']} one-to-many, {counts['one-to-one']} one-to-one, "
        f"{len(junctions)} junction table(s); all foreign keys resolve",
        **common,
    )


def check_many(ddls: list[str], **kwargs: Any) -> list[dict[str, Any]]:
    """``check`` over a batch. Each script gets its own connection - schemas must not leak."""
    return [check(ddl, **kwargs) for ddl in ddls]


REFERENCE_DDL = """
CREATE TABLE author (
    id        INTEGER PRIMARY KEY,
    email     VARCHAR(200) NOT NULL UNIQUE,
    name      TEXT NOT NULL
);
CREATE TABLE profile (
    author_id INTEGER PRIMARY KEY REFERENCES author(id),
    bio       TEXT
);
CREATE TABLE publisher (
    id        INTEGER PRIMARY KEY,
    name      TEXT NOT NULL UNIQUE
);
CREATE TABLE book (
    id           INTEGER PRIMARY KEY,
    isbn         CHAR(13) NOT NULL UNIQUE,
    title        TEXT NOT NULL,
    publisher_id INTEGER NOT NULL REFERENCES publisher(id),
    published_on DATE
);
CREATE TABLE tag (
    id   INTEGER PRIMARY KEY,
    slug TEXT NOT NULL UNIQUE
);
CREATE TABLE book_tag (
    book_id INTEGER NOT NULL REFERENCES book(id),
    tag_id  INTEGER NOT NULL REFERENCES tag(id),
    PRIMARY KEY (book_id, tag_id)
);
CREATE TABLE book_author (
    book_id   INTEGER NOT NULL REFERENCES book(id),
    author_id INTEGER NOT NULL REFERENCES author(id),
    position  SMALLINT NOT NULL DEFAULT 1,
    PRIMARY KEY (book_id, author_id)
);
CREATE TABLE review (
    id        INTEGER PRIMARY KEY,
    book_id   INTEGER NOT NULL REFERENCES book(id),
    author_id INTEGER REFERENCES author(id),
    stars     SMALLINT NOT NULL,
    body      TEXT
);
"""

#: Eleven scripts, each broken in exactly one way, used to prove the checker rejects.
#: The comment on each says whether SQLite's own execution catches it (it mostly does not).
BAD_DDL: dict[str, str] = {
    # --- invisible to execution; caught only by the static pass ---
    "dangling_parent_table": """
        CREATE TABLE post (
            id      INTEGER PRIMARY KEY,
            user_id INTEGER NOT NULL REFERENCES users(id)
        );
    """,
    "dangling_parent_column": """
        CREATE TABLE users (id INTEGER PRIMARY KEY, email TEXT NOT NULL UNIQUE);
        CREATE TABLE post (
            id      INTEGER PRIMARY KEY,
            user_id INTEGER NOT NULL REFERENCES users(uid)
        );
    """,
    "non_unique_fk_parent": """
        CREATE TABLE users (id INTEGER PRIMARY KEY, dept TEXT);
        CREATE TABLE post (
            id   INTEGER PRIMARY KEY,
            dept TEXT NOT NULL REFERENCES users(dept)
        );
    """,
    "parent_without_primary_key": """
        CREATE TABLE users (name TEXT, email TEXT);
        CREATE TABLE post (
            id      INTEGER PRIMARY KEY,
            user_id INTEGER NOT NULL REFERENCES users
        );
    """,
    "bogus_column_type": """
        CREATE TABLE users (id INTEGER PRIMARY KEY, email TEXTT NOT NULL UNIQUE);
    """,
    "hallucinated_type": """
        CREATE TABLE users (id INTEGER PRIMARY KEY, joined BANANA);
    """,
    "table_without_key": """
        CREATE TABLE audit_log (message TEXT, at DATETIME);
    """,
    "empty_schema": """
        -- the model emitted only a comment
        PRAGMA journal_mode = WAL;
    """,
    # --- caught by execution itself ---
    "duplicate_table": """
        CREATE TABLE users (id INTEGER PRIMARY KEY);
        CREATE TABLE users (id INTEGER PRIMARY KEY, email TEXT);
    """,
    "duplicate_column": """
        CREATE TABLE users (id INTEGER PRIMARY KEY, email TEXT, email TEXT);
    """,
    "syntax_error": """
        CREATE TABLE users (id INTEGER PRIMARY KEY,,, email TEXT);
    """,
}


def _benchmark(iterations: int = 1000) -> None:  # pragma: no cover - reporting helper
    import statistics

    result = check(REFERENCE_DDL)
    print(f"reference schema: ok={result['ok']} {result['detail']}")
    samples = []
    for _ in range(iterations):
        start = time.perf_counter()
        check(REFERENCE_DDL)
        samples.append((time.perf_counter() - start) * 1000.0)
    print(
        f"per call over {iterations}: mean {statistics.mean(samples):.2f} ms, "
        f"median {statistics.median(samples):.2f} ms, max {max(samples):.2f} ms"
    )
    start = time.perf_counter()
    for _ in range(iterations):
        con = sqlite3.connect(":memory:")
        con.execute("PRAGMA foreign_keys = ON")
        con.close()
    print(f"of which connection setup: {(time.perf_counter() - start) / iterations * 1000:.2f} ms")
    print(f"postgres probe: {postgres_available()}")
    for name, ddl in BAD_DDL.items():
        outcome = check(ddl)
        print(f"  {'REJECTED' if not outcome['ok'] else 'PASSED  '} {name:26s} {outcome['kind']}")


if __name__ == "__main__":  # pragma: no cover
    import argparse

    parser = argparse.ArgumentParser(description="Phase 12.3.7 SQL DDL checker")
    parser.add_argument("--benchmark", action="store_true")
    args = parser.parse_args()
    if args.benchmark:
        _benchmark()
    else:
        print(check(REFERENCE_DDL)["detail"])
