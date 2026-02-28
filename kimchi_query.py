"""SQLite query and semantic helpers for Kimchi."""

from __future__ import annotations

import json
import math
import sqlite3
from pathlib import Path

from kimchi_embed import embed_texts


def cell_path(home: Path, cell: str = "codex_code") -> Path:
    return home / "cells" / f"{cell}.db"


def open_db(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=30000")
    return conn


def run_sql_readonly(conn: sqlite3.Connection, sql: str) -> list[dict]:
    q = sql.strip()
    if not q:
        return []

    head = q.split(None, 1)[0].upper()
    if head not in {"SELECT", "WITH", "PRAGMA", "EXPLAIN"}:
        raise RuntimeError("Read-only SQL only")

    rows = conn.execute(q).fetchall()
    return [dict(row) for row in rows]


def inspect_schema(conn: sqlite3.Connection) -> dict:
    tables = [
        row[0]
        for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name").fetchall()
    ]
    views = [
        row[0]
        for row in conn.execute("SELECT name FROM sqlite_master WHERE type='view' ORDER BY name").fetchall()
    ]
    presets = []
    if "quick_queries" in tables:
        presets = [row[0] for row in conn.execute("SELECT query_key FROM quick_queries ORDER BY query_key").fetchall()]
    return {"tables": tables, "views": views, "presets": presets}


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na <= 1e-12 or nb <= 1e-12:
        return 0.0
    return dot / (na * nb)


def semantic_search(conn: sqlite3.Connection, query: str, limit: int = 10) -> list[dict]:
    qvec = embed_texts([query], prefix="search_query: ")[0]
    rows = conn.execute(
        """
        SELECT card_key, text_body, vec_blob, born_unix
        FROM pantry_cards
        WHERE vec_blob IS NOT NULL
        """
    ).fetchall()

    scored: list[dict] = []
    for row in rows:
        try:
            dvec = json.loads(row["vec_blob"])
            score = _cosine(qvec, dvec)
        except Exception:
            continue
        scored.append(
            {
                "id": row["card_key"],
                "content": row["text_body"],
                "timestamp": row["born_unix"],
                "score": score,
            }
        )

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:limit]


def _file_history_rows(conn: sqlite3.Connection, file_query: str) -> list[sqlite3.Row]:
    pattern = f"%{file_query.strip()}%"
    if not file_query.strip():
        return []
    return conn.execute(
        """
        WITH target_runs AS (
            SELECT DISTINCT rc.run_key
            FROM op_notes op
            JOIN run_cards rc ON rc.card_key = op.card_key
            WHERE LOWER(COALESCE(op.file_path_hint, '')) LIKE LOWER(?)
        )
        SELECT
            pc.card_key,
            pc.text_body,
            pc.vec_blob,
            pc.born_unix,
            rc.run_key,
            rc.turn_no,
            cm.mark_kind,
            cm.actor_role,
            op.op_code,
            op.file_path_hint,
            op.pass_flag
        FROM run_cards rc
        JOIN pantry_cards pc ON pc.card_key = rc.card_key
        LEFT JOIN card_marks cm ON cm.card_key = rc.card_key
        LEFT JOIN op_notes op ON op.card_key = rc.card_key
        WHERE rc.run_key IN (SELECT run_key FROM target_runs)
        ORDER BY pc.born_unix DESC, rc.turn_no DESC
        """,
        (pattern,),
    ).fetchall()


def _history_item(row: sqlite3.Row) -> dict:
    return {
        "id": row["card_key"],
        "run_key": row["run_key"],
        "turn_no": row["turn_no"],
        "kind": row["mark_kind"],
        "role": row["actor_role"],
        "op_code": row["op_code"],
        "file_path_hint": row["file_path_hint"],
        "pass_flag": row["pass_flag"],
        "timestamp": row["born_unix"],
        "content": row["text_body"],
    }


def file_history_search(conn: sqlite3.Connection, file_query: str, limit: int = 20, semantic_query: str = "") -> list[dict]:
    """Return history cards from sessions that touched a specific file path."""
    rows = _file_history_rows(conn, file_query)
    if not rows:
        return []

    query = semantic_query.strip()
    if not query:
        return [_history_item(row) for row in rows[:limit]]

    qvec = embed_texts([query], prefix="search_query: ")[0]
    scored: list[dict] = []
    for row in rows:
        try:
            vec = json.loads(row["vec_blob"]) if row["vec_blob"] else None
            score = _cosine(qvec, vec) if isinstance(vec, list) and vec else 0.0
        except Exception:
            score = 0.0
        item = _history_item(row)
        item["score"] = score
        scored.append(item)

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:limit]


def schema_similarity_search(conn: sqlite3.Connection, query: str, limit: int = 10) -> list[dict]:
    """Search schema objects (table/view definitions) by semantic similarity."""
    rows = conn.execute(
        """
        SELECT type, name, sql
        FROM sqlite_master
        WHERE type IN ('table', 'view')
          AND name NOT LIKE 'sqlite_%'
        ORDER BY type, name
        """
    ).fetchall()

    objects: list[dict] = []
    texts: list[str] = []
    for row in rows:
        obj_type = str(row["type"])
        name = str(row["name"])
        sql = str(row["sql"] or "").strip()
        if not sql and obj_type == "table":
            # Fallback textual schema from columns when CREATE SQL isn't available.
            safe_name = name.replace("'", "''")
            cols = conn.execute(f"PRAGMA table_info('{safe_name}')").fetchall()
            col_text = ", ".join(f"{c['name']} {c['type']}" for c in cols) if cols else ""
            sql = f"{name}({col_text})"

        text = f"{obj_type} {name}\n{sql}".strip()
        objects.append({"type": obj_type, "name": name, "sql": sql})
        texts.append(text)

    if not objects:
        return []

    qvec = embed_texts([query], prefix="search_query: ")[0]
    dvecs = embed_texts(texts, prefix="search_document: ")

    scored: list[dict] = []
    for obj, vec in zip(objects, dvecs):
        scored.append(
            {
                "type": obj["type"],
                "name": obj["name"],
                "sql": obj["sql"],
                "score": _cosine(qvec, vec),
            }
        )

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:limit]
