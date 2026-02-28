#!/usr/bin/env python3
"""Kimchi: codex-only local memory engine (all app logic lives in 3 files)."""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import time
from pathlib import Path

from kimchi_embed import embed_texts
from kimchi_query import (
    cell_path,
    inspect_schema,
    open_db,
    run_sql_readonly,
    schema_similarity_search,
    semantic_search,
)

KIMCHI_HOME = Path(os.environ.get("KIMCHI_HOME", Path.home() / ".kimchi"))
DEFAULT_CODEX_SESSION_ROOT = Path.home() / ".codex" / "sessions"


def ensure_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS pantry_cards (
            card_key TEXT PRIMARY KEY,
            text_body TEXT NOT NULL,
            vec_blob TEXT,
            born_unix INTEGER
        );

        CREATE TABLE IF NOT EXISTS batch_runs (
            run_key TEXT PRIMARY KEY,
            source_uri TEXT,
            run_label TEXT,
            started_unix INTEGER,
            finished_unix INTEGER,
            item_count INTEGER DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS run_cards (
            card_key TEXT NOT NULL,
            run_key TEXT NOT NULL,
            lane_tag TEXT DEFAULT 'codex-log',
            turn_no INTEGER,
            PRIMARY KEY (card_key, run_key)
        );

        CREATE TABLE IF NOT EXISTS card_marks (
            card_key TEXT PRIMARY KEY,
            mark_kind TEXT,
            actor_role TEXT,
            turn_index INTEGER,
            trace_ref TEXT
        );

        CREATE TABLE IF NOT EXISTS op_notes (
            card_key TEXT PRIMARY KEY,
            op_code TEXT,
            file_path_hint TEXT,
            pass_flag INTEGER
        );

        CREATE TABLE IF NOT EXISTS quick_queries (
            query_key TEXT PRIMARY KEY,
            query_note TEXT,
            query_sql TEXT
        );

        INSERT OR IGNORE INTO quick_queries(query_key, query_note, query_sql)
        VALUES ('search', 'Schema inventory', 'SELECT name, type FROM sqlite_master ORDER BY type, name');
        """
    )


def _safe_mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except Exception:
        return 0.0


def _iter_session_files(session_root: Path, max_sessions: int = 0, session_id: str = "") -> list[Path]:
    if not session_root.exists():
        return []

    sid = session_id.strip()
    if sid:
        direct = session_root / f"{sid}.json"
        if direct.exists():
            files = [direct]
        else:
            # Best-effort fuzzy lookup when session id is partial/normalized differently.
            files = sorted(session_root.rglob(f"*{sid}*.json"), key=_safe_mtime, reverse=True)
    else:
        files = sorted(session_root.rglob("*.json"), key=_safe_mtime, reverse=True)

    if max_sessions and max_sessions > 0:
        return files[:max_sessions]
    return files


def _parse_timestamp(value: object) -> int:
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return int(time.time())


def _extract_text(content: object) -> str:
    if isinstance(content, str):
        return content.strip()

    if isinstance(content, dict):
        for key in ("text", "content", "result", "output", "summary"):
            value = content.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

    if isinstance(content, list):
        parts: list[str] = []
        for row in content:
            if isinstance(row, str) and row.strip():
                parts.append(row.strip())
            elif isinstance(row, dict):
                text = row.get("text") or row.get("content")
                if isinstance(text, str) and text.strip():
                    parts.append(text.strip())
        return "\n".join(parts).strip()

    return ""


def _session_id(path: Path, payload: dict) -> str:
    sid = ((payload.get("session") or {}).get("id") or "").strip()
    if sid:
        return sid
    return path.stem


def _parse_tool_args(value: object) -> dict:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, dict):
                return parsed
            return {"raw": value}
        except Exception:
            return {"raw": value}
    return {}


def index_codex_sessions(
    conn: sqlite3.Connection,
    session_root: Path,
    max_sessions: int = 0,
    session_id: str = "",
) -> dict:
    ensure_schema(conn)
    session_files = _iter_session_files(session_root, max_sessions=max_sessions, session_id=session_id)

    inserted_sessions = 0
    inserted_chunks = 0

    for path in session_files:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue

        if not isinstance(payload, dict):
            continue

        sid = _session_id(path, payload)
        items = payload.get("items")
        if not isinstance(items, list):
            continue

        conn.execute(
            "INSERT OR IGNORE INTO batch_runs(run_key, source_uri, run_label, started_unix, item_count) VALUES (?, ?, ?, ?, 0)",
            (sid, f"codex:{sid}", path.name, int(time.time())),
        )

        chunk_rows: list[tuple] = []
        texts_for_embedding: list[str] = []

        for idx, item in enumerate(items, start=1):
            if not isinstance(item, dict):
                continue

            item_type = item.get("type")
            role = str(item.get("role") or "assistant")
            ts = _parse_timestamp(item.get("timestamp"))
            card_key = f"{sid}_{idx}"

            chunk_type = ""
            text = ""
            tool_name = None
            target_file = None
            success = None

            if item_type == "message":
                text = _extract_text(item.get("content"))
                if not text:
                    continue
                chunk_type = "user_prompt" if role == "user" else "assistant_message"

            elif item_type == "reasoning":
                text = _extract_text(item.get("summary"))
                if not text:
                    continue
                chunk_type = "assistant_reasoning"

            elif item_type == "function_call":
                chunk_type = "tool_call"
                tool_name = str(item.get("name") or "unknown")
                args = _parse_tool_args(item.get("arguments"))
                target_file = args.get("file_path") or args.get("path") or args.get("target_file")
                text = json.dumps(args, ensure_ascii=False)
                status = str(item.get("status") or "").lower().strip()
                if status:
                    success = int(status in {"ok", "success", "completed"})

            elif item_type == "function_call_output":
                chunk_type = "tool_output"
                text = _extract_text(item.get("output"))
                if not text:
                    continue

            else:
                continue

            chunk_rows.append(
                (
                    card_key,
                    text,
                    ts,
                    sid,
                    idx,
                    chunk_type,
                    role,
                    item.get("id"),
                    tool_name,
                    target_file,
                    success,
                )
            )
            texts_for_embedding.append(text)

        if not chunk_rows:
            continue

        vectors = embed_texts(texts_for_embedding, prefix="search_document: ")

        for row, vector in zip(chunk_rows, vectors):
            (
                card_key,
                text,
                ts,
                sid,
                idx,
                chunk_type,
                role,
                entry_uuid,
                tool_name,
                target_file,
                success,
            ) = row

            conn.execute(
                "INSERT OR REPLACE INTO pantry_cards(card_key, text_body, vec_blob, born_unix) VALUES (?, ?, ?, ?)",
                (card_key, text, json.dumps(vector), ts),
            )
            conn.execute(
                "INSERT OR REPLACE INTO run_cards(card_key, run_key, turn_no) VALUES (?, ?, ?)",
                (card_key, sid, idx),
            )
            conn.execute(
                "INSERT OR REPLACE INTO card_marks(card_key, mark_kind, actor_role, turn_index, trace_ref) VALUES (?, ?, ?, ?, ?)",
                (card_key, chunk_type, role, idx, entry_uuid),
            )
            if tool_name:
                conn.execute(
                    "INSERT OR REPLACE INTO op_notes(card_key, op_code, file_path_hint, pass_flag) VALUES (?, ?, ?, ?)",
                    (card_key, tool_name, target_file, success),
                )
            inserted_chunks += 1

        conn.execute(
            "UPDATE batch_runs SET item_count = COALESCE(item_count, 0) + ?, finished_unix = ? WHERE run_key = ?",
            (len(chunk_rows), int(time.time()), sid),
        )
        inserted_sessions += 1

    conn.commit()
    return {
        "sessions": inserted_sessions,
        "chunks": inserted_chunks,
        "source_files": len(session_files),
        "session_root": str(session_root),
    }


def cmd_init(args: argparse.Namespace) -> None:
    if args.embed_url:
        os.environ["KIMCHI_EMBED_URL"] = args.embed_url

    home = Path(args.home).expanduser()
    session_root = Path(args.session_root).expanduser()
    db = cell_path(home, args.cell)

    conn = open_db(db)
    try:
        ensure_schema(conn)
        stats = index_codex_sessions(
            conn,
            session_root=session_root,
            max_sessions=args.max_sessions,
            session_id=args.session_id,
        )
    finally:
        conn.close()

    print(json.dumps({"home": str(home), "cell": str(db), "indexed": stats}, indent=2, ensure_ascii=False))


def cmd_index(args: argparse.Namespace) -> None:
    if args.embed_url:
        os.environ["KIMCHI_EMBED_URL"] = args.embed_url

    home = Path(args.home).expanduser()
    session_root = Path(args.session_root).expanduser()
    db = cell_path(home, args.cell)

    conn = open_db(db)
    try:
        ensure_schema(conn)
        stats = index_codex_sessions(
            conn,
            session_root=session_root,
            max_sessions=args.max_sessions,
            session_id=args.session_id,
        )
    finally:
        conn.close()

    print(json.dumps(stats, indent=2, ensure_ascii=False))


def cmd_search(args: argparse.Namespace) -> None:
    if args.embed_url:
        os.environ["KIMCHI_EMBED_URL"] = args.embed_url

    home = Path(args.home).expanduser()
    db = cell_path(home, args.cell)
    if not db.exists():
        raise SystemExit(f"Cell not found: {db}. Run `kimchi init` first.")

    conn = open_db(db)
    try:
        if args.schema_similar:
            result = schema_similarity_search(conn, args.schema_similar, limit=args.limit)
        elif args.semantic:
            result = semantic_search(conn, args.semantic, limit=args.limit)
        elif args.query.startswith("@"):
            preset_name = args.query[1:].strip()
            if preset_name == "search":
                result = inspect_schema(conn)
            else:
                row = conn.execute(
                    "SELECT query_sql FROM quick_queries WHERE query_key = ?",
                    (preset_name,),
                ).fetchone()
                if not row:
                    raise RuntimeError(f"Unknown preset: @{preset_name}")
                result = run_sql_readonly(conn, row["query_sql"])
        else:
            result = run_sql_readonly(conn, args.query)
    finally:
        conn.close()

    print(json.dumps(result, indent=2, ensure_ascii=False))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="kimchi", description="Kimchi codex memory engine")
    sub = parser.add_subparsers(dest="command")

    def add_common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--home", default=str(KIMCHI_HOME), help="Kimchi home directory")
        p.add_argument("--cell", default="codex_code", help="SQLite cell name")
        p.add_argument("--embed-url", default="", help="Embedding endpoint URL")
        p.add_argument("--session-root", default=str(DEFAULT_CODEX_SESSION_ROOT), help="Codex sessions root path")
        p.add_argument("--max-sessions", type=int, default=0, help="Index at most N session files (0 = all)")
        p.add_argument("--session-id", default="", help="Index only matching session id (exact or fuzzy)")

    p_init = sub.add_parser("init", help="Initialize schema and index Codex sessions")
    add_common(p_init)

    p_index = sub.add_parser("index", help="Re-index Codex sessions")
    add_common(p_index)

    p_search = sub.add_parser("search", help="Search data via SQL or semantic query")
    add_common(p_search)
    p_search.add_argument("query", nargs="?", default="@search", help="SQL query or @preset")
    p_search.add_argument("--semantic", default="", help="Semantic text query")
    p_search.add_argument("--schema-similar", default="", help="Find similar table/view schema definitions")
    p_search.add_argument("--limit", type=int, default=10, help="Semantic result limit")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "init":
        cmd_init(args)
    elif args.command == "index":
        cmd_index(args)
    elif args.command == "search":
        cmd_search(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
