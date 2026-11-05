#!/usr/bin/env python3
"""Local PersonaNet session notes in Postgres, using the existing Docker container.

Examples:
  uv run python scripts/12_session_log.py --init
  uv run python scripts/12_session_log.py --log .local/session-notes.json
  uv run python scripts/12_session_log.py --list
  uv run python scripts/12_session_log.py --add "Finding" --body "What changed and why"
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import subprocess
import uuid

SCHEMA = """
CREATE TABLE IF NOT EXISTS projects (
    slug text PRIMARY KEY,
    name text NOT NULL,
    description text NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS sessions (
    id text PRIMARY KEY,
    project_slug text NOT NULL REFERENCES projects(slug),
    title text NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS session_entries (
    id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    session_id text NOT NULL REFERENCES sessions(id),
    entry_key text NOT NULL,
    kind text NOT NULL,
    title text NOT NULL,
    body text NOT NULL,
    metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE (session_id, entry_key)
);
"""


def literal(value: str) -> str:
    """SQL text literal; connections explicitly enable standard-conforming strings."""
    if "\x00" in value:
        raise ValueError("Postgres text cannot contain a NUL character")
    return "'" + value.replace("'", "''") + "'"


def query(container: str, database: str, sql: str) -> str:
    result = subprocess.run(
        ["docker", "exec", "-i", container, "psql", "-X", "-qAt",
         "-U", "postgres", "-d", database, "--set", "ON_ERROR_STOP=1"],
        input="SET standard_conforming_strings = on;\n" + sql,
        text=True, capture_output=True,
    )
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or "Database command failed")
    return result.stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--container", default="supabase_db_pixis-finops")
    parser.add_argument("--database", default="personanet")
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--init", action="store_true")
    action.add_argument("--log", type=Path, help="Import a JSON session with stable entry keys")
    action.add_argument("--list", action="store_true")
    action.add_argument("--add", metavar="TITLE", help="Append one note")
    parser.add_argument("--body", default="")
    parser.add_argument("--kind", default="note")
    parser.add_argument("--session", default=datetime.now(timezone.utc).strftime("%Y-%m-%d") + "-linkedin-demo")
    args = parser.parse_args()
    if not re.fullmatch(r"[a-z][a-z0-9_]*", args.database):
        parser.error("--database must be a lowercase SQL identifier")

    def execute(sql: str, database: str = args.database) -> str:
        return query(args.container, database, sql)

    if args.init:
        exists = execute(f"SELECT 1 FROM pg_database WHERE datname = {literal(args.database)};", "postgres")
        if exists != "1":
            execute(f'CREATE DATABASE "{args.database}";', "postgres")
        execute("BEGIN;\n" + SCHEMA + """
INSERT INTO projects (slug, name, description)
VALUES ('personanet', 'PersonaNet', 'Frozen LLM personality-head research, demo decisions, and session findings')
ON CONFLICT (slug) DO NOTHING;
COMMIT;
""")
        print(f"PersonaNet project ready in {args.database} ({args.container}).")
        return

    if args.list:
        print(execute("""
SELECT coalesce(jsonb_pretty(jsonb_agg(to_jsonb(notes) ORDER BY notes.id)), '[]')
FROM (
    SELECT e.id, s.project_slug, e.session_id, e.kind, e.title, e.body,
           e.metadata, e.created_at
    FROM session_entries e JOIN sessions s ON s.id = e.session_id
    WHERE s.project_slug = 'personanet'
) notes;
"""))
        return

    if args.log:
        session = json.loads(args.log.read_text())
    else:
        if not args.body.strip():
            parser.error("--add requires a nonempty --body")
        session = {
            "session_id": args.session, "title": "PersonaNet demo development",
            "entries": [{"key": str(uuid.uuid4()), "kind": args.kind,
                         "title": args.add, "body": args.body}],
        }
    session_id = literal(session["session_id"])
    statements = ["BEGIN;", f"""
INSERT INTO sessions (id, project_slug, title)
VALUES ({session_id}, 'personanet', {literal(session['title'])})
ON CONFLICT (id) DO NOTHING;
"""]
    for entry in session["entries"]:
        values = ", ".join(literal(value) for value in (
            entry["key"], entry["kind"], entry["title"], entry["body"],
            json.dumps(entry.get("metadata", {})),
        ))
        statements.append(f"""
INSERT INTO session_entries (session_id, entry_key, kind, title, body, metadata)
VALUES ({session_id}, {values})
ON CONFLICT (session_id, entry_key) DO NOTHING;
""")
    statements.extend(["COMMIT;", f"SELECT count(*) FROM session_entries WHERE session_id = {session_id};"])
    count = execute("\n".join(statements))
    print(f"Session {session['session_id']}: {count} stored entries (duplicate import keys skipped).")


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, OSError, ValueError, KeyError) as exc:
        raise SystemExit(f"Session logging failed: {exc}") from exc
