# PersonaNet development sessions

## Persistent session context

The user requested that project conversations, decisions, and findings be logged
in the local database so future sessions can build on them.

- At the start of development, read the ledger with
  `uv run python scripts/12_session_log.py --list` when Docker is available.
- After meaningful changes or experiments, append a concise note with
  `scripts/12_session_log.py --session <session-id> --kind <kind> --add <title> --body <summary>`.
  Record the request, decision, relevant settings, verification results, and
  unresolved issues. Label hypotheses and unverified claims accordingly.
- The current demo session is `2026-11-05-linkedin-demo`. Use a new descriptive
  session ID for later work. Do not copy secrets or unrelated project data into notes.
- If the database is unavailable, keep a gitignored draft under `.local/` and
  report the logging blocker. Database access should not block unrelated work.
- See `docs/SESSION_LOG.md` for the schema, import format, and connection details.

## Demo verification

Run `uv run python -m unittest discover -s tests -v` for demo regression checks.
Keep live demonstrations distinct from the recorded results in `docs/REPORT.md`.
