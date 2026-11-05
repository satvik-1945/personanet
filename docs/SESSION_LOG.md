# Local project and session notes

PersonaNet keeps development decisions and findings in a dedicated **`personanet`**
Postgres database on the existing local Docker container
`supabase_db_pixis-finops`. The `personanet` project record groups sessions and
their timestamped entries. Tables: `projects`, `sessions`, `session_entries`.

The CLI uses `docker exec` and the container's local Postgres administrator;
no database credentials are written into this repository. Docker must be running.

```bash
# Create the dedicated database, tables, and project (safe to rerun)
uv run python scripts/12_session_log.py --init

# Read the saved entries
uv run python scripts/12_session_log.py --list

# Append a finding from this session
uv run python scripts/12_session_log.py \
  --session 2026-11-05-linkedin-demo \
  --kind finding \
  --add "Experiment finding" \
  --body "Describe the observation, settings, and next step."
```

`--container` and `--database` override the local defaults. Without `--session`,
new notes use the current UTC date followed by `-linkedin-demo`.

## Import a session

Local drafts live under gitignored `.local/`. The initial session summary is
`.local/session-notes.json` and can be imported with:

```bash
uv run python scripts/12_session_log.py --log .local/session-notes.json
```

JSON shape:

```json
{
  "session_id": "2026-11-05-linkedin-demo",
  "title": "Demo development",
  "entries": [
    {
      "key": "unique-finding-key",
      "kind": "finding",
      "title": "Short title",
      "body": "Observation and supporting context",
      "metadata": {"command": "optional verification command"}
    }
  ]
}
```

Imports are transactional and skip duplicate `(session_id, entry_key)` pairs.
Existing notes remain intact; corrections should be appended as new entries.
Timestamps represent insertion time, not exact original chat-message time.

This is an explicit development-session ledger, not a background chat recorder
or automatic model training pipeline. Future sessions can read the ledger at
the start and append decisions and verified results as work proceeds. The
repository's `AGENTS.md` instructs coding sessions to follow that workflow. The
database persists with the existing container's data volume; deleting that
volume would remove the notes, so include it in your normal database backups.
