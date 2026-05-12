# athctl

Test-harness CLI for the Athletly backend. Bootstraps a throwaway Supabase user that reuses Roman's Garmin tokens, drives chat sessions, snapshots state, and validates agent behavior end-to-end.

## What it is

`athctl` is a single-tenant developer tool. It lives next to the backend and imports backend modules directly (`src.services.providers`, `src.db.client`, `src.config`) so the same code paths that production hits are exercised under test.

Source user (the Garmin token donor) is hardcoded to Roman's real account: `d511a8ce-9a35-476e-bbc1-840f3e7ee81e`. The harness is not designed for multi-tenant use.

## Install

From the backend repo root:

```
uv pip install -e tools/athctl/
```

This registers the `athctl` console script. Because the install is editable, code changes under `tools/athctl/athctl/` take effect immediately.

The CLI imports the backend's `src` package, so install it inside the same Python env that has the backend deps (`uv sync` first if you have not).

## Usage

```
athctl --help
athctl init --help
athctl init
athctl --json init
```

### Global flags

- `--json`  emit machine-readable JSON instead of human output
- `--quiet` suppress progress chatter (errors still go to stderr)

### Commands implemented

- `init` - create a fresh Supabase auth user, clone Garmin tokens, run initial sync (activities 180d, daily 30d, sleep 30d), optionally import Strava bootstrap tokens, write `~/.athctl/state.json`.

### Commands stubbed

These print "not yet implemented (Task #N)" today; they are part of the planned surface area so scripts can target stable paths:

- `provider list | status`
- `garmin sync` / `strava sync` (shortcuts)
- `state show`
- `chat <message>`
- `reset`, `trigger`, `snapshot`, `trace`, `validate`

## State file

`~/.athctl/state.json` is the single source of truth for "which test user is active". Schema:

```
{
  "active_user_id": "<uuid>",
  "active_user_email": "claude-test+<ts>@athletly.dev",
  "last_session_id": null,
  "created_at": "<iso8601>"
}
```

Hand-editing is fine; it gets rewritten on the next `athctl init`.

## Environment

Reads the backend's `.env` automatically. Required:

- `SUPABASE_URL`
- `SUPABASE_SERVICE_ROLE_KEY` (admin user creation, RLS bypass)

Optional:

- `STRAVA_TEST_ACCESS_TOKEN` and `STRAVA_TEST_REFRESH_TOKEN` for the Strava bootstrap step. Missing tokens skip Strava without failing init.
- `ATHLETLY_BACKEND_ROOT` to point at a non-default backend root (otherwise athctl walks up from cwd).

## Exit codes

| Code | Name              | Meaning                                          |
|------|-------------------|--------------------------------------------------|
| 0    | EXIT_OK           | Success                                          |
| 2    | EXIT_AUTH         | Supabase admin or Garmin token operation failed  |
| 3    | EXIT_NETWORK      | Garmin/Strava call failed mid-flight             |
| 4    | EXIT_STATE        | No active test user (run `athctl init`)          |
| 5    | EXIT_BACKEND_DOWN | Supabase write failed unexpectedly               |

## Limitations / TODO

- `--reuse-existing` flag on `init` is accepted but currently ignored (always creates a new user).
- The other subgroups are stubs; see the task list in the backend repo for the roadmap.
