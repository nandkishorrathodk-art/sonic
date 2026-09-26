# SONIC Configuration Reference

Configuration is managed via `sonic-core/sonic/config.py` (`Settings`) and environment variables.

## Core Environment Variables

| Variable | Default | Description |
|---|---|---|
| `APP_ENV` | `production` | Runtime environment (`development` \| `staging` \| `production`) |
| `JWT_SECRET` | *(required)* | Secret key for signing bearer tokens |
| `SONIC_MEMORY_DB_PATH` | `./sonic_data.db` | Persistent SQLite store for Graph, Vector, and Evolution memory |
| `SONIC_BEING_DB_PATH` | `./sonic_data.db` | Persistent SQLite store for `Being`, `BeingMind`, and `BeingVault` |
| `SONIC_WORKSTATION_STATE_PATH` | `sonic_data/workstations.json` | Persistent workspace attachment index |
| `SONIC_ENABLE_BEING_LIFE_LOOP` | `0` | Enable always-on autonomous `BeingLifeLoop` (`1` to enable) |
| `SONIC_SELF_CODER_ENABLED` | `true` | Enable the independent periodic self-coder loop; disabling it stops automatic code improvement |
| `SONIC_SELF_CODER_INTERVAL_SECONDS` | `3600` | Seconds between background self-coder runs (`60`–`86400`) |
| `SONIC_SELF_CODER_TENANT` | `SONIC_BEING_TENANT` or `default` | Tenant namespace for background candidate reports |
| `SONIC_SELF_CODER_EVERY_TICKS` | `20` | Cadence for optional direct `BeingLifeLoop` self-coder integrations (`1`–`10000`); production background self-coding uses the interval above |
| `SONIC_SELF_CODER_CANDIDATE_DIR` | `sonic_data/self_coder_candidates` | Directory for tenant-scoped candidate records and temporary Git worktrees |
| `NEXT_PUBLIC_API_URL` | `http://127.0.0.1:12000` | Backend URL used by `sonic-dashboard` |

## Self-Coder Autonomous Workflow

- The periodic self-coder runs independently of `SONIC_ENABLE_BEING_LIFE_LOOP`; it is enabled by default and can be stopped with `SONIC_SELF_CODER_ENABLED=false`. It runs immediately at backend startup and then at `SONIC_SELF_CODER_INTERVAL_SECONDS`.
- Candidates are created and tested in detached worktrees from the committed `HEAD`; uncommitted active-checkout changes are left untouched and listed in the report as excluded. A passing candidate is automatically revalidated and committed to `sonic/self-coder/<candidate_id>`. The active branch is not changed and nothing is pushed remotely.
- Discovery uses real test failures and read-only code-audit findings. Audit findings are hypotheses: the tool-using coding agent must inspect the source, establish a diagnosis, and the candidate must pass import, full test, and security-regression checks before its branch commit is created.
- Every run stores the diagnosis, expected impact (explicitly not a production measurement), changed files/diff, test and security receipts, commit hash/branch, and errors in candidate records. Inspect `GET /self-coder/status`, run immediately with `POST /self-coder/run`, and list/read history under `/self-coder/candidates`.
- `POST /self-coder/candidates/{candidate_id}/approve` remains for older verified records. New candidates do not wait for human approval. Reject/delete endpoints remain available for uncommitted historical candidate records.

## Dynamic Application & Target Policy

- **Dynamic Binaries**: Applications are discovered from `$PATH` and POSIX entries (`x-www-browser`, `sensible-browser`, `xdg-open`) without vendor binary hardcoding.
- **Target Scope**: Test and documentation targets use RFC 2606 `.test` domains (`https://app.target.test`, `https://git.target.test/org/repo`) and RFC 5737 documentation network addresses (`198.51.100.1`).
