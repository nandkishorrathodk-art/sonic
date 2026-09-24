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
| `NEXT_PUBLIC_API_URL` | `http://127.0.0.1:12000` | Backend URL used by `sonic-dashboard` |

## Dynamic Application & Target Policy

- **Dynamic Binaries**: Applications are discovered from `$PATH` and POSIX entries (`x-www-browser`, `sensible-browser`, `xdg-open`) without vendor binary hardcoding.
- **Target Scope**: Test and documentation targets use RFC 2606 `.test` domains (`https://app.target.test`, `https://git.target.test/org/repo`) and RFC 5737 documentation network addresses (`198.51.100.1`).

