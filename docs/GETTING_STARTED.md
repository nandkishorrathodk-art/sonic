# Getting Started with SONIC

SONIC is an **Autonomous AI-Human Digital Being** operating its own dedicated personal computer workstation (`DISPLAY=:99` graphical desktop + dedicated headless terminal).

## 1. Prerequisites

- Python 3.12+ (`pip install -e sonic-core[dev]`)
- Node.js 20+ (`cd sonic-dashboard && npm install`)
- Docker Desktop / Engine (for isolated `sonic-desktop-workstation` container execution)

## 2. Starting the Backend (`sonic-core`)

```bash
APP_ENV=development python -m uvicorn sonic.api.main:app --port 12000
```

Verify health:
```bash
curl http://127.0.0.1:12000/health
```

## 3. Starting the Workstation Dashboard (`sonic-dashboard`)

```bash
cd sonic-dashboard
NEXT_PUBLIC_API_URL=http://127.0.0.1:12000 npx next dev -p 12001
```

## 4. Dual-Plane Workstation Usage

- **Workstation Application Plane (`DISPLAY=:99`)**: Launches and interacts with any OS-installed application dynamically via POSIX alternatives (`x-www-browser`, `sensible-browser`, `xdg-open`, and `$PATH` binaries).
- **Operator & Sandbox Plane**: Executes terminal, file, git, and subagent branch workflows (`https://app.target.test`, `https://git.target.test/org/repo`).

