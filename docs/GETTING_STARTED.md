# Getting started

This guide starts SONIC for local development. It does not certify the local
environment as production-safe or prove that a live desktop task can be
completed.

## Requirements

- Python 3.12+
- Node.js and npm for the dashboard
- Docker Engine or Docker Desktop for the local workstation and optional services
- A model-provider credential configured in `configs/models.yaml` if you need
  model-backed reasoning

## Configure and start the local workstation

Copy the environment template and set a strong, local-only VNC password:

```powershell
Copy-Item .env.example .env
```

Set `SONIC_VNC_PASSWORD` in `.env`, and add the credentials for the provider
you intend to use. Do not commit `.env`. The Neo4j credentials in the local
compose example are for development only.

Start compose from the repository root:

```powershell
docker compose up -d
```

The compose file starts the workstation and supporting services; it does not
start the FastAPI API or dashboard. VNC/noVNC ports are bound to localhost.
The background self-coder is enabled by default and may create candidate
commits on its own branch when a model is available. For an ordinary local
development session, explicitly disable it in the API shell as shown below.

## Start the backend

From the repository root:

```powershell
python -m pip install -e "sonic-core[dev]"
$env:APP_ENV = "development"
$env:SONIC_SELF_CODER_ENABLED = "false"
$env:CORS_ORIGINS = "http://127.0.0.1:12001,http://localhost:12001"
python -m uvicorn sonic.api.main:app --host 127.0.0.1 --port 12000
```

Check the API at `http://127.0.0.1:12000/health`; interactive API docs are at
`http://127.0.0.1:12000/docs`.

## Start the dashboard

In a separate PowerShell window:

```powershell
Set-Location sonic-dashboard
npm install
$env:NEXT_PUBLIC_API_URL = "http://127.0.0.1:12000"
npm run dev
```

Open `http://127.0.0.1:12001`. The dashboard needs the backend URL configured
for the local environment if it differs from its default.

## Run tests

From the repository root:

```powershell
python -m pytest sonic-core/tests/
```

Some tests require live Docker or a real workstation. Read the reported skips
and failures; a test-suite pass alone does not establish real multi-application
GUI acceptance. See [the product contract](SONIC_COMPUTER_USER_GOAL.md).
