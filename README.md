# Tiny Sonic

Tiny Sonic is the minimal local control plane extracted from SONIC.

It provides:

- a small FastAPI backend;
- a safe health endpoint;
- a simple conversational endpoint;
- no database, cloud provider, or desktop dependency.

## Run

```powershell
python -m pip install -r requirements.txt
python -m uvicorn app:app --host 127.0.0.1 --port 12000
```

Then open `http://127.0.0.1:12000/docs`.

The full SONIC codebase remains on its original branch.
