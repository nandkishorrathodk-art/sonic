# SONIC Configuration Reference

Configuration is managed via `services/core/sonic/config.py` (`Settings`) and environment variables.

## Core Environment Variables

| Variable | Default | Description |
|---|---|---|
| `APP_ENV` | `production` | Runtime environment (`development` \| `staging` \| `production`) |
| `JWT_SECRET` | *(required)* | Secret key for signing bearer tokens |
| `SONIC_MEMORY_DB_PATH` | `./sonic_data.db` | Persistent SQLite path used by graph, vector, and selected Being stores |
| `SONIC_EMBEDDING_PROVIDER` | `legacy_hash` | `legacy_hash`, `openai_compatible`, or `sentence_transformers`; legacy mode is not semantic embedding |
| `SONIC_EMBEDDING_BASE_URL` | *(unset)* | Base URL for an OpenAI-compatible `/embeddings` endpoint; may be a local service |
| `SONIC_EMBEDDING_MODEL` | *(unset)* | Model name required when `SONIC_EMBEDDING_PROVIDER=openai_compatible` |
| `SONIC_EMBEDDING_API_KEY` | *(unset)* | Optional bearer token for an OpenAI-compatible embedding endpoint |
| `SONIC_EMBEDDING_TIMEOUT_SECONDS` | `30` | Request timeout for OpenAI-compatible embedding requests |
| `SONIC_EMBEDDING_LOCAL_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Model loaded when `SONIC_EMBEDDING_PROVIDER=sentence_transformers` |
| `SONIC_BEING_DB_PATH` | `./sonic_data.db` | Persistent SQLite store for `Being`, `BeingMind`, and `BeingVault` |
| `SONIC_BEING_VAULT_KEY` | *(required to use the vault)* | URL-safe base64 encoding of 32 random bytes; encrypts credentials, tokens, profile URLs, and saved session state |
| `SONIC_WORKSTATION_STATE_PATH` | `sonic_data/workstations.json` | Persistent workspace attachment index |
| `SONIC_ENABLE_BEING_LIFE_LOOP` | `0` | Enable the optional background `BeingLifeLoop` (`1` to enable; off by default) |
| `SONIC_SELF_CODER_ENABLED` | `true` | Enable the independent periodic self-coder loop; disabling it stops automatic code improvement |
| `SONIC_SELF_CODER_INTERVAL_SECONDS` | `3600` | Seconds between background self-coder runs (`60`–`86400`) |
| `SONIC_SELF_CODER_TENANT` | `SONIC_BEING_TENANT` or `default` | Tenant namespace for background candidate reports |
| `SONIC_SELF_CODER_EVERY_TICKS` | `20` | Cadence for optional direct `BeingLifeLoop` self-coder integrations (`1`–`10000`); production background self-coding uses the interval above |
| `SONIC_SELF_CODER_CANDIDATE_DIR` | `sonic_data/self_coder_candidates` | Directory for tenant-scoped candidate records and temporary Git worktrees |
| `NEXT_PUBLIC_API_URL` | `http://127.0.0.1:12000` | Backend URL used by `apps/dashboard` |

### Semantic retrieval embeddings

The existing `legacy_hash` mode keeps older installations working, but its
hashed n-gram vectors are not learned semantic embeddings. To use a local
OpenAI-compatible embedding server or hosted endpoint, configure:

```text
SONIC_EMBEDDING_PROVIDER=openai_compatible
SONIC_EMBEDDING_BASE_URL=http://localhost:11434/v1
SONIC_EMBEDDING_MODEL=<model-supported-by-the-endpoint>
```

Set `SONIC_EMBEDDING_API_KEY` only when the endpoint requires authentication.
Alternatively, install the optional `rag-local` extra and select
`SONIC_EMBEDDING_PROVIDER=sentence_transformers`; set
`SONIC_EMBEDDING_LOCAL_MODEL` to a locally available model identifier. The
Sentence Transformers model is loaded only when this provider is selected; the
model may need to be fetched if it is not cached locally. Hosted endpoints
receive the source text being embedded, so select a provider consistent with
your data-handling requirements.

Each row records its embedding model identity and vector dimension. Search never
compares vectors from different models or dimensions; it can still use lexical
overlap for rows not yet re-embedded. After selecting a new embedding model, call
`get_vector_memory().reembed_all()` to migrate existing rows in resumable
batches. Back up the SQLite database first. Provider errors are explicit and
do not silently fall back to hashed vectors.

Run the fixed paraphrase-retrieval check from `services/core/` after configuring a
real provider:

```text
python -m sonic.memory.retrieval_eval
```

It reports Recall@3 and mean reciprocal rank (MRR) for six SONIC memory
queries. The evaluation corpus is deliberately small and is a regression
signal, not proof of general-purpose retrieval quality.

### Agentic retrieval

Cross-mission lessons in the computer-use reasoning path use a bounded
agentic-retrieval step when a model router is available. SONIC searches the
tenant's vector memory, asks the router whether the retrieved passages address
the current goal, and—when necessary—runs up to two further retrieval-planning
rounds. The planner can propose up to four total search queries; retrieved
results are merged by document ID and ranked with reciprocal-rank fusion, so
sources that appear near the top for multiple queries are favored. The top
lessons are passed to the normal reasoning prompt with their original evidence.
The planner cannot write memory, create evidence, or execute actions.

This adds up to two text-only reasoning calls per mission's first lesson lookup.
Queries and results remain tenant-scoped. If the planner is unavailable or
returns an invalid plan, SONIC logs the condition and uses the initial
retrieval; vector-store/embedding errors are not hidden. Agentic query planning
improves retrieval coverage only when the underlying index contains useful
documents; it does not verify their truth or guarantee answer quality.

### BeingVault key setup and legacy data

Provision `SONIC_BEING_VAULT_KEY` from a secret manager and keep it stable across
restarts and all backend workers that access the same Being database. Generate
a new key with:

```powershell
python -c "import base64,secrets; print(base64.urlsafe_b64encode(secrets.token_bytes(32)).decode())"
```

Do not commit the key or store it alongside the database. Vault reads and writes
fail closed when the key is missing, malformed, or different from the key used
to encrypt the records.

Databases created before vault encryption may contain plaintext credentials or
session state. Back up the database to protected storage, configure the key,
stop other database users, and explicitly run
`BeingVault.migrate_legacy_data()` before retrieving those records. Protect and
retire the pre-migration backup separately because it still contains plaintext.
Do not delete or discard it until the encrypted records have been verified.

## LLM Cost Controls

The limits in `configs/models.yaml` are checked by the model router using
configured model token pricing. Text-only requests are estimated before
dispatch, engagement estimates are reserved against concurrent requests, and
response usage/cost is recorded afterward. Engagement totals are maintained
per router process, not in shared persistent storage.

For multimodal requests, image-token cost is not included in the preflight
estimate because no provider-specific image pricing is configured. Vision
requests are still allowed; their reported response cost or token usage is
accounted after completion, and the router logs/reports that a hard cap was not
guaranteed for the image portion. Providers must return usable cost or token
usage for the completed call to be accounted.

## Self-Coder Autonomous Workflow

- The periodic self-coder runs independently of `SONIC_ENABLE_BEING_LIFE_LOOP`; it is enabled by default and can be stopped with `SONIC_SELF_CODER_ENABLED=false`. It runs immediately at backend startup and then at `SONIC_SELF_CODER_INTERVAL_SECONDS`.
- Candidates are created and tested in detached worktrees from the committed `HEAD`; uncommitted active-checkout changes are left untouched and listed in the report as excluded. A passing candidate is automatically revalidated and committed to `sonic/self-coder/<candidate_id>`. The active branch is not changed and nothing is pushed remotely.
- Discovery uses real test failures and read-only code-audit findings. Audit findings are hypotheses: the tool-using coding agent must inspect the source, establish a diagnosis, and the candidate must pass import, full test, and security-regression checks before its branch commit is created.
- Every run stores the diagnosis, expected impact (explicitly not a production measurement), changed files/diff, test and security receipts, commit hash/branch, and errors in candidate records. Inspect `GET /self-coder/status`, run immediately with `POST /self-coder/run`, and list/read history under `/self-coder/candidates`.
- `POST /self-coder/candidates/{candidate_id}/approve` remains for older verified records. New candidates do not wait for human approval. Reject/delete endpoints remain available for uncommitted historical candidate records.

## Dynamic Application & Target Policy

- **Dynamic Binaries**: Applications are discovered from `$PATH` and POSIX entries (`x-www-browser`, `sensible-browser`, `xdg-open`) without vendor binary hardcoding.
- **Target Scope**: Test and documentation targets use RFC 2606 `.test` domains (`https://app.target.test`, `https://git.target.test/org/repo`) and RFC 5737 documentation network addresses (`198.51.100.1`).
