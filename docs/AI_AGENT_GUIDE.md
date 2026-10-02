# AI agent and contributor guide

This guide is for people and coding agents changing the SONIC repository. It
does not describe an autonomous worker as more capable than current evidence
supports.

## Before changing code

1. Read [`AGENTS.md`](../AGENTS.md) for repository-specific invariants.
2. Read the [product and GUI acceptance contract](SONIC_COMPUTER_USER_GOAL.md)
   before changing workstation, perception, orchestration or Judge behavior.
3. Resolve subsystem ownership using
   [`architecture.py`](../sonic-core/sonic/organization/architecture.py).
4. Inspect the current call path and tests; historical phase notes are not
   proof that a module is still wired.
5. Keep changes focused. Preserve unrelated worktree changes.

## Development setup

The backend workspace requires Python 3.12 or later. From the repository root:

```powershell
python -m pip install -e "sonic-core[dev]"
python -m pytest sonic-core/tests/
```

For the dashboard:

```powershell
Set-Location sonic-dashboard
npm install
npm run lint
npm run build
```

The local workstation and optional supporting services are configured with
`docker-compose.yml`. Follow [Getting Started](GETTING_STARTED.md) and provide
the required local VNC password. Tests which need live Docker or a real desktop
may be skipped or fail when that infrastructure is unavailable; report the
actual result rather than describing such runs as passed.

## Computer-use change requirements

- Route every actionable workstation objective through Boss orchestration.
- Delegate bounded tasks to ordinary `ComputerUseAgent` workers.
- Preserve the assigned plane; GUI-only tasks cannot silently become terminal tasks.
- Use fresh state and explicit coordinate metadata. Reject ambiguity and
  out-of-bounds targets rather than guessing.
- Keep screenshots available when visual evidence is required; do not require
  full-frame transfer when fresh, unambiguous structured state is enough.
- Forward operator interruption through Boss to delegated workers.
- Treat worker completion as a proposal. Only independent Judge evidence can
  confirm GUI task completion.
- Add regression tests for changed behavior; distinguish mocked tests from
  live workstation acceptance.

For end-to-end acceptance, follow the measurable criteria in the product
contract. A passing unit test is not evidence that an arbitrary desktop task
works.

## General engineering expectations

- Keep tenant and workspace boundaries intact.
- Do not weaken policy, egress or approval checks to make a test pass.
- Surface failures explicitly; do not mask them with success-shaped fallbacks.
- Use existing patterns and test commands, and run the smallest relevant suite
  before escalating to the full suite.
- Do not make network, workstation or release claims without checking the
  actual configured provider and production wiring.
- Keep documentation synchronized with changed behavior.
