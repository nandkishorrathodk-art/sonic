# Contributing to SONIC

Thank you for contributing to SONIC. SONIC is developed as a multi-disciplinary autonomous AI system, so contributions must preserve architectural boundaries, safety invariants, reproducibility, and evidence-based evaluation.

## Development model

- `main` is the protected integration/release branch.
- `development` is the active integration branch for ongoing engineering.
- Contributors work from short-lived branches created from `development`.
- Changes enter through pull requests.
- CI must pass before merge.
- Behavioural changes require tests or an explicit documented reason why automated coverage is not practical.

## Branch naming

Use one of:

- `feat/<area>/<short-name>`
- `fix/<area>/<short-name>`
- `refactor/<area>/<short-name>`
- `test/<area>/<short-name>`
- `docs/<area>/<short-name>`
- `research/<area>/<short-name>`

Examples:

- `feat/cognitive/memory-retrieval-v2`
- `fix/computer/window-focus`
- `test/evaluation/browser-agent`

## Engineering ownership

SONIC is organized around product responsibilities:

1. **Cognitive Systems** — reasoning, agents, planning, memory, knowledge, curiosity and evolution.
2. **Computer & Interaction** — GUI, browser, perception, DOM/CDP grounding and motor control.
3. **Platform & Systems** — APIs, execution runtime, Rust systems, infrastructure, data services and CI/CD.
4. **Security & Safety** — policy enforcement, sandboxing, isolation, permissions and security engineering.
5. **Evaluation & Reliability** — tests, benchmarks, falsification, regression and failure analysis.
6. **Experience & Interface** — dashboard, observability and human interaction.
7. **Research & Evolution Lab** — cross-functional experimental work.

See [docs/EMPLOYEE_ARCHITECTURE.md](docs/EMPLOYEE_ARCHITECTURE.md) for the people and ownership model.

## Pull request requirements

Every PR should explain:

- What changed?
- Why is the change needed?
- Which subsystem owns it?
- How was it tested?
- What evidence demonstrates the expected behaviour?
- Are there security, safety, performance, migration, or compatibility implications?

Keep PRs focused. Avoid mixing unrelated refactors with feature work.

## SONIC invariants

Do not introduce:

- hardcoded application-specific automation when dynamic grounding is required;
- puppet-style coordinate scripts;
- silent shell failure masking;
- unverified success claims;
- unsafe host escapes from protected execution paths;
- new external/mock production endpoints without explicit justification;
- state mutations that bypass the typed state/evidence flow.

If an invariant must change, treat it as an architectural change and document the rationale in the PR.

## Testing

At minimum, run the relevant local checks before opening a PR:

```bash
uv run --locked --all-packages --extra dev ruff check services/core/sonic apps/cli/sonic_cli
uv run --locked --all-packages --extra dev python -m pytest
```

For changes touching the safety boundary, also run the P0 security/safety suite described in the repository documentation.

The CI workflow is authoritative for merge gates.

## Review standard

Reviewers should evaluate:

1. correctness;
2. security and safety;
3. test/evidence quality;
4. architectural fit;
5. failure handling;
6. observability;
7. maintainability;
8. performance where relevant.

A green test suite is necessary, not sufficient, for autonomous-agent changes.

## Contributor progression

Contributors can grow from task-level work to subsystem ownership based on demonstrated engineering quality, reliable reviews, test ownership, and independent delivery. Role titles should reflect demonstrated responsibility rather than time spent in the project.

## Questions

Open a focused GitHub issue for architectural questions or implementation blockers. Do not publish sensitive security findings in public issues; follow [SECURITY.md](SECURITY.md).
