# Services

- [`core/`](core/) contains SONIC's FastAPI backend, orchestration, computer-use
  runtime, safety enforcement, service-owned tests and production container
  build.

Product interfaces are under [`../apps/`](../apps/); reusable components are
under [`../packages/`](../packages/).

The root `pyproject.toml` includes `services/core` in the Python workspace.
Run the complete Python suite from the repository root with
`uv run --locked --all-packages --extra dev python -m pytest`.
