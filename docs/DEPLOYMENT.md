# Deployment notes

SONIC's compose and provider configurations are environment-specific. A
successful container start is not evidence of secure production deployment or
reliable computer-use task completion.

## Local development

Use [`docker-compose.yml`](../docker-compose.yml) for the local workstation and
supporting services. Start the API and dashboard separately as documented in
[Getting Started](GETTING_STARTED.md). The local compose file contains
development-only service settings; do not reuse its example credentials in a
shared or production environment.

VNC/noVNC are published to loopback by default. Keep them local or put them
behind an independently secured access mechanism; do not expose raw VNC ports
to an untrusted network.

## Production compose

[`docker-compose.prod.yml`](../docker-compose.prod.yml) is a configuration
template, not a production-readiness certification. It requires strong
secrets, including `SECRET_KEY`, `NEO4J_PASSWORD`, `DAYTONA_API_KEY`, and
`GROQ_API_KEY` (the configured default model provider), and sets
`SONIC_USE_DAYTONA_CLOUD=1` for the workstation routes. Optional provider keys
are forwarded from the deployment environment when configured. This is
distinct from the local Docker workstation path.

The staging stack in [`infra/docker-compose.staging.yml`](../infra/docker-compose.staging.yml)
also requires configured `SONIC_SITE_HOST`, `SONIC_API_HOST`, `POSTGRES_PASSWORD`,
`REDIS_PASSWORD`, `NEO4J_PASSWORD`, `JWT_SECRET`, `GROQ_API_KEY`, and
`DAYTONA_API_KEY` values. Caddy terminates TLS for the two configured hostnames;
the `.test` entries in `.env.example` are placeholders and are not deployable DNS.

The production Compose template persists SQLite-backed service and Being data
in the named `sonic_data_prod` volume mounted at `/app/sonic_data`. Include this
volume in the deployment's backup and restore procedure. Staging mounts the
repository's `configs/` read-only and stores local runtime state in its
`sonic-data` named volume. Its Caddy ingress requires `SONIC_SITE_HOST` and
`SONIC_API_HOST` to be real DNS hostnames before deployment; the `.test`
examples in `.env.example` are placeholders only.

Before deployment:

1. Review the compose file, provider tenancy model, network exposure, volumes,
   resource limits and backup/recovery approach for the exact environment.
2. Provide secrets through a managed secret store or protected environment;
   never commit `.env` or credentials.
3. Restrict access to the API, dashboard, databases, Redis and workstation.
4. Verify that production auth, tenant isolation, provider and policy settings
   are actually active.
5. Run deployment-specific checks and a controlled workstation acceptance
   task; do not infer readiness from unit tests or container health alone.

Production deployment requires an explicitly configured remote workstation
provider; it must not silently fall back to a shared local Docker desktop.
Confirm provider behavior and authorization boundaries before enabling operator
tasks.

## Operational truth

The product contract requires fresh desktop observations, grounded actions,
action receipts and independent Judge verification for GUI success. A healthy
API, a reachable VNC endpoint, an open application or a model response is not
completion evidence. Record blocked or unverified outcomes as such.
