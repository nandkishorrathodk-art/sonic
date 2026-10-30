# SONIC Production Deployment Guide

This guide covers deploying the SONIC AI-Human Digital Being workstation and control plane using `docker-compose.prod.yml`.

## 1. Environment Preparation

Provide required secrets via environment variables (never hardcoded):
- `JWT_SECRET`: High-entropy signing secret (minimum 32 bytes)
- `POSTGRES_PASSWORD`: Database credential
- `REDIS_PASSWORD`: Queue and session cache credential
- `NEO4J_PASSWORD`: Persistent graph memory credential
- `SONIC_ALLOW_HOST_EXECUTION=false`: Hard-enforced fail-closed sandbox execution outside development

## 2. Launching the Stack

```bash
docker compose -f docker-compose.prod.yml up -d --build
```

## 3. Workstation & Network Egress Security

- All desktop GUI actions (`DISPLAY=:99`) resolve applications dynamically via `$PATH`, `x-www-browser`, and `xdg-open`.
- All outbound traffic is governed by `SealedActionPolicy` and the egress filter (`169.254.169.254` and private ranges blocked by default; authorized targets such as `https://app.target.test` are DNS-verified).

