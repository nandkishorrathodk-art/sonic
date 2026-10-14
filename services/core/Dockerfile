# ============================================
# SONIC Core — Production Dockerfile
# Multi-stage build compiling Native Rust Kernel (sonic-kernel-rs)
# and Python runtime
# ============================================

# Stage 1: Build Native Rust Kernel
FROM rust:1.85-slim AS rust-builder
WORKDIR /kernel-build
COPY packages/kernel/Cargo.toml packages/kernel/Cargo.lock* ./
COPY packages/kernel/src ./src
RUN cargo build --locked --release

# Stage 2: Python builder
FROM python:3.12-slim AS builder
WORKDIR /app
ARG INSTALL_AVCS_GROUNDING=0
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY services/core/pyproject.toml .
# AVCS's forge.vision package initializer eagerly imports its cortex modules,
# so PyTorch is required even when only the pixel-reflex path is enabled.
RUN pip install --no-cache-dir --upgrade pip && \
    if [ "$INSTALL_AVCS_GROUNDING" = "1" ]; then \
      pip install --no-cache-dir ".[avcs-grounding]"; \
    else \
      pip install --no-cache-dir . "torch>=2.0.0"; \
    fi

# Stage 3: Runtime
FROM python:3.12-slim AS runtime
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    nmap \
    ca-certificates \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install compiled native Rust kernel binary
COPY --from=rust-builder /kernel-build/target/release/sonic-kernel /usr/local/bin/sonic-kernel
RUN chmod +x /usr/local/bin/sonic-kernel

# Create unprivileged runner user
RUN useradd -m -u 1000 sonic && chown -R sonic:sonic /app

COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

COPY --chown=sonic:sonic services/core/sonic /app/sonic
COPY --chown=sonic:sonic packages/avcs/forge/vision /app/packages/avcs/forge/vision
COPY --chown=sonic:sonic packages/avcs/avcs /app/packages/avcs/avcs

RUN mkdir -p /app/sonic_data && chown sonic:sonic /app/sonic_data

ENV PYTHONPATH=/app/packages/avcs

# Exercise the same PNG capture entry point used by the workstation provider.
# This verifies pixel-reflex packaging only; no grounding checkpoint is bundled
# or presumed to be available.
RUN python -c \
    "from io import BytesIO; import numpy as np; from PIL import Image; from sonic.computer_use.avcs_perception import AVCSPerception; frame = BytesIO(); Image.fromarray(np.zeros((16, 16, 3), dtype=np.uint8)).save(frame, format='PNG'); state = AVCSPerception().perceive_png(frame.getvalue()); assert state['engine'] == 'avcs_reflex' and (state['width'], state['height']) == (16, 16)"

USER sonic
EXPOSE 8000

ENV APP_ENV=production \
    PYTHONUNBUFFERED=1 \
    SONIC_KERNEL_BIN=/usr/local/bin/sonic-kernel

CMD ["uvicorn", "sonic.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
