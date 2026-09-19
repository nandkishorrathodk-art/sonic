# ⚡ SONIC-REDA (A-SEA)
### Autonomous Self-Evolving Penetration Architect & Cyber Workstation

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![Next.js 14](https://img.shields.io/badge/Next.js-14-black.svg)](https://nextjs.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688.svg)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg)](https://www.docker.com/)
[![License: Proprietary](https://img.shields.io/badge/license-Proprietary-red.svg)]()

> **Identity:** A-SEA (Autonomous Self-Evolving Penetration Architect)  
> **Core Concept:** An autonomous AI-Human Hacker Being operating its own dedicated personal computer workstation with full graphical desktop agency, headless terminal execution, persistent cognitive mind, and an immutable safety envelope. Every finding is backed by empirical receipts — zero success-by-decree.

---

## 📑 Table of Contents
- [Overview](#-overview)
- [Core Architecture & Operational Model](#-core-architecture--operational-model)
- [Key Capabilities & Subsystems](#-key-capabilities--subsystems)
- [Repository Structure](#-repository-structure)
- [Quick Start Guide](#-quick-start-guide)
- [The Strict Core Invariants](#-the-strict-core-invariants)
- [Testing & Quality Assurance](#-testing--quality-assurance)
- [License](#-license)

---

## 🌟 Overview

**SONIC-REDA** is an autonomous offensive-security artificial intelligence system. Unlike conventional security scanners that blindly run keyword scripts or chatbots that hallucinate vulnerabilities, SONIC operates as an **independent agentic entity** with its own virtual computer environment.

It autonomously:
1. **Perceives** graphical application windows (Display `:99`) and headless terminal outputs simultaneously.
2. **Reasons** from first-principles using multimodal Large Language Models (LLMs) and Vision-Language Models (VLMs).
3. **Authors** its own custom tools (**Toolsmith**) and invents novel attack methods (**Method Lab**) when off-the-shelf utilities are missing or insufficient.
4. **Reproduces & Verifies** every vulnerability empirically in sandboxed runtime containers — zero unverified claims.
5. **Survives restarts** with persistent SQLite-backed cognitive affect, vector embeddings, and Neo4j graph memory.

---

## 🖥️ Core Architecture & Operational Model

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                    SONIC A-SEA BEING                                   │
│            Persistent Mind (sqlite) │ Vector Memory (Embeddings) │ Graph Memory        │
│                Curiosity Loop │ Self-Reflection │ Craft Ledger (Host Disk)             │
└───────────────────────────────────────────┬────────────────────────────────────────────┘
                                            │
                                            ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                              COGNITIVE ORCHESTRATION ENGINE                            │
│                 BossAgent (Multi-Phase Planner) ↔ TaskGraph Dependency Wave            │
│                       LLM Multi-Provider Router (Anthropic/OpenAI/Groq/Ollama)         │
│                 6+1 Research Swarm (Network, Web, API, Auth, Logic, Cloud, Falsify)    │
└───────────────────────────────────────────┬────────────────────────────────────────────┘
                                            │
                                            ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                              IMMUTABLE SAFETY ENVELOPE (SHA-256)                       │
│                     SealedActionPolicy ↔ ScopeChecker ↔ SandboxTierRouter              │
│                       Zero Host Escapes │ Egress Filter │ Rate Limiting                │
└───────────────────────────────────────────┬────────────────────────────────────────────┘
                                            │
                                            ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        DUAL-PLANE DOCKER WORKSTATION PROVIDER                          │
│                                                                                        │
│  🖥️ Workstation Application Plane (GUI)       💻 Operator Sandbox Plane (Headless)     │
│  - Virtual X11 Display (:99)                  - Non-graphical high-throughput terminal  │
│  - XFCE4 Desktop & Dynamic Window Tiling      - Custom Python probes & AST auditing    │
│  - x11vnc (:5900) + noVNC Websockify (:6080)  - Native Rust kernel bridges             │
│  - Real Multimodal Vision Grounding           - Toolsmith compilation & execution      │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

### 1. Dual-Plane Operational Model
SONIC does not silo itself into a terminal-only script or a sluggish GUI-only recorder:
- **Workstation Application Plane (Computer-Use)**: Interacts with the full graphical desktop environment (X11 via `GUI_*`, `APP_*`, `BROWSER_*`). Operates any arbitrary desktop application, tool, debugger, or browser installed on the OS via visual perception without hardcoded app names.
- **Operator & Sandbox Plane (Direct Headless Execution)**: Dedicated high-speed execution plane (`TERMINAL_EXEC`, `SECURITY_TOOL`, `FILE_*`, `GIT_*`). Heavy scripts, socket listeners, and source audits execute directly in the container sandbox without cluttering the screen or wasting visual tokens.
- **Zero Puppet Scripts**: SONIC autonomously decides which plane and tool to use based on observations and goals.

### 2. Persistent Being & Craft
- **Identity & Affect (`sonic/being/identity.py`)**: Persists being identity, mood metrics (`curiosity_drive`, `focus`, `satiety`), and learned facts across reboots.
- **Always-On Life Loop (`sonic/being/life_loop.py`)**: Runs when no operator task is active, exploring, hypothesizing, and learning autonomously under the safety envelope.
- **Durable Craft (`sonic/being/craft.py`)**: Persists authored scripts and notes to host storage (`sonic_data/craft/`).

---

## ⚡ Key Capabilities & Subsystems

| Subsystem | Module Path | Purpose |
| :--- | :--- | :--- |
| **Boss Agent** | `sonic/computer_use/boss.py` | Multi-phase strategic planner; decomposes goals into dependency waves. |
| **Computer-Use Agent** | `sonic/computer_use/agent.py` | Core observe $\to$ reason $\to$ act $\to$ verify closed loop. |
| **Vision Grounding** | `sonic/computer_use/grounding.py` | Resolves visual elements to exact pixel coordinates from raw screenshots. |
| **Docker Computer** | `sonic/computer/docker_computer.py` | Direct workstation container control (XFCE4, Xvfb `:99`, x11vnc, wmctrl). |
| **Safety Envelope** | `sonic/safety/action_policy.py` | Fail-closed policy engine enforcing egress, rate limits, and path bounds. |
| **Sealed Policy** | `sonic/safety/sealed.py` | SHA-256 tamper-evident cryptographic wrapper around safety policies. |
| **Toolsmith** | `sonic/evolution/toolsmith/` | Generates, tests, and deploys custom Python tools on-the-fly. |
| **Method Lab** | `sonic/evolution/method_lab/` | Synthesizes novel exploit vectors and fuzzing payloads. |
| **Evidence Engine** | `sonic/evidence/` | Verifies and cryptographically hashes reproduction receipts for all findings. |
| **Research Swarm** | `sonic/researcher/` | 6+1 concurrent specialist agents auditing attack surfaces in parallel. |
| **Graph & Vector Memory** | `sonic/memory/` | Dual Neo4j / SQLite property graph and embedding vector storage. |

---

## 📁 Repository Structure

```
.
├── sonic-core/                         # Python Backend (FastAPI, Python 3.12+)
│   ├── sonic/
│   │   ├── api/routes/                 # REST Endpoints (Workstation, Sessions, Auth, Missions)
│   │   ├── being/                      # Persistent Being Identity, Affect, Craft, & Life Loop
│   │   ├── computer/                   # Docker Workstation & Sandbox Providers
│   │   ├── computer_use/               # BossAgent, ComputerUseAgent, Visual Grounding
│   │   ├── evidence/                   # Empirical Evidence Engine & Receipts
│   │   ├── evolution/                  # Toolsmith, Method Lab, Dynamic Strategy Engine
│   │   ├── llm/                        # Multi-LLM Providers (Anthropic, OpenAI, Groq, Local)
│   │   ├── memory/                     # Neo4j & Persistent SQLite Graph Memory, Vector Store
│   │   ├── researcher/                 # 6+1 Parallel Specialist Swarm
│   │   ├── safety/                     # Cryptographic Sealed Action Policy & Scope Enforcement
│   │   └── tools/                      # Security Tool Adapter Registry (nmap, nuclei, ffuf)
│   └── tests/                          # 350+ Pytest Test Suites
├── sonic-dashboard/                    # Cyber Dashboard (Next.js 14, React, Tailwind CSS)
│   ├── app/                            # Next.js App Router (Workstation, Missions, Evidence)
│   ├── components/                     # Workstation GUI, Live VNC Stream, Terminal Timeline
│   └── lib/                            # API Client, Token Auto-Refresh, WebSocket Streaming
├── sonic-cli/                          # Command-Line Interface (Typer + Rich)
├── docker/                             # Workstation Container Configs & Entrypoints
├── configs/                            # System Safety Rules & Model Routing YAMLs
├── AGENTS.md                           # Core Repository Memory & Operational Rules
└── README.md                           # This Documentation
```

---

## 🚀 Quick Start Guide

### Prerequisites
- **Python 3.12+**
- **Node.js 20+** (for dashboard)
- **Docker & Docker Compose**
- LLM API Key (Groq, Anthropic, OpenAI, xAI, or local Ollama/vLLM)

### 1. Clone & Configure Environment
```bash
git clone https://github.com/nandkishorrathodk-art/sonic.git
cd sonic

# Create local environment configuration
cp .env.example .env
```
Edit `.env` to configure your preferred LLM provider (`GROQ_API_KEY`, `ANTHROPIC_API_KEY`, or `OPENAI_API_KEY`).

### 2. Start Infrastructure & Workstation Container
```bash
# Start Neo4j & Redis (Optional: backend auto-falls back to persistent SQLite if offline)
docker-compose up -d

# Start the Native Docker Cyber Workstation
docker start sonic-desktop-workstation
```
> The live XFCE4 desktop stream is available at: **`http://localhost:6080/vnc.html`**

### 3. Install & Start Backend
```bash
# Install backend in editable mode
pip install -e "sonic-core[dev]"

# Launch the FastAPI backend on port 12000
APP_ENV=development python -m uvicorn sonic.api.main:app --port 12000 --reload
```
> Interactive API Documentation: **`http://127.0.0.1:12000/docs`**

### 4. Install & Start Next.js Dashboard
```bash
cd sonic-dashboard
npm install
npm run dev -p 12001
```
> Open your browser at: **`http://127.0.0.1:12001`**

---

## 🛡️ The Strict Core Invariants

The following invariants are hard-enforced in code and prompts:

1. **Zero Hardcoded Application Names:** Control flow and intent routing never assume fixed binaries (`chrome`, `firefox`, `wireshark`). Applications are discovered dynamically from `$PATH`, `.desktop` entries, and active OS window titles.
2. **Zero Scripted Puppet Heuristics:** No rigid keyword trees, canned checklists, or regex intent steering. The agent reasons freely from live observations.
3. **Zero Host Execution:** All actions, terminal commands, and tool authoring execute inside isolated Docker container sandboxes. Host escape attempts trigger immediate fail-closed termination.
4. **Empirical Falsification:** Every vulnerability finding, milestone completion, and knowledge claim must be backed by verifiable in-sandbox execution receipts.
5. **Fail-Closed Safety Envelope:** Safety policies are cryptographically hashed (`SealedActionPolicy`). Any unauthorized runtime modification triggers an immediate `DENY`.

---

## 🧪 Testing & Quality Assurance

SONIC features extensive test suites ensuring regression-free execution across safety, reasoning, and tools:

```bash
# Run the core regression and security suites
python -m pytest sonic-core/tests/test_phase_plan6_safety_envelope.py \
                 sonic-core/tests/test_ai_and_control_upgrade.py \
                 sonic-core/tests/test_p0_security_hardening.py \
                 sonic-core/tests/test_workstation_boss_integration.py \
                 sonic-core/tests/test_safety_sealed_policy.py
```

---

## 📄 License

**Copyright (c) 2026 Nandkishor Rathod. All Rights Reserved.**

This project is proprietary software. No entity or individual may copy, clone, distribute, modify, or commercially exploit this software without express prior written permission. See [LICENSE](LICENSE) for full legal terms.
