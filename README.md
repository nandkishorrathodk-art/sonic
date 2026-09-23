# SONIC-REDA
### Next-Generation Autonomous AI Bug Hunting System
# ⚡ SONIC-REDA (A-SEA)
### Autonomous Self-Evolving Penetration Architect & Cyber Workstation

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![Next.js 14](https://img.shields.io/badge/Next.js-14-black.svg)](https://nextjs.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688.svg)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg)](https://www.docker.com/)
[![License: Proprietary](https://img.shields.io/badge/license-Proprietary-red.svg)]()

> **Codename:** SONIC A-SEA (Autonomous Self-Evolving Penetration Architect)  
> **Status:** Phase 22 — Native Docker Cyber Workstation & Autonomous Reasoning  
> **Identity:** A-SEA (Autonomous Self-Evolving Penetration Architect)  
> **Core Concept:** An autonomous AI-Human Hacker Being operating its own dedicated personal computer workstation with full graphical desktop agency, headless terminal execution, persistent cognitive mind, and an immutable safety envelope. Every finding is backed by empirical receipts — zero success-by-decree.

## What is SONIC-REDA?
---

The most powerful, self-evolving, multi-agent AI red-team system that combines:
## 📑 Table of Contents
- [Overview](#-overview)
- [Core Architecture: The AI Being & Workstation](#-core-architecture-the-ai-being--workstation)
- [Key Capabilities & Subsystems](#-key-capabilities--subsystems)
- [Deep AI / ML Architecture](#-deep-ai--ml-architecture)
- [Repository Structure](#-repository-structure)
- [Quick Start Guide](#-quick-start-guide)
- [The Strict Core Invariants](#-the-strict-core-invariants)
- [Testing & Quality Assurance](#-testing--quality-assurance)
- [License](#-license)

- 🖥️ **Devin-style** full virtual computer control
- 🔴 **RedAmon-level** autonomous kill-chain power
- 🧠 **True multi-agent swarm intelligence** with Agent-to-Agent Graph Memory
- 🎯 **Autonomous pentest loop** — Observe → Think → Act → Re-probe, keeps
  going until the kill-chain is proven (no more one-shot-and-stop)
- 📋 **Mandatory Evidence Engine** — zero hallucination tolerance, every
  finding ships with the real request + response that proved it
- 🔄 **Controlled self-development** with measurable improvement
- 🛡️ **Production-grade safety** + rollback (default-deny egress, scope, rate-limit)
---

## Architecture
## 🌟 Overview

**SONIC-REDA** is an autonomous offensive-security artificial intelligence system. Unlike conventional security scanners that blindly run keyword scripts or chatbots that hallucinate vulnerabilities, SONIC operates as an **independent agentic entity** with its own virtual computer environment.

It autonomously:
1. **Perceives** graphical application windows (Display `:99`) and headless terminal outputs simultaneously.
2. **Reasons** from first-principles using multimodal Large Language Models (LLMs) and Vision-Language Models (VLMs).
3. **Authors** its own custom tools (**Toolsmith**) and invents novel attack methods (**Method Lab**) when off-the-shelf utilities are missing or insufficient.
4. **Reproduces & Verifies** every vulnerability empirically in sandboxed runtime containers — zero unverified claims.
5. **Survives restarts** with persistent SQLite-backed cognitive affect, vector embeddings, and Neo4j graph memory.

---

## 🖥️ Core Architecture: The AI Being & Workstation

```
┌──────────────────────────────────────────────┐
│       Immutable Safety & Scope Layer         │
├──────────────────────────────────────────────┤
│       Meta / Self-Development Layer          │
├──────────────────────────────────────────────┤
│       Sonic Agentic Graph (Core Brain)       │
│  Orchestrator ↔ Recon ↔ Reasoning ↔ Dynamic │
│  Hypothesis ↔ Verifier ↔ Exploit ↔ CodeFix  │
├──────────────────────────────────────────────┤
│       Virtual Computer Layer (Sandbox)       │
└──────────────────────────────────────────────┘
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

## Quick Start
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

## 🧠 Deep AI / ML Architecture

SONIC is designed from the ground up as an **AI/ML-Native system**, eliminating rigid heuristics in favor of probabilistic reasoning and learned representations:

1. **Multimodal Perception & Computer Vision:**
   - Screenshots of Display `:99` are captured via `scrot -p` and processed directly by Vision-Language Models (VLMs) like Claude 3.5 Sonnet, GPT-4o, Gemini 1.5 Pro, or local Qwen2-VL.
   - Grounding algorithms convert bounding-box tokens (`<|box_start|>ymin,xmin,ymax,xmax<|box_end|>`) into sub-pixel click targets.

2. **Cognitive ReAct Decision Loop:**
   - Every step follows `Observation -> Multimodal LLM Chain-of-Thought -> Structured Action -> Environment Feedback`.
   - Stderr reflection: Execution failures prompt self-correction, syntax adjustments, and dynamic pivoting.

3. **High-Dimensional Vector Memory (RAG):**
   - Past penetration testing outcomes, CVE insights, and payloads are embedded via dense vector models (`text-embedding-3`, `BGE-m3`, `SentenceTransformers`) and retrieved using cosine similarity.

4. **Curiosity & Novelty Engine:**
   - Quantifies the information gain of new server responses, UI states, and files using vector distance. High novelty states incentivize deeper autonomous probing during idle curiosity loops.

5. **Clustering & Anomaly Detection:**
   - Analyzes response latency, status codes, and DOM trees using clustering algorithms (DBSCAN / K-Means) to surface blind vulnerabilities without pre-canned regexes.

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
- Python 3.12+
- Docker & Docker Compose
- Node.js 20+ (for dashboard)
- A Google Cloud project with OAuth2 credentials
- **Python 3.12+**
- **Node.js 20+** (for dashboard)
- **Docker & Docker Compose**
- LLM API Key (Groq, Anthropic, OpenAI, xAI, or local Ollama/vLLM)

### Setup

### 1. Clone & Configure Environment
```bash
# 1. Clone and enter the project
git clone https://github.com/nandkishorrathodk-art/sonic.git
cd sonic

# 2. Copy environment template
# Create local environment configuration
cp .env.example .env
# Edit .env with your API keys and Google OAuth credentials
```
Edit `.env` to configure your preferred LLM provider (`GROQ_API_KEY`, `ANTHROPIC_API_KEY`, or `OPENAI_API_KEY`).

# 3. Start infrastructure (Neo4j + Redis)
#    Neo4j is OPTIONAL in dev — the backend falls back to an in-memory
#    graph store if localhost:7687 is unreachable.
### 2. Start Infrastructure & Workstation Container
```bash
# Start Neo4j & Redis (Optional: backend auto-falls back to persistent SQLite if offline)
docker-compose up -d

# 4. Install Python dependencies
pip install -e sonic-core[dev]
pip install -e sonic-cli[dev]
# Start the Native Docker Cyber Workstation
docker start sonic-desktop-workstation
```
> The live XFCE4 desktop stream is available at: **`http://localhost:6080/vnc.html`**

# 5. Start the backend
#    uvicorn is not on PATH; invoke it as a module:
### 3. Install & Start Backend
```bash
# Install backend in editable mode
pip install -e "sonic-core[dev]"

# Launch the FastAPI backend on port 12000
APP_ENV=development python -m uvicorn sonic.api.main:app --port 12000 --reload
```
> Interactive API Documentation: **`http://127.0.0.1:12000/docs`**

# 6. Start Native Docker Cyber Workstation
docker start sonic-desktop-workstation
# Stream live XFCE4 desktop at: http://localhost:6080/vnc.html
### 4. Install & Start Next.js Dashboard
```bash
cd sonic-dashboard
npm install
npm run dev -p 12001
```
> Open your browser at: **`http://127.0.0.1:12001`**

# 7. Launch Autonomous Agent Mission
sonic/
├── sonic-core/          # 🧠 Core Python backend (FastAPI)
│   └── sonic/
│       ├── auth/        # Google OAuth2 + JWT
│       ├── llm/         # Abstract LLM Provider + Model Router
│       ├── agents/      # Multi-agent system
│       ├── memory/      # Neo4j Graph Memory
│       ├── safety/      # Immutable Safety Layer
│       ├── evidence/    # Evidence Engine
│       └── api/         # REST API
├── sonic-cli/           # ⌨️ CLI (Typer + Rich)
├── sonic-dashboard/     # 🖥️ Next.js Dashboard
└── configs/             # 📋 Shared YAML configs
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

## Core Principles
---

1. **Evidence > Claims** — No finding without proof
2. **Speed without chaos** — Sonic fast, never reckless
3. **Shared intelligence** — Graph Memory for all agents
4. **Self-improvement** — Only through measured experiments
5. **Immutable safety** — Core rules cannot be modified by agents
6. **Human authority** — Final say on high-risk actions
## 📄 License

## License

Proprietary — All rights reserved.
Proprietary — All rights reserved.  
Unauthorized copying, modification, distribution, or commercial deployment is strictly prohibited.
