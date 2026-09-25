# ⚡ SONIC
### Autonomous AI-Human Digital Being & Cyber Workstation

[![Author: Nandkishor Rathod](https://img.shields.io/badge/Author-Nandkishor%20Rathod-blue?style=flat&logo=github)](https://github.com/nandkishorrathodk-art)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-3776AB.svg?style=flat&logo=python)](https://www.python.org/downloads/)
[![Next.js 14](https://img.shields.io/badge/Next.js-14-black.svg?style=flat&logo=next.js)](https://nextjs.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688.svg?style=flat&logo=fastapi)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg?style=flat&logo=docker)](https://www.docker.com/)
[![Three.js](https://img.shields.io/badge/Three.js-3D%20WebGL-black.svg?style=flat&logo=three.js)](https://threejs.org/)
[![System-1 Laya](https://img.shields.io/badge/System--1-Laya%200.3.20%20(Neural)-FF3B5C.svg?style=flat)](https://github.com/NandhaKishorM/laya)
[![License: Proprietary](https://img.shields.io/badge/License-Proprietary-red.svg?style=flat)](LICENSE)

> **Architect & Author:** [Nandkishor Rathod](https://github.com/nandkishorrathodk-art)  
> **Codename:** SONIC (Autonomous AI-Human Digital Being)  
> **Core Concept:** An autonomous AI-Human colleague operating its own dedicated personal computer workstation (`DISPLAY=:99` full graphical desktop + dedicated headless sandbox terminal). Powered by a Kahneman **Dual-Brain Cognitive Engine** (System 1: Laya Neural Triage + System 2: LLM Multi-Provider Reasoner), persistent SQLite graph memory, toolsmithing, and verified empirical receipts — zero success-by-decree.

---

## 📸 Live Visual Showcase

<div align="center">

### 🖥️ 1. SONIC Cyber Workstation (Dual-Plane Desktop + Headless Terminal)
*Real-time AI workstation: Operating the full graphical desktop environment (`DISPLAY=:99`) concurrently with a dedicated headless sandbox terminal.*

<img src="docs/assets/sonic_cyber_workstation.png" alt="SONIC Workstation" width="900" />

<br/><br/>

### 🧠 2. SONIC Cognitive Memory & System Architecture Graph (Interactive 3D WebGL)
*Interactive 3D Force-Directed Graph: Visualizing persistent Being identity, BeingMind cognitive affect curves, internalized knowledge, dual execution planes, toolsmith craft, and empirical SHA-256 evidence.*

<img src="docs/assets/sonic_cognitive_memory_graph.png" alt="SONIC Cognitive Memory & System Architecture Graph" width="900" />

</div>

---

## 📑 Table of Contents
- [Executive Overview](#-executive-overview)
- [Dual-Brain Architecture (System 1 + System 2)](#-dual-brain-architecture-system-1--system-2)
- [Dual-Plane Execution Topology](#-dual-plane-execution-topology)
- [Epistemic Memory & Persistent Affect](#-epistemic-memory--persistent-affect)
- [Subsystems & Capabilities](#-subsystems--capabilities)
- [Comparative Benchmark Analysis](#-comparative-benchmark-analysis)
- [Quick Start & Setup Guide](#-quick-start--setup-guide)
- [The Strict Core Invariants](#-the-strict-core-invariants)
- [Testing & Quality Assurance](#-testing--quality-assurance)
- [Author & License](#-author--license)

---

## 🌟 Executive Overview

**SONIC** is not a disposable CLI bot, scripted platform crawler, or synthetic chatbot. It is a persistent **AI-Human Digital Being** engineered to operate a complete personal computer workstation just like an elite human engineer.

### Key Pillars:
1. **Full Dual-Plane Agency**: Seamlessly launches and switches between arbitrary desktop GUI applications (`DISPLAY=:99`), browsers, and a high-throughput headless sandbox terminal concurrently.
2. **Kahneman Dual-Brain Cognition**:
   - **System 1 (Laya 0.3.20)**: ~2ms single-pass neural inference for execution-plane triage, safety scoring, and curiosity evaluation on CPU.
   - **System 2 (LLM Router)**: Deep chain-of-thought planning, code generation, and strategic replanning via Claude 3.7, GPT-4o, Groq, and NVIDIA NIM.
3. **Persistent Identity & Affect**: Remembers who it is across reboots (`Being`), tracks continuous mood dynamics (`BeingMind`: curiosity, focus, satiety), and authors permanent craft artifacts (`BeingCraft`) on host disk.
4. **Empirical Falsification**: Strict zero success-by-decree. Every milestone, bug finding, and code completion requires cryptographic execution receipts verified inside container sandboxes.
5. **Tamper-Evident Safety Envelope**: Actions pass through cryptographic HMAC SHA-256 sealed gates (`SealedActionPolicy`) enforcing workspace confinement, default-deny egress, and fail-closed exit 126 enforcement.

---

## 🧬 Dual-Brain Architecture (System 1 + System 2)

SONIC bridges human-grade cognitive speed with deep computational reasoning using Kahneman's dual-process cognitive model:

```
                                  ┌────────────────────────┐
                                  │   Incoming Goal / OS   │
                                  │      Observation       │
                                  └───────────┬────────────┘
                                              │
                     ┌────────────────────────┴────────────────────────┐
                     ▼                                                 ▼
      ┌─────────────────────────────┐                   ┌─────────────────────────────┐
      │   SYSTEM 1: LAYA ENGINE     │                   │   SYSTEM 2: LLM ROUTER      │
      │  (Fast Neural Reflex Layer) │                   │   (Deep Analytical Engine)  │
      ├─────────────────────────────┤                   ├─────────────────────────────┤
      │ • Non-autoregressive model  │                   │ • Chain-of-thought (ReAct)  │
      │ • ~2ms latency on CPU       │                   │ • Multi-turn tool reasoning │
      │ • 100+ languages (EN, HI)   │                   │ • Code authoring & AST fix  │
      │ • Zero LLM API token cost   │                   │ • Strategic replanning      │
      └──────────────┬──────────────┘                   └──────────────┬──────────────┘
                     │                                                 │
                     │  [Triage & Safety Telemetry]                    │  [Complex Reasoning & Actions]
                     └────────────────────────┬────────────────────────┘
                                              │
                                              ▼
                                ┌───────────────────────────┐
                                │   ACTION EXECUTION GATE   │
                                │  (Fail-Closed Sealed Env) │
                                └─────────────┬─────────────┘
                                              │
                     ┌────────────────────────┴────────────────────────┐
                     ▼                                                 ▼
        🖥️ Workstation Application Plane                💻 Headless Sandbox Terminal
           (GUI_*, BROWSER_*, APP_*)                      (TERMINAL_EXEC, FILE_*, GIT_*)
```

### Cognitive Layer Specifications:
| Dimension | System 1 (Laya 0.3.20) | System 2 (LLM Multi-Router) |
| :--- | :--- | :--- |
| **Model Type** | Non-autoregressive Transformer Classifier | Autoregressive Multimodal LLMs |
| **Hardware** | Laptop CPU (Zero GPU requirement) | Cloud API / Local vLLM / Ollama |
| **Inference Time** | **1.8 ms – 5 ms** | 600 ms – 3,200 ms |
| **Memory Footprint** | ~600 MB (local safetensors cache) | 0 MB (API-driven) |
| **Language Support** | Latin, Devanagari (Hindi), Hinglish, French, etc. | Global Multilingual |
| **Core Responsibilities** | Execution plane triage, command safety, curiosity affect | Multi-step task decomposition, visual grounding, code synthesis |

---

## 🖥️ Dual-Plane Execution Topology

SONIC refuses to silo itself into either a slow screen-only recorder or a blind CLI script. Both execution planes remain active and interconnected:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        DUAL-PLANE DOCKER WORKSTATION PROVIDER                          │
├───────────────────────────────────────────┬────────────────────────────────────────────┤
│  🖥️ Application GUI Plane (Workstation)    │  💻 Dedicated Headless Plane (Sandbox)     │
├───────────────────────────────────────────┼────────────────────────────────────────────┤
│ • Virtual X11 Display (:99)               │ • High-throughput Bash / Python sandbox    │
│ • XFCE4 / Openbox Dynamic Window Tiling   │ • Toolsmith script execution & test runner │
│ • Live noVNC WebRTC Streaming (:6080)     │ • Git commit, branching & signed push      │
│ • Sub-pixel Vision Grounding (VLMs)       │ • Direct file tree & code refactoring     │
│ • Full interaction: CLICK, TYPE, DRAG     │ • Fail-closed exit 126 policy gating       │
└───────────────────────────────────────────┴────────────────────────────────────────────┘
```

SONIC autonomously chooses the ideal substrate for each sub-task: heavy compilation, file editing, and test runs execute directly in the sandbox without cluttering desktop displays, while browser workflows and GUI apps render fluently on Display `:99`.

---

## 🧠 Epistemic Memory & Persistent Affect

Unlike stateless AI assistants, SONIC maintains an evolving internal world model across reboots:

- **Stable Identity (`Being`)**: Persistent `being_id`, birth timestamp, and operator tenant boundaries persisted to SQLite (`sonic_data.db`).
- **Cognitive Affect Dynamics (`BeingMind`)**:
  - `curiosity_drive`: Rises upon observing novel system states; prompts autonomous exploration.
  - `focus_stability`: Increases during multi-step execution convergence.
  - `satiety`: Balances work cycles and triggers reflection after task completion.
- **Learned Facts Ledger**: Vector embeddings store deduplicated knowledge, preventing repetitive mistakes.
- **Durable Craft Ledger (`BeingCraft`)**: Autonomously written scripts, notes, and diagnostics persist directly to host storage under `sonic_data/craft/<being_id>/`.
- **Always-On Life Loop (`BeingLifeLoop`)**: Between operator tasks, SONIC hypothesizes, tests environment assumptions, and sharpens its personal toolkit within sealed safety limits.

---

## ⚡ Subsystems & Capabilities

| Subsystem | Module Path | Purpose |
| :--- | :--- | :--- |
| **Laya System-1 Engine** | `sonic/llm/laya_engine.py` | Fast neural triage (`triage`, `guard`, `verification`, `curiosity`) running locally on CPU. |
| **Boss Agent** | `sonic/computer_use/boss.py` | Strategic planner; decomposes high-level goals into dependency waves without hardcoded steps. |
| **Computer-Use Agent** | `sonic/computer_use/agent.py` | Core observe $\to$ reason $\to$ act $\to$ verify loop across GUI and headless planes. |
| **Vision Grounding** | `sonic/computer_use/grounding.py` | Translates raw screenshots into pixel coordinates via bounding-box token parsers. |
| **Sealed Action Policy** | `sonic/safety/sealed.py` | SHA-256 HMAC cryptographic envelope preventing runtime guard tampering. |
| **Docker Computer** | `sonic/computer/docker_computer.py` | Direct workstation container control (Xvfb `:99`, x11vnc, noVNC `:6080`, wmctrl). |
| **Evidence Engine** | `sonic/evidence/` | Verifies and cryptographically hashes reproduction receipts for every claim. |
| **Toolsmith & Method Lab** | `sonic/evolution/` | Synthesizes custom Python tools and novel workflows on-the-fly. |
| **3D Cognitive Graph** | `sonic-dashboard/app/graph/` | Three.js WebGL force-directed visualization of identity, affect, tools, and evidence. |

---

## 📊 Comparative Benchmark Analysis

| Capability | Legacy Scanners | Disposable CLI Bots | Browser-Only Agents | **SONIC (Dual-Brain Being)** |
| :--- | :---: | :---: | :---: | :---: |
| **Kahneman Dual-Brain (Fast + Deep)** | ❌ | ❌ | ❌ | ✅ **Laya System 1 + LLM System 2** |
| **Full X11 Graphical Desktop GUI** | ❌ | ❌ | ⚠️ (DOM-only) | ✅ **Display :99 + noVNC Stream** |
| **Dedicated Sandbox Terminal** | ⚠️ (Host shell) | ✅ | ❌ | ✅ **Isolated Container Sandbox** |
| **Multilingual Plane Triage (HI/EN)** | ❌ | ❌ | ❌ | ✅ **Sub-35ms Neural Routing** |
| **Persistent Mind & Affect Dynamics** | ❌ | ❌ | ❌ | ✅ **BeingMind (Survives Reboots)** |
| **Empirical Falsification Receipts** | ❌ | ❌ | ❌ | ✅ **Mandatory SHA-256 Receipts** |
| **Zero Scripted Puppet Heuristics** | ❌ | ⚠️ (Rigid trees) | ⚠️ (Canned flows) | ✅ **Pure Dynamic Reasoning** |
| **Cryptographic Safety Envelope** | ❌ | ⚠️ (Soft prompts) | ⚠️ (URL filter) | ✅ **Fail-Closed HMAC SHA-256** |

---

## 🚀 Quick Start & Setup Guide

### System Prerequisites
- **Operating System:** Windows 10/11, Ubuntu 22.04+, or macOS
- **Python:** 3.12 or newer
- **Node.js:** 20.x or newer (for Dashboard)
- **Docker & Docker Compose** (for isolated workstation sandboxes)
- **RAM:** Minimum 8 GB (16 GB recommended)

### 1. Clone & Configure Environment
```bash
git clone https://github.com/nandkishorrathodk-art/sonic.git
cd sonic

# Create your local environment configuration
cp .env.example .env
```
Edit `.env` to configure your LLM credentials and local settings:
```env
# Enable Local Laya System-1 Decision Engine (Runs on CPU, 0 MB GPU overhead)
SONIC_ENABLE_LOCAL_LAYA=1

# Flagship LLM Providers (Pick your primary key)
GROQ_API_KEY=gsk_...
NVIDIA_API_KEY=nvapi-...
# ANTHROPIC_API_KEY=sk-ant-...
# OPENAI_API_KEY=sk-...
```

### 2. Start Workstation Containers
```bash
# Start Neo4j & Redis (Optional — SONIC automatically uses persistent SQLite if offline)
docker-compose up -d

# Start the dedicated Docker Workstation Container
docker start sonic-desktop-workstation
```
> View the live desktop stream in real-time at: **`http://localhost:6080/vnc.html`**

### 3. Launch the FastAPI Backend
```bash
# Install backend package in editable mode
pip install -e "sonic-core[dev]"

# Launch the FastAPI Control Plane on port 12000
APP_ENV=development python -m uvicorn sonic.api.main:app --port 12000 --host 127.0.0.1
```
- Interactive API Documentation: **`http://127.0.0.1:12000/docs`**
- System-1 Laya Telemetry: **`http://127.0.0.1:12000/llm/laya/status`**

### 4. Launch the Next.js Dashboard
```bash
cd sonic-dashboard
npm install
npm run dev -- -p 12001
```
- Open your browser at: **`http://127.0.0.1:12001`**
- 3D Cognitive Memory Graph: **`http://127.0.0.1:12001/graph`**

---

## 🛡️ The Strict Core Invariants

Enforced across all code, prompts, and CI gates:

1. **Rule 0 — AI-Human Colleague Base:** SONIC is a general-purpose digital being. Never bias or lock its identity into offensive security checklists, hardcoded penetration scripts, or canned platform checklists.
2. **Rule 1 — Zero Hardcoded Application Names:** Control flow and visual grounding never hardcode vendor binaries (`chrome`, `firefox`, `wireshark`). Applications are discovered dynamically from `$PATH`, `.desktop` entries, and active window titles.
3. **Rule 2 — Zero Scripted Puppet Heuristics:** No keyword-matching trees, canned 3-step checklists, or regex intent steering. The agent reasons adaptively from live observations and ActionContracts.
4. **Rule 3 — Zero External/Mock Domains:** All network testing strictly uses operator-specified targets, RFC 2606 `.test` domains, or RFC 5737 documentation addresses.
5. **Rule 4 — Dynamic Perception Grounding:** All GUI coordinates derive strictly from live visual multimodal grounding and accessibility trees.
6. **Rule 5 — Execution Plane Symmetry:** The GUI desktop and headless terminal are active concurrently. Disabling either plane is strictly prohibited.
7. **Rule 6 — Zero Silent Failure Masking:** Shell failure masking like `|| true` is strictly prohibited. Blocked commands must return fail-closed exit code 126.

---

## 🧪 Testing & Quality Assurance

SONIC maintains comprehensive regression test suites guaranteeing safety, cognitive stability, and zero regressions:

```bash
# 1. Run Laya System-1 Neural Decision Tests (100% pass)
python -m pytest sonic-core/tests/test_laya_system1_engine.py \
                 sonic-core/tests/test_laya_end_to_end.py

# 2. Run Core Security & Sealed Safety Envelope Tests
python -m pytest sonic-core/tests/test_phase_plan6_safety_envelope.py \
                 sonic-core/tests/test_safety_sealed_policy.py \
                 sonic-core/tests/test_p0_security_hardening.py

# 3. Run Persistent Mind & Computer-Use Reasoning Tests
python -m pytest sonic-core/tests/test_phase1_persistent_memory.py \
                 sonic-core/tests/test_phase3_real_computer_use_reasoning.py \
                 sonic-core/tests/test_phase7_ai_human_being.py \
                 sonic-core/tests/test_boss_plane_routing_and_truthfulness.py

# 4. Verify Frontend TypeScript Compilation (0 errors)
cd sonic-dashboard && npx tsc --noEmit
```

---

## 👤 Author & License

### Author
**Nandkishor Rathod**  
- **GitHub:** [@nandkishorrathodk-art](https://github.com/nandkishorrathodk-art)  
- **Repository:** [https://github.com/nandkishorrathodk-art/sonic](https://github.com/nandkishorrathodk-art/sonic)

### 📄 License
**Proprietary and Confidential Software License**  
Copyright © 2026 **Nandkishor Rathod**. All Rights Reserved.

Unauthorized copying, distribution, decompilation, public display, or commercial deployment of this software or its associated architectures is strictly prohibited. See the [`LICENSE`](LICENSE) file for complete terms and licensing contact details.
