# SONIC — System Definition & Architecture

> **Status:** Grounded in the actual codebase (`sonic-core/sonic/`).
> Every claim maps to real modules, classes, and invariants in the source.

---

## 1. Formal Definition

SONIC is a **fail-closed, tamper-evident, self-improving Autonomous AI-Human
Digital Being** that operates its own dedicated personal computer workstation
(`DISPLAY=:99` full graphical desktop + dedicated headless terminal) to perform
software engineering, web browsing, account management, empirical research, and
verification — generating and empirically falsifying hypotheses, and authoring
its own tools and methods inside an unbreachable safety envelope that the agent
cannot widen at runtime.

### Plain-language version

> "An autonomous AI-Human colleague operating its own isolated computer
> workstation (`DISPLAY=:99` GUI + headless terminal), perceiving applications
> multimodally (screenshot + accessibility tree + CDP DOM + terminal),
> launching dynamic subagent teams in isolated branch workspaces, testing
> hypotheses empirically in-sandbox, and authoring reusable tools (`BeingCraft`,
> `Toolsmith`) — with every action governed by a tamper-evident safety policy."

---

## 2. What makes this a complete AI-Human Digital Being

| Capability | Module | What it actually does |
|---|---|---|
| **Own computer** | `computer/`, `sandbox/` | Provisions isolated cloud/Docker sandboxes with real GUI desktops (`DISPLAY=:99`) |
| **Multi-modal perception** | `daytona_computer.py` | Real screenshots + AT-SPI accessibility-tree text extraction + CDP DOM grounding |
| **Closed-loop reasoning** | `computer_use/agent.py` | observe → reason(LLM) → act → verify, goal-aware early stop |
| **Dynamic subagent teamwork** | `computer_use/teamwork.py` | `DEFINE_SUBAGENT`, `INVOKE_SUBAGENT`, `SEND_MESSAGE`, `MANAGE_SUBAGENTS` with `branch` isolation + `JudgeAgent` verification |
| **Epistemic reasoning** | `research/epistemic.py` | First-class Unknowns, competing hypotheses, contradiction tracking |
| **Experiment design** | `research/experiment_designer.py` | Discriminating experiments to separate hypotheses + adversarial falsification |
| **Information-gain selection** | `research/information_gain.py` | Ranks actions by expected information/confidence gain vs cost/risk |
| **Tool authoring** | `being/toolsmith.py` | Writes new tools, registers them ONLY after in-sandbox verification (exit 0 + non-empty) |
| **Method invention** | `being/method_lab.py` | Synthesizes novel workflows/methods, confirms only after empirical reproduction |
| **Evidence & trust** | `evidence/` | Custody chain, independent adversarial verifier, false-positive dedup, confidence bands |
| **Self-directed life loop** | `being/life_loop.py` | Always-on curiosity tick that pursues goals even without operator prompts |
| **Tamper-evident safety** | `safety/sealed.py` | SHA-256-sealed policy; runtime mutation → DENY |
| **Fail-closed execution** | `sandbox/providers/` | Every provider: exit 126 + "host fallback prohibited" on any failure |

---

## Persistent AI Employee Organization

`sonic/organization/architecture.py` is the canonical repository ownership
manifest. It maps cognitive, computer, platform, safety, evaluation,
experience, and research subsystems to departments and teams. The
tenant-scoped `sonic/organization/store.py` mirrors that manifest in a
persistent subsystem registry and validates employee ownership and work-item
assignment against it. The dashboard organization page reads and updates this
data through `/organization`. Role profiles are
not human user accounts and do not claim that an agent is currently running.
Subagent calls may include an `employee_id`; the runtime resolves that profile
within the active tenant, uses its role prompt and ownership context, and
derives a sealed, read-only action allowlist when the profile is read-only.

Issue and pull-request URLs are references rather than an outbound GitHub
automation feature. Configure `SONIC_GITHUB_WEBHOOK_SECRET` and a GitHub webhook
for `pull_request`, `pull_request_review`, and `check_suite` to record signed
review and aggregate CI receipts. Only those verified webhook events can set
review/evaluation results or release a merged work item. This registry does not
create GitHub issues, branches, or pull requests, or merge/push code. Ordinary
work-item status and assignment remain operator-managed. A work item can now
follow the enforced chain `department -> team -> subsystem -> employee ->
project -> work item`; release still requires signed external GitHub receipts.

---

## 3. The Six-Layer Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│  L6  Being Life Loop        — always-on curiosity, self-directed  │
│      (being/life_loop.py)      goals within the safety envelope   │
├──────────────────────────────────────────────────────────────────┤
│  L5  Research Engine         — hypotheses, experiments,           │
│      (research/, researcher/)   information-gain action selection  │
├──────────────────────────────────────────────────────────────────┤
│  L4  ComputerUseAgent        — closed loop: observe→reason→act→   │
│      (computer_use/agent.py)   verify, with vision-in-the-loop    │
├──────────────────────────────────────────────────────────────────┤
│  L3  Safety Gate (Sealed)    — fail-closed, tamper-evident        │
│      (safety/sealed.py)        SHA-256 seal, frozen egress snapshot│
├──────────────────────────────────────────────────────────────────┤
│  L2  Computer Provider       — GUI + terminal + fs + git + apps   │
│      (computer/)               real mouse/keyboard, AT-SPI text    │
├──────────────────────────────────────────────────────────────────┤
│  L1  Compute Sandbox         — isolated workspace, execute/destroy│
│      (sandbox/)                Daytona / Docker / LocalDev         │
└──────────────────────────────────────────────────────────────────┘
```

### Data flow (one cognitive step)

```
Being Life Loop (L6)
  └─ proposes curiosity goal from live observation
Research Engine (L5)
  └─ generates hypotheses + designs discriminating experiment
  └─ information-gain ranks candidate actions
ComputerUseAgent (L4)
  └─ observe(): screenshot + status + files + git + terminal + DOM
  └─ choose_action(): LLM(weights=observation+history+hypotheses)
  └─ execute_action():
       │
       ▼
   Safety Gate (L3) ── DENY? → BLOCKED, never reaches provider
       │              (rate / allowlist / path / command / egress)
       ▼ ALLOW
   Computer Provider (L2)
     ├─ GUI? → DaytonaComputerProvider → SDK mouse/keyboard (`x-www-browser` / `xdg-open`)
     └─ Terminal? → UnifiedComputerProvider → compute.execute()
       │
       ▼
   Compute Sandbox (L1)
     └─ sandbox.process.exec() / docker exec → real isolated execution
       │
       ▼
   ExecResult / ScreenObservation (returned up the stack)
       │
       ▼
   Evidence Engine (L5) — custody chain, confidence scoring, dedup
   World Model (L5) — update verified knowledge, stop conditions
```

---

## 4. The Five Pillars (each a real, auditable invariant)

### Pillar 1 — Fail-Closed Isolation (L1)

**Invariant:** *A failed sandbox never falls back to the host OS.*

- `ComputeProvider` is an ABC with a 6-method contract: `create_workspace`,
  `execute`, `read_file`, `write_file`, `destroy_workspace`, `get_state`.
- **Every** concrete provider enforces the same fail-closed return on any
  failure: `exit_code=126`, `stderr="FAIL-CLOSED: Host fallback is prohibited."`
- Provider selection (`sandbox/factory.py`): Daytona → Docker → LocalDev.
  The `LocalDevProvider` blocks all commands unless `allow_host_execution=True`,
  which a **hard guard** refuses outside `APP_ENV=development`.
- Resources are capped (`--memory`, `--cpus`, `--security-opt no-new-privileges`).
- Timeout kills the **entire process group** (`start_new_session=True` +
  `os.killpg`), not just the shell — no orphaned child processes leak.

### Pillar 2 — Tamper-Evident Safety Envelope (L3)

**Invariant:** *The agent cannot widen its own permissions at runtime.*

The `ActionPolicy` is sealed via `SealedActionPolicy`:

1. At `seal()` time, a **SHA-256 hash** is recorded over the canonical
   serialization of every safety-relevant field (allowed_types,
   security_tool_targets, rate cap, approval flag, workspace root, **and a copy
   of the egress blocked-networks list**).
2. After sealing, mutation is **refused** — fields are frozen (frozenset,
   tuple).
3. On **every** `evaluate()`, the seal hash is recomputed and compared.
   Mismatch → `DENY` + `safety_policy_tampered` log.
4. The egress check uses the **frozen snapshot**, so runtime mutation of
   `egress.BLOCKED_NETWORKS` cannot widen what this policy permits.

The gate is a 5-step fail-closed pipeline, evaluated **before** any action
reaches the provider:

```
1. Rate limit          (max actions/min)
2. Action allowlist    (unknown type → DENY)        ← fail-closed
3. Path confinement    (workspace-root traversal → DENY)
4. Command risk        (L2_FORBIDDEN → DENY; L1 → approval gate)
5. Egress filter       (metadata/loopback/RFC1918 → DENY)
```

### Pillar 3 — Defence-in-Depth Egress (L3↔L1)

**Invariant:** *No network path reaches cloud metadata, loopback, or private ranges — including via IPv6 encodings.*

- Default-deny: a target is allowed only if it is not in any blocked range.
- Blocked: `169.254.169.254/32` (metadata), `127.0.0.0/8`, `10.0.0.0/8`,
  `172.16.0.0/12`, `192.168.0.0/16`, `0.0.0.0/8`, `::1/128`, `fc00::/7` (ULA).
- **IPv4-mapped IPv6**: `::ffff:169.254.169.254` is blocked via
  `::ffff:0.0.0.0/96` and a normalization helper (`_ip_blocked`) that converts
  mapped v6 → v4 before checking.
- **NAT64** well-known prefix `64:ff9b::/96` is also blocked.
- Domains are **DNS-resolved** and every resolved IP is checked.

### Pillar 4 — Epistemic, Falsification-Driven Research (L5)

**Invariant:** *The system tracks what it does NOT know and designs experiments to resolve it.*

- **Unknown-First Reasoning** (`research/epistemic.py`): `Unknown` is a
  first-class object with importance, confidence, possible actions.
- **Competing Hypotheses**: multiple explanations held simultaneously, each
  with evidence links and a confidence breakdown.
- **Experiment Designer**: generates *discriminating experiments* to separate
  competing hypotheses, plus **adversarial falsification challenges**.
- **Information-Gain Selection** (`research/information_gain.py`): ranks
  candidate actions by expected information gain / confidence gain vs cost.

### Pillar 5 — Honest Self-Improvement (L6)

**Invariant:** *A self-authored tool or method is registered ONLY after it reproduces a real result in-sandbox — never by decree.*

- **Toolsmith** (`being/toolsmith.py`): authors a new utility/script when no
  registered tool covers an observation gap, and registers it **only after**
  `confirm_and_register()` runs it in the sandbox and gets `exit 0` + non-empty
  stdout.
- **Method Lab** (`being/method_lab.py`): synthesizes novel workflows and
  confirms them only after empirical verification.
- **Being Life Loop** (`being/life_loop.py`): an always-on `asyncio.Task` that
  runs when no operator goal is active.

---

## 5. Multi-Modal Perception — More Than Pixels

`DaytonaComputerProvider` captures a *structured* observation, not just a bitmap:

| Channel | Source | Why it matters |
|---|---|---|
| **Screenshot** | `sandbox.computer_use.screenshot.take_full_screen()` | Vision model input |
| **Visible text** | AT-SPI accessibility tree walk (`_extract_visible_text`) | Agent reads on-screen text directly, not via OCR |
| **Detected controls** | AT-SPI node roles (button, link, entry) | Structured click targets |
| **Terminal output** | `computer.terminal("echo __sonic_obs_ready__")` | Real shell state |
| **Filesystem** | `computer.list_files()` | What exists on disk |
| **Git state** | `computer.git_action("status")` | Branch + cleanliness |
| **Browser DOM** | `_observe_browser()` (CDP DOM grounding via `x-www-browser`) | URL, title, interactive elements |

**"Never fabricate a pixel, never fabricate content."** If no live desktop
exists, `screenshot()` returns `desktop_state="NO_DISPLAY"` with an empty
bitmap — it does not synthesize a fake screen.

---

## 6. End-to-End Call Chains

### A. GUI click at (350, 200)

```
BeingLifeLoop.tick() → propose("open x-www-browser")
  └─ ComputerUseAgent.run_mission()
       ├─ observe() → screenshot + AT-SPI text + status + files + git + DOM
       ├─ choose_action() → LLM → (GUI_CLICK, "(350,200)", {x:350,y:200})
       └─ execute_action(GUI_CLICK, ...)
            ├─ SealedActionPolicy.evaluate("GUI_CLICK", ...)  ← seal re-checked
            │   └─ ALLOW (rate OK, type allowed)
            └─ DaytonaComputerProvider.gui_action(CLICK, x=350, y=200)
                 └─ sandbox.computer_use.mouse.click(350, 200, "left")
                      └─ Daytona SDK → remote X11 desktop, real mouse event
```

### B. Run diagnostic command on `app.target.test`

```
execute_action(TERMINAL_EXEC, "app.target.test", {command:"curl -I https://app.target.test"})
  └─ SealedActionPolicy.evaluate("TERMINAL_EXEC", ...)
       ├─ rate OK
       ├─ type allowed
       └─ _check_command("curl -I https://app.target.test")
  └─ UnifiedComputerProvider.terminal()
       └─ compute.execute(workspace_id, "curl -I https://app.target.test", timeout=60)
```

### C. Browse to `http://internal.target.test/` (SSRF attempt blocked)

```
execute_action(BROWSER_NAVIGATE, "internal.target.test", {url:"http://internal.target.test/"})
  └─ SealedActionPolicy.evaluate("BROWSER_NAVIGATE", ...)
       └─ _check_egress("http://internal.target.test/", "browser")
            └─ egress.is_target_allowed()
                 ├─ DNS resolve → 127.0.0.1
                 ├─ 127.0.0.0/8 in frozen BLOCKED_NETWORKS → True
                 └─ return (False, "Target IP 127.0.0.1 is in blocked network")
  └─ verdict.allowed=False → status="BLOCKED", reason logged
  └─ return trace (NEVER reaches provider — no recovery)
```

### D. Self-author a tool (honest loop)

```
BeingLifeLoop.tick() → observation gap: no parser for X format
  └─ Toolsmith.author(name="parse_x", source=<LLM-authored Python>)
       └─ BeingCraft.persist() (survives restart)
       └─ confirm_and_register(name, source)
            └─ compute.execute(ws, "python /path/parse_x.py")
                 └─ run strictly in sandbox
            ├─ exit 0 + non-empty stdout? → reproduced=True, register
            └─ exit 126 / fail / empty?    → reproduced=False, NOT registered
```

---

## 7. Authorization & Trust Boundary

- **Scope**: `safety/scope.py` — `ScopeChecker.is_target_in_scope` uses
  `fullmatch` (not `endswith`); `_domain_to_regex` escapes then restores globs.
  Subdomain enumeration uses `_is_subdomain_of` (dot-boundary check) —
  `evil-target.test` is NOT accepted as a subdomain of `target.test`.
- **Tenancy**: sessions and engagements are tenant-scoped; cross-tenant runs are refused.
- **Audit**: every `UnifiedComputerProvider` op records a `ComputerAuditEvent`
  (actor, action, resource, result).
- **Evidence custody**: `evidence/custody.py` — provenance chain for every
  finding; `evidence/independent_verifier.py` — adversarial reviewer that
  challenges claims to prevent self-confirmation bias.
