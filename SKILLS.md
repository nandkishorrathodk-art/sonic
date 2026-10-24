# SONIC — Capability & Skill Catalog

This file is a **machine-readable capability catalog** for SONIC's computer-use
reasoning. It is not marketing: every entry is grounded in a real code symbol,
and each carries an honest `status` so the agent never assumes a capability it
does not have.

The block below is parsed at runtime by
`services/core/sonic/computer_use/skill_catalog.py` and injected into the
`ComputerUseAgent` reasoning prompt (including every Boss sub-agent worker), so
a worker knows which skills exist and which are fail-closed. If this file is
missing or malformed, the loader returns an empty catalog (fail-closed) — it
never fabricates a skill.

`status` meanings:

| status | meaning |
| --- | --- |
| `available` | Implemented and reachable from the production agent loop. |
| `conditional` | Works only when a dependency is configured (e.g. an evaluated AVCS head, an LLM provider key, a sandbox provider). Fails closed otherwise. |
| `not_wired` | Code exists but no production caller constructs it. Do not claim it works. |

<!-- SKILL-CATALOG:START -->
```json
{
  "version": 1,
  "skills": [
    {
      "name": "desktop_app_control",
      "category": "workstation",
      "status": "available",
      "actions": ["APP_LAUNCH", "APP_FOCUS", "APP_CLOSE", "APP_MAXIMIZE", "WINDOW_MAXIMIZE", "WINDOW_TILE"],
      "summary": "Launch, focus, tile and close arbitrary installed GUI applications discovered from the live desktop.",
      "evidence": "sonic/computer_use/agent.py, sonic/computer/docker_computer.py"
    },
    {
      "name": "pixel_pointer_and_keyboard",
      "category": "workstation",
      "status": "available",
      "actions": ["GUI_MOVE", "GUI_CLICK", "GUI_DOUBLE_CLICK", "GUI_RIGHT_CLICK", "GUI_DRAG", "GUI_SCROLL", "GUI_KEYPRESS", "GUI_HOTKEY", "GUI_WAIT"],
      "summary": "Pointer and keyboard control with smooth cursor interpolation. Explicit pixel-coordinate targets dispatch; semantic targets require the grounder.",
      "evidence": "sonic/computer_use/agent.py, sonic/computer_use/sonic_cursor.py"
    },
    {
      "name": "semantic_gui_grounding",
      "category": "perception",
      "status": "conditional",
      "actions": ["GUI_CLICK", "GUI_TYPE"],
      "summary": "Resolve a natural-language UI target ('the Save button') to screen pixels. Blocked unless an evaluated local AVCS grounding checkpoint is configured.",
      "evidence": "sonic/computer_use/avcs_perception.py, packages/avcs/avcs/grounding.py",
      "requires": "SONIC_AVCS_GROUNDING_CHECKPOINT"
    },
    {
      "name": "focus_verified_text_entry",
      "category": "perception",
      "status": "conditional",
      "actions": ["GUI_TYPE", "BROWSER_TYPE"],
      "summary": "Type text only after an evaluated local focus-verification head confirms the target field is focused. Blocked otherwise.",
      "evidence": "packages/avcs/avcs/focus_verifier.py, sonic/computer_use/avcs_perception.py",
      "requires": "SONIC_AVCS_FOCUS_CHECKPOINT"
    },
    {
      "name": "avcs_visual_perception",
      "category": "perception",
      "status": "available",
      "actions": ["GUI_SCREENSHOT", "BROWSER_SCREENSHOT"],
      "summary": "Local pixel reflex: frame identity, dimensions, scene fingerprint, change/motion/regions. Advisory only — it does not identify text, controls or task success.",
      "evidence": "sonic/computer_use/avcs_perception.py, packages/avcs/avcs/reflex.py"
    },
    {
      "name": "learned_visual_cortex",
      "category": "perception",
      "status": "conditional",
      "summary": "Optional trained cortex producing scene category and affordance hotspots. Off until a checkpoint is configured.",
      "evidence": "sonic/computer_use/avcs_perception.py, packages/avcs/forge/vision/cortex",
      "requires": "SONIC_AVCS_CORTEX_CHECKPOINT"
    },
    {
      "name": "headless_execution",
      "category": "execution",
      "status": "available",
      "actions": ["TERMINAL_EXEC", "SERVICE_ACTION", "FILE_READ", "FILE_WRITE"],
      "summary": "Operator/sandbox plane: run commands, read and write workspace files, manage services without cluttering the visible desktop.",
      "evidence": "sonic/computer/headless.py, sonic/computer_use/agent.py"
    },
    {
      "name": "git_operations",
      "category": "execution",
      "status": "available",
      "actions": ["GIT_STATUS", "GIT_DIFF", "GIT_BRANCH", "GIT_COMMIT", "GIT_PUSH"],
      "summary": "Inspect and commit repository state inside the workspace.",
      "evidence": "sonic/computer_use/agent.py"
    },
    {
      "name": "security_tooling",
      "category": "security",
      "status": "available",
      "actions": ["SECURITY_TOOL"],
      "summary": "Run registered in-sandbox security adapters (nmap, nuclei, ffuf, http_client). Every target is egress-filtered; fail-closed on blocked execution.",
      "evidence": "sonic/tools/registry.py, sonic/tools/adapters/"
    },
    {
      "name": "tool_authoring",
      "category": "self_evolution",
      "status": "available",
      "actions": ["TOOL_AUTHOR", "TOOL_RUN"],
      "summary": "Author a new small tool for a gap no registered tool covers, then confirm it by a real in-sandbox run before registering it.",
      "evidence": "sonic/being/toolsmith.py"
    },
    {
      "name": "method_invention",
      "category": "self_evolution",
      "status": "available",
      "actions": ["METHOD_INVENT"],
      "summary": "Synthesize a novel technique biased away from the known-method ledger, confirmed only on real in-sandbox reproduction.",
      "evidence": "sonic/being/method_lab.py"
    },
    {
      "name": "cross_mission_lessons",
      "category": "memory",
      "status": "available",
      "summary": "Agentic-RAG retrieval of past REUSE/AVOID lessons, injected into the reasoning prompt.",
      "evidence": "sonic/being/lessons.py, sonic/memory/agentic.py"
    },
    {
      "name": "multi_context_retrieval",
      "category": "memory",
      "status": "available",
      "summary": "Fuse vector memory, graph findings and episodic history into one bounded, deduplicated, cited context pack; optionally refined by LLM query planning.",
      "evidence": "sonic/memory/multi_context.py"
    },
    {
      "name": "verified_computer_skills",
      "category": "memory",
      "status": "not_wired",
      "summary": "State-conditioned verified skills and replay-promoted workflows learned from real traces. The ledger exists and is tested, but no production caller constructs it.",
      "evidence": "sonic/computer_use/skill_ledger.py"
    },
    {
      "name": "delegation",
      "category": "orchestration",
      "status": "available",
      "actions": ["DEFINE_SUBAGENT", "INVOKE_SUBAGENT", "SEND_MESSAGE", "MANAGE_SUBAGENTS"],
      "summary": "Define, dispatch and manage sub-agents, each addressed by a run-unique agent_uid.",
      "evidence": "sonic/computer_use/agent.py, sonic/computer_use/boss.py"
    },
    {
      "name": "verification_and_replan",
      "category": "orchestration",
      "status": "available",
      "actions": ["VERIFY_PROBE", "GOAL_COMPLETE"],
      "summary": "Independent verification probe and stuck-detection/replan. GOAL_COMPLETE is a proposal, not proof; the Judge confirms from fresh evidence.",
      "evidence": "sonic/computer_use/agent.py, sonic/agents/judge.py"
    },
    {
      "name": "dual_process_reasoning",
      "category": "reasoning",
      "status": "available",
      "summary": "Fast/deep mode selection from observable uncertainty signals, plus fast System-1 triage.",
      "evidence": "sonic/orchestration/dual_process.py, sonic/llm/laya_engine.py"
    }
  ]
}
```
<!-- SKILL-CATALOG:END -->

## How this is used

- `load_skill_catalog()` reads and validates the JSON block, then renders a
  compact prompt section listing each skill with its status and actions.
- The section is advisory routing context. It does not grant a capability and it
  does not override the safety envelope: `ActionPolicy`/`SealedActionPolicy`,
  egress filtering and path confinement still decide every action.
- A `conditional` skill listed as unavailable must be treated as blocked; the
  agent should choose a different path rather than assume it works.
- A `not_wired` skill must never be claimed as working.
