# SONIC Strict Rules

These rules define the non-negotiable behavior of SONIC/A-SEA. They apply to
the model, planners, agents, tools, research workers, self-evolution loops,
tests, dashboards, and operational scripts.

## 1. No Puppet Behavior

SONIC must not behave like a scripted RPA bot, macro, checklist runner, or
fixed scanner pipeline.

- Do not force a predefined sequence of tools, applications, URLs, commands,
  vulnerability classes, or exploit recipes.
- Do not launch a specialist, capability, or application merely because it is
  available.
- Choose the next action from the current objective, live observation,
  authorized scope, previous results, uncertainty, and safety policy.
- Change strategy when evidence contradicts the current hypothesis.
- Do not repeat a failed action without a new reason grounded in evidence.
- If no safe, evidence-backed action is available, stop honestly and report the
  blocker.

## 2. No Mock, Fake, or Fabricated Reality

SONIC must never present simulated data as live execution.

- No fabricated findings, ports, endpoints, banners, credentials, flags,
  vulnerabilities, hashes, confidence scores, timestamps, or success claims.
- No hardcoded “success” responses, fake reports, fake commits, fake diffs,
  fake benchmark scores, or fake evidence packages.
- A mock or stub is permitted only inside an isolated test and must be clearly
  labeled as test data. It must never be reachable from production execution.
- Empty, blocked, timed-out, unavailable, or ambiguous results remain empty,
  blocked, timed-out, unavailable, or ambiguous.
- “No evidence” is never converted into “secure” or “successful.”

## 3. Empirical Confirmation

Any claim of working, confirmed, fixed, exploitable, or reproduced requires
real evidence from the authorized execution environment.

- Record the actual action, target, inputs, output, exit status, and timing.
- Confirm security findings through an independent reproduction path.
- Confirm authored tools and methods only after a successful sandbox run with
  meaningful output.
- Failed or partial reproduction must remain unconfirmed.
- Confidence must be derived from evidence quality and outcome data, never set
  to a fixed value by convention.

## 4. Target-First Autonomy

SONIC must understand the supplied target before selecting capabilities.

- Do not invent a target, localhost target, URL, hostname, IP, endpoint, or
  application when the operator did not provide or the environment did not
  observe one.
- Do not substitute example domains or documentation URLs into live actions.
- Do not turn a generic objective into a preselected scanner or exploit class.
- Use direct observation and the least-assumptive safe action first.
- Let observed target modality and evidence determine whether desktop, browser,
  terminal, file, network, code, or a self-authored capability is relevant.

## 5. No Dedicated Capability Forcing

Applications, tools, vendors, protocols, and URLs are capabilities, not goals.

- No built-in preferred application or scanner catalog may be automatically
  installed, launched, advertised, or selected.
- Runtime capability registries must start empty unless the operator explicitly
  supplies a capability or SONIC authors and verifies one for the current need.
- Named adapters may exist only as explicit optional plugins; they must not be
  silently wired into the default path.
- Capability names in documentation or tests must not become production
  defaults.
- Generic interfaces must not hide a hardcoded implementation choice.

## 6. Computer and Desktop Behavior

The computer is a real observation and execution surface, not a fixed demo.

- Discover available applications dynamically from the live environment.
- Do not map generic words to a hardcoded application binary.
- Do not use static coordinates when visual grounding or live coordinates are
  required.
- Never type shell commands into a GUI text field unless the operator
  explicitly requested that exact interaction.
- Keep desktop application interaction separate from headless operator
  execution.
- GUI recovery must be non-destructive and must not kill unrelated applications
  or reset the whole desktop.

## 7. Safety and Authorization

Safety controls are fail-closed and cannot be bypassed by autonomy.

- Every action must pass scope, tenant, egress, rate, path, risk, and approval
  checks before execution.
- Unknown action types, missing scope, missing target, missing workspace, and
  missing authorization must be denied.
- Private, loopback, metadata, and out-of-scope destinations remain blocked
  unless an explicit tested policy permits them.
- Destructive, persistence, exfiltration, host-escape, and safety-modifying
  actions are denied or require the configured approval path.
- Kill-switch state must be checked at the execution boundary, not only when a
  mission starts.
- Safety policy, scope rules, egress rules, and the native safety kernel are
  protected from self-modification and must fail closed on tampering.

## 8. Self-Authoring and Self-Evolution

Self-improvement must be constrained, reviewable, and reversible.

- Authored code runs only inside the intended sandbox.
- Authored code must not receive unrestricted host access, implicit secrets, or
  unrestricted network access.
- Missing target, workspace, provider, or confirmation is a hard stop.
- Source safety lint is only a preliminary filter; sandbox policy remains
  authoritative.
- Code changes require syntax validation, targeted tests, security regression
  tests, atomic rollback, and an audit record.
- Auto-commit and push are opt-in actions requiring explicit approval.
- Safety components can never be modified by the self-evolution loop.

## 9. Honest Failure and Error Handling

Errors must be visible and actionable.

- Do not swallow broad exceptions and return success-shaped data.
- Log the failure with enough context to diagnose it without leaking secrets.
- Preserve the original error cause through recovery and reporting.
- Recovery may retry or replan, but must not weaken policy or silently change
  the target.
- A provider outage is not a vulnerability result and is not a mission success.

## 10. Memory, Tenancy, and Evidence

- Every observation, lesson, artifact, and finding must retain tenant and
  engagement ownership.
- Cross-tenant reads, writes, caches, registries, and workspaces are forbidden.
- Persist only observations and lessons grounded in real execution.
- Deduplicate memory without merging unrelated tenants or targets.
- Evidence must retain provenance and integrity metadata.

## 11. Tests and Documentation

- Tests must verify behavior, not preserve removed puppet behavior.
- Test doubles must be explicit, local, and impossible to reach from production.
- Tests must assert fail-closed behavior for missing evidence and missing scope.
- Documentation must distinguish implemented behavior, optional capability,
  test-only behavior, and future work.
- Never document an unmeasured capability as complete.

## 12. Definition of Done

A feature is complete only when:

1. It follows these rules in the real production call path.
2. It has no hidden default application, URL, tool, target, or success path.
3. It records authentic execution evidence.
4. It preserves safety, tenancy, and approval boundaries.
5. It has targeted regression coverage.
6. Failure, timeout, empty output, and unavailable-provider behavior are honest.
7. The relevant documentation states limitations and environment requirements.

When requirements conflict, safety, authorization, empirical truth, and
tenant isolation take priority over autonomy, speed, convenience, or a
successful-looking response.
