# SONIC Agent Operating Contract

This file is the short operational entrypoint for agents working on SONIC.
The complete non-negotiable policy is in
[strict_rule.md](./strict_rule.md). Read it before modifying code, prompts,
tests, infrastructure, or documentation.

## Agent identity

SONIC is an autonomous, target-first security research system operating inside
an authorized and isolated environment. It is not a macro, RPA script, fixed
scanner chain, browser-only bot, mock demonstrator, or success-reporting
facade.

## Required operating loop

Use this loop for every objective:

```text
Understand objective and authorization
  -> observe live target/environment
  -> identify knowns, unknowns, and constraints
  -> choose the least-assumptive safe action
  -> execute inside the authorized substrate
  -> inspect authentic output
  -> verify independently
  -> learn and replan
```

The loop may stop when the objective is verified, scope is exhausted, the
operator must approve the next action, or no safe evidence-backed progress is
possible.

## Hard prohibitions

Never add or restore:

- hardcoded target URLs, example targets, localhost fallbacks, or invented
  endpoints;
- forced application launchers, static application maps, fixed GUI
  coordinates, or browser-only workflows;
- automatic scanner/tool chains, named capability defaults, or preferred
  vendor integrations;
- fake findings, fake evidence, fake confidence, fake hashes, fake commits,
  fake benchmark scores, or fake success fallbacks;
- silent host execution, scope bypasses, egress bypasses, tenant mixing, or
  safety-policy mutation;
- automatic code commit/push without explicit approval.

## Implementation requirements

- Prefer generic interfaces and live capability discovery.
- Keep the default capability registry empty.
- Require explicit target and workspace values for active work.
- Route all execution through the safety and authorization boundary.
- Preserve exact exit codes, outputs, provenance, and failure causes.
- Make recovery non-destructive; replan instead of repeating a failed script.
- Make authored tools and methods prove themselves in the sandbox before
  registration or reporting.
- Update tests when behavior intentionally changes; remove tests that only
  enforce deleted puppet behavior.

## Review checklist

Before declaring work complete, search the changed call path for:

1. hardcoded URLs, targets, application names, scanner names, and wordlists;
2. default objects or fallback values that create fictional work;
3. broad exception handlers that hide execution failure;
4. unguarded subprocess, browser, GUI, network, or file operations;
5. missing tenant, scope, egress, approval, or kill-switch checks;
6. claims that are not backed by an actual output and independent verification.

A feature that looks autonomous but uses a fixed recipe is a regression. A
feature that reports success without proof is a regression. When uncertain,
fail closed and report what is missing.
