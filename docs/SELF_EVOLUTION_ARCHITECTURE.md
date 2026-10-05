# SONIC Self-Evolution Architecture

This diagram describes the current code paths, not a guarantee that every
candidate will be fixed successfully. SONIC's Self-Evolution is a local
codebase workflow; it is separate from Boss's orchestration of ordinary
computer-use tasks.

## Entry points and system boundaries

```mermaid
flowchart LR
    OP["Operator"] --> CLI["SONIC CLI"]

    CLI -->|"`/auto [N]`"| AUTO["ContinuousAutoEvolutionRunner<br/>local checkout"]
    AUTO --> SELF["SelfCoder + candidate verification"]

    CLI -->|"`sonic evolution run`"| ECLI["Evolution CLI"]
    ECLI --> QUEUE["EvolutionGoalDirector<br/>queued goals"]

    CLI -->|"ordinary task prompt"| API["Backend API"]
    API --> BOSS["BossAgent"]
    BOSS --> WORKERS["ComputerUseAgent workers"]
    WORKERS --> DESKTOP["Visible desktop / task tools"]

    SELF -. "not routed through" .-> BOSS
    QUEUE -. "not routed through" .-> BOSS
```

- **`/auto`** starts a local continuous runner from the CLI's selected Git
  checkout. It does not send the work to the backend Boss.
- **`sonic evolution run`** drains goals already stored in the evolution goal
  queue. It is a distinct evolution entry point, not the `/auto` loop.
- **Ordinary prompts** go through the backend and Boss; they are not the
  Self-Evolution coder.

## `/auto` candidate lifecycle

```mermaid
flowchart TD
    START["Operator runs `/auto N`"] --> CLI["CLI resolves local Git repo root"]
    CLI --> RUNNER["ContinuousAutoEvolutionRunner"]

    RUNNER --> SCAN["Signal discovery on active checkout"]
    SCAN --> PYTEST["Run `sonic-core/tests/`<br/>attribute failures to editable source<br/>(timeout: 300 s)"]
    SCAN --> AUDIT["Static audit + TODO/FIXME/HACK<br/>and no-op marker scan"]
    PYTEST --> SIGNAL["Signals: test failures first,<br/>then audit and marker findings"]
    AUDIT --> SIGNAL

    SIGNAL --> FOUND{"Any signal?"}
    FOUND -->|No| IDLE["Wait / next bounded cycle"]
    IDLE --> SCAN
    FOUND -->|Yes| CANDIDATE["SelfCoderCandidateStore"]

    SIGNAL -->|"pass exact selected signal"| CANDIDATE
    CANDIDATE --> BASE["Record HEAD/base state"]
    BASE --> WT["Create detached Git worktree"]
    WT --> CODER["Create SelfCoder in worktree"]
    CODER --> PINNED["SelfCoder.run() uses the supplied signal<br/>without rediscovering"]
    PINNED --> CYCLE["SelfCoder.run_cycle(signal, target, category)"]

    CYCLE --> INVESTIGATE["CodingAgent investigates with read-only tools<br/>read / grep / inspect tests"]
    INVESTIGATE --> GOAL["EvolutionGoalDirector creates<br/>file-scoped goal"]
    GOAL --> PATCH["PatchSynthesizer asks LLM<br/>for a unified diff"]
    PATCH --> PATCHCHECK{"Diff applies, changes real code,<br/>and avoids protected paths?"}
    PATCHCHECK -->|No| FAIL["Reject / record no candidate"]
    PATCHCHECK -->|Yes| EVOLVER["CodebaseEvolver applies patch<br/>with snapshots"]
    EVOLVER --> GATES["AST + target tests + import checks<br/>+ security regression"]
    GATES --> GATEPASS{"All gates pass?"}
    GATEPASS -->|No| ROLLBACK["Rollback snapshots;<br/>candidate discarded"]
    GATEPASS -->|Yes| REPORT["Report-only result:<br/>capture exact diff, restore worktree"]

    REPORT --> RETEST["CandidateStore reapplies exact diff<br/>in detached worktree"]
    RETEST --> FULL["Import checks + full configured test dirs<br/>`sonic-core/tests` and `sonic-cli/tests`<br/>+ security regression"]
    FULL --> FULLPASS{"All verification receipts pass?"}
    FULLPASS -->|No| ROLLBACK
    FULLPASS -->|Yes| RECORD["Save candidate + integrity hash;<br/>remove detached worktree"]

    RECORD --> BASEMATCH{"Active HEAD still matches candidate base,<br/>and touched files have no conflict?"}
    BASEMATCH -->|No| DISCARD["Mark stale / conflict;<br/>do not apply"]
    BASEMATCH -->|Yes| APPLY["Apply exact verified diff<br/>to active checkout"]
    APPLY --> POST["Re-run import, test and security checks"]
    POST --> POSTPASS{"Post-apply checks pass?"}
    POSTPASS -->|No| ROLLBACKACTIVE["Restore active files"]
    POSTPASS -->|Yes| COMMIT{"`/auto` auto_commit enabled?"}
    COMMIT -->|Yes| LOCAL["Commit changed files locally;<br/>bump version + journal"]
    COMMIT -->|No| APPLIED["Leave verified patch applied;<br/>no commit"]
    LOCAL --> NEXT["Report result; next bounded cycle"]
    APPLIED --> NEXT
    FAIL --> NEXT
    ROLLBACK --> NEXT
    DISCARD --> NEXT
    ROLLBACKACTIVE --> NEXT
    NEXT --> SCAN
```

### What the coder actually does

1. **Finds a signal:** `SelfCoder.discover_signals()` runs pytest and looks for
   source-attributed failures, then static-audit findings, then TODO-like
   markers. A TODO is a work signal, not proof of a bug.
2. **Investigates:** `CodingAgent` asks the configured LLM to inspect the real
   repository using bounded tools. The investigation in this flow is
   read-only.
3. **Proposes:** `PatchSynthesizer` asks the LLM for a unified diff and checks
   that it applies cleanly, changes actual code, and does not touch protected
   files.
4. **Verifies and promotes:** `CodebaseEvolver` runs code and safety gates;
   `SelfCoderCandidateStore` repeats verification in a detached worktree before
   applying the exact tested patch to the active checkout. The `/auto` runner
   enables local commit by default; it does **not** push to a remote.

## Separate queued-goal evolution path

```mermaid
flowchart LR
    OP["Operator / automation"] --> CMD["`sonic evolution run`"]
    CMD --> DIRECTOR["EvolutionGoalDirector"]
    DIRECTOR --> QUEUE["Read queued goals<br/>in priority order"]
    QUEUE --> ANALYZE["Analyze goal + target files"]
    ANALYZE --> SYNTH["PatchSynthesizer"]
    SYNTH --> EVOLVE["CodebaseEvolver"]
    EVOLVE --> CHECKS["Protected paths + diff/AST<br/>tests + imports + security"]
    CHECKS --> RESULT{"Checks pass?"}
    RESULT -->|No| ROLLBACK["Rollback / retain failed goal"]
    RESULT -->|Yes| COMMIT["Commit locally + version/journal"]
    COMMIT --> PUSH{"Explicit `--push`?"}
    PUSH -->|Yes| REMOTE["Push"]
    PUSH -->|No| DONE["Finish locally"]
```

This lane uses the queued `EvolutionGoalDirector` and `CodebaseEvolver`
directly. It does not use the `/auto` runner's outer signal scan and candidate
worktree lifecycle.

## Current caveats from the implementation and recent test

- `/auto` now passes its selected signal (description, target file, and
  category) into the candidate worktree. The candidate uses that signal instead
  of rediscovering a different issue. This does not by itself make arbitrary
  external repositories or test suites discoverable: the outer signal scan
  still searches the configured SONIC source roots and runs the configured test
  scan.
- The signal scan runs the full `sonic-core/tests/` suite with a 300-second
  timeout; it is not yet an efficient arbitrary external-demo test selector.
- In the recent demo attempt, the candidate did not produce a patch: the first
  `/auto` run reached a SelfCoder cleanup error; after that lifecycle fix, a
  target-scoped attempt was rejected when the configured providers returned
  NVIDIA 403 and Groq 429. The demo's 23-contract baseline had 22 failures and
  1 pass, but no demo fix was verified.
- Passing unit tests or a model-generated diff alone does not establish
  successful evolution. A candidate is only promoted after the configured
  verification and active-checkout conflict checks.

## Source files

- [`sonic-cli/sonic_cli/repl.py`](../sonic-cli/sonic_cli/repl.py) — `/auto`
  entry point.
- [`continuous_runner.py`](../sonic-core/sonic/evolution/continuous_runner.py) —
  continuous `/auto` loop.
- [`self_coder.py`](../sonic-core/sonic/evolution/self_coder.py) — signals,
  report-only cycles, and target test discovery.
- [`coding_agent.py`](../sonic-core/sonic/evolution/coding_agent.py) — bounded
  tool-using investigation.
- [`patch_synthesizer.py`](../sonic-core/sonic/evolution/patch_synthesizer.py) —
  LLM unified-diff proposal and validation.
- [`self_coder_candidates.py`](../sonic-core/sonic/evolution/self_coder_candidates.py) —
  detached worktree verification and guarded application.
- [`codebase_evolver.py`](../sonic-core/sonic/evolution/codebase_evolver.py) —
  protected-path, AST, test, import, security, rollback, and promotion gates.
- [`goal_director.py`](../sonic-core/sonic/evolution/goal_director.py) and
  [`cli.py`](../sonic-core/sonic/evolution/cli.py) — queued-goal CLI path.
