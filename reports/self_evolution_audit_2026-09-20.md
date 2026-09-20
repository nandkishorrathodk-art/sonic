# Self-Coder / Self-Evolution Audit — is SONIC really upgrading its own code?

Question asked: "self-coder jo SONIC ko upgrade karta rehta hai — kya wo thinking
& reasoning karke sach me kaam kar raha hai, jaise ek software engineer?"

Short answer: **No.** The *scaffolding* (atomic snapshots, AST gate, test gate,
version bump, journal) is real and decent, but the part that actually **writes
code** is a stub, the LLM is never wired, the queue is never drained
autonomously, and the one human gate that exists is bypassed by default. As it
stands the "self-coder" can only append a timestamp comment to a file (proven
below) — it cannot fix or upgrade its own source.

All findings verified against `main` @ `5308ced`. Every claim marked 🔬 was
executed against the real code path.

---

## 1. The self-coder has no code-generation brain (CRITICAL — Rule 15)

`sonic/evolution/goal_director.py:568-575`:
```python
def _synthesize_patch_fallback(self, goal, target_file, context):
    """Generates a safe fallback code comment/annotation when no external LLM patch is supplied."""
    orig = context.get("content", "")
    timestamp = _now()
    comment = f"\n# [SONIC-EVOLUTION] Goal: {goal.title} ({timestamp})\n"
    if orig and orig not in comment:
        return orig + comment      # <-- the entire "fix" is a comment append
    return orig
```

`execute_goal` (`goal_director.py:475-478`) sets `code_diff = custom_patch`; when
none is supplied (there is no caller that supplies one) it falls back to the
stub above and hands it to the evolver as if it were a real patch.

🔬 **Proven end-to-end.** For a `BUG_FIX` goal on `sonic/evolution/__init__.py`:
```
synthesized 'patch' last line: '# [SONIC-EVOLUTION] Goal: Fix the parser bug (2026-09-20T10:02:52...'
apply_patch -> True | File content written and validated | +2/-0
original AST == new AST: True          # no semantic change at all
```
So the evolver reports “File content written and validated, +2 lines”, the
journal would log a successful upgrade, the version bumps — and **nothing was
fixed**. This is exactly the "success-by-decree" Rule 15 forbids, in the self-
evolution path.

## 2. The LLM is deliberately stubbed out of the self-coder (CRITICAL)

`EvolutionGoalDirector.__init__` accepts `llm_client` (`goal_director.py:120`)
and stores it (`:138`) — but **it is never referenced anywhere else in the file**
(grep: only the two lines). There is no prompt, no `ModelRouter`, no code
synthesis call. Every production construction passes no LLM:
- `engine.py:106` builds `EvolutionGoalDirector(...)` without `llm_client`
- `cli.py:118` builds it without `llm_client`

So the self-upgrade path is *structurally incapable* of reasoning about code.
The only genuine LLM code-generation in the repo is `CodeFixAgent`
(`sonic/agents/codefix.py:65-115`, uses `CODEFIX_SYSTEM` + `think(task_type="coding")`)
and `ToolsmithLoop._llm_propose_tool` (`being/toolsmith.py:119-155`) — neither is
wired to the codebase-evolution engine. (`CodeFixAgent` is imported only by
`sonic/swarm.py:20`, i.e. it writes fixes for *target* vulnerabilities, not for
SONIC itself.)

## 3. The queue is never drained autonomously (HIGH)

`EvolutionEngine.__init__` auto-queues a perpetual self-update goal
(`engine.py:98-104`), and `run_evolution_queue` (`engine.py:267-273`) exists —
but 🔬 grepping the whole tree, `run_continuous` / `run_evolution_queue` /
`execute_goal` are only ever called from `sonic/evolution/cli.py` (a human CLI).
There is no scheduler, background task, or life-loop hook. So nothing ever runs
the queue in the autonomous being: the "continuous self-evolution" is manual.

## 4. Self-update goal is re-created on every boot (HIGH — dedup bug)

`engine.py:101` checks `g.title.startswith("Self‑Update")` — that literal uses
**U+2011 non-breaking hyphen** (🔬 codepoint `0x2011`). The goal that is actually
created, `create_self_update_goal` (`goal_director.py:74`), uses a normal ASCII
hyphen `"Self-Update: maintain autonomous assessment reliability"`.
🔬 `"Self-Update: ...".startswith("Self\u2011Update")` → **False**.
So the dedup never fires and a fresh self-update goal is appended every time an
`EvolutionEngine` is constructed (backend boot, per BOSS, per agent). Meanwhile
`get_evolution_status` (`engine.py:279`) filters with the *ASCII* hyphen, so these
duplicate goals are also hidden from the status. Queue grows unbounded; a future
`run_continuous` would spin on stale no-op goals.

## 5. The one human-approval gate is bypassed by default (HIGH — Rule 11/19)

`EvolutionPipeline` correctly stages proposals and only promotes with explicit
operator approval in its own API (`pipeline.py:113-131`), and `CodebaseEvolver`
labels “awaiting approval” when `auto_promote=False` (`codebase_evolver.py:615-617`).
But:
- `EvolutionEngine.evolve_codebase` defaults `auto_promote=True` (`engine.py:209`)
- `EvolutionEngine.execute_evolution_goal` defaults `auto_promote=True` (`engine.py:254`)
- `run_continuous` hardcodes `auto_promote=True` (`goal_director.py:600`)
- the only real caller (`cli.py:188`) passes `auto_promote=True`

So the safe "CANARY → AWAITING_APPROVAL → operator" path in the docstring is
never the default path; the default is autonomous commit (+ optional push). This
is the same issue as audit H6, restated for the self-coder.

## 6. Verification gate can be trivially passed / skipped (HIGH — Rule 15)

`CodebaseEvolver.run_tests` (`codebase_evolver.py:334-346`): default test path is
a single file `sonic-core/tests/test_evolution_strategy_and_engine.py`; the pass
predicate is `returncode == 0 and failed == 0`.
- 🔬/code-read: a run that collects **zero** tests (or all-skipped) yields
  `passed_tests=0, failed_tests=0, total_tests=0` → `passed=True`. A patch whose
  tests all fail to collect still "passes" the gate.
- The regex only looks for `"N passed"` / `"N failed"`; a run with `0 passed`
  prints no `"N passed"`, so `passed=0, failed=0` and `is_success` is True.

`run_security_regression` (`codebase_evolver.py:365-370`): `if not existing:
return True` — if `test_p0_security_hardening.py` is absent, the security gate
**passes by absence**. And the gate outcome flips hard on an environment detail:
🔬 with an interpreter lacking `pytest` the whole cycle instantly ROLLS BACK
(`Verification tests failed: No module named pytest`) before any real check.

## 7. Protected-safety lists have silently drifted (MEDIUM)

Two independent copies:
- `pipeline.py:17-24` — 6 entries (no Rust kernel)
- `codebase_evolver.py:45-53` — 7 entries (adds `sonic-kernel-rs/src/safety`)

So *where* the protection is enforced depends on the code path
(`submit_proposal` vs `evolve_impl`/`apply_patch`/`analyze_goal`). Add the
substring-normalization bypass from the previous audit (H5:
`sonic//safety/...`, `sonic/./safety/...` both report `protected=False`), and the
safety-kernel immutability that the docs promise is unreliable.

## 8. The whole `sonic/meta/` self-development stack is dead (MEDIUM)

`meta/experiment.py`, `meta/evaluator.py`, `meta/canary.py`, `meta/benchmark.py`
contain a genuinely better design (baseline-vs-candidate benchmark, zero-
regression evaluator with FP/safety/recall gates, canary + auto-rollback). But
🔬 the only importers are `api/routes/experiments.py`, `api/routes/live.py`,
`api/routes/security.py` — and `experiments.py` calls `CanaryPipeline` with
*caller-supplied* eval callbacks; nothing connects it to `EvolutionEngine` or to
actual code patches. The evaluator's `BenchmarkResult` F1/recall/false-positive
metrics (`meta/benchmark.py`) are never fed by the self-coder. So the rigorous
evaluation machinery and the code-writing path are two disconnected universes.

## 9. What *is* genuinely working (be fair)

- `CodebaseEvolver` snapshots every file and can atomically roll back
  (`_record_snapshot` / `rollback_snapshots`) — real and correct.
- AST pre-flight before writing (`validate_ast`) — real.
- `apply_patch` correctly handles both unified diffs (`git apply`) and full-file
  replacement with diff stats — real.
- `VersionTracker` persists semver bumps to SQLite — real.
- `EvolutionJournal` persists an audit journal + appends to `evolution.md` — real
  (though `evolution.md` is currently just the empty header, i.e. no cycle has
  ever actually recorded a real upgrade).
- `Lesson` for AVOID on failure is recorded (`codebase_evolver.py:563-580`).
- `ToolsmithLoop` and `MethodLab` *do* drive an LLM and confirm by real sandbox
  execution — honest design.

## 10. Bottom line

| Capability an engineer has | Does the self-coder do it? |
|---|---|
| Understand the goal | ❌ keyword file-discovery only (`_discover_candidate_files`) |
| Read the code | ✅ reads file + symbols |
| Write a fix | ❌ **appends a comment** (`_synthesize_patch_fallback`) |
| Use an LLM to write code | ❌ `llm_client` is dead; no prompt |
| Run tests | ✅ but gate is bypassable (0-collected = pass) |
| Security regression | ⚠️ passes if the test file is missing |
| Commit / version | ✅ but `auto_promote=True` default, no approval |
| Autonomous loop | ❌ queue is never drained outside the CLI |
| Journal / audit | ✅ |

**Verdict:** SONIC has an honest *guard/ledger* around self-modification, and a
real LLM code-writer elsewhere (`CodeFixAgent`/Toolsmith) — but the *codebase
self-coder* itself cannot author code. It is currently a well-guarded no-op that
would, if ever run, commit timestamp comments and bump versions. That is the
opposite of "thinking and reasoning like a software engineer".

## Suggested priority to make it real

1. Wire a real LLM patch generator into `EvolutionGoalDirector` (reuse
   `CodeFixAgent`/`CODEFIX_SYSTEM` or a new `self_evolution_patch` prompt) and
   feed its diff as `code_diff`.
2. Make the test gate require `passed_tests > 0` and treat
   missing/0-collected/unparseable output as failure; make the security gate
   fail-closed when its suite is absent.
3. Change `auto_promote` default to `False`; keep the `AWAITING_APPROVAL` stage
   as the default exit.
4. Fix the U+2011 hyphen dedup bug (or use a stable `category`/idempotency key).
5. Unify `_PROTECTED_COMPONENTS` into one module and normalize paths
   (`Path.resolve()` + `relative_to` instead of substring matching).
6. Connect `sonic/meta/` (benchmark + evaluator) to the code path, or delete it.
7. Add a scheduler (life-loop hook / cron) only *after* 1-5 are real.