"""
Tests for SONIC's autonomous self-coder (real code paths, no mocked gates).

Covers:
1. PatchSynthesizer accepts a real unified diff and reports its change size.
2. PatchSynthesizer rejects a comment-only (no-op) response — the exact bug the
   old `_synthesize_patch_fallback` stub shipped.
3. PatchSynthesizer rejects a diff that does not apply cleanly.
4. PatchSynthesizer refuses a diff that touches a protected safety component.
5. The verification gate fails closed on a 0-collected test run.
6. The dedup hyphen bug is fixed (engine does not re-create the self-update goal).
7. SelfCoder drives a goal from a real signal to a locally committed change
   through an LLM-provided patch, and honestly reports no-change when the LLM
   declines.
"""

from __future__ import annotations

import asyncio
import subprocess
from pathlib import Path

import pytest

from sonic.evolution.codebase_evolver import CodebaseEvolver
from sonic.evolution.patch_synthesizer import PatchSynthesizer, _extract_diff
from sonic.evolution.self_coder import SelfCoder


class _StubLLM:
    """Minimal router stub returning a scripted response (never a real network)."""

    def __init__(self, contents: list[str]):
        self._contents = list(contents)
        self.calls = 0

    async def complete(self, request, task_type=None, provider_name=None):  # noqa: ANN001
        self.calls += 1
        idx = min(self.calls - 1, len(self._contents) - 1)
        content = self._contents[idx]

        class _Resp:
            def __init__(self, c: str):
                self.content = c
                self.model = "stub"
                self.provider = "stub"

        return _Resp(content)


def _git(cwd: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=str(cwd), check=True, capture_output=True, text=True)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    (root / "sonic-core" / "sonic" / "agents").mkdir(parents=True)
    (root / "sonic-core" / "sonic" / "safety").mkdir(parents=True)
    (root / "sonic-core" / "tests").mkdir(parents=True)
    (root / "sonic-core" / "sonic" / "agents" / "widget.py").write_text(
        "def total(items):\n    return sum(items)\n", encoding="utf-8"
    )
    (root / "sonic-core" / "sonic" / "safety" / "kernel.py").write_text(
        "GUARD = True\n", encoding="utf-8"
    )
    _git(root, "init")
    _git(root, "config", "user.email", "t@t.t")
    _git(root, "config", "user.name", "t")
    _git(root, "add", "-A")
    _git(root, "commit", "-m", "init")
    return root


class _Goal:
    goal_id = "g1"
    title = "Fix total() to ignore None"
    description = "total() crashes on None entries"
    category = "bug_fix"


REAL_DIFF = (
    "diff --git a/sonic-core/sonic/agents/widget.py b/sonic-core/sonic/agents/widget.py\n"
    "--- a/sonic-core/sonic/agents/widget.py\n"
    "+++ b/sonic-core/sonic/agents/widget.py\n"
    "@@ -1,2 +1,2 @@\n"
    " def total(items):\n"
    "-    return sum(items)\n"
    "+    return sum(i for i in items if i is not None)\n"
)

NOOP_DIFF = (
    "diff --git a/sonic-core/sonic/agents/widget.py b/sonic-core/sonic/agents/widget.py\n"
    "--- a/sonic-core/sonic/agents/widget.py\n"
    "+++ b/sonic-core/sonic/agents/widget.py\n"
    "@@ -1,2 +1,3 @@\n"
    " def total(items):\n"
    "     return sum(items)\n"
    "+# [SONIC-EVOLUTION] Goal: Fix the parser (timestamp)\n"
)

BAD_DIFF = (
    "diff --git a/sonic-core/sonic/agents/widget.py b/sonic-core/sonic/agents/widget.py\n"
    "--- a/sonic-core/sonic/agents/widget.py\n"
    "+++ b/sonic-core/sonic/agents/widget.py\n"
    "@@ -1,2 +1,2 @@\n"
    " def total(items):\n"
    "-    return sum(items) + 999\n"
    "+    return 0\n"
)


def test_patch_synthesizer_accepts_real_diff(repo: Path):
    llm = _StubLLM([f"```diff\n{REAL_DIFF}```"])
    synth = PatchSynthesizer(llm=llm, repo_root=repo, is_protected=lambda _p: False)
    res = asyncio.run(synth.synthesize(_Goal(), ["sonic-core/sonic/agents/widget.py"]))
    assert res.proposal is not None
    assert res.proposal.change_lines >= 1
    assert res.proposal.files == ["sonic-core/sonic/agents/widget.py"]


def test_patch_synthesizer_rejects_comment_only_noop(repo: Path):
    llm = _StubLLM([f"```diff\n{NOOP_DIFF}```"])
    synth = PatchSynthesizer(llm=llm, repo_root=repo, is_protected=lambda _p: False)
    res = asyncio.run(synth.synthesize(_Goal(), ["sonic-core/sonic/agents/widget.py"]))
    assert res.proposal is None
    assert "comment" in res.reason.lower() or "no real code change" in res.reason.lower()


def test_patch_synthesizer_rejects_nonapplying_diff(repo: Path):
    llm = _StubLLM([f"```diff\n{BAD_DIFF}```"])
    synth = PatchSynthesizer(llm=llm, repo_root=repo, is_protected=lambda _p: False)
    res = asyncio.run(synth.synthesize(_Goal(), ["sonic-core/sonic/agents/widget.py"]))
    assert res.proposal is None
    assert "apply" in res.reason.lower()


def test_patch_synthesizer_rejects_protected_target(repo: Path):
    protected_diff = REAL_DIFF.replace("agents/widget.py", "safety/kernel.py")
    llm = _StubLLM([f"```diff\n{protected_diff}```"])
    synth = PatchSynthesizer(
        llm=llm,
        repo_root=repo,
        is_protected=lambda p: "safety" in p,
    )
    res = asyncio.run(synth.synthesize(_Goal(), ["sonic-core/sonic/safety/kernel.py"]))
    assert res.proposal is None
    assert "protected" in res.reason.lower()


def test_extract_diff_handles_bare_and_fenced():
    assert _extract_diff(f"```diff\n{REAL_DIFF}\n```").startswith("diff --git")
    assert _extract_diff(REAL_DIFF).startswith("diff --git")


def test_verification_gate_fails_on_zero_tests(repo: Path):
    evolver = CodebaseEvolver(repo_root=repo)
    # A path with no tests collected must NOT pass the gate.
    result = evolver.run_tests(["sonic-core/tests/does_not_exist.py"])
    assert result.passed is False


def test_security_regression_fails_closed_when_missing(repo: Path):
    evolver = CodebaseEvolver(repo_root=repo)
    # No test_p0_security_hardening.py exists in this temp repo -> must fail closed.
    assert evolver.run_security_regression() is False


def test_self_update_goal_dedup_is_idempotent(repo: Path, tmp_path: Path):
    from sonic.evolution.evolution_journal import EvolutionJournal
    from sonic.evolution.goal_director import EvolutionGoalDirector
    from sonic.evolution.version_tracker import VersionTracker

    db = str(tmp_path / "evo.db")
    vtracker = VersionTracker(db_path=db, repo_root=repo)
    journal = EvolutionJournal(db_path=db, repo_root=repo)
    evolver = CodebaseEvolver(repo_root=repo)
    director = EvolutionGoalDirector(
        repo_root=repo, db_path=db, evolver=evolver,
        version_tracker=vtracker, journal=journal,
    )
    # Build the engine twice; the perpetual self-update goal must appear once.
    from sonic.evolution.engine import EvolutionEngine

    EvolutionEngine(codebase_evolver=evolver, version_tracker=vtracker, journal=journal, goal_director=director)
    EvolutionEngine(codebase_evolver=evolver, version_tracker=vtracker, journal=journal, goal_director=director)
    self_update_goals = [g for g in director.list_goals() if g.title.replace("\u2011", "-").startswith("Self-Update")]
    assert len(self_update_goals) == 1


def test_self_coder_end_to_end_commits_a_real_change(repo: Path, tmp_path: Path):
    """The self-coder turns a real signal into an LLM patch -> verified -> commit."""
    db = str(tmp_path / "evo.db")

    # Real unit test that will pass after the patch and fail before it.
    (repo / "sonic-core" / "tests" / "test_widget.py").write_text(
        "from sonic.agents.widget import total\n\n"
        "def test_total_ignores_none():\n"
        "    assert total([1, None, 2]) == 3\n",
        encoding="utf-8",
    )
    # Make the package importable for pytest.
    (repo / "sonic-core" / "sonic" / "__init__.py").write_text("", encoding="utf-8")
    (repo / "sonic-core" / "sonic" / "agents" / "__init__.py").write_text("", encoding="utf-8")
    # The security regression gate is fail-closed; provide a real (trivial) suite.
    (repo / "sonic-core" / "tests" / "test_p0_security_hardening.py").write_text(
        "def test_safety_present():\n    assert True\n",
        encoding="utf-8",
    )
    (repo / "conftest.py").write_text(
        "import sys\nfrom pathlib import Path\n"
        "sys.path.insert(0, str(Path(__file__).parent / 'sonic-core'))\n",
        encoding="utf-8",
    )
    _git(repo, "add", "-A")
    _git(repo, "commit", "-m", "add test")

    # A genuine TODO marker is the discovery signal.
    (repo / "sonic-core" / "sonic" / "agents" / "widget.py").write_text(
        "def total(items):\n    # TODO: ignore None entries\n    return sum(items)\n",
        encoding="utf-8",
    )
    _git(repo, "add", "-A")
    _git(repo, "commit", "-m", "add todo")

    from sonic.evolution.evolution_journal import EvolutionJournal
    from sonic.evolution.goal_director import EvolutionGoalDirector
    from sonic.evolution.version_tracker import VersionTracker

    vtracker = VersionTracker(db_path=db, repo_root=repo)
    journal = EvolutionJournal(db_path=db, repo_root=repo)
    evolver = CodebaseEvolver(repo_root=repo)
    director = EvolutionGoalDirector(
        repo_root=repo, db_path=db, evolver=evolver,
        version_tracker=vtracker, journal=journal,
        test_paths=["sonic-core/tests/test_widget.py"],
    )
    llm = _StubLLM([
        "```diff\ndiff --git a/sonic-core/sonic/agents/widget.py b/sonic-core/sonic/agents/widget.py\n"
        "--- a/sonic-core/sonic/agents/widget.py\n"
        "+++ b/sonic-core/sonic/agents/widget.py\n"
        "@@ -1,3 +1,2 @@\n def total(items):\n-    # TODO: ignore None entries\n-    return sum(items)\n"
        "+    return sum(i for i in items if i is not None)\n```"
    ])
    coder = SelfCoder(repo_root=repo, llm=llm, evolver=evolver, director=director)
    report = asyncio.run(coder.run(max_cycles=1))

    # report-only by default: the change is verified (awaiting_approval) but the tree is restored.
    assert report.changed is False
    assert report.cycles and report.cycles[0].stage in ("promoted", "awaiting_approval")
    assert report.cycles[0].stage != "rolled_back"
    assert report.cycles[0].tree_action == "rolled_back"
    assert report.cycles[0].proposed_diff
    # The source file must be exactly as it was before the run.
    assert (repo / "sonic-core" / "sonic" / "agents" / "widget.py").read_text() == (
        "def total(items):\n    # TODO: ignore None entries\n    return sum(items)\n"
    )


def test_self_coder_allow_writes_commits(repo: Path, tmp_path: Path):
    """With allow_writes=True the verified change is committed locally."""
    db = str(tmp_path / "evo.db")
    (repo / "sonic-core" / "tests" / "test_widget.py").write_text(
        "from sonic.agents.widget import total\n\n"
        "def test_total_ignores_none():\n"
        "    assert total([1, None, 2]) == 3\n",
        encoding="utf-8",
    )
    (repo / "sonic-core" / "sonic" / "__init__.py").write_text("", encoding="utf-8")
    (repo / "sonic-core" / "sonic" / "agents" / "__init__.py").write_text("", encoding="utf-8")
    (repo / "sonic-core" / "tests" / "test_p0_security_hardening.py").write_text(
        "def test_safety_present():\n    assert True\n", encoding="utf-8"
    )
    (repo / "conftest.py").write_text(
        "import sys\nfrom pathlib import Path\n"
        "sys.path.insert(0, str(Path(__file__).parent / 'sonic-core'))\n",
        encoding="utf-8",
    )
    _git(repo, "add", "-A")
    _git(repo, "commit", "-m", "add tests")
    (repo / "sonic-core" / "sonic" / "agents" / "widget.py").write_text(
        "def total(items):\n    # TODO: ignore None entries\n    return sum(items)\n", encoding="utf-8"
    )
    _git(repo, "add", "-A")
    _git(repo, "commit", "-m", "todo")

    from sonic.evolution.evolution_journal import EvolutionJournal
    from sonic.evolution.goal_director import EvolutionGoalDirector
    from sonic.evolution.version_tracker import VersionTracker

    vtracker = VersionTracker(db_path=db, repo_root=repo)
    journal = EvolutionJournal(db_path=db, repo_root=repo)
    evolver = CodebaseEvolver(repo_root=repo)
    director = EvolutionGoalDirector(
        repo_root=repo, db_path=db, evolver=evolver,
        version_tracker=vtracker, journal=journal,
        test_paths=["sonic-core/tests/test_widget.py"],
    )
    llm = _StubLLM([
        "```diff\ndiff --git a/sonic-core/sonic/agents/widget.py b/sonic-core/sonic/agents/widget.py\n"
        "--- a/sonic-core/sonic/agents/widget.py\n"
        "+++ b/sonic-core/sonic/agents/widget.py\n"
        "@@ -1,3 +1,2 @@\n def total(items):\n-    # TODO: ignore None entries\n-    return sum(items)\n"
        "+    return sum(i for i in items if i is not None)\n```"
    ])
    coder = SelfCoder(repo_root=repo, llm=llm, evolver=evolver, director=director, allow_writes=True)
    report = asyncio.run(coder.run(max_cycles=1))
    assert report.changed is True
    assert report.cycles[0].tree_action == "committed"
    assert "is not None" in (repo / "sonic-core" / "sonic" / "agents" / "widget.py").read_text()


def test_normalize_bare_hunks_fills_line_numbers(repo: Path):
    """Models often emit a bare '@@' header; we must derive the real line numbers."""
    from sonic.evolution.patch_synthesizer import _normalize_hunks

    bare = (
        "diff --git a/sonic-core/sonic/agents/widget.py b/sonic-core/sonic/agents/widget.py\n"
        "--- a/sonic-core/sonic/agents/widget.py\n"
        "+++ b/sonic-core/sonic/agents/widget.py\n"
        "@@\n"
        " def total(items):\n"
        "-    return sum(items)\n"
        "+    return sum(i for i in items if i is not None)\n"
    )

    def reader(rel: str) -> str:
        return (repo / rel).read_text(encoding="utf-8")

    norm, err = _normalize_hunks(bare, reader)
    assert err == ""
    assert "@@ -1,2 +1,2 @@" in norm


def test_synthesizer_accepts_bare_hunk_diff(repo: Path):
    """An otherwise-correct patch with bare '@@' must not be thrown away."""
    bare = (
        "```diff\n"
        "diff --git a/sonic-core/sonic/agents/widget.py b/sonic-core/sonic/agents/widget.py\n"
        "--- a/sonic-core/sonic/agents/widget.py\n"
        "+++ b/sonic-core/sonic/agents/widget.py\n"
        "@@\n"
        " def total(items):\n"
        "-    return sum(items)\n"
        "+    return sum(i for i in items if i is not None)\n"
        "```"
    )
    llm = _StubLLM([bare])
    synth = PatchSynthesizer(llm=llm, repo_root=repo, is_protected=lambda _p: False)
    res = asyncio.run(synth.synthesize(_Goal(), ["sonic-core/sonic/agents/widget.py"]))
    assert res.proposal is not None
    assert "@@ -1,2 +1,2 @@" in res.proposal.diff


def test_import_integrity_gate_blocks_broken_import(repo: Path):
    """A patch adding a top-level import of a missing module must be blocked,
    even when the test subset never imports the changed file."""
    evolver = CodebaseEvolver(repo_root=repo)
    target = repo / "sonic-core" / "sonic" / "agents" / "widget.py"
    target.write_text(
        "from sonic.agents.nonexistent_helper import guard\n\n"
        "def total(items):\n    return guard(items)\n",
        encoding="utf-8",
    )
    ok, msg = evolver.run_import_checks(["sonic-core/sonic/agents/widget.py"])
    assert ok is False
    assert "nonexistent_helper" in msg


def test_import_integrity_gate_passes_clean_file(repo: Path):
    evolver = CodebaseEvolver(repo_root=repo)
    ok, msg = evolver.run_import_checks(["sonic-core/sonic/agents/widget.py"])
    assert ok is True


def test_report_only_run_writes_no_side_effects(repo: Path, tmp_path: Path):
    """report_only must leave the tree (and evolution.md) untouched."""
    db = str(tmp_path / "evo.db")
    from sonic.evolution.evolution_journal import EvolutionJournal
    from sonic.evolution.goal_director import EvolutionGoalDirector
    from sonic.evolution.version_tracker import VersionTracker

    (repo / "sonic-core" / "sonic" / "agents" / "widget.py").write_text(
        "def total(items):\n    # TODO: handle None\n    return sum(items)\n", encoding="utf-8"
    )

    vtracker = VersionTracker(db_path=db, repo_root=repo)
    journal = EvolutionJournal(db_path=db, repo_root=repo)
    evolver = CodebaseEvolver(repo_root=repo)
    director = EvolutionGoalDirector(
        repo_root=repo, db_path=db, evolver=evolver,
        version_tracker=vtracker, journal=journal,
    )
    # Capture the baseline *after* constructing the journal (constructing it
    # writes the evolution.md header).
    evo_md = repo / "evolution.md"
    before = evo_md.read_text(encoding="utf-8") if evo_md.exists() else None
    target = "sonic-core/sonic/agents/widget.py"
    goal = director.submit_goal(title="t", description="d", target_files=[target])
    # a patch that fails verification, in report-only mode
    asyncio.run(director.execute_goal(
        goal.goal_id, custom_patch="def total(items):\n    return 'nonsense'\n",
        test_paths=["sonic-core/tests/test_does_not_exist.py"], report_only=True,
    ))
    src_now = (repo / target).read_text(encoding="utf-8")
    assert "return sum(items)" in src_now  # rolled back
    after = evo_md.read_text(encoding="utf-8") if evo_md.exists() else None
    assert after == before  # no journal side-effect


def test_coding_agent_uses_tools_before_answering(tmp_path: Path):
    """The agent must be able to explore the real repo with tools and report."""
    repo = tmp_path / "repo"
    (repo / "sonic-core" / "sonic" / "x").mkdir(parents=True)
    (repo / "sonic-core" / "sonic" / "x" / "mod.py").write_text("VALUE = 42\n", encoding="utf-8")

    from sonic.evolution.coding_agent import CodingAgent
    from sonic.llm.schemas import LLMResponse, ToolCall

    class ToolThenFinish:
        """First turn: read_file. Second turn: finish with what it saw."""
        def __init__(self):
            self.n = 0

        async def complete(self, request, task_type=None, provider_name=None):
            self.n += 1
            if self.n == 1:
                return LLMResponse(content="", tool_calls=[
                    ToolCall(id="c1", name="read_file", arguments={"path": "sonic-core/sonic/x/mod.py"})
                ])
            if self.n == 2:
                # Confirm the tool output actually reached the model's context.
                tool_msgs = [m.content for m in request.messages if m.role.value == "tool"]
                saw = any("VALUE = 42" in (c or "") for c in tool_msgs)
                return LLMResponse(content="", tool_calls=[
                    ToolCall(id="c2", name="finish", arguments={"diagnosis": f"saw_value={saw}"})
                ])
            return LLMResponse(content="done")

    agent = CodingAgent(llm=ToolThenFinish(), repo_root=repo, max_steps=5, allow_write=False)
    res = asyncio.run(agent.run("What is VALUE?"))
    assert res.finished is True
    assert "saw_value=True" in res.diagnosis
    assert any("read_file" in c for c in res.tool_calls)


def test_coding_agent_tools_are_real(tmp_path: Path):
    """list_dir/grep/read_file return real repo content; escapes are refused."""
    repo = tmp_path / "repo"
    (repo / "pkg").mkdir(parents=True)
    (repo / "pkg" / "a.py").write_text("def alpha():\n    return 1\n", encoding="utf-8")

    from sonic.evolution.coding_agent import CodingAgent

    agent = CodingAgent(llm=None, repo_root=repo, allow_write=False)
    d = agent._t_list_dir("pkg")
    assert d.ok and "a.py" in d.output
    r = agent._t_read_file("pkg/a.py")
    assert r.ok and "def alpha" in r.output
    g = agent._t_grep("alpha")
    assert g.ok and "pkg/a.py" in g.output
    # path escape refused
    esc = agent._t_read_file("../../etc/passwd")
    assert esc.ok is False


def test_coding_agent_refuses_protected_write(tmp_path: Path):
    repo = tmp_path / "repo"
    (repo / "sonic-core" / "sonic" / "safety").mkdir(parents=True)

    from sonic.evolution.coding_agent import CodingAgent

    agent = CodingAgent(llm=None, repo_root=repo, allow_write=True, is_protected=lambda p: "safety" in p)
    res = agent._t_write_file("sonic-core/sonic/safety/kernel.py", "x=1\n")
    assert res.ok is False
    assert "protected" in res.output.lower()


def test_coding_agent_write_stages_in_overlay(tmp_path: Path):
    repo = tmp_path / "repo"
    (repo / "pkg").mkdir(parents=True)
    (repo / "pkg" / "a.py").write_text("old\n", encoding="utf-8")

    from sonic.evolution.coding_agent import CodingAgent

    agent = CodingAgent(llm=None, repo_root=repo, allow_write=True)
    res = agent._t_write_file("pkg/a.py", "new\n")
    assert res.ok is True
    # read_file sees the staged content, but disk is unchanged (no commit).
    assert "new" in agent._t_read_file("pkg/a.py").output
    assert (repo / "pkg" / "a.py").read_text() == "old\n"


def test_report_only_rollback_restores_cross_file_patch(tmp_path: Path):
    """Regression: report-only must restore a file the patch touched even when it
    is not the signal's target. Root cause was clear_snapshots() running during
    the cycle, leaving nothing to roll back from."""
    repo = tmp_path / "repo"
    (repo / "sonic-core" / "sonic" / "agents").mkdir(parents=True)
    (repo / "sonic-core" / "tests").mkdir(parents=True)
    (repo / "sonic-core" / "sonic" / "__init__.py").write_text("", encoding="utf-8")
    (repo / "sonic-core" / "sonic" / "agents" / "__init__.py").write_text("", encoding="utf-8")
    orig = "def total(items):\n    # TODO: ignore None entries\n    return sum(items)\n"
    (repo / "sonic-core" / "sonic" / "agents" / "widget.py").write_text(orig, encoding="utf-8")
    (repo / "sonic-core" / "tests" / "test_widget.py").write_text(
        "from sonic.agents.widget import total\n\ndef test_total_ignores_none():\n    assert total([1,None,2])==3\n",
        encoding="utf-8",
    )
    (repo / "sonic-core" / "tests" / "test_p0_security_hardening.py").write_text(
        "def test_s(): assert True\n", encoding="utf-8")
    (repo / "conftest.py").write_text(
        "import sys\nfrom pathlib import Path\nsys.path.insert(0,str(Path(__file__).parent/'sonic-core'))\n",
        encoding="utf-8")
    _git(repo, "init")
    _git(repo, "config", "user.email", "t@t.t")
    _git(repo, "config", "user.name", "t")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-m", "init")

    from sonic.evolution.evolution_journal import EvolutionJournal
    from sonic.evolution.goal_director import EvolutionGoalDirector
    from sonic.evolution.version_tracker import VersionTracker

    db = str(tmp_path / "evo.db")
    vtracker = VersionTracker(db_path=db, repo_root=repo)
    journal = EvolutionJournal(db_path=db, repo_root=repo)
    evolver = CodebaseEvolver(repo_root=repo)
    director = EvolutionGoalDirector(
        repo_root=repo, db_path=db, evolver=evolver,
        version_tracker=vtracker, journal=journal,
        test_paths=["sonic-core/tests/test_widget.py"],
    )
    patch = (
        "diff --git a/sonic-core/sonic/agents/widget.py b/sonic-core/sonic/agents/widget.py\n"
        "--- a/sonic-core/sonic/agents/widget.py\n+++ b/sonic-core/sonic/agents/widget.py\n"
        "@@\n def total(items):\n-    # TODO: ignore None entries\n-    return sum(items)\n"
        "+    return sum(i for i in items if i is not None)\n"
    )

    class _L:
        async def complete(self, req, task_type=None, provider_name=None):
            return type("R", (), {"content": f"```diff\n{patch}```", "model": "s", "provider": "s"})()

    coder = SelfCoder(repo_root=repo, llm=_L(), evolver=evolver, director=director, agentic=False)
    # Signal targets the TEST file, but the patch edits widget.py.
    cycle = asyncio.run(coder.run_cycle(
        "Repair the failing test test_widget.py::test_total_ignores_none",
        "sonic-core/tests/test_widget.py", "bug_fix",
    ))
    assert cycle.tree_action == "rolled_back"
    assert (repo / "sonic-core" / "sonic" / "agents" / "widget.py").read_text() == orig
    # git must report a clean tree (no stray modification).
    status = subprocess.run(["git", "status", "--porcelain"], cwd=repo, capture_output=True, text=True).stdout
    assert "widget.py" not in status


def test_auditor_is_read_only_and_finds_defects(tmp_path: Path):
    repo = tmp_path / "repo"
    (repo / "sonic-core" / "sonic" / "x").mkdir(parents=True)
    (repo / "configs").mkdir(parents=True)
    (repo / "configs" / "safety_rules.yaml").write_text("kill_switch:\n  enabled: true\n", encoding="utf-8")
    (repo / "configs" / "models.yaml").write_text("default_provider: x\n", encoding="utf-8")
    bad = (
        "def f(x=[]):\n"
        "    try:\n"
        "        return x\n"
        "    except:\n"
        "        pass\n"
        "\ndef g(s):\n"
        "    return eval(s)  # FIXME: unsafe\n"
    )
    (repo / "sonic-core" / "sonic" / "x" / "evil.py").write_text(bad, encoding="utf-8")

    from sonic.evolution.codebase_auditor import CodebaseAuditor
    from sonic.evolution.codebase_report import narrate

    auditor = CodebaseAuditor(repo_root=repo, source_roots=("sonic-core/sonic",))
    report = auditor.audit(run_tests=False)
    kinds = {f.kind for f in report.findings}
    assert "bare_except" in kinds
    assert "dynamic_exec" in kinds
    assert "mutable_default" in kinds
    assert "todo_marker" in kinds
    # Read-only: the defective file is unchanged.
    assert (repo / "sonic-core" / "sonic" / "x" / "evil.py").read_text() == bad
    text = narrate(report)
    assert "Sir," in text
    assert "did not commit" in text.lower() or "nothing was committed" in text.lower()


def test_self_coder_reports_no_change_when_llm_declines(repo: Path, tmp_path: Path):
    db = str(tmp_path / "evo.db")
    from sonic.evolution.evolution_journal import EvolutionJournal
    from sonic.evolution.goal_director import EvolutionGoalDirector
    from sonic.evolution.version_tracker import VersionTracker

    (repo / "sonic-core" / "sonic" / "agents" / "widget.py").write_text(
        "def total(items):\n    # TODO: handle None\n    return sum(items)\n", encoding="utf-8"
    )
    vtracker = VersionTracker(db_path=db, repo_root=repo)
    journal = EvolutionJournal(db_path=db, repo_root=repo)
    evolver = CodebaseEvolver(repo_root=repo)
    director = EvolutionGoalDirector(
        repo_root=repo, db_path=db, evolver=evolver,
        version_tracker=vtracker, journal=journal,
    )
    llm = _StubLLM(["I cannot make this change safely."])
    coder = SelfCoder(repo_root=repo, llm=llm, evolver=evolver, director=director)
    report = asyncio.run(coder.run(max_cycles=1))
    assert report.changed is False
    assert report.cycles[0].stage != "promoted"
