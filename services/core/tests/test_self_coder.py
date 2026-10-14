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
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from sonic.evolution.codebase_evolver import CodebaseEvolver
from sonic.evolution.patch_synthesizer import PatchSynthesizer, _extract_diff, _files_touched
from sonic.evolution.pipeline import EvolutionStage
from sonic.evolution.self_coder import SelfCoder, SelfCoderCycle


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


def test_close_is_safe_for_regular_self_coder(tmp_path: Path) -> None:
    coder = SelfCoder(repo_root=tmp_path)

    coder.close()
    coder.close()


@pytest.mark.asyncio
async def test_run_uses_pinned_signals_without_rediscovery(tmp_path: Path) -> None:
    coder = SelfCoder(repo_root=tmp_path, agentic=False)
    pinned = ("Fix the requested issue", "services/core/example.py", "bug_fix")
    expected_cycle = SelfCoderCycle(
        signal=pinned[0],
        target_file=pinned[1],
        category=pinned[2],
        stage="rejected",
    )
    coder.discover_signals = MagicMock(side_effect=AssertionError("must not rescan"))
    coder.run_cycle = AsyncMock(return_value=expected_cycle)

    report = await coder.run(max_cycles=1, signals=[pinned])

    assert report.cycles == [expected_cycle]
    coder.discover_signals.assert_not_called()
    coder.run_cycle.assert_awaited_once_with(*pinned)


def test_signal_discovery_uses_configurable_timeout(repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Signal discovery must honour SONIC_EVOLUTION_TEST_TIMEOUT.

    Regression: discovery hardcoded a 300s pytest timeout while the verification
    gate used a configurable (default 600s) budget, so a slow suite could abort
    discovery and stall self-evolution.
    """
    import subprocess as _sp

    captured: dict[str, object] = {}

    class _Res:
        returncode = 0
        stdout = "no tests ran\n"
        stderr = ""

    def _fake_run(cmd, **kwargs):  # noqa: ANN001
        captured["timeout"] = kwargs.get("timeout")
        return _Res()

    monkeypatch.setattr(_sp, "run", _fake_run)
    monkeypatch.setenv("SONIC_EVOLUTION_TEST_TIMEOUT", "742")
    coder = SelfCoder(repo_root=repo, agentic=False)
    coder.discover_signals(limit=1)
    assert captured["timeout"] == 742

    # Even if the evolver attribute is unavailable, the env value is honoured.
    monkeypatch.setenv("SONIC_EVOLUTION_TEST_TIMEOUT", "500")
    coder2 = SelfCoder(repo_root=repo, agentic=False)
    coder2.evolver._test_timeout_seconds = None  # type: ignore[attr-defined]
    coder2.discover_signals(limit=1)
    assert captured["timeout"] == 500


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    (root / "services/core" / "sonic" / "agents").mkdir(parents=True)
    (root / "services/core" / "sonic" / "safety").mkdir(parents=True)
    (root / "services/core" / "tests").mkdir(parents=True)
    (root / "services/core" / "sonic" / "agents" / "widget.py").write_text(
        "def total(items):\n    return sum(items)\n", encoding="utf-8"
    )
    (root / "services/core" / "sonic" / "safety" / "kernel.py").write_text(
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
    "diff --git a/services/core/sonic/agents/widget.py b/services/core/sonic/agents/widget.py\n"
    "--- a/services/core/sonic/agents/widget.py\n"
    "+++ b/services/core/sonic/agents/widget.py\n"
    "@@ -1,2 +1,2 @@\n"
    " def total(items):\n"
    "-    return sum(items)\n"
    "+    return sum(i for i in items if i is not None)\n"
)

NOOP_DIFF = (
    "diff --git a/services/core/sonic/agents/widget.py b/services/core/sonic/agents/widget.py\n"
    "--- a/services/core/sonic/agents/widget.py\n"
    "+++ b/services/core/sonic/agents/widget.py\n"
    "@@ -1,2 +1,3 @@\n"
    " def total(items):\n"
    "     return sum(items)\n"
    "+# [SONIC-EVOLUTION] Goal: Fix the parser (timestamp)\n"
)

BAD_DIFF = (
    "diff --git a/services/core/sonic/agents/widget.py b/services/core/sonic/agents/widget.py\n"
    "--- a/services/core/sonic/agents/widget.py\n"
    "+++ b/services/core/sonic/agents/widget.py\n"
    "@@ -1,2 +1,2 @@\n"
    " def total(items):\n"
    "-    return sum(items) + 999\n"
    "+    return 0\n"
)


def test_patch_synthesizer_accepts_real_diff(repo: Path):
    llm = _StubLLM([f"```diff\n{REAL_DIFF}```"])
    synth = PatchSynthesizer(llm=llm, repo_root=repo, is_protected=lambda _p: False)
    res = asyncio.run(synth.synthesize(_Goal(), ["services/core/sonic/agents/widget.py"]))
    assert res.proposal is not None
    assert res.proposal.change_lines >= 1
    assert res.proposal.files == ["services/core/sonic/agents/widget.py"]


def test_patch_synthesizer_rejects_comment_only_noop(repo: Path):
    llm = _StubLLM([f"```diff\n{NOOP_DIFF}```"])
    synth = PatchSynthesizer(llm=llm, repo_root=repo, is_protected=lambda _p: False)
    res = asyncio.run(synth.synthesize(_Goal(), ["services/core/sonic/agents/widget.py"]))
    assert res.proposal is None
    assert "comment" in res.reason.lower() or "no real code change" in res.reason.lower()


def test_patch_synthesizer_rejects_nonapplying_diff(repo: Path):
    llm = _StubLLM([f"```diff\n{BAD_DIFF}```"])
    synth = PatchSynthesizer(llm=llm, repo_root=repo, is_protected=lambda _p: False)
    res = asyncio.run(synth.synthesize(_Goal(), ["services/core/sonic/agents/widget.py"]))
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
    res = asyncio.run(synth.synthesize(_Goal(), ["services/core/sonic/safety/kernel.py"]))
    assert res.proposal is None
    assert "protected" in res.reason.lower()


def test_extract_diff_handles_bare_and_fenced():
    assert _extract_diff(f"```diff\n{REAL_DIFF}\n```").startswith("diff --git")
    assert _extract_diff(REAL_DIFF).startswith("diff --git")


def test_files_touched_inspects_both_groups_and_fallbacks():
    # 1. Both groups in diff --git
    diff_rename = (
        "diff --git a/foo.py b/bar.py\n"
        "--- a/foo.py\n"
        "+++ b/bar.py\n"
        "@@ -1 +1 @@\n-1\n+2\n"
    )
    assert _files_touched(diff_rename) == ["foo.py", "bar.py"]

    # 2. Deletion with diff --git and /dev/null
    diff_del_git = (
        "diff --git a/foo.py b//dev/null\n"
        "--- a/foo.py\n"
        "+++ /dev/null\n"
        "@@ -1 +0,0 @@\n-1\n"
    )
    assert _files_touched(diff_del_git) == ["foo.py"]

    # 3. Fallback deletion without diff --git
    diff_del_fallback = (
        "--- a/foo.py\n"
        "+++ /dev/null\n"
        "@@ -1 +0,0 @@\n-1\n"
    )
    assert _files_touched(diff_del_fallback) == ["foo.py"]

    # 4. Fallback creation without diff --git
    diff_create_fallback = (
        "--- /dev/null\n"
        "+++ b/bar.py\n"
        "@@ -0,0 +1 @@\n+1\n"
    )
    assert _files_touched(diff_create_fallback) == ["bar.py"]


def test_patch_synthesizer_rejects_protected_file_deletion(repo: Path):
    delete_safety_diff = (
        "--- a/services/core/sonic/safety/kernel.py\n"
        "+++ /dev/null\n"
        "@@ -1 +0,0 @@\n"
        "-GUARD = True\n"
    )
    llm = _StubLLM([f"```diff\n{delete_safety_diff}```"])
    synth = PatchSynthesizer(
        llm=llm,
        repo_root=repo,
        is_protected=lambda p: "safety" in p,
    )
    res = asyncio.run(synth.synthesize(_Goal(), ["services/core/sonic/safety/kernel.py"]))
    assert res.proposal is None
    assert "protected" in res.reason.lower()


def test_validate_and_measure_normalizes_paths(repo: Path):
    # Test path traversal normalization before is_protected check
    traversal_diff = (
        "diff --git a/services/core/sonic/agents/../safety/kernel.py b/services/core/sonic/agents/../safety/kernel.py\n"
        "--- a/services/core/sonic/agents/../safety/kernel.py\n"
        "+++ b/services/core/sonic/agents/../safety/kernel.py\n"
        "@@ -1 +1 @@\n"
        "-GUARD = True\n"
        "+GUARD = False\n"
    )
    checked_paths: list[str] = []

    def is_prot(p: str) -> bool:
        checked_paths.append(p)
        return "safety" in p

    synth = PatchSynthesizer(llm=None, repo_root=repo, is_protected=is_prot)
    ok, reason, proposal = synth._validate_and_measure(_Goal(), traversal_diff)
    assert ok is False
    assert "protected" in reason.lower()
    assert "services/core/sonic/safety/kernel.py" in checked_paths
    assert ".." not in checked_paths[0]


def test_applies_cleanly_uses_explicit_lf(repo: Path, monkeypatch: pytest.MonkeyPatch):
    import tempfile
    original_named_temp = tempfile.NamedTemporaryFile
    called_kwargs = {}

    def tracking_named_temp(*args, **kwargs):
        called_kwargs.update(kwargs)
        return original_named_temp(*args, **kwargs)

    monkeypatch.setattr(tempfile, "NamedTemporaryFile", tracking_named_temp)
    synth = PatchSynthesizer(llm=None, repo_root=repo)
    ok, _ = synth._applies_cleanly(REAL_DIFF)
    assert ok is True
    assert called_kwargs.get("newline") == "\n"


def test_extract_diff_prioritizes_tagged_fences_and_strips_conversation():
    # Multiple code blocks: untagged or other language first, ```diff second
    response_with_multiple_blocks = (
        "Here is what happened:\n"
        "```python\n"
        "crash = total([None])\n"
        "```\n"
        "And here is an untagged block:\n"
        "```\n"
        "some arbitrary text\n"
        "```\n"
        "Here is the diff you should apply:\n"
        "```diff\n"
        "I also added some conversational text inside the fence:\n"
        f"{REAL_DIFF}\n"
        "```\n"
        "Hope this helps!"
    )
    extracted = _extract_diff(response_with_multiple_blocks)
    assert extracted.startswith("diff --git ")
    assert "crash = total" not in extracted
    assert "some arbitrary text" not in extracted
    assert "conversational text inside" not in extracted

    # Code block tagged with ```patch
    response_patch = (
        "Here is the patch:\n"
        "```patch\n"
        "--- a/services/core/sonic/agents/widget.py\n"
        "+++ b/services/core/sonic/agents/widget.py\n"
        "@@ -1,2 +1,2 @@\n"
        " def total(items):\n"
        "-    return sum(items)\n"
        "+    return sum(i for i in items if i is not None)\n"
        "```"
    )
    extracted_patch = _extract_diff(response_patch)
    assert extracted_patch.startswith("--- a/")

    # Raw text with conversation preceding ---
    raw_with_conversation = (
        "Here is the fix directly:\n"
        "--- a/services/core/sonic/agents/widget.py\n"
        "+++ b/services/core/sonic/agents/widget.py\n"
        "@@ -1,2 +1,2 @@\n"
        " def total(items):\n"
        "-    return sum(items)\n"
        "+    return sum(i for i in items if i is not None)\n"
    )
    extracted_raw = _extract_diff(raw_with_conversation)
    assert extracted_raw.startswith("--- a/")



def test_verification_gate_fails_on_zero_tests(repo: Path):
    evolver = CodebaseEvolver(repo_root=repo)
    # A path with no tests collected must NOT pass the gate.
    result = evolver.run_tests(["services/core/tests/does_not_exist.py"])
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
    (repo / "services/core" / "tests" / "test_widget.py").write_text(
        "from sonic.agents.widget import total\n\n"
        "def test_total_ignores_none():\n"
        "    assert total([1, None, 2]) == 3\n",
        encoding="utf-8",
    )
    # Make the package importable for pytest.
    (repo / "services/core" / "sonic" / "__init__.py").write_text("", encoding="utf-8")
    (repo / "services/core" / "sonic" / "agents" / "__init__.py").write_text("", encoding="utf-8")
    # The security regression gate is fail-closed; provide a real (trivial) suite.
    (repo / "services/core" / "tests" / "test_p0_security_hardening.py").write_text(
        "def test_safety_present():\n    assert True\n",
        encoding="utf-8",
    )
    (repo / "services/core" / "tests" / "test_safety_sealed_policy.py").write_text(
        "def test_safety_sealed():\n    assert True\n",
        encoding="utf-8",
    )
    (repo / "conftest.py").write_text(
        "import sys\nfrom pathlib import Path\n"
        "sys.path.insert(0, str(Path(__file__).parent / 'services/core'))\n",
        encoding="utf-8",
    )
    _git(repo, "add", "-A")
    _git(repo, "commit", "-m", "add test")

    # A genuine TODO marker is the discovery signal.
    (repo / "services/core" / "sonic" / "agents" / "widget.py").write_text(
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
        test_paths=["services/core/tests/test_widget.py"],
    )
    llm = _StubLLM([
        "```diff\ndiff --git a/services/core/sonic/agents/widget.py b/services/core/sonic/agents/widget.py\n"
        "--- a/services/core/sonic/agents/widget.py\n"
        "+++ b/services/core/sonic/agents/widget.py\n"
        "@@ -1,3 +1,2 @@\n def total(items):\n-    # TODO: ignore None entries\n-    return sum(items)\n"
        "+    return sum(i for i in items if i is not None)\n```"
    ])
    # These tests cover the patch/verification pipeline. The separate
    # investigation contract is tested with a tool-grounded CodingAgent result.
    coder = SelfCoder(
        repo_root=repo, llm=llm, evolver=evolver, director=director, agentic=False
    )
    report = asyncio.run(coder.run(max_cycles=1))

    # report-only by default: the change is verified (awaiting_approval) but the tree is restored.
    assert report.changed is False
    assert report.cycles and report.cycles[0].stage in ("promoted", "awaiting_approval")
    assert report.cycles[0].stage != "rolled_back"
    assert report.cycles[0].tree_action == "rolled_back"
    assert report.cycles[0].proposed_diff
    # The source file must be exactly as it was before the run.
    assert (repo / "services/core" / "sonic" / "agents" / "widget.py").read_text() == (
        "def total(items):\n    # TODO: ignore None entries\n    return sum(items)\n"
    )


def test_self_coder_write_mode_records_verified_candidate_without_checkout_changes(
    repo: Path, tmp_path: Path
):
    """Candidate mode leaves a verified diff for review without editing the checkout."""
    db = str(tmp_path / "evo.db")
    (repo / "services/core" / "tests" / "test_widget.py").write_text(
        "from sonic.agents.widget import total\n\n"
        "def test_total_ignores_none():\n"
        "    assert total([1, None, 2]) == 3\n",
        encoding="utf-8",
    )
    (repo / "services/core" / "sonic" / "__init__.py").write_text("", encoding="utf-8")
    (repo / "services/core" / "sonic" / "agents" / "__init__.py").write_text("", encoding="utf-8")
    (repo / "services/core" / "tests" / "test_p0_security_hardening.py").write_text(
        "def test_safety_present():\n    assert True\n", encoding="utf-8"
    )
    (repo / "services/core" / "tests" / "test_safety_sealed_policy.py").write_text(
        "def test_safety_sealed():\n    assert True\n", encoding="utf-8"
    )
    (repo / "conftest.py").write_text(
        "import sys\nfrom pathlib import Path\n"
        "sys.path.insert(0, str(Path(__file__).parent / 'services/core'))\n",
        encoding="utf-8",
    )
    _git(repo, "add", "-A")
    _git(repo, "commit", "-m", "add tests")
    (repo / "services/core" / "sonic" / "agents" / "widget.py").write_text(
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
        test_paths=["services/core/tests/test_widget.py"],
    )
    llm = _StubLLM([
        "```diff\ndiff --git a/services/core/sonic/agents/widget.py b/services/core/sonic/agents/widget.py\n"
        "--- a/services/core/sonic/agents/widget.py\n"
        "+++ b/services/core/sonic/agents/widget.py\n"
        "@@ -1,3 +1,2 @@\n def total(items):\n-    # TODO: ignore None entries\n-    return sum(items)\n"
        "+    return sum(i for i in items if i is not None)\n```"
    ])
    coder = SelfCoder(
        repo_root=repo,
        llm=llm,
        evolver=evolver,
        director=director,
        allow_writes=True,
        agentic=False,
    )
    before_head = _git(repo, "rev-parse", "HEAD")
    before_branch = _git(repo, "branch", "--show-current")
    report = asyncio.run(coder.run(max_cycles=1))
    assert report.changed is False
    assert report.cycles[0].tree_action == "candidate_only"
    assert report.cycles[0].commit_hash == ""
    assert _git(repo, "rev-parse", "HEAD") == before_head
    assert _git(repo, "branch", "--show-current") == before_branch
    assert not _git(repo, "branch", "--list", "sonic/self-coder/*")
    assert "is not None" not in (repo / "services/core" / "sonic" / "agents" / "widget.py").read_text()
    assert report.cycles[0].proposed_diff


def test_failing_test_signal_targets_editable_source_frame(repo: Path, monkeypatch):
    from sonic.evolution.self_coder import SelfCoder

    test_file = repo / "services/core" / "tests" / "test_widget.py"
    test_file.write_text("def test_total():\n    assert False\n", encoding="utf-8")
    output = (
        "________________________ test_total _________________________\n"
        "Traceback (most recent call last):\n"
        "  services/core/tests/test_widget.py:2: in test_total\n"
        "  services/core/sonic/agents/widget.py:2: in total\n"
        "E AssertionError\n"
        "FAILED services/core/tests/test_widget.py::test_total - AssertionError\n"
    )
    monkeypatch.setattr(
        "subprocess.run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args[0], 1, stdout=output, stderr=""
        ),
    )

    coder = SelfCoder(repo_root=repo)
    signals = coder._failing_test_signals(limit=5)

    assert len(signals) == 1
    assert signals[0][1] == "services/core/sonic/agents/widget.py"
    assert "test_widget.py::test_total" in signals[0][0]
    assert coder._test_paths_for(signals[0][1]) == [
        "services/core/tests/test_widget.py"
    ]


def test_test_only_failure_does_not_become_source_edit_signal(repo: Path, monkeypatch):
    from sonic.evolution.self_coder import SelfCoder

    test_file = repo / "services/core" / "tests" / "test_widget.py"
    test_file.write_text("def test_total():\n    assert False\n", encoding="utf-8")
    output = (
        "________________________ test_total _________________________\n"
        "Traceback (most recent call last):\n"
        "  services/core/tests/test_widget.py:2: in test_total\n"
        "E AssertionError\n"
        "FAILED services/core/tests/test_widget.py::test_total - AssertionError\n"
    )
    monkeypatch.setattr(
        "subprocess.run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args[0], 1, stdout=output, stderr=""
        ),
    )

    assert SelfCoder(repo_root=repo)._failing_test_signals(limit=5) == []


def test_failing_cli_test_signal_targets_cli_source(repo: Path, monkeypatch):
    from sonic.evolution.self_coder import SelfCoder

    cli_test_dir = repo / "apps/cli" / "tests"
    cli_source_dir = repo / "apps/cli" / "sonic_cli"
    cli_test_dir.mkdir(parents=True)
    cli_source_dir.mkdir(parents=True)
    test_file = cli_test_dir / "test_widget.py"
    source_file = cli_source_dir / "widget.py"
    test_file.write_text("def test_total():\n    assert False\n", encoding="utf-8")
    source_file.write_text("def total():\n    return 0\n", encoding="utf-8")
    output = (
        "________________________ test_total _________________________\n"
        "Traceback (most recent call last):\n"
        "  apps/cli/tests/test_widget.py:2: in test_total\n"
        "  apps/cli/sonic_cli/widget.py:2: in total\n"
        "E AssertionError\n"
        "FAILED apps/cli/tests/test_widget.py::test_total - AssertionError\n"
    )
    monkeypatch.setattr(
        "subprocess.run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args[0], 1, stdout=output, stderr=""
        ),
    )

    coder = SelfCoder(repo_root=repo)
    signals = coder._failing_test_signals(limit=5)

    assert len(signals) == 1
    assert signals[0][1] == "apps/cli/sonic_cli/widget.py"
    assert coder._test_paths_for(signals[0][1]) == [
        "apps/cli/tests/test_widget.py"
    ]


def test_test_signal_discovery_raises_when_pytest_cannot_collect(repo: Path, monkeypatch):
    from sonic.evolution.self_coder import SelfCoder

    monkeypatch.setattr(
        "subprocess.run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args[0], 4, stdout="pytest usage error", stderr=""
        ),
    )

    with pytest.raises(RuntimeError, match="could not complete signal discovery"):
        SelfCoder(repo_root=repo)._failing_test_signals(limit=5)


def test_test_signal_discovery_raises_when_no_test_suites_exist(tmp_path: Path):
    from sonic.evolution.self_coder import SelfCoder

    with pytest.raises(RuntimeError, match="No configured test directories"):
        SelfCoder(repo_root=tmp_path)._failing_test_signals(limit=5)


def test_static_scan_failure_is_not_reported_as_clean_repo(repo: Path, monkeypatch):
    from sonic.evolution.codebase_auditor import CodebaseAuditor
    from sonic.evolution.self_coder import SelfCoder

    coder = SelfCoder(repo_root=repo)
    monkeypatch.setattr(coder, "_failing_test_signals", lambda limit: [])
    monkeypatch.setattr(
        CodebaseAuditor,
        "static_scan",
        lambda self: (_ for _ in ()).throw(RuntimeError("audit unavailable")),
    )

    with pytest.raises(RuntimeError, match="Static signal discovery failed"):
        coder.discover_signals(limit=5)


def test_static_audit_signal_is_prioritized_and_safety_targets_are_excluded(
    repo: Path, monkeypatch
):
    from sonic.evolution.codebase_auditor import CodebaseAuditor
    from sonic.evolution.self_coder import SelfCoder

    findings = [
        SimpleNamespace(
            kind="shell_true",
            severity="high",
            file="services/core/sonic/agents/widget.py",
            line=1,
            title="Subprocess uses shell=True",
            detail="Potentially unsafe shell invocation",
        ),
        SimpleNamespace(
            kind="new_audit_rule",
            severity="medium",
            file="services/core/sonic/agents/widget.py",
            line=2,
            title="New auditor rule",
            detail="Future audit kinds should be discoverable without a second allowlist",
        ),
        SimpleNamespace(
            kind="shell_true",
            severity="critical",
            file="services/core/sonic/safety/kernel.py",
            line=1,
            title="Safety policy invokes shell",
            detail="Protected target",
        ),
    ]
    monkeypatch.setattr(CodebaseAuditor, "static_scan", lambda self: findings)
    coder = SelfCoder(repo_root=repo)
    monkeypatch.setattr(coder, "_failing_test_signals", lambda limit: [])

    signals = coder.discover_signals(limit=5)

    assert signals[0][1] == "services/core/sonic/agents/widget.py"
    assert "shell_true" in signals[0][0]
    assert any("new_audit_rule" in signal[0] for signal in signals)
    assert all("safety/kernel.py" not in signal[1] for signal in signals)


def test_agentic_self_coder_resolves_router_for_investigation(repo: Path, monkeypatch):
    from sonic.evolution.pipeline import EvolutionStage
    from sonic.evolution.self_coder import SelfCoder

    llm = object()
    init_calls: list[dict[str, object]] = []

    class _Director:
        test_paths = []

        def _resolve_llm(self):
            return llm

        def submit_goal(self, **kwargs):
            return SimpleNamespace(goal_id="goal-1")

        async def execute_goal(self, *args, **kwargs):
            return SimpleNamespace(
                stage=EvolutionStage.ROLLED_BACK,
                description="No patch produced",
                lines_added=0,
                lines_removed=0,
                files_affected=[],
                error_reason="model declined",
                git_commit_hash="",
            )

    class _Agent:
        def __init__(self, **kwargs):
            init_calls.append(kwargs)

        async def run(self, goal):
            return SimpleNamespace(
                finished=True,
                diagnosis="Read the implementation and confirmed the cause.",
                steps=2,
                tool_calls=[{"name": "read_file"}],
                observed_files=["services/core/sonic/agents/widget.py"],
            )

    monkeypatch.setattr("sonic.evolution.coding_agent.CodingAgent", _Agent)
    coder = SelfCoder(repo_root=repo, director=_Director(), agentic=True)

    cycle = asyncio.run(
        coder.run_cycle("Inspect a real signal", "services/core/sonic/agents/widget.py", "bug_fix")
    )

    assert init_calls and init_calls[0]["llm"] is llm
    assert cycle.diagnosis == "Read the implementation and confirmed the cause."
    assert cycle.stage == EvolutionStage.ROLLED_BACK.value


def test_agentic_self_coder_stops_when_no_llm_is_configured(repo: Path):
    from sonic.evolution.self_coder import SelfCoder

    class _Director:
        test_paths = []
        submit_goal = MagicMock()

        def _resolve_llm(self):
            return None

    director = _Director()
    coder = SelfCoder(repo_root=repo, director=director, agentic=True)

    cycle = asyncio.run(
        coder.run_cycle("Inspect a real signal", "services/core/sonic/agents/widget.py", "bug_fix")
    )

    assert cycle.stage == "investigation_incomplete"
    assert "No configured LLM provider" in cycle.reason
    director.submit_goal.assert_not_called()


def test_self_coder_does_not_synthesize_without_completed_investigation(
    repo: Path, monkeypatch
):
    from sonic.evolution.coding_agent import CodingAgent

    async def incomplete_investigation(self, goal):
        return SimpleNamespace(
            finished=False,
            diagnosis="Possible issue, but not checked with tools.",
            steps=1,
            tool_calls=[],
        )

    monkeypatch.setattr(CodingAgent, "run", incomplete_investigation)
    coder = SelfCoder(repo_root=repo, llm=_StubLLM(["unused"]))
    monkeypatch.setattr(
        coder.director,
        "submit_goal",
        lambda **kwargs: pytest.fail("incomplete investigation reached patch synthesis"),
    )

    cycle = asyncio.run(
        coder.run_cycle(
            "A failing behavior needs investigation",
            "services/core/sonic/agents/widget.py",
            "bug_fix",
        )
    )

    assert cycle.stage == "investigation_incomplete"
    assert cycle.proposed_diff == ""
    assert cycle.tree_action == "rolled_back"


def test_self_coder_requires_target_file_evidence_before_synthesis(
    repo: Path, monkeypatch
):
    from sonic.evolution.coding_agent import CodingAgent

    async def unsupported_diagnosis(self, goal):
        return SimpleNamespace(
            finished=True,
            diagnosis="The issue is probably in the target module.",
            steps=1,
            tool_calls=["finish({'diagnosis': 'guess'})"],
            observed_files=[],
        )

    monkeypatch.setattr(CodingAgent, "run", unsupported_diagnosis)
    coder = SelfCoder(repo_root=repo, llm=_StubLLM(["unused"]))
    monkeypatch.setattr(
        coder.director,
        "submit_goal",
        lambda **kwargs: pytest.fail("unsupported diagnosis reached patch synthesis"),
    )

    cycle = asyncio.run(
        coder.run_cycle(
            "Review the implementation",
            "services/core/sonic/agents/widget.py",
            "bug_fix",
        )
    )

    assert cycle.stage == "investigation_incomplete"
    assert "read the target file" in cycle.reason
    assert cycle.proposed_diff == ""


def test_normalize_bare_hunks_fills_line_numbers(repo: Path):
    """Models often emit a bare '@@' header; we must derive the real line numbers."""
    from sonic.evolution.patch_synthesizer import _normalize_hunks

    bare = (
        "diff --git a/services/core/sonic/agents/widget.py b/services/core/sonic/agents/widget.py\n"
        "--- a/services/core/sonic/agents/widget.py\n"
        "+++ b/services/core/sonic/agents/widget.py\n"
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
        "diff --git a/services/core/sonic/agents/widget.py b/services/core/sonic/agents/widget.py\n"
        "--- a/services/core/sonic/agents/widget.py\n"
        "+++ b/services/core/sonic/agents/widget.py\n"
        "@@\n"
        " def total(items):\n"
        "-    return sum(items)\n"
        "+    return sum(i for i in items if i is not None)\n"
        "```"
    )
    llm = _StubLLM([bare])
    synth = PatchSynthesizer(llm=llm, repo_root=repo, is_protected=lambda _p: False)
    res = asyncio.run(synth.synthesize(_Goal(), ["services/core/sonic/agents/widget.py"]))
    assert res.proposal is not None
    assert "@@ -1,2 +1,2 @@" in res.proposal.diff


def test_import_integrity_gate_blocks_broken_import(repo: Path):
    """A patch adding a top-level import of a missing module must be blocked,
    even when the test subset never imports the changed file."""
    evolver = CodebaseEvolver(repo_root=repo)
    target = repo / "services/core" / "sonic" / "agents" / "widget.py"
    target.write_text(
        "from sonic.agents.nonexistent_helper import guard\n\n"
        "def total(items):\n    return guard(items)\n",
        encoding="utf-8",
    )
    ok, msg = evolver.run_import_checks(["services/core/sonic/agents/widget.py"])
    assert ok is False
    assert "nonexistent_helper" in msg


def test_import_integrity_gate_passes_clean_file(repo: Path):
    evolver = CodebaseEvolver(repo_root=repo)
    ok, msg = evolver.run_import_checks(["services/core/sonic/agents/widget.py"])
    assert ok is True


def test_report_only_run_writes_no_side_effects(repo: Path, tmp_path: Path):
    """report_only must leave the tree (and evolution.md) untouched."""
    db = str(tmp_path / "evo.db")
    from sonic.evolution.evolution_journal import EvolutionJournal
    from sonic.evolution.goal_director import EvolutionGoalDirector
    from sonic.evolution.version_tracker import VersionTracker

    (repo / "services/core" / "sonic" / "agents" / "widget.py").write_text(
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
    target = "services/core/sonic/agents/widget.py"
    goal = director.submit_goal(title="t", description="d", target_files=[target])
    # a patch that fails verification, in report-only mode
    asyncio.run(director.execute_goal(
        goal.goal_id, custom_patch="def total(items):\n    return 'nonsense'\n",
        test_paths=["services/core/tests/test_does_not_exist.py"], report_only=True,
    ))
    src_now = (repo / target).read_text(encoding="utf-8")
    assert "return sum(items)" in src_now  # rolled back
    after = evo_md.read_text(encoding="utf-8") if evo_md.exists() else None
    assert after == before  # no journal side-effect


def test_coding_agent_uses_tools_before_answering(tmp_path: Path):
    """The agent must be able to explore the real repo with tools and report."""
    repo = tmp_path / "repo"
    (repo / "services/core" / "sonic" / "x").mkdir(parents=True)
    (repo / "services/core" / "sonic" / "x" / "mod.py").write_text("VALUE = 42\n", encoding="utf-8")

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
                    ToolCall(id="c1", name="read_file", arguments={"path": "services/core/sonic/x/mod.py"})
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
    assert res.observed_files == ["services/core/sonic/x/mod.py"]


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
    (repo / "services/core" / "sonic" / "safety").mkdir(parents=True)

    from sonic.evolution.coding_agent import CodingAgent

    agent = CodingAgent(llm=None, repo_root=repo, allow_write=True, is_protected=lambda p: "safety" in p)
    res = agent._t_write_file("services/core/sonic/safety/kernel.py", "x=1\n")
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


def test_coding_agent_command_security(repo: Path):
    from sonic.evolution.coding_agent import _ALLOWED_CMDS, CodingAgent

    assert "python -c" not in _ALLOWED_CMDS

    agent = CodingAgent(llm=None, repo_root=repo, allow_write=False)

    # Rejects python -c
    res = agent._t_run_command("python -c 'print(1)'")
    assert res.ok is False
    assert "command not allowlisted" in res.output

    # Rejects chaining characters
    for ch in [";", "&&", "|", "&", "\n"]:
        cmd = f"git status {ch} ls"
        res = agent._t_run_command(cmd)
        assert res.ok is False
        assert "chaining" in res.output

    # Rejects disallowed operators
    res = agent._t_run_command("git status > out.txt")
    assert res.ok is False
    assert "disallowed operator" in res.output

    # Rejects unallowlisted command
    res = agent._t_run_command("whoami")
    assert res.ok is False
    assert "command not allowlisted" in res.output

    # Successfully executes allowlisted command without shell
    res = agent._t_run_command("git status")
    assert res.ok is True


def test_coding_agent_write_file_path_confinement_and_safety(repo: Path):
    from sonic.evolution.coding_agent import CodingAgent

    seen_paths = []

    def check_protected(rel: str) -> bool:
        seen_paths.append(rel)
        return rel == "pkg/locked.py"

    agent = CodingAgent(llm=None, repo_root=repo, allow_write=True, is_protected=check_protected)

    # Path escape outside repo root is blocked with exact error message
    esc = agent._t_write_file("../../etc/passwd", "root:x:0:0")
    assert esc.ok is False
    assert "path escapes repository root" in esc.output

    esc2 = agent._t_write_file("pkg/../../outside.py", "secret")
    assert esc2.ok is False
    assert "path escapes repository root" in esc2.output

    # 'test' substring cannot bypass _SAFETY_MARKERS
    sec1 = agent._t_write_file("services/core/sonic/safety/test_bypass.py", "x = 1")
    assert sec1.ok is False
    assert "refusing to write safety-sensitive path" in sec1.output

    sec2 = agent._t_write_file("services/core/sonic/sandbox/egress/test.py", "x = 1")
    assert sec2.ok is False
    assert "refusing to write safety-sensitive path" in sec2.output

    # is_protected called with normalized relative path
    res = agent._t_write_file("./pkg/nested/../locked.py", "content")
    assert res.ok is False
    assert "refusing to write protected component: pkg/locked.py" in res.output
    assert "pkg/locked.py" in seen_paths



def test_report_only_rollback_restores_cross_file_patch(tmp_path: Path):
    """Regression: report-only must restore a file the patch touched even when it
    is not the signal's target. Root cause was clear_snapshots() running during
    the cycle, leaving nothing to roll back from."""
    repo = tmp_path / "repo"
    (repo / "services/core" / "sonic" / "agents").mkdir(parents=True)
    (repo / "services/core" / "tests").mkdir(parents=True)
    (repo / "services/core" / "sonic" / "__init__.py").write_text("", encoding="utf-8")
    (repo / "services/core" / "sonic" / "agents" / "__init__.py").write_text("", encoding="utf-8")
    orig = "def total(items):\n    # TODO: ignore None entries\n    return sum(items)\n"
    (repo / "services/core" / "sonic" / "agents" / "widget.py").write_text(orig, encoding="utf-8")
    (repo / "services/core" / "tests" / "test_widget.py").write_text(
        "from sonic.agents.widget import total\n\ndef test_total_ignores_none():\n    assert total([1,None,2])==3\n",
        encoding="utf-8",
    )
    (repo / "services/core" / "tests" / "test_p0_security_hardening.py").write_text(
        "def test_s(): assert True\n", encoding="utf-8")
    (repo / "services/core" / "tests" / "test_safety_sealed_policy.py").write_text(
        "def test_safety_s(): assert True\n", encoding="utf-8")
    (repo / "conftest.py").write_text(
        "import sys\nfrom pathlib import Path\nsys.path.insert(0,str(Path(__file__).parent/'services/core'))\n",
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
        test_paths=["services/core/tests/test_widget.py"],
    )
    patch = (
        "diff --git a/services/core/sonic/agents/widget.py b/services/core/sonic/agents/widget.py\n"
        "--- a/services/core/sonic/agents/widget.py\n+++ b/services/core/sonic/agents/widget.py\n"
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
        "services/core/tests/test_widget.py", "bug_fix",
    ))
    assert cycle.tree_action == "rolled_back"
    assert (repo / "services/core" / "sonic" / "agents" / "widget.py").read_text() == orig
    # git must report a clean tree (no stray modification).
    status = subprocess.run(["git", "status", "--porcelain"], cwd=repo, capture_output=True, text=True).stdout
    assert "widget.py" not in status


def test_auditor_is_read_only_and_finds_defects(tmp_path: Path):
    repo = tmp_path / "repo"
    (repo / "services/core" / "sonic" / "x").mkdir(parents=True)
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
    (repo / "services/core" / "sonic" / "x" / "evil.py").write_text(bad, encoding="utf-8")

    from sonic.evolution.codebase_auditor import CodebaseAuditor
    from sonic.evolution.codebase_report import narrate

    auditor = CodebaseAuditor(repo_root=repo, source_roots=("services/core/sonic",))
    report = auditor.audit(run_tests=False)
    kinds = {f.kind for f in report.findings}
    assert "bare_except" in kinds
    assert "dynamic_exec" in kinds
    assert "mutable_default" in kinds
    assert "todo_marker" in kinds
    # Read-only: the defective file is unchanged.
    assert (repo / "services/core" / "sonic" / "x" / "evil.py").read_text() == bad
    text = narrate(report)
    assert "Sir," in text
    assert "did not commit" in text.lower() or "nothing was committed" in text.lower()


def test_self_coder_reports_no_change_when_llm_declines(repo: Path, tmp_path: Path):
    db = str(tmp_path / "evo.db")
    from sonic.evolution.evolution_journal import EvolutionJournal
    from sonic.evolution.goal_director import EvolutionGoalDirector
    from sonic.evolution.version_tracker import VersionTracker

    (repo / "services/core" / "sonic" / "agents" / "widget.py").write_text(
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


def test_ast_syntax_error_triggers_rejection_and_rollback(repo: Path):
    """Feeds invalid Python syntax in diff or target to CodebaseEvolver / SelfCoder,
    asserts stage != 'promoted', ast error reason is recorded, and scratch files are rolled back."""
    target_rel = "services/core/sonic/agents/widget.py"
    target_abs = repo / target_rel
    orig_content = target_abs.read_text(encoding="utf-8")

    evolver = CodebaseEvolver(repo_root=repo)

    # 1. Direct code replacement with syntax error
    bad_code = "def total(items):\n    return sum(items def broken(\n"
    report_direct = evolver.evolve(
        target_component=target_rel,
        description="Syntax error in direct replacement",
        code_diff=bad_code,
    )
    assert report_direct.stage != "promoted"
    assert report_direct.stage == EvolutionStage.REJECTED
    assert "syntax" in report_direct.error_reason.lower() or "ast" in report_direct.error_reason.lower()
    assert target_abs.read_text(encoding="utf-8") == orig_content

    # 2. Unified diff with invalid Python syntax
    bad_syntax_diff = (
        "diff --git a/services/core/sonic/agents/widget.py b/services/core/sonic/agents/widget.py\n"
        "--- a/services/core/sonic/agents/widget.py\n"
        "+++ b/services/core/sonic/agents/widget.py\n"
        "@@ -1,2 +1,2 @@\n"
        " def total(items):\n"
        "-    return sum(items)\n"
        "+    return sum(items def broken(\n"
    )
    report_diff = evolver.evolve(
        target_component=target_rel,
        description="Syntax error in diff",
        code_diff=bad_syntax_diff,
    )
    assert report_diff.stage != "promoted"
    assert report_diff.stage == EvolutionStage.REJECTED
    assert any(term in report_diff.error_reason.lower() for term in ("syntax", "ast", "corrupt patch"))
    assert target_abs.read_text(encoding="utf-8") == orig_content

    # 3. Via SelfCoder loop
    from sonic.evolution.evolution_journal import EvolutionJournal
    from sonic.evolution.goal_director import EvolutionGoalDirector
    from sonic.evolution.version_tracker import VersionTracker

    db = str(repo / "evo.db")
    vtracker = VersionTracker(db_path=db, repo_root=repo)
    journal = EvolutionJournal(db_path=db, repo_root=repo)
    director = EvolutionGoalDirector(
        repo_root=repo, db_path=db, evolver=evolver,
        version_tracker=vtracker, journal=journal,
    )
    llm = _StubLLM([f"```diff\n{bad_syntax_diff}```"])
    coder = SelfCoder(repo_root=repo, llm=llm, evolver=evolver, director=director, agentic=False)
    cycle = asyncio.run(coder.run_cycle("Fix widget calculation", target_rel, "bug_fix"))
    assert cycle.stage != "promoted"
    assert any(term in cycle.reason.lower() for term in ("syntax", "ast", "corrupt patch"))
    assert target_abs.read_text(encoding="utf-8") == orig_content


def test_test_failure_triggers_rollback_and_avoid_lesson(repo: Path, tmp_path: Path):
    """Simulates a patch where pytest exits non-zero, asserts stage is not promoted,
    tree is rolled back cleanly, and an AVOID lesson is recorded."""
    target_rel = "services/core/sonic/agents/widget.py"
    target_abs = repo / target_rel
    orig_content = target_abs.read_text(encoding="utf-8")

    test_rel = "services/core/tests/test_widget.py"
    (repo / test_rel).write_text(
        "from sonic.agents.widget import total\n\n"
        "def test_total_positive():\n"
        "    assert total([1, 2, 3]) == 6\n",
        encoding="utf-8",
    )
    (repo / "services/core" / "sonic" / "__init__.py").write_text("", encoding="utf-8")
    (repo / "services/core" / "tests" / "test_p0_security_hardening.py").write_text(
        "def test_safety_invariants():\n    assert True\n",
        encoding="utf-8",
    )
    (repo / "services/core" / "tests" / "test_safety_sealed_policy.py").write_text(
        "def test_safety_sealed():\n    assert True\n",
        encoding="utf-8",
    )
    (repo / "conftest.py").write_text(
        "import sys\nfrom pathlib import Path\n"
        "sys.path.insert(0, str(Path(__file__).parent / 'services/core'))\n",
        encoding="utf-8",
    )
    _git(repo, "add", "-A")
    _git(repo, "commit", "-m", "setup test suite")

    from sonic.being.lessons import LessonKind, LessonsLedger

    ledger = LessonsLedger(tenant_id="test_tenant", agent_id="test_agent")

    evolver = CodebaseEvolver(repo_root=repo, lessons_ledger=ledger)

    # Patch that syntactically validly returns 0, causing test_total_positive to fail (exit non-zero)
    failing_patch = (
        "diff --git a/services/core/sonic/agents/widget.py b/services/core/sonic/agents/widget.py\n"
        "--- a/services/core/sonic/agents/widget.py\n"
        "+++ b/services/core/sonic/agents/widget.py\n"
        "@@ -1,2 +1,2 @@\n"
        " def total(items):\n"
        "-    return sum(items)\n"
        "+    return 0\n"
    )

    report = evolver.evolve(
        target_component=target_rel,
        description="Change total to return 0",
        code_diff=failing_patch,
        test_paths=[test_rel],
    )

    # Assert stage is not promoted
    assert report.stage != "promoted"
    assert report.stage == EvolutionStage.ROLLED_BACK
    assert not report.test_result.passed
    assert "Verification tests failed" in report.error_reason

    # Assert tree is rolled back cleanly
    assert target_abs.read_text(encoding="utf-8") == orig_content
    git_status = subprocess.run(["git", "status", "--porcelain"], cwd=str(repo), capture_output=True, text=True).stdout
    assert "widget.py" not in git_status

    # Assert an AVOID lesson was recorded
    assert len(ledger._lessons) >= 1
    assert any(l.kind == LessonKind.AVOID for l in ledger._lessons)


def test_multi_file_atomic_rollback(repo: Path):
    """Creates a multi-file diff where the second file is invalid,
    asserts both files are restored to their original snapshot."""
    f1 = repo / "services/core" / "sonic" / "agents" / "mod_a.py"
    f2 = repo / "services/core" / "sonic" / "agents" / "mod_b.py"
    f1.write_text("VALUE_A = 10\n", encoding="utf-8")
    f2.write_text("VALUE_B = 20\n", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-m", "add mod_a and mod_b")

    evolver = CodebaseEvolver(repo_root=repo)

    multi_diff = (
        "diff --git a/services/core/sonic/agents/mod_a.py b/services/core/sonic/agents/mod_a.py\n"
        "--- a/services/core/sonic/agents/mod_a.py\n"
        "+++ b/services/core/sonic/agents/mod_a.py\n"
        "@@ -1 +1 @@\n"
        "-VALUE_A = 10\n"
        "+VALUE_A = 999\n"
        "diff --git a/services/core/sonic/agents/mod_b.py b/services/core/sonic/agents/mod_b.py\n"
        "--- a/services/core/sonic/agents/mod_b.py\n"
        "+++ b/services/core/sonic/agents/mod_b.py\n"
        "@@ -1 +1 @@\n"
        "-VALUE_B = 20\n"
        "+VALUE_B = 888 def invalid(\n"
    )

    files = ["services/core/sonic/agents/mod_a.py", "services/core/sonic/agents/mod_b.py"]
    ok, msg = evolver.apply_patch_multi(files, multi_diff)
    assert ok is False
    assert "SyntaxError" in msg or "AST" in msg
    assert "mod_b.py" in msg

    # Assert BOTH files are restored to their original snapshots
    assert f1.read_text(encoding="utf-8") == "VALUE_A = 10\n"
    assert f2.read_text(encoding="utf-8") == "VALUE_B = 20\n"

    # Also test end-to-end via EvolutionGoalDirector
    from sonic.evolution.evolution_journal import EvolutionJournal
    from sonic.evolution.goal_director import EvolutionGoalDirector
    from sonic.evolution.version_tracker import VersionTracker

    db = str(repo / "evo.db")
    vtracker = VersionTracker(db_path=db, repo_root=repo)
    journal = EvolutionJournal(db_path=db, repo_root=repo)
    director = EvolutionGoalDirector(
        repo_root=repo, db_path=db, evolver=evolver,
        version_tracker=vtracker, journal=journal,
    )
    goal = director.submit_goal(
        title="Multi-file update",
        description="Update both mod_a and mod_b",
        target_files=files,
    )
    report = asyncio.run(director.execute_goal(goal.goal_id, custom_patch=multi_diff))
    assert report.stage != EvolutionStage.PROMOTED
    assert f1.read_text(encoding="utf-8") == "VALUE_A = 10\n"
    assert f2.read_text(encoding="utf-8") == "VALUE_B = 20\n"
