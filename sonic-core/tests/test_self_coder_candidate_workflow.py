from __future__ import annotations

import asyncio
import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.dependencies.utils import get_dependant
from fastapi.testclient import TestClient

from sonic.api.routes.self_coder import (
    _build_coder,
    self_coder_approve_candidate,
    self_coder_status,
)
from sonic.api.routes.self_coder import (
    router as self_coder_router,
)
from sonic.auth.middleware import get_current_user, require_operator
from sonic.auth.models import User, UserRole
from sonic.evolution.codebase_evolver import CodebaseEvolver
from sonic.evolution.pipeline import EvolutionStage
from sonic.evolution.self_coder_candidates import SelfCoderCandidateStore
from sonic.evolution.self_coder_config import get_self_coder_config

DIFF = """diff --git a/sonic-core/app.py b/sonic-core/app.py
--- a/sonic-core/app.py
+++ b/sonic-core/app.py
@@ -1,2 +1,2 @@
 def value():
-    return "base"
+    return "candidate"
"""


def _git(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=root, check=True, capture_output=True, text=True
    )
    return result.stdout.strip()


@pytest.fixture
def candidate_repo(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "repository"
    tests = root / "sonic-core" / "tests"
    tests.mkdir(parents=True)
    (root / "sonic-core" / "app.py").write_text(
        'def value():\n    return "base"\n', encoding="utf-8"
    )
    (root / "sonic-core" / "security_gate.py").write_text(
        'def allowed(value):\n    return value != "blocked"\n', encoding="utf-8"
    )
    (tests / "test_candidate_app.py").write_text(
        "import sys\n"
        "from pathlib import Path\n"
        "sys.path.insert(0, str(Path(__file__).parents[1]))\n"
        "from app import value\n"
        "def test_candidate_behavior():\n"
        '    assert value() == "candidate"\n',
        encoding="utf-8",
    )
    (tests / "test_p0_security_hardening.py").write_text(
        "import sys\n"
        "from pathlib import Path\n"
        "sys.path.insert(0, str(Path(__file__).parents[1]))\n"
        "from security_gate import allowed\n"
        "def test_blocked_input_is_rejected():\n"
        '    assert allowed("blocked") is False\n',
        encoding="utf-8",
    )
    _git(root, "init")
    _git(root, "checkout", "-b", "feature")
    _git(root, "config", "user.name", "Test Operator")
    _git(root, "config", "user.email", "operator@example.test")
    _git(root, "add", "-A")
    _git(root, "commit", "-m", "base")
    return root, tmp_path / "candidate-store"


def _coder_factory(root: Path):
    class CandidateCoder:
        def __init__(self):
            self.evolver = CodebaseEvolver(repo_root=root)

        async def run(self, max_cycles: int, excluded_signals=None):
            assert max_cycles == 1
            report = self.evolver.evolve(
                target_component="sonic-core/app.py",
                description="Verify candidate behavior",
                code_diff=DIFF,
                test_paths=["sonic-core/tests/test_candidate_app.py"],
                report_only=True,
            )
            cycle = SimpleNamespace(
                target_file="sonic-core/app.py",
                signal="Verify candidate behavior",
                stage=report.stage.value,
                proposed_diff=DIFF if report.stage == EvolutionStage.AWAITING_APPROVAL else "",
                patch_files=report.files_affected,
                reason=report.error_reason,
                goal_id="",
                version_after="",
                commit_hash="",
                lines_added=1,
                lines_removed=1,
            )
            return SimpleNamespace(
                cycles=[cycle],
                to_text=lambda: f"verification={report.stage.value}",
            )

    return CandidateCoder()


def _new_candidate(root: Path, storage: Path, tenant: str = "tenant-a"):
    store = SelfCoderCandidateStore(root, storage)
    result = asyncio.run(
        store.run_candidate(tenant, 1, lambda worktree: _coder_factory(worktree))
    )
    return store, result["candidates"][0]


def test_candidate_run_isolated_from_active_checkout(candidate_repo):
    root, storage = candidate_repo
    before_head = _git(root, "rev-parse", "HEAD")
    before_branch = _git(root, "branch", "--show-current")
    before_status = _git(root, "status", "--porcelain")

    store, candidate = _new_candidate(root, storage)

    assert candidate["status"] == "auto_promoted"
    assert candidate["verification"]["full_test_suite"] is True
    assert candidate["verification"]["security_regression"] is True
    assert candidate["commit"]
    assert candidate["commit_branch"].startswith("sonic/self-coder/")
    assert "Expected impact" in candidate["report"]
    assert _git(root, "rev-parse", "HEAD") == before_head
    assert _git(root, "branch", "--show-current") == before_branch
    assert _git(root, "status", "--porcelain") == before_status
    assert "base" in (root / "sonic-core" / "app.py").read_text(encoding="utf-8")
    assert store.get(candidate["candidate_id"], "tenant-a")["diff"] == DIFF


def test_candidate_run_preserves_dirty_active_checkout(candidate_repo):
    root, storage = candidate_repo
    (root / "operator-work.txt").write_text("keep this change\n", encoding="utf-8")
    store = SelfCoderCandidateStore(root, storage)

    result = asyncio.run(
        store.run_candidate(
            "tenant-a", 1, lambda worktree: _coder_factory(worktree)
        )
    )

    assert (root / "operator-work.txt").read_text(encoding="utf-8") == (
        "keep this change\n"
    )
    assert result["candidates"][0]["status"] == "auto_promoted"
    assert "operator-work.txt" in result["candidates"][0]["working_tree_changes_excluded"]
    assert not list((storage / "worktrees").iterdir())


def test_candidate_auto_commits_to_local_branch_without_approval(candidate_repo):
    root, storage = candidate_repo
    original_branch = _git(root, "branch", "--show-current")
    original_head = _git(root, "rev-parse", "HEAD")
    store, candidate = _new_candidate(root, storage)

    assert _git(root, "rev-parse", "HEAD") == original_head
    assert _git(root, "branch", "--show-current") == original_branch
    assert candidate["status"] == "auto_promoted"

    promoted = store.get(candidate["candidate_id"], "tenant-a")

    assert promoted["status"] == "auto_promoted"
    assert promoted["commit"]
    assert promoted["commit_branch"].startswith("sonic/self-coder/")
    assert _git(root, "branch", "--show-current") == original_branch
    assert _git(root, "rev-parse", "HEAD") == original_head
    candidate_head = _git(root, "rev-parse", promoted["commit_branch"])
    assert candidate_head != original_head
    assert "candidate" in _git(root, "show", f"{promoted['commit_branch']}:sonic-core/app.py")
    with pytest.raises(ValueError, match="retained for audit"):
        store.cleanup(promoted["candidate_id"], "tenant-a")


def test_legacy_verified_candidate_can_still_be_manually_promoted(candidate_repo):
    root, storage = candidate_repo
    store, candidate = _new_candidate(root, storage)
    branch = candidate["commit_branch"]
    _git(root, "branch", "-D", branch)
    candidate["status"] = "verified"
    candidate["commit"] = ""
    candidate["commit_branch"] = ""
    candidate.pop("promotion", None)
    store.save(candidate)

    approved = store.approve(candidate["candidate_id"], "tenant-a")

    assert approved["status"] == "approved"
    assert approved["commit"]
    assert approved["commit_branch"] == branch


def test_handled_signal_is_excluded_from_later_candidate_runs(candidate_repo):
    root, storage = candidate_repo
    store, candidate = _new_candidate(root, storage)
    candidate["status"] = "rejected"
    store.save(candidate)
    captured = {}

    class CapturingCoder:
        async def run(self, max_cycles: int, excluded_signals=None):
            captured["max_cycles"] = max_cycles
            captured["excluded_signals"] = excluded_signals
            return SimpleNamespace(cycles=[], to_text=lambda: "No new signal")

    result = asyncio.run(
        store.run_candidate(
            "tenant-a", 1, lambda _worktree: CapturingCoder()
        )
    )

    assert captured["max_cycles"] == 1
    assert captured["excluded_signals"] == {
        ("sonic-core/app.py", "Verify candidate behavior")
    }
    assert result["candidates"][0]["status"] == "no_candidate"


def test_rejection_and_stale_revalidation_do_not_commit(candidate_repo):
    root, storage = candidate_repo
    store, rejected = _new_candidate(root, storage)
    _git(root, "branch", "-D", rejected["commit_branch"])
    rejected["status"] = "verified"
    rejected["commit"] = ""
    rejected["commit_branch"] = ""
    rejected.pop("promotion", None)
    store.save(rejected)
    head_before_reject = _git(root, "rev-parse", "HEAD")
    rejected_record = store.reject(rejected["candidate_id"], "tenant-a", "Not wanted")
    assert rejected_record["status"] == "rejected"
    assert _git(root, "rev-parse", "HEAD") == head_before_reject
    store.cleanup(rejected["candidate_id"], "tenant-a")
    assert store.get(rejected["candidate_id"], "tenant-a") is None

    store, stale = _new_candidate(root, storage)
    changed = root / "unrelated.txt"
    changed.write_text("operator change\n", encoding="utf-8")
    _git(root, "add", "unrelated.txt")
    _git(root, "commit", "-m", "advance base")
    current_head = _git(root, "rev-parse", "HEAD")
    stale_record = store.get(stale["candidate_id"], "tenant-a")
    old_branch = stale_record["commit_branch"]
    _git(root, "branch", "-D", old_branch)
    stale_record["status"] = "verified"
    stale_record["commit"] = ""
    stale_record["commit_branch"] = ""
    stale_record.pop("promotion", None)
    store.save(stale_record)
    with pytest.raises(ValueError, match="base commit changed"):
        store.approve(stale["candidate_id"], "tenant-a")
    assert _git(root, "rev-parse", "HEAD") == current_head
    assert not _git(root, "branch", "--list", f"sonic/self-coder/{stale['candidate_id']}")
    assert store.get(stale["candidate_id"], "tenant-a")["status"] == "stale"


def test_tampered_candidate_is_not_approved(candidate_repo):
    root, storage = candidate_repo
    store, candidate = _new_candidate(root, storage)
    _git(root, "branch", "-D", candidate["commit_branch"])
    candidate["status"] = "verified"
    candidate["commit"] = ""
    candidate["commit_branch"] = ""
    candidate.pop("promotion", None)
    store.save(candidate)
    before_head = _git(root, "rev-parse", "HEAD")
    record_path = storage / f"{candidate['candidate_id']}.json"
    record = json.loads(record_path.read_text(encoding="utf-8"))
    record["diff"] = record["diff"].replace('"candidate"', '"altered"')
    record_path.write_text(json.dumps(record), encoding="utf-8")

    with pytest.raises(ValueError, match="integrity check failed"):
        store.approve(candidate["candidate_id"], "tenant-a")
    assert _git(root, "rev-parse", "HEAD") == before_head
    assert store.get(candidate["candidate_id"], "tenant-a")["status"] == "tampered"


def test_candidate_tenant_isolation_and_operator_dependency(candidate_repo):
    root, storage = candidate_repo
    store, candidate = _new_candidate(root, storage)
    assert store.get(candidate["candidate_id"], "tenant-b") is None
    with pytest.raises(KeyError, match="Candidate not found"):
        store.approve(candidate["candidate_id"], "tenant-b")
    route_dependencies = get_dependant(
        path="/self-coder/candidates/{candidate_id}/approve",
        call=self_coder_approve_candidate,
    ).dependencies
    assert any(dependency.call is require_operator for dependency in route_dependencies)


def test_verification_rejects_all_skipped_test_runs(candidate_repo, monkeypatch):
    root, _ = candidate_repo
    evolver = CodebaseEvolver(repo_root=root)
    monkeypatch.setattr(
        "sonic.evolution.codebase_evolver.subprocess.run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args[0], 0, stdout="5 skipped in 0.01s\n", stderr=""
        ),
    )
    result = evolver.run_tests(["sonic-core/tests"])
    assert result.passed is False
    assert result.skipped_tests == 5


def test_test_runner_prefers_isolated_worktree_imports(candidate_repo, monkeypatch):
    root, _ = candidate_repo
    evolver = CodebaseEvolver(repo_root=root)
    captured = {}

    def run(*args, **kwargs):
        captured["env"] = kwargs["env"]
        return subprocess.CompletedProcess(args[0], 0, stdout="1 passed in 0.01s", stderr="")

    monkeypatch.setattr("sonic.evolution.codebase_evolver.subprocess.run", run)
    result = evolver.run_tests(["sonic-core/tests"])
    assert result.passed is True
    assert Path(captured["env"]["PYTHONPATH"].split(os.pathsep)[0]) == root / "sonic-core"


def test_candidate_approval_endpoint_enforces_operator_rbac():
    app = FastAPI()
    app.include_router(self_coder_router, prefix="/self-coder")
    app.dependency_overrides[get_current_user] = lambda: User(
        email="auditor@example.test",
        name="Auditor",
        tenant_id="tenant-a",
        role=UserRole.AUDITOR,
    )
    response = TestClient(app).post(
        "/self-coder/candidates/0123456789abcdef0123456789abcdef/approve"
    )
    assert response.status_code == 403


def test_api_coder_reuses_evolution_engine_configuration(tmp_path, monkeypatch):
    import sonic.api.routes.self_coder as self_coder_route

    router = SimpleNamespace(providers=[object()])
    strategy = object()
    method_lab = object()
    toolsmith = object()
    lessons = object()
    template = SimpleNamespace(
        llm_client=router,
        strategy_engine=strategy,
        method_lab=method_lab,
        toolsmith=toolsmith,
        lessons_ledger=lessons,
    )
    monkeypatch.setattr(self_coder_route, "_configured_evolution_engine", template)

    coder = _build_coder(tmp_path, require_llm=True)

    assert coder.evolution_engine.llm_client is router
    assert coder.evolution_engine.strategy_engine is strategy
    assert coder.evolution_engine.method_lab is method_lab
    assert coder.evolution_engine.toolsmith is toolsmith
    assert coder.evolution_engine.lessons_ledger is lessons
    assert coder.evolver is coder.evolution_engine.codebase_evolver
    assert coder.director is coder.evolution_engine.goal_director
    assert coder.allow_writes is False
    assert coder.push is False
    assert coder.director.list_goals() == []


@pytest.mark.asyncio
async def test_self_coder_status_reports_background_loop_independently(monkeypatch):
    user = User(email="operator@example.test", name="Operator", role=UserRole.OPERATOR)
    monkeypatch.delenv("SONIC_ENABLE_BEING_LIFE_LOOP", raising=False)
    monkeypatch.setenv("SONIC_SELF_CODER_EVERY_TICKS", "7")
    monkeypatch.setenv("SONIC_SELF_CODER_INTERVAL_SECONDS", "900")
    status = await self_coder_status(user=user)
    assert status["enabled"] is False
    assert "background self-coder is not running" in status["reason"].lower()
    assert status["every_ticks"] == 7
    assert status["interval_seconds"] == 900


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        (
            {
                "SONIC_SELF_CODER_ENABLED": "no",
                "SONIC_SELF_CODER_EVERY_TICKS": "3",
            },
            (False, 3),
        ),
        ({"SONIC_SELF_CODER_EVERY_TICKS": "10000"}, (True, 10000)),
    ],
)
def test_self_coder_config_parses_explicit_values(values, expected):
    config = get_self_coder_config(values)
    assert (config.enabled, config.every_ticks) == expected
    assert config.interval_seconds == 3600


@pytest.mark.parametrize("value", ["0", "-1", "bad", "10001"])
def test_self_coder_interval_rejects_invalid_values(value):
    with pytest.raises(ValueError, match="SONIC_SELF_CODER_EVERY_TICKS"):
        get_self_coder_config({"SONIC_SELF_CODER_EVERY_TICKS": value})


@pytest.mark.parametrize("value", ["0", "59", "86401", "bad"])
def test_self_coder_interval_seconds_rejects_invalid_values(value):
    with pytest.raises(ValueError, match="SONIC_SELF_CODER_INTERVAL_SECONDS"):
        get_self_coder_config({"SONIC_SELF_CODER_INTERVAL_SECONDS": value})
