"""
Tests for Continuous Codebase Self-Evolution Engine (`CodebaseEvolver`)
=====================================================================
Verifies:
1. Safety Invariant: AI is strictly blocked from proposing edits to safety kernels.
2. Syntax Pre-flight: Python AST catches syntax errors before touching files.
3. Atomic Rollback: When tests fail, all modifications are cleanly rolled back.
4. Lessons Integration: Test failures automatically record [AVOID] lessons.
5. Evolution Summary Report: Generates rich, structured markdown reports.
6. Auto-Promotion Gate: Commits and pushes without human approval when tests pass.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from sonic.being.lessons import LessonKind, LessonsLedger
from sonic.evolution.codebase_evolver import (
    CodebaseEvolver,
    EvolutionSummaryReport,
    SafetyInvariantViolation,
    TestExecutionResult,
)
from sonic.evolution.pipeline import EvolutionPipeline, EvolutionStage


@pytest.fixture
def temp_repo(tmp_path):
    """Creates a temporary isolated repository structure for evolution testing."""
    repo = tmp_path / "mock_repo"
    repo.mkdir()

    # Create directory structure
    (repo / "sonic" / "safety").mkdir(parents=True)
    (repo / "sonic" / "tools" / "adapters").mkdir(parents=True)
    (repo / "sonic" / "research").mkdir(parents=True)
    (repo / "tests").mkdir(parents=True)

    # Populate a protected safety file
    (repo / "sonic" / "safety" / "kernel.py").write_text(
        "# PROTECTED SAFETY KERNEL\nclass SafetyKernel:\n    pass\n", encoding="utf-8"
    )

    # Populate an evolvable tool adapter file
    (repo / "sonic" / "tools" / "adapters" / "custom_parser.py").write_text(
        "def parse_status(code: int) -> str:\n    return 'OK' if code == 200 else 'ERR'\n",
        encoding="utf-8",
    )

    # Populate a test file
    (repo / "tests" / "test_parser.py").write_text(
        "from sonic.tools.adapters.custom_parser import parse_status\n\n"
        "def test_status():\n    assert parse_status(200) == 'OK'\n",
        encoding="utf-8",
    )

    return repo


class TestSafetyInvariants:
    def test_rejects_modification_to_protected_safety_components(self, temp_repo):
        evolver = CodebaseEvolver(repo_root=temp_repo)

        protected_targets = [
            "sonic/safety/kernel.py",
            "sonic/safety/sealed.py",
            "sonic/safety/action_policy.py",
            "sonic/kernel/action_broker.py",
            "sonic/sandbox/egress.py",
            "sonic-kernel-rs/src/safety/mod.rs",
            "sonic/auth/models.py",
            "sonic/sandbox/providers/docker.py",
            "sonic/evolution/patch_synthesizer.py",
            "sonic/evolution/codebase_evolver.py",
            "sonic/evolution/goal_director.py",
            "sonic/evolution/pipeline.py",
            "sonic/evolution/self_coder.py",
            "sonic/evolution/coding_agent",
            "sonic-core/tests/test_p0_security_hardening.py",
            "sonic/agents/judge.py",
            "sonic/evidence/collector.py",
            "sonic-core/tests/test_safety_sealed_policy.py",
            "sonic/evolution/self_coder_candidates.py",
            "sonic/evolution/continuous_runner.py",
            "sonic/sandbox/factory.py",
            "tests/test_p0_security_hardening.py",
            "tests/test_safety_sealed_policy.py",
        ]

        for target in protected_targets:
            assert evolver.is_protected(target) is True
            report = evolver.evolve(
                target_component=target,
                description="Bypass safety filters for speed",
                code_diff="def bypass(): pass",
            )
            assert report.stage == EvolutionStage.REJECTED
            assert "Safety Invariant Violation" in report.error_reason

        # Ensure protected file was NOT touched
        kernel_content = (temp_repo / "sonic" / "safety" / "kernel.py").read_text(encoding="utf-8")
        assert "PROTECTED SAFETY KERNEL" in kernel_content


class TestSyntaxAndStaging:
    def test_catches_ast_syntax_errors_before_touching_disk(self, temp_repo):
        evolver = CodebaseEvolver(repo_root=temp_repo)
        target = "sonic/tools/adapters/custom_parser.py"

        orig_content = (temp_repo / target).read_text(encoding="utf-8")

        # Propose broken Python code with syntax error
        broken_code = "def parse_status(code: int) -> str\n    return 'broken'"

        report = evolver.evolve(
            target_component=target,
            description="Broken syntax upgrade",
            code_diff=broken_code,
        )

        assert report.stage == EvolutionStage.REJECTED
        assert "SyntaxError" in report.error_reason
        assert report.ast_validated is False

        # File content must remain completely unchanged
        current_content = (temp_repo / target).read_text(encoding="utf-8")
        assert current_content == orig_content


class TestRollbackAndLessons:
    def test_rolls_back_and_records_avoid_lesson_when_tests_fail(self, temp_repo, tmp_path):
        ledger = LessonsLedger(tenant_id="test-tenant", agent_id="test-agent")
        evolver = CodebaseEvolver(repo_root=temp_repo, lessons_ledger=ledger)
        target = "sonic/tools/adapters/custom_parser.py"
        orig_content = (temp_repo / target).read_text(encoding="utf-8")

        # Code is valid Python AST, but changes behavior and breaks the test
        breaking_code = "def parse_status(code: int) -> str:\n    return 'ALWAYS_FAIL'\n"

        # Mock test runner to simulate failed tests
        mock_res = TestExecutionResult(
            passed=False,
            total_tests=5,
            passed_tests=3,
            failed_tests=2,
            duration_seconds=1.2,
            error_message="AssertionError: 'ALWAYS_FAIL' != 'OK'",
            raw_output="FAILED test_parser.py::test_status",
        )

        with patch.object(evolver, "run_tests", return_value=mock_res):
            report = evolver.evolve(
                target_component=target,
                description="Mutate status parsing return value",
                code_diff=breaking_code,
                auto_promote=True,
            )

        assert report.stage == EvolutionStage.ROLLED_BACK
        assert report.test_result.passed is False
        assert "Verification tests failed" in report.error_reason

        # Verified: Atomic rollback restored the original file
        assert (temp_repo / target).read_text(encoding="utf-8") == orig_content

        # Verified: AVOID lesson was recorded in the ledger
        lessons = ledger.all()
        assert len(lessons) >= 1
        assert any(l.kind == LessonKind.AVOID for l in lessons)


class TestSuccessfulEvolutionAndSummary:
    def test_successful_upgrade_with_auto_promotion_and_markdown_summary(self, temp_repo):
        evolver = CodebaseEvolver(repo_root=temp_repo)
        target = "sonic/tools/adapters/custom_parser.py"

        # Enhanced, backwards-compatible implementation
        enhanced_code = (
            "def parse_status(code: int) -> str:\n"
            "    if code in (200, 201, 204):\n"
            "        return 'OK'\n"
            "    return 'ERR'\n"
        )

        mock_test_res = TestExecutionResult(
            passed=True,
            total_tests=4,
            passed_tests=4,
            failed_tests=0,
            duration_seconds=0.45,
            raw_output="4 passed in 0.45s",
        )

        with patch.object(evolver, "run_tests", return_value=mock_test_res), \
             patch.object(evolver, "run_security_regression", return_value=True), \
             patch.object(evolver, "commit_and_push", return_value=(True, "a1b2c3d", True, "Pushed to origin/main")):

            report = evolver.evolve(
                target_component=target,
                description="Support HTTP 201 and 204 in parse_status",
                code_diff=enhanced_code,
                auto_promote=True,
                push=True,
            )

        assert report.stage == EvolutionStage.PROMOTED
        assert report.ast_validated is True
        assert report.test_result.passed is True
        assert report.security_regression_passed is True
        assert report.git_committed is True
        assert report.git_commit_hash == "a1b2c3d"
        assert report.git_pushed is True

        # Markdown summary verification
        md = report.to_markdown()
        assert "# 🧬 Codebase Self-Evolution Summary:" in md
        assert "PROMOTED (AUTO-MERGED)" in md
        assert "Support HTTP 201 and 204 in parse_status" in md
        assert "Passed: `4` / `4`" in md
        assert "a1b2c3d" in md
        assert "Origin Updated" in md

    def test_manual_approval_mode_when_auto_promote_is_false(self, temp_repo):
        evolver = CodebaseEvolver(repo_root=temp_repo)
        target = "sonic/tools/adapters/custom_parser.py"

        enhanced_code = "def parse_status(code: int) -> str:\n    return 'OK'\n"

        mock_test_res = TestExecutionResult(passed=True, total_tests=2, passed_tests=2)

        with patch.object(evolver, "run_tests", return_value=mock_test_res), \
             patch.object(evolver, "run_security_regression", return_value=True):

            report = evolver.evolve(
                target_component=target,
                description="Minor refactor",
                code_diff=enhanced_code,
                auto_promote=False,
            )

        assert report.stage == EvolutionStage.AWAITING_APPROVAL
        assert report.git_committed is False
        assert "AWAITING OPERATOR APPROVAL" in report.to_markdown()


class TestCodebaseEvolverBugFixes:
    def test_gate5_auto_promote_stages_all_pre_applied_files(self, temp_repo):
        evolver = CodebaseEvolver(repo_root=temp_repo)
        target = "sonic/tools/adapters/custom_parser.py"
        other_file = "tests/test_parser.py"

        mock_test_res = TestExecutionResult(passed=True, total_tests=2, passed_tests=2)
        mock_commit_and_push = MagicMock(return_value=(True, "commit123", True, "Pushed"))

        with patch.object(evolver, "run_tests", return_value=mock_test_res), \
             patch.object(evolver, "run_security_regression", return_value=True), \
             patch.object(evolver, "commit_and_push", mock_commit_and_push):

            report = evolver.evolve(
                target_component=target,
                description="Multi-file upgrade",
                code_diff="pass",
                auto_promote=True,
                push=True,
                pre_applied_files=[target, other_file],
            )

        assert report.stage == EvolutionStage.PROMOTED
        mock_commit_and_push.assert_called_once_with(
            proposal_id=report.proposal_id,
            description="Multi-file upgrade",
            files=[target, other_file],
            push=True,
        )

    def test_apply_patch_passes_newline_lf_to_named_temporary_file(self, temp_repo):
        evolver = CodebaseEvolver(repo_root=temp_repo)
        target = "sonic/tools/adapters/custom_parser.py"
        unified_diff = (
            "--- a/sonic/tools/adapters/custom_parser.py\n"
            "+++ b/sonic/tools/adapters/custom_parser.py\n"
            "@@ -1,2 +1,2 @@\n"
            "-def parse_status(code: int) -> str:\n"
            "+def parse_status(code: int) -> str:\n"
        )

        with patch("tempfile.NamedTemporaryFile") as mock_ntf, \
             patch("subprocess.run") as mock_run:
            mock_file = MagicMock()
            mock_file.name = "dummy.patch"
            mock_ntf.return_value.__enter__.return_value = mock_file
            mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

            evolver.apply_patch(target, unified_diff)

            mock_ntf.assert_called_once_with(
                "w",
                suffix=".patch",
                delete=False,
                encoding="utf-8",
                newline="\n",
            )

    def test_apply_patch_multi_catches_general_exception_and_rolls_back(self, temp_repo):
        evolver = CodebaseEvolver(repo_root=temp_repo)
        file1 = "sonic/tools/adapters/custom_parser.py"
        orig_content = (temp_repo / file1).read_text(encoding="utf-8")

        with patch.object(evolver, "apply_patch", return_value=(True, "ok", 1, 1)), \
             patch("ast.parse", side_effect=ValueError("malformed token or null byte")):
            diff = (
                "--- a/sonic/tools/adapters/custom_parser.py\n"
                "+++ b/sonic/tools/adapters/custom_parser.py\n"
                "@@ -1 +1 @@\n"
                "-old\n"
                "+new\n"
            )
            ok, msg = evolver.apply_patch_multi([file1], diff)

        assert ok is False
        assert "AST verification failed" in msg or "malformed token" in msg
        # Rollback was executed
        assert (temp_repo / file1).read_text(encoding="utf-8") == orig_content

    def test_diff_files_captures_deleted_files_with_dev_null(self):
        # Case 1: git diff format with /dev/null
        git_diff_dev_null = (
            "diff --git a/sonic/tools/old_module.py b//dev/null\n"
            "deleted file mode 100644\n"
            "--- a/sonic/tools/old_module.py\n"
            "+++ /dev/null\n"
        )
        files = CodebaseEvolver._diff_files(git_diff_dev_null)
        assert files == ["sonic/tools/old_module.py"]

        # Case 2: git diff format with b/dev/null
        git_diff_b_dev_null = (
            "diff --git a/sonic/tools/removed.py b/dev/null\n"
            "deleted file mode 100644\n"
        )
        files2 = CodebaseEvolver._diff_files(git_diff_b_dev_null)
        assert files2 == ["sonic/tools/removed.py"]

        # Case 3: unified diff without diff --git header
        unified_diff_dev_null = (
            "--- a/sonic/tools/legacy.py\n"
            "+++ /dev/null\n"
            "@@ -1,5 +0,0 @@\n"
        )
        files3 = CodebaseEvolver._diff_files(unified_diff_dev_null)
        assert files3 == ["sonic/tools/legacy.py"]

    def test_rejects_repository_escape_and_protected_secondary_diff_path(self, temp_repo, tmp_path):
        evolver = CodebaseEvolver(repo_root=temp_repo)
        target = "sonic/tools/adapters/custom_parser.py"
        outside = tmp_path / "outside.py"
        outside.write_text("VALUE = 1\n", encoding="utf-8")

        escaped = evolver.evolve(
            target_component="../outside.py",
            description="Escape repository root",
            code_diff="VALUE = 2\n",
        )
        assert escaped.stage == EvolutionStage.REJECTED
        assert "escapes repository root" in escaped.error_reason
        assert outside.read_text(encoding="utf-8") == "VALUE = 1\n"

        protected_diff = (
            "diff --git a/sonic/tools/adapters/custom_parser.py b/sonic/tools/adapters/custom_parser.py\n"
            "--- a/sonic/tools/adapters/custom_parser.py\n"
            "+++ b/sonic/tools/adapters/custom_parser.py\n"
            "@@ -1 +1 @@\n"
            "-old\n"
            "+new\n"
            "diff --git a/sonic/auth/models.py b/sonic/auth/models.py\n"
            "--- a/sonic/auth/models.py\n"
            "+++ b/sonic/auth/models.py\n"
            "@@ -1 +1 @@\n"
            "-old\n"
            "+new\n"
        )
        result = evolver.evolve(
            target_component=target,
            description="Try to modify protected auth code",
            code_diff=protected_diff,
        )
        assert result.stage == EvolutionStage.REJECTED
        assert "Safety Invariant Violation" in result.error_reason

    def test_failed_commit_rolls_back_and_is_not_promoted(self, temp_repo):
        evolver = CodebaseEvolver(repo_root=temp_repo)
        target = "sonic/tools/adapters/custom_parser.py"
        original = (temp_repo / target).read_text(encoding="utf-8")
        replacement = "def parse_status(code: int) -> str:\n    return 'NEW'\n"
        passing = TestExecutionResult(passed=True, total_tests=1, passed_tests=1)

        with patch.object(evolver, "run_tests", return_value=passing), \
             patch.object(evolver, "run_import_checks", return_value=(True, "ok")), \
             patch.object(evolver, "run_security_regression", return_value=True), \
             patch.object(evolver, "commit_and_push", return_value=(False, "", False, "commit denied")):
            report = evolver.evolve(
                target_component=target,
                description="Commit failure must not promote",
                code_diff=replacement,
                auto_promote=True,
            )

        assert report.stage == EvolutionStage.ROLLED_BACK
        assert report.git_committed is False
        assert "commit denied" in report.error_reason
        assert (temp_repo / target).read_text(encoding="utf-8") == original

    def test_legitimate_authoring_module_not_blocked_by_auth_substring(self, temp_repo):
        evolver = CodebaseEvolver(repo_root=temp_repo)
        # sonic/authoring/generator.py must NOT be blocked because sonic/auth is only a substring
        assert evolver.is_protected("sonic/authoring/generator.py") is False
        assert evolver.is_protected("sonic-core/sonic/authoring/generator.py") is False

        pipeline = EvolutionPipeline()
        prop, msg = pipeline.submit_proposal(
            target_component="sonic/authoring/generator.py",
            description="Add prompt template generator",
            code_diff="def generate(): pass\n",
        )
        assert prop.stage != EvolutionStage.REJECTED
        assert "Cannot self-modify" not in msg

    def test_timeout_ceiling_and_floor(self, monkeypatch):
        monkeypatch.setenv("SONIC_EVOLUTION_TEST_TIMEOUT", "30")
        assert CodebaseEvolver._resolve_test_timeout() == 120

        monkeypatch.setenv("SONIC_EVOLUTION_TEST_TIMEOUT", "3600")
        assert CodebaseEvolver._resolve_test_timeout() == 1800

        monkeypatch.setenv("SONIC_EVOLUTION_TEST_TIMEOUT", "600")
        assert CodebaseEvolver._resolve_test_timeout() == 600

    def test_security_regression_requires_both_suites_on_disk(self, temp_repo):
        evolver = CodebaseEvolver(repo_root=temp_repo)
        p0_path = temp_repo / "sonic-core" / "tests" / "test_p0_security_hardening.py"
        safety_path = temp_repo / "sonic-core" / "tests" / "test_safety_sealed_policy.py"
        p0_path.parent.mkdir(parents=True, exist_ok=True)
        p0_path.write_text("def test_p0(): pass\n", encoding="utf-8")

        # Missing test_safety_sealed_policy.py -> must fail closed
        if safety_path.exists():
            safety_path.unlink()
        assert evolver.run_security_regression() is False

        # Create both -> should run tests
        safety_path.write_text("def test_safety(): pass\n", encoding="utf-8")
        with patch.object(evolver, "run_tests", return_value=TestExecutionResult(passed=True)):
            assert evolver.run_security_regression() is True

    def test_test_subprocess_env_scrubs_secrets(self):
        from sonic.evolution.codebase_evolver import _scrub_env_secrets

        dirty_env = {
            "PATH": "/usr/bin:/bin",
            "PYTHONPATH": "/repo/sonic-core",
            "JWT_SECRET": "super_secret_jwt",
            "SECRET_KEY": "super_secret_key",
            "OPENAI_API_KEY": "sk-1234567890",
            "ANTHROPIC_API_KEY": "ant-key-123",
            "DATABASE_URL": "postgresql://user:pass@localhost/db",
            "NEO4J_PASSWORD": "mypassword",
            "GH_TOKEN": "ghp_abcdef",
            "MY_SERVICE_TOKEN": "tok-123",
            "APP_ENV": "test",
        }
        clean_env = _scrub_env_secrets(dirty_env)
        assert "PATH" in clean_env
        assert "PYTHONPATH" in clean_env
        assert "APP_ENV" in clean_env
        assert "JWT_SECRET" not in clean_env
        assert "SECRET_KEY" not in clean_env
        assert "OPENAI_API_KEY" not in clean_env
        assert "ANTHROPIC_API_KEY" not in clean_env
        assert "DATABASE_URL" not in clean_env
        assert "NEO4J_PASSWORD" not in clean_env
        assert "GH_TOKEN" not in clean_env
        assert "MY_SERVICE_TOKEN" not in clean_env

    def test_protected_component_in_nested_worktree_path_matches(self):
        from sonic.evolution.pipeline import is_protected_component

        assert is_protected_component("worktrees/evo-123/sonic/kernel/action_broker.py") is True
        assert is_protected_component("worktrees/evo-123/sonic/sandbox/egress.py") is True
        assert is_protected_component("worktrees/evo-123/sonic/safety/sealed.py") is True
        assert is_protected_component("worktrees/evo-123/sonic/agents/judge.py") is True
        assert is_protected_component("worktrees/evo-123/sonic/authoring/generator.py") is False

    def test_security_regression_resolves_when_repo_root_is_sonic_core(self, tmp_path):
        # When repo_root is already inside sonic-core
        tests_dir = tmp_path / "tests"
        tests_dir.mkdir(parents=True)
        (tests_dir / "test_p0_security_hardening.py").write_text("def test_p0(): pass\n", encoding="utf-8")
        (tests_dir / "test_safety_sealed_policy.py").write_text("def test_safety(): pass\n", encoding="utf-8")

        evolver = CodebaseEvolver(repo_root=tmp_path)
        with patch.object(evolver, "run_tests", return_value=TestExecutionResult(passed=True)) as mock_run:
            assert evolver.run_security_regression() is True
            mock_run.assert_called_once_with(["tests/test_p0_security_hardening.py", "tests/test_safety_sealed_policy.py"])

