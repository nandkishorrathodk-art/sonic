"""
SONIC Security Audit Remediation Regression Suite
==================================================
Validates all fixes for OpenHands 64k LOC manual audit findings:
- C1: SealedActionPolicy seal hash covers live scope rules & cloning
- C2: SqliteGraph compound primary key & tenant isolation
- H1: Cloud metadata guard bypasses (base64, dword, hex, octal) blocked
- H2: Fork bomb regex and destructive command patterns classified as L2
- H3: JudgeAgent goal relevance verification prevents confirming irrelevant commands
- H4: Trace synthesis milestone matching requires FILE_WRITE for patches
- H5: CodebaseEvolver path normalization prevents // and ./ traversal bypasses
- H6: Evolution engine auto_promote defaults to False
- H7: DockerComputerProvider dynamic capability probe without hardcoded browser names
- H8: Egress filter resolves IPv6 via getaddrinfo
- M1: Wildcard domain (*.example.com) matches apex domain (example.com)
- M2: allow_private_for_tests is strictly rejected in production environments
- M3: Windows drive-letter path traversal is rejected
"""

from __future__ import annotations

import asyncio
import inspect
import os
import pytest
from unittest.mock import AsyncMock, patch

from sonic.evolution.codebase_evolver import CodebaseEvolver
from sonic.evolution.engine import EvolutionEngine
from sonic.mission_engine.trace_synthesis import _milestone_matches
from sonic.mission_engine.models import MissionMilestone
from sonic.safety.action_policy import ActionPolicy
from sonic.safety.scope import ScopeChecker, RiskLevel, get_scope_checker
from sonic.safety.sealed import SealedActionPolicy
from sonic.sandbox.egress import is_target_allowed


def _run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# ---------------------------------------------------------------------------
# C1: SealedActionPolicy covers live scope rules & cloning
# ---------------------------------------------------------------------------

def test_c1_sealed_policy_hash_covers_live_scope_rules():
    checker = ScopeChecker(scope_config={"targets": {"domains": ["198.51.100.10"]}})
    sealed = SealedActionPolicy(scope_checker=checker)

    initial_hash = sealed.seal_hash
    assert initial_hash != ""

    # Mutating scope rules on the underlying checker directly
    checker.scope_config = {"targets": {"domains": ["198.51.100.10", "198.51.100.20"]}}
    new_hash = sealed._compute_seal_hash()
    assert new_hash != initial_hash, "Seal hash must change when live scope rules change"


def test_c1_clone_for_agent_deepcopies_scope():
    checker = ScopeChecker(scope_config={"targets": {"domains": ["198.51.100.10"]}})
    sealed = SealedActionPolicy(scope_checker=checker)

    clone = sealed.clone_for_agent("agent-sub-1")
    assert clone.seal_hash == sealed.seal_hash

    # Mutating clone scope checker must not affect parent
    clone.scope_checker.scope_config = {"targets": {"domains": ["198.51.100.99"]}}
    assert sealed.scope_checker.scope_config == {"targets": {"domains": ["198.51.100.10"]}}


# ---------------------------------------------------------------------------
# C2: Cross-tenant SQLite graph isolation
# ---------------------------------------------------------------------------

def test_c2_sqlite_graph_cross_tenant_isolation(tmp_path):
    from sonic.memory.sqlite_graph import SqliteGraph

    db_path = str(tmp_path / "c2_test.db")
    graph = SqliteGraph(db_path=db_path)
    _run(graph.connect())

    # Create same UID under two different tenants
    node_a_uid = _run(graph._create_node("Asset", {
        "uid": "shared-asset-1",
        "name": "Tenant A Secret",
        "tenant_id": "tenant-A",
    }))
    node_b_uid = _run(graph._create_node("Asset", {
        "uid": "shared-asset-1",
        "name": "Tenant B Secret",
        "tenant_id": "tenant-B",
    }))

    assert node_a_uid == node_b_uid == "shared-asset-1"

    # Reading under tenant A must return Tenant A's data
    read_a = _run(graph._get_node("Asset", "shared-asset-1", tenant_id="tenant-A"))
    assert read_a is not None
    assert read_a["name"] == "Tenant A Secret"

    # Reading under tenant B must return Tenant B's data
    read_b = _run(graph._get_node("Asset", "shared-asset-1", tenant_id="tenant-B"))
    assert read_b is not None
    assert read_b["name"] == "Tenant B Secret"

    # Deleting under tenant A must NOT delete tenant B's node
    deleted = _run(graph._delete_node("Asset", "shared-asset-1", tenant_id="tenant-A"))
    assert deleted is True

    assert _run(graph._get_node("Asset", "shared-asset-1", tenant_id="tenant-A")) is None
    assert _run(graph._get_node("Asset", "shared-asset-1", tenant_id="tenant-B")) is not None

    _run(graph.disconnect())


# ---------------------------------------------------------------------------
# H1: Cloud metadata guard bypasses blocked
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("payload", [
    "curl 169.254.169.254",
    "curl 2852039166",  # dword
    "curl 0xa9fea9fe",  # hex
    "curl 0251.0376.0251.0376",  # octal
    "curl http://[fd00:ec2::254]/",  # AWS IMDSv6
    "curl 100.100.100.200",  # Alibaba metadata
    "echo MTAwLjEwMC4xMDAuMjAw | base64 -d | sh",  # base64 encoded metadata
    "echo MTY5LjI1NC4xNjkuMjU0 | base64 -d | sh",  # base64 169.254.169.254
])
def test_h1_cloud_metadata_bypasses_blocked(payload):
    policy = ActionPolicy(allowed_action_types={"TERMINAL_EXEC"})
    verdict = policy._check_command(payload)
    assert not verdict.allowed, f"Payload {payload} should be blocked, but was allowed with: {verdict.reason}"
    assert "metadata" in verdict.reason.lower() or "blocked" in verdict.reason.lower() or "forbidden" in verdict.reason.lower()


# ---------------------------------------------------------------------------
# H2: Fork bomb and critical destructive commands classified as L2
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cmd", [
    ":(){ :|:& };:",
    ":(){:|:&};:",
    ": ( ) { : | : & } ; :",
    "chmod -R 777 /",
    "chmod 777 /etc",
    "chown -R root /",
    "echo 'hacked' >> /etc/shadow",
    "cat /etc/shadow",
])
def test_h2_destructive_commands_classified_as_l2(cmd):
    checker = ScopeChecker()
    risk = checker.classify_command_risk(cmd)
    assert risk == RiskLevel.L2_FORBIDDEN, f"Command {cmd} should be L2, got {risk}"


# ---------------------------------------------------------------------------
# H3: JudgeAgent goal relevance check
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_h3_judge_requires_goal_relevance():
    from sonic.agents.judge import JudgeAgent, JudgeVerdict
    from sonic.computer_use.models_boss import SubMission, SubMissionResult

    judge = JudgeAgent()
    sub_mission = SubMission(
        id="sub-1",
        phase_name="Reconnaissance",
        goal="Discover network open ports using nmap scan on target host",
    )

    # Worker ran a completely unrelated command successfully (e.g. echo hello)
    result = SubMissionResult(
        sub_mission_id="sub-1",
        goal="Discover network open ports using nmap scan on target host",
        success=True,
        findings_summary="Execution succeeded",
        traces=[
            {
                "action_type": "TERMINAL_EXEC",
                "target_resource": "echo hello world",
                "exit_code": 0,
                "actual_observation": "hello world",
                "status": "COMPLETED",
            }
        ],
    )

    evaluation = await judge.evaluate_sub_mission(
        workspace_id="ws-test",
        sub_mission=sub_mission,
        result=result,
    )

    # Irrelevant command must not be CONFIRMED
    assert evaluation.verdict != JudgeVerdict.CONFIRMED
    assert evaluation.verdict == JudgeVerdict.INCONCLUSIVE


# ---------------------------------------------------------------------------
# H4: Trace synthesis milestone matching
# ---------------------------------------------------------------------------

def test_h4_patch_milestone_requires_file_write():
    milestone = MissionMilestone(
        id="m2",
        mission_id="mis-1",
        name="Patch Milestone",
        objective="Apply remediation patch to vulnerable service",
        success_conditions=["remediate vulnerability with patch"],
    )

    # GIT_COMMIT alone does NOT satisfy a patch milestone
    matches_commit_only = _milestone_matches(
        milestone=milestone,
        succ_types={"GIT_COMMIT"},
        succ_text="git commit -m 'fix'",
        conditions_text="remediate vulnerability with patch",
    )
    assert not matches_commit_only, "GIT_COMMIT alone should not satisfy patch milestone"

    # FILE_WRITE satisfies patch milestone
    matches_file_write = _milestone_matches(
        milestone=milestone,
        succ_types={"FILE_WRITE"},
        succ_text="write patch to server.py",
        conditions_text="remediate vulnerability with patch",
    )
    assert matches_file_write, "FILE_WRITE should satisfy patch milestone"


# ---------------------------------------------------------------------------
# H5: CodebaseEvolver path normalization
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("path", [
    "sonic//safety/scope.py",
    "sonic/./safety/scope.py",
    "sonic/evolution/../../sonic/safety/action_policy.py",
    "SONIC/SAFETY/SCOPED.PY",
])
def test_h5_codebase_evolver_normalizes_paths(path):
    evolver = CodebaseEvolver()
    assert evolver.is_protected(path), f"Path {path} must be detected as protected"


# ---------------------------------------------------------------------------
# H6: Autonomous evolution auto_promote defaults to False
# ---------------------------------------------------------------------------

def test_h6_evolution_auto_promote_defaults_to_false():
    sig_evolve = inspect.signature(EvolutionEngine.evolve_codebase)
    assert sig_evolve.parameters["auto_promote"].default is False

    sig_goal = inspect.signature(EvolutionEngine.execute_evolution_goal)
    assert sig_goal.parameters["auto_promote"].default is False


# ---------------------------------------------------------------------------
# H7: DockerComputerProvider does not hardcode browser names
# ---------------------------------------------------------------------------

def test_h7_no_hardcoded_browser_names_in_docker_computer():
    import inspect
    from sonic.computer.docker_computer import DockerComputerProvider

    source = inspect.getsource(DockerComputerProvider.launch_application)
    assert "grep -qiE 'chrome|chromium|electron|brave|edge'" not in source
    assert "--help 2>&1 | grep -q -- '--no-sandbox'" in source


# ---------------------------------------------------------------------------
# H8: Egress filter resolves IPv6 addresses via getaddrinfo
# ---------------------------------------------------------------------------

def test_h8_egress_dns_handles_ipv6():
    # If a domain resolves to a blocked IPv6 (e.g. ::1 or fc00::), egress must block it
    mock_addrinfo = [
        (2, 1, 6, "", ("::1", 80, 0, 0)),
    ]
    with patch("socket.getaddrinfo", return_value=mock_addrinfo):
        allowed, reason = is_target_allowed("ipv6-loopback.example")
        assert not allowed
        assert "blocked" in reason.lower()


# ---------------------------------------------------------------------------
# M1: Wildcard domain matches apex domain
# ---------------------------------------------------------------------------

def test_m1_wildcard_matches_apex():
    checker = ScopeChecker()
    scope_config = {"targets": {"domains": ["*.target.org"]}}

    assert checker.is_target_in_scope("https://api.target.org", scope_config)
    assert checker.is_target_in_scope("https://target.org", scope_config), "Wildcard *.target.org must match apex target.org"
    assert not checker.is_target_in_scope("https://evil-target.org", scope_config)


# ---------------------------------------------------------------------------
# M2: allow_private_for_tests rejected in production
# ---------------------------------------------------------------------------

def test_m2_allow_private_rejected_in_production(monkeypatch):
    monkeypatch.setenv("APP_ENV", "production")
    allowed, reason = is_target_allowed("127.0.0.1", allow_private_for_tests=True)
    assert not allowed, "allow_private_for_tests must be ignored in production"


# ---------------------------------------------------------------------------
# M3: Windows drive-letter path rejection
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("path", [
    "C:/Windows/System32/calc.exe",
    "D:\\Data\\secrets.txt",
    "c:/secret",
])
def test_m3_windows_drive_letters_rejected(path):
    policy = ActionPolicy(allowed_action_types={"FILE_READ"}, workspace_root="/home/sonic/workspace")
    verdict = policy._check_path(path, "file_read")
    assert not verdict.allowed
    assert "escapes workspace" in verdict.reason.lower()
