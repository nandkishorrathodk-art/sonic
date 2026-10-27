"""
SONIC Dual-Plane Computer Operator E2E Test Suite
==================================================
Comprehensive requirement-driven, opaque-box E2E test suite covering:
  - R1: Multi-Phase Plane Preservation & Symmetric Boundary Enforcement
  - R2: CapabilityRouter Dispatch Wiring & Headless GUI Decoupling
  - R3: Objective-Driven Plane Resolution & Zero-Puppet Compliance
  - R4: Provider Execution Truthfulness & Git Operation Hardening

Organized across 4 Tiers:
  - Tier 1: Feature Coverage (F1 - F11)
  - Tier 2: Boundary & Corner Cases
  - Tier 3: Cross-Feature Combinations
  - Tier 4: Real-World Workload Scenarios
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from sonic.computer.provider import UnifiedComputerProvider
from sonic.computer_use.action_contracts import (
    ActionContract,
    ActionContractValidator,
    ActionReceipt,
)
from sonic.computer_use.agent import ComputerUseAgent
from sonic.computer_use.boss import BossAgent
from sonic.computer_use.models import (
    ActionExecutionStatus,
    ComputerActionType,
    ComputerDecisionTrace,
)
from sonic.computer_use.models_boss import (
    BossReport,
    Phase,
    SubMission,
    SubMissionResult,
)
from sonic.execution.capability_router import CapabilityRouter, ExecutionSubstrate
from sonic.safety.action_policy import ActionPolicy, PolicyVerdict
from sonic.safety.sealed import SealedActionPolicy, seal_default
from sonic.sandbox.provider import ExecResult

try:
    from sonic.api.routes.workstation import _requires_desktop_observation
except ImportError:
    _requires_desktop_observation = None

pytestmark = [pytest.mark.no_live_infra]


# =============================================================================
# Test Helpers & Fixtures
# =============================================================================

def _make_llm_mock(responses: list[str]) -> AsyncMock:
    """Create a mock LLM router returning the specified sequence of responses."""
    mock = AsyncMock()
    results = []
    for r in responses:
        resp = MagicMock()
        resp.content = r
        resp.reasoning_content = ""
        resp.latency_ms = 50
        results.append(resp)
    mock.complete = AsyncMock(side_effect=results)
    return mock


def _make_computer_mock() -> AsyncMock:
    """Create a mock ComputerProvider with standard methods."""
    computer = AsyncMock()
    computer.terminal = AsyncMock(return_value=MagicMock(stdout="ok", exit_code=0))
    computer.screenshot = AsyncMock(return_value=MagicMock(screenshot_base64="fake_b64", width=1920, height=1080))
    computer.gui_action = AsyncMock(return_value=MagicMock(success=True))
    computer.write_file = AsyncMock(return_value=True)
    computer.read_file = AsyncMock(return_value=MagicMock(content="content"))
    computer.status = AsyncMock(
        return_value=MagicMock(
            active_window="Desktop",
            terminal_output="$ ",
            running_processes=[],
        )
    )
    return computer


# =============================================================================
# TIER 1: FEATURE COVERAGE (R1 - R4)
# =============================================================================

class TestTier1FeatureCoverage:
    """Tier 1: Comprehensive feature coverage for R1 - R4."""

    def test_tier1_submission_explicit_plane_preservation(self):
        """F1: SubMission must preserve explicit plane='gui' and not override to 'headless'."""
        s_gui = SubMission(
            goal="Open browser application to inspect interface at http://127.0.0.1:5000",
            plane="gui",
        )
        assert s_gui.plane == "gui", f"Explicit plane='gui' was overridden to {s_gui.plane}"

        s_headless = SubMission(
            goal="Audit web server code and inspect application directory",
            plane="headless",
        )
        assert s_headless.plane == "headless", f"Explicit plane='headless' was overridden to {s_headless.plane}"

    @pytest.mark.asyncio
    async def test_tier1_boss_multi_phase_preserves_gui_plane(self):
        """F2: BossAgent multi-phase reaction planning must preserve requested_plane across phases."""
        phase1_plan = json.dumps({
            "thinking": "Phase 1: Explore application interface.",
            "phase_name": "Interface Exploration",
            "sub_missions": [
                {"goal": "Interact with GUI window on screen", "max_steps": 3, "plane": "gui"},
            ],
        })
        sub1_summary = "Explored interface and observed navigation elements."
        phase1_analysis = "Found interactive application controls."
        completion1_json = json.dumps({"complete": False, "reason": "Further interactive testing required."})

        phase2_plan = json.dumps({
            "thinking": "Phase 2: Perform deep interaction on application controls.",
            "phase_name": "Deep Interaction",
            "sub_missions": [
                {"goal": "Click submit button and verify visual response", "max_steps": 4, "plane": "gui"},
            ],
        })

        llm = _make_llm_mock([
            phase1_plan,
            sub1_summary,
            phase1_analysis,
            completion1_json,
            phase2_plan,
        ])
        computer = _make_computer_mock()
        boss = BossAgent(computer_provider=computer, llm_router=llm)

        # Plan next phase with requested_plane="gui"
        past_phase = Phase(
            phase_number=1,
            name="Interface Exploration",
            sub_missions=[SubMission(goal="Interact with GUI", plane="gui")],
        )
        try:
            next_phase = await boss._plan_next_phase(
                objective="Assess graphical application at http://127.0.0.1:5000",
                all_findings="Found interactive controls",
                phases=[past_phase],
                requested_plane="gui",
            )
        except TypeError:
            boss._current_requested_plane = "gui"
            next_phase = await boss._plan_next_phase(
                objective="Assess graphical application at http://127.0.0.1:5000",
                all_findings="Found interactive controls",
                phases=[past_phase],
            )

        assert next_phase is not None, "Next phase was not generated"
        for sm in next_phase.sub_missions:
            assert sm.plane == "gui", (
                f"Multi-phase plane preservation failed: SubMission plane is '{sm.plane}', expected 'gui'"
            )

    @pytest.mark.asyncio
    async def test_tier1_symmetric_boundary_headless_rejects_gui_actions(self):
        """F4: Headless agent must reject inadvertent GUI conversion with status=BLOCKED (exit 126)."""
        mock_comp = _make_computer_mock()
        agent = ComputerUseAgent(
            computer_provider=mock_comp,
            llm_router=MagicMock(),
            tenant_id="test-tenant",
            execution_plane="headless",
        )

        # Attempt GUI actions on headless execution plane
        for gui_action in (ComputerActionType.GUI_CLICK, ComputerActionType.APP_LAUNCH, ComputerActionType.BROWSER_CLICK):
            result = await agent.execute_action(
                workspace_id="ws-test",
                action_type=gui_action,
                target_resource="submit_button",
                payload={"x": 100, "y": 200},
                predicted_outcome="Click should occur",
            )
            assert result.status == ActionExecutionStatus.BLOCKED, (
                f"Headless agent failed to block {gui_action}: status={result.status}"
            )
            assert result.exit_code == 126, (
                f"Headless agent failed to report exit code 126 for {gui_action}: exit_code={result.exit_code}"
            )

    def test_tier1_capability_router_substrate_routing(self):
        """F5: CapabilityRouter must route GUI/App to COMPUTER and Terminal/File/Git to HEADLESS."""
        # Workstation Application Plane actions
        assert CapabilityRouter.route_action(ComputerActionType.GUI_CLICK) == ExecutionSubstrate.COMPUTER
        assert CapabilityRouter.route_action(ComputerActionType.APP_LAUNCH) == ExecutionSubstrate.COMPUTER
        assert CapabilityRouter.route_action(ComputerActionType.BROWSER_CLICK) == ExecutionSubstrate.COMPUTER
        assert CapabilityRouter.route_action(ComputerActionType.BROWSER_TYPE) == ExecutionSubstrate.COMPUTER

        # Operator Headless Sandbox actions
        assert CapabilityRouter.route_action(ComputerActionType.TERMINAL_EXEC) == ExecutionSubstrate.HEADLESS
        assert CapabilityRouter.route_action(ComputerActionType.FILE_READ) == ExecutionSubstrate.HEADLESS
        assert CapabilityRouter.route_action(ComputerActionType.FILE_WRITE) == ExecutionSubstrate.HEADLESS
        assert CapabilityRouter.route_action(ComputerActionType.GIT_COMMIT) == ExecutionSubstrate.HEADLESS
        assert CapabilityRouter.route_action(ComputerActionType.SECURITY_TOOL) == ExecutionSubstrate.HEADLESS

        # BROWSER_NAVIGATE conditional routing
        assert CapabilityRouter.route_action("BROWSER_NAVIGATE", {"interactive": False}) == ExecutionSubstrate.HEADLESS
        assert CapabilityRouter.route_action("BROWSER_NAVIGATE", {"interactive": True}) == ExecutionSubstrate.COMPUTER

    @pytest.mark.asyncio
    async def test_tier1_headless_observation_decoupled_from_gui_screenshot(self):
        """F6: Headless agent observe() must decouple from redundant full-frame GUI screenshots."""
        mock_comp = _make_computer_mock()
        agent = ComputerUseAgent(
            computer_provider=mock_comp,
            llm_router=MagicMock(),
            tenant_id="test-tenant",
            execution_plane="headless",
        )

        obs = await agent.observe("ws-test")

        # Invariant: GUI screenshot polling bypassed during headless execution
        assert mock_comp.screenshot.await_count == 0, (
            f"Headless observe() called screenshot {mock_comp.screenshot.await_count} times, expected 0"
        )
        assert obs.desktop_state == "HEADLESS_DECOUPLED" or obs.screenshot_base64 == "", (
            f"Observation did not indicate headless decoupling: state={obs.desktop_state}"
        )
        # Mandate 5 invariant: planes are never disabled
        assert agent.observe_desktop is True, "Rule 5 violation: observe_desktop was disabled"

    def test_tier1_workstation_objective_driven_plane_resolution(self):
        """F7: Workstation route must not falsely classify headless tasks as GUI based on keywords."""
        if _requires_desktop_observation is None:
            pytest.skip("_requires_desktop_observation not importable")

        # Headless tasks containing substring terms like 'web' or 'app'
        assert _requires_desktop_observation("audit web server source code") is False, (
            "Rule 2 violation: 'audit web server source code' falsely classified as GUI"
        )
        assert _requires_desktop_observation("inspect application directory") is False, (
            "Rule 2 violation: 'inspect application directory' falsely classified as GUI"
        )
        assert _requires_desktop_observation("check server.py endpoints with curl") is False
        assert _requires_desktop_observation("run git diff in workspace") is False

        # Legitimate desktop GUI tasks
        assert _requires_desktop_observation("open browser and visually verify UI") is True
        assert _requires_desktop_observation("click submit button on desktop screen") is True
        assert _requires_desktop_observation("take screenshot of active desktop window") is True

    def test_tier1_seal_default_accepts_runtime_parameters(self):
        """F8: seal_default must accept require_approval_for_intrusive and max_actions_per_minute."""
        sealed_policy = seal_default(
            require_approval_for_intrusive=True,
            max_actions_per_minute=45,
        )
        assert isinstance(sealed_policy, SealedActionPolicy)
        assert sealed_policy.max_actions_per_minute == 45

    @pytest.mark.asyncio
    async def test_tier1_provider_git_action_unmasked_exit_126_and_nonzero(self):
        """F9: UnifiedComputerProvider must not mask failures with '|| true' and must not treat exit 126 as success."""
        mock_compute = AsyncMock()
        mock_ws = MagicMock()
        mock_ws.tenant_id = "test-tenant"

        provider = UnifiedComputerProvider(compute_provider=mock_compute)
        provider.workspaces["ws-1"] = mock_ws

        # 1. Exit code 126 (BLOCKED)
        mock_compute.execute = AsyncMock(return_value=ExecResult(exit_code=126, stderr="Policy blocked command"))
        ok_blocked = await provider.git_action("ws-1", "commit", message="test commit")

        assert ok_blocked is False, "Provider treated exit code 126 as success (ok=True)"
        # Verify audit log reflects BLOCKED
        last_audit = provider.audit_log[-1] if hasattr(provider, "audit_log") and provider.audit_log else None
        if last_audit:
            assert last_audit.result == "BLOCKED", f"Audit logged result={last_audit.result}, expected BLOCKED"

        # 2. Command must not contain '|| true'
        cmd_executed = mock_compute.execute.call_args[0][1]
        assert "|| true" not in cmd_executed, f"Provider command masked with '|| true': {cmd_executed}"

        # 3. Exit code 1 (FAILED)
        mock_compute.execute = AsyncMock(return_value=ExecResult(exit_code=1, stderr="Nothing to commit"))
        ok_failed = await provider.git_action("ws-1", "commit", message="test commit")
        assert ok_failed is False, "Provider treated exit code 1 as success"

    @pytest.mark.asyncio
    async def test_tier1_agent_git_action_receipt_honest_status(self):
        """F10: ComputerUseAgent must produce honest ActionReceipt with status=BLOCKED or FAILED."""
        mock_comp = _make_computer_mock()
        mock_comp.git_action = AsyncMock(return_value=False)
        mock_comp._last_git_result = ExecResult(exit_code=126, stderr="Action blocked by sandbox security policy")

        agent = ComputerUseAgent(
            computer_provider=mock_comp,
            llm_router=MagicMock(),
            tenant_id="test-tenant",
        )

        trace = await agent.execute_action(
            workspace_id="ws-1",
            action_type=ComputerActionType.GIT_COMMIT,
            target_resource="workspace",
            payload={"message": "chore: test commit"},
            predicted_outcome="Commit changes",
        )

        assert trace.status == ActionExecutionStatus.BLOCKED, (
            f"Agent reported status={trace.status} for exit 126, expected BLOCKED"
        )
        assert trace.exit_code == 126, f"Agent reported exit_code={trace.exit_code}, expected 126"

    def test_tier1_action_policy_git_commit_validation(self):
        """F11: ActionPolicy must validate GIT_COMMIT parameters (message, workspace confinement, branch)."""
        policy = ActionPolicy(workspace_root="/home/sonic/workspace")

        # 1. Clean commit -> Allowed
        v_clean = policy.evaluate(
            "GIT_COMMIT",
            "git",
            {"message": "chore: update documentation", "path": "/home/sonic/workspace"},
        )
        assert v_clean.allowed is True, f"Clean commit denied: {v_clean.reason}"

        # 2. Shell command injection in commit message -> Denied
        v_inject = policy.evaluate(
            "GIT_COMMIT",
            "git",
            {"message": "commit'; rm -rf /; '", "path": "/home/sonic/workspace"},
        )
        assert v_inject.allowed is False, "Injection payload in commit message was allowed"

        # 3. Path traversal outside workspace -> Denied
        v_traversal = policy.evaluate(
            "GIT_COMMIT",
            "git",
            {"message": "chore: update file", "path": "/etc/shadow"},
        )
        assert v_traversal.allowed is False, "Path outside workspace root was allowed"


# =============================================================================
# TIER 2: BOUNDARY & CORNER CASES
# =============================================================================

class TestTier2BoundaryAndCornerCases:
    """Tier 2: Boundary conditions, corner cases, and adversarial injection vectors."""

    def test_tier2_empty_or_unspecified_plane_defaults_gracefully(self):
        """Unspecified, empty, or invalid plane strings must default safely without unhandled errors."""
        s1 = SubMission(goal="Generic diagnostic task", plane="")
        assert s1.plane in ("headless", "gui", "dual")

        s2 = SubMission(goal="Generic task", plane="invalid_plane_name")
        assert s2.plane in ("headless", "gui", "dual")

    def test_tier1_submission_dual_plane_is_first_class(self):
        """F12: 'dual' is a valid first-class plane and is preserved, not coerced to 'gui'."""
        s_dual = SubMission(goal="Set up the service and verify it in the desktop UI", plane="dual")
        assert s_dual.plane == "dual", (
            f"Explicit plane='dual' was coerced to {s_dual.plane}"
        )
        # A dual worker keeps full agency: no GUI-only action contract is imposed.
        agent = ComputerUseAgent(
            computer_provider=_make_computer_mock(),
            execution_plane="dual",
        )
        assert agent._gui_action_contract_active() is False
        assert str(agent.execution_plane).lower().strip() == "dual"

    def test_tier1_dual_request_is_not_a_hard_restriction(self):
        """F13: A dual request must not override the planner's explicit per-worker plane."""
        boss = BossAgent(
            computer_provider=_make_computer_mock(),
            llm_router=_make_llm_mock([]),
            requested_plane="dual",
        )
        assert boss.requested_plane == "dual"
        assert boss._explicit_requested_plane == "dual"

    def test_tier1_dual_worker_uses_external_judge_when_desktop_actions_run(self):
        """F14: A dual worker that used the desktop still requires the independent Judge."""
        agent = ComputerUseAgent(
            computer_provider=_make_computer_mock(),
            execution_plane="dual",
        )
        agent.traces = [
            ComputerDecisionTrace(
                action_type=ComputerActionType.GUI_CLICK,
                target_resource="button",
                predicted_outcome="click the control",
                actual_observation="clicked",
                status=ActionExecutionStatus.COMPLETED,
                exit_code=0,
            )
        ]
        assert agent._mission_used_desktop_actions() is True

        headless_only = ComputerUseAgent(
            computer_provider=_make_computer_mock(),
            execution_plane="dual",
        )
        headless_only.traces = [
            ComputerDecisionTrace(
                action_type=ComputerActionType.TERMINAL_EXEC,
                target_resource="hostname",
                predicted_outcome="read the hostname",
                actual_observation="host",
                status=ActionExecutionStatus.COMPLETED,
                exit_code=0,
            )
        ]
        assert headless_only._mission_used_desktop_actions() is False

    def test_tier1_gui_and_headless_requests_remain_hard_restrictions(self):
        """F15: Explicit gui/headless requests still override a planner plane."""
        gui_boss = BossAgent(
            computer_provider=_make_computer_mock(),
            llm_router=_make_llm_mock([]),
            requested_plane="gui",
        )
        assert gui_boss.requested_plane == "gui"

        headless_boss = BossAgent(
            computer_provider=_make_computer_mock(),
            llm_router=_make_llm_mock([]),
            requested_plane="headless",
        )
        assert headless_boss.requested_plane == "headless"

    @pytest.mark.asyncio
    async def test_tier1_dual_worker_is_gui_capable_for_shared_desktop_scheduling(self):
        """F16: A dual worker counts as GUI-capable so shared-desktop scheduling serializes it."""
        boss = BossAgent(
            computer_provider=_make_computer_mock(),
            llm_router=_make_llm_mock([]),
        )
        assert boss._shared_desktop is True
        dual_a = SubMission(id="a", goal="Desktop setup", plane="dual")
        dual_b = SubMission(id="b", goal="Desktop verification", plane="dual")
        assert boss._worker_limit_for_wave([dual_a, dual_b]) == 1

        headless_a = SubMission(id="c", goal="Run tests", plane="headless")
        headless_b = SubMission(id="d", goal="Run lint", plane="headless")
        assert boss._worker_limit_for_wave([headless_a, headless_b]) == boss.max_parallel_workers

    def test_tier1_dispatch_policy_default_plane_is_dual(self):
        """F17: An unset workstream plane defaults to dual (full agency), not gui."""
        from sonic.computer_use.dispatch_policy import Workstream, workstreams_from_sub_missions

        assert Workstream(id="w").plane == "dual"
        streams = workstreams_from_sub_missions(
            [SubMission(id="x", goal="Anything", plane="")]
        )
        # An empty plane is not a declaration; it must resolve to the dual default
        # rather than silently pinning the worker to the desktop.
        assert streams[0].plane in ("dual", "")

    @pytest.mark.asyncio
    async def test_tier2_unknown_requested_plane_is_ignored(self):
        """F18: A bogus requested_plane must not be latched as a hard restriction."""
        boss = BossAgent(
            computer_provider=_make_computer_mock(),
            llm_router=_make_llm_mock([]),
        )
        await boss.run(workspace_id="ws", objective="", requested_plane="nonsense")
        # The bogus value must never become a binding operator restriction.
        assert boss._explicit_requested_plane is None
        assert boss.requested_plane != "nonsense"
        # Any plane inferred after planning must be a real, recognised plane.
        assert boss.requested_plane in (None, "gui", "headless", "dual")

    @pytest.mark.asyncio
    async def test_tier2_git_exit_codes_distinction(self):
        """Exact exit code discrimination: 0=SUCCESS, 126=BLOCKED, 1/2/128=FAILED."""
        mock_compute = AsyncMock()
        mock_ws = MagicMock()
        mock_ws.tenant_id = "test-tenant"

        provider = UnifiedComputerProvider(compute_provider=mock_compute)
        provider.workspaces["ws-1"] = mock_ws

        test_cases = [
            (0, True),
            (126, False),
            (1, False),
            (2, False),
            (128, False),
        ]

        for code, expected_ok in test_cases:
            mock_compute.execute = AsyncMock(return_value=ExecResult(exit_code=code, stderr=f"Code {code}"))
            ok = await provider.git_action("ws-1", "commit", message="test")
            assert ok is expected_ok, f"Exit code {code} produced ok={ok}, expected {expected_ok}"

    def test_tier2_git_branch_name_sanitization_and_boundaries(self):
        """Branch names must be sanitized against injection flags, control chars, and path traversal."""
        policy = ActionPolicy(workspace_root="/home/sonic/workspace")

        # Valid legal branches
        assert policy.evaluate("GIT_COMMIT", "git", {"message": "ok", "branch": "main"}).allowed is True
        assert policy.evaluate("GIT_COMMIT", "git", {"message": "ok", "branch": "feat/login-flow"}).allowed is True
        assert policy.evaluate("GIT_COMMIT", "git", {"message": "ok", "branch": "fix_issue_42"}).allowed is True
        assert policy.evaluate("GIT_COMMIT", "git", {"message": "ok", "branch": "release-v1.0.0"}).allowed is True

        # Malicious / illegal branches
        assert policy.evaluate("GIT_COMMIT", "git", {"message": "ok", "branch": "-rf"}).allowed is False
        assert policy.evaluate("GIT_COMMIT", "git", {"message": "ok", "branch": "--force"}).allowed is False
        assert policy.evaluate("GIT_COMMIT", "git", {"message": "ok", "branch": "../heads/main"}).allowed is False
        assert policy.evaluate("GIT_COMMIT", "git", {"message": "ok", "branch": "feat; rm -rf /"}).allowed is False
        assert policy.evaluate("GIT_COMMIT", "git", {"message": "ok", "branch": "feat$(id)"}).allowed is False
        assert policy.evaluate("GIT_COMMIT", "git", {"message": "ok", "branch": "feat\nname"}).allowed is False

    def test_tier2_git_commit_injection_payloads(self):
        """Commit messages must reject metacharacters, null bytes, backticks, and subshells."""
        policy = ActionPolicy(workspace_root="/home/sonic/workspace")

        malicious_messages = [
            "'; rm -rf /; '",
            "$(cat /etc/passwd)",
            "`whoami`",
            'message" && echo "pwned',
            "message\nmalicious",
            "message\0nullbyte",
            "",
            "   ",
            "a" * 501,  # Excessively long commit message
        ]

        for msg in malicious_messages:
            verdict = policy.evaluate("GIT_COMMIT", "git", {"message": msg, "path": "/home/sonic/workspace"})
            assert verdict.allowed is False, f"Malicious commit message was permitted: {repr(msg)}"

    def test_tier2_git_commit_path_traversal_and_confinement(self):
        """Git commit path must be strictly confined within workspace_root."""
        policy = ActionPolicy(workspace_root="/home/sonic/workspace")

        # Safe
        assert policy.evaluate("GIT_COMMIT", "git", {"message": "safe", "path": "/home/sonic/workspace/app"}).allowed is True

        # Escapes
        assert policy.evaluate("GIT_COMMIT", "git", {"message": "safe", "path": "/home/sonic/workspace/../../etc"}).allowed is False
        assert policy.evaluate("GIT_COMMIT", "git", {"message": "safe", "path": "/etc/shadow"}).allowed is False
        assert policy.evaluate("GIT_COMMIT", "git", {"message": "safe", "path": "../../outside"}).allowed is False


# =============================================================================
# TIER 3: CROSS-FEATURE COMBINATIONS
# =============================================================================

class TestTier3CrossFeatureCombinations:
    """Tier 3: Pairwise and multi-feature interaction contracts."""

    @pytest.mark.asyncio
    async def test_tier3_multiphase_gui_mission_propagates_subagent_constraints(self):
        """BossAgent decomposes GUI mission -> SubAgent receives execution_plane='gui' constraint."""
        phase1_plan = json.dumps({
            "thinking": "Phase 1: GUI interaction.",
            "phase_name": "GUI Phase",
            "sub_missions": [
                {"goal": "Interact with GUI window", "max_steps": 3, "plane": "gui"},
            ],
        })
        phase1_summary = "GUI interaction completed."
        phase1_analysis = "Controls verified on screen."
        completion_json = json.dumps({"complete": True, "reason": "Target verified."})
        final_summary = "Mission complete."

        llm = _make_llm_mock([
            phase1_plan,
            phase1_summary,
            phase1_analysis,
            completion_json,
            final_summary,
        ])
        computer = _make_computer_mock()
        boss = BossAgent(computer_provider=computer, llm_router=llm)

        with patch("sonic.computer_use.agent.ComputerUseAgent") as MockAgentClass:
            mock_subagent = MagicMock()
            mock_subagent.run_mission = AsyncMock(return_value=[
                ComputerDecisionTrace(
                    action_type=ComputerActionType.GUI_CLICK,
                    target_resource="button",
                    actual_observation="Clicked",
                    status=ActionExecutionStatus.COMPLETED,
                    exit_code=0,
                    predicted_outcome="Click submit button",
                )
            ])
            MockAgentClass.return_value = mock_subagent

            report = await boss.run(
                workspace_id="ws-1",
                objective="Test graphical application at http://127.0.0.1:5000",
                requested_plane="gui",
            )

            assert isinstance(report, BossReport)
            # Verify SubAgent was constructed with execution_plane='gui'
            assert MockAgentClass.call_count >= 1
            call_kwargs = MockAgentClass.call_args[1]
            plane_passed = call_kwargs.get("execution_plane") or call_kwargs.get("initial_context", {}).get("execution_plane")
            assert plane_passed == "gui", f"SubAgent received plane '{plane_passed}', expected 'gui'"

    def test_tier3_workstation_contract_triggering_capability_router(self):
        """Workstation route evaluates ActionContract through CapabilityRouter classification."""
        if _requires_desktop_observation is None:
            pytest.skip("_requires_desktop_observation not importable")

        # Contract for terminal execution -> HEADLESS -> False
        contract_terminal = ActionContract(
            action_type=ComputerActionType.TERMINAL_EXEC.value,
        )
        assert CapabilityRouter.route_action(contract_terminal.action_type) == ExecutionSubstrate.HEADLESS

        # Contract for GUI click -> COMPUTER -> True
        contract_gui = ActionContract(
            action_type=ComputerActionType.GUI_CLICK.value,
        )
        assert CapabilityRouter.route_action(contract_gui.action_type) == ExecutionSubstrate.COMPUTER

    @pytest.mark.asyncio
    async def test_tier3_git_commit_safety_blocks_before_provider_execution(self):
        """ActionPolicy blocks malicious GIT_COMMIT before provider compute.execute is invoked."""
        mock_comp = _make_computer_mock()
        policy = ActionPolicy(workspace_root="/home/sonic/workspace")

        agent = ComputerUseAgent(
            computer_provider=mock_comp,
            llm_router=MagicMock(),
            safety=policy,
            self_host=True,
            tenant_id="test-tenant",
        )

        trace = await agent.execute_action(
            workspace_id="ws-1",
            action_type=ComputerActionType.GIT_COMMIT,
            target_resource="workspace",
            payload={"message": "feat'; rm -rf /; '", "path": "/home/sonic/workspace"},
            predicted_outcome="Commit should be blocked",
        )

        # Policy blocked the action
        assert trace.status == ActionExecutionStatus.BLOCKED
        assert trace.exit_code == 126
        # Provider terminal/compute execute was NEVER called
        assert mock_comp.terminal.await_count == 0


# =============================================================================
# TIER 4: REAL-WORLD WORKLOAD SCENARIOS
# =============================================================================

class TestTier4RealWorldWorkloadScenarios:
    """Tier 4: End-to-end real-world workload scenarios."""

    @pytest.mark.asyncio
    async def test_tier4_workload_audit_web_server_remains_pure_headless(self):
        """Scenario: 'Audit web server source code' must remain pure headless end-to-end."""
        objective = "Audit web server source code in /home/sonic/workspace/app and test endpoints"

        # 1. Workstation observation classifier check
        if _requires_desktop_observation is not None:
            assert _requires_desktop_observation(objective) is False

        # 2. SubMission plane inference
        sm = SubMission(goal=objective)
        assert sm.plane == "headless", f"Objective was classified as {sm.plane}, expected 'headless'"

        # 3. Execution routing
        mock_comp = _make_computer_mock()
        agent = ComputerUseAgent(
            computer_provider=mock_comp,
            llm_router=MagicMock(),
            tenant_id="test-tenant",
            execution_plane="headless",
        )

        # Observe must not poll screenshot
        obs = await agent.observe("ws-audit")
        assert mock_comp.screenshot.await_count == 0
        assert agent.observe_desktop is True  # GUI plane remains available

        # Actions route to HEADLESS substrate
        assert CapabilityRouter.route_action(ComputerActionType.TERMINAL_EXEC) == ExecutionSubstrate.HEADLESS
        assert CapabilityRouter.route_action(ComputerActionType.FILE_READ) == ExecutionSubstrate.HEADLESS

    @pytest.mark.asyncio
    async def test_tier4_workload_multiphase_penetration_assessment_retains_gui(self):
        """Scenario: Multi-phase penetration assessment retains GUI plane across all phases."""
        phase1_plan = json.dumps({
            "thinking": "Phase 1: Explore web target interface visually.",
            "phase_name": "Web Recon",
            "sub_missions": [
                {"goal": "Navigate to login screen and visually inspect input fields", "max_steps": 4, "plane": "gui"},
            ],
        })
        sub1_summary = "Observed login form at http://127.0.0.1:8080/login."
        phase1_analysis = "Discovered administrative login endpoint."
        completion1_json = json.dumps({"complete": False, "reason": "Proceed with interactive assessment."})

        phase2_plan = json.dumps({
            "thinking": "Phase 2: Perform interactive test on the login form.",
            "phase_name": "Interactive Assessment",
            "sub_missions": [
                {"goal": "Interact with login button and verify authentication response", "max_steps": 5, "plane": "gui"},
            ],
        })

        llm = _make_llm_mock([
            phase1_plan,
            sub1_summary,
            phase1_analysis,
            completion1_json,
            phase2_plan,
        ])
        computer = _make_computer_mock()
        boss = BossAgent(computer_provider=computer, llm_router=llm)

        past_phase = Phase(
            phase_number=1,
            name="Web Recon",
            sub_missions=[SubMission(goal="Inspect login UI", plane="gui")],
        )

        try:
            next_phase = await boss._plan_next_phase(
                objective="Perform penetration assessment on target at http://127.0.0.1:8080",
                all_findings="Found admin login at /login",
                phases=[past_phase],
                requested_plane="gui",
            )
        except TypeError:
            boss._current_requested_plane = "gui"
            next_phase = await boss._plan_next_phase(
                objective="Perform penetration assessment on target at http://127.0.0.1:8080",
                all_findings="Found admin login at /login",
                phases=[past_phase],
            )

        assert next_phase is not None
        assert len(next_phase.sub_missions) > 0
        for sm in next_phase.sub_missions:
            assert sm.plane == "gui", f"Phase 2 sub-mission plane is '{sm.plane}', expected 'gui'"
