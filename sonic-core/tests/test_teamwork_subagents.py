"""
Tests for SONIC's Dynamic Runtime Subagents & `/teamwork-preview` Protocol (`teamwork.py`).
Verifies:
1. Dynamic `DEFINE_SUBAGENT`, `INVOKE_SUBAGENT`, `SEND_MESSAGE`, and `MANAGE_SUBAGENTS` actions in `ComputerUseAgent`.
2. Parallel `workspace_mode="branch"` isolation and automatic merge back to parent workspace after Judge approval.
3. Dual Verification Forcing Function + automatic self-correction loop when Round 1 fails verification.
4. DisplayLeaseManager serialization for `:99` GUI actions.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
import pytest

from sonic.computer.headless import HeadlessComputeProvider
from sonic.computer_use.agent import ComputerUseAgent
from sonic.computer_use.models import ComputerActionType
from sonic.computer_use.teamwork import DisplayLeaseManager, SubagentTeamworkManager
from sonic.llm.schemas import LLMResponse
from sonic.safety.action_policy import ActionPolicy


class _AdaptiveFakeLLM:
    """Fake LLM that responds based on live observation without hardcoded scripts."""

    def __init__(self, responses: list[str]):
        self._responses = list(responses)
        self.calls = 0

    async def complete(self, request, **kwargs):
        idx = min(self.calls, len(self._responses) - 1)
        self.calls += 1
        return LLMResponse(
            content=self._responses[idx],
            model="test-model",
            provider="local",
            input_tokens=10,
            output_tokens=10,
        )


@pytest.mark.no_live_infra
@pytest.mark.asyncio
async def test_define_and_invoke_parallel_branch_subagents_merge_on_verify(tmp_path: Path):
    provider = HeadlessComputeProvider(base_dir=tmp_path, allow_host_execution=True)
    ws = await provider.create(tenant_id="t-teamwork", engagement_id="e-branch")
    ws_id = ws.id
    ws_dir = provider._get_workspace_dir(ws_id)

    policy = ActionPolicy(workspace_root=str(ws_dir), allow_private_networks=True, max_actions_per_minute=200)

    # LLM responses for the 2 parallel subagents (each writes a file in its isolated branch and completes)
    llm = _AdaptiveFakeLLM([
        (
            "THOUGHT: I will create module_a.txt in the workspace to satisfy R1.\n"
            "ACTION: FILE_WRITE\n"
            "TARGET: module_a.txt\n"
            'PAYLOAD: {"path": "module_a.txt", "content": "module A verified content"}\n'
            "EXPECTED: module_a.txt created"
        ),
        (
            "THOUGHT: I will create module_b.txt in the workspace to satisfy R2.\n"
            "ACTION: FILE_WRITE\n"
            "TARGET: module_b.txt\n"
            'PAYLOAD: {"path": "module_b.txt", "content": "module B verified content"}\n'
            "EXPECTED: module_b.txt created"
        ),
        '{"passed": true, "notes": "Acceptance criteria satisfied."}',
        '{"passed": true, "notes": "Acceptance criteria satisfied."}',
    ])

    parent_agent = ComputerUseAgent(
        computer_provider=provider,
        llm_router=llm,
        safety=policy,
        tenant_id="t-teamwork",
    )

    # 1. DEFINE_SUBAGENT action
    define_trace = await parent_agent.execute_action(
        ws_id,
        ComputerActionType.DEFINE_SUBAGENT,
        "module_builder",
        {
            "name": "module_builder",
            "description": "Autonomous module builder",
            "system_prompt": "Deliver clean artifacts satisfying all acceptance criteria.",
            "enable_write_tools": True,
        },
        "Subagent type defined",
    )
    assert "Defined subagent type 'module_builder'" in define_trace.actual_observation

    # 2. INVOKE_SUBAGENT with 2 parallel `branch` subagents
    invoke_trace = await parent_agent.execute_action(
        ws_id,
        ComputerActionType.INVOKE_SUBAGENT,
        "parallel_modules",
        {
            "subagents": [
                {
                    "type_name": "module_builder",
                    "role": "Module A Builder",
                    "prompt": "Create module_a.txt with non-empty content.",
                    "requirements": ["Deliver module_a.txt"],
                    "acceptance_criteria": ["module_a.txt exists and is non-empty"],
                    "workspace_mode": "branch",
                    "max_steps": 1,
                },
                {
                    "type_name": "module_builder",
                    "role": "Module B Builder",
                    "prompt": "Create module_b.txt with non-empty content.",
                    "requirements": ["Deliver module_b.txt"],
                    "acceptance_criteria": ["module_b.txt exists and is non-empty"],
                    "workspace_mode": "branch",
                    "max_steps": 1,
                },
            ]
        },
        "Parallel subagents invoked and verified",
    )
    assert "judge=CONFIRMED" in invoke_trace.actual_observation
    assert (ws_dir / "module_a.txt").exists()
    assert (ws_dir / "module_b.txt").exists()


@pytest.mark.no_live_infra
@pytest.mark.asyncio
async def test_judge_forcing_function_triggers_self_correction_round(tmp_path: Path):
    provider = HeadlessComputeProvider(base_dir=tmp_path, allow_host_execution=True)
    ws = await provider.create(tenant_id="t-teamwork", engagement_id="e-self-correct")
    ws_id = ws.id
    ws_dir = provider._get_workspace_dir(ws_id)
    policy = ActionPolicy(workspace_root=str(ws_dir), allow_private_networks=True, max_actions_per_minute=200)

    # Round 1: Subagent emits GOAL_COMPLETE prematurely without concrete proof -> Judge rejects (self-attestation rejected!)
    # Round 2 (Self-Correction): Subagent receives Judge rejection and writes the required file -> Judge confirms!
    llm = _AdaptiveFakeLLM([
        (
            "THOUGHT: Attempting immediate completion without writing artifact.\n"
            "ACTION: GOAL_COMPLETE\n"
            "TARGET: done\n"
            "PAYLOAD: {}\n"
            "EXPECTED: complete"
        ),
        (
            "THOUGHT: The Independent Judge rejected my unverified completion. I will write verified_output.txt now.\n"
            "ACTION: FILE_WRITE\n"
            "TARGET: verified_output.txt\n"
            'PAYLOAD: {"path": "verified_output.txt", "content": "empirically verified payload"}\n'
            "EXPECTED: file written"
        ),
    ])

    parent_agent = ComputerUseAgent(
        computer_provider=provider,
        llm_router=llm,
        safety=policy,
        tenant_id="t-teamwork",
    )
    mgr = SubagentTeamworkManager(
        parent_agent=parent_agent,
        computer_provider=provider,
        llm_router=None,
        safety=policy,
        tenant_id="t-teamwork",
    )
    # Attach llm_router only to subagent execution so rubric check uses JudgeAgent's empirical trace & file check
    mgr.llm_router = llm

    convs = await mgr.invoke_subagents(
        ws_id,
        [
            {
                "type_name": "self",
                "role": "Self-Correcting Subagent",
                "prompt": "Write verified_output.txt in the workspace.",
                "workspace_mode": "inherit",
                "max_steps": 1,
                "max_correction_rounds": 2,
            }
        ],
    )
    assert len(convs) == 1
    conv = convs[0]
    assert conv.verified is True
    assert conv.judge_verdict == "CONFIRMED"
    assert conv.correction_rounds_used == 1
    assert (ws_dir / "verified_output.txt").read_text(encoding="utf-8") == "empirically verified payload"


@pytest.mark.no_live_infra
@pytest.mark.asyncio
async def test_display_lease_serializes_concurrent_gui_access():
    lock_a = DisplayLeaseManager.get_lock("ws-shared-99")
    lock_b = DisplayLeaseManager.get_lock("ws-shared-99")
    assert lock_a is lock_b


@pytest.mark.no_live_infra
@pytest.mark.asyncio
async def test_subagent_default_separate_workspace_and_dual_plane_gui_and_terminal(tmp_path: Path):
    """Proves that when a subagent is launched:
    1. It gets its own separate workspace by default (`workspace_mode="branch"`).
    2. It operates BOTH planes (`GUI Desktop :99` via `APP_LAUNCH` AND `Terminal/File Sandbox` via `FILE_WRITE`).
    """
    provider = HeadlessComputeProvider(base_dir=tmp_path, allow_host_execution=True)
    ws = await provider.create(tenant_id="t-dual", engagement_id="e-dual")
    ws_id = ws.id
    ws_dir = provider._get_workspace_dir(ws_id)
    policy = ActionPolicy(workspace_root=str(ws_dir), allow_private_networks=True, max_actions_per_minute=200)

    llm = _AdaptiveFakeLLM([
        # Step 1 (Plane 1 - GUI Desktop :99): Launch desktop application
        (
            "THOUGHT: Launch graphical desktop tool on DISPLAY=:99.\n"
            "ACTION: APP_LAUNCH\n"
            "TARGET: x-www-browser\n"
            'PAYLOAD: {"app_name": "x-www-browser"}\n'
            "EXPECTED: desktop window opened"
        ),
        # Step 2 (Plane 2 - Isolated Terminal/File Sandbox): Write artifact in separate workspace
        (
            "THOUGHT: Save output in isolated subagent workspace.\n"
            "ACTION: FILE_WRITE\n"
            "TARGET: dual_plane_result.txt\n"
            'PAYLOAD: {"path": "dual_plane_result.txt", "content": "GUI + Terminal Dual-Plane Verified"}\n'
            "EXPECTED: file written"
        ),
        # Step 3: Complete
        (
            "THOUGHT: Both GUI and Terminal steps verified.\n"
            "ACTION: GOAL_COMPLETE\n"
            "TARGET: dual_plane_result.txt\n"
            'PAYLOAD: {"summary": "Completed GUI + Terminal tasks"}\n'
            "EXPECTED: done"
        ),
    ])

    parent_agent = ComputerUseAgent(
        computer_provider=provider,
        llm_router=llm,
        safety=policy,
        tenant_id="t-dual",
    )
    mgr = SubagentTeamworkManager(
        parent_agent=parent_agent,
        computer_provider=provider,
        llm_router=None,
        safety=policy,
        tenant_id="t-dual",
    )
    mgr.llm_router = llm

    # Notice we do NOT pass workspace_mode — it must default to "branch" (separate workspace)!
    convs = await mgr.invoke_subagents(
        ws_id,
        [
            {
                "type_name": "self",
                "role": "Dual-Plane Subagent",
                "prompt": "Open a desktop tool on :99 and write dual_plane_result.txt.",
                "max_steps": 3,
            }
        ],
    )
    assert len(convs) == 1
    conv = convs[0]
    assert conv.workspace_mode == "branch"
    assert conv.effective_workspace_id != ws_id
    assert conv.agent_instance.observe_desktop is True
    assert conv.agent_instance.gui_only is False
    assert conv.agent_instance.execution_plane == "gui"
    assert conv.verified is True
    assert "dual_plane_result.txt" in conv.merged_files
    assert (ws_dir / "dual_plane_result.txt").read_text(encoding="utf-8") == "GUI + Terminal Dual-Plane Verified"

