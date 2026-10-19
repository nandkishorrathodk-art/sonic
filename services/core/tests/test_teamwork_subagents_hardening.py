"""
Hardening and Edge-Case Tests for SONIC's Subagent Teamwork Architecture.
Verifies:
1. INVOKE_SUBAGENT empty prompt rejection and multiline string normalization for requirements/criteria.
2. SEND_MESSAGE preservation of existing criteria and verification commands when omitted.
3. SEND_MESSAGE resolution of conversation_id from payload.
4. MANAGE_SUBAGENTS flexible resolution of target and payload for kill/kill_all.
5. DisplayLeaseManager 60-second watchdog TTL breaking stale locks.
6. ComputerUseAgent holistic goal decomposition and rubric bullet exclusion.
7. ComputerUseAgent reasoning context formatting for assigned subagents and system prompt overlay.
8. Independent Judge failing closed on inconclusive verdicts and unconfirmed acceptance criteria.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
import time
import pytest

from sonic.agents.judge import JudgeVerdict
from sonic.computer.headless import HeadlessComputeProvider
from sonic.computer_use.agent import ComputerUseAgent
from sonic.computer_use.models import ActionExecutionStatus, ComputerActionType
from sonic.computer_use.teamwork import DisplayLeaseManager, SubagentTaskSpec, SubagentTeamworkManager, _get_field
from sonic.llm.schemas import LLMRequest, LLMResponse


class _HardeningStubLLM:
    def __init__(self, response: str = '{"passed": true, "notes": "ok"}'):
        self.response = response
        self.calls = 0

    async def complete(self, request: LLMRequest, **kwargs) -> LLMResponse:
        self.calls += 1
        return LLMResponse(
            content=self.response,
            model="test-stub",
            provider="local",
            input_tokens=10,
            output_tokens=10,
        )


@pytest.mark.no_live_infra
@pytest.mark.asyncio
async def test_invoke_subagent_empty_prompt_rejected(tmp_path: Path):
    """Empty or blank prompt must be rejected with ActionExecutionStatus.FAILED."""
    provider = HeadlessComputeProvider(base_dir=tmp_path, allow_host_execution=True)
    ws = await provider.create(tenant_id="t-hardening", engagement_id="e-empty")
    agent = ComputerUseAgent(computer_provider=provider, tenant_id="t-hardening")

    # 1. Blank string prompt
    trace = await agent.execute_action(
        ws.id,
        ComputerActionType.INVOKE_SUBAGENT,
        "",
        {"prompt": "   "},
        "attempt empty prompt",
    )
    assert trace.status == ActionExecutionStatus.FAILED
    assert "Cannot invoke subagents with empty prompt" in trace.actual_observation

    # 2. Subagents array with empty prompt
    trace2 = await agent.execute_action(
        ws.id,
        ComputerActionType.INVOKE_SUBAGENT,
        "subagent_task",
        {"subagents": [{"role": "Worker", "prompt": ""}]},
        "attempt empty subagent in array",
    )
    assert trace2.status == ActionExecutionStatus.FAILED
    assert "Cannot invoke subagents with empty prompt" in trace2.actual_observation


@pytest.mark.no_live_infra
@pytest.mark.asyncio
async def test_invoke_subagent_string_requirements_and_criteria_normalization(tmp_path: Path):
    """Passing multiline or single string for requirements and criteria must NOT character-split."""
    provider = HeadlessComputeProvider(base_dir=tmp_path, allow_host_execution=True)
    ws = await provider.create(tenant_id="t-hardening", engagement_id="e-norm")
    agent = ComputerUseAgent(computer_provider=provider, tenant_id="t-hardening")

    # Manually invoke teamwork manager with string requirements and criteria
    mgr = SubagentTeamworkManager(
        parent_agent=agent,
        computer_provider=provider,
        llm_router=_HardeningStubLLM(),
        safety=None,
        tenant_id="t-hardening",
    )

    convs = await mgr.invoke_subagents(
        ws.id,
        [
            {
                "type_name": "self",
                "role": "Normalizer Worker",
                "prompt": "Test normalization.",
                "requirements": "R1: First requirement\nR2: Second requirement",
                "acceptance_criteria": "Criterion A\nCriterion B",
                "max_steps": 1,
            }
        ],
    )
    assert len(convs) == 1
    spec = convs[0].last_spec
    assert spec is not None
    assert spec.requirements == ["R1: First requirement", "R2: Second requirement"]
    assert spec.acceptance_criteria == ["Criterion A", "Criterion B"]


@pytest.mark.no_live_infra
@pytest.mark.asyncio
async def test_send_message_preserves_existing_criteria_and_command(tmp_path: Path):
    """send_message must NOT wipe out existing acceptance_criteria or verification_command when omitted."""
    provider = HeadlessComputeProvider(base_dir=tmp_path, allow_host_execution=True)
    ws = await provider.create(tenant_id="t-hardening", engagement_id="e-preserve")

    llm = _HardeningStubLLM(
        "THOUGHT: Write file.\nACTION: FILE_WRITE\nTARGET: f.txt\nPAYLOAD: {\"path\": \"f.txt\", \"content\": \"c\"}\nEXPECTED: ok"
    )
    agent = ComputerUseAgent(computer_provider=provider, llm_router=llm, tenant_id="t-hardening")
    mgr = SubagentTeamworkManager(
        parent_agent=agent,
        computer_provider=provider,
        llm_router=llm,
        safety=None,
        tenant_id="t-hardening",
    )

    convs = await mgr.invoke_subagents(
        ws.id,
        [
            {
                "type_name": "self",
                "role": "Preservation Worker",
                "prompt": "Initial setup.",
                "requirements": ["R1: setup"],
                "acceptance_criteria": ["File f.txt exists"],
                "verification_command": "python -c \"pass\"",
                "max_steps": 1,
            }
        ],
    )
    conv = convs[0]
    orig_criteria = list(conv.last_spec.acceptance_criteria)
    orig_command = conv.last_spec.verification_command

    # Now call send_message with NO criteria and NO verification command
    continued_conv = await mgr.send_message(
        conversation_id=conv.conversation_id,
        message="Follow up message without specifying criteria.",
        acceptance_criteria=None,
        verification_command="",
        max_steps=1,
    )
    assert continued_conv.last_spec.acceptance_criteria == orig_criteria
    assert continued_conv.last_spec.verification_command == orig_command

    # Also test passing empty list [] - it must NOT wipe out existing criteria either
    continued_conv2 = await mgr.send_message(
        conversation_id=conv.conversation_id,
        message="Another follow up with empty list criteria.",
        acceptance_criteria=[],
        verification_command="",
        max_steps=1,
    )
    assert continued_conv2.last_spec.acceptance_criteria == orig_criteria


@pytest.mark.no_live_infra
@pytest.mark.asyncio
async def test_manage_subagents_kill_and_kill_all(tmp_path: Path):
    """manage_subagents kill and kill_all properly mark conversations canceled and terminate tasks."""
    provider = HeadlessComputeProvider(base_dir=tmp_path, allow_host_execution=True)
    ws = await provider.create(tenant_id="t-hardening", engagement_id="e-manage")
    agent = ComputerUseAgent(computer_provider=provider, tenant_id="t-hardening")
    mgr = SubagentTeamworkManager(
        parent_agent=agent,
        computer_provider=provider,
        llm_router=_HardeningStubLLM(),
        safety=None,
        tenant_id="t-hardening",
    )

    # Launch two dummy conversations
    convs = await mgr.invoke_subagents(
        ws.id,
        [
            {"role": "Worker 1", "prompt": "Task 1", "max_steps": 1},
            {"role": "Worker 2", "prompt": "Task 2", "max_steps": 1},
        ],
    )
    c1, c2 = convs[0], convs[1]

    # Kill single subagent
    listing = mgr.manage_subagents(action="kill", conversation_ids=[c1.conversation_id])
    assert any(item["conversationId"] == c1.conversation_id and item["state"] == "canceled" for item in listing)

    # Kill all
    c2.state = "running"
    listing_all = mgr.manage_subagents(action="kill_all")
    assert any(item["conversationId"] == c2.conversation_id and item["state"] == "canceled" for item in listing_all)


@pytest.mark.no_live_infra
@pytest.mark.asyncio
async def test_display_lease_timeout_never_breaks_an_active_owner():
    """Elapsed ownership time never permits a second physical-display owner."""
    display = ":9988"
    key = f"display:{display}"
    lock = await DisplayLeaseManager.acquire(
        "ws-ttl", display=display, timeout=1.0, owner="stale-worker"
    )
    DisplayLeaseManager._lock_metadata[key]["acquired_at"] = time.time() - 70.0

    with pytest.raises(asyncio.TimeoutError):
        await DisplayLeaseManager.acquire(
            "ws-ttl",
            display=display,
            timeout=0.1,
            owner="fresh-worker",
        )
    assert lock.locked()
    assert DisplayLeaseManager._lock_metadata[key]["owner"] == "stale-worker"

    DisplayLeaseManager.release(lock, display=display)
    assert not lock.locked()
    new_lock = await DisplayLeaseManager.acquire(
        "ws-ttl", display=display, timeout=1.0, owner="fresh-worker"
    )
    assert new_lock.locked()
    assert DisplayLeaseManager._lock_metadata[key]["owner"] == "fresh-worker"
    DisplayLeaseManager.release(new_lock, display=display)


@pytest.mark.no_live_infra
@pytest.mark.asyncio
async def test_decompose_goal_holistic_and_rubric_filtering(tmp_path: Path):
    """Subagent worker pursues goal holistically and does not split rubric bullets into procedural sub-goals."""
    provider = HeadlessComputeProvider(base_dir=tmp_path, allow_host_execution=True)

    # 1. Subagent worker context (is_subagent_worker=True)
    worker = ComputerUseAgent(
        computer_provider=provider,
        is_subagent_worker=True,
        tenant_id="t-hardening",
    )
    goal_with_rubric = (
        "Implement auth module\n"
        "- **R1**: Create auth.py with verify() function\n"
        "- **R2**: Implement token validation\n"
        "- [ ] Token expires after 3600 seconds\n"
        "- [ ] Unit tests pass"
    )
    checklist = await worker.decompose_goal(goal_with_rubric)
    assert len(checklist.sub_goals) == 1
    assert "Pursue objective:" in checklist.sub_goals[0].description

    # 2. General agent with rubric bullets (not a subagent worker)
    # Even outside worker context, rubric bullets like - **R1**: and - [ ] must not be treated as sequential steps
    general_agent = ComputerUseAgent(
        computer_provider=provider,
        is_subagent_worker=False,
        initial_context={},
        tenant_id="t-hardening",
    )
    cl2 = await general_agent.decompose_goal(goal_with_rubric)
    # Sub-goals should not contain raw rubric lines
    for sg in cl2.sub_goals:
        assert not sg.description.startswith("- [ ]")
        assert not sg.description.startswith("- **R")


@pytest.mark.no_live_infra
@pytest.mark.asyncio
async def test_reasoning_prompt_subagent_role_banner_and_overlay(tmp_path: Path):
    """Reasoning prompt formats assigned subagent role banner and prepends system_prompt_overlay."""
    provider = HeadlessComputeProvider(base_dir=tmp_path, allow_host_execution=True)
    ws = await provider.create(tenant_id="t-hardening", engagement_id="e-prompt")

    agent = ComputerUseAgent(
        computer_provider=provider,
        initial_context={"subagent_role": "Security Specialist", "silent_background": True},
        system_prompt_overlay="CRITICAL SYSTEM OVERLAY INSTRUCTION",
        tenant_id="t-hardening",
    )
    obs = await agent.observe(ws.id)
    sys_prompt, user_prompt = agent._build_reasoning_context("test mission", obs, 1, "", "")

    assert sys_prompt.startswith("CRITICAL SYSTEM OVERLAY INSTRUCTION")
    assert "=== ASSIGNED SUBAGENT ROLE & WORKSPACE: Security Specialist ===" in user_prompt
    assert "=== ADVISORY INITIAL CONTEXT (UNVERIFIED) ===" not in user_prompt


@pytest.mark.no_live_infra
@pytest.mark.asyncio
async def test_independent_judge_fails_closed_on_unparseable_or_inconclusive(tmp_path: Path):
    """Judge fails closed when acceptance criteria evaluation is falsified or inconclusive."""
    provider = HeadlessComputeProvider(base_dir=tmp_path, allow_host_execution=True)

    # LLM returns unparseable rejection
    llm = _HardeningStubLLM("Unparseable output without JSON.")
    agent = ComputerUseAgent(computer_provider=provider, llm_router=llm, tenant_id="t-hardening")
    mgr = SubagentTeamworkManager(
        parent_agent=agent,
        computer_provider=provider,
        llm_router=llm,
        safety=None,
        tenant_id="t-hardening",
    )

    spec = SubagentTaskSpec(
        prompt="Deliver artifact",
        acceptance_criteria=["Criterion 1 must pass"],
        max_steps=1,
        max_correction_rounds=0,
    )
    conv = (await mgr.invoke_subagents(
        "ws-test",
        [spec],
    ))[0]

    # Verify that judge verdict is FALSIFIED (not promoted to CONFIRMED)
    assert conv.verified is False
    assert conv.judge_verdict == JudgeVerdict.FALSIFIED.value


@pytest.mark.no_live_infra
@pytest.mark.asyncio
async def test_get_field_helper_and_permissive_subagent_parsing(tmp_path: Path):
    """Verify _get_field alias/casing resolution and permissive subagent parsing in invoke_subagents."""
    # 1. Direct unit checks for _get_field
    sample = {
        "Prompt": "Do work",
        "Role": "Lead Worker",
        "TypeName": "research",
        "Workspace": "inherit",
        "Model": "flash",
        "VerificationCommand": "pytest -k unit",
        "empty_key": "",
        "fallback_target": "recovered_value",
    }
    assert _get_field(sample, "prompt") == "Do work"
    assert _get_field(sample, "PROMPT") == "Do work"
    assert _get_field(sample, "goal", "task", "prompt") == "Do work"
    assert _get_field(sample, "role", "title", "name") == "Lead Worker"
    assert _get_field(sample, "type_name", "type") == "research"
    assert _get_field(sample, "workspace_mode", "workspace") == "inherit"
    assert _get_field(sample, "model_tier", "model") == "flash"
    assert _get_field(sample, "verification_command", "verify_cmd") == "pytest -k unit"
    assert _get_field(sample, "empty_key", "fallback_target") == "recovered_value"
    assert _get_field(sample, "missing_key", default="default_val") == "default_val"
    assert _get_field({}, "key", default=42) == 42
    assert _get_field(None, "key", default=None) is None

    # 2. Permissive parsing inside invoke_subagents across varied LLM payload schemas
    provider = HeadlessComputeProvider(base_dir=tmp_path, allow_host_execution=True)
    ws = await provider.create(tenant_id="t-hardening", engagement_id="e-parse")
    agent = ComputerUseAgent(computer_provider=provider, tenant_id="t-hardening")
    mgr = SubagentTeamworkManager(
        parent_agent=agent,
        computer_provider=provider,
        llm_router=_HardeningStubLLM(),
        safety=None,
        tenant_id="t-hardening",
    )

    convs = await mgr.invoke_subagents(
        ws.id,
        [
            # Worker 1: Uppercase / PascalCase standard keys
            {
                "Prompt": "PascalCase prompt",
                "Role": "Pascal Worker",
                "TypeName": "self",
                "Workspace": "branch",
                "Model": "pro",
                "VerificationCommand": "echo 1",
                "MaxSteps": 5,
                "MaxCorrectionRounds": 1,
            },
            # Worker 2: goal alias with criteria and verify_cmd
            {
                "goal": "Goal alias prompt",
                "title": "Goal Worker",
                "type": "self",
                "workspace": "inherit",
                "model": "flash",
                "verify_cmd": "echo 2",
                "criteria": ["Goal criterion passes"],
            },
            # Worker 3: task alias with empty prompt fallback
            {
                "prompt": "   ",
                "task": "Task alias recovered prompt",
                "name": "Task Worker",
                "workspace_mode": "branch",
                "requirements": "R1: Requirement text",
            },
            # Worker 4: instruction alias
            {
                "instruction": "Instruction alias prompt",
                "role": "Instruction Worker",
            },
        ],
    )

    assert len(convs) == 4
    # Worker 1 checks
    s1 = convs[0].last_spec
    assert s1.prompt == "PascalCase prompt"
    assert s1.role == "Pascal Worker"
    assert s1.workspace_mode == "branch"
    assert s1.model_tier == "pro"
    assert s1.verification_command == "echo 1"
    assert s1.max_steps == 5

    # Worker 2 checks
    s2 = convs[1].last_spec
    assert s2.prompt == "Goal alias prompt"
    assert s2.role == "Goal Worker"
    assert s2.workspace_mode == "inherit"
    assert s2.model_tier == "flash"
    assert s2.verification_command == "echo 2"
    assert s2.acceptance_criteria == ["Goal criterion passes"]

    # Worker 3 checks
    s3 = convs[2].last_spec
    assert s3.prompt == "Task alias recovered prompt"
    assert s3.role == "Task Worker"
    assert s3.requirements == ["R1: Requirement text"]

    # Worker 4 checks
    s4 = convs[3].last_spec
    assert s4.prompt == "Instruction alias prompt"
    assert s4.role == "Instruction Worker"


@pytest.mark.no_live_infra
@pytest.mark.asyncio
async def test_transparent_subagent_error_reporting_on_launch_failure(tmp_path: Path):
    """When a subagent launch fails with an exception, an errored SubagentConversation is recorded and returned."""
    provider = HeadlessComputeProvider(base_dir=tmp_path, allow_host_execution=True)
    ws = await provider.create(tenant_id="t-hardening", engagement_id="e-err-report")
    agent = ComputerUseAgent(computer_provider=provider, tenant_id="t-hardening")
    mgr = SubagentTeamworkManager(
        parent_agent=agent,
        computer_provider=provider,
        llm_router=_HardeningStubLLM(),
        safety=None,
        tenant_id="t-hardening",
    )

    orig_prepare = mgr._prepare_subagent_workspace

    async def _failing_prepare(ws_id: str, conv_id: str, w_mode: str):
        # Fail specifically for the second worker
        if "Worker-2" in conv_id or getattr(_failing_prepare, "counter", 0) == 1:
            _failing_prepare.counter += 1
            raise RuntimeError("Simulated workspace preparation failure")
        _failing_prepare.counter += 1
        return await orig_prepare(ws_id, conv_id, w_mode)

    _failing_prepare.counter = 0
    mgr._prepare_subagent_workspace = _failing_prepare

    async def _mock_execute_loop(conv, spec, goal):
        conv.state = "completed"
        conv.state_detail = "Completed successfully"

    mgr._execute_and_verify_loop = _mock_execute_loop

    convs = await mgr.invoke_subagents(
        ws.id,
        [
            {"role": "Worker 1", "prompt": "Task 1", "max_steps": 1},
            {"role": "Worker 2 (Failing)", "prompt": "Task 2", "max_steps": 1},
            {"role": "Worker 3", "prompt": "Task 3", "max_steps": 1},
        ],
    )

    # Must return all 3 records, NOT just 2!
    assert len(convs) == 3

    # Worker 1 succeeds
    assert convs[0].role == "Worker 1"
    assert convs[0].state in ("idle", "running", "completed")

    # Worker 2 has state="errored" with transparent details
    err_conv = convs[1]
    assert err_conv.role == "Worker 2 (Failing)"
    assert err_conv.state == "errored"
    assert "Launch failed: Simulated workspace preparation failure" in err_conv.state_detail
    assert err_conv.conversation_id in mgr.conversations
    assert mgr.conversations[err_conv.conversation_id].state == "errored"

    # Worker 3 succeeds
    assert convs[2].role == "Worker 3"
    assert convs[2].state in ("idle", "running", "completed")



@pytest.mark.asyncio
@pytest.mark.parametrize("parent_change", ["modified", "deleted", "created"])
async def test_subagent_merge_preserves_concurrent_parent_changes(tmp_path, parent_change):
    from types import SimpleNamespace
    from sonic.computer_use.teamwork import SubagentConversation

    parent = tmp_path / "parent"
    isolated = tmp_path / "worker"
    parent.mkdir()
    isolated.mkdir()
    if parent_change != "created":
        (parent / "shared.txt").write_text("baseline")
        (isolated / "shared.txt").write_text("baseline")
    manager = SubagentTeamworkManager(
        parent_agent=SimpleNamespace(),
        computer_provider=SimpleNamespace(_get_workspace_dir=lambda _: parent),
        llm_router=None,
        safety=None,
    )
    manager._baseline_snapshots["worker"] = manager._snapshot_dir(isolated)
    (isolated / "first.txt").write_text("must not be partially merged")
    (isolated / "shared.txt").write_text("worker change")
    if parent_change == "deleted":
        (parent / "shared.txt").unlink()
    else:
        (parent / "shared.txt").write_text("parent change")
    conversation = SubagentConversation(
        conversation_id="worker", type_name="self", role="Worker", workspace_mode="branch",
        parent_workspace_id="parent", effective_workspace_id="worker", isolated_dir=isolated,
    )

    with pytest.raises(RuntimeError, match="conflict"):
        await manager._merge_isolated_workspace(conversation)

    assert not (parent / "first.txt").exists()
    if parent_change == "deleted":
        assert not (parent / "shared.txt").exists()
    else:
        assert (parent / "shared.txt").read_text() == "parent change"


@pytest.mark.asyncio
async def test_subagent_merge_updates_baseline_and_blocks_symlink_escape(tmp_path):
    from types import SimpleNamespace
    from sonic.computer_use.teamwork import SubagentConversation

    parent = tmp_path / "parent"
    isolated = tmp_path / "worker"
    parent.mkdir()
    isolated.mkdir()
    manager = SubagentTeamworkManager(
        parent_agent=SimpleNamespace(),
        computer_provider=SimpleNamespace(_get_workspace_dir=lambda _: parent),
        llm_router=None, safety=None,
    )
    conversation = SubagentConversation(
        conversation_id="worker", type_name="self", role="Worker", workspace_mode="branch",
        parent_workspace_id="parent", effective_workspace_id="worker", isolated_dir=isolated,
    )
    manager._baseline_snapshots["worker"] = {}
    for content in ("first change", "continued change"):
        (isolated / "shared.txt").write_text(content)
        assert await manager._merge_isolated_workspace(conversation) == ["shared.txt"]
        assert (parent / "shared.txt").read_text() == content

    external = tmp_path / "outside.txt"
    external.write_text("keep outside content")
    (parent / "shared.txt").unlink()
    (parent / "shared.txt").symlink_to(external)
    (isolated / "shared.txt").write_text("must not escape")
    with pytest.raises(RuntimeError, match="unsafe file path"):
        await manager._merge_isolated_workspace(conversation)
    assert external.read_text() == "keep outside content"
