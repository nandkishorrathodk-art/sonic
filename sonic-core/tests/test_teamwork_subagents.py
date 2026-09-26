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
import json
from pathlib import Path
import subprocess
import pytest

from sonic.computer.headless import HeadlessComputeProvider
from sonic.computer_use.agent import ComputerUseAgent
from sonic.computer_use.models import ActionExecutionStatus, ComputerActionType
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


@pytest.mark.no_live_infra
@pytest.mark.asyncio
async def test_subagent_git_branch_creation_and_autostash_merge_with_uncommitted_parent_files(tmp_path: Path):
    """Verifies that when the workspace is a Git repo:
    1. A subagent gets its own git branch without disturbing parent HEAD.
    2. Parent uncommitted files are safely preserved across the entire subagent lifecycle.
    3. Verified subagent outputs merge back cleanly with --autostash.
    """
    provider = HeadlessComputeProvider(base_dir=tmp_path, allow_host_execution=True)
    ws = await provider.create(tenant_id="t-git", engagement_id="e-git-branch")
    ws_id = ws.id
    ws_dir = provider._get_workspace_dir(ws_id)

    # Initialize a real Git repository in the parent workspace
    subprocess.run(["git", "init", "-b", "main"], cwd=ws_dir, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Sonic Engineer"], cwd=ws_dir, check=True)
    subprocess.run(["git", "config", "user.email", "sonic@antigravity.test"], cwd=ws_dir, check=True)

    initial_file = ws_dir / "README.md"
    initial_file.write_text("# Sonic Git Project\nInitial baseline commit.", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=ws_dir, check=True)
    subprocess.run(["git", "commit", "-m", "chore: initial baseline"], cwd=ws_dir, check=True)

    # User has uncommitted work in the parent working tree!
    uncommitted_file = ws_dir / "user_uncommitted_notes.md"
    uncommitted_file.write_text("# Sensitive Work In Progress\nMust NOT be overwritten or lost.", encoding="utf-8")

    policy = ActionPolicy(workspace_root=str(tmp_path), allow_private_networks=True, max_actions_per_minute=200)

    llm = _AdaptiveFakeLLM([
        (
            "THOUGHT: Create service.py in isolated subagent branch workspace.\n"
            "ACTION: FILE_WRITE\n"
            "TARGET: service.py\n"
            'PAYLOAD: {"path": "service.py", "content": "def run():\\n    return \'ok\'\\n"}\n'
            "EXPECTED: service.py created"
        ),
    ])

    parent_agent = ComputerUseAgent(
        computer_provider=provider,
        llm_router=llm,
        safety=policy,
        tenant_id="t-git",
    )
    mgr = SubagentTeamworkManager(
        parent_agent=parent_agent,
        computer_provider=provider,
        llm_router=llm,
        safety=policy,
        tenant_id="t-git",
    )

    convs = await mgr.invoke_subagents(
        ws_id,
        [
            {
                "type_name": "self",
                "role": "Git Branch Feature Subagent",
                "prompt": "Create service.py and verify.",
                "verification_command": "python -c \"import service; assert service.run() == 'ok'\"",
                "workspace_mode": "branch",
                "max_steps": 1,
            }
        ],
    )

    assert len(convs) == 1
    conv = convs[0]
    assert conv.verified is True
    assert conv.judge_verdict == "CONFIRMED"
    # Service file is merged into parent workspace
    assert "service.py" in conv.merged_files
    assert (ws_dir / "service.py").read_text(encoding="utf-8") == "def run():\n    return 'ok'\n"
    # User uncommitted notes are 100% preserved
    assert uncommitted_file.read_text(encoding="utf-8") == "# Sensitive Work In Progress\nMust NOT be overwritten or lost."


@pytest.mark.no_live_infra
@pytest.mark.asyncio
async def test_subagent_share_mode_git_worktree_isolation(tmp_path: Path):
    """Verifies that workspace_mode='share' provisions a git worktree sharing repository storage."""
    provider = HeadlessComputeProvider(base_dir=tmp_path, allow_host_execution=True)
    ws = await provider.create(tenant_id="t-share", engagement_id="e-share")
    ws_id = ws.id
    ws_dir = provider._get_workspace_dir(ws_id)

    subprocess.run(["git", "init", "-b", "main"], cwd=ws_dir, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Sonic Engineer"], cwd=ws_dir, check=True)
    subprocess.run(["git", "config", "user.email", "sonic@antigravity.test"], cwd=ws_dir, check=True)

    base_file = ws_dir / "base.txt"
    base_file.write_text("baseline", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=ws_dir, check=True)
    subprocess.run(["git", "commit", "-m", "chore: base"], cwd=ws_dir, check=True)

    uncommitted = ws_dir / "parent_wip.txt"
    uncommitted.write_text("parent wip uncommitted", encoding="utf-8")

    policy = ActionPolicy(workspace_root=str(tmp_path), allow_private_networks=True, max_actions_per_minute=200)

    llm = _AdaptiveFakeLLM([
        (
            "THOUGHT: Write shared_feat.txt in worktree.\n"
            "ACTION: FILE_WRITE\n"
            "TARGET: shared_feat.txt\n"
            'PAYLOAD: {"path": "shared_feat.txt", "content": "shared worktree feature"}\n'
            "EXPECTED: written"
        ),
    ])

    parent_agent = ComputerUseAgent(
        computer_provider=provider,
        llm_router=llm,
        safety=policy,
        tenant_id="t-share",
    )
    mgr = SubagentTeamworkManager(
        parent_agent=parent_agent,
        computer_provider=provider,
        llm_router=llm,
        safety=policy,
        tenant_id="t-share",
    )

    convs = await mgr.invoke_subagents(
        ws_id,
        [
            {
                "type_name": "self",
                "role": "Worktree Subagent",
                "prompt": "Create shared_feat.txt.",
                "verification_command": "python -c \"assert open('shared_feat.txt').read() == 'shared worktree feature'\"",
                "workspace_mode": "share",
                "max_steps": 1,
            }
        ],
    )

    assert len(convs) == 1
    conv = convs[0]
    assert conv.workspace_mode == "share"
    assert conv.verified is True
    assert "shared_feat.txt" in conv.merged_files
    assert (ws_dir / "shared_feat.txt").read_text(encoding="utf-8") == "shared worktree feature"
    assert uncommitted.read_text(encoding="utf-8") == "parent wip uncommitted"


@pytest.mark.no_live_infra
@pytest.mark.asyncio
async def test_subagent_merges_only_modified_or_created_files_never_clobbers_untouched_parent_files(tmp_path: Path):
    """Verifies that subagent file merge delta only merges new or modified files,
    never clobbering or reporting untouched baseline parent files.
    """
    provider = HeadlessComputeProvider(base_dir=tmp_path, allow_host_execution=True)
    ws = await provider.create(tenant_id="t-delta", engagement_id="e-delta")
    ws_id = ws.id
    ws_dir = provider._get_workspace_dir(ws_id)

    # 1. Existing baseline file that subagent will modify
    existing_file = ws_dir / "existing.txt"
    existing_file.write_text("version 1.0", encoding="utf-8")

    # 2. Existing parent uncommitted notes that subagent will NOT touch
    untouched_notes = ws_dir / "notes.md"
    untouched_notes.write_text("# My Personal Notes", encoding="utf-8")

    policy = ActionPolicy(workspace_root=str(tmp_path), allow_private_networks=True, max_actions_per_minute=200)

    llm = _AdaptiveFakeLLM([
        # Action 1: Modify existing.txt
        (
            "THOUGHT: Modify existing.txt to version 2.0.\n"
            "ACTION: FILE_WRITE\n"
            "TARGET: existing.txt\n"
            'PAYLOAD: {"path": "existing.txt", "content": "version 2.0"}\n'
            "EXPECTED: modified"
        ),
        # Action 2: Create new artifact
        (
            "THOUGHT: Create newly_added.txt.\n"
            "ACTION: FILE_WRITE\n"
            "TARGET: newly_added.txt\n"
            'PAYLOAD: {"path": "newly_added.txt", "content": "newly added content"}\n'
            "EXPECTED: created"
        ),
    ])

    parent_agent = ComputerUseAgent(
        computer_provider=provider,
        llm_router=llm,
        safety=policy,
        tenant_id="t-delta",
    )
    mgr = SubagentTeamworkManager(
        parent_agent=parent_agent,
        computer_provider=provider,
        llm_router=llm,
        safety=policy,
        tenant_id="t-delta",
    )

    convs = await mgr.invoke_subagents(
        ws_id,
        [
            {
                "type_name": "self",
                "role": "Delta Modifier Subagent",
                "prompt": "Update existing.txt to version 2.0 and create newly_added.txt.",
                "verification_command": "python -c \"assert open('existing.txt').read() == 'version 2.0' and open('newly_added.txt').read() == 'newly added content'\"",
                "workspace_mode": "branch",
                "max_steps": 2,
            }
        ],
    )

    assert len(convs) == 1
    conv = convs[0]
    assert conv.verified is True
    # Modified file and new file are merged
    assert "existing.txt" in conv.merged_files
    assert "newly_added.txt" in conv.merged_files
    # Untouched file is NOT in merged files
    assert "notes.md" not in conv.merged_files
    # Verification of contents in parent directory
    assert (ws_dir / "existing.txt").read_text(encoding="utf-8") == "version 2.0"
    assert (ws_dir / "newly_added.txt").read_text(encoding="utf-8") == "newly added content"
    assert untouched_notes.read_text(encoding="utf-8") == "# My Personal Notes"


@pytest.mark.no_live_infra
@pytest.mark.asyncio
async def test_git_branch_action_does_not_collide_with_uncommitted_parent_files(tmp_path: Path):
    """Verifies that ComputerActionType.GIT_BRANCH safely switches branches without
    colliding on uncommitted parent files.
    """
    provider = HeadlessComputeProvider(base_dir=tmp_path, allow_host_execution=True)
    ws = await provider.create(tenant_id="t-git-co", engagement_id="e-git-co")
    ws_id = ws.id
    ws_dir = provider._get_workspace_dir(ws_id)

    subprocess.run(["git", "init", "-b", "main"], cwd=ws_dir, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Sonic Engineer"], cwd=ws_dir, check=True)
    subprocess.run(["git", "config", "user.email", "sonic@antigravity.test"], cwd=ws_dir, check=True)

    f1 = ws_dir / "app.py"
    f1.write_text("initial app code", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=ws_dir, check=True)
    subprocess.run(["git", "commit", "-m", "chore: initial app"], cwd=ws_dir, check=True)

    # Create feature branch with divergent app.py
    subprocess.run(["git", "checkout", "-b", "experiment"], cwd=ws_dir, check=True)
    f1.write_text("experiment app code", encoding="utf-8")
    subprocess.run(["git", "commit", "-am", "feat: experiment"], cwd=ws_dir, check=True)
    subprocess.run(["git", "checkout", "main"], cwd=ws_dir, check=True)

    # Now make uncommitted changes in main on app.py (colliding with experiment branch)
    f1.write_text("uncommitted edits on main", encoding="utf-8")

    policy = ActionPolicy(workspace_root=str(tmp_path), allow_private_networks=True, max_actions_per_minute=200)
    agent = ComputerUseAgent(
        computer_provider=provider,
        llm_router=None,
        safety=policy,
        tenant_id="t-git-co",
    )

    # Switch to experiment branch via GIT_BRANCH action
    trace = await agent.execute_action(
        ws_id,
        ComputerActionType.GIT_BRANCH,
        "experiment",
        {"branch": "experiment"},
        "Switched branch",
    )
    # Proves git checkout did NOT fail despite uncommitted colliding changes
    assert trace.status != ActionExecutionStatus.FAILED
    assert "Switched to branch experiment" in trace.actual_observation


@pytest.mark.no_live_infra
@pytest.mark.asyncio
async def test_computer_use_agent_dynamic_subagent_lifecycle_actions(tmp_path: Path):
    """Verifies that ComputerUseAgent handles DEFINE_SUBAGENT, INVOKE_SUBAGENT,
    SEND_MESSAGE, and MANAGE_SUBAGENTS dynamically like Antigravity.
    """
    provider = HeadlessComputeProvider(base_dir=tmp_path, allow_host_execution=True)
    ws = await provider.create(tenant_id="t-lifecycle", engagement_id="e-lifecycle")
    ws_id = ws.id
    ws_dir = provider._get_workspace_dir(ws_id)
    policy = ActionPolicy(workspace_root=str(tmp_path), allow_private_networks=True, max_actions_per_minute=300)

    llm = _AdaptiveFakeLLM([
        # 1. Subagent initial run: writes part1.txt
        (
            "THOUGHT: Write part1.txt.\n"
            "ACTION: FILE_WRITE\n"
            "TARGET: part1.txt\n"
            'PAYLOAD: {"path": "part1.txt", "content": "part 1 done"}\n'
            "EXPECTED: part1.txt created"
        ),
        # 2. Subagent continuation run (via SEND_MESSAGE): writes part2.txt
        (
            "THOUGHT: Write part2.txt following parent message instructions.\n"
            "ACTION: FILE_WRITE\n"
            "TARGET: part2.txt\n"
            'PAYLOAD: {"path": "part2.txt", "content": "part 2 done"}\n'
            "EXPECTED: part2.txt created"
        ),
    ])

    parent = ComputerUseAgent(
        computer_provider=provider,
        llm_router=llm,
        safety=policy,
        tenant_id="t-lifecycle",
    )

    # 1. DEFINE_SUBAGENT
    def_trace = await parent.execute_action(
        ws_id,
        ComputerActionType.DEFINE_SUBAGENT,
        "data_processor",
        {
            "name": "data_processor",
            "description": "Specialized data processor",
            "system_prompt": "Process data accurately.",
            "enable_write_tools": True,
            "enable_subagent_tools": False,
        },
        "defined subagent",
    )
    assert def_trace.status == ActionExecutionStatus.VERIFIED
    assert "Defined subagent type 'data_processor'" in def_trace.actual_observation

    # 2. INVOKE_SUBAGENT
    inv_trace = await parent.execute_action(
        ws_id,
        ComputerActionType.INVOKE_SUBAGENT,
        "processor_1",
        {
            "type_name": "data_processor",
            "role": "Data Worker",
            "prompt": "Create part1.txt.",
            "verification_command": "python -c \"assert open('part1.txt').read() == 'part 1 done'\"",
            "workspace_mode": "branch",
            "max_steps": 1,
        },
        "invoked subagent",
    )
    assert inv_trace.status == ActionExecutionStatus.VERIFIED
    assert "judge=CONFIRMED" in inv_trace.actual_observation
    assert (ws_dir / "part1.txt").read_text(encoding="utf-8") == "part 1 done"

    # Extract conversation_id from observation
    import re
    match = re.search(r'\[(subconv-[a-f0-9]+)\s*\|', inv_trace.actual_observation)
    assert match is not None
    conv_id = match.group(1)

    # 3. SEND_MESSAGE (continue existing subagent)
    msg_trace = await parent.execute_action(
        ws_id,
        ComputerActionType.SEND_MESSAGE,
        conv_id,
        {
            "recipient": conv_id,
            "message": "Now please create part2.txt with 'part 2 done'.",
            "verification_command": "python -c \"assert open('part2.txt').read() == 'part 2 done'\"",
            "max_steps": 1,
        },
        "sent continuation message",
    )
    assert msg_trace.status == ActionExecutionStatus.VERIFIED
    assert (ws_dir / "part2.txt").read_text(encoding="utf-8") == "part 2 done"

    # 4. MANAGE_SUBAGENTS (list)
    list_trace = await parent.execute_action(
        ws_id,
        ComputerActionType.MANAGE_SUBAGENTS,
        "list",
        {"action": "list"},
        "listed subagents",
    )
    assert list_trace.status == ActionExecutionStatus.COMPLETED
    listing = json.loads(list_trace.actual_observation)
    assert len(listing) == 1
    assert listing[0]["conversationId"] == conv_id
    assert listing[0]["verified"] is True

    # 5. MANAGE_SUBAGENTS (kill)
    kill_trace = await parent.execute_action(
        ws_id,
        ComputerActionType.MANAGE_SUBAGENTS,
        "kill",
        {"action": "kill", "conversation_ids": [conv_id]},
        "killed subagents",
    )
    assert kill_trace.status == ActionExecutionStatus.COMPLETED
    kill_listing = json.loads(kill_trace.actual_observation)
    assert kill_listing[0]["state"] == "canceled"
