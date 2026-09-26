"""
Tests for SONIC's Subagent Live Capability, Parallel Branch Isolation,
Programmatic Verification Forcing Function, Zero-Collision Merging,
and Multi-Agent Outcome Synthesis.

Verifies:
1. SONIC can launch 3 subagents concurrently with distinct roles:
   - Subagent 1: Math Service
   - Subagent 2: Text Processor
   - Subagent 3: System Reporter
2. All 3 execute concurrently in isolated branch workspaces (`workspace_mode="branch"`).
3. Each subagent fulfills its specific requirements and passes its verification command (exits 0).
4. All 3 successfully merge their code and artifacts back into the parent workspace with zero collisions.
5. BossAgent or ComputerUseAgent synthesizes the final combined outcome with verified receipts.
6. Branch isolation prevents unverified code/failures from polluting the parent workspace.
7. Dual Verification Forcing Function and Self-Correction loop resolve initial rejections.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
import pytest
from unittest.mock import MagicMock

from sonic.agents.judge import JudgeVerdict
from sonic.computer.headless import HeadlessComputeProvider
from sonic.computer_use.agent import ComputerUseAgent
from sonic.computer_use.boss import BossAgent
from sonic.computer_use.models import ActionExecutionStatus, ComputerActionType
from sonic.computer_use.models_boss import Phase, SubMission, SubMissionResult
from sonic.computer_use.teamwork import SubagentTeamworkManager, SubagentTaskSpec
from sonic.llm.schemas import LLMRequest, LLMResponse
from sonic.safety.action_policy import ActionPolicy


class _SmartLiveCapabilityLLM:
    """Smart contextual fake LLM that routes requests adaptively to subagents,
    Judge evaluations, and synthesis reasoning without race conditions.
    """

    def __init__(self):
        self.call_history: list[str] = []
        self.parent_step = 0

    async def complete(self, request: LLMRequest, **kwargs) -> LLMResponse:
        msgs = getattr(request, "messages", []) or []
        combined_text = " ".join(str(m.content or "") for m in msgs)
        self.call_history.append(combined_text)

        # 1. Independent Acceptance Criteria Judge Rubric Evaluation
        if (
            "independent acceptance criteria judge" in combined_text.lower()
            or getattr(request, "task_type", "") == "verification"
        ):
            return LLMResponse(
                content='{"passed": true, "notes": "Objective acceptance criteria and execution receipts verified."}',
                model="test-model",
                provider="local",
                input_tokens=10,
                output_tokens=10,
            )

        # 2. BossAgent Final Synthesis Report
        if "boss agent creating an empirical final report" in combined_text.lower():
            return LLMResponse(
                content=(
                    "Final Empirical Synthesis: All 3 subagents (Math Service, Text Processor, "
                    "System Reporter) successfully executed in isolated branch workspaces, "
                    "passed programmatic verification commands with exit code 0, and merged "
                    "their verified code into the parent workspace with zero collisions."
                ),
                model="test-model",
                provider="local",
                input_tokens=20,
                output_tokens=20,
            )

        # 3. BossAgent Phase Findings Aggregation / Synthesis
        if "boss agent analyzing reports from your worker agents" in combined_text.lower():
            return LLMResponse(
                content=(
                    "Phase 1 Synthesis: Math Service, Text Processor, and System Reporter have all "
                    "completed their modules and produced verified unit test receipts."
                ),
                model="test-model",
                provider="local",
                input_tokens=20,
                output_tokens=20,
            )

        # 4. Subagent 1: Math Service
        if "role mandate (math service)" in combined_text.lower() or "[role]: math service" in combined_text.lower():
            math_code = (
                "def add(a, b):\n"
                "    return a + b\n\n"
                "def multiply(a, b):\n"
                "    return a * b\n\n"
                "def factorial(n):\n"
                "    if n <= 1:\n"
                "        return 1\n"
                "    return n * factorial(n - 1)\n"
            )
            resp_content = (
                "THOUGHT: Implement MathService with add, multiply, and factorial functions.\n"
                "ACTION: FILE_WRITE\n"
                "TARGET: math_service.py\n"
                f'PAYLOAD: {{"path": "math_service.py", "content": {json.dumps(math_code)}}}\n'
                "EXPECTED: math_service.py created"
            )
            return LLMResponse(
                content=resp_content,
                model="test-model",
                provider="local",
                input_tokens=15,
                output_tokens=15,
            )

        # 5. Subagent 2: Text Processor
        if "role mandate (text processor)" in combined_text.lower() or "[role]: text processor" in combined_text.lower():
            text_code = (
                "import re\n\n"
                "def word_count(text):\n"
                "    return len(text.strip().split())\n\n"
                "def slugify(text):\n"
                "    clean = re.sub(r'[^\\w\\s-]', '', text.lower()).strip()\n"
                "    return re.sub(r'[-\\s]+', '-', clean)\n\n"
                "def reverse_words(text):\n"
                "    return ' '.join(reversed(text.strip().split()))\n"
            )
            resp_content = (
                "THOUGHT: Implement TextProcessor with word_count, slugify, and reverse_words functions.\n"
                "ACTION: FILE_WRITE\n"
                "TARGET: text_processor.py\n"
                f'PAYLOAD: {{"path": "text_processor.py", "content": {json.dumps(text_code)}}}\n'
                "EXPECTED: text_processor.py created"
            )
            return LLMResponse(
                content=resp_content,
                model="test-model",
                provider="local",
                input_tokens=15,
                output_tokens=15,
            )

        # 6. Subagent 3: System Reporter
        if "role mandate (system reporter)" in combined_text.lower() or "[role]: system reporter" in combined_text.lower():
            sys_code = (
                "import json\nimport sys\n\n"
                "def get_report():\n"
                "    return {'status': 'HEALTHY', 'subagents_active': 3, 'runtime': 'python'}\n\n"
                "def write_report_file(path='system_report.json'):\n"
                "    data = get_report()\n"
                "    with open(path, 'w', encoding='utf-8') as f:\n"
                "        json.dump(data, f, indent=2)\n"
                "    return data\n"
            )
            resp_content = (
                "THOUGHT: Implement SystemReporter that collects metrics and writes system_report.json.\n"
                "ACTION: FILE_WRITE\n"
                "TARGET: system_reporter.py\n"
                f'PAYLOAD: {{"path": "system_reporter.py", "content": {json.dumps(sys_code)}}}\n'
                "EXPECTED: system_reporter.py created"
            )
            return LLMResponse(
                content=resp_content,
                model="test-model",
                provider="local",
                input_tokens=15,
                output_tokens=15,
            )

        # 7. Parent Agent Autonomous Multi-Agent Synthesis Loop
        if self.parent_step == 0:
            self.parent_step += 1
            # Step 1: Parent invokes the 3 parallel branch subagents
            invoke_payload = {
                "subagents": [
                    {
                        "type_name": "math_service",
                        "role": "Math Service",
                        "prompt": "Implement MathService in math_service.py with add, multiply, and factorial.",
                        "requirements": ["Deliver math_service.py"],
                        "acceptance_criteria": ["math_service.py exists and functions pass tests"],
                        "verification_command": (
                            'python -c "import math_service; '
                            'assert math_service.add(3, 4) == 7; '
                            'assert math_service.multiply(3, 4) == 12; '
                            'assert math_service.factorial(5) == 120; '
                            'print(\'MATH_VERIFIED\')"'
                        ),
                        "workspace_mode": "branch",
                        "max_steps": 1,
                    },
                    {
                        "type_name": "text_processor",
                        "role": "Text Processor",
                        "prompt": "Implement TextProcessor in text_processor.py with word_count, slugify, and reverse_words.",
                        "requirements": ["Deliver text_processor.py"],
                        "acceptance_criteria": ["text_processor.py exists and functions pass tests"],
                        "verification_command": (
                            'python -c "import text_processor; '
                            'assert text_processor.word_count(\'sonic subagents active\') == 3; '
                            'assert text_processor.slugify(\'Hello World SONIC\') == \'hello-world-sonic\'; '
                            'assert text_processor.reverse_words(\'one two three\') == \'three two one\'; '
                            'print(\'TEXT_VERIFIED\')"'
                        ),
                        "workspace_mode": "branch",
                        "max_steps": 1,
                    },
                    {
                        "type_name": "system_reporter",
                        "role": "System Reporter",
                        "prompt": "Implement SystemReporter in system_reporter.py that collects metrics and writes system_report.json.",
                        "requirements": ["Deliver system_reporter.py"],
                        "acceptance_criteria": ["system_reporter.py exists and system_report.json created"],
                        "verification_command": (
                            'python -c "import system_reporter; '
                            'r = system_reporter.write_report_file(); '
                            'assert r[\'status\'] == \'HEALTHY\'; '
                            'print(\'SYSTEM_VERIFIED\')"'
                        ),
                        "workspace_mode": "branch",
                        "max_steps": 1,
                    },
                ]
            }
            return LLMResponse(
                content=(
                    "THOUGHT: Launch 3 concurrent subagents in isolated branch workspaces.\n"
                    "ACTION: INVOKE_SUBAGENT\n"
                    "TARGET: parallel_services\n"
                    f"PAYLOAD: {json.dumps(invoke_payload)}\n"
                    "EXPECTED: Subagents executed and verified"
                ),
                model="test-model",
                provider="local",
                input_tokens=25,
                output_tokens=25,
            )

        elif self.parent_step == 1:
            self.parent_step += 1
            # Step 2: Parent synthesizes by authoring integrated_pipeline.py combining all 3 services
            integration_code = (
                "import json\n"
                "import math_service\n"
                "import text_processor\n"
                "import system_reporter\n\n"
                "def run_integrated_pipeline():\n"
                "    fact = math_service.factorial(5)\n"
                "    slug = text_processor.slugify('SONIC Multi Agent Live Synthesis')\n"
                "    wc = text_processor.word_count('All three subagents merged cleanly')\n"
                "    sys_data = system_reporter.write_report_file('system_report.json')\n"
                "    combined = {\n"
                "        'pipeline_status': 'SYNTHESIS_CONFIRMED',\n"
                "        'math_result': {'factorial_5': fact},\n"
                "        'text_result': {'slug': slug, 'word_count': wc},\n"
                "        'system_status': sys_data,\n"
                "    }\n"
                "    with open('synthesis_receipt.json', 'w', encoding='utf-8') as f:\n"
                "        json.dump(combined, f, indent=2)\n"
                "    print('COMBINED_SYNTHESIS_SUCCESS')\n"
                "    return combined\n\n"
                "if __name__ == '__main__':\n"
                "    run_integrated_pipeline()\n"
            )
            return LLMResponse(
                content=(
                    "THOUGHT: All 3 subagents merged successfully. Author integrated_pipeline.py to synthesize their outcomes.\n"
                    "ACTION: FILE_WRITE\n"
                    "TARGET: integrated_pipeline.py\n"
                    f'PAYLOAD: {{"path": "integrated_pipeline.py", "content": {json.dumps(integration_code)}}}\n'
                    "EXPECTED: integrated_pipeline.py authored"
                ),
                model="test-model",
                provider="local",
                input_tokens=20,
                output_tokens=20,
            )

        elif self.parent_step == 2:
            self.parent_step += 1
            # Step 3: Run the integrated synthesis script in the parent workspace
            return LLMResponse(
                content=(
                    "THOUGHT: Execute the integrated pipeline script to generate verified synthesis receipts.\n"
                    "ACTION: TERMINAL_EXEC\n"
                    "TARGET: python integrated_pipeline.py\n"
                    'PAYLOAD: {"command": "python integrated_pipeline.py"}\n'
                    "EXPECTED: COMBINED_SYNTHESIS_SUCCESS"
                ),
                model="test-model",
                provider="local",
                input_tokens=15,
                output_tokens=15,
            )

        # Step 4: Finalize goal
        return LLMResponse(
            content=(
                "THOUGHT: Multi-agent synthesis verified with concrete receipts.\n"
                "ACTION: GOAL_COMPLETE\n"
                "TARGET: goal-complete\n"
                'PAYLOAD: {"summary": "Completed concurrent subagent execution, branch merging, and verified outcome synthesis."}\n'
                "EXPECTED: done"
            ),
            model="test-model",
            provider="local",
            input_tokens=10,
            output_tokens=10,
        )


@pytest.mark.no_live_infra
@pytest.mark.asyncio
async def test_three_concurrent_subagents_branch_isolation_and_merge(tmp_path: Path):
    """Proves:
    a) SONIC can launch 3 subagents concurrently with distinct roles (Math Service, Text Processor, System Reporter).
    b) All 3 execute concurrently in isolated branch workspaces (`workspace_mode="branch"`).
    c) Each subagent fulfills its specific requirements and passes its programmatic verification command (exit 0).
    d) All 3 successfully merge their code and artifacts back into the parent workspace with zero collisions.
    """
    provider = HeadlessComputeProvider(base_dir=tmp_path, allow_host_execution=True)
    parent_ws = await provider.create(tenant_id="t-teamwork-live", engagement_id="e-3-subagents")
    parent_ws_id = parent_ws.id
    ws_dir = provider._get_workspace_dir(parent_ws_id)

    policy = ActionPolicy(workspace_root=str(ws_dir), allow_private_networks=True, max_actions_per_minute=300)
    llm = _SmartLiveCapabilityLLM()

    parent_agent = ComputerUseAgent(
        computer_provider=provider,
        llm_router=llm,
        safety=policy,
        tenant_id="t-teamwork-live",
        enable_llm_decomposition=False,
    )

    mgr = SubagentTeamworkManager(
        parent_agent=parent_agent,
        computer_provider=provider,
        llm_router=llm,
        safety=policy,
        tenant_id="t-teamwork-live",
    )

    # Register specialized subagent types
    mgr.define_subagent(
        name="math_service",
        description="Math service implementation and algorithmic verification engineer",
        system_prompt="Deliver mathematically verified algorithms and clean modules.",
        enable_write_tools=True,
    )
    mgr.define_subagent(
        name="text_processor",
        description="Text processing and string manipulation engineer",
        system_prompt="Deliver clean string processing and text sanitization modules.",
        enable_write_tools=True,
    )
    mgr.define_subagent(
        name="system_reporter",
        description="System health inspection and reporting engineer",
        system_prompt="Collect and format runtime system diagnostics and reports.",
        enable_write_tools=True,
    )

    # 3 Concurrent Subagent Task Specifications
    specs = [
        SubagentTaskSpec(
            type_name="math_service",
            role="Math Service",
            prompt="Implement MathService in math_service.py with add, multiply, and factorial.",
            requirements=["Deliver math_service.py with add, multiply, and factorial"],
            acceptance_criteria=["math_service.py exists and functions pass tests"],
            verification_command=(
                'python -c "import math_service; '
                'assert math_service.add(3, 4) == 7; '
                'assert math_service.multiply(3, 4) == 12; '
                'assert math_service.factorial(5) == 120; '
                'print(\'MATH_VERIFIED\')"'
            ),
            workspace_mode="branch",
            max_steps=1,
            max_correction_rounds=1,
        ),
        SubagentTaskSpec(
            type_name="text_processor",
            role="Text Processor",
            prompt="Implement TextProcessor in text_processor.py with word_count, slugify, and reverse_words.",
            requirements=["Deliver text_processor.py with word_count, slugify, and reverse_words"],
            acceptance_criteria=["text_processor.py exists and functions pass tests"],
            verification_command=(
                'python -c "import text_processor; '
                'assert text_processor.word_count(\'sonic subagents active\') == 3; '
                'assert text_processor.slugify(\'Hello World SONIC\') == \'hello-world-sonic\'; '
                'assert text_processor.reverse_words(\'one two three\') == \'three two one\'; '
                'print(\'TEXT_VERIFIED\')"'
            ),
            workspace_mode="branch",
            max_steps=1,
            max_correction_rounds=1,
        ),
        SubagentTaskSpec(
            type_name="system_reporter",
            role="System Reporter",
            prompt="Implement SystemReporter in system_reporter.py that collects metrics and writes system_report.json.",
            requirements=["Deliver system_reporter.py with get_report and write_report_file"],
            acceptance_criteria=["system_reporter.py exists and system_report.json created"],
            verification_command=(
                'python -c "import system_reporter; '
                'r = system_reporter.write_report_file(); '
                'assert r[\'status\'] == \'HEALTHY\'; '
                'print(\'SYSTEM_VERIFIED\')"'
            ),
            workspace_mode="branch",
            max_steps=1,
            max_correction_rounds=1,
        ),
    ]

    # Execute all 3 concurrently in parallel isolated branches
    t_start = asyncio.get_running_loop().time()
    convs = await mgr.invoke_subagents(parent_ws_id, specs)
    elapsed = asyncio.get_running_loop().time() - t_start

    # (a) 3 subagents launched concurrently
    assert len(convs) == 3

    # (b) All 3 executed in isolated branch workspaces
    effective_ids = [c.effective_workspace_id for c in convs]
    assert len(set(effective_ids)) == 3, "Each subagent must receive a distinct isolated workspace ID"
    for c in convs:
        assert c.workspace_mode == "branch"
        assert c.effective_workspace_id != parent_ws_id, "Branch workspace must not be the parent workspace ID"
        assert c.isolated_dir is not None
        assert c.isolated_dir != ws_dir, "Isolated directory must not be the parent workspace directory"

    # (c) Each subagent fulfilled requirements and passed verification command
    for c in convs:
        assert c.verified is True, f"Subagent {c.role} must be verified (notes: {c.judge_notes})"
        assert c.judge_verdict == JudgeVerdict.CONFIRMED.value, f"Judge must confirm {c.role}"
        assert c.state == "idle"
        assert len(c.traces) >= 1

    # (d) Zero-collision merge back into parent workspace
    assert (ws_dir / "math_service.py").exists(), "math_service.py must be merged into parent workspace"
    assert (ws_dir / "text_processor.py").exists(), "text_processor.py must be merged into parent workspace"
    assert (ws_dir / "system_reporter.py").exists(), "system_reporter.py must be merged into parent workspace"
    assert (ws_dir / "system_report.json").exists(), "system_report.json must be merged into parent workspace"

    # Concrete verification: Run cross-check command in parent workspace to prove all 3 modules coexist and function
    res = await provider.execute(
        workspace_id=parent_ws_id,
        command=(
            'python -c "'
            'import math_service, text_processor, system_reporter; '
            'assert math_service.factorial(4) == 24; '
            'assert text_processor.slugify(\'SONIC Zero Collision\') == \'sonic-zero-collision\'; '
            'assert system_reporter.get_report()[\'status\'] == \'HEALTHY\'; '
            'print(\'CROSS_CHECK_ALL_3_MODULES_OK\')"'
        ),
        cwd=str(ws_dir),
    )
    assert res.exit_code == 0, f"Parent cross-check failed: {res.stderr}"
    assert "CROSS_CHECK_ALL_3_MODULES_OK" in res.stdout


@pytest.mark.no_live_infra
@pytest.mark.asyncio
async def test_computer_use_agent_end_to_end_synthesis_of_three_subagents(tmp_path: Path):
    """Proves:
    e) ComputerUseAgent executes INVOKE_SUBAGENT, coordinates 3 subagents, merges them,
       and autonomously synthesizes the combined outcome via an integrated pipeline script.
    """
    provider = HeadlessComputeProvider(base_dir=tmp_path, allow_host_execution=True)
    parent_ws = await provider.create(tenant_id="t-teamwork-synth", engagement_id="e-synth")
    parent_ws_id = parent_ws.id
    ws_dir = provider._get_workspace_dir(parent_ws_id)

    policy = ActionPolicy(workspace_root=str(ws_dir), allow_private_networks=True, max_actions_per_minute=300)
    llm = _SmartLiveCapabilityLLM()

    agent = ComputerUseAgent(
        computer_provider=provider,
        llm_router=llm,
        safety=policy,
        tenant_id="t-teamwork-synth",
        enable_llm_decomposition=False,
    )

    # Mission goal requiring concurrent delegation followed by synthesis
    goal = "Build MathService, TextProcessor, and SystemReporter via concurrent subagents, then synthesize the combined outcome."
    traces = await agent.run_mission(workspace_id=parent_ws_id, goal=goal, steps=4)

    assert len(traces) >= 3

    # Step 1 trace: INVOKE_SUBAGENT
    invoke_trace = traces[0]
    assert invoke_trace.action_type == ComputerActionType.INVOKE_SUBAGENT
    assert invoke_trace.status == ActionExecutionStatus.VERIFIED
    assert "judge=CONFIRMED" in invoke_trace.actual_observation

    # Verify all 3 subagent files exist in parent workspace
    assert (ws_dir / "math_service.py").exists()
    assert (ws_dir / "text_processor.py").exists()
    assert (ws_dir / "system_reporter.py").exists()

    # Step 2 trace: FILE_WRITE integrated_pipeline.py
    write_trace = traces[1]
    assert write_trace.action_type == ComputerActionType.FILE_WRITE
    assert (ws_dir / "integrated_pipeline.py").exists()

    # Step 3 trace: TERMINAL_EXEC python integrated_pipeline.py
    exec_trace = traces[2]
    assert exec_trace.action_type == ComputerActionType.TERMINAL_EXEC
    assert exec_trace.exit_code == 0
    assert "COMBINED_SYNTHESIS_SUCCESS" in exec_trace.actual_observation

    # Verify synthesis receipt was produced by the integrated pipeline
    synthesis_receipt_file = ws_dir / "synthesis_receipt.json"
    assert synthesis_receipt_file.exists(), "synthesis_receipt.json must be written by synthesis pipeline"
    receipt_data = json.loads(synthesis_receipt_file.read_text(encoding="utf-8"))
    assert receipt_data["pipeline_status"] == "SYNTHESIS_CONFIRMED"
    assert receipt_data["math_result"]["factorial_5"] == 120
    assert receipt_data["text_result"]["slug"] == "sonic-multi-agent-live-synthesis"
    assert receipt_data["text_result"]["word_count"] == 5
    assert receipt_data["system_status"]["status"] == "HEALTHY"


@pytest.mark.no_live_infra
@pytest.mark.asyncio
async def test_boss_agent_coordinates_three_subagents_and_synthesizes_final_report(tmp_path: Path):
    """Proves:
    e) BossAgent orchestrates subagents across distinct roles, gathers empirical trace receipts,
       and synthesizes a verified, comprehensive final report.
    """
    provider = HeadlessComputeProvider(base_dir=tmp_path, allow_host_execution=True)
    ws = await provider.create(tenant_id="t-boss-synth", engagement_id="e-boss")
    ws_id = ws.id
    ws_dir = provider._get_workspace_dir(ws_id)

    llm = _SmartLiveCapabilityLLM()
    policy = ActionPolicy(workspace_root=str(ws_dir), allow_private_networks=True, max_actions_per_minute=300)

    boss = BossAgent(
        computer_provider=provider,
        llm_router=llm,
        safety=policy,
        tenant_id="t-boss-synth",
        max_parallel_workers=4,
    )

    # Mock 3 completed worker results with concrete ground-truth traces
    trace_math = MagicMock(
        target_resource="math_service.py",
        payload='{"path": "math_service.py"}',
        status="VERIFIED",
        exit_code=0,
        actual_observation="Wrote patch (142 bytes) to math_service.py. MATH_VERIFIED.",
        verification_evidence="exit_code=0",
    )
    trace_text = MagicMock(
        target_resource="text_processor.py",
        payload='{"path": "text_processor.py"}',
        status="VERIFIED",
        exit_code=0,
        actual_observation="Wrote patch (218 bytes) to text_processor.py. TEXT_VERIFIED.",
        verification_evidence="exit_code=0",
    )
    trace_sys = MagicMock(
        target_resource="system_reporter.py",
        payload='{"path": "system_reporter.py"}',
        status="VERIFIED",
        exit_code=0,
        actual_observation="Wrote patch (190 bytes) to system_reporter.py. SYSTEM_VERIFIED.",
        verification_evidence="exit_code=0",
    )

    res_math = SubMissionResult(
        sub_mission_id="sub-math-1",
        goal="Implement MathService module and algorithms",
        role="Math Service",
        success=True,
        traces=[trace_math],
        findings_summary="MathService delivered with add, multiply, and factorial passing tests.",
        actions_taken=1,
    )
    res_text = SubMissionResult(
        sub_mission_id="sub-text-2",
        goal="Implement TextProcessor module and string utilities",
        role="Text Processor",
        success=True,
        traces=[trace_text],
        findings_summary="TextProcessor delivered with word_count, slugify, and reverse_words passing tests.",
        actions_taken=1,
    )
    res_sys = SubMissionResult(
        sub_mission_id="sub-sys-3",
        goal="Implement SystemReporter and generate health report",
        role="System Reporter",
        success=True,
        traces=[trace_sys],
        findings_summary="SystemReporter delivered with system_report.json generated.",
        actions_taken=1,
    )

    phase = Phase(
        phase_number=1,
        name="Concurrent Core Services Development",
        sub_missions=[
            SubMission(id="sub-math-1", goal=res_math.goal, role="Math Service", plane="headless"),
            SubMission(id="sub-text-2", goal=res_text.goal, role="Text Processor", plane="headless"),
            SubMission(id="sub-sys-3", goal=res_sys.goal, role="System Reporter", plane="headless"),
        ],
        results=[res_math, res_text, res_sys],
    )
    boss.phases = [phase]
    boss.all_traces = [trace_math, trace_text, trace_sys]

    # Aggregate phase findings
    phase_synthesis = await boss._aggregate_findings(
        phase=phase,
        objective="Deliver 3 concurrent core services: Math, Text, and System Reporter",
    )
    assert "Math Service" in phase_synthesis or "MathService" in phase_synthesis

    # Synthesize final report
    report = await boss._synthesize_final_report(
        objective="Deliver 3 concurrent core services: Math, Text, and System Reporter",
        t_start=0.0,
    )

    assert report.status == "COMPLETE"
    assert "Math Service" in report.findings_summary or "MathService" in report.findings_summary
    assert "Text Processor" in report.findings_summary or "TextProcessor" in report.findings_summary
    assert "System Reporter" in report.findings_summary or "SystemReporter" in report.findings_summary
    assert "[Insert" not in report.findings_summary


@pytest.mark.no_live_infra
@pytest.mark.asyncio
async def test_branch_isolation_prevents_unverified_files_from_polluting_parent(tmp_path: Path):
    """Proves:
    If a branch subagent's programmatic verification command fails,
    its artifacts remain quarantined in the isolated directory and are NEVER merged
    into the parent workspace.
    """
    provider = HeadlessComputeProvider(base_dir=tmp_path, allow_host_execution=True)
    parent_ws = await provider.create(tenant_id="t-isolation", engagement_id="e-iso")
    parent_ws_id = parent_ws.id
    ws_dir = provider._get_workspace_dir(parent_ws_id)

    policy = ActionPolicy(workspace_root=str(ws_dir), allow_private_networks=True, max_actions_per_minute=200)

    # LLM that writes a faulty or malicious draft file
    class _FailingSubagentLLM:
        async def complete(self, request, **kwargs):
            return LLMResponse(
                content=(
                    "THOUGHT: Write unverified file.\n"
                    "ACTION: FILE_WRITE\n"
                    "TARGET: unverified_draft.py\n"
                    'PAYLOAD: {"path": "unverified_draft.py", "content": "# broken implementation\\nassert False\\n"}\n'
                    "EXPECTED: file written"
                ),
                model="test-model",
                provider="local",
                input_tokens=10,
                output_tokens=10,
            )

    parent_agent = ComputerUseAgent(
        computer_provider=provider,
        llm_router=_FailingSubagentLLM(),
        safety=policy,
        tenant_id="t-isolation",
        enable_llm_decomposition=False,
    )
    mgr = SubagentTeamworkManager(
        parent_agent=parent_agent,
        computer_provider=provider,
        llm_router=_FailingSubagentLLM(),
        safety=policy,
        tenant_id="t-isolation",
    )

    # Subagent with a verification command that intentionally fails
    convs = await mgr.invoke_subagents(
        parent_ws_id,
        [
            SubagentTaskSpec(
                type_name="self",
                role="Unverified Worker",
                prompt="Write unverified_draft.py in branch workspace.",
                verification_command='python -c "assert False, \'Deliberate verification rejection\'"',
                workspace_mode="branch",
                max_steps=1,
                max_correction_rounds=0,  # No correction rounds to test direct rejection
            )
        ],
    )

    assert len(convs) == 1
    conv = convs[0]
    assert conv.verified is False
    assert conv.judge_verdict == JudgeVerdict.FALSIFIED.value
    assert conv.state == "errored"

    # Crucial assertion: Parent workspace MUST NOT contain the unverified file!
    assert not (ws_dir / "unverified_draft.py").exists(), (
        "Unverified artifacts must never be merged into the parent workspace!"
    )
    # The artifact exists only in the quarantined isolated directory
    assert conv.isolated_dir is not None
    assert (conv.isolated_dir / "unverified_draft.py").exists()


@pytest.mark.no_live_infra
@pytest.mark.asyncio
async def test_subagent_self_correction_loop_on_initial_judge_falsification(tmp_path: Path):
    """Proves:
    Dual Verification Forcing Function:
    Round 1: Worker attempts premature completion without writing artifact -> Judge falsifies.
    Round 2 (Self-Correction): Runtime feeds Judge failure evidence back to the worker via send_message.
    Worker writes the implementation -> Verification passes -> Judge confirms -> Clean merge!
    """
    provider = HeadlessComputeProvider(base_dir=tmp_path, allow_host_execution=True)
    parent_ws = await provider.create(tenant_id="t-self-correct", engagement_id="e-correct")
    parent_ws_id = parent_ws.id
    ws_dir = provider._get_workspace_dir(parent_ws_id)

    policy = ActionPolicy(workspace_root=str(ws_dir), allow_private_networks=True, max_actions_per_minute=200)

    class _SelfCorrectingLLM:
        def __init__(self):
            self.round = 0

        async def complete(self, request, **kwargs):
            # Round 1: Premature completion attempt without writing required code
            if self.round == 0:
                self.round += 1
                return LLMResponse(
                    content=(
                        "THOUGHT: Premature completion attempt without writing math_service.py.\n"
                        "ACTION: GOAL_COMPLETE\n"
                        "TARGET: done\n"
                        "PAYLOAD: {}\n"
                        "EXPECTED: complete"
                    ),
                    model="test-model",
                    provider="local",
                    input_tokens=10,
                    output_tokens=10,
                )
            # Round 2: Receives Judge rejection and writes the complete code satisfying all assertions
            math_code = (
                "def add(a, b):\n"
                "    return a + b\n\n"
                "def factorial(n):\n"
                "    return 1 if n <= 1 else n * factorial(n - 1)\n"
            )
            return LLMResponse(
                content=(
                    "THOUGHT: Judge rejected unverified completion. Authoring math_service.py now.\n"
                    "ACTION: FILE_WRITE\n"
                    "TARGET: math_service.py\n"
                    f'PAYLOAD: {{"path": "math_service.py", "content": {json.dumps(math_code)}}}\n'
                    "EXPECTED: math_service.py authored"
                ),
                model="test-model",
                provider="local",
                input_tokens=15,
                output_tokens=15,
            )

    llm = _SelfCorrectingLLM()
    parent_agent = ComputerUseAgent(
        computer_provider=provider,
        llm_router=llm,
        safety=policy,
        tenant_id="t-self-correct",
        enable_llm_decomposition=False,
    )
    mgr = SubagentTeamworkManager(
        parent_agent=parent_agent,
        computer_provider=provider,
        llm_router=None,  # Rely on programmatic verification command & trace evaluation
        safety=policy,
        tenant_id="t-self-correct",
    )
    mgr.llm_router = llm

    convs = await mgr.invoke_subagents(
        parent_ws_id,
        [
            SubagentTaskSpec(
                type_name="self",
                role="Self-Correcting Math Worker",
                prompt="Implement add and factorial in math_service.py.",
                verification_command='python -c "import math_service; assert math_service.add(2, 3) == 5; assert math_service.factorial(4) == 24; print(\'ROUND2_VERIFIED\')"',
                workspace_mode="branch",
                max_steps=1,
                max_correction_rounds=2,
            )
        ],
    )

    assert len(convs) == 1
    conv = convs[0]
    assert conv.verified is True
    assert conv.judge_verdict == JudgeVerdict.CONFIRMED.value
    assert conv.correction_rounds_used == 1, "Must have self-corrected in round 2"
    assert (ws_dir / "math_service.py").exists(), "Must be cleanly merged after round 2 verification passed"

    # Verify content in parent workspace has the corrected factorial function
    content = (ws_dir / "math_service.py").read_text(encoding="utf-8")
    assert "def factorial" in content
    assert "def add" in content
