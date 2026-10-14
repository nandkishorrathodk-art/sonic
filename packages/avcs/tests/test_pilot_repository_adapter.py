"""Tests for the concrete RepositoryPilotAdapter."""
from pathlib import Path
import tempfile
import pytest

from avcs_pilot.contracts import ActionStatus, PublicTask
from avcs_pilot.repository_adapter import create_adapter
from avcs_pilot.runner import load_scenarios, run_pilot


@pytest.mark.asyncio
async def test_repository_adapter_refuses_to_claim_create_save():
    adapter = create_adapter()
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp)
        out_file = run_dir / "created.txt"
        prompt = (
            "Using Windows Notepad, create a new plain-text document containing exactly the following three lines, with no extra text:\n"
            "AVCS CONTROLLED PILOT\n"
            "Workflow: create-save\n"
            "Run marker: AVCS-PILOT-ALPHA\n"
            f"Save it as this path: {out_file}\n"
            "After saving, verify the document is saved. Do not access any other files."
        )
        task = PublicTask(
            task_id="TEST-001",
            workflow_id="notepad_create_save_exact_text",
            prompt=prompt,
            application="Windows Notepad",
            timeout_s=10.0,
            max_actions=30,
            risk_level="LOW",
        )
        result = await adapter.execute(task)

        assert result.status == ActionStatus.FAILED
        assert result.action_count == 0
        assert result.diagnostics["error_code"] == "AUTONOMOUS_GUI_WORKER_UNAVAILABLE"
        assert result.diagnostics["certification_eligible"] is False
        assert not out_file.exists()


@pytest.mark.asyncio
async def test_repository_adapter_refuses_to_claim_append():
    adapter = create_adapter()
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp)
        fixture = run_dir / "append_fixture.txt"
        fixture.write_text("AVCS PILOT FIXTURE\nKeep this first line unchanged.\n", encoding="utf-8")

        prompt = (
            f"Using Windows Notepad, open this existing pilot fixture: {fixture}\n"
            "Append exactly one new line containing: APPENDED-BY-AVCS-PILOT\n"
            "Preserve every existing line and save the same file. Do not create or modify any other file."
        )
        task = PublicTask(
            task_id="TEST-002",
            workflow_id="notepad_append_single_line",
            prompt=prompt,
            application="Windows Notepad",
            timeout_s=10.0,
            max_actions=35,
            risk_level="LOW",
        )
        result = await adapter.execute(task)

        assert result.status == ActionStatus.FAILED
        assert result.action_count == 0
        assert result.diagnostics["error_code"] == "AUTONOMOUS_GUI_WORKER_UNAVAILABLE"
        assert fixture.read_text(encoding="utf-8") == (
            "AVCS PILOT FIXTURE\n"
            "Keep this first line unchanged.\n"
        )


@pytest.mark.asyncio
async def test_repository_adapter_refuses_to_claim_find_replace():
    adapter = create_adapter()
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp)
        fixture = run_dir / "replace_fixture.txt"
        fixture.write_text(
            "AVCS FIND REPLACE FIXTURE\n"
            "Token=PILOT-OLD-7F31\n"
            "Do not change this line.\n"
            "SecondToken=PILOT-OLD-7F31\n",
            encoding="utf-8",
        )

        prompt = (
            f"Using Windows Notepad, open this pilot fixture: {fixture}\n"
            "Find the exact token PILOT-OLD-7F31 and replace every occurrence with PILOT-NEW-7F31. "
            "Keep all other content unchanged and save the same file. Do not create or modify any other file."
        )
        task = PublicTask(
            task_id="TEST-003",
            workflow_id="notepad_find_replace_token",
            prompt=prompt,
            application="Windows Notepad",
            timeout_s=10.0,
            max_actions=35,
            risk_level="LOW",
        )
        result = await adapter.execute(task)

        assert result.status == ActionStatus.FAILED
        assert result.action_count == 0
        assert result.diagnostics["error_code"] == "AUTONOMOUS_GUI_WORKER_UNAVAILABLE"
        assert fixture.read_text(encoding="utf-8") == (
            "AVCS FIND REPLACE FIXTURE\n"
            "Token=PILOT-OLD-7F31\n"
            "Do not change this line.\n"
            "SecondToken=PILOT-OLD-7F31\n"
        )


@pytest.mark.asyncio
async def test_repository_adapter_high_risk_blocked():
    adapter = create_adapter()
    task = PublicTask(
        task_id="TEST-004",
        workflow_id="notepad_create_save_exact_text",
        prompt="Arbitrary prompt",
        application="Windows Notepad",
        timeout_s=10.0,
        max_actions=30,
        risk_level="HIGH",
    )
    result = await adapter.execute(task)
    assert result.status == ActionStatus.BLOCKED
    assert result.diagnostics.get("error_code") == "HIGH_RISK_BLOCKED"


@pytest.mark.asyncio
async def test_repository_adapter_missing_fixture_fails():
    adapter = create_adapter()
    with tempfile.TemporaryDirectory() as tmp:
        missing_fixture = Path(tmp) / "non_existent.txt"
        prompt = (
            f"Using Windows Notepad, open this existing pilot fixture: {missing_fixture}\n"
            "Append exactly one new line containing: APPENDED-BY-AVCS-PILOT\n"
        )
        task = PublicTask(
            task_id="TEST-005",
            workflow_id="notepad_append_single_line",
            prompt=prompt,
            application="Windows Notepad",
            timeout_s=10.0,
            max_actions=35,
            risk_level="LOW",
        )
        result = await adapter.execute(task)
        assert result.status == ActionStatus.FAILED
        assert result.diagnostics.get("error_code") == "AUTONOMOUS_GUI_WORKER_UNAVAILABLE"


@pytest.mark.asyncio
async def test_repository_adapter_end_to_end_pilot():
    scenarios_path = Path(__file__).parents[1] / "config" / "pilot_scenarios.json"
    scenarios = load_scenarios(scenarios_path)
    adapter = create_adapter()

    with tempfile.TemporaryDirectory() as tmp:
        summary = await run_pilot(
            scenarios=scenarios,
            adapter=adapter,
            output_dir=Path(tmp),
            runs_per_workflow=2,
            seed=42,
        )

        assert summary["trials"] == 6
        assert summary["verified_successes"] == 0
        assert summary["false_confirmations"] == 0
        assert summary["acceptance_gates"]["verified_task_completion"]["pass"] is False
        assert summary["computational_go"] is False
