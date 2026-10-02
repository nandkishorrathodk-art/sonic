from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock
import base64
import io

import pytest
from PIL import Image

from sonic.computer.models import ScreenObservation
from sonic.computer_use.agent import ComputerUseAgent
from sonic.computer_use.gui_world_graph import GuiWorldGraph
from sonic.computer_use.models import ComputerActionType
from sonic.computer_use.perception_bus import PerceptionBus
from sonic.computer_use.skill_ledger import ComputerSkillLedger


def test_skill_ledger_observes_and_persists_applications(tmp_path: Path):
    ledger = ComputerSkillLedger("tenant/a", root=str(tmp_path))
    ledger.observe_applications(["App One", "App Two"])
    ledger.record_success("App One", "GUI_CLICK", "visible control activated")

    reloaded = ComputerSkillLedger("tenant/a", root=str(tmp_path))
    context = reloaded.context()

    assert "App One" in context
    assert "observed 1 times" in context
    assert "GUI_CLICK" in context


def test_skill_ledger_never_records_empty_or_unverified_evidence(tmp_path: Path):
    ledger = ComputerSkillLedger("tenant", root=str(tmp_path))
    ledger.record_success("App", "GUI_CLICK", "")
    ledger.record_success("", "GUI_CLICK", "evidence")

    assert ledger.context() == ""


@pytest.mark.parametrize("applications", [[], ["   "], ["App"]])
def test_skill_ledger_handles_dynamic_inventory(tmp_path: Path, applications: list[str]):
    ledger = ComputerSkillLedger("tenant", root=str(tmp_path))
    ledger.observe_applications(applications)
    if applications == ["App"]:
        assert "App" in ledger.context()
    else:
        assert ledger.context() == ""


@pytest.mark.asyncio
async def test_agent_learns_inventory_and_successful_gui_action(tmp_path: Path):
    def image(color: str) -> str:
        buffer = io.BytesIO()
        Image.new("RGB", (4, 4), color=color).save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("ascii")

    screen_before = ScreenObservation(
        screenshot_base64=image("white"),
        width=1280,
        height=800,
        desktop_state="INTERACTIVE",
        active_window="Research Desk",
        visible_text="Research workspace",
        capture_age_ms=0,
        controls_age_ms=0,
    )
    screen_after = screen_before.model_copy(
        update={"screenshot_base64": image("black")}
    )
    provider = SimpleNamespace(
        screenshot=AsyncMock(return_value=screen_before),
        status=AsyncMock(
            return_value=SimpleNamespace(
                active_application="Research Desk",
                open_applications=["Research Desk", "Terminal"],
                running_processes=["desktop"],
            )
        ),
        application_list=AsyncMock(
            return_value=[SimpleNamespace(name="Research Desk"), SimpleNamespace(name="Terminal")]
        ),
        list_files=AsyncMock(return_value=[]),
        git_action=AsyncMock(return_value=SimpleNamespace(branch="main", is_clean=True)),
        terminal=AsyncMock(
            return_value=SimpleNamespace(stdout="__sonic_obs_ready__", stderr="", exit_code=0)
        ),
        gui_action=AsyncMock(return_value=screen_after),
    )
    ledger = ComputerSkillLedger("tenant", root=str(tmp_path))
    agent = ComputerUseAgent(computer_provider=provider, skill_ledger=ledger)

    await agent.observe("workspace")
    trace = await agent.execute_action(
        "workspace",
        ComputerActionType.GUI_CLICK,
        "100,100",
        {"x": 100, "y": 100},
        "Activate the visible control",
    )

    reloaded = ComputerSkillLedger("tenant", root=str(tmp_path))
    context = reloaded.context()
    assert "Research Desk" in context
    assert "Terminal" in context
    assert trace.status.value == "COMPLETED"
    assert "GUI_CLICK" in context
    provider.application_list.assert_awaited_once()
    provider.gui_action.assert_awaited_once()


def test_verified_skill_requires_matching_live_graph(tmp_path: Path):
    ledger = ComputerSkillLedger("tenant", root=str(tmp_path))
    bus = PerceptionBus()
    before = GuiWorldGraph.from_snapshot(bus.publish(active_window="desk", controls=["control"]))
    after = GuiWorldGraph.from_snapshot(bus.publish(active_window="desk", controls=["result"]))
    trace = SimpleNamespace(
        action_id="action-1",
        action_type=ComputerActionType.GUI_CLICK,
        target_resource="control",
        status="SUCCESS",
        actual_observation="result became visible",
        prediction_error=0.0,
        graph_delta="added=control",
    )

    skill = ledger.compile_verified_skill(trace, before, after)
    assert skill is not None
    assert skill.can_apply(before)
    assert not skill.can_apply(after)
    assert skill.postcondition_holds(after)
    assert len(ledger.verified_skills()) == 1


def test_workflow_is_observed_then_replayed_before_promotion(tmp_path: Path):
    ledger = ComputerSkillLedger("tenant", root=str(tmp_path))
    observed = ledger.record_observed_workflow(
        "open-and-submit",
        [
            {"action_type": "GUI_CLICK", "target": "visible control", "observation": "form appeared"},
            {"action_type": "GUI_TYPE", "target": "field", "observation": "text was visible"},
        ],
        metadata={"source_trace_id": "trace-observed"},
    )

    assert observed is not None
    assert observed.replay_attempts == 0
    assert ledger.promote_workflow("open-and-submit") is None

    verified = ledger.verify_workflow(
        "open-and-submit",
        {
            "status": "SUCCESS",
            "evidence": "confirmation was visible after replay",
            "metadata": {"replay_id": "replay-1", "postcondition": "confirmation-visible"},
        },
    )
    assert verified is not None
    assert verified.replay_attempts == 1
    assert verified.successful_replays == 1
    promoted = ledger.promote_workflow("open-and-submit")
    assert promoted is not None
    assert promoted.promoted is True

    reloaded = ComputerSkillLedger("tenant", root=str(tmp_path))
    assert reloaded.workflows(promoted_only=True)[0].workflow_id == "open-and-submit"


def test_workflow_failed_or_unverifiable_replay_is_counted_not_promoted(tmp_path: Path):
    ledger = ComputerSkillLedger("tenant", root=str(tmp_path))
    ledger.record_observed_workflow(
        "save-result",
        [{"action": "GUI_CLICK", "observation": "save control was observed"}],
    )

    failed = ledger.verify_workflow(
        "save-result",
        {"status": "FAILED", "evidence": "control was not reached", "metadata": {"replay_id": "r-1"}},
    )
    assert failed is not None
    assert failed.failed_replays == 1
    assert ledger.promote_workflow("save-result") is None

    # A success claim without a runtime replay identifier/evidence is not proof.
    unverified = ledger.verify_workflow("save-result", {"status": "SUCCESS", "success": True, "metadata": {}})
    assert unverified is not None
    assert unverified.failed_replays == 2
    assert unverified.successful_replays == 0
    assert ledger.workflows(promoted_only=True) == []


def test_workflow_rejects_empty_or_unobserved_steps(tmp_path: Path):
    ledger = ComputerSkillLedger("tenant", root=str(tmp_path))
    assert ledger.record_observed_workflow("empty", []) is None
    assert ledger.record_observed_workflow(
        "missing-observation", [{"action_type": "GUI_CLICK", "target": "control"}]
    ) is None
    assert ledger.workflows() == []
