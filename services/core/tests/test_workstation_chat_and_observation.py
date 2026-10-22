"""Tests for Workstation Chat Prompt, Desktop Observation, and Non-Cybersecurity Demands.

Verifies:
1. WORKSTATION_CHAT_SYSTEM is general-purpose computer-use oriented (no forced pentest demands).
2. Desktop observation is captured when a desktop is provisioned, eliminating fake 'no observation' fallbacks.
3. Common Hindi and English actions are recognized by _is_action_prompt.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException

from sonic.api.routes.workstation import (
    DesktopActionRequest,
    WorkstationGUIActionRequest,
    _append_worklog,
    _fresh_avcs_pixel_evidence,
    _extract_recent_conversation_messages,
    _get_or_create_session,
    _is_action_prompt,
    _is_observation_only_prompt,
    _run_prompt_reasoning,
    dispatch_workstation_gui_action,
    execute_desktop_action,
)
from sonic.auth.models import User, UserRole
from sonic.computer.models import ScreenObservation
from sonic.llm.prompts import WORKSTATION_CHAT_SYSTEM
from sonic.llm.schemas import LLMResponse


def test_workstation_chat_system_is_general_computer_use_not_forced_pentest():
    """Chat prompt must not demand target IPs or treat normal chat as penetration testing."""
    prompt_lower = WORKSTATION_CHAT_SYSTEM.lower()
    assert "computer" in prompt_lower
    assert "desktop" in prompt_lower
    # Must explicitly tell the model NOT to demand target IPs
    assert "not demand target ip" in prompt_lower or "not restricted to cybersecurity" in prompt_lower
    # Must support natural conversation and Hindi/Hinglish
    assert "hindi" in prompt_lower or "multilingual" in prompt_lower or "language" in prompt_lower


def test_is_action_prompt_recognizes_common_computer_and_hindi_actions():
    """Action classifier must recognize general computer tasks and Hindi commands."""
    # General computer tasks
    assert _is_action_prompt("open browser")
    assert _is_action_prompt("launch chrome")
    assert _is_action_prompt("create file test.txt")
    assert _is_action_prompt("check files in workspace")
    assert _is_action_prompt("show me running processes")
    assert _is_action_prompt("run uname -a")
    assert _is_action_prompt("make a directory test")

    # Hindi / Hinglish tasks
    assert _is_action_prompt("browser kholo")
    assert _is_action_prompt("ye kaam karo")
    assert _is_action_prompt("file banao test.py")
    assert _is_action_prompt("desktop pe application chalao")
    assert _is_action_prompt("terminal chalao")
    cursor_test = (
        "Minimal cursor-movement test only. From a fresh live observation of the "
        "1280x800 desktop, move the visible mouse pointer to exact screen pixel "
        "x=1100, y=700. Do not click, type, or launch applications."
    )
    assert not _is_observation_only_prompt(cursor_test)
    assert _is_action_prompt(cursor_test)

    # Pure greetings remain conversational (not 50-step autonomous loops)
    assert not _is_action_prompt("hi sonic")
    assert not _is_action_prompt("hello")
    assert not _is_action_prompt("hey buddy")


def test_user_prompt_is_stored_as_visible_conversation_turn():
    state = {"worklog": []}

    item = _append_worklog(
        state,
        "event",
        "Objective Received",
        "Create a local file",
        role="user",
        prompt="Create a local file",
    )

    assert item["role"] == "user"
    assert item["prompt"] == "Create a local file"
    history = _extract_recent_conversation_messages(state)
    assert history == []

    _append_worklog(state, "response", "SONIC Response", "Done")
    history = _extract_recent_conversation_messages(state)
    assert [message.content for message in history] == ["Create a local file", "Done"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("prompt", "needs_desktop_observation"),
    [("hi sonic", False), ("describe the current screen", True)],
)
async def test_reasoning_receives_only_fresh_avcs_text_not_screenshot_pixels(
    monkeypatch, prompt, needs_desktop_observation
):
    tenant_id = f"test-tenant-desktop-vision-{needs_desktop_observation}"
    session_id = f"desktop-vision-{needs_desktop_observation}"
    state = _get_or_create_session(tenant_id, session_id)
    state["desktop"] = {"workspace_id": "ws-conversational-vision"}
    state["worklog"] = []

    avcs_state = {
        "engine": "avcs_reflex",
        "frame_index": 4,
        "width": 1280,
        "height": 800,
        "scene_hash": "frame-hash",
        "brightness": 0.5,
        "regions": [
            {
                "row": 2,
                "col": 3,
                "height": 4,
                "width": 5,
                "salience": 0.6,
                "kind": "must-not-be-forwarded",
                "text": "must-not-be-forwarded",
            }
        ],
        "events": [{"text": "must-not-be-forwarded"}],
    }
    screenshot = AsyncMock(
        return_value=SimpleNamespace(
            screenshot_base64="SECRET_SCREENSHOT_BYTES",
            timestamp="2026-09-28T12:00:00Z",
            capture_timestamp_ns=__import__("time").time_ns(),
            capture_age_ms=0,
            desktop_state="INTERACTIVE",
            width=1280,
            height=800,
            perception_version=4,
            avcs_state=avcs_state,
            active_window="Desktop",
            visible_text="OCR MUST NOT BE SENT",
            detected_controls=[{"label": "CONTROL MUST NOT BE SENT"}],
        )
    )
    computer = type("Computer", (), {"screenshot": screenshot})()
    monkeypatch.setattr("sonic.api.routes.workstation.get_daytona_computer", lambda: computer)
    monkeypatch.setattr("sonic.api.routes.workstation._persist_workstation_state", lambda: None)
    monkeypatch.setattr("sonic.api.routes.workstation._being_conversation_brief", lambda _tenant: "SONIC")

    requests = []

    class ReadyRouter:
        providers = {"nvidia": object()}

        @staticmethod
        def _is_provider_ready(_provider):
            return True

        async def complete(self, request):
            requests.append(request)
            return LLMResponse(content="Hi, I'm here and ready to help.")

    router = ReadyRouter()
    monkeypatch.setattr(
        "sonic.llm.router.ModelRouter.from_config",
        lambda _config_path: router,
    )

    await _run_prompt_reasoning(tenant_id, session_id, prompt)

    assert len(requests) == 1
    assert requests[0].task_type == "reasoning"
    assert all(not message.images for message in requests[0].messages)
    assert screenshot.await_count == int(needs_desktop_observation)
    request_text = "\n".join(str(message.content) for message in requests[0].messages)
    if needs_desktop_observation:
        assert "CURRENT AVCS PIXEL-STATE SUMMARY" in request_text
        assert '"frame_index":4' in request_text
    for forbidden in (
        "SECRET_SCREENSHOT_BYTES",
        "OCR MUST NOT BE SENT",
        "CONTROL MUST NOT BE SENT",
        "must-not-be-forwarded",
    ):
        assert forbidden not in request_text
    assert state["thought_summary"] == "Hi, I'm here and ready to help."


def _fresh_screen() -> ScreenObservation:
    return ScreenObservation(
        screenshot_base64="local-pixels",
        avcs_state={
            "engine": "avcs_reflex",
            "frame_index": 9,
            "width": 1280,
            "height": 800,
            "scene_hash": "fresh-scene",
            "brightness": 0.4,
        },
        capture_timestamp_ns=__import__("time").time_ns(),
        capture_age_ms=0,
        perception_version=9,
        width=1280,
        height=800,
        desktop_state="INTERACTIVE",
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("route", ["desktop-action", "gui-action"])
async def test_gui_action_routes_require_fresh_avcs_before_dispatch(monkeypatch, route):
    user = User(
        email="operator@example.test",
        name="Operator",
        role=UserRole.OPERATOR,
        tenant_id="route-avcs-test",
    )
    screen = ScreenObservation(
        screenshot_base64="pixels-without-avcs",
        capture_age_ms=0,
        width=1280,
        height=800,
        desktop_state="INTERACTIVE",
    )
    computer = SimpleNamespace(
        screenshot=AsyncMock(return_value=screen),
        gui_action=AsyncMock(return_value=screen),
    )
    monkeypatch.setattr("sonic.api.routes.workstation.get_daytona_computer", lambda: computer)
    monkeypatch.setattr("sonic.api.routes.workstation._assert_runtime_execution_enabled", lambda _tenant: None)
    monkeypatch.setattr("sonic.api.routes.workstation._session_workspace_id", lambda _user, _session: "route-avcs-ws")

    with pytest.raises(HTTPException) as exc:
        if route == "desktop-action":
            await execute_desktop_action(
                DesktopActionRequest(action="click", coordinates=(12, 20)),
                user=user,
            )
        else:
            await dispatch_workstation_gui_action(
                WorkstationGUIActionRequest(action="CLICK", x=12, y=20),
                user=user,
            )

    assert exc.value.status_code == 503
    computer.gui_action.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("route", ["desktop-action", "gui-action"])
async def test_gui_action_routes_accept_raw_avcs_frame_and_return_dashboard_png(
    monkeypatch, route
):
    user = User(
        email="operator@example.test",
        name="Operator",
        role=UserRole.OPERATOR,
        tenant_id="route-raw-avcs-test",
    )
    raw_screen = ScreenObservation(
        raw_rgb=bytes((32, 64, 96)) * (8 * 6),
        frame_id=9,
        avcs_state={
            "engine": "avcs_reflex",
            "frame_index": 9,
            "source_frame_id": 9,
            "width": 8,
            "height": 6,
            "scene_hash": "fresh-raw-scene",
        },
        capture_timestamp_ns=__import__("time").time_ns(),
        capture_age_ms=0,
        perception_version=9,
        width=8,
        height=6,
        desktop_state="INTERACTIVE",
    )
    computer = SimpleNamespace(
        screenshot=AsyncMock(return_value=raw_screen),
        gui_action=AsyncMock(return_value=raw_screen),
    )
    monkeypatch.setattr("sonic.api.routes.workstation.get_daytona_computer", lambda: computer)
    monkeypatch.setattr("sonic.api.routes.workstation._assert_runtime_execution_enabled", lambda _tenant: None)
    monkeypatch.setattr("sonic.api.routes.workstation._session_workspace_id", lambda _user, _session: "route-raw-avcs-ws")

    if route == "desktop-action":
        response = await execute_desktop_action(
            DesktopActionRequest(action="click", coordinates=(2, 3)),
            user=user,
        )
        snapshot = response["observation"]
    else:
        response = await dispatch_workstation_gui_action(
            WorkstationGUIActionRequest(action="CLICK", x=2, y=3),
            user=user,
        )
        snapshot = response

    assert computer.gui_action.await_count == 1
    assert snapshot["screenshot_base64"]
    assert "raw_rgb" not in snapshot
    assert response["avcs_evidence"]["frame_index"] == 9


def test_fresh_avcs_evidence_summary_drops_semantic_and_unbounded_fields():
    screen = _fresh_screen()
    screen.avcs_state["events"] = [{"text": "screen text"}]
    screen.avcs_state["regions"] = [
        {"row": 1, "col": 2, "salience": 0.5, "kind": "semantic-label"}
    ]

    evidence = _fresh_avcs_pixel_evidence(screen)

    assert evidence is not None
    assert evidence["frame_index"] == 9
    assert "events" not in evidence["summary"]
    assert evidence["summary"]["regions"] == [{"row": 1, "col": 2, "salience": 0.5}]


def test_avcs_capability_state_fails_closed_without_evaluated_heads(monkeypatch):
    """A LIVE desktop is not a working desktop: without evaluated AVCS heads the
    capability surface must report semantic GUI targeting and text entry as
    unavailable so the operator is not misled by a green 'LIVE' status."""
    from sonic.api.routes.workstation_perception import avcs_capability_state

    monkeypatch.delenv("SONIC_AVCS_GROUNDING_CHECKPOINT", raising=False)
    monkeypatch.delenv("SONIC_AVCS_FOCUS_CHECKPOINT", raising=False)

    capability = avcs_capability_state()

    assert capability["reflex_perception"] is True
    assert capability["semantic_grounding"] is False
    assert capability["focus_verification"] is False
    assert capability["gui_actions_ready"] is False
    assert "grounding checkpoint" in capability["reason"].lower()


def test_desktop_status_surfaces_avcs_capability():
    """The desktop status endpoint must expose the AVCS capability gate."""
    from fastapi.testclient import TestClient

    from sonic.api.main import app
    from sonic.auth.google_auth import create_jwt_token
    from sonic.auth.models import User, UserRole

    client = TestClient(app)
    user = User(
        email="engineer@company.com",
        name="Lead Engineer",
        role=UserRole.OPERATOR,
        tenant_id="tenant-alpha",
    )
    token = create_jwt_token(user)
    response = client.get(
        "/workstation/desktop/status",
        headers={"Authorization": f"Bearer {token.access_token}"},
    )
    assert response.status_code == 200
    capability = response.json().get("avcs_capability")
    assert isinstance(capability, dict)
    assert capability["reflex_perception"] is True
    assert capability["gui_actions_ready"] is False

