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

from sonic.api.routes.workstation import (
    _append_worklog,
    _extract_recent_conversation_messages,
    _is_action_prompt,
    _is_observation_only_prompt,
    _get_or_create_session,
    _run_prompt_reasoning,
)
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
    ("prompt", "needs_desktop_vision"),
    [("hi sonic", False), ("describe the current screen", True)],
)
async def test_desktop_vision_is_only_used_when_prompt_needs_it(
    monkeypatch, prompt, needs_desktop_vision
):
    tenant_id = f"test-tenant-desktop-vision-{needs_desktop_vision}"
    session_id = f"desktop-vision-{needs_desktop_vision}"
    state = _get_or_create_session(tenant_id, session_id)
    state["desktop"] = {"workspace_id": "ws-conversational-vision"}
    state["worklog"] = []

    screenshot = AsyncMock(
        return_value=SimpleNamespace(
            screenshot_base64="aGVsbG8=",
            timestamp="2026-09-28T12:00:00Z",
            desktop_state="INTERACTIVE",
            width=1280,
            height=800,
            active_window="Desktop",
            visible_text="",
            detected_controls=[],
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
    assert (requests[0].task_type == "vision") is needs_desktop_vision
    assert any(message.images for message in requests[0].messages) is needs_desktop_vision
    assert screenshot.await_count == int(needs_desktop_vision)
    assert state["thought_summary"] == "Hi, I'm here and ready to help."
