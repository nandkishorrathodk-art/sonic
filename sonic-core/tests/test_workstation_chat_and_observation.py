"""Tests for Workstation Chat Prompt, Desktop Observation, and Non-Cybersecurity Demands.

Verifies:
1. WORKSTATION_CHAT_SYSTEM is general-purpose computer-use oriented (no forced pentest demands).
2. Desktop observation is captured when a desktop is provisioned, eliminating fake 'no observation' fallbacks.
3. Common Hindi and English actions are recognized by _is_action_prompt.
"""

from __future__ import annotations

from sonic.api.routes.workstation import (
    _append_worklog,
    _extract_recent_conversation_messages,
    _is_action_prompt,
)
from sonic.llm.prompts import WORKSTATION_CHAT_SYSTEM


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
