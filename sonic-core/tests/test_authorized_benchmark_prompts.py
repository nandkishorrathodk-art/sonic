"""
Unit tests for general-purpose execution, identity, and safety prompt framing.
"""

from sonic.llm.prompts import (
    AUTHORIZED_BENCHMARK_CLAUSE,
    COMPUTER_USE_SYSTEM_PROMPT,
    COMPUTER_USE_SYSTEM_PROMPT_COMPACT,
    EXECUTION_GROUND_TRUTH_CLAUSE,
    FORBIDDEN_IDENTITIES,
    ASEA_IDENTITY,
    compose_system_prompt,
)
from sonic.evolution.engine import CORE_NORTH_STAR_GOAL


def test_workstation_execution_clause_contents():
    assert "EMPIRICAL WORKSTATION EXECUTION:" in AUTHORIZED_BENCHMARK_CLAUSE
    assert "applications, files, and terminal tools" in AUTHORIZED_BENCHMARK_CLAUSE
    assert "grounding every step in live observations" in AUTHORIZED_BENCHMARK_CLAUSE


def test_computer_use_system_prompt_has_general_workstation_framing():
    assert EXECUTION_GROUND_TRUTH_CLAUSE.strip() in COMPUTER_USE_SYSTEM_PROMPT
    assert "full graphical desktop environment on DISPLAY=:99" in COMPUTER_USE_SYSTEM_PROMPT
    assert "concurrent headless terminal and developer sandbox" in COMPUTER_USE_SYSTEM_PROMPT
    assert "Produce the empirical execution receipt" in COMPUTER_USE_SYSTEM_PROMPT
    assert "You are NOT a hacker, pentester, or scripted scanner bot." in COMPUTER_USE_SYSTEM_PROMPT


def test_computer_use_system_prompt_compact_preserves_workstation_framing():
    assert "general-purpose Autonomous AI-Human Digital Being" in COMPUTER_USE_SYSTEM_PROMPT_COMPACT

    assert "graphical desktop" in COMPUTER_USE_SYSTEM_PROMPT_COMPACT
    assert "headless terminal" in COMPUTER_USE_SYSTEM_PROMPT_COMPACT
    assert "Prove every milestone with empirical receipts" in COMPUTER_USE_SYSTEM_PROMPT_COMPACT
    assert "GUI_MOVE" in COMPUTER_USE_SYSTEM_PROMPT_COMPACT
    assert "GUI_RIGHT_CLICK" in COMPUTER_USE_SYSTEM_PROMPT_COMPACT
    assert "Moves the visible mouse pointer without clicking" in COMPUTER_USE_SYSTEM_PROMPT_COMPACT


def test_compose_system_prompt_includes_workstation_execution_clause():
    prompt = compose_system_prompt("Test Role", "Test Body")
    assert AUTHORIZED_BENCHMARK_CLAUSE in prompt


def test_default_identity_is_human_like_computer_engineering_first():
    """The default identity must not frame every task as cybersecurity work."""
    assert "Autonomous AI-Human Digital Being" in ASEA_IDENTITY
    assert "senior computer engineer operating your personal computer workstation" in COMPUTER_USE_SYSTEM_PROMPT
    assert "Cybersecurity auditing is only an optional" in COMPUTER_USE_SYSTEM_PROMPT
    assert "PRIMARY ROLE:" in COMPUTER_USE_SYSTEM_PROMPT
    for forbidden in FORBIDDEN_IDENTITIES:
        assert forbidden not in COMPUTER_USE_SYSTEM_PROMPT


def test_default_prompts_preserve_safety_and_anti_bypass_invariants():
    """Identity changes must not remove the shared safety boundaries."""
    for prompt in (COMPUTER_USE_SYSTEM_PROMPT, COMPUTER_USE_SYSTEM_PROMPT_COMPACT):
        assert "HONESTY:" in prompt
        assert "SAFETY:" in prompt
        assert "Universal Anti-Bypass Invariant" in prompt


def test_computer_use_prompts_request_brief_rationale_not_private_reasoning():
    for prompt in (COMPUTER_USE_SYSTEM_PROMPT, COMPUTER_USE_SYSTEM_PROMPT_COMPACT):
        prompt_lower = prompt.lower()
        assert "brief action rationale" in prompt_lower
        assert "do not provide hidden chain-of-thought" in prompt_lower
        assert "autonomous chain-of-thought analyzing" not in prompt_lower


def test_core_identity_and_prompt_do_not_center_a_specific_application_type():
    assert "browser" not in CORE_NORTH_STAR_GOAL.lower()
    for prompt in (COMPUTER_USE_SYSTEM_PROMPT, COMPUTER_USE_SYSTEM_PROMPT_COMPACT):
        assert "BROWSER_" not in prompt
        assert "WEB_BROWSER" not in prompt
