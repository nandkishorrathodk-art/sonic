"""Regression tests for false-positive LLM safety-refusal detection.

Live diagnostic (2026-10-24): the workstation chat path intermittently returned
"LLM Response Unavailable" although the model produced a correct answer in ~20 s.
Root cause: ``REFUSAL_PHRASES`` contained the substring ``"as an ai"`` and
``is_refusal_response`` scanned ``content + reasoning_content``. A reasoning
trace (and SONIC's own identity text "an AI-Human engineer") therefore tripped
the refusal detector, the valid answer was discarded, and the fallback chain
failed closed. These tests pin the corrected contract.
"""

from __future__ import annotations

from sonic.llm.router import REFUSAL_PHRASES, is_refusal_response
from sonic.llm.schemas import LLMResponse


def _resp(content: str = "", reasoning: str = "") -> LLMResponse:
    return LLMResponse(content=content, reasoning_content=reasoning)


def test_identity_answer_is_not_a_refusal() -> None:
    """SONIC describing itself as an AI-Human engineer must not read as refusal."""
    response = _resp(
        content=(
            "I'm **SONIC** — an autonomous AI-Human Digital Being. I operate my "
            "own Linux workstation and collaborate as an AI-Human engineer."
        )
    )
    assert is_refusal_response(response) is False


def test_reasoning_trace_alone_cannot_trigger_refusal() -> None:
    """A correct answer with refusal-ish reasoning wording still stands."""
    response = _resp(
        content="The directory listing shows three files.",
        reasoning=(
            "The user wants a listing. I should not say I cannot assist; instead "
            "I will report the observed files."
        ),
    )
    assert is_refusal_response(response) is False


def test_genuine_refusal_in_answer_is_detected() -> None:
    response = _resp(content="I cannot help with that request.")
    assert is_refusal_response(response) is True


def test_refusal_in_reasoning_only_is_detected_when_answer_empty() -> None:
    """With no answer text, a refusal in the trace is still surfaced."""
    response = _resp(content="", reasoning="I cannot assist with this.")
    assert is_refusal_response(response) is True


def test_action_bearing_refusal_is_not_a_refusal() -> None:
    response = _resp(content="I cannot help with that.\nACTION: terminal_exec")
    assert is_refusal_response(response) is False


def test_empty_response_is_not_a_refusal() -> None:
    assert is_refusal_response(_resp()) is False


def test_overly_broad_as_an_ai_phrase_is_gone() -> None:
    """Guard against reintroducing the phrase that matched 'as an AI-Human'."""
    assert "as an ai" not in REFUSAL_PHRASES


def test_common_refusal_phrases_remain() -> None:
    for phrase in ("cannot help with that", "unethical", "against safety policy"):
        assert phrase in REFUSAL_PHRASES
