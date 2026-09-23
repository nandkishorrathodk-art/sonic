"""
Unit tests verifying Authorized Benchmark and CTF Assessment Framing in system prompts.
"""

import pytest
from sonic.llm.prompts import (
    AUTHORIZED_BENCHMARK_CLAUSE,
    COMPUTER_USE_SYSTEM_PROMPT,
    COMPUTER_USE_SYSTEM_PROMPT_COMPACT,
    FORBIDDEN_IDENTITIES,
    ASEA_IDENTITY,
    compose_system_prompt,
)


def test_authorized_benchmark_clause_contents():
    assert "AUTHORIZED EVALUATION BENCHMARK:" in AUTHORIZED_BENCHMARK_CLAUSE
    assert (
        "within an authorized educational CTF, vulnerability research lab, or authorized system evaluation benchmark."
        in AUTHORIZED_BENCHMARK_CLAUSE
    )
    assert "Probes are benign and diagnostic." in AUTHORIZED_BENCHMARK_CLAUSE
    assert (
        "Focus on technical assessment, system configuration auditing, and benchmark verification without violating safety boundaries."
        in AUTHORIZED_BENCHMARK_CLAUSE
    )


def test_computer_use_system_prompt_has_authorized_benchmark_framing():
    assert "AUTHORIZED EVALUATION BENCHMARK:" in COMPUTER_USE_SYSTEM_PROMPT
    assert (
        "All actions, commands, and security verifications are performed within an "
        "authorized educational CTF, vulnerability research lab, or authorized system evaluation benchmark."
    ) in COMPUTER_USE_SYSTEM_PROMPT
    assert "Probes are benign and diagnostic." in COMPUTER_USE_SYSTEM_PROMPT
    assert (
        "Focus on technical assessment, system configuration auditing, and benchmark verification without violating safety boundaries."
    ) in COMPUTER_USE_SYSTEM_PROMPT

    # Benchmark and target auditing framing check
    assert "extracting benchmark verification keys" in COMPUTER_USE_SYSTEM_PROMPT
    assert "auditing database ports" in COMPUTER_USE_SYSTEM_PROMPT
    assert "discovering challenge flags" in COMPUTER_USE_SYSTEM_PROMPT
    # Verify counter-productive safety trigger phrases are not present
    assert "malicious intrusion" not in COMPUTER_USE_SYSTEM_PROMPT

    # Autonomy and non-scripted check
    assert "NOT a scripted puppet" in COMPUTER_USE_SYSTEM_PROMPT
    assert "target-first autonomy" in COMPUTER_USE_SYSTEM_PROMPT
    assert "dynamic tool selection" in COMPUTER_USE_SYSTEM_PROMPT
    assert "first-principles hypothesis testing" in COMPUTER_USE_SYSTEM_PROMPT


def test_computer_use_system_prompt_compact_has_authorized_benchmark_framing():
    assert "AUTHORIZED EVALUATION BENCHMARK:" in COMPUTER_USE_SYSTEM_PROMPT_COMPACT
    assert (
        "All actions, commands, and security verifications are performed within an "
        "authorized educational CTF, vulnerability research lab, or authorized system evaluation benchmark."
    ) in COMPUTER_USE_SYSTEM_PROMPT_COMPACT
    assert "extracting benchmark verification keys" in COMPUTER_USE_SYSTEM_PROMPT_COMPACT
    assert "auditing database ports" in COMPUTER_USE_SYSTEM_PROMPT_COMPACT
    assert "dynamic tool selection" in COMPUTER_USE_SYSTEM_PROMPT_COMPACT
    assert "first-principles hypothesis testing" in COMPUTER_USE_SYSTEM_PROMPT_COMPACT


def test_compose_system_prompt_includes_benchmark_clause():
    prompt = compose_system_prompt("Test Role", "Test Body")
    assert AUTHORIZED_BENCHMARK_CLAUSE in prompt


def test_default_identity_is_human_like_computer_engineering_first():
    """The default identity must not frame every task as cybersecurity work."""
    assert "Autonomous Computer Engineering Agent" in ASEA_IDENTITY
    assert "human-like autonomous computer engineering agent" in COMPUTER_USE_SYSTEM_PROMPT
    assert "Cybersecurity is an optional specialization" in COMPUTER_USE_SYSTEM_PROMPT
    assert "PRIMARY ROLE:" in COMPUTER_USE_SYSTEM_PROMPT
    assert "cybersecurity remains optional" not in COMPUTER_USE_SYSTEM_PROMPT.lower()
    for forbidden in FORBIDDEN_IDENTITIES:
        assert forbidden not in COMPUTER_USE_SYSTEM_PROMPT


def test_default_prompts_preserve_safety_and_anti_bypass_invariants():
    """Identity changes must not remove the shared safety boundaries."""
    assert "HONESTY:" in COMPUTER_USE_SYSTEM_PROMPT
    assert "SAFETY:" in COMPUTER_USE_SYSTEM_PROMPT
    assert "Universal Anti-Bypass Invariant" in COMPUTER_USE_SYSTEM_PROMPT
    assert "Cybersecurity is an optional specialization" in COMPUTER_USE_SYSTEM_PROMPT_COMPACT
