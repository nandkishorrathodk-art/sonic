"""Comprehensive Safety and Tool Sandboxing Audit Tests for DeveloperAgent.

Audits:
1. Path safety: confinement within repo root (prevents ../ traversal and absolute path escaping).
2. Subprocess safety: timeout handling, hanging process termination, clean stdout/stderr capture.
3. Secret safety: zero API key leakage (GROQ_API_KEY, OpenAI, bearer tokens) in logs, repr, or console outputs.
4. Error feedback: clear actionable error strings for non-existent files, missing edit targets, and duplicate targets.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from sonic_cli.developer_agent import (
    DeveloperAgent,
    LLMConfig,
    _resolve_workspace_path,
    edit_file,
    run_command,
    sanitize_secrets,
    view_file,
    write_file,
)

# ==============================================================================
# 1. Path Safety & Confinement Tests
# ==============================================================================


def test_resolve_workspace_path_confinement(tmp_path: Path):
    """Paths attempting to traverse outside workspace root must raise PermissionError."""
    # Valid relative path
    valid = _resolve_workspace_path("sub/file.txt", workspace_dir=tmp_path)
    assert valid == (tmp_path / "sub" / "file.txt").resolve()

    # Valid absolute path inside workspace
    inside_abs = (tmp_path / "inside.txt").resolve()
    assert _resolve_workspace_path(inside_abs, workspace_dir=tmp_path) == inside_abs

    # Traversal attack via ../../
    with pytest.raises(PermissionError) as exc_info:
        _resolve_workspace_path("../../outside.txt", workspace_dir=tmp_path)
    assert "escapes repository workspace root" in str(exc_info.value)

    # Empty path check
    with pytest.raises(ValueError):
        _resolve_workspace_path("", workspace_dir=tmp_path)


def test_view_file_blocks_path_traversal(tmp_path: Path):
    """view_file must refuse reading files outside repository workspace."""
    # ../../ attack
    res_traversal = view_file("../../secret.txt", workspace_dir=tmp_path)
    assert "Error:" in res_traversal
    assert "escapes repository workspace root" in res_traversal

    # Absolute path escaping repo (e.g. system root)
    outside = Path(tmp_path.parent / "unauthorized.txt")
    outside.write_text("classified data", encoding="utf-8")
    res_abs = view_file(outside, workspace_dir=tmp_path)
    assert "Error:" in res_abs
    assert "escapes repository workspace root" in res_abs


def test_write_file_blocks_path_traversal(tmp_path: Path):
    """write_file must refuse writing files outside repository workspace."""
    # ../../ write attack
    res_traversal = write_file("../../evil.sh", "#!/bin/sh\necho hacked", workspace_dir=tmp_path)
    assert "Error:" in res_traversal
    assert "escapes repository workspace root" in res_traversal

    # Absolute path escaping repo
    outside_file = tmp_path.parent / "escape.txt"
    res_abs = write_file(outside_file, "escaped content", workspace_dir=tmp_path)
    assert "Error:" in res_abs
    assert "escapes repository workspace root" in res_abs
    assert not outside_file.exists()


def test_edit_file_blocks_path_traversal(tmp_path: Path):
    """edit_file must refuse editing files outside repository workspace."""
    # Traversal attack
    res = edit_file("../../target.txt", "old", "new", workspace_dir=tmp_path)
    assert "Error:" in res
    assert "escapes repository workspace root" in res


# ==============================================================================
# 2. Subprocess Execution & Timeout Safety Tests
# ==============================================================================


def test_run_command_timeout_clean_termination(tmp_path: Path):
    """run_command must terminate hanging processes on timeout and report diagnostic info."""
    # Command that would sleep for 10 seconds, but timeout in 1s
    res = run_command(
        'python -c "import time; time.sleep(10)"',
        timeout=1,
        workspace_dir=tmp_path,
    )
    assert "Error: Command timed out after 1 seconds" in res


def test_run_command_partial_output_on_timeout(tmp_path: Path):
    """run_command captures partial stdout before timeout occurs."""
    cmd = 'python -u -c "import time, sys; sys.stdout.write(\'starting_work\\n\'); sys.stdout.flush(); time.sleep(5)"'
    res = run_command(cmd, timeout=1, workspace_dir=tmp_path)
    assert "timed out after 1 seconds" in res
    assert "starting_work" in res


def test_run_command_clean_capture(tmp_path: Path):
    """run_command cleanly separates exit code, stdout, and stderr."""
    cmd = 'python -c "import sys; sys.stdout.write(\'normal_out\\n\'); sys.stderr.write(\'error_out\\n\'); sys.exit(3)"'
    res = run_command(cmd, workspace_dir=tmp_path)
    assert "Exit code: 3" in res
    assert "STDOUT:\nnormal_out" in res
    assert "STDERR:\nerror_out" in res


def test_run_command_unicode_handling(tmp_path: Path):
    """run_command handles non-ASCII and Unicode characters safely without decoding crashes."""
    cmd = 'python -c "print(\'🚀 Unicode test: こんにちは \')"'
    res = run_command(cmd, workspace_dir=tmp_path)
    assert "Exit code: 0" in res
    assert "Unicode test:" in res


# ==============================================================================
# 3. Secret Safety & Sanitization Tests
# ==============================================================================


def test_sanitize_secrets_env_masking(monkeypatch):
    """sanitize_secrets must mask sensitive environment variable values."""
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test_secret_key_1234567890abcdef")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-proj-super_confidential_openai_token_xyz")
    monkeypatch.setenv("JWT_SECRET", "super_secret_jwt_passphrase_here")

    raw_text = (
        "Connecting with GROQ_API_KEY=gsk_test_secret_key_1234567890abcdef and "
        "OPENAI_API_KEY=sk-proj-super_confidential_openai_token_xyz. "
        "JWT token signing key is super_secret_jwt_passphrase_here."
    )
    cleaned = sanitize_secrets(raw_text)

    assert "gsk_test_secret_key_1234567890abcdef" not in cleaned
    assert "sk-proj-super_confidential_openai_token_xyz" not in cleaned
    assert "super_secret_jwt_passphrase_here" not in cleaned
    assert "[REDACTED_API_KEY]" in cleaned or "[REDACTED_KEY]" in cleaned


def test_sanitize_secrets_regex_patterns():
    """sanitize_secrets regex filters catch common key formats even without matching env."""
    text_with_groq = "Request failed with key: gsk_abcdef1234567890abcdef1234567890"
    cleaned_groq = sanitize_secrets(text_with_groq)
    assert "abcdef1234567890abcdef1234567890" not in cleaned_groq
    assert "[REDACTED_KEY]" in cleaned_groq

    text_with_bearer = "Authorization: Bearer mySecretTokenValue1234567890"
    cleaned_bearer = sanitize_secrets(text_with_bearer)
    assert "mySecretTokenValue1234567890" not in cleaned_bearer
    assert "Bearer [REDACTED_TOKEN]" in cleaned_bearer


def test_llm_config_repr_masks_api_key():
    """LLMConfig __repr__ must omit api_key to prevent accidental logging leaks."""
    cfg = LLMConfig(provider="groq", api_key="gsk_secret_value_not_to_be_printed")
    repr_str = repr(cfg)
    assert "gsk_secret_value_not_to_be_printed" not in repr_str
    assert "api_key" not in repr_str


def test_developer_agent_execute_tool_masks_secrets(tmp_path: Path):
    """DeveloperAgent.execute_tool must mask secrets in tool execution results."""
    secret = "gsk_secret_agent_key_998877665544332211"
    config = LLMConfig(provider="groq", api_key=secret)
    agent = DeveloperAgent(workspace_dir=tmp_path, config=config)

    # Tool output that echoes the secret
    result = agent.execute_tool("run_command", {"command": f'python -c "print(\'{secret}\')"', "timeout": 10})
    assert secret not in result
    assert "[REDACTED_API_KEY]" in result or "[REDACTED_KEY]" in result


# ==============================================================================
# 4. Error Feedback & Self-Correction Tests
# ==============================================================================


def test_file_not_found_actionable_feedback(tmp_path: Path):
    """Missing file errors provide actionable feedback to re-inspect workspace."""
    res_view = view_file("missing_module.py", workspace_dir=tmp_path)
    assert "Error: File not found at 'missing_module.py'" in res_view
    assert "view_file" in res_view

    res_edit = edit_file("nonexistent.py", "target", "replace", workspace_dir=tmp_path)
    assert "Error: File not found at 'nonexistent.py'" in res_edit
    assert "view_file" in res_edit


def test_edit_file_target_not_found_self_correction_guidance(tmp_path: Path):
    """When target_content is missing, error instructs the LLM to inspect using view_file."""
    test_f = tmp_path / "code.py"
    test_f.write_text("def calculate(x, y):\n    return x + y\n", encoding="utf-8")

    res = edit_file(
        test_f,
        target_content="def calculate(a, b):\n    return a + b\n",
        replacement_content="def calculate(a, b):\n    return a * b\n",
        workspace_dir=tmp_path,
    )
    assert "Error: target_content not found in" in res
    assert "view_file" in res
    assert "indentation" in res


def test_edit_file_multiple_occurrences_guidance(tmp_path: Path):
    """When target_content matches multiple times, error instructs LLM to add surrounding context."""
    test_f = tmp_path / "repeated.py"
    test_f.write_text("x = 1\ny = 2\nx = 1\n", encoding="utf-8")

    res = edit_file(test_f, target_content="x = 1", replacement_content="x = 10", workspace_dir=tmp_path)
    assert "occurs 2 times" in res
    assert "view_file" in res
    assert "surrounding context" in res


def test_edit_file_empty_target_rejected(tmp_path: Path):
    """Empty target_content is explicitly rejected."""
    test_f = tmp_path / "empty_test.py"
    test_f.write_text("content", encoding="utf-8")

    res = edit_file(test_f, target_content="", replacement_content="new", workspace_dir=tmp_path)
    assert "Error: target_content cannot be empty" in res


def test_edit_file_on_directory_rejected(tmp_path: Path):
    """edit_file on a directory returns clear error."""
    sub_dir = tmp_path / "folder"
    sub_dir.mkdir()

    res = edit_file(sub_dir, target_content="x", replacement_content="y", workspace_dir=tmp_path)
    assert "is a directory, not a file" in res


@pytest.mark.parametrize("provider", ["groq", "openai", "openrouter", "deepseek", "nvidia"])
def test_llm_config_does_not_borrow_another_providers_key(monkeypatch, provider, tmp_path):
    """A missing provider key must not send unrelated credentials to its API."""
    key_vars = {
        "groq": "GROQ_API_KEY",
        "openai": "OPENAI_API_KEY",
        "openrouter": "OPENROUTER_API_KEY",
        "deepseek": "DEEPSEEK_API_KEY",
        "nvidia": "NVIDIA_API_KEY",
    }
    for name, variable in key_vars.items():
        if name == provider:
            monkeypatch.delenv(variable, raising=False)
        else:
            monkeypatch.setenv(variable, f"synthetic-{name}-credential")

    config = LLMConfig.from_env(provider=provider)
    assert config.api_key == ""
    agent = DeveloperAgent(workspace_dir=tmp_path, config=config)
    monkeypatch.setattr(agent, "_get_client", lambda: pytest.fail("Missing key must block HTTP requests"))
    with pytest.raises(ValueError, match=key_vars[provider]):
        agent._call_single_llm([])
    assert LLMConfig.from_env(provider=provider, api_key="explicit-test-key").api_key == "explicit-test-key"


@pytest.mark.parametrize("file_newline", ["\n", "\r\n"])
@pytest.mark.parametrize("target_newline", ["\n", "\r\n"])
@pytest.mark.parametrize("replacement_newline", ["\n", "\r\n"])
def test_edit_file_preserves_line_endings(tmp_path, file_newline, target_newline, replacement_newline):
    """Edits from a model must preserve the existing file's newline convention."""
    target = tmp_path / "code.py"
    target.write_bytes(file_newline.join(["before", "old", "value", "after", ""]).encode())
    result = edit_file(
        target,
        target_newline.join(["old", "value"]),
        replacement_newline.join(["new", "value"]),
        workspace_dir=tmp_path,
    )
    assert "Successfully edited" in result
    assert target.read_bytes() == file_newline.join(["before", "new", "value", "after", ""]).encode()


def test_edit_file_rejects_invalid_utf8_without_corrupting_file(tmp_path):
    target = tmp_path / "legacy.txt"
    original = b"old value\nlegacy byte: \xff\n"
    target.write_bytes(original)

    result = edit_file(target, "old value", "new value", workspace_dir=tmp_path)

    assert result.startswith("Error")
    assert target.read_bytes() == original


def test_edit_file_preserves_unedited_mixed_line_endings(tmp_path):
    target = tmp_path / "mixed.txt"
    target.write_bytes(b"before\r\nold\r\nvalue\r\nafter\n")

    result = edit_file(target, "old\nvalue", "new\nvalue", workspace_dir=tmp_path)

    assert "Successfully edited" in result
    assert target.read_bytes() == b"before\r\nnew\r\nvalue\r\nafter\n"


def test_edit_file_rejects_ambiguous_mixed_line_endings(tmp_path):
    target = tmp_path / "mixed.txt"
    original = b"old\r\nvalue\r\nold\nvalue\n"
    target.write_bytes(original)

    result = edit_file(target, "old\nvalue", "new\nvalue", workspace_dir=tmp_path)

    assert "occurs 2 times" in result
    assert target.read_bytes() == original
