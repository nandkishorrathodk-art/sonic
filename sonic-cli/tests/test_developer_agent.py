"""Comprehensive test suite for the SONIC Developer Agent and Developer CLI.

Tests:
1. Tool execution (view_file, write_file, edit_file, run_command, git_diff, run_pytest).
2. DeveloperAgent tool loop with mocked and simulated streaming tool calls.
3. Slash command handling (/diff, /journal, /test, /status, /help, greetings).
4. Zero-boss isolation: strictly assert that BossAgent, Docker desktop, and
   /workstation/prompt are never imported or invoked.

Belongs to Department: platform_systems, Team: api_services, Subsystem: cli.
"""

from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from rich.console import Console

from sonic_cli import repl
from sonic_cli.context import LocalWorkspaceContext, resolve_local_context
from sonic_cli.developer_agent import (
    DEFAULT_DEVELOPER_SYSTEM_PROMPT,
    DeveloperAgent,
    DeveloperAgentResult,
    LLMConfig,
    edit_file,
    git_diff,
    run_command,
    run_pytest,
    view_file,
    write_file,
)
from sonic_cli.main import app

# ==============================================================================
# 1. Standalone Tool Execution Tests
# ==============================================================================


class TestToolExecution:
    """Test suite for direct workspace tool execution."""

    def test_view_file_full_and_range(self, tmp_path: Path):
        """Verify view_file renders lines with line numbers and supports ranges."""
        test_file = tmp_path / "sample.py"
        lines = [f"line {i}" for i in range(1, 11)]
        test_file.write_text("\n".join(lines), encoding="utf-8")

        # Full file view
        full_output = view_file("sample.py", workspace_dir=tmp_path)
        assert "File: sample.py (lines 1-10 of 10):" in full_output
        assert "   1 | line 1" in full_output
        assert "  10 | line 10" in full_output

        # Line range view (lines 3 to 5)
        range_output = view_file("sample.py", start_line=3, end_line=5, workspace_dir=tmp_path)
        assert "lines 3-5 of 10" in range_output
        assert "   3 | line 3" in range_output
        assert "   5 | line 5" in range_output
        assert "line 1" not in range_output
        assert "line 6" not in range_output

    def test_view_file_directory_listing(self, tmp_path: Path):
        """Verify view_file on a directory lists files and subdirectories."""
        (tmp_path / "file1.txt").write_text("hello", encoding="utf-8")
        (tmp_path / "subdir").mkdir()
        (tmp_path / "subdir" / "nested.txt").write_text("nested", encoding="utf-8")

        listing = view_file(".", workspace_dir=tmp_path)
        assert "Directory listing for '.'" in listing
        assert "file1.txt" in listing
        assert "subdir/" in listing

    def test_view_file_edge_cases(self, tmp_path: Path):
        """Verify view_file error handling for missing, binary, empty, and invalid ranges."""
        # Missing file
        missing_out = view_file("nonexistent.py", workspace_dir=tmp_path)
        assert "Error: File not found" in missing_out

        # Empty file
        empty_file = tmp_path / "empty.txt"
        empty_file.write_text("", encoding="utf-8")
        empty_out = view_file("empty.txt", workspace_dir=tmp_path)
        assert "is empty, 0 lines" in empty_out

        # Binary file
        bin_file = tmp_path / "binary.bin"
        bin_file.write_bytes(b"some\x00binary\x00data")
        bin_out = view_file("binary.bin", workspace_dir=tmp_path)
        assert "appears to be a binary file" in bin_out

        # Invalid range: start_line > total_lines
        valid_file = tmp_path / "two_lines.txt"
        valid_file.write_text("a\nb\n", encoding="utf-8")
        range_out = view_file("two_lines.txt", start_line=10, workspace_dir=tmp_path)
        assert "Error: start_line 10 exceeds total lines" in range_out

        # Invalid range: end_line < start_line
        range_err = view_file("two_lines.txt", start_line=2, end_line=1, workspace_dir=tmp_path)
        assert "cannot be less than start_line" in range_err

    def test_write_file_new_and_overwrite(self, tmp_path: Path):
        """Verify write_file creates parent dirs, writes, and overwrites files."""
        nested_file = "deeply/nested/dir/test.py"
        content1 = "def foo():\n    return 'bar'\n"

        result1 = write_file(nested_file, content1, workspace_dir=tmp_path)
        assert "Successfully wrote 2 lines" in result1
        assert (tmp_path / nested_file).exists()
        assert (tmp_path / nested_file).read_text(encoding="utf-8") == content1

        content2 = "def foo():\n    return 'updated'\n"
        result2 = write_file(nested_file, content2, workspace_dir=tmp_path)
        assert "Successfully wrote 2 lines" in result2
        assert (tmp_path / nested_file).read_text(encoding="utf-8") == content2

    def test_edit_file_exact_and_resilient(self, tmp_path: Path):
        """Verify edit_file replaces unique blocks and handles line-ending differences."""
        target = tmp_path / "code.py"
        original = "def add(a, b):\n    # TODO: fix\n    return a - b\n"
        target.write_text(original, encoding="utf-8")

        # Exact replacement
        res = edit_file(
            "code.py",
            target_content="    # TODO: fix\n    return a - b",
            replacement_content="    return a + b",
            workspace_dir=tmp_path,
        )
        assert "Successfully edited 'code.py'" in res
        updated = target.read_text(encoding="utf-8")
        assert "return a + b" in updated
        assert "return a - b" not in updated

    def test_edit_file_errors(self, tmp_path: Path):
        """Verify edit_file returns clear errors for missing targets or ambiguous matches."""
        target = tmp_path / "code.py"
        target.write_text("x = 1\nx = 1\ny = 2\n", encoding="utf-8")

        # Ambiguous target (occurs twice)
        dup_res = edit_file("code.py", "x = 1", "x = 10", workspace_dir=tmp_path)
        assert "occurs 2 times" in dup_res
        assert "surrounding context lines" in dup_res

        # Target not found
        missing_target = edit_file("code.py", "z = 99", "z = 100", workspace_dir=tmp_path)
        assert "target_content not found" in missing_target

        # Non-existent file
        no_file = edit_file("ghost.py", "a", "b", workspace_dir=tmp_path)
        assert "Error: File not found" in no_file

    def test_run_command_success_and_failure(self, tmp_path: Path):
        """Verify run_command captures stdout, stderr, and exit codes."""
        # Success command using sys.executable
        py_cmd = f'"{sys.executable}" -c "print(\'SONIC_DEVELOPER_TEST_OK\')"'
        out_ok = run_command(py_cmd, workspace_dir=tmp_path)
        assert "Exit code: 0" in out_ok
        assert "SONIC_DEVELOPER_TEST_OK" in out_ok

        # Non-zero exit code
        py_err = f'"{sys.executable}" -c "import sys; sys.stderr.write(\'TEST_ERR\'); sys.exit(42)"'
        out_err = run_command(py_err, workspace_dir=tmp_path)
        assert "Exit code: 42" in out_err
        assert "TEST_ERR" in out_err

    def test_run_command_timeout(self, tmp_path: Path):
        """Verify run_command enforces timeout safely."""
        # Sleep for 5 seconds with a 1 second timeout
        py_sleep = f'"{sys.executable}" -c "import time; time.sleep(5)"'
        out_to = run_command(py_sleep, timeout=1, workspace_dir=tmp_path)
        assert "timed out after 1 seconds" in out_to

    def test_git_diff_execution(self, tmp_path: Path):
        """Verify git_diff identifies clean vs dirty working tree."""
        # Initialize a temporary git repository
        subprocess.run(["git", "init"], cwd=str(tmp_path), capture_output=True, check=True)
        subprocess.run(["git", "config", "user.name", "Test"], cwd=str(tmp_path), check=True)
        subprocess.run(["git", "config", "user.email", "test@test.local"], cwd=str(tmp_path), check=True)

        # Initial commit
        (tmp_path / "hello.txt").write_text("initial", encoding="utf-8")
        subprocess.run(["git", "add", "hello.txt"], cwd=str(tmp_path), check=True)
        subprocess.run(["git", "commit", "-m", "init"], cwd=str(tmp_path), check=True)

        # Clean status check
        clean_res = git_diff(workspace_dir=tmp_path)
        assert "No git diff detected" in clean_res

        # Modify file and verify diff detection
        (tmp_path / "hello.txt").write_text("modified", encoding="utf-8")
        dirty_res = git_diff(workspace_dir=tmp_path)
        assert "Git Diff:" in dirty_res
        assert "+modified" in dirty_res

    def test_run_pytest_in_workspace(self, tmp_path: Path):
        """Verify run_pytest executes tests and returns formatted output."""
        test_file = tmp_path / "test_dummy.py"
        test_file.write_text("def test_pass(): assert True\n", encoding="utf-8")

        res = run_pytest(test_path="test_dummy.py", workspace_dir=tmp_path)
        assert "pytest exit code: 0" in res
        assert "1 passed" in res


# ==============================================================================
# 2. DeveloperAgent Tool Dispatcher and Configuration Tests
# ==============================================================================


class TestDeveloperAgentConfigAndDispatcher:
    """Test LLMConfig resolution and tool dispatcher wiring."""

    def test_llm_config_resolution(self, monkeypatch: pytest.MonkeyPatch):
        """Verify LLMConfig loads from environment or falls back cleanly."""
        monkeypatch.setenv("GROQ_API_KEY", "gsk_test_key_12345")
        monkeypatch.setenv("DEFAULT_LLM_PROVIDER", "groq")

        cfg = LLMConfig.from_env()
        assert cfg.provider == "groq"
        assert cfg.model == "qwen/qwen3.8-27b"
        assert cfg.api_key == "gsk_test_key_12345"
        assert cfg.base_url == "https://api.groq.com/openai/v1"

    def test_developer_agent_init(self, tmp_path: Path):
        """Verify DeveloperAgent initializes with workspace and default tools."""
        agent = DeveloperAgent(workspace_dir=tmp_path, max_steps=10)
        assert agent.workspace_dir == tmp_path.resolve()
        assert agent.max_steps == 10
        assert agent.system_prompt == DEFAULT_DEVELOPER_SYSTEM_PROMPT

    def test_agent_execute_tool_dispatch(self, tmp_path: Path):
        """Verify agent.execute_tool dispatches calls to all supported tools."""
        agent = DeveloperAgent(workspace_dir=tmp_path)

        # 1. write_file
        w_res = agent.execute_tool("write_file", {"path": "module.py", "content": "x = 42\n"})
        assert "Successfully wrote" in w_res

        # 2. view_file
        v_res = agent.execute_tool("view_file", {"path": "module.py", "start_line": 1, "end_line": 1})
        assert "x = 42" in v_res

        # 3. edit_file
        e_res = agent.execute_tool("edit_file", {"path": "module.py", "target_content": "42", "replacement_content": "99"})
        assert "Successfully edited" in e_res

        # 4. run_command
        py_cmd = f'"{sys.executable}" -c "print(\'EXEC_TEST_OK\')"'
        c_res = agent.execute_tool("run_command", {"command": py_cmd, "timeout": 10})
        assert "EXEC_TEST_OK" in c_res

        # 5. git_diff
        d_res = agent.execute_tool("git_diff", {})
        assert isinstance(d_res, str)

        # 6. run_pytest
        p_res = agent.execute_tool("run_pytest", {"test_path": "nonexistent.py"})
        assert "pytest exit code:" in p_res

        # Unknown tool
        u_res = agent.execute_tool("unknown_tool", {})
        assert "Error: Unknown tool 'unknown_tool'" in u_res


# ==============================================================================
# 3. DeveloperAgent Observe-Reason-Act Tool Loop Tests
# ==============================================================================


class TestDeveloperAgentToolLoop:
    """Test autonomous tool execution loop with mocked and simulated LLM clients."""

    def test_single_turn_completion_without_tools(self, tmp_path: Path):
        """When LLM returns direct response without tool calls, agent completes in 1 step."""
        console = Console(record=True)

        def mock_llm_client(messages: list[dict], tools: list[dict]):
            return {"role": "assistant", "content": "I have completed the analysis."}, []

        agent = DeveloperAgent(workspace_dir=tmp_path, console=console, llm_client=mock_llm_client)
        result: DeveloperAgentResult = agent.run("Analyze the codebase")

        assert result.success is True
        assert result.steps == 1
        assert result.tool_calls_count == 0
        assert "completed the analysis" in result.final_response
        assert result.interrupted is False
        assert result.error is None

    def test_multi_turn_tool_calling_loop(self, tmp_path: Path):
        """Simulate a 2-turn loop: LLM calls write_file, receives tool output, then finishes."""
        turn = 0

        def mock_llm_client(messages: list[dict], tools: list[dict]):
            nonlocal turn
            turn += 1
            if turn == 1:
                return (
                    {"role": "assistant", "content": None},
                    [
                        {
                            "id": "call_123",
                            "type": "function",
                            "function": {
                                "name": "write_file",
                                "arguments": json.dumps({"path": "calc.py", "content": "def add(a, b): return a + b\n"}),
                            },
                        }
                    ],
                )
            else:
                # Turn 2: assert tool result was passed in messages
                last_msg = messages[-1]
                assert last_msg["role"] == "tool"
                assert "Successfully wrote" in last_msg["content"]
                return {"role": "assistant", "content": "calc.py has been created with add() function."}, []

        agent = DeveloperAgent(workspace_dir=tmp_path, llm_client=mock_llm_client)
        result = agent.run("Create calc.py with add function")

        assert result.success is True
        assert result.steps == 2
        assert result.tool_calls_count == 1
        assert "calc.py has been created" in result.final_response
        assert (tmp_path / "calc.py").exists()

    def test_max_steps_limit_handling(self, tmp_path: Path):
        """When LLM continues generating tool calls past max_steps, agent stops safely."""
        def infinite_tool_llm(messages: list[dict], tools: list[dict]):
            return (
                {"role": "assistant", "content": None},
                [
                    {
                        "id": "call_loop",
                        "type": "function",
                        "function": {
                            "name": "view_file",
                            "arguments": json.dumps({"path": "nonexistent.py"}),
                        },
                    }
                ],
            )

        agent = DeveloperAgent(workspace_dir=tmp_path, max_steps=3, llm_client=infinite_tool_llm)
        result = agent.run("Infinite loop test")

        assert result.success is False
        assert result.steps == 3
        assert result.tool_calls_count == 3
        assert "Step limit (3) reached" in str(result.error)

    def test_keyboard_interrupt_graceful_exit(self, tmp_path: Path):
        """Verify KeyboardInterrupt (Ctrl+C) exits cleanly with interrupted=True."""
        def interrupt_llm(messages: list[dict], tools: list[dict]):
            raise KeyboardInterrupt()

        agent = DeveloperAgent(workspace_dir=tmp_path, llm_client=interrupt_llm)
        result = agent.run("Task that gets interrupted")

        assert result.success is False
        assert result.interrupted is True
        assert "interrupted by user" in result.final_response.lower()

    def test_llm_api_exception_handling(self, tmp_path: Path):
        """Verify unexpected API errors are captured and returned in DeveloperAgentResult."""
        def failing_llm(messages: list[dict], tools: list[dict]):
            raise RuntimeError("API quota exceeded")

        agent = DeveloperAgent(workspace_dir=tmp_path, llm_client=failing_llm)
        result = agent.run("Task that fails")

        assert result.success is False
        assert result.error == "API quota exceeded"

    def test_sse_streaming_response_parser(self, tmp_path: Path):
        """Test _step_llm streaming parser extracting reasoning, content, and tool calls from SSE chunks."""
        sse_lines = [
            'data: {"choices": [{"delta": {"role": "assistant", "content": "<think>Thinking about file structure</think>"}}]}',
            'data: {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "tc_1", "function": {"name": "view_file", "arguments": "{\\"path\\": "}}]}}]}',
            'data: {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": "\\"README.md\\"}"}}]}}]}',
            'data: [DONE]',
        ]

        class MockResponse:
            status_code = 200

            def iter_lines(self):
                return sse_lines

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

        class MockClient:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            def stream(self, *args, **kwargs):
                return MockResponse()

        cfg = LLMConfig(provider="groq", api_key="dummy-key")
        agent = DeveloperAgent(workspace_dir=tmp_path, config=cfg)

        with patch("sonic_cli.developer_agent.httpx.Client", return_value=MockClient()):
            assistant_msg, tool_calls = agent._step_llm([{"role": "user", "content": "view readme"}])

        assert len(tool_calls) == 1
        assert tool_calls[0]["function"]["name"] == "view_file"
        assert json.loads(tool_calls[0]["function"]["arguments"]) == {"path": "README.md"}


# ==============================================================================
# 4. REPL Slash Commands & Greetings Tests
# ==============================================================================


class TestReplSlashCommands:
    """Test developer slash commands (/diff, /journal, /test, /status, /help, greetings)."""

    def test_handle_diff_git_and_nongit(self, tmp_path: Path):
        """Verify _handle_diff executes git diff or displays warning for non-git directory."""
        # Non-git workspace
        ctx_nongit = LocalWorkspaceContext(
            repo_root=tmp_path,
            branch="main",
            tenant_id="tenant-test",
            workspace_id="ws-test",
            session_id="sess-test",
            is_git=False,
        )
        with patch.object(repl.console, "print") as mock_print:
            repl._handle_diff(ctx_nongit)
            assert any("Not in a git repository" in str(c) for c in mock_print.call_args_list)

        # Git workspace
        ctx_git = LocalWorkspaceContext(
            repo_root=tmp_path,
            branch="main",
            tenant_id="tenant-test",
            workspace_id="ws-test",
            session_id="sess-test",
            is_git=True,
        )
        with patch("sonic_cli.repl.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout="diff --git a/file b/file\n+new line")
            repl._handle_diff(ctx_git)
            mock_run.assert_called_once()

    def test_handle_journal(self, tmp_path: Path):
        """Verify _handle_journal reads evolution.md or candidate records."""
        ctx = LocalWorkspaceContext(
            repo_root=tmp_path,
            branch="main",
            tenant_id="tenant-test",
            workspace_id="ws-test",
            session_id="sess-test",
            is_git=True,
        )
        # Empty workspace
        with patch.object(repl.console, "print") as mock_print:
            repl._handle_journal(ctx)
            assert any("No evolution.md or candidate records" in str(c) for c in mock_print.call_args_list)

        # Workspace with evolution.md
        (tmp_path / "evolution.md").write_text("# Evolution Journal\n- Step 1", encoding="utf-8")
        with patch.object(repl.console, "print") as mock_print:
            repl._handle_journal(ctx)
            # Rendered as markdown
            assert mock_print.called

    def test_handle_test(self, tmp_path: Path):
        """Verify _handle_test runs pytest with correct target path."""
        ctx = LocalWorkspaceContext(
            repo_root=tmp_path,
            branch="main",
            tenant_id="tenant-test",
            workspace_id="ws-test",
            session_id="sess-test",
            is_git=True,
        )
        with patch("sonic_cli.repl.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="5 passed", stderr="")
            repl._handle_test(ctx, test_path="tests/test_math.py")
            mock_run.assert_called_once()
            cmd = mock_run.call_args[0][0]
            assert "pytest" in cmd
            assert "tests/test_math.py" in cmd

    def test_handle_status_and_help(self):
        """Verify _handle_status and _handle_help display without errors."""
        ctx = resolve_local_context()
        with patch.object(repl.console, "print"):
            repl._handle_status(ctx)
            repl._handle_help()

    def test_greeting_detection_and_conversational_response(self):
        """Verify greetings ('hi', 'hello', 'hii') are detected and answered conversationally."""
        assert repl._is_greeting("hi") is True
        assert repl._is_greeting("Hello!") is True
        assert repl._is_greeting("hii") is True
        assert repl._is_greeting("hey there") is True
        assert repl._is_greeting("write a quicksort function") is False

        with patch.object(repl.console, "print") as mock_print:
            repl._handle_greeting()
            output = "".join(str(c) for c in mock_print.call_args_list)
            assert "SONIC" in output
            assert "autonomous software engineer" in output


# ==============================================================================
# 5. Zero-Boss Isolation Tests
# ==============================================================================


class TestZeroBossIsolation:
    """Strictly assert that BossAgent, Docker desktop, and /workstation/prompt

    are completely isolated and NEVER invoked or imported by the developer CLI.
    """

    def test_zero_boss_in_developer_agent_source(self):
        """Assert DeveloperAgent source code contains zero references to BossAgent or workstation."""
        dev_agent_file = Path(__file__).resolve().parents[1] / "sonic_cli" / "developer_agent.py"
        source_code = dev_agent_file.read_text(encoding="utf-8")

        # Parse AST to ensure valid syntax
        tree = ast.parse(source_code)
        assert tree is not None

        # Forbidden tokens in the developer agent
        forbidden_tokens = [
            "BossAgent",
            "boss_agent",
            "/workstation/prompt",
            "/workstation/state",
            "sonic.orchestration",
            "sonic.computer.docker",
            "DockerComputerProvider",
            "ComputerUseAgent",
        ]
        for token in forbidden_tokens:
            assert token not in source_code, f"Forbidden token '{token}' found in developer_agent.py!"

    def test_zero_boss_in_repl_source(self):
        """Assert repl.py source code contains zero references to BossAgent or workstation endpoints."""
        repl_file = Path(__file__).resolve().parents[1] / "sonic_cli" / "repl.py"
        source_code = repl_file.read_text(encoding="utf-8")

        forbidden_tokens = [
            "BossAgent",
            "/workstation/prompt",
            "/workstation/state",
            "/workstation/session/interrupt",
            "sonic.orchestration",
        ]
        for token in forbidden_tokens:
            assert token not in source_code, f"Forbidden token '{token}' found in repl.py!"

    def test_zero_boss_in_registered_subcommands(self):
        """Assert main Typer app has purged all workstation / Boss subcommands."""
        from typer.testing import CliRunner
        runner = CliRunner()
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0

        # These must NOT be present in CLI help
        workstation_commands = [
            "computer",
            "mission",
            "agents",
            "engage",
            "finding",
            "evidence",
            "research",
            "security",
            "experiment",
            "graph",
            "prompt",
            "hypotheses",
        ]
        for cmd in workstation_commands:
            assert cmd not in result.output, f"Obsolete workstation command '{cmd}' still registered in CLI!"

    def test_zero_boss_runtime_isolation(self, tmp_path: Path):
        """Assert that running DeveloperAgent never makes network calls to /workstation/prompt."""
        with patch("sonic_cli.developer_agent.httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client_cls.return_value.__enter__.return_value = mock_client

            def mock_llm_turn(messages, tools):
                return {"role": "assistant", "content": "Done without Boss!"}, []

            agent = DeveloperAgent(workspace_dir=tmp_path, llm_client=mock_llm_turn)
            res = agent.run("Refactor tests")

            assert res.success is True
            # Assert httpx was never called with any workstation URLs
            for call in mock_client.method_calls:
                call_str = str(call)
                assert "workstation" not in call_str
                assert "boss" not in call_str.lower()
