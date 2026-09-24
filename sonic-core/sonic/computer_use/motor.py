"""
SONIC A-SEA — System 1 Motor Reflexes, Human Cadence & Hotkeys
===============================================================
Provides human-like motor reflexes, human typing cadence, hotkey navigation,
reflexive backtracking (modal dismissal / history back), and browser tab
budget hygiene for the SONIC Computer.
"""

from __future__ import annotations

import asyncio
import inspect
import os
import shlex
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from sonic.computer.models import GUIAction, GUIActionType
from sonic.logger import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class BurstActionResult:
    """The observed outcome of one action in a motor burst."""

    index: int
    action: GUIAction
    success: bool
    result: Any = None
    error: str | None = None
    verification: str | None = None


@dataclass
class BurstResult:
    """Truthful summary of a burst; only dispatched actions appear in results."""

    success: bool
    completed: int
    stopped: bool
    stop_reason: str | None = None
    actions: list[BurstActionResult] = field(default_factory=list)
    elapsed_seconds: float = 0.0

    @property
    def action_results(self) -> list[BurstActionResult]:
        """Descriptive alias for callers that prefer the explicit name."""
        return self.actions


class MotorReflexes:
    """System 1 Motor Reflexes and human physical cadence engine."""

    def __init__(self, computer: Any):
        self.computer = computer

    @staticmethod
    async def _call_hook(hook: Callable[..., Any], *args: Any) -> Any:
        """Call a hook without imposing a needless callback signature."""
        try:
            parameters = inspect.signature(hook).parameters
            positional = [
                p for p in parameters.values()
                if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
            ]
        except (TypeError, ValueError):
            positional = None
        if positional is None:
            value = hook(*args)
        else:
            # The final argument is the provider's observed result.  This makes
            # one-argument verifiers useful while retaining the full form.
            count = min(len(positional), len(args))
            value = hook(*args[-count:] if count else ())
        return await value if inspect.isawaitable(value) else value

    @staticmethod
    def _failure_from_dispatch(value: Any) -> str | None:
        """Extract explicit provider failure, never infer success from a message."""
        if value is None:
            return "GUI provider returned no execution result"
        status = value.get("status") if isinstance(value, dict) else getattr(value, "status", None)
        if status is not None and str(status).upper() in {"FAILED", "FAILURE", "BLOCKED", "ERROR"}:
            return f"GUI provider status: {status}"
        exit_code = value.get("exit_code") if isinstance(value, dict) else getattr(value, "exit_code", None)
        if exit_code is not None and exit_code != 0:
            return f"GUI provider exit code: {exit_code}"
        return None

    async def execute_burst(
        self,
        workspace_id: str,
        actions: Sequence[GUIAction],
        *,
        interval_seconds: float = 0.05,
        verifier: Callable[..., Awaitable[Any] | Any] | None = None,
        observation_hook: Callable[..., Awaitable[Any] | Any] | None = None,
    ) -> BurstResult:
        """Dispatch typed GUI actions at a controlled cadence, stopping fail-closed.

        ``verifier`` and ``observation_hook`` may accept (result), (action, result),
        or (workspace_id, action, result).  A verifier must return truthy; a hook
        may return ``False`` or ``{"ok": False}`` to interrupt the burst.  When
        no hook is supplied, an available computer ``observe`` method is called
        for fresh state, but its observation is not treated as a success verdict.
        """
        if not isinstance(interval_seconds, (int, float)) or isinstance(interval_seconds, bool):
            raise ValueError("interval_seconds must be a number")
        if interval_seconds < 0 or interval_seconds > 1.0:
            raise ValueError("interval_seconds must be between 0 and 1 second")
        if actions is None or isinstance(actions, (str, bytes)):
            raise ValueError("actions must be a sequence of GUIAction values")
        action_list = list(actions)
        if any(not isinstance(action, GUIAction) for action in action_list):
            raise TypeError("every burst action must be a GUIAction")

        started = time.perf_counter()
        result = BurstResult(success=True, completed=0, stopped=False)
        for index, action in enumerate(action_list):
            if index:
                await asyncio.sleep(interval_seconds)
            try:
                dispatched = await self.computer.gui_action(workspace_id, action)
            except Exception as exc:
                detail = BurstActionResult(index, action, False, error=str(exc))
                result.actions.append(detail)
                result.success = False
                result.stopped = True
                result.stop_reason = f"execution failed: {exc}"
                break

            failure = self._failure_from_dispatch(dispatched)
            if failure:
                result.actions.append(BurstActionResult(index, action, False, dispatched, failure))
                result.success = False
                result.stopped = True
                result.stop_reason = failure
                break

            detail = BurstActionResult(index, action, True, dispatched)
            result.actions.append(detail)
            result.completed += 1

            hook = observation_hook
            if hook is not None:
                try:
                    observed = await self._call_hook(hook, workspace_id, action, dispatched)
                except Exception as exc:
                    result.actions[-1] = BurstActionResult(
                        index, action, False, dispatched, f"observation failed: {exc}",
                        "observation error",
                    )
                    result.completed -= 1
                    result.success = False
                    result.stopped = True
                    result.stop_reason = f"observation failed: {exc}"
                    break
                if observed is False or (isinstance(observed, dict) and observed.get("ok") is False):
                    result.actions[-1] = BurstActionResult(
                        index, action, False, dispatched, "unexpected state/modal change",
                        "observation rejected",
                    )
                    result.completed -= 1
                    result.success = False
                    result.stopped = True
                    result.stop_reason = "observation rejected the action"
                    break
            elif hasattr(self.computer, "observe"):
                try:
                    # Computer observation APIs conventionally take the
                    # workspace, unlike the richer optional hook contract.
                    await self._call_hook(self.computer.observe, workspace_id)
                except Exception as exc:
                    result.actions[-1] = BurstActionResult(
                        index, action, False, dispatched, f"observation failed: {exc}",
                        "observation error",
                    )
                    result.completed -= 1
                    result.success = False
                    result.stopped = True
                    result.stop_reason = f"observation failed: {exc}"
                    break
            if verifier is not None:
                try:
                    verdict = await self._call_hook(verifier, workspace_id, action, dispatched)
                except Exception as exc:
                    result.actions[-1] = BurstActionResult(
                        index, action, False, dispatched, f"verification failed: {exc}",
                        "verification error",
                    )
                    result.completed -= 1
                    result.success = False
                    result.stopped = True
                    result.stop_reason = f"verification failed: {exc}"
                    break
                if verdict is not True and not (isinstance(verdict, dict) and verdict.get("ok") is True):
                    result.actions[-1] = BurstActionResult(
                        index, action, False, dispatched, "verifier interrupted burst",
                        "verification rejected",
                    )
                    result.completed -= 1
                    result.success = False
                    result.stopped = True
                    result.stop_reason = "verifier interrupted burst"
                    break
        result.elapsed_seconds = time.perf_counter() - started
        return result

    async def _exec_cmd(self, cmd: str, workspace_id: str = "") -> tuple[int, str, str]:
        """Execute a shell command via _docker_exec or terminal."""
        if hasattr(self.computer, "_docker_exec"):
            try:
                res = await self.computer._docker_exec(cmd)
                if isinstance(res, tuple) and len(res) >= 3:
                    return res
                elif isinstance(res, tuple) and len(res) == 2:
                    return res[0], res[1], ""
            except Exception:
                pass
        if hasattr(self.computer, "terminal"):
            try:
                res = await self.computer.terminal(workspace_id, cmd)
                if hasattr(res, "exit_code"):
                    return res.exit_code, getattr(res, "stdout", ""), getattr(res, "stderr", "")
            except Exception:
                pass
        return 1, "", "No execution method available"

    def _get_display(self, workspace_id: str | None = None) -> str:
        """Resolve dynamic X11 DISPLAY for the workstation."""
        if hasattr(self.computer, "_get_display"):
            try:
                disp = self.computer._get_display(workspace_id)
                if disp:
                    return disp
            except Exception:
                pass
        return os.environ.get("DISPLAY") or ":99"

    async def _send_hotkey(self, workspace_id: str, key: str) -> None:
        """Send a keyboard shortcut or keypress to the workstation."""
        safe_key = shlex.quote(key)
        disp = self._get_display(workspace_id)
        cmd = f"DISPLAY={disp} xdotool key --clearmodifiers {safe_key}"
        if hasattr(self.computer, "_docker_exec"):
            try:
                res = await self.computer._docker_exec(cmd)
                if isinstance(res, tuple):
                    return
            except Exception:
                pass
        if hasattr(self.computer, "terminal"):
            try:
                await self.computer.terminal(workspace_id, cmd)
                return
            except Exception:
                pass
        if hasattr(self.computer, "gui_action"):
            await self.computer.gui_action(
                workspace_id, GUIAction(action=GUIActionType.KEYPRESS, key=key)
            )
            return

    async def human_type(self, workspace_id: str, text: str, delay_ms: int = 25) -> str:
        """
        Executes xdotool type --delay {delay_ms} --clearmodifiers {safe_text} via
        computer._docker_exec (or computer.gui_action). If delay_ms > 0, types with
        natural human cadence (preventing React debounce drops and anti-bot flags).
        """
        safe_text = shlex.quote(text)
        delay_arg = f"--delay {delay_ms} " if delay_ms > 0 else ""
        disp = self._get_display(workspace_id)
        cmd = f"DISPLAY={disp} xdotool type {delay_arg}--clearmodifiers {safe_text}"

        if hasattr(self.computer, "_docker_exec"):
            try:
                res = await self.computer._docker_exec(cmd)
                if isinstance(res, tuple) and len(res) >= 2:
                    out = res[1]
                    return out.strip() if out and str(out).strip() else f"typed: {text}"
            except Exception:
                pass

        if hasattr(self.computer, "gui_action"):
            await self.computer.gui_action(
                workspace_id, GUIAction(action=GUIActionType.TYPE, text=text, delay_ms=delay_ms)
            )
            return f"typed: {text}"
        elif hasattr(self.computer, "terminal"):
            try:
                res = await self.computer.terminal(workspace_id, cmd)
                out = getattr(res, "stdout", "")
                return out.strip() if out and out.strip() else f"typed: {text}"
            except Exception:
                pass
        return f"typed: {text}"

    async def hotkey_navigate(self, workspace_id: str, url: str, delay_ms: int = 25) -> str:
        """
        Focuses address bar with Ctrl + L (hotkey), types URL with human cadence,
        then presses Return.
        """
        await self._send_hotkey(workspace_id, "ctrl+l")
        await asyncio.sleep(0.05)
        await self.human_type(workspace_id, url, delay_ms=delay_ms)
        await asyncio.sleep(0.05)
        await self._send_hotkey(workspace_id, "Return")
        return f"navigated: {url}"

    async def hotkey_close_tab(self, workspace_id: str) -> str:
        """Presses Ctrl + W to close the active browser tab."""
        await self._send_hotkey(workspace_id, "ctrl+w")
        return "closed_tab"

    async def hotkey_new_tab(self, workspace_id: str, url: str = "") -> str:
        """Presses Ctrl + T, and if url provided, hotkey_navigates to it."""
        await self._send_hotkey(workspace_id, "ctrl+t")
        if url:
            await asyncio.sleep(0.05)
            return await self.hotkey_navigate(workspace_id, url)
        return "new_tab"

    async def backtrack(self, workspace_id: str, reason: str = "modal_blocked") -> str:
        """
        Human reflexive backtracking: presses Escape (to dismiss modal/dialog).
        If reason is "wrong_page", sends Alt + Left (browser back).
        """
        if reason == "wrong_page":
            await self._send_hotkey(workspace_id, "alt+Left")
        else:
            await self._send_hotkey(workspace_id, "Escape")
        return f"backtracked: {reason}"

    async def backtrack_reflex(self, workspace_id: str, reason: str = "modal_blocked") -> str:
        """Alias for backtrack."""
        return await self.backtrack(workspace_id, reason)

    async def hotkey_switch_app(self, workspace_id: str) -> str:
        """Switch active window via Alt + Tab."""
        await self._send_hotkey(workspace_id, "alt+Tab")
        return "switched_app"

    async def enforce_tab_budget(self, workspace_id: str, max_tabs: int = 3) -> int:
        """
        Checks open browser tabs via explicit `get_open_tab_count` API on `self.computer`.
        Never counts general X11 windows from `wmctrl -l` or `status.open_applications`
        and never sends `Ctrl+1` + `Ctrl+W` to close arbitrary desktop windows.
        Returns the number of excess tabs closed.
        """
        count = 0
        if hasattr(self.computer, "get_open_tab_count"):
            res = self.computer.get_open_tab_count(workspace_id)
            count = await res if asyncio.iscoroutine(res) else int(res or 0)
        elif hasattr(self.computer, "open_tabs"):
            val = self.computer.open_tabs
            count = len(val) if isinstance(val, (list, tuple, set)) else int(val or 0)
        elif hasattr(self.computer, "tab_count"):
            val = self.computer.tab_count
            count = val() if callable(val) else int(val or 0)
        else:
            return 0

        excess = max(0, count - max_tabs)
        for _ in range(excess):
            await self.hotkey_close_tab(workspace_id)
            await asyncio.sleep(0.15)
        return excess

    async def handle_gtk_file_dialog(
        self,
        workspace_id: str,
        file_path: str,
        delay_ms: int = 25,
    ) -> str:
        """
        Interacts with GTK file chooser dialogs (Browsers, GUI applications, dialogs) by:
          1. Pressing Ctrl + L to expose the direct path entry field.
          2. Typing the target file path with natural human cadence.
          3. Pressing Return to confirm file selection.
        """
        await asyncio.sleep(0.3)
        await self._send_hotkey(workspace_id, "ctrl+l")
        await asyncio.sleep(0.1)
        await self.human_type(workspace_id, file_path, delay_ms=delay_ms)
        await asyncio.sleep(0.1)
        await self._send_hotkey(workspace_id, "Return")
        return f"file_selected: {file_path}"

    async def two_stage_click(
        self,
        workspace_id: str,
        target_window: str,
        x: int,
        y: int,
        button: int = 1,
        settle_seconds: float = 0.15,
    ) -> str:
        """
        Eliminates X11 click-to-focus event swallowing:
          Stage 1: Explicitly activate/raise target_window.
          Stage 2: Settle for 150ms to allow window manager focus transition.
          Stage 3: Dispatch physical click event at coordinate (x, y).
        """
        safe_target = shlex.quote(target_window)
        disp = self._get_display(workspace_id)
        # Stage 1: Focus/raise window
        focus_cmd = (
            f"DISPLAY={disp} wmctrl -a {safe_target} || "
            f"DISPLAY={disp} xdotool search --name {safe_target} windowactivate --sync"
        )
        await self._exec_cmd(focus_cmd, workspace_id)

        # Stage 2: Settle for window manager focus transition
        if settle_seconds > 0:
            await asyncio.sleep(settle_seconds)

        # Stage 3: Physical mouse move and click (dispatch exactly once)
        if hasattr(self.computer, "gui_action"):
            try:
                act_type = GUIActionType.RIGHT_CLICK if button == 3 else GUIActionType.CLICK
                try:
                    action_obj = GUIAction(action=act_type, x=x, y=y, button=button)
                except TypeError:
                    action_obj = GUIAction(action=act_type, x=x, y=y)
                await self.computer.gui_action(
                    workspace_id,
                    action_obj,
                )
            except Exception:
                click_cmd = f"DISPLAY={disp} xdotool mousemove {x} {y} click {button}"
                await self._exec_cmd(click_cmd, workspace_id)
        else:
            click_cmd = f"DISPLAY={disp} xdotool mousemove {x} {y} click {button}"
            await self._exec_cmd(click_cmd, workspace_id)

        return f"two_stage_clicked: window={target_window} at=({x},{y})"

    async def send_application_shortcut(self, workspace_id: str, shortcut: str) -> str:
        """Send a standard keyboard shortcut (e.g. 'ctrl+r', 'f5', 'ctrl+f') to the active application."""
        await self._send_hotkey(workspace_id, shortcut)
        return f"shortcut_sent: {shortcut}"
