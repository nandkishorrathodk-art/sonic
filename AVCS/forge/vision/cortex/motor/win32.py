"""Native Windows Win32 OS Cursor Driver.

Hardened hardware-level cursor steering using Windows user32.dll & SendInput.
Features Per-Monitor V2 DPI awareness, high-precision timers, and multi-monitor support.
"""
from __future__ import annotations

import atexit
import ctypes
from ctypes import wintypes
import time
from typing import List, Set, Tuple

from forge.vision.cortex.motor.failsafe import CornerBreakoutFailSafe

# Input Types
INPUT_MOUSE = 0
INPUT_KEYBOARD = 1
INPUT_HARDWARE = 2

# Win32 Mouse Event Flags
MOUSEEVENTF_MOVE = 0x0001
MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004
MOUSEEVENTF_RIGHTDOWN = 0x0008
MOUSEEVENTF_RIGHTUP = 0x0010
MOUSEEVENTF_MIDDLEDOWN = 0x0020
MOUSEEVENTF_MIDDLEUP = 0x0040
MOUSEEVENTF_ABSOLUTE = 0x8000
MOUSEEVENTF_VIRTUALDESK = 0x4000

# Win32 Keyboard Event Flags
KEYEVENTF_EXTENDEDKEY = 0x0001
KEYEVENTF_KEYUP = 0x0002
KEYEVENTF_UNICODE = 0x0004
KEYEVENTF_SCANCODE = 0x0008

# Standard Virtual-Key Codes
VK_BACK = 0x08
VK_TAB = 0x09
VK_RETURN = 0x0D
VK_SHIFT = 0x10
VK_CONTROL = 0x11
VK_MENU = 0x12  # Alt
VK_ESCAPE = 0x1B
VK_SPACE = 0x20
VK_PRIOR = 0x21  # Page Up
VK_NEXT = 0x22   # Page Down
VK_END = 0x23
VK_HOME = 0x24
VK_LEFT = 0x25
VK_UP = 0x26
VK_RIGHT = 0x27
VK_DOWN = 0x28
VK_INSERT = 0x2D
VK_DELETE = 0x2E

EXTENDED_VKS = {
    VK_LEFT, VK_UP, VK_RIGHT, VK_DOWN,
    VK_DELETE, VK_INSERT, VK_HOME, VK_END,
    VK_PRIOR, VK_NEXT,
}

KEY_NAME_TO_VK = {
    "enter": VK_RETURN,
    "return": VK_RETURN,
    "tab": VK_TAB,
    "space": VK_SPACE,
    "backspace": VK_BACK,
    "back": VK_BACK,
    "delete": VK_DELETE,
    "del": VK_DELETE,
    "insert": VK_INSERT,
    "ins": VK_INSERT,
    "home": VK_HOME,
    "end": VK_END,
    "pageup": VK_PRIOR,
    "pgup": VK_PRIOR,
    "pagedown": VK_NEXT,
    "pgdn": VK_NEXT,
    "escape": VK_ESCAPE,
    "esc": VK_ESCAPE,
    "up": VK_UP,
    "down": VK_DOWN,
    "left": VK_LEFT,
    "right": VK_RIGHT,
    "shift": VK_SHIFT,
    "ctrl": VK_CONTROL,
    "control": VK_CONTROL,
    "alt": VK_MENU,
    "win": 0x5B,
    "windows": 0x5B,
    "cmd": 0x5B,
}

# Register function keys F1 through F24
for _i in range(1, 25):
    KEY_NAME_TO_VK[f"f{_i}"] = 0x70 + _i - 1

SM_CXSCREEN = 0
SM_CYSCREEN = 1
SM_XVIRTUALSCREEN = 76
SM_YVIRTUALSCREEN = 77
SM_CXVIRTUALSCREEN = 78
SM_CYVIRTUALSCREEN = 79

# Ctypes 64-bit SendInput Structures
ULONG_PTR = ctypes.c_size_t


class MOUSEINPUT(ctypes.Structure):
    _fields_ = [
        ("dx", wintypes.LONG),
        ("dy", wintypes.LONG),
        ("mouseData", wintypes.DWORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ULONG_PTR),
    ]


class KEYBDINPUT(ctypes.Structure):
    _fields_ = [
        ("wVk", wintypes.WORD),
        ("wScan", wintypes.WORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ULONG_PTR),
    ]


class HARDWAREINPUT(ctypes.Structure):
    _fields_ = [
        ("uMsg", wintypes.DWORD),
        ("wU56SDFGHJKLLKJUTREIOOIMNBVYUITYUL", wintypes.WORD),
        ("wParamH", wintypes.WORD),
    ]


class _INPUT_UNION(ctypes.Union):
    _fields_ = [
        ("mi", MOUSEINPUT),
        ("ki", KEYBDINPUT),
        ("hi", HARDWAREINPUT),
    ]


class INPUT(ctypes.Structure):
    _anonymous_ = ("_u",)
    _fields_ = [
        ("type", wintypes.DWORD),
        ("_u", _INPUT_UNION),
    ]


class Win32CursorDriver:
    """Hardware cursor driver for Microsoft Windows with SendInput & DPI awareness."""

    def __init__(self, failsafe: bool = True) -> None:
        self.user32 = ctypes.windll.user32
        self._attach_input_desktop()
        self._init_dpi_awareness()
        self._setup_ctypes_signatures()
        self._init_high_res_timer()
        self.failsafe = CornerBreakoutFailSafe(enabled=failsafe)
        self._active_keys: Set[int] = set()
        atexit.register(self.panic_release)

    def _attach_input_desktop(self) -> None:
        """Attaches current thread to interactive input desktop to enable physical input routing."""
        try:
            h_desk = self.user32.OpenInputDesktop(0, False, 0x01FF)
            if h_desk:
                self.user32.SetThreadDesktop(h_desk)
        except Exception:
            pass

    def _init_dpi_awareness(self) -> None:
        """Enables Per-Monitor V2 DPI awareness to prevent coordinate scaling drift."""
        try:
            # DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2 = -4
            self.user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))
        except Exception:
            try:
                ctypes.windll.shcore.SetProcessDpiAwareness(2)
            except Exception:
                try:
                    self.user32.SetProcessDPIAware()
                except Exception:
                    pass

    def _setup_ctypes_signatures() -> None:
        pass

    def _setup_ctypes_signatures(self) -> None:
        """Declares strict 64-bit argument and return types for user32 APIs."""
        self.user32.GetSystemMetrics.argtypes = [ctypes.c_int]
        self.user32.GetSystemMetrics.restype = ctypes.c_int

        self.user32.GetCursorPos.argtypes = [ctypes.POINTER(wintypes.POINT)]
        self.user32.GetCursorPos.restype = wintypes.BOOL

        self.user32.SetCursorPos.argtypes = [ctypes.c_int, ctypes.c_int]
        self.user32.SetCursorPos.restype = wintypes.BOOL

        self.user32.SendInput.argtypes = [wintypes.UINT, ctypes.POINTER(INPUT), ctypes.c_int]
        self.user32.SendInput.restype = wintypes.UINT

    def _init_high_res_timer(self) -> None:
        """Requests 1ms OS scheduling resolution via winmm.dll."""
        try:
            self.winmm = ctypes.windll.winmm
            self.winmm.timeBeginPeriod.argtypes = [wintypes.UINT]
            self.winmm.timeBeginPeriod.restype = wintypes.UINT
            self.winmm.timeEndPeriod.argtypes = [wintypes.UINT]
            self.winmm.timeEndPeriod.restype = wintypes.UINT
            self.winmm.timeBeginPeriod(1)
        except Exception:
            self.winmm = None

    def __del__(self) -> None:
        if getattr(self, "winmm", None):
            try:
                self.winmm.timeEndPeriod(1)
            except Exception:
                pass

    @property
    def screen_size(self) -> Tuple[int, int]:
        """Returns primary display resolution in physical pixels: (width, height)."""
        width = int(self.user32.GetSystemMetrics(SM_CXSCREEN))
        height = int(self.user32.GetSystemMetrics(SM_CYSCREEN))
        return (width, height)

    @property
    def virtual_screen_rect(self) -> Tuple[int, int, int, int]:
        """Returns virtual desktop bounds (x, y, width, height) across all monitors."""
        vx = int(self.user32.GetSystemMetrics(SM_XVIRTUALSCREEN))
        vy = int(self.user32.GetSystemMetrics(SM_YVIRTUALSCREEN))
        vw = int(self.user32.GetSystemMetrics(SM_CXVIRTUALSCREEN))
        vh = int(self.user32.GetSystemMetrics(SM_CYVIRTUALSCREEN))
        return (vx, vy, vw, vh)

    def get_position_pixels(self) -> Tuple[int, int]:
        """Returns current physical mouse coordinates in screen pixels."""
        pt = wintypes.POINT()
        self.user32.GetCursorPos(ctypes.byref(pt))
        return (int(pt.x), int(pt.y))

    def get_position_normalized(self) -> Tuple[float, float]:
        """Returns mouse position normalized to primary display [0.0, 1.0]."""
        px, py = self.get_position_pixels()
        w, h = self.screen_size
        nx = max(0.0, min(1.0, px / max(1, w - 1)))
        ny = max(0.0, min(1.0, py / max(1, h - 1)))
        return (nx, ny)

    def check_hardware_failsafe(self) -> None:
        """Queries physical mouse position and verifies safety fail-safe."""
        hw_x, hw_y = self.get_position_pixels()
        w, h = self.screen_size
        self.failsafe.check(hw_x, hw_y, w, h)

    def move_to_pixels(self, x: int, y: int) -> None:
        """Moves cursor to absolute pixel coordinates with fail-safe guard."""
        # 1. Guard against user breakout before dispatching movement
        self.check_hardware_failsafe()

        w, h = self.screen_size
        clamped_x = max(0, min(w - 1, x))
        clamped_y = max(0, min(h - 1, y))

        # 2. Guard against commanding into breakout corner
        self.failsafe.check(clamped_x, clamped_y, w, h)
        ret = bool(self.user32.SetCursorPos(clamped_x, clamped_y))
        if not ret:
            import threading
            def _th_move() -> None:
                h_desk = self.user32.OpenInputDesktop(0, False, 0x01FF)
                if h_desk:
                    self.user32.SetThreadDesktop(h_desk)
                self.user32.SetCursorPos(clamped_x, clamped_y)
            th = threading.Thread(target=_th_move, name="AVCS-CursorMoveWorker")
            th.start()
            th.join(timeout=0.5)

    def move_to_normalized(self, nx: float, ny: float) -> None:
        w, h = self.screen_size
        px = int(round(nx * (w - 1)))
        py = int(round(ny * (h - 1)))
        self.move_to_pixels(px, py)

    def move_relative(self, dx: int, dy: int) -> None:
        curr_x, curr_y = self.get_position_pixels()
        self.move_to_pixels(curr_x + dx, curr_y + dy)

    def _send_mouse_input(self, flags: int, data: int = 0) -> None:
        """Dispatches serialized mouse event via SendInput."""
        inp = INPUT()
        inp.type = INPUT_MOUSE
        inp.mi = MOUSEINPUT(
            dx=0,
            dy=0,
            mouseData=data,
            dwFlags=flags,
            time=0,
            dwExtraInfo=0,
        )
        ret = self.user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(INPUT))
        if ret == 0:
            import threading
            def _th_mouse() -> None:
                h_desk = self.user32.OpenInputDesktop(0, False, 0x01FF)
                if h_desk:
                    self.user32.SetThreadDesktop(h_desk)
                self.user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(INPUT))
            th = threading.Thread(target=_th_mouse, name="AVCS-MouseInputWorker")
            th.start()
            th.join(timeout=0.5)

    def mouse_down(self, button: str = "left") -> None:
        self.check_hardware_failsafe()
        flag = (
            MOUSEEVENTF_RIGHTDOWN if button == "right" else
            MOUSEEVENTF_MIDDLEDOWN if button == "middle" else
            MOUSEEVENTF_LEFTDOWN
        )
        self._send_mouse_input(flag)

    def mouse_up(self, button: str = "left") -> None:
        flag = (
            MOUSEEVENTF_RIGHTUP if button == "right" else
            MOUSEEVENTF_MIDDLEUP if button == "middle" else
            MOUSEEVENTF_LEFTUP
        )
        self._send_mouse_input(flag)

    def click(self, button: str = "left", hold_duration: float = 0.02) -> None:
        self.check_hardware_failsafe()
        self.mouse_down(button=button)
        if hold_duration > 0:
            time.sleep(hold_duration)
        self.mouse_up(button=button)

    def double_click(self, button: str = "left", interval: float = 0.08) -> None:
        self.click(button=button)
        time.sleep(interval)
        self.click(button=button)

    def _send_key_input(self, vk: int = 0, scan: int = 0, flags: int = 0) -> None:
        """Dispatches serialized keyboard event via SendInput."""
        if vk in EXTENDED_VKS:
            flags |= KEYEVENTF_EXTENDEDKEY
        inp = INPUT()
        inp.type = INPUT_KEYBOARD
        inp.ki = KEYBDINPUT(
            wVk=vk,
            wScan=scan,
            dwFlags=flags,
            time=0,
            dwExtraInfo=0,
        )
        ret = self.user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(INPUT))
        if ret == 0:
            import threading
            def _th_key() -> None:
                h_desk = self.user32.OpenInputDesktop(0, False, 0x01FF)
                if h_desk:
                    self.user32.SetThreadDesktop(h_desk)
                self.user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(INPUT))
            th = threading.Thread(target=_th_key, name="AVCS-KeyInputWorker")
            th.start()
            th.join(timeout=0.5)

    def _resolve_vk_with_shift(self, vk_or_name: int | str) -> Tuple[int, bool]:
        if isinstance(vk_or_name, int):
            return vk_or_name, False
        name = str(vk_or_name).lower().strip()
        if name in KEY_NAME_TO_VK:
            return KEY_NAME_TO_VK[name], False
        if len(name) == 1:
            if name.isalnum():
                return ord(name.upper()), False
            # Fallback to VkKeyScanW for symbols and punctuation
            res = self.user32.VkKeyScanW(ord(name))
            if res != -1:
                vk = res & 0xFF
                shift_state = (res >> 8) & 0xFF
                needs_shift = bool(shift_state & 0x01)
                return vk, needs_shift
        raise ValueError(f"Unknown key name: '{vk_or_name}'")

    def _resolve_vk(self, vk_or_name: int | str) -> int:
        vk, _ = self._resolve_vk_with_shift(vk_or_name)
        return vk

    def key_down(self, vk_or_name: int | str) -> None:
        self.check_hardware_failsafe()
        vk = self._resolve_vk(vk_or_name)
        self._send_key_input(vk=vk, flags=0)
        self._active_keys.add(vk)

    def key_up(self, vk_or_name: int | str) -> None:
        vk = self._resolve_vk(vk_or_name)
        self._send_key_input(vk=vk, flags=KEYEVENTF_KEYUP)
        self._active_keys.discard(vk)

    def press_key(self, vk_or_name: int | str, hold_duration: float = 0.02) -> None:
        self.check_hardware_failsafe()
        vk, needs_shift = self._resolve_vk_with_shift(vk_or_name)
        if needs_shift:
            self.key_down(VK_SHIFT)
            time.sleep(0.01)
        try:
            self._send_key_input(vk=vk, flags=0)
            self._active_keys.add(vk)
            if hold_duration > 0:
                time.sleep(hold_duration)
            self._send_key_input(vk=vk, flags=KEYEVENTF_KEYUP)
            self._active_keys.discard(vk)
        finally:
            if needs_shift:
                time.sleep(0.01)
                self.key_up(VK_SHIFT)

    def press_hotkey(self, *keys: int | str, delay_between: float = 0.02) -> None:
        """Presses multiple keys simultaneously in sequence, then releases in reverse order."""
        self.check_hardware_failsafe()
        vks = [self._resolve_vk(k) for k in keys]
        pressed: List[int] = []
        try:
            for vk in vks:
                self.key_down(vk)
                pressed.append(vk)
                if delay_between > 0:
                    time.sleep(delay_between)
        finally:
            for vk in reversed(pressed):
                try:
                    self.key_up(vk)
                except Exception:
                    pass
                if delay_between > 0:
                    time.sleep(delay_between)

    def type_char(self, char: str, dwell_ms: float = 5.0) -> None:
        """Sends a single Unicode character (including 32-bit emojis/surrogate pairs) to active window."""
        self.check_hardware_failsafe()
        # Encode to UTF-16-LE to handle BMP characters and surrogate pairs for SMP characters (emojis)
        utf16_bytes = char.encode("utf-16-le")
        code_units = [int.from_bytes(utf16_bytes[i:i+2], "little") for i in range(0, len(utf16_bytes), 2)]
        for cu in code_units:
            # Unicode KeyDown
            self._send_key_input(vk=0, scan=cu, flags=KEYEVENTF_UNICODE)
            if dwell_ms > 0:
                time.sleep(dwell_ms / 1000.0)
            # Unicode KeyUp
            self._send_key_input(vk=0, scan=cu, flags=KEYEVENTF_UNICODE | KEYEVENTF_KEYUP)

    def type_text(
        self,
        text: str,
        key_delay_ms: int = 25,
        clear_first: bool = False,
        press_enter: bool = False,
        literal_mode: bool = False,
    ) -> None:
        """Types string directly into active focus using SendInput Unicode events.
        
        Args:
            text: Text string to type.
            key_delay_ms: Inter-character delay in ms (default 25ms to satisfy modern WinUI/TSF event loops).
            clear_first: Select all (Ctrl+A) and delete before typing.
            press_enter: Press Enter after finishing typing.
            literal_mode: When True, prevents interpreting escape characters (e.g. \\n, \\t) as Enter/Tab.
        """
        self.check_hardware_failsafe()
        if clear_first:
            self.press_hotkey("ctrl", "a")
            time.sleep(0.04)
            self.press_key("backspace")
            time.sleep(0.04)

        delay_sec = max(0.0, key_delay_ms / 1000.0)
        if literal_mode:
            for ch in text:
                self.check_hardware_failsafe()
                self.type_char(ch)
                if delay_sec > 0:
                    time.sleep(delay_sec)
        else:
            normalized_text = text.replace("\r\n", "\n").replace("\r", "\n")
            for ch in normalized_text:
                self.check_hardware_failsafe()
                if ch == "\n":
                    self.press_key(VK_RETURN)
                elif ch == "\t":
                    self.press_key(VK_TAB)
                elif ch == "\b":
                    self.press_key(VK_BACK)
                else:
                    self.type_char(ch)
                if delay_sec > 0:
                    time.sleep(delay_sec)

        if press_enter:
            time.sleep(0.04)
            self.press_key(VK_RETURN)

    def force_foreground_window(self, hwnd: int, timeout: float = 3.0) -> bool:
        """Robustly brings a window to the foreground on Windows 11, bypassing foreground lock restrictions."""
        if not hwnd or not self.user32.IsWindow(hwnd):
            return False

        t_end = time.perf_counter() + timeout
        while time.perf_counter() < t_end:
            fg_hwnd = self.user32.GetForegroundWindow()
            if fg_hwnd == hwnd:
                return True

            # 1. Un-minimize if iconic
            if self.user32.IsIconic(hwnd):
                self.user32.ShowWindow(hwnd, 9)  # SW_RESTORE
                time.sleep(0.05)

            # 2. Simulate Alt-key poke to gain foreground rights
            self.user32.keybd_event(VK_MENU, 0, 0, 0)
            self.user32.keybd_event(VK_MENU, 0, KEYEVENTF_KEYUP, 0)

            # 3. AttachThreadInput bypass
            cur_thread = ctypes.windll.kernel32.GetCurrentThreadId()
            fg_thread = self.user32.GetWindowThreadProcessId(fg_hwnd, None)
            target_thread = self.user32.GetWindowThreadProcessId(hwnd, None)

            attached_fg = False
            attached_target = False
            try:
                if fg_thread and fg_thread != cur_thread:
                    attached_fg = bool(self.user32.AttachThreadInput(cur_thread, fg_thread, True))
                if target_thread and target_thread != cur_thread:
                    attached_target = bool(self.user32.AttachThreadInput(cur_thread, target_thread, True))

                self.user32.SetForegroundWindow(hwnd)
                self.user32.BringWindowToTop(hwnd)
                self.user32.ShowWindow(hwnd, 5)  # SW_SHOW
            finally:
                if attached_fg:
                    self.user32.AttachThreadInput(cur_thread, fg_thread, False)
                if attached_target:
                    self.user32.AttachThreadInput(cur_thread, target_thread, False)

            time.sleep(0.1)
            if self.user32.GetForegroundWindow() == hwnd:
                return True

        return self.user32.GetForegroundWindow() == hwnd

    def assert_foreground_window(self, hwnd: int) -> None:
        """Fail-closed assertion: raises RuntimeError if target hwnd does not own foreground focus."""
        current = self.user32.GetForegroundWindow()
        if current != hwnd:
            raise RuntimeError(
                f"Fail-closed focus guard violation: Target HWND {hwnd} is not foreground (current foreground: {current})."
            )

    def panic_release(self) -> None:
        """Emergency release for all currently held keys and mouse buttons."""
        for flag in (MOUSEEVENTF_LEFTUP, MOUSEEVENTF_RIGHTUP, MOUSEEVENTF_MIDDLEUP):
            try:
                self._send_mouse_input(flag)
            except Exception:
                pass
        for vk in list(self._active_keys):
            try:
                self._send_key_input(vk=vk, flags=KEYEVENTF_KEYUP)
            except Exception:
                pass
        self._active_keys.clear()

    def close(self) -> None:
        """Closes driver and safely releases any held hardware inputs."""
        self.panic_release()

