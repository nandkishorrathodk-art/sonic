"""Raw screen capture sources.

Two sources, one interface:

* :class:`ScreenCapture` grabs the real desktop with ``mss`` (fast, ~5-15 ms at
  1080p) and returns a raw ``(H, W, 3)`` uint8 array.
* :class:`SyntheticScreen` renders a scripted "desktop" in memory so the whole
  pipeline can be developed and tested on a headless box with no display.

Both hand the reflex layer nothing but pixels -- there is no other channel.
"""
from __future__ import annotations

import sys
import time
from typing import Iterator, Optional, Tuple

import numpy as np


class CaptureError(RuntimeError):
    """Raised when a real screen cannot be grabbed (no display, permissions)."""


class ScreenCapture:
    """Real desktop capture via ``mss``."""

    def __init__(self, monitor: int = 1) -> None:
        self.monitor = monitor
        self._sct = None

    def open(self) -> "ScreenCapture":
        try:
            import mss
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise CaptureError("mss is not installed; `pip install mss`") from exc
        try:
            if sys.platform == "win32":
                try:
                    import ctypes
                    u32 = ctypes.windll.user32
                    hdesk = u32.OpenInputDesktop(0, False, 0x01FF)
                    if hdesk:
                        u32.SetThreadDesktop(hdesk)
                except Exception:
                    pass
            # mss renamed the factory: `MSS` is current, `mss` is deprecated.
            factory = getattr(mss, "MSS", None) or mss.mss
            self._sct = factory()
            # Touch the monitor list so a missing display fails here, not later.
            _ = self._sct.monitors[self.monitor]
        except Exception as exc:  # pragma: no cover - depends on host display
            raise CaptureError(f"cannot open display: {exc}") from exc
        return self

    def grab(self) -> np.ndarray:
        if self._sct is None:
            self.open()
        assert self._sct is not None
        shot = self._sct.grab(self._sct.monitors[self.monitor])
        arr = np.asarray(shot)
        # mss delivers BGRA on Windows/X11; convert to genuine RGB for neural cortex
        if arr.ndim == 3 and arr.shape[-1] >= 3:
            return arr[:, :, 2::-1].copy()
        return arr.copy()

    def close(self) -> None:
        if self._sct is not None:
            try:
                self._sct.close()
            finally:
                self._sct = None

    def __enter__(self) -> "ScreenCapture":
        return self.open()

    def __exit__(self, *exc) -> None:
        self.close()

    @staticmethod
    def available() -> bool:
        """True if a real screen can actually be grabbed on this host."""
        try:
            with ScreenCapture() as cap:
                cap.grab()
            return True
        except Exception:
            return False


class SyntheticScreen:
    """A scripted desktop rendered to pixels, for headless development.

    It draws a background, a couple of "windows", and a cursor that moves each
    tick -- enough structure for layout, motion, change and overlay logic to be
    exercised without a display.
    """

    def __init__(self, width: int = 640, height: int = 360, seed: int = 7) -> None:
        self.width = width
        self.height = height
        self.rng = np.random.default_rng(seed)
        self.tick = 0

    def grab(self) -> np.ndarray:
        h, w = self.height, self.width
        img = np.zeros((h, w, 3), dtype=np.uint8)
        img[..., 0] = np.linspace(24, 72, w, dtype=np.uint8)[None, :]
        img[..., 1] = np.linspace(24, 56, h, dtype=np.uint8)[:, None]
        img[..., 2] = 48

        # Two static "windows".
        img[int(h * 0.15):int(h * 0.55), int(w * 0.10):int(w * 0.45)] = 205
        img[int(h * 0.30):int(h * 0.85), int(w * 0.52):int(w * 0.92)] = 120

        # A sliding "window" large enough for tile-level motion to register.
        wx = int(w * 0.05 + (w * 0.35) * (0.5 + 0.5 * np.sin(self.tick * 0.25)))
        img[int(h * 0.60):int(h * 0.85), wx:wx + int(w * 0.18)] = 235

        # A moving cursor (the thing motion detection must find).
        cx = int(w * 0.5 + 0.35 * w * np.sin(self.tick * 0.35))
        cy = int(h * 0.5 + 0.3 * h * np.cos(self.tick * 0.35))
        r = 6
        y0, y1 = max(0, cy - r), min(h, cy + r)
        x0, x1 = max(0, cx - r), min(w, cx + r)
        img[y0:y1, x0:x1] = 255

        # A blinking overlay rectangle in the corner on odd ticks.
        if self.tick % 2 == 1:
            img[0:24, w - 120:w] = 250

        self.tick += 1
        return img

    def stream(self, count: int, interval: float = 0.0) -> Iterator[np.ndarray]:
        for _ in range(count):
            yield self.grab()
            if interval:
                time.sleep(interval)


class FrameSource:
    """Wrap any user-supplied frame provider (webcam, video, remote stream).

    ``provider`` must return an ``(H, W, 3)`` uint8 array per call.  This is the
    seam for non-screen sources without the pipeline knowing anything about
    them.
    """

    def __init__(self, provider) -> None:
        self.provider = provider

    def grab(self) -> np.ndarray:
        frame = self.provider()
        arr = np.asarray(frame)
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        if arr.shape[-1] == 4:
            arr = arr[:, :, :3]
        return arr.astype(np.uint8, copy=False)

    def close(self) -> None:
        close = getattr(self.provider, "close", None)
        if callable(close):
            close()


class Win32FastCapture:
    """High-speed native Windows desktop capture via GDI DIBSection.

    Avoids DWM GPU readback stall and redundant Python conversions by
    memory-mapping a top-down 32-bit DIB directly into a numpy buffer.
    Supports native resolution capture as well as hardware-accelerated StretchBlt scaling.
    """

    def __init__(
        self,
        monitor: int = 1,
        target_size: Optional[Tuple[int, int]] = None,
    ) -> None:
        self.monitor = monitor
        self.target_size = target_size
        self._user32 = None
        self._gdi32 = None
        self._h_screen_dc = None
        self._h_mem_dc = None
        self._h_bitmap = None
        self._bits_ptr = None
        self._mem_view = None
        self._raw_np: Optional[np.ndarray] = None
        self._src_x = 0
        self._src_y = 0
        self._src_w = 0
        self._src_h = 0
        self._dst_w = 0
        self._dst_h = 0
        self._h_desk = None
        self._is_open = False

    def open(self) -> "Win32FastCapture":
        if sys.platform != "win32":
            raise CaptureError("Win32FastCapture is only supported on Windows")
        try:
            import ctypes
            from ctypes import wintypes
        except ImportError as exc:
            raise CaptureError("ctypes is required for Win32FastCapture") from exc

        self._user32 = ctypes.windll.user32
        self._gdi32 = ctypes.windll.gdi32

        self._h_desk = None
        # Ensure thread desktop access
        try:
            hdesk = self._user32.OpenInputDesktop(0, False, 0x01FF)
            if hdesk:
                self._h_desk = hdesk
                self._user32.SetThreadDesktop(hdesk)
        except Exception:
            pass

        self._src_w = self._user32.GetSystemMetrics(0)
        self._src_h = self._user32.GetSystemMetrics(1)
        self._src_x = 0
        self._src_y = 0

        if self.target_size is not None:
            self._dst_w, self._dst_h = self.target_size
        else:
            self._dst_w, self._dst_h = self._src_w, self._src_h

        class BITMAPINFOHEADER(ctypes.Structure):
            _fields_ = [
                ("biSize", wintypes.DWORD),
                ("biWidth", wintypes.LONG),
                ("biHeight", wintypes.LONG),
                ("biPlanes", wintypes.WORD),
                ("biBitCount", wintypes.WORD),
                ("biCompression", wintypes.DWORD),
                ("biSizeImage", wintypes.DWORD),
                ("biXPelsPerMeter", wintypes.LONG),
                ("biYPelsPerMeter", wintypes.LONG),
                ("biClrUsed", wintypes.DWORD),
                ("biClrImportant", wintypes.DWORD),
            ]

        class BITMAPINFO(ctypes.Structure):
            _fields_ = [("bmiHeader", BITMAPINFOHEADER), ("bmiColors", wintypes.DWORD * 3)]

        bmi = BITMAPINFO()
        bmi.bmiHeader.biSize = ctypes.sizeof(BITMAPINFOHEADER)
        bmi.bmiHeader.biWidth = self._dst_w
        bmi.bmiHeader.biHeight = -self._dst_h  # Negative height = top-down DIB
        bmi.bmiHeader.biPlanes = 1
        bmi.bmiHeader.biBitCount = 32
        bmi.bmiHeader.biCompression = 0  # BI_RGB

        self._h_screen_dc = self._user32.GetDC(0)
        if not self._h_screen_dc:
            raise CaptureError("Failed to acquire Windows screen DC")

        self._h_mem_dc = self._gdi32.CreateCompatibleDC(self._h_screen_dc)
        if not self._h_mem_dc:
            self._user32.ReleaseDC(0, self._h_screen_dc)
            raise CaptureError("Failed to create Windows compatible DC")

        self._bits_ptr = ctypes.c_void_p()
        self._h_bitmap = self._gdi32.CreateDIBSection(
            self._h_mem_dc,
            ctypes.byref(bmi),
            0,
            ctypes.byref(self._bits_ptr),
            None,
            0,
        )
        if not self._h_bitmap or not self._bits_ptr:
            self._cleanup_handles()
            raise CaptureError("Failed to allocate Win32 DIBSection")

        self._gdi32.SelectObject(self._h_mem_dc, self._h_bitmap)

        buffer_from_memory = ctypes.pythonapi.PyMemoryView_FromMemory
        buffer_from_memory.restype = ctypes.py_object
        buffer_from_memory.argtypes = [ctypes.c_void_p, ctypes.c_ssize_t, ctypes.c_int]

        self._mem_view = buffer_from_memory(self._bits_ptr, self._dst_w * self._dst_h * 4, 0x200)
        self._raw_np = np.frombuffer(self._mem_view, dtype=np.uint8).reshape((self._dst_h, self._dst_w, 4))
        self._is_open = True
        return self

    def grab(self) -> np.ndarray:
        if not self._is_open:
            self.open()
        assert self._raw_np is not None
        SRCCOPY = 0x00CC0020
        if self._dst_w == self._src_w and self._dst_h == self._src_h:
            ret = self._gdi32.BitBlt(
                self._h_mem_dc,
                0,
                0,
                self._dst_w,
                self._dst_h,
                self._h_screen_dc,
                self._src_x,
                self._src_y,
                SRCCOPY,
            )
        else:
            HALFTONE = 4
            self._gdi32.SetStretchBltMode(self._h_mem_dc, HALFTONE)
            self._gdi32.SetBrushOrgEx(self._h_mem_dc, 0, 0, None)
            ret = self._gdi32.StretchBlt(
                self._h_mem_dc,
                0,
                0,
                self._dst_w,
                self._dst_h,
                self._h_screen_dc,
                self._src_x,
                self._src_y,
                self._src_w,
                self._src_h,
                SRCCOPY,
            )
        if not ret:
            raise CaptureError("Win32 BitBlt/StretchBlt failed")
        self._gdi32.GdiFlush()
        # 32-bit DIB memory order is BGRA -> convert to RGB via zero-strided copy
        return self._raw_np[:, :, 2::-1].copy()

    def _cleanup_handles(self) -> None:
        if self._gdi32 and self._h_bitmap:
            try:
                self._gdi32.DeleteObject(self._h_bitmap)
            except Exception:
                pass
            self._h_bitmap = None
        if self._gdi32 and self._h_mem_dc:
            try:
                self._gdi32.DeleteDC(self._h_mem_dc)
            except Exception:
                pass
            self._h_mem_dc = None
        if self._user32 and self._h_screen_dc:
            try:
                self._user32.ReleaseDC(0, self._h_screen_dc)
            except Exception:
                pass
            self._h_screen_dc = None
        if self._user32 and getattr(self, "_h_desk", None):
            try:
                self._user32.CloseDesktop(self._h_desk)
            except Exception:
                pass
            self._h_desk = None
        self._is_open = False

    def close(self) -> None:
        self._cleanup_handles()

    def __enter__(self) -> "Win32FastCapture":
        return self.open()

    def __exit__(self, *exc) -> None:
        self.close()

    @staticmethod
    def available() -> bool:
        if sys.platform != "win32":
            return False
        try:
            with Win32FastCapture() as cap:
                arr = cap.grab()
                return arr.ndim == 3 and arr.shape[2] == 3
        except Exception:
            return False


class ThreadedScreenCapture:
    """High-frequency asynchronous double-buffered frame capture worker.

    Decouples frame acquisition from the neural perception loop so that
    calling `.grab()` returns the latest captured desktop frame in sub-0.01 ms.
    """

    def __init__(self, source: object, target_fps: float = 60.0) -> None:
        self.source = source
        self.target_fps = target_fps
        self._running = False
        import threading
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._latest_frame: Optional[np.ndarray] = None
        self._seq: int = 0
        self._last_grab_time: float = 0.0
        self._last_error: Optional[Exception] = None

    def open(self) -> "ThreadedScreenCapture":
        if hasattr(self.source, "open") and callable(self.source.open):
            try:
                self.source.open()
            except Exception:
                pass
        self._latest_frame = self.source.grab()
        self._running = True
        import threading
        self._thread = threading.Thread(target=self._worker, daemon=True, name="AVCS-CaptureWorker")
        self._thread.start()
        return self

    def _worker(self) -> None:
        if sys.platform == "win32":
            try:
                import ctypes
                u32 = ctypes.windll.user32
                hdesk = u32.OpenInputDesktop(0, False, 0x01FF)
                if hdesk:
                    u32.SetThreadDesktop(hdesk)
            except Exception:
                pass
        sleep_interval = 1.0 / max(1.0, self.target_fps * 1.2) if self.target_fps > 0 else 0.005
        while self._running:
            try:
                frame = self.source.grab()
                now = time.perf_counter()
                with self._lock:
                    self._latest_frame = frame
                    self._seq += 1
                    self._last_grab_time = now
                    self._last_error = None
            except Exception as exc:
                with self._lock:
                    self._last_error = exc
            time.sleep(sleep_interval)

    def grab(self) -> np.ndarray:
        if not self._running or self._latest_frame is None:
            self.open()
        with self._lock:
            if self._last_error is not None:
                raise self._last_error
            return self._latest_frame

    def close(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=0.5)
            self._thread = None
        if hasattr(self.source, "close") and callable(self.source.close):
            self.source.close()

    def trigger_motion(self, x: float, y: float) -> None:
        """Forwards motion triggers to underlying source if supported."""
        if hasattr(self.source, "trigger_motion") and callable(self.source.trigger_motion):
            self.source.trigger_motion(x, y)

    def __enter__(self) -> "ThreadedScreenCapture":
        return self.open()

    def __exit__(self, *exc) -> None:
        self.close()

    @property
    def sequence_number(self) -> int:
        with self._lock:
            return self._seq


def open_capture(
    prefer_real: bool = True,
    monitor: int = 1,
    threaded: bool = False,
    target_size: Optional[Tuple[int, int]] = None,
) -> object:
    """Return a real capture if a display exists, else a synthetic one.

    If `threaded=True`, wraps the capture source in a high-speed background
    double-buffered worker so `.grab()` is non-blocking (<0.01ms).
    """
    cap = None
    if prefer_real:
        if sys.platform == "win32" and Win32FastCapture.available():
            try:
                cap = Win32FastCapture(monitor=monitor, target_size=target_size).open()
            except Exception:
                cap = None
        if cap is None and ScreenCapture.available():
            try:
                cap = ScreenCapture(monitor=monitor).open()
            except Exception:
                cap = None

    if cap is None:
        cap = SyntheticScreen()

    if threaded:
        return ThreadedScreenCapture(cap).open()
    return cap

