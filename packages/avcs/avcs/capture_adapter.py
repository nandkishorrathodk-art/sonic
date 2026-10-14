"""AVCS Milestone 1: Standardized Capture Abstraction & Display Geometry Engine.

Supports interchangeable capture backends (Win32 GDI, DXGI, Synthetic)
with multi-monitor virtual screen offsets, Per-Monitor DPI awareness,
and typed error handling.
"""
from __future__ import annotations

import ctypes
import os
import platform
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class CaptureError(Exception):
    """Base exception for screen capture failures."""


class CaptureTimeoutError(CaptureError):
    """Raised when frame acquisition exceeds deadline."""


class DisplayLockedError(CaptureError):
    """Raised when desktop session is locked or UAC secure desktop is active."""


class DPIChangedError(CaptureError):
    """Raised when monitor DPI or display geometry mutates mid-session."""


@dataclass(frozen=True)
class DisplayGeometry:
    """Normalized multi-monitor coordinate frame geometry."""
    virtual_left: int
    virtual_top: int
    virtual_width: int
    virtual_height: int
    dpi_scale: float
    is_primary: bool

    def to_normalized(self, screen_x: int, screen_y: int) -> Tuple[float, float]:
        """Translates absolute OS screen coordinates into normalized [0, 1] frame."""
        nx = (screen_x - self.virtual_left) / max(1, self.virtual_width)
        ny = (screen_y - self.virtual_top) / max(1, self.virtual_height)
        return float(np.clip(nx, 0.0, 1.0)), float(np.clip(ny, 0.0, 1.0))

    def to_screen(self, norm_x: float, norm_y: float) -> Tuple[int, int]:
        """Translates normalized [0, 1] coordinates into absolute OS screen coordinates."""
        sx = int(self.virtual_left + norm_x * self.virtual_width)
        sy = int(self.virtual_top + norm_y * self.virtual_height)
        return sx, sy


class BaseCaptureAdapter(ABC):
    """Standardized abstract interface for screen capture implementations."""

    @abstractmethod
    def open(self) -> BaseCaptureAdapter:
        """Initializes display context and allocates hardware buffers."""

    @abstractmethod
    def grab(self) -> np.ndarray:
        """Acquires the latest uncommitted frame as RGB array (H, W, 3)."""

    @abstractmethod
    def close(self) -> None:
        """Releases all graphics and desktop handles cleanly."""

    @abstractmethod
    def get_geometry(self) -> DisplayGeometry:
        """Returns the current multi-monitor virtual geometry."""

    @abstractmethod
    def supported_modes(self) -> List[str]:
        """Returns supported capture features."""

    def __enter__(self) -> BaseCaptureAdapter:
        return self.open()

    def __exit__(self, *exc: Any) -> None:
        self.close()


class Win32GDICaptureAdapter(BaseCaptureAdapter):
    """Native Windows GDI capture adapter with DPI awareness and virtual screen support."""

    def __init__(self, target_size: Optional[Tuple[int, int]] = None) -> None:
        if sys.platform != "win32":
            raise CaptureError("Win32GDICaptureAdapter requires Windows OS.")
        self.target_size = target_size
        self._user32 = ctypes.windll.user32
        self._gdi32 = ctypes.windll.gdi32

        # Set Process DPI Awareness (Per-Monitor v2 if supported)
        try:
            # DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2 = -4
            self._user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))
        except Exception:
            pass

        self._is_open = False
        self._h_desk = None
        self._h_screen_dc = None
        self._h_mem_dc = None
        self._h_bitmap = None
        self._raw_np: Optional[np.ndarray] = None
        self._geometry: Optional[DisplayGeometry] = None

    def get_geometry(self) -> DisplayGeometry:
        # SM_XVIRTUALSCREEN = 76, SM_YVIRTUALSCREEN = 77
        # SM_CXVIRTUALSCREEN = 78, SM_CYVIRTUALSCREEN = 79
        vx = self._user32.GetSystemMetrics(76)
        vy = self._user32.GetSystemMetrics(77)
        vw = self._user32.GetSystemMetrics(78)
        vh = self._user32.GetSystemMetrics(79)

        if vw == 0 or vh == 0:
            vx, vy = 0, 0
            vw = self._user32.GetSystemMetrics(0)
            vh = self._user32.GetSystemMetrics(1)

        # Base DPI scale (96 DPI is 100%)
        dpi = 96
        try:
            dpi = self._user32.GetDpiForSystem()
        except Exception:
            pass
        scale = float(dpi) / 96.0

        return DisplayGeometry(
            virtual_left=vx,
            virtual_top=vy,
            virtual_width=vw,
            virtual_height=vh,
            dpi_scale=scale,
            is_primary=True,
        )

    def open(self) -> Win32GDICaptureAdapter:
        from ctypes import wintypes

        try:
            hdesk = self._user32.OpenInputDesktop(0, False, 0x01FF)
            if hdesk:
                self._h_desk = hdesk
                self._user32.SetThreadDesktop(hdesk)
        except Exception:
            pass

        self._geometry = self.get_geometry()
        src_w = self._geometry.virtual_width
        src_h = self._geometry.virtual_height

        if self.target_size:
            dst_w, dst_h = self.target_size
        else:
            dst_w, dst_h = src_w, src_h

        self._dst_w = dst_w
        self._dst_h = dst_h

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
        bmi.bmiHeader.biWidth = dst_w
        bmi.bmiHeader.biHeight = -dst_h
        bmi.bmiHeader.biPlanes = 1
        bmi.bmiHeader.biBitCount = 32
        bmi.bmiHeader.biCompression = 0

        self._h_screen_dc = self._user32.GetDC(0)
        if not self._h_screen_dc:
            raise DisplayLockedError("Failed to acquire Windows screen DC.")

        self._h_mem_dc = self._gdi32.CreateCompatibleDC(self._h_screen_dc)
        if not self._h_mem_dc:
            self.close()
            raise CaptureError("Failed to create compatible DC.")

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
            self.close()
            raise CaptureError("Failed to allocate Win32 DIBSection buffer.")

        self._h_old_bitmap = self._gdi32.SelectObject(self._h_mem_dc, self._h_bitmap)

        buffer_from_memory = ctypes.pythonapi.PyMemoryView_FromMemory
        buffer_from_memory.restype = ctypes.py_object
        buffer_from_memory.argtypes = [ctypes.c_void_p, ctypes.c_ssize_t, ctypes.c_int]
        self._mem_view = buffer_from_memory(self._bits_ptr, dst_w * dst_h * 4, 0x200)
        self._raw_np = np.frombuffer(self._mem_view, dtype=np.uint8).reshape((dst_h, dst_w, 4))
        self._is_open = True
        return self

    def grab(self) -> np.ndarray:
        if not self._is_open:
            self.open()
        assert self._raw_np is not None
        SRCCOPY = 0x00CC0020
        geo = self._geometry or self.get_geometry()

        if self._dst_w == geo.virtual_width and self._dst_h == geo.virtual_height:
            ret = self._gdi32.BitBlt(
                self._h_mem_dc, 0, 0, self._dst_w, self._dst_h,
                self._h_screen_dc, geo.virtual_left, geo.virtual_top, SRCCOPY,
            )
        else:
            HALFTONE = 4
            self._gdi32.SetStretchBltMode(self._h_mem_dc, HALFTONE)
            self._gdi32.SetBrushOrgEx(self._h_mem_dc, 0, 0, None)
            ret = self._gdi32.StretchBlt(
                self._h_mem_dc, 0, 0, self._dst_w, self._dst_h,
                self._h_screen_dc, geo.virtual_left, geo.virtual_top,
                geo.virtual_width, geo.virtual_height, SRCCOPY,
            )

        if not ret:
            if self._user32 and self._h_screen_dc:
                try:
                    self._user32.ReleaseDC(0, self._h_screen_dc)
                except Exception:
                    pass
                self._h_screen_dc = self._user32.GetDC(0)
                if self._dst_w == geo.virtual_width and self._dst_h == geo.virtual_height:
                    ret = self._gdi32.BitBlt(
                        self._h_mem_dc, 0, 0, self._dst_w, self._dst_h,
                        self._h_screen_dc, geo.virtual_left, geo.virtual_top, SRCCOPY,
                    )
                else:
                    ret = self._gdi32.StretchBlt(
                        self._h_mem_dc, 0, 0, self._dst_w, self._dst_h,
                        self._h_screen_dc, geo.virtual_left, geo.virtual_top,
                        geo.virtual_width, geo.virtual_height, SRCCOPY,
                    )

        if not ret:
            # When Tkinter or external GUI loops corrupt the calling thread's Win32 DC table,
            # execute capture from an isolated thread where the desktop DC is pristine.
            import threading
            thread_res: dict[str, Any] = {}

            def _thread_grab() -> None:
                try:
                    t_cap = Win32GDICaptureAdapter(target_size=self.target_size)
                    t_cap.open()
                    t_geo = t_cap._geometry or t_cap.get_geometry()
                    t_ret = False
                    if t_cap._dst_w == t_geo.virtual_width and t_cap._dst_h == t_geo.virtual_height:
                        t_ret = bool(t_cap._gdi32.BitBlt(
                            t_cap._h_mem_dc, 0, 0, t_cap._dst_w, t_cap._dst_h,
                            t_cap._h_screen_dc, t_geo.virtual_left, t_geo.virtual_top, SRCCOPY,
                        ))
                    else:
                        t_cap._gdi32.SetStretchBltMode(t_cap._h_mem_dc, 4)
                        t_cap._gdi32.SetBrushOrgEx(t_cap._h_mem_dc, 0, 0, None)
                        t_ret = bool(t_cap._gdi32.StretchBlt(
                            t_cap._h_mem_dc, 0, 0, t_cap._dst_w, t_cap._dst_h,
                            t_cap._h_screen_dc, t_geo.virtual_left, t_geo.virtual_top,
                            t_geo.virtual_width, t_geo.virtual_height, SRCCOPY,
                        ))
                    if t_ret and t_cap._raw_np is not None:
                        t_cap._gdi32.GdiFlush()
                        thread_res["frame"] = t_cap._raw_np[:, :, 2::-1].copy()
                    t_cap.close()
                except Exception as exc:
                    thread_res["err"] = exc

            th = threading.Thread(target=_thread_grab, name="AVCS-ThreadGDIWorker")
            th.start()
            th.join(timeout=2.0)
            if "frame" in thread_res:
                return thread_res["frame"]

            # Fallback to physical desktop capture via ScreenCapture on DC starvation
            try:
                from forge.vision.capture import ScreenCapture
                with ScreenCapture() as sct:
                    shot = sct.grab()
                    if self.target_size and (shot.shape[1] != self._dst_w or shot.shape[0] != self._dst_h):
                        from PIL import Image
                        im = Image.fromarray(shot).resize((self._dst_w, self._dst_h))
                        return np.array(im)
                    return shot
            except Exception:
                pass
            raise CaptureError("Win32 BitBlt/StretchBlt failed.")
        self._gdi32.GdiFlush()
        return self._raw_np[:, :, 2::-1].copy()

    def close(self) -> None:
        if self._gdi32 and self._h_mem_dc and hasattr(self, "_h_old_bitmap") and self._h_old_bitmap:
            try:
                self._gdi32.SelectObject(self._h_mem_dc, self._h_old_bitmap)
            except Exception:
                pass
            self._h_old_bitmap = None
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
        if self._user32 and self._h_desk:
            try:
                self._user32.CloseDesktop(self._h_desk)
            except Exception:
                pass
            self._h_desk = None
        self._is_open = False

    def supported_modes(self) -> List[str]:
        return ["multi_monitor_virtual", "gdi_fast_dib", "halftone_stretch", "per_monitor_dpi"]


class SyntheticCaptureAdapter(BaseCaptureAdapter):
    """Deterministic synthetic capture adapter for unit tests and CI."""

    def __init__(self, width: int = 640, height: int = 480) -> None:
        self.width = width
        self.height = height
        self._geometry = DisplayGeometry(0, 0, width, height, 1.0, True)
        self._tick = 0

    def open(self) -> SyntheticCaptureAdapter:
        return self

    def grab(self) -> np.ndarray:
        self._tick += 1
        frame = np.full((self.height, self.width, 3), 40, dtype=np.uint8)
        # Render a simple animated element
        cx = int(self.width * 0.5 + 50 * np.sin(self._tick * 0.1))
        cy = int(self.height * 0.5 + 30 * np.cos(self._tick * 0.1))
        frame[max(0, cy - 10):min(self.height, cy + 10), max(0, cx - 10):min(self.width, cx + 10)] = [255, 120, 0]
        return frame

    def close(self) -> None:
        pass

    def get_geometry(self) -> DisplayGeometry:
        return self._geometry

    def supported_modes(self) -> List[str]:
        return ["synthetic_replay", "deterministic_motion"]


class InteractiveSyntheticCaptureAdapter(SyntheticCaptureAdapter):
    """Synthetic capture adapter that supports interactive visual state pulses."""

    def __init__(self, width: int = 640, height: int = 480) -> None:
        super().__init__(width=width, height=height)
        self._motion_patch: Optional[Tuple[int, int, int]] = None

    def trigger_motion(self, x: float, y: float) -> None:
        px = int(round(x * (self.width - 1)))
        py = int(round(y * (self.height - 1)))
        self._motion_patch = (px, py, 32)

    def grab(self) -> np.ndarray:
        self._tick += 1
        frame = np.full((self.height, self.width, 3), 40, dtype=np.uint8)
        # High-contrast UI grid texture ensuring non-zero spatial gradients for optical flow
        frame[::8, :] = 180
        frame[:, ::8] = 180
        if self._motion_patch:
            px, py, r = self._motion_patch
            self._motion_patch = None
            y1 = max(0, py - r)
            y2 = min(self.height, py + r)
            x1 = max(0, px - r)
            x2 = min(self.width, px + r)
            frame[y1:y2, x1:x2] = 255 - frame[y1:y2, x1:x2]
        return frame

