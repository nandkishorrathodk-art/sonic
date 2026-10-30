"""Unit Tests for Native Desktop Computer GUI Application."""
from __future__ import annotations

import pytest

# The desktop GUI needs a Tk-enabled Python *and* an X display. A bare CI runner
# has neither, and a top-level import would abort collection of the whole suite,
# so skip the module instead of failing it. importorskip() is not enough here:
# a Tk build without libtk raises ImportError from inside tkinter's own import.
try:
    import tkinter as tk
except ImportError as exc:  # pragma: no cover - environment dependent
    pytest.skip(f"tkinter unavailable: {exc}", allow_module_level=True)

try:
    _probe = tk.Tk()
    _probe.withdraw()
    _probe.destroy()
except Exception as exc:  # pragma: no cover - environment dependent
    pytest.skip(f"no usable X display for Tk: {exc}", allow_module_level=True)

from avcs import SyntheticScreen  # noqa: E402
from avcs.gui import AVCSDesktopApp  # noqa: E402


def test_desktop_gui_app_initialization_and_widgets():
    """Verify AVCSDesktopApp builds native window, widgets, and bindings."""
    root = tk.Tk()
    root.withdraw()  # keep window off-screen during headless test

    app = AVCSDesktopApp(root=root, fps=10, source=SyntheticScreen(640, 480))

    assert root.title() == "AVCS // Artificial Visual Cortex - Desktop Station"
    assert app.canvas is not None
    assert app.query_entry is not None
    assert app.btn_look is not None
    assert app.btn_steer is not None
    assert app.lbl_category is not None
    assert app.lbl_stability is not None
    assert app.lbl_surprise is not None
    assert app.lbl_latency is not None

    # Test query input change
    app.query_entry.delete(0, tk.END)
    app.query_entry.insert(0, "Close Window")
    assert app.query_entry.get() == "Close Window"

    # Test overlay toggle callbacks
    app.var_fovea.set(False)
    app.var_motion.set(False)
    app.var_pursuit.set(False)
    app._on_overlay_toggle()
    assert app.show_fovea is False
    assert app.show_motion is False
    assert app.show_pursuit is False

    app.var_top.set(True)
    app._on_top_toggle()
    assert app.always_on_top is True

    # Clean teardown
    app.on_close()

