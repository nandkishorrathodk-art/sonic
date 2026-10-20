"""Live GUI acceptance probe: drive the REAL DockerComputerProvider like the CUA agent.

Not a unit test. Requires a running `sonic-desktop-workstation` container.
Run:  python scripts/live_gui_acceptance.py
"""
from __future__ import annotations

import asyncio
import base64
import sys
import time

sys.path.insert(0, "services/core")

from sonic.computer.docker_computer import DockerComputerProvider  # noqa: E402
from sonic.computer.models import (  # noqa: E402
    GUIAction,
    GUIActionType,
)


def _png_size(b64: str | None) -> tuple[int, int] | None:
    if not b64:
        return None
    raw = base64.b64decode(b64)
    if raw[:8] != b"\x89PNG\r\n\x1a\n":
        return None
    import struct

    w, h = struct.unpack(">II", raw[16:24])
    return w, h


async def main() -> int:
    comp = DockerComputerProvider()
    wid = comp.container_name
    print(f"[*] provider container = {wid}")
    await comp.create("default", "default", )
    state = await comp.status(wid)
    print(f"[*] status = {state.status if hasattr(state, 'status') else state}")

    # ---- 1. OBSERVE (fresh frame, as the agent would) ----
    t0 = time.perf_counter()
    obs = await comp.screenshot(wid)
    dt = (time.perf_counter() - t0) * 1000
    size = _png_size(obs.screenshot_base64)
    print(f"[1] screenshot: {dt:.0f} ms  png_size={size}  state={obs.desktop_state} "
          f"age={obs.capture_age_ms}ms  active_window={obs.active_window!r}")
    print(f"    raw_rgb bytes={len(obs.raw_rgb)}  dims={obs.width}x{obs.height}  frame_id={obs.frame_id}")
    print(f"    avcs_state keys: {list((obs.avcs_state or {}).keys())}")
    print(f"    visible_text (first 120): {(obs.visible_text or '')[:120]!r}")
    print(f"    detected_controls: {len(obs.detected_controls or [])}")

    # ---- 2. LAUNCH a non-browser app (thunar) ----
    ok = await comp.launch_application(wid, "thunar")
    print(f"[2] launch thunar -> {ok}")
    await asyncio.sleep(1.5)
    obs2 = await comp.screenshot(wid)
    print(f"    after launch, active_window={obs2.active_window!r} title=?")

    # ---- 3. terminal exec (headless execution plane, same container) ----
    code, out, err = await comp._docker_exec("echo SONIC_LIVE_OK && uname -s")
    print(f"[3] exec echo -> exit={code} out={out.strip()!r}")

    # ---- 4. GUI TYPE into the terminal via real gui_action ----
    # Focus the terminal window first
    await comp.gui_action(wid, GUIAction(action=GUIActionType.SELECT_WINDOW, app_name="thunar"))
    await asyncio.sleep(0.5)
    obs3 = await comp.screenshot(wid)
    active = getattr(obs3, "active_window", None)
    print(f"[4] active_window after focus = {active}")

    # Type a marker command + Enter, then verify via the persisted shell output
    type_res = await comp.gui_action(
        wid, GUIAction(action=GUIActionType.TYPE, text="echo HUMAN_LIKE_TYPED_$((6*7))")
    )
    print(f"    TYPE dispatched, desktop_state={getattr(type_res, 'desktop_state', None)}")
    await asyncio.sleep(0.3)
    key_res = await comp.gui_action(wid, GUIAction(action=GUIActionType.KEYPRESS, key="Return"))
    await asyncio.sleep(1.0)
    print(f"    KEYPRESS Return dispatched, state={getattr(key_res, 'desktop_state', None)}")

    # ---- 5. OBSERVE post-action, check the terminal window title changed/still live ----
    obs4 = await comp.screenshot(wid)
    print(f"[5] post-action active_window={obs4.active_window!r} "
          f"visible_text_tail={(obs4.visible_text or '')[-160:]!r}")

    # ---- 6. Human-visible cursor MOVE (smooth glide) ----
    await comp.gui_action(wid, GUIAction(action=GUIActionType.MOVE, x=960, y=540))
    pos = await comp._docker_exec("DISPLAY=:99 xdotool getmouselocation")
    print(f"[6] cursor after MOVE -> {pos[1].strip()!r}")

    # ---- 7. Screenshot pixels actually changed after typing? (visual delta proxy) ----
    import hashlib

    def md5(b) -> str:
        return hashlib.md5(b if isinstance(b, bytes) else (b or "").encode()).hexdigest()[:10]

    pre = obs3.raw_rgb or obs3.screenshot_base64
    post = obs4.raw_rgb or obs4.screenshot_base64
    print(f"[7] frame hashes: pre-type={md5(pre)} post-type={md5(post)}  "
          f"changed={md5(pre) != md5(post)}  (raw pre={len(obs3.raw_rgb)}B post={len(obs4.raw_rgb)}B)")
    print(f"    avcs scene_changed pre={obs3.avcs_state.get('scene_changed')} "
          f"post={obs4.avcs_state.get('scene_changed')} "
          f"change_fraction={obs4.avcs_state.get('change_fraction')}")

    # Save the final frame so a human can eyeball it
    if obs4.raw_rgb:
        import numpy as np
        from PIL import Image

        arr = np.frombuffer(obs4.raw_rgb, dtype=np.uint8).reshape(obs4.height, obs4.width, 3)
        Image.fromarray(arr).save("/tmp/sonic_live_final.png")
        print("[*] final raw frame saved to /tmp/sonic_live_final.png")
    elif obs4.screenshot_base64:
        with open("/tmp/sonic_live_final.png", "wb") as fh:
            fh.write(base64.b64decode(obs4.screenshot_base64))
        print("[*] final png frame saved to /tmp/sonic_live_final.png")

    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
