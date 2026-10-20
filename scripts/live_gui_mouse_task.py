"""Live, human-like mouse + GUI acceptance against the real Docker workstation.

Drives the real DockerComputerProvider and the real ComputerUseAgent loop with a
scripted router (no live LLM needed) to exercise:
  - APP_LAUNCH (verified window appears)
  - GUI_MOVE  (cursor physically glides, verified)
  - GUI_CLICK (coordinate click on a real widget -> scene change)
  - GUI_TYPE  (blocked fail-closed without a focus-verification head)

Every claim is checked against independent evidence (cursor position, window
list, pixel scene hash) rather than trusting the agent's own status.
"""

from __future__ import annotations

import asyncio
import hashlib
import sys

sys.path.insert(0, "services/core")

from sonic.computer.docker_computer import DockerComputerProvider
from sonic.computer_use.agent import ComputerUseAgent
from sonic.computer_use.models import ComputerActionType

MARKER = "@@RESULT@@"


def md5(obs) -> str:
    return hashlib.md5(obs.raw_rgb).hexdigest()[:10]


async def cursor(comp: DockerComputerProvider, wid: str) -> tuple[int, int]:
    _, out, _ = await comp._docker_exec("DISPLAY=:99 xdotool getmouselocation --shell")
    x = y = -1
    for line in out.splitlines():
        if line.startswith("X="):
            x = int(line[2:])
        elif line.startswith("Y="):
            y = int(line[2:])
    return x, y


def log(*args) -> None:
    print(MARKER, *args, flush=True)


async def main() -> None:
    comp = DockerComputerProvider()
    wid = comp.container_name
    await comp.create("default", "default")
    await comp.cleanup_desktop_windows(wid)
    await asyncio.sleep(1.0)

    agent = ComputerUseAgent(
        computer_provider=comp, execution_plane="gui", gui_only=True, max_actions=10
    )

    # 1. Launch a real GUI app and confirm a window actually mapped.
    launch = await agent.execute_action(
        wid, ComputerActionType.APP_LAUNCH, "thunar", {"app_name": "thunar"}, "open thunar"
    )
    _, wl, _ = await comp._docker_exec("DISPLAY=:99 wmctrl -l | grep -ci thunar")
    log("APP_LAUNCH", launch.status.value, "| real thunar windows =", wl.strip())
    await asyncio.sleep(1.0)

    # 2. Prime a fresh observation, then glide the mouse with GUI_MOVE.
    await agent.observe(wid)
    start = await cursor(comp, wid)
    move = await agent.execute_action(
        wid, ComputerActionType.GUI_MOVE, "", {"x": 640, "y": 400}, "move mouse"
    )
    end = await cursor(comp, wid)
    log("GUI_MOVE", move.status.value, f"| cursor {start} -> {end} (want (640, 400))")

    # 3. Coordinate click on the file-manager sidebar (Places pane).
    await agent.observe(wid)
    pre = await comp.screenshot(wid)
    click = await agent.execute_action(
        wid, ComputerActionType.GUI_CLICK, "", {"x": 120, "y": 250}, "click sidebar item"
    )
    await asyncio.sleep(1.2)
    post = await comp.screenshot(wid)
    log(
        "GUI_CLICK",
        click.status.value,
        "| scene changed =",
        md5(pre) != md5(post),
        f"{md5(pre)} -> {md5(post)}",
    )

    # 4. Type into the focused field (expected fail-closed without a focus head).
    typing = await agent.execute_action(
        wid, ComputerActionType.GUI_TYPE, "", {"x": 120, "y": 250, "text": "hello"}, "type"
    )
    log("GUI_TYPE", typing.status.value, "|", str(typing.actual_observation)[:90])

    # 5. Semantic click without a configured grounder (expected fail-closed).
    await agent.observe(wid)
    semantic = await agent.execute_action(
        wid, ComputerActionType.GUI_CLICK, "the Back button", {}, "semantic click"
    )
    log("GUI_CLICK semantic", semantic.status.value, "|", str(semantic.actual_observation)[:90])

    await comp.cleanup_desktop_windows(wid)


if __name__ == "__main__":
    asyncio.run(main())
