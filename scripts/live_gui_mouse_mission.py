"""Live GUI mission that exercises the MOUSE through the real run_mission loop.

Uses a scripted router (no live LLM key needed) to drive the real
observe -> reason -> ground -> act -> observe machinery and prove the mouse
actually moves/clicks on the real Docker workstation desktop.

Independent evidence: xdotool cursor position, wmctrl window list, and the
AVCS raw pixel frame (not the agent's self-reported status).

Run:  python scripts/live_gui_mouse_mission.py
"""

from __future__ import annotations

import asyncio
import hashlib
import sys

sys.path.insert(0, "services/core")

from sonic.computer.docker_computer import DockerComputerProvider
from sonic.computer_use.agent import ComputerUseAgent


class StubRouter:
    def __init__(self, script: list[str]):
        self.script = script
        self.i = 0
        self.prompts: list[str] = []

    async def complete(self, req):
        from sonic.llm.schemas import LLMResponse

        msgs = getattr(req, "messages", []) or []
        self.prompts.append("\n".join(str(getattr(m, "content", "")) for m in msgs))
        resp = self.script[min(self.i, len(self.script) - 1)]
        self.i += 1
        return LLMResponse(content=resp)


async def cursor(comp: DockerComputerProvider) -> tuple[int, int]:
    _, out, _ = await comp._docker_exec("DISPLAY=:99 xdotool getmouselocation --shell")
    x = y = -1
    for line in out.splitlines():
        if line.startswith("X="):
            x = int(line[2:])
        elif line.startswith("Y="):
            y = int(line[2:])
    return x, y


def md5(obs) -> str:
    return hashlib.md5(obs.raw_rgb).hexdigest()[:10]


async def main() -> int:
    comp = DockerComputerProvider()
    wid = comp.container_name
    await comp.create("default", "default")
    await comp.cleanup_desktop_windows(wid)
    await asyncio.sleep(1.0)

    script = [
        'ACTION: APP_LAUNCH\nTARGET: thunar\nPAYLOAD: {"app_name": "thunar"}\nEXPECTED: a Thunar window opens',
        'ACTION: GUI_MOVE\nTARGET: \nPAYLOAD: {"x": 500, "y": 300}\nEXPECTED: the mouse cursor glides to (500, 300)',
        'ACTION: GUI_CLICK\nTARGET: \nPAYLOAD: {"x": 140, "y": 260, "postcondition": {"screen_changed": true}}\nEXPECTED: clicking a real widget in the file manager',
        "ACTION: GOAL_COMPLETE\nTARGET: goal_complete\nPAYLOAD: {}\nEXPECTED: hand off for verification",
    ]
    router = StubRouter(script)
    agent = ComputerUseAgent(
        computer_provider=comp,
        llm_router=router,
        max_actions=10,
        execution_plane="gui",
        gui_only=True,
    )

    pre_cursor = await cursor(comp)
    traces = await agent.run_mission(
        wid,
        goal="Open the file manager, glide the mouse, and click within its window.",
        steps=6,
    )
    post_cursor = await cursor(comp)

    print("===== MISSION TRACES =====")
    for t in traces:
        print(
            f"  step={getattr(t,'step_index','?')} action={getattr(t,'action_type','?')} "
            f"status={getattr(t,'status','?')}"
        )
        res = getattr(t, "result", "") or getattr(t, "actual_observation", "") or ""
        print(f"      result: {str(res)[:160]}")

    _, wl, _ = await comp._docker_exec("DISPLAY=:99 wmctrl -l | grep -ci thunar")
    _, loc, _ = await comp._docker_exec("DISPLAY=:99 xdotool getmouselocation --shell")
    print("\n===== INDEPENDENT CHECK =====")
    print(f"  cursor {pre_cursor} -> {post_cursor}")
    print(f"  thunar windows = {wl.strip()}")
    print(f"  raw mouselocation = {loc.strip().replace(chr(10), ' ')}")

    last = router.prompts[-1] if router.prompts else ""
    print("\n===== PROMPT CONTAINS =====")
    print(f"  'avcs'={'avcs' in last.lower()}  len={len(last)}")

    await comp.cleanup_desktop_windows(wid)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
