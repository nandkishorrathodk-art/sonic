"""Live GUI mission: drive the REAL ComputerUseAgent loop against the REAL
Docker workststation, with a scripted router that stands in for the LLM brain.

This isolates the observe -> reason -> ground -> act -> observe machinery from
the (unconfigured) live LLM credentials. It is the honest way to test the CUA
path end-to-end when no vision LLM key is present.

Run:  python scripts/live_gui_mission.py
"""
from __future__ import annotations

import asyncio
import sys

sys.path.insert(0, "services/core")

from sonic.computer.docker_computer import DockerComputerProvider  # noqa: E402
from sonic.computer_use.agent import ComputerUseAgent  # noqa: E402


class StubRouter:
    """Returns a fixed script of ACTION/TARGET/PAYLOAD/EXPECTED blocks, one per call."""

    def __init__(self, script: list[str]):
        self.script = script
        self.i = 0
        self.prompts: list[str] = []

    async def complete(self, req):
        from sonic.llm.schemas import LLMResponse

        # capture the last user message for inspection
        try:
            msgs = getattr(req, "messages", []) or []
            self.prompts.append("\n".join(str(getattr(m, "content", "")) for m in msgs))
        except Exception:
            pass
        resp = self.script[min(self.i, len(self.script) - 1)]
        self.i += 1
        return LLMResponse(content=resp)


async def main() -> int:
    comp = DockerComputerProvider()
    wid = comp.container_name
    await comp.create("default", "default")

    # Clean desktop first
    await comp.cleanup_desktop_windows(wid)
    await asyncio.sleep(1.0)

    script = [
        # step 1: launch a real installed GUI app (In thunar) via visible desktop
        'ACTION: APP_LAUNCH\nTARGET: thunar\nPAYLOAD: {"app_name": "thunar"}\nEXPECTED: a Thunar file manager window opens on the desktop',
        # step 2: focus it explicitly
        'ACTION: APP_FOCUS\nTARGET: thunar\nPAYLOAD: {"app_name": "thunar"}\nEXPECTED: the Thunar window is active',
        # step 3: give up control - let the loop report completion
        "ACTION: GOAL_COMPLETE\nTARGET: goal_complete\nPAYLOAD: {}\nEXPECTED: hand off for independent verification",
    ]
    router = StubRouter(script)

    agent = ComputerUseAgent(
        computer_provider=comp,
        llm_router=router,
        max_actions=10,
        execution_plane="gui",
        gui_only=True,
    )

    traces = await agent.run_mission(
        wid,
        goal="Open the Thunar file manager on the desktop and confirm its window is visible.",
        steps=6,
    )

    print("\n===== MISSION TRACES =====")
    for t in traces:
        print(f"  step={getattr(t,'step_index','?')} action={getattr(t,'action_type','?')} "
              f"status={getattr(t,'status','?')} target={getattr(t,'target','')!r}")
        res = getattr(t, "result", "") or getattr(t, "reason", "")
        print(f"      result: {str(res)[:200]}")

    # Independent verification: is a Thunar window really on screen?
    code, out, _ = await comp._docker_exec(
        "DISPLAY=:99 wmctrl -l | grep -i thunar"
    )
    print(f"\n===== INDEPENDENT CHECK =====\n  thunar windows -> {out.strip()!r}")
    print(f"  VERDICT: {'PASS' if out.strip() else 'FAIL'}")

    # Did the agent receive the AVCS state + a fresh frame in its prompt?
    last = router.prompts[-1] if router.prompts else ""
    print("\n===== PROMPT CONTAINS =====\n"
          f"  'avcs'/{'avcs' in last.lower()}  "
          f"'scene_hash'={'scene_hash' in last}  "
          f"'change_fraction'={'change_fraction' in last}  "
          f"len={len(last)}")

    await comp.cleanup_desktop_windows(wid)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
