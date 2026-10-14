"""Runnable in-memory example: python examples/demo.py"""
import asyncio
import sys
from pathlib import Path

# Ensure repository root is on sys.path
repo_root = str(Path(__file__).resolve().parent.parent)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from avcs_sdk import AVCSClient, ActionClass, ActionStatus
try:
    from examples.demo_backend import DemoBackend
except ImportError:
    from demo_backend import DemoBackend


async def main() -> None:
    async with AVCSClient(DemoBackend()) as avcs:
        observation = await avcs.observe()
        targets = await avcs.find_targets("Save button", observation=observation)
        if not targets:
            raise RuntimeError("No target found")
        result = await avcs.click(
            target=targets[0],
            action_class=ActionClass.CONDITIONAL,
            expected_postcondition="The in-memory demo save state is true",
        )
        print(f"status={result.status.value}, verified={bool(result.evidence and result.evidence.verified)}")
        if result.status is not ActionStatus.CONFIRMED:
            raise RuntimeError(result.message or "Demo action did not confirm")


if __name__ == "__main__":
    asyncio.run(main())
