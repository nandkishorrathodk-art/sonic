"""Minimal API example; an actual repository backend must be supplied."""
import asyncio
import sys
from pathlib import Path

# Ensure repository root is on sys.path
repo_root = str(Path(__file__).resolve().parent.parent)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from avcs_sdk import AVCSClient, ActionClass, ActionStatus
try:
    from avcs_sdk.adapters.repository_backend import RepositoryAVCSBackend
except ImportError:
    try:
        from examples.backend_template import RepositoryAVCSBackend
    except ImportError:
        from backend_template import RepositoryAVCSBackend


async def main() -> None:
    async with AVCSClient(RepositoryAVCSBackend()) as avcs:
        observation = await avcs.observe()
        targets = await avcs.find_targets("Save button", observation=observation)
        if not targets:
            raise RuntimeError("Save button not found")
        result = await avcs.click(
            target=targets[0],
            action_class=ActionClass.CONDITIONAL,
            expected_postcondition="The document save state is confirmed",
        )
        if result.status is not ActionStatus.CONFIRMED:
            raise RuntimeError(f"Save action not confirmed: {result.status.value}")


if __name__ == "__main__":
    asyncio.run(main())
