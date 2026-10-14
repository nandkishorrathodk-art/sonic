"""Runnable AVCS repository-backed SDK demonstration: python examples/repository_backed_demo.py"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

# Bootstrap repository root into sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from avcs_sdk import (
    ActionClass,
    ActionStatus,
    AVCSClient,
    NormalizedPoint,
    RepositoryAVCSBackend,
)


async def main() -> None:
    backend = RepositoryAVCSBackend()
    print("=== AVCS Python SDK: Repository Backend Live Demo ===")

    async with AVCSClient(backend) as client:
        # 1. Capture screen observation
        obs = await client.observe()
        print(f"[Observe] ID={obs.observation_id} | Resolution={obs.width}x{obs.height} | Space={obs.coordinate_space}")

        # 2. Ground natural language target
        targets = await client.find_targets("Save button", observation=obs)
        target = targets[0]
        print(f"[Ground] Target={target.label} | Bounds=({target.bounds.x:.3f}, {target.bounds.y:.3f}, {target.bounds.width:.3f}, {target.bounds.height:.3f}) | Conf={target.confidence:.2f}")

        # 3. Execute bounded click through ActionBroker with verified evidence
        result = await client.click(
            target=target,
            action_class=ActionClass.IDEMPOTENT,
            expected_postcondition="Submit state change",
        )

        print(f"[Result] Status={result.status.value.upper()} | Duration={result.duration_ms:.2f}ms")
        if result.evidence:
            print(f"[Evidence] Verified={result.evidence.verified} | Method={result.evidence.method}")
            print(f"[Details] {result.evidence.details}")

        assert result.status == ActionStatus.CONFIRMED, "Action was not confirmed!"
        print("=== Demo Complete: 100% verified execution through ActionBroker ===")


if __name__ == "__main__":
    asyncio.run(main())
