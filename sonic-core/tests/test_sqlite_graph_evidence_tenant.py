"""Tenant-isolation regressions for SQLite evidence lookup."""

import asyncio

from sonic.memory.schemas import RelationshipType
from sonic.memory.sqlite_graph import SqliteGraph


def test_find_evidence_checks_evidence_node_tenant(tmp_path):
    async def run():
        graph = SqliteGraph(db_path=str(tmp_path / "evidence_tenant.db"))
        await graph.connect()
        await graph._create_node(
            "Evidence",
            {
                "uid": "evidence-owned-by-b",
                "tenant_id": "tenant-B",
                "finding_id": "finding-shared",
                "content": "tenant B evidence",
            },
        )
        await graph.create_relationship(
            "Finding",
            "finding-shared",
            "Evidence",
            "evidence-owned-by-b",
            RelationshipType.HAS_EVIDENCE,
            tenant_id="tenant-A",
        )
        await graph._create_node(
            "Evidence",
            {
                "uid": "evidence-owned-by-a",
                "tenant_id": "tenant-A",
                "finding_id": "finding-shared",
                "content": "tenant A evidence",
            },
        )
        await graph.create_relationship(
            "Finding",
            "finding-shared",
            "Evidence",
            "evidence-owned-by-a",
            RelationshipType.HAS_EVIDENCE,
            tenant_id="tenant-A",
        )

        result = await graph.find_evidence("finding-shared", tenant_id="tenant-A")
        await graph.disconnect()
        return result

    results = asyncio.run(run())
    assert [result["uid"] for result in results] == ["evidence-owned-by-a"]
