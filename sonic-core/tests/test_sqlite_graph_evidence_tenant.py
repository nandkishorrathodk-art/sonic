"""Tenant-isolation regressions for SQLite evidence lookup."""

import asyncio

from sonic.memory.schemas import ALL_TENANTS, RelationshipType
from sonic.memory.sqlite_graph import SqliteGraph


def test_find_evidence_checks_evidence_node_tenant(tmp_path):
    async def run():
        graph = SqliteGraph(db_path=str(tmp_path / "evidence_tenant.db"))
        await graph.connect()
        await graph._create_node(
            "Finding",
            {"uid": "finding-shared", "tenant_id": "tenant-A", "title": "tenant A"},
        )
        await graph._create_node(
            "Finding",
            {"uid": "finding-shared", "tenant_id": "tenant-B", "title": "tenant B"},
        )
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


def test_sqlite_graph_unscoped_reads_default_tenant_only(tmp_path):
    async def run():
        graph = SqliteGraph(db_path=str(tmp_path / "default_scope.db"))
        await graph.connect()
        await graph._create_node("Asset", {"uid": "default-asset", "name": "default"})
        await graph._create_node(
            "Asset", {"uid": "private-asset", "name": "private", "tenant_id": "tenant-private"}
        )

        implicit = await graph._get_node("Asset", "private-asset")
        default_list = await graph._list_nodes("Asset")
        all_list = await graph._list_nodes("Asset", tenant_id=ALL_TENANTS)
        await graph.disconnect()
        return implicit, default_list, all_list

    implicit, default_list, all_list = asyncio.run(run())
    assert implicit is None
    assert [node["uid"] for node in default_list] == ["default-asset"]
    assert {node["uid"] for node in all_list} == {"default-asset", "private-asset"}


def test_sqlite_graph_relationship_requires_same_tenant_endpoints(tmp_path):
    async def run():
        graph = SqliteGraph(db_path=str(tmp_path / "relationship_scope.db"))
        await graph.connect()
        await graph._create_node("Asset", {"uid": "asset", "tenant_id": "tenant-A"})
        await graph._create_node("Engagement", {"uid": "engagement", "tenant_id": "tenant-B"})
        cross_tenant = await graph.create_relationship(
            "Asset", "asset", "Engagement", "engagement", "PART_OF", tenant_id="tenant-A"
        )
        await graph.disconnect()
        return cross_tenant

    assert asyncio.run(run()) is False
