import asyncio

import pytest

from sonic.memory.graph import GraphMemory
from sonic.memory.inmemory import InMemoryGraph
from sonic.memory.schemas import ALL_TENANTS


def test_in_memory_graph_defaults_unscoped_reads_to_default_tenant():
    async def run():
        graph = InMemoryGraph()
        await graph._create_node("Asset", {"uid": "shared", "tenant_id": "tenant-a"})
        await graph._create_node("Asset", {"uid": "default", "name": "default"})
        return (
            await graph._get_node("Asset", "shared"),
            await graph._get_node("Asset", "shared", tenant_id=ALL_TENANTS),
            await graph.query("ignored"),
        )

    scoped, all_tenants, default_query = asyncio.run(run())
    assert scoped is None
    assert all_tenants["tenant_id"] == "tenant-a"
    assert [node["uid"] for node in default_query] == ["default"]


def test_in_memory_graph_rejects_cross_tenant_relationship_and_tenant_mutation():
    async def run():
        graph = InMemoryGraph()
        await graph._create_node("Asset", {"uid": "asset", "tenant_id": "tenant-a"})
        await graph._create_node("Engagement", {"uid": "eng", "tenant_id": "tenant-b"})
        relation = await graph.create_relationship(
            "Asset", "asset", "Engagement", "eng", "PART_OF", tenant_id="tenant-a"
        )
        mutation = await graph._update_node(
            "Asset", "asset", {"tenant_id": "tenant-b"}, tenant_id="tenant-a"
        )
        wrong_label_delete = await graph._delete_node(
            "Engagement", "asset", tenant_id="tenant-a"
        )
        return relation, mutation, wrong_label_delete

    assert asyncio.run(run()) == (False, False, False)


class _FakeResult:
    async def single(self):
        return {"props": {"uid": "asset", "tenant_id": "default"}}

    async def data(self):
        return []


class _FakeSession:
    def __init__(self, calls):
        self.calls = calls

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_exc):
        return None

    async def run(self, query, *args, **kwargs):
        self.calls.append((query, args, kwargs))
        return _FakeResult()


class _FakeDriver:
    def __init__(self):
        self.calls = []

    def session(self):
        return _FakeSession(self.calls)


def test_neo4j_graph_crud_is_tenant_scoped_by_default():
    async def run():
        graph = GraphMemory()
        graph._driver = _FakeDriver()
        node = await graph._get_node("Asset", "asset")
        return node, graph._driver.calls

    node, calls = asyncio.run(run())
    query, _args, params = calls[0]
    assert node["uid"] == "asset"
    assert "n.tenant_id = $tenant_id" in query
    assert params["tenant_id"] == "default"


def test_neo4j_graph_raw_cypher_requires_explicit_all_tenant_scope():
    async def run():
        graph = GraphMemory()
        graph._driver = _FakeDriver()
        with pytest.raises(PermissionError, match="all-tenant"):
            await graph.query("MATCH (n) RETURN n")
        return await graph.query("MATCH (n) RETURN n", tenant_id=ALL_TENANTS)

    assert asyncio.run(run()) == []
