"""
SONIC-REDA — Multi-Tenant In-Memory Graph Store (Dev/Test Fallback)
=====================================================================
Drop-in replacement for Neo4j GraphMemory that works without external dependencies.
Enforces multi-tenant isolation across all node queries and relationships.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

from sonic.logger import get_logger
from sonic.memory.schemas import (
    ALL_TENANTS,
    AgentNode,
    AssetNode,
    EngagementNode,
    EvidenceNode,
    FindingNode,
    HypothesisNode,
    RelationshipType,
    TechniqueNode,
)

logger = get_logger(__name__)


def _in_scope(node: dict[str, Any], tenant_id: str | None) -> bool:
    return tenant_id == ALL_TENANTS or node.get("tenant_id") == (tenant_id or "default")


class InMemoryGraph:
    """
    Multi-tenant in-memory graph store implementing the same interface as GraphMemory.
    Works without Neo4j — perfect for development and testing.
    """

    def __init__(self):
        self._nodes: dict[str, dict[str, Any]] = {}  # uid -> {label, ...props}
        self._relationships: list[dict[str, Any]] = []
        self._connected = True

    async def connect(self) -> bool:
        self._connected = True
        logger.info("inmemory_graph_ready")
        return True

    async def disconnect(self) -> None:
        self._connected = False

    @property
    def is_connected(self) -> bool:
        return self._connected

    async def health_check(self) -> bool:
        return self._connected

    async def init_schema(self) -> bool:
        return True

    # ============================================
    # Generic CRUD
    # ============================================

    async def _create_node(self, label: str, props: dict[str, Any]) -> str | None:
        uid = props.get("uid") or f"{label.lower()}-{uuid.uuid4().hex[:8]}"
        props["uid"] = uid
        props["_label"] = label
        if "tenant_id" not in props or not props["tenant_id"]:
            props["tenant_id"] = "default"
        if "created_at" not in props:
            props["created_at"] = datetime.now(UTC).isoformat()
        self._nodes[uid] = props
        return uid

    async def _get_node(self, label: str, uid: str, tenant_id: str | None = None) -> dict | None:
        node = self._nodes.get(uid)
        if node and node.get("_label") == label:
            if not _in_scope(node, tenant_id):
                return None
            return {k: v for k, v in node.items() if k != "_label"}
        return None

    async def _update_node(self, label: str, uid: str, updates: dict[str, Any], tenant_id: str | None = None) -> bool:
        if tenant_id == ALL_TENANTS:
            return False
        node = self._nodes.get(uid)
        if not node or node.get("_label") != label:
            return False
        if not _in_scope(node, tenant_id):
            return False
        if "tenant_id" in updates and updates["tenant_id"] != node.get("tenant_id"):
            return False
        node.update(updates)
        return True

    async def _delete_node(self, label: str, uid: str, tenant_id: str | None = None) -> bool:
        if tenant_id == ALL_TENANTS:
            return False
        node = self._nodes.get(uid)
        if not node or node.get("_label") != label:
            return False
        if not _in_scope(node, tenant_id):
            return False
        del self._nodes[uid]
        self._relationships = [
            r for r in self._relationships
            if r["from_uid"] != uid and r["to_uid"] != uid
        ]
        return True

    async def create_relationship(
        self, from_label: str, from_uid: str,
        to_label: str, to_uid: str,
        rel_type: str, props: dict[str, Any] | None = None,
        tenant_id: str | None = None,
    ) -> bool:
        if tenant_id == ALL_TENANTS:
            return False
        scope = tenant_id or "default"
        source = await self._get_node(from_label, from_uid, tenant_id=scope)
        target = await self._get_node(to_label, to_uid, tenant_id=scope)
        if source is None or target is None:
            return False
        self._relationships.append({
            "from_uid": from_uid, "from_label": from_label,
            "to_uid": to_uid, "to_label": to_label,
            "type": rel_type, "props": props or {},
            "tenant_id": scope,
        })
        return True

    # ============================================
    # Typed Multi-Tenant CRUD
    # ============================================

    async def create_engagement(self, engagement: EngagementNode) -> str | None:
        return await self._create_node("Engagement", engagement.model_dump())

    async def get_engagement(self, uid: str, tenant_id: str | None = None) -> dict | None:
        return await self._get_node("Engagement", uid, tenant_id=tenant_id)

    async def update_engagement(self, uid: str, tenant_id: str | None = None, **updates: Any) -> bool:
        return await self._update_node("Engagement", uid, updates, tenant_id=tenant_id)

    async def list_engagements(self, status: str | None = None, tenant_id: str | None = None) -> list[dict]:
        results = []
        for n in self._nodes.values():
            if n.get("_label") == "Engagement":
                if not _in_scope(n, tenant_id):
                    continue
                if status and n.get("status") != status:
                    continue
                results.append({k: v for k, v in n.items() if k != "_label"})
        return results

    async def create_asset(self, asset: AssetNode) -> str | None:
        uid = await self._create_node("Asset", asset.model_dump())
        if uid and asset.engagement_id:
            await self.create_relationship(
                "Asset", uid, "Engagement", asset.engagement_id,
                RelationshipType.PART_OF, tenant_id=asset.tenant_id,
            )
        return uid

    async def get_asset(self, uid: str, tenant_id: str | None = None) -> dict | None:
        return await self._get_node("Asset", uid, tenant_id=tenant_id)

    async def find_assets(self, engagement_id: str, asset_type: str | None = None, tenant_id: str | None = None) -> list[dict]:
        results = []
        for n in self._nodes.values():
            if n.get("_label") == "Asset" and n.get("engagement_id") == engagement_id:
                if not _in_scope(n, tenant_id):
                    continue
                if asset_type and n.get("asset_type") != asset_type:
                    continue
                results.append({k: v for k, v in n.items() if k != "_label"})
        return results

    async def upsert_asset(self, asset: AssetNode) -> str:
        for uid, n in self._nodes.items():
            if (n.get("_label") == "Asset" and n.get("value") == asset.value
                    and n.get("asset_type") == asset.asset_type
                    and n.get("engagement_id") == asset.engagement_id
                    and n.get("tenant_id") == asset.tenant_id):
                n.update({"metadata": asset.metadata})
                return uid
        return await self._create_node("Asset", asset.model_dump()) or ""

    async def create_finding(self, finding: FindingNode) -> str | None:
        uid = await self._create_node("Finding", finding.model_dump())
        if uid and finding.engagement_id:
            await self.create_relationship(
                "Finding", uid, "Engagement", finding.engagement_id,
                RelationshipType.PART_OF, tenant_id=finding.tenant_id,
            )
        return uid

    async def get_finding(self, uid: str, tenant_id: str | None = None) -> dict | None:
        return await self._get_node("Finding", uid, tenant_id=tenant_id)

    async def update_finding(self, uid: str, tenant_id: str | None = None, **updates: Any) -> bool:
        return await self._update_node("Finding", uid, updates, tenant_id=tenant_id)

    async def find_findings(
        self,
        engagement_id: str,
        severity: str | None = None,
        status: str | None = None,
        min_confidence: int = 0,
        tenant_id: str | None = None,
    ) -> list[dict]:
        results = []
        for n in self._nodes.values():
            if n.get("_label") != "Finding" or n.get("engagement_id") != engagement_id:
                continue
            if not _in_scope(n, tenant_id):
                continue
            if severity and n.get("severity") != severity:
                continue
            if status and n.get("status") != status:
                continue
            if n.get("confidence_score", 0) < min_confidence:
                continue
            results.append({k: v for k, v in n.items() if k != "_label"})
        return results

    async def create_hypothesis(self, hypothesis: HypothesisNode) -> str | None:
        return await self._create_node("Hypothesis", hypothesis.model_dump())

    async def update_hypothesis(self, uid: str, tenant_id: str | None = None, **updates: Any) -> bool:
        return await self._update_node("Hypothesis", uid, updates, tenant_id=tenant_id)

    async def find_hypotheses(self, engagement_id: str, status: str | None = None, tenant_id: str | None = None) -> list[dict]:
        results = []
        for n in self._nodes.values():
            if n.get("_label") != "Hypothesis" or n.get("engagement_id") != engagement_id:
                continue
            if not _in_scope(n, tenant_id):
                continue
            if status and n.get("status") != status:
                continue
            results.append({k: v for k, v in n.items() if k != "_label"})
        return results

    async def create_evidence(self, evidence: EvidenceNode) -> str | None:
        uid = await self._create_node("Evidence", evidence.model_dump())
        if uid and evidence.finding_id:
            await self.create_relationship(
                "Finding", evidence.finding_id, "Evidence", uid,
                RelationshipType.HAS_EVIDENCE, tenant_id=evidence.tenant_id,
            )
        return uid

    async def find_evidence(self, finding_id: str, tenant_id: str | None = None) -> list[dict]:
        """List all evidence attached to a finding (tenant-isolated)."""
        results: list[dict] = []
        for r in self._relationships:
            if r.get("type") != RelationshipType.HAS_EVIDENCE or r.get("from_uid") != finding_id:
                continue
            if tenant_id != ALL_TENANTS and r.get("tenant_id") != (tenant_id or "default"):
                continue
            ev_uid = r.get("to_uid")
            node = self._nodes.get(ev_uid)
            if node and node.get("_label") == "Evidence":
                results.append({k: v for k, v in node.items() if k != "_label"})
        return results

    async def create_technique(self, technique: TechniqueNode) -> str | None:
        return await self._create_node("Technique", technique.model_dump())

    async def register_agent(self, agent: AgentNode) -> str | None:
        return await self._create_node("Agent", agent.model_dump())

    # ============================================
    # Multi-Tenant Query & Search
    # ============================================

    async def add_node(self, label: str, data: dict[str, Any]) -> str | None:
        return await self._create_node(label, data)

    async def list_nodes(
        self,
        label: str,
        tenant_id: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        if not label.isidentifier():
            raise ValueError("Graph node label must be an identifier")
        if limit < 1:
            raise ValueError("Graph node limit must be positive")
        nodes = [
            {key: value for key, value in node.items() if key != "_label"}
            for node in self._nodes.values()
            if node.get("_label") == label and _in_scope(node, tenant_id)
        ]
        nodes.sort(key=lambda node: node.get("recorded_at", ""), reverse=True)
        return nodes[:limit]

    async def query(self, query_str: str, params: dict[str, Any] | None = None, tenant_id: str | None = None) -> list[dict]:
        results = []
        for n in self._nodes.values():
            if not _in_scope(n, tenant_id):
                continue
            results.append({k: v for k, v in n.items() if k != "_label"})
        return results[:20]

    async def search_findings(self, search_text: str, limit: int = 20, tenant_id: str | None = None) -> list[dict]:
        results = []
        search_lower = search_text.lower()
        for n in self._nodes.values():
            if n.get("_label") != "Finding":
                continue
            if not _in_scope(n, tenant_id):
                continue
            text = f"{n.get('title', '')} {n.get('description', '')} {n.get('vulnerability_class', '')}".lower()
            if search_lower in text:
                results.append({"finding": {k: v for k, v in n.items() if k != "_label"}, "score": 1.0})
        return results[:limit]

    async def get_engagement_summary(self, engagement_id: str, tenant_id: str | None = None) -> dict:
        eng = await self.get_engagement(engagement_id, tenant_id=tenant_id)
        assets = await self.find_assets(engagement_id, tenant_id=tenant_id)
        findings = await self.find_findings(engagement_id, tenant_id=tenant_id)
        hypotheses = await self.find_hypotheses(engagement_id, tenant_id=tenant_id)
        return {
            "engagement": eng or {},
            "asset_count": len(assets),
            "finding_count": len(findings),
            "hypothesis_count": len(hypotheses),
        }

    async def get_stats(self, tenant_id: str | None = None) -> dict:
        labels = {}
        filtered_nodes = [n for n in self._nodes.values() if _in_scope(n, tenant_id)]
        scope = tenant_id or "default"
        relationships = [
            r for r in self._relationships
            if tenant_id == ALL_TENANTS or r.get("tenant_id") == scope
        ]
        for n in filtered_nodes:
            lbl = n.get("_label", "Unknown")
            labels[lbl] = labels.get(lbl, 0) + 1
        return {
            "connected": True,
            "in_memory": True,
            "total_nodes": len(filtered_nodes),
            "total_relationships": len(relationships),
            **{k.lower() + "s": v for k, v in labels.items()},
        }
