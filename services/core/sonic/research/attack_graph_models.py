"""Shared typed models for the attack graph and its pathfinding engines."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class AttackNodeType(StrEnum):
    ENTRY_POINT = "entry_point"
    VULNERABILITY = "vulnerability"
    CREDENTIAL = "credential"
    ASSET = "asset"
    OBJECTIVE = "objective"


class AttackNode(BaseModel):
    """Vertex representing a target state, primitive, or asset in the attack surface."""

    id: str
    name: str
    node_type: AttackNodeType
    severity: str = "medium"
    metadata: dict[str, Any] = Field(default_factory=dict)


class AttackEdge(BaseModel):
    """Directed transition representing exploitation, pivot, or escalation."""

    source_id: str
    target_id: str
    technique: str
    confidence: float = 1.0
    prerequisites: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class AttackPath(BaseModel):
    """A complete multi-hop exploitation route from entry point to objective."""

    path_id: str
    nodes: list[AttackNode]
    edges: list[AttackEdge]
    total_hops: int
    compound_confidence: float
