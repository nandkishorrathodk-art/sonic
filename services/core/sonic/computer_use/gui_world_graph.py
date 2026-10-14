"""Generic, observation-grounded graph of the current desktop state."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from sonic.computer_use.perception_models import PerceptionSnapshot


@dataclass(frozen=True)
class WorldNode:
    node_id: str
    kind: str
    label: str
    state_version: int
    attributes: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class WorldEdge:
    source: str
    relation: str
    target: str


@dataclass(frozen=True)
class GuiWorldGraph:
    state_version: int
    nodes: tuple[WorldNode, ...]
    edges: tuple[WorldEdge, ...]

    @staticmethod
    def _id(version: int, kind: str, label: str) -> str:
        raw = f"{version}:{kind}:{label}".encode()
        return hashlib.blake2s(raw, digest_size=8).hexdigest()

    @classmethod
    def from_snapshot(cls, snapshot: PerceptionSnapshot) -> GuiWorldGraph:
        nodes: list[WorldNode] = []
        edges: list[WorldEdge] = []

        def add(kind: str, label: str, **attributes: str) -> str:
            node_id = cls._id(snapshot.version, kind, label)
            nodes.append(
                WorldNode(
                    node_id=node_id,
                    kind=kind,
                    label=label,
                    state_version=snapshot.version,
                    attributes=tuple(sorted((key, str(value)) for key, value in attributes.items())),
                )
            )
            return node_id

        desktop_id = add("desktop", "desktop")
        active_id = add("window", snapshot.active_window) if snapshot.active_window else ""
        if active_id:
            edges.append(WorldEdge(desktop_id, "contains", active_id))
            edges.append(WorldEdge(active_id, "focused", desktop_id))

        window_ids: dict[str, str] = {}
        for label in snapshot.windows:
            if not label:
                continue
            window_id = add("window", label)
            window_ids[label] = window_id
            edges.append(WorldEdge(desktop_id, "contains", window_id))
            if label == snapshot.active_window:
                edges.append(WorldEdge(window_id, "focused", desktop_id))

        parent_id = active_id or desktop_id
        for label in snapshot.controls:
            if not label:
                continue
            control_id = add("control", label)
            edges.append(WorldEdge(parent_id, "contains", control_id))

        for line in snapshot.visible_text.splitlines():
            label = line.strip()
            if label:
                text_id = add("text", label)
                edges.append(WorldEdge(parent_id, "displays", text_id))

        for label in snapshot.processes:
            if not label:
                continue
            process_id = add("process", label)
            edges.append(WorldEdge(desktop_id, "runs", process_id))

        return cls(snapshot.version, tuple(nodes), tuple(edges))

    def summary(self, limit: int = 20) -> str:
        """Render compact graph facts for an LLM without inventing relations."""
        lines = [
            f"GUI world graph (observation v{self.state_version}): "
            f"{len(self.nodes)} nodes, {len(self.edges)} relations."
        ]
        for edge in self.edges[:limit]:
            source = next((node.label for node in self.nodes if node.node_id == edge.source), edge.source)
            target = next((node.label for node in self.nodes if node.node_id == edge.target), edge.target)
            lines.append(f"- {source} --{edge.relation}--> {target}")
        return "\n".join(lines)

    def fingerprint(self) -> str:
        """Return a stable content fingerprint independent of observation version."""
        labels = {node.node_id: (node.kind, node.label) for node in self.nodes}
        content = "|".join(
            f"{kind}:{label}:{attributes}"
            for kind, label, attributes in sorted(
                [
                    (node.kind, node.label, node.attributes)
                    for node in self.nodes
                ],
                key=repr,
            )
        ) + "||" + "|".join(
            f"{labels.get(edge.source, ('unknown', edge.source))}:"
            f"{edge.relation}:{labels.get(edge.target, ('unknown', edge.target))}"
            for edge in self.edges
        )
        return hashlib.blake2s(content.encode("utf-8"), digest_size=12).hexdigest()

    def delta(self, previous: GuiWorldGraph | None) -> dict[str, tuple[str, ...]]:
        """Describe observed graph additions/removals without inferring intent."""
        if previous is None:
            return {"added": tuple(node.label for node in self.nodes), "removed": ()}
        current_nodes = {(node.kind, node.label) for node in self.nodes}
        previous_nodes = {(node.kind, node.label) for node in previous.nodes}
        return {
            "added": tuple(sorted(f"{kind}:{label}" for kind, label in current_nodes - previous_nodes)),
            "removed": tuple(sorted(f"{kind}:{label}" for kind, label in previous_nodes - current_nodes)),
        }
