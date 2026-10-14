from sonic.computer_use.gui_world_graph import GuiWorldGraph
from sonic.computer_use.perception_bus import PerceptionBus, PerceptionSnapshot
from sonic.computer_use.perception_models import PerceptionSnapshot as PerceptionSnapshotContract


def test_snapshot_contract_is_shared_without_bus_import_dependency():
    assert PerceptionSnapshot is PerceptionSnapshotContract


def test_world_graph_is_built_from_one_live_snapshot():
    bus = PerceptionBus()
    snapshot = bus.publish(
        active_window="focused window",
        windows=["focused window", "another window"],
        processes=["process-a"],
        visible_text="A visible label\nA second label",
        controls=["submit control", "text input"],
    )

    graph = bus.world_graph()

    assert graph.state_version == snapshot.version
    assert {node.kind for node in graph.nodes} == {
        "desktop", "window", "process", "text", "control"
    }
    assert any(edge.relation == "focused" for edge in graph.edges)
    assert "submit control" in graph.summary()


def test_world_graph_changes_version_when_perception_is_invalidated():
    bus = PerceptionBus()
    first = bus.publish(windows=["first"])
    first_graph = GuiWorldGraph.from_snapshot(first)
    bus.invalidate()
    second_graph = bus.world_graph()

    assert second_graph.state_version > first_graph.state_version
    first_node = next(node for node in first_graph.nodes if node.label == "first")
    second_node = next(node for node in second_graph.nodes if node.label == "first")
    assert second_node.state_version > first_node.state_version
    assert second_node.node_id != first_node.node_id


def test_world_graph_does_not_invent_controls_or_coordinates():
    bus = PerceptionBus()
    bus.publish(active_window="unknown application")
    graph = bus.world_graph()

    assert not any(node.kind == "control" for node in graph.nodes)
    assert all("x=" not in node.label and "y=" not in node.label for node in graph.nodes)


def test_graph_delta_reports_only_observed_node_changes():
    bus = PerceptionBus()
    first = bus.publish(windows=["one"])
    first_graph = GuiWorldGraph.from_snapshot(first)
    second = bus.publish(windows=["two"])
    delta = GuiWorldGraph.from_snapshot(second).delta(first_graph)

    assert "window:two" in delta["added"]
    assert "window:one" in delta["removed"]


def test_fingerprint_is_stable_across_observation_versions():
    bus = PerceptionBus()
    first = GuiWorldGraph.from_snapshot(bus.publish(windows=["same"], controls=["control"]))
    second = GuiWorldGraph.from_snapshot(bus.publish(windows=["same"], controls=["control"]))

    assert first.state_version != second.state_version
    assert first.fingerprint() == second.fingerprint()
