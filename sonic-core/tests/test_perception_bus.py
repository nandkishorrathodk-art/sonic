from sonic.computer_use.perception_bus import PerceptionBus


def test_perception_bus_versions_and_rejects_stale_futures():
    bus = PerceptionBus()
    first = bus.publish(
        screenshot_base64="frame-a",
        width=1280,
        height=800,
        active_window="Browser",
        windows=["Browser", "Terminal"],
        processes=["browser"],
        controls=["submit"],
    )
    target = bus.register_target(
        "submit", (10, 20, 100, 40), source="accessibility", confidence=1.0
    )
    future = bus.prepare("click", "submit")

    assert first.version == 1
    assert target.state_version == 1
    assert bus.resolve("submit") == target
    assert bus.is_current(future)
    assert bus.current().windows == ("Browser", "Terminal")
    assert bus.current().processes == ("browser",)

    bus.publish(screenshot_base64="frame-b")
    assert bus.current().version == 2
    assert bus.resolve("submit") is None
    assert not bus.is_current(future)


def test_perception_bus_clamps_confidence_and_hashes_frames():
    bus = PerceptionBus()
    snapshot = bus.publish(screenshot_base64="same-frame")
    match = bus.register_target(
        "target", (0, 0, 1, 1), source="dom", confidence=4.0
    )

    assert len(snapshot.screen_hash) == 32
    assert match.confidence == 1.0
    assert bus.last_update_latency_ns >= 0


def test_perception_bus_invalidates_targets_after_external_action():
    bus = PerceptionBus()
    bus.publish(screenshot_base64="frame")
    bus.register_target("menu", (1, 2, 3, 4), source="atspi", confidence=1.0)
    future = bus.prepare("click", "menu")

    version = bus.invalidate()

    assert version == 2
    assert bus.resolve("menu") is None
    assert not bus.is_current(future)


def test_perception_bus_normalizes_explicit_corner_bboxes_before_centering():
    bus = PerceptionBus()
    bus.publish(
        screenshot_base64="frame",
        width=1280,
        height=800,
        controls=[
            {
                "name": source,
                "role": "button",
                "bbox": (100, 200, 180, 230),
                "bbox_format": "xyxy",
                "source": source,
            }
            for source in ("ocr", "atspi", "x11")
        ],
    )

    for source in ("ocr", "atspi", "x11"):
        match = bus.resolve(source)
        assert match is not None
        assert match.bbox == (100, 200, 80, 30)
        assert match.center == (140, 215)


def test_perception_bus_rejects_invalid_bbox_format_and_empty_rectangles():
    bus = PerceptionBus()
    bus.publish(
        screenshot_base64="frame",
        controls=[
            {
                "name": "unknown-format",
                "bbox": (10, 20, 30, 40),
                "bbox_format": "guess",
            },
            {
                "name": "inverted-rectangle",
                "bbox": (30, 40, 10, 20),
                "bbox_format": "xyxy",
            },
        ],
    )

    assert bus.resolve("unknown-format") is None
    assert bus.resolve("inverted-rectangle") is None


def test_perception_bus_fails_closed_for_duplicate_semantic_targets():
    bus = PerceptionBus()
    bus.publish(
        screenshot_base64="frame",
        width=1280,
        height=800,
        controls=[
            {
                "name": "Submit",
                "role": "button",
                "bbox": (100, 200, 80, 30),
                "attributes": {"aria-label": "Submit login"},
            },
            {
                "name": "Submit",
                "role": "button",
                "bbox": (900, 700, 80, 30),
                "attributes": {"aria-label": "Submit payment"},
            },
        ],
    )

    assert bus.resolve("Submit") is None
    assert bus.resolve("Submit button") is None
    assert bus.resolve("Submit login") is not None
    assert bus.resolve("Submit login").center == (140, 215)
    assert bus.resolve("Submit payment") is not None
    assert bus.resolve("Submit payment").center == (940, 715)
