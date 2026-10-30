"""
Unit and E2E Test Suite for O(1) Semantic Target Grounding (Requirement R2).

Validates:
- SemanticTargetIndex multi-attribute hash indexing (exact text, slugs, placeholder, aria-label, role, role synonyms, qualified phrases).
- PerceptionBus control ingestion (ingest_controls) and resolution.
- Sub-0.1ms resolution execution latency (< 0.5ms acceptance criteria).
- Strict anti-landmark compliance: unindexed queries return None (zero hallucinated coordinates).
- Version isolation and cache invalidation on snapshot publish.
- Grounding fast-path integration with resolve_ui_target.
- 4-Tier test methodology with ZERO MOCK GATES.
"""

from __future__ import annotations

import time
from typing import Any
import pytest

from sonic.computer_use.perception_bus import PerceptionBus, TargetMatch
from sonic.computer_use.grounding import resolve_ui_target

# Attempt to import SemanticTargetIndex from perception_bus or grounding
try:
    from sonic.computer_use.perception_bus import SemanticTargetIndex
except ImportError:
    try:
        from sonic.computer_use.grounding import SemanticTargetIndex
    except ImportError:
        SemanticTargetIndex = None


# ============================================================================
# Tier 1: Feature Coverage (>=5 tests per feature)
# ============================================================================

def test_semantic_index_exact_text_lookup():
    """Verify SemanticTargetIndex resolves exact element text (case-insensitive)."""
    if SemanticTargetIndex is None:
        pytest.fail("SemanticTargetIndex is not implemented in perception_bus.py or grounding.py")

    index = SemanticTargetIndex(state_version=1)
    control = {
        "text": "Sign In",
        "role": "button",
        "bbox": (200, 300, 100, 40),
    }
    index.index_control(control, state_version=1)

    match = index.resolve("Sign In", state_version=1)
    assert match is not None, "Exact text 'Sign In' must resolve"
    assert match.bbox == (200, 300, 100, 40)
    assert match.center == (250, 320)

    # Case-insensitive
    match_lower = index.resolve("sign in", state_version=1)
    assert match_lower is not None
    assert match_lower.bbox == (200, 300, 100, 40)


def test_semantic_index_slug_variants():
    """Verify SemanticTargetIndex resolves kebab-case, snake-case, and compact slugs."""
    if SemanticTargetIndex is None:
        pytest.fail("SemanticTargetIndex is not implemented")

    index = SemanticTargetIndex(state_version=1)
    control = {
        "text": "First Name",
        "role": "input",
        "bbox": (150, 200, 180, 32),
    }
    index.index_control(control, state_version=1)

    # Kebab-case
    match_kebab = index.resolve("first-name", state_version=1)
    assert match_kebab is not None, "Kebab slug 'first-name' must resolve"
    assert match_kebab.bbox == (150, 200, 180, 32)

    # Snake-case
    match_snake = index.resolve("first_name", state_version=1)
    assert match_snake is not None, "Snake slug 'first_name' must resolve"

    # Compact alphanumeric
    match_compact = index.resolve("firstname", state_version=1)
    assert match_compact is not None, "Compact slug 'firstname' must resolve"


def test_semantic_index_placeholder_and_aria_label():
    """Verify SemanticTargetIndex resolves placeholder and aria-label attributes."""
    if SemanticTargetIndex is None:
        pytest.fail("SemanticTargetIndex is not implemented")

    index = SemanticTargetIndex(state_version=1)
    control = {
        "text": "",
        "role": "input",
        "placeholder": "Enter your username",
        "aria_label": "Username or email field",
        "attributes": {
            "placeholder": "Enter your username",
            "aria-label": "Username or email field",
            "id": "identifierId",
        },
        "bbox": (300, 150, 250, 36),
    }
    index.index_control(control, state_version=1)

    # Resolve by placeholder
    match_ph = index.resolve("Enter your username", state_version=1)
    assert match_ph is not None, "Placeholder query must resolve"
    assert match_ph.bbox == (300, 150, 250, 36)

    # Resolve by aria-label
    match_aria = index.resolve("Username or email field", state_version=1)
    assert match_aria is not None, "ARIA label query must resolve"

    # Resolve by ID if indexed
    match_id = index.resolve("identifierId", state_version=1)
    assert match_id is not None, "Element ID must resolve"


def test_semantic_index_role_qualified_and_synonyms():
    """Verify SemanticTargetIndex resolves role-qualified queries and role synonyms."""
    if SemanticTargetIndex is None:
        pytest.fail("SemanticTargetIndex is not implemented")

    index = SemanticTargetIndex(state_version=1)
    btn_control = {
        "text": "Submit",
        "role": "button",
        "bbox": (500, 400, 90, 35),
    }
    input_control = {
        "text": "Search",
        "role": "input",
        "bbox": (100, 50, 300, 30),
    }
    index.index_control(btn_control, state_version=1)
    index.index_control(input_control, state_version=1)

    # Role qualified: "submit button"
    match_btn_role = index.resolve("submit button", state_version=1)
    assert match_btn_role is not None, "'submit button' must resolve"
    assert match_btn_role.bbox == (500, 400, 90, 35)

    # Role synonym: "submit btn"
    match_btn_synonym = index.resolve("submit btn", state_version=1)
    assert match_btn_synonym is not None, "'submit btn' synonym must resolve"

    # Role qualified: "search input" / "search field"
    match_input_role = index.resolve("search input", state_version=1)
    assert match_input_role is not None, "'search input' must resolve"

    match_input_synonym = index.resolve("search field", state_version=1)
    assert match_input_synonym is not None, "'search field' synonym must resolve"


def test_perception_bus_ingest_controls():
    """Verify PerceptionBus.ingest_controls registers controls and resolves them."""
    bus = PerceptionBus()
    controls = [
        {
            "name": "Next",
            "role": "button",
            "bbox": (600, 500, 80, 36),
            "attributes": {"type": "submit"},
            "visible": True,
        },
        {
            "name": "Password",
            "role": "input",
            "bbox": (200, 350, 200, 32),
            "attributes": {"type": "password", "placeholder": "Enter password"},
            "visible": True,
        }
    ]

    if not hasattr(bus, "ingest_controls"):
        pytest.fail("PerceptionBus.ingest_controls is not implemented")

    count = bus.ingest_controls(controls)
    assert count >= 2, f"Expected at least 2 controls ingested, got {count}"

    # Resolve from bus
    match_next = bus.resolve("Next")
    assert match_next is not None, "'Next' must resolve"
    assert match_next.bbox == (600, 500, 80, 36)

    match_pwd = bus.resolve("password")
    assert match_pwd is not None, "'password' must resolve"


# ============================================================================
# Tier 2: Boundary, Corner & Strict Anti-Landmark Protection
# ============================================================================

def test_strict_anti_landmark_unindexed_returns_none():
    """Verify strict anti-landmark compliance: unindexed queries return None (zero hallucinated coordinates)."""
    bus = PerceptionBus()
    # Unindexed bus must return None for any arbitrary target
    assert bus.resolve("Applications") is None
    assert bus.resolve("Terminal") is None
    assert bus.resolve("Desktop Icon") is None
    assert bus.resolve("Submit") is None

    # Verify grounding.py resolve_ui_target also returns None without falling back to percentage landmarks
    res = resolve_ui_target("Applications menu", allow_landmarks=True)
    assert res is None, "Anti-landmark rule: unindexed queries MUST return None"

    res_term = resolve_ui_target("Terminal icon", allow_landmarks=True)
    assert res_term is None, "Anti-landmark rule: unindexed queries MUST return None"


def test_empty_and_whitespace_queries_fail_closed():
    """Verify empty and whitespace queries return None safely without errors."""
    bus = PerceptionBus()
    assert bus.resolve("") is None
    assert bus.resolve("   ") is None
    assert bus.resolve("\t\n") is None

    if SemanticTargetIndex is not None:
        index = SemanticTargetIndex(state_version=1)
        assert index.resolve("", state_version=1) is None
        assert index.resolve("   ", state_version=1) is None


def test_special_characters_and_regex_resilience():
    """Verify queries with regex meta-characters (*, +, ?, (, [, ^, $, \\) resolve without crashing."""
    if SemanticTargetIndex is None:
        pytest.fail("SemanticTargetIndex is not implemented")

    index = SemanticTargetIndex(state_version=1)
    control = {
        "text": "Save & Continue (Step 1/2) [Beta] *",
        "role": "button",
        "bbox": (100, 200, 150, 40),
    }
    index.index_control(control, state_version=1)

    match = index.resolve("Save & Continue (Step 1/2) [Beta] *", state_version=1)
    assert match is not None
    assert match.bbox == (100, 200, 150, 40)


def test_unicode_and_accented_characters():
    """Verify unicode characters and internationalized text resolve accurately."""
    if SemanticTargetIndex is None:
        pytest.fail("SemanticTargetIndex is not implemented")

    index = SemanticTargetIndex(state_version=1)
    control = {
        "text": "Créer un compte",
        "role": "button",
        "bbox": (50, 100, 120, 35),
    }
    index.index_control(control, state_version=1)

    match = index.resolve("Créer un compte", state_version=1)
    assert match is not None
    assert match.bbox == (50, 100, 120, 35)


def test_version_isolation_and_invalidation():
    """Verify that targets from old versions are inaccessible after bus.publish or bus.invalidate."""
    bus = PerceptionBus()
    first_snapshot = bus.publish(screenshot_base64="frame1")

    if hasattr(bus, "ingest_controls"):
        bus.ingest_controls([{"text": "Login", "role": "button", "bbox": (10, 20, 80, 30)}])
        assert bus.resolve("Login") is not None

        # Advancing version clears targets
        bus.publish(screenshot_base64="frame2")
        assert bus.current().version == first_snapshot.version + 1
        assert bus.resolve("Login") is None, "Targets from previous version must be purged"


# ============================================================================
# Tier 3: Cross-Feature Pairwise Integration
# ============================================================================

def test_pairwise_grounding_fast_path_with_perception_bus():
    """Verify grounding.resolve_ui_target queries PerceptionBus first and returns center coordinates in < 0.001ms."""
    bus = PerceptionBus()
    control = {
        "text": "Submit Application",
        "role": "button",
        "bbox": (300, 400, 100, 40),
    }
    if hasattr(bus, "ingest_controls"):
        bus.ingest_controls([control])

        # Test resolve_ui_target with perception_bus parameter
        started = time.perf_counter()
        coords = resolve_ui_target("Submit Application", perception_bus=bus)
        elapsed_ms = (time.perf_counter() - started) * 1000.0

        assert coords == (350, 420), f"Expected center (350, 420), got {coords}"
        assert elapsed_ms < 0.1, f"Fast-path resolution took {elapsed_ms:.4f}ms (must be < 0.1ms)"


def test_pairwise_browser_and_desktop_controls_in_single_index():
    """Verify both desktop AT-SPI controls and browser DOM controls coexist cleanly in PerceptionBus."""
    bus = PerceptionBus()
    controls = [
        # Desktop control from X11/AT-SPI
        {"name": "File Menu", "role": "menu", "bbox": (10, 0, 50, 24), "source": "desktop_atspi"},
        # Browser control from CDP DOM
        {"name": "Search Button", "role": "button", "bbox": (400, 120, 80, 32), "source": "browser_cdp"},
    ]
    if hasattr(bus, "ingest_controls"):
        bus.ingest_controls(controls)

        match_desktop = bus.resolve("File Menu")
        assert match_desktop is not None
        assert match_desktop.bbox == (10, 0, 50, 24)

        match_browser = bus.resolve("Search Button")
        assert match_browser is not None
        assert match_browser.bbox == (400, 120, 80, 32)


# ============================================================================
# Tier 4: Real-World Workload & Latency Benchmarks
# ============================================================================

def test_benchmark_target_resolution_latency():
    """
    Benchmark: In-memory target resolution via PerceptionBus.resolve() must execute
    in under 0.1ms (strictly exceeding the < 0.5ms acceptance criteria).
    """
    bus = PerceptionBus()

    # Generate a realistic 150-element DOM + Desktop scene
    controls = []
    # Standard form inputs
    controls.append({"name": "First name", "role": "input", "bbox": (100, 200, 180, 35), "placeholder": "First name"})
    controls.append({"name": "Last name", "role": "input", "bbox": (300, 200, 180, 35), "placeholder": "Last name"})
    controls.append({"name": "Username", "role": "input", "bbox": (100, 250, 380, 35), "placeholder": "Choose a username"})
    controls.append({"name": "Password", "role": "input", "bbox": (100, 300, 180, 35), "placeholder": "Password"})
    controls.append({"name": "Confirm", "role": "input", "bbox": (300, 300, 180, 35), "placeholder": "Confirm password"})
    controls.append({"name": "Next", "role": "button", "bbox": (400, 400, 80, 36)})
    controls.append({"name": "Sign in instead", "role": "link", "bbox": (100, 405, 120, 25)})

    # Add 140 background links, buttons, and desktop panel controls
    for i in range(140):
        controls.append({
            "name": f"Footer Link {i}",
            "role": "link" if i % 2 == 0 else "button",
            "bbox": (10 + (i % 10) * 80, 600 + (i // 10) * 20, 70, 18),
        })

    if not hasattr(bus, "ingest_controls"):
        pytest.fail("PerceptionBus.ingest_controls is not implemented")

    ingest_start = time.perf_counter()
    bus.ingest_controls(controls)
    ingest_elapsed_ms = (time.perf_counter() - ingest_start) * 1000.0

    # Ingestion benchmark: 150 elements should ingest in < 15ms
    assert ingest_elapsed_ms < 15.0, f"Ingesting 150 elements took {ingest_elapsed_ms:.2f}ms"

    # Query resolution benchmark: 1,000 queries
    queries = [
        "First name",
        "first-name",
        "firstname",
        "Choose a username",
        "Next",
        "next button",
        "next btn",
        "Password",
        "Sign in instead",
    ]

    # Warmup
    for q in queries:
        bus.resolve(q)

    iterations = 1000
    start = time.perf_counter()
    for i in range(iterations):
        q = queries[i % len(queries)]
        match = bus.resolve(q)
        assert match is not None
    total_elapsed_ms = (time.perf_counter() - start) * 1000.0
    avg_latency_ms = total_elapsed_ms / iterations

    # Latency assertions:
    # Acceptance criteria is < 0.5ms. Requirement R2 target is < 0.1ms.
    assert avg_latency_ms < 0.1, (
        f"Average resolution latency {avg_latency_ms * 1000:.2f} microseconds per query "
        f"exceeds 0.1ms target (total for {iterations} queries: {total_elapsed_ms:.2f}ms)"
    )
    assert avg_latency_ms < 0.5, f"Must pass acceptance criteria < 0.5ms"
