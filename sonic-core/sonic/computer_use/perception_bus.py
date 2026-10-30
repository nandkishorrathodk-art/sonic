"""Low-latency, versioned perception state and O(1) semantic target index for computer-use actions.

The perception bus provides:
1. Low-latency, atomic, versioned perception snapshots across desktop and browser environments.
2. High-speed in-memory multi-attribute hash indexing (SemanticTargetIndex) supporting exact text,
   kebab/snake/compact slugs, placeholder, aria-label, role, role synonyms, and role-qualified phrases.
3. O(1) target resolution resolving natural language queries to absolute screen coordinates (x, y)
   in < 0.1ms with zero model roundtrips.
4. Strict anti-landmark compliance: unindexed queries fail closed to None without hallucinated coordinates.
"""

from __future__ import annotations

import hashlib
import math
import re
import time
from dataclasses import dataclass, field
from threading import RLock
from typing import Any

_RE_SPACES = re.compile(r"\s+")
_RE_KEBAB = re.compile(r"[\s_]+")
_RE_SNAKE = re.compile(r"[\s\-]+")
_RE_COMPACT = re.compile(r"[^a-z0-9]")
_RE_CAMEL = re.compile(r"[A-Z]?[a-z0-9]+|[A-Z]+(?=[A-Z][a-z0-9]|\b)")
_RE_SEARCH = re.compile(r"\bsearch\b")

ROLE_SYNONYMS: dict[str, tuple[str, ...]] = {
    "button": ("button", "btn", "push-button", "pushbutton", "clickable"),
    "push-button": ("button", "btn", "push-button", "pushbutton", "clickable"),
    "pushbutton": ("button", "btn", "push-button", "pushbutton", "clickable"),
    "btn": ("button", "btn", "push-button", "pushbutton", "clickable"),
    "entry": ("input", "field", "entry", "box", "text", "textbox", "text-field"),
    "input": ("input", "field", "entry", "box", "text", "textbox", "text-field"),
    "textbox": ("input", "field", "entry", "box", "text", "textbox", "text-field"),
    "text": ("input", "field", "entry", "box", "text", "textbox", "text-field"),
    "text-field": ("input", "field", "entry", "box", "text", "textbox", "text-field"),
    "password": ("password", "password input", "password field", "input", "field"),
    "search": ("search", "searchbar", "search-bar", "search_bar", "searchbox", "search-box", "search field", "search input", "search bar"),
    "searchbar": ("search", "searchbar", "search-bar", "search_bar", "searchbox", "search-box", "search field", "search input", "search bar"),
    "searchbox": ("search", "searchbar", "search-bar", "search_bar", "searchbox", "search-box", "search field", "search input", "search bar"),
    "link": ("link", "url", "hyperlink", "anchor", "a"),
    "combobox": ("combobox", "combo-box", "dropdown", "drop-down", "select", "menu"),
    "select": ("combobox", "combo-box", "dropdown", "drop-down", "select", "menu"),
    "dropdown": ("combobox", "combo-box", "dropdown", "drop-down", "select", "menu"),
    "checkbox": ("checkbox", "check-box", "check"),
    "radio": ("radio", "radiobutton", "radio-button"),
    "radiobutton": ("radio", "radiobutton", "radio-button"),
    "tab": ("tab", "page"),
    "menuitem": ("menuitem", "menu-item", "menu item", "option"),
    "dialog": ("dialog", "modal", "window", "popup"),
    "modal": ("dialog", "modal", "window", "popup"),
    "image": ("image", "img", "icon", "picture"),
    "img": ("image", "img", "icon", "picture"),
    "icon": ("image", "img", "icon", "picture"),
}


def _flatten_controls(items: Any) -> list[Any]:
    """Recursively flatten nested lists or tuples of controls."""
    flattened: list[Any] = []
    if isinstance(items, (list, tuple)):
        for item in items:
            if isinstance(item, (list, tuple)) and not isinstance(item, (str, bytes)):
                flattened.extend(_flatten_controls(item))
            elif item is not None:
                flattened.append(item)
    elif items is not None:
        flattened.append(items)
    return flattened


def _extract_control_data(control: Any) -> dict[str, Any] | None:
    """Extract normalized fields (name, role, bbox, attributes, source) from any control representation."""
    if control is None or isinstance(control, (str, bytes)):
        return None

    if isinstance(control, dict):
        visible = control.get("visible", True)
        if visible is False:
            return None
        name = str(control.get("name") or control.get("text") or "").strip()
        role = str(control.get("role") or control.get("tag") or "").strip()
        bbox = control.get("bbox")
        bbox_format = str(control.get("bbox_format") or "xywh").strip().lower()
        raw_attrs = control.get("attributes")
        attrs: dict[str, Any] = dict(raw_attrs) if isinstance(raw_attrs, dict) else {}
        for k in (
            "placeholder",
            "aria-label",
            "aria_label",
            "id",
            "name",
            "type",
            "value",
            "title",
            "selector",
            "editable",
            "focused",
            "window",
            "freshness_ms",
        ):
            if k in control and k not in attrs and control[k] is not None:
                attrs[k] = control[k]
        source = str(control.get("source") or "ui_control")
        confidence = control.get("confidence")
        confidence_provided = "confidence" in control
    else:
        visible = getattr(control, "visible", True)
        if visible is False:
            return None
        name = str(getattr(control, "name", "") or getattr(control, "text", "") or "").strip()
        role = str(getattr(control, "role", "") or getattr(control, "tag", "") or "").strip()
        bbox = getattr(control, "bbox", None)
        bbox_format = str(getattr(control, "bbox_format", "xywh") or "xywh").strip().lower()
        raw_attrs = getattr(control, "attributes", None)
        attrs = dict(raw_attrs) if isinstance(raw_attrs, dict) else {}
        for k in (
            "placeholder",
            "aria-label",
            "aria_label",
            "id",
            "name",
            "type",
            "value",
            "title",
            "selector",
            "editable",
            "focused",
            "window",
            "freshness_ms",
        ):
            val = getattr(control, k, None)
            if val is not None and k not in attrs:
                attrs[k] = val
        source = str(getattr(control, "source", None) or type(control).__name__)
        confidence = getattr(control, "confidence", None)
        confidence_provided = confidence is not None

    if not bbox or len(bbox) < 4:
        return None
    try:
        clean_bbox = (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
    except (ValueError, TypeError):
        return None
    if bbox_format == "xyxy":
        x1, y1, x2, y2 = clean_bbox
        clean_bbox = (x1, y1, x2 - x1, y2 - y1)
    elif bbox_format != "xywh":
        return None
    if clean_bbox[2] <= 0 or clean_bbox[3] <= 0:
        return None

    # Must have at least some identifiable content
    has_identity = bool(
        name
        or attrs.get("placeholder")
        or attrs.get("aria-label")
        or attrs.get("aria_label")
        or attrs.get("id")
        or attrs.get("name")
        or attrs.get("value")
        or attrs.get("title")
        or attrs.get("selector")
        or (role and role.lower() in ("search", "searchbar", "searchbox"))
    )
    if not has_identity:
        return None

    return {
        "name": name,
        "role": role,
        "bbox": clean_bbox,
        "attributes": attrs,
        "source": source,
        "confidence": confidence,
        "confidence_provided": confidence_provided,
        "freshness_ms": attrs.get("freshness_ms"),
    }


def _format_controls_summary(controls_seq: Any) -> tuple[str, ...]:
    """Extract descriptive control names or roles as a string tuple for snapshot summary."""
    if not controls_seq:
        return ()
    summary: list[str] = []
    items = _flatten_controls(controls_seq)
    for c in items:
        if isinstance(c, str):
            summary.append(c)
        elif isinstance(c, dict):
            name = c.get("name") or c.get("text") or c.get("role") or c.get("selector") or ""
            if name:
                summary.append(str(name))
        else:
            name = (
                getattr(c, "name", None)
                or getattr(c, "text", None)
                or getattr(c, "role", None)
                or str(c)
            )
            if name:
                summary.append(str(name))
    return tuple(summary)


@dataclass(frozen=True)
class TargetMatch:
    """A semantic target resolved against one perception snapshot."""

    query: str
    source: str = "indexed_ui"
    bbox: tuple[int, int, int, int] = (0, 0, 0, 0)
    confidence: float = 1.0
    state_version: int = 0
    observed_at_ns: int = 0
    role: str = ""
    attributes: dict[str, Any] = field(default_factory=dict)
    center: tuple[int, int] | None = None

    def __post_init__(self) -> None:
        if self.bbox:
            object.__setattr__(self, "bbox", tuple(int(v) for v in self.bbox))
        if self.center is None and self.bbox:
            x, y, w, h = self.bbox
            object.__setattr__(
                self, "center", (int(x + max(0, w // 2)), int(y + max(0, h // 2)))
            )
        elif self.center is not None:
            cx, cy = self.center
            object.__setattr__(self, "center", (int(cx), int(cy)))
        object.__setattr__(
            self,
            "confidence",
            max(0.0, min(1.0, float(self.confidence))),
        )
        if not self.observed_at_ns:
            object.__setattr__(self, "observed_at_ns", time.perf_counter_ns())
        if self.attributes is None:
            object.__setattr__(self, "attributes", {})


@dataclass(frozen=True)
class PerceptionSnapshot:
    """Versioned state shared by all workers for a single computer."""

    version: int
    screen_hash: str
    width: int
    height: int
    active_window: str
    windows: tuple[str, ...]
    processes: tuple[str, ...]
    visible_text: str
    controls: tuple[str, ...]
    browser_state: dict[str, Any]
    observed_at_ns: int


@dataclass(frozen=True)
class ActionFuture:
    """A prepared action that is valid only for its originating snapshot."""

    action: str
    target: TargetMatch | None
    state_version: int
    created_at_ns: int


class SemanticTargetIndex:
    """Ultra-fast, in-memory multi-attribute hash index for UI controls.

    Indexes controls across multiple attributes:
    - Exact text content and inner names
    - Kebab-case, snake_case, and compact alphanumeric slugs
    - Placeholder text and stripped input prefixes
    - ARIA labels and accessibility attributes
    - HTML/accessibility roles, role synonyms, and role-qualified phrases
    - Identifiers, input names, and CSS selectors

    Resolves natural language queries in < 0.1ms (O(1)) with zero model calls.
    Unindexed queries strictly return None (fail closed).
    """

    def __init__(self, state_version: int = 0) -> None:
        self._lock = RLock()
        self._version = state_version
        self._index: dict[str, TargetMatch] = {}
        self._confidences: dict[str, float] = {}
        self._matches: list[TargetMatch] = []
        self._ambiguous_keys: set[str] = set()

    @property
    def version(self) -> int:
        with self._lock:
            return self._version

    def clear(self, state_version: int | None = None) -> None:
        """Clear all indexed entries and optionally advance state version."""
        with self._lock:
            self._index.clear()
            self._confidences.clear()
            self._matches.clear()
            self._ambiguous_keys.clear()
            if state_version is not None:
                self._version = state_version

    def _generate_keys_for_phrase(
        self, phrase: str, role: str, base_conf: float
    ) -> list[tuple[str, float]]:
        """Generate normalized keys, slugs, and role-qualified phrases for a text candidate."""
        results: list[tuple[str, float]] = []
        if not phrase:
            return results

        raw = phrase.strip().lower()
        if not raw:
            return results

        # 1. Exact raw & whitespace-normalized
        results.append((raw, base_conf))
        norm_spaces = " ".join(raw.split())
        if norm_spaces != raw:
            results.append((norm_spaces, base_conf))

        # 2. Slugs
        if "_" in norm_spaces:
            kebab = _RE_KEBAB.sub("-", norm_spaces)
            snake = _RE_SNAKE.sub("_", norm_spaces)
        else:
            kebab = norm_spaces.replace(" ", "-")
            snake = norm_spaces.replace(" ", "_")
        results.append((kebab, base_conf * 0.98))
        results.append((snake, base_conf * 0.98))

        compact = _RE_COMPACT.sub("", norm_spaces)
        if len(compact) >= 2:
            results.append((compact, base_conf * 0.95))

        # 3. CamelCase / PascalCase decomposition
        words = _RE_CAMEL.findall(phrase)
        if len(words) > 1:
            joined = " ".join(w.lower() for w in words)
            if joined != norm_spaces:
                results.append((joined, base_conf * 0.96))
                results.append((joined.replace(" ", "-"), base_conf * 0.95))
                results.append((joined.replace(" ", "_"), base_conf * 0.95))
                cc_compact = _RE_COMPACT.sub("", joined)
                if len(cc_compact) >= 2:
                    results.append((cc_compact, base_conf * 0.93))

        # 4. Stripping placeholder / action prefixes
        stripped_variants: list[str] = []
        prefixes = (
            "enter your ",
            "enter ",
            "type your ",
            "type ",
            "input your ",
            "input ",
            "search for ",
            "search ",
            "fill in your ",
            "fill in ",
            "choose your ",
            "choose ",
            "select your ",
            "select ",
            "click to ",
            "click ",
        )
        for pfx in prefixes:
            if norm_spaces.startswith(pfx):
                stripped = norm_spaces[len(pfx):].strip()
                if stripped:
                    stripped_variants.append(stripped)
                    results.append((stripped, base_conf * 0.95))
                    results.append((stripped.replace(" ", "-"), base_conf * 0.94))
                    results.append((stripped.replace(" ", "_"), base_conf * 0.94))
                    sc = _RE_COMPACT.sub("", stripped)
                    if len(sc) >= 2:
                        results.append((sc, base_conf * 0.92))

        # 5. Role combinations & Role Synonyms
        role_clean = role.strip().lower()
        synonyms = ROLE_SYNONYMS.get(role_clean, (role_clean,) if role_clean else ())

        base_pool = [norm_spaces] + stripped_variants
        for base_item in base_pool:
            for syn in synonyms:
                if not syn:
                    continue
                if not base_item.endswith(syn) and not base_item.startswith(syn):
                    q1 = f"{base_item} {syn}"
                    results.append((q1, base_conf * 0.95))
                    results.append((q1.replace(" ", "-"), base_conf * 0.94))
                    results.append((q1.replace(" ", "_"), base_conf * 0.94))
                    q2 = f"{syn} {base_item}"
                    results.append((q2, base_conf * 0.93))
                elif base_item.endswith(f" {syn}"):
                    base_without_role = base_item[: -len(syn) - 1].strip()
                    if base_without_role:
                        results.append((base_without_role, base_conf * 0.96))

        # 6. Special role aliases
        if role_clean in ("search", "searchbar", "searchbox") or (
            role_clean in ("input", "entry", "textbox", "text", "combobox")
            and "search" in norm_spaces
        ):
            for search_alias in (
                "searchbar",
                "search-bar",
                "search_bar",
                "searchbox",
                "search-box",
                "search field",
                "search input",
                "search bar",
            ):
                results.append((search_alias, base_conf * 0.90))

        return results

    def index_match(
        self,
        match: TargetMatch,
        *,
        state_version: int | None = None,
        query: str | None = None,
        role: str = "",
        attributes: dict[str, Any] | None = None,
    ) -> None:
        """Register a TargetMatch into the index with multi-attribute expansion."""
        with self._lock:
            version = self._version if state_version is None else state_version
            if version != self._version:
                return

            target_role = role or match.role
            target_attrs = attributes if attributes is not None else match.attributes

            candidates: list[tuple[str, float]] = []
            seen_phrases: set[str] = set()

            def _add_candidate(text: Any, conf: float) -> None:
                if text:
                    s = str(text).strip()
                    s_lower = s.lower()
                    if s_lower and s_lower not in seen_phrases:
                        seen_phrases.add(s_lower)
                        candidates.append((s, conf))

            _add_candidate(query or match.query, match.confidence)

            if target_attrs:
                _add_candidate(target_attrs.get("aria-label") or target_attrs.get("aria_label"), match.confidence * 0.96)
                _add_candidate(target_attrs.get("placeholder"), match.confidence * 0.95)
                _add_candidate(target_attrs.get("title"), match.confidence * 0.93)
                _add_candidate(target_attrs.get("value"), match.confidence * 0.92)
                _add_candidate(target_attrs.get("id"), match.confidence * 0.90)
                _add_candidate(target_attrs.get("name"), match.confidence * 0.90)
                _add_candidate(target_attrs.get("selector"), match.confidence * 0.88)

            for phrase, conf in candidates:
                for key_str, k_conf in self._generate_keys_for_phrase(
                    phrase, target_role, conf
                ):
                    key_clean = key_str.strip().lower()
                    if not key_clean:
                        continue
                    existing_match = self._index.get(key_clean)
                    if (
                        existing_match is not None
                        and existing_match.bbox != match.bbox
                    ):
                        self._ambiguous_keys.add(key_clean)
                    existing_conf = self._confidences.get(key_clean, -1.0)
                    if key_clean not in self._index or k_conf > existing_conf:
                        self._index[key_clean] = match
                        self._confidences[key_clean] = k_conf

            self._matches.append(match)

    def index_target(
        self,
        query: str,
        bbox: tuple[int, int, int, int],
        *,
        source: str = "indexed_ui",
        confidence: float = 1.0,
        state_version: int | None = None,
        role: str = "",
        attributes: dict[str, Any] | None = None,
    ) -> TargetMatch:
        """Create and index a TargetMatch directly."""
        with self._lock:
            version = self._version if state_version is None else state_version
            match = TargetMatch(
                query=query,
                source=source,
                bbox=tuple(int(v) for v in bbox),
                confidence=confidence,
                state_version=version,
                observed_at_ns=time.perf_counter_ns(),
                role=role,
                attributes=dict(attributes or {}),
            )
            self.index_match(
                match,
                state_version=version,
                query=query,
                role=role,
                attributes=attributes,
            )
            return match

    def index_control(
        self,
        control: Any,
        *,
        state_version: int | None = None,
        source_override: str | None = None,
    ) -> TargetMatch | None:
        """Index a single structured desktop or browser control."""
        c_data = _extract_control_data(control)
        if not c_data:
            return None

        with self._lock:
            version = self._version if state_version is None else state_version
            source = source_override or c_data["source"]
            primary_name = c_data["name"] or c_data["attributes"].get("placeholder") or c_data["attributes"].get("aria-label") or c_data["role"]
            raw_confidence = c_data["confidence"]
            if c_data["confidence_provided"]:
                try:
                    confidence = float(raw_confidence)
                except (TypeError, ValueError):
                    confidence = 0.0
                if not math.isfinite(confidence) or confidence < 0.0 or confidence > 100.0:
                    confidence = 0.0
                elif confidence > 1.0:
                    confidence /= 100.0
            else:
                confidence = 0.0 if source.lower() == "ocr" else 0.98
            confidence = max(0.0, min(1.0, confidence))
            match = TargetMatch(
                query=primary_name,
                source=source,
                bbox=c_data["bbox"],
                confidence=confidence,
                state_version=version,
                observed_at_ns=time.perf_counter_ns(),
                role=c_data["role"],
                attributes=c_data["attributes"],
            )
            freshness_ms = c_data["freshness_ms"]
            try:
                age = None if freshness_ms is None else float(freshness_ms)
                is_fresh = age is None or (
                    math.isfinite(age) and 0.0 <= age <= 250.0
                )
            except (TypeError, ValueError):
                is_fresh = False
            if confidence >= 0.6 and is_fresh:
                self.index_match(
                    match,
                    state_version=version,
                    query=primary_name,
                    role=c_data["role"],
                    attributes=c_data["attributes"],
                )
            else:
                self._matches.append(match)
            return match

    def ingest(
        self,
        controls: Any,
        *,
        state_version: int | None = None,
        source_override: str | None = None,
    ) -> int:
        """Ingest a collection of desktop AT-SPI2 and CDP browser controls.

        Returns:
            Number of successfully indexed controls.
        """
        flattened = _flatten_controls(controls)
        count = 0
        with self._lock:
            for item in flattened:
                match = self.index_control(
                    item,
                    state_version=state_version,
                    source_override=source_override,
                )
                if match is not None:
                    count += 1
        return count

    def lookup(
        self, query: str, *, state_version: int | None = None
    ) -> TargetMatch | None:
        """O(1) multi-attribute lookup resolving a natural language query in < 0.1ms.

        Returns TargetMatch or None (fail-closed if unindexed).
        """
        if not query:
            return None

        with self._lock:
            if state_version is not None and state_version != self._version:
                return None

            q = query.strip().lower()
            if not q:
                return None

            # 1. Exact query match
            match = None if q in self._ambiguous_keys else self._index.get(q)
            if match is not None:
                return match

            # 2. Whitespace-normalized
            q_norm = re.sub(r"\s+", " ", q)
            match = (
                None
                if q_norm in self._ambiguous_keys
                else self._index.get(q_norm)
            )
            if match is not None:
                return match

            # 3. Slugs (kebab, snake, compact)
            q_kebab = re.sub(r"[\s_]+", "-", q_norm)
            match = (
                None
                if q_kebab in self._ambiguous_keys
                else self._index.get(q_kebab)
            )
            if match is not None:
                return match

            q_snake = re.sub(r"[\s\-]+", "_", q_norm)
            match = (
                None
                if q_snake in self._ambiguous_keys
                else self._index.get(q_snake)
            )
            if match is not None:
                return match

            q_compact = re.sub(r"[^a-z0-9]", "", q_norm)
            if len(q_compact) >= 2:
                match = (
                    None
                    if q_compact in self._ambiguous_keys
                    else self._index.get(q_compact)
                )
                if match is not None:
                    return match

            # 4. Punctuation-stripped query (e.g. "Search...", "Submit!")
            q_clean_punct = re.sub(r"^[^\w]+|[^\w]+$", "", q_norm)
            if q_clean_punct and q_clean_punct != q_norm:
                match = (
                    None
                    if q_clean_punct in self._ambiguous_keys
                    else self._index.get(q_clean_punct)
                )
                if match is not None:
                    return match

            # 5. Action / filler prefixes: "click submit", "click on sign in"
            action_prefixes = (
                "click on the ",
                "click on ",
                "click the ",
                "click ",
                "press the ",
                "press ",
                "tap on ",
                "tap ",
                "select the ",
                "select ",
            )
            for pfx in action_prefixes:
                if q_norm.startswith(pfx):
                    sub_q = q_norm[len(pfx):].strip()
                    sub_match = self.lookup(sub_q, state_version=state_version)
                    if sub_match is not None:
                        return sub_match

            # 6. Role-suffix stripping (e.g. query is "submit button", base was "submit")
            role_suffixes = (
                " button",
                " btn",
                " input",
                " field",
                " entry",
                " box",
                " link",
                " tab",
                " icon",
                " dropdown",
                " menu",
                " checkbox",
                " text",
            )
            for sfx in role_suffixes:
                if q_norm.endswith(sfx):
                    sub_q = q_norm[: -len(sfx)].strip()
                    sub_match = self.lookup(sub_q, state_version=state_version)
                    if sub_match is not None:
                        return sub_match

            # 7. Role-prefix stripping (e.g. query is "button submit")
            role_prefixes = (
                "button ",
                "btn ",
                "input ",
                "field ",
                "box ",
                "link ",
                "tab ",
                "icon ",
            )
            for pfx in role_prefixes:
                if q_norm.startswith(pfx):
                    sub_q = q_norm[len(pfx):].strip()
                    sub_match = self.lookup(sub_q, state_version=state_version)
                    if sub_match is not None:
                        return sub_match

            return None

    def resolve(
        self, query: str, *, state_version: int | None = None
    ) -> TargetMatch | None:
        """Alias to lookup for interface uniformity."""
        return self.lookup(query, state_version=state_version)

    def resolve_coordinates(
        self, query: str, *, state_version: int | None = None
    ) -> tuple[int, int] | None:
        """Resolve a query to absolute screen coordinates (cx, cy) or None."""
        match = self.lookup(query, state_version=state_version)
        if match is not None:
            return match.center
        return None

    def items(
        self, state_version: int | None = None
    ) -> list[tuple[str, TargetMatch]]:
        """Return all indexed (key, TargetMatch) pairs for the given or current version."""
        with self._lock:
            if state_version is not None and state_version != self._version:
                return []
            return list(self._index.items())

    def __len__(self) -> int:
        with self._lock:
            return len(self._index)

    def observed_matches(
        self, *, state_version: int | None = None
    ) -> tuple[TargetMatch, ...]:
        """Return controls grounded in the current observation, without aliases."""
        with self._lock:
            if state_version is not None and state_version != self._version:
                return ()
            return tuple(self._matches)

    def __contains__(self, query: str) -> bool:
        return self.lookup(query) is not None


class PerceptionBus:
    """Thread-safe state index with O(1) snapshot and stale-state checks."""

    def __init__(self) -> None:
        self._lock = RLock()
        self._version = 0
        self._snapshot = PerceptionSnapshot(
            version=0,
            screen_hash="",
            width=0,
            height=0,
            active_window="",
            windows=(),
            processes=(),
            visible_text="",
            controls=(),
            browser_state={},
            observed_at_ns=time.perf_counter_ns(),
        )
        self._targets: dict[tuple[int, str], TargetMatch] = {}
        self._semantic_index = SemanticTargetIndex(state_version=0)
        self._last_publish_latency_ns = 0

    @property
    def target_index(self) -> SemanticTargetIndex:
        """Direct access to the underlying SemanticTargetIndex."""
        return self._semantic_index

    @staticmethod
    def _screen_hash(screenshot_base64: str) -> str:
        if not screenshot_base64:
            return ""
        return hashlib.blake2s(
            screenshot_base64.encode("utf-8"), digest_size=16
        ).hexdigest()

    def publish(
        self,
        *,
        screenshot_base64: str = "",
        width: int = 0,
        height: int = 0,
        active_window: str = "",
        windows: list[str] | tuple[str, ...] = (),
        processes: list[str] | tuple[str, ...] = (),
        visible_text: str = "",
        controls: list[str] | tuple[str, ...] | list[Any] = (),
        browser_state: dict[str, Any] | None = None,
    ) -> PerceptionSnapshot:
        """Publish one live observation and atomically advance its version."""
        started = time.perf_counter_ns()
        with self._lock:
            now = time.perf_counter_ns()
            self._version += 1

            controls_summary = _format_controls_summary(controls)

            self._snapshot = PerceptionSnapshot(
                version=self._version,
                screen_hash=self._screen_hash(screenshot_base64),
                width=max(0, int(width)),
                height=max(0, int(height)),
                active_window=str(active_window or ""),
                windows=tuple(str(item) for item in windows),
                processes=tuple(str(item) for item in processes),
                visible_text=str(visible_text or ""),
                controls=controls_summary,
                browser_state=dict(browser_state or {}),
                observed_at_ns=now,
            )
            self._targets.clear()
            self._semantic_index.clear(state_version=self._version)

            # Auto-ingest structured controls if provided
            if controls:
                self.ingest_controls(controls, state_version=self._version)

            # Auto-ingest browser interactive elements if provided
            if browser_state and isinstance(browser_state, dict):
                elements = browser_state.get("interactive_elements")
                if elements and isinstance(elements, (list, tuple)):
                    self.ingest_controls(
                        elements, source="browser_dom", state_version=self._version
                    )

            self._last_publish_latency_ns = max(0, time.perf_counter_ns() - started)
            return self._snapshot

    def current(self) -> PerceptionSnapshot:
        """Return the latest snapshot without copying image data."""
        with self._lock:
            return self._snapshot

    def world_graph(self):
        """Build a generic graph from the current live snapshot."""
        from sonic.computer_use.gui_world_graph import GuiWorldGraph

        with self._lock:
            return GuiWorldGraph.from_snapshot(self._snapshot)

    def register_target(
        self,
        query: str,
        bbox: tuple[int, int, int, int],
        *,
        source: str,
        confidence: float,
        state_version: int | None = None,
        role: str = "",
        attributes: dict[str, Any] | None = None,
    ) -> TargetMatch:
        """Index a semantic target against the current state version."""
        with self._lock:
            version = self._version if state_version is None else state_version
            clamped_conf = max(0.0, min(1.0, float(confidence)))
            clean_bbox = tuple(int(value) for value in bbox)
            match = TargetMatch(
                query=query,
                source=source,
                bbox=clean_bbox,
                confidence=clamped_conf,
                state_version=version,
                observed_at_ns=time.perf_counter_ns(),
                role=role,
                attributes=dict(attributes or {}),
            )
            self._targets[(version, query.strip().lower())] = match
            self._semantic_index.index_match(
                match,
                state_version=version,
                query=query,
                role=role,
                attributes=attributes,
            )
            return match

    def ingest_controls(
        self,
        controls: Any,
        *,
        source: str | None = None,
        state_version: int | None = None,
    ) -> int:
        """Ingests desktop AT-SPI2 and browser CDP controls into the in-memory index.

        Args:
            controls: Single control, list/tuple of controls, or nested collections.
            source: Optional source override string.
            state_version: Optional version to index against (defaults to current).

        Returns:
            Number of successfully indexed controls.
        """
        with self._lock:
            version = self._version if state_version is None else state_version
            count = self._semantic_index.ingest(
                controls,
                state_version=version,
                source_override=source,
            )
            return count

    def resolve(self, query: str) -> TargetMatch | None:
        """Resolve a target only from the current, non-stale snapshot."""
        if not query:
            return None
        with self._lock:
            clean_q = query.strip().lower()
            exact = self._targets.get((self._version, clean_q))
            if exact is not None:
                return exact
            return self._semantic_index.lookup(query, state_version=self._version)

    def resolve_coordinates(self, query: str) -> tuple[int, int] | None:
        """Resolve a query to absolute screen coordinates (cx, cy) or None."""
        match = self.resolve(query)
        if match is not None:
            return match.center
        return None

    def prepare(self, action: str, query: str = "") -> ActionFuture:
        """Prepare an action for the current version; execution validates it."""
        with self._lock:
            return ActionFuture(
                action=action,
                target=self.resolve(query) if query else None,
                state_version=self._version,
                created_at_ns=time.perf_counter_ns(),
            )

    def is_current(self, future: ActionFuture) -> bool:
        """Return false when a new observation invalidated a prepared action."""
        with self._lock:
            return future.state_version == self._version

    def invalidate(self) -> int:
        """Advance the version when an external action may have changed UI state."""
        with self._lock:
            self._version += 1
            self._targets.clear()
            self._semantic_index.clear(state_version=self._version)
            self._snapshot = PerceptionSnapshot(
                version=self._version,
                screen_hash=self._snapshot.screen_hash,
                width=self._snapshot.width,
                height=self._snapshot.height,
                active_window=self._snapshot.active_window,
                windows=self._snapshot.windows,
                processes=self._snapshot.processes,
                visible_text=self._snapshot.visible_text,
                controls=self._snapshot.controls,
                browser_state=dict(self._snapshot.browser_state),
                observed_at_ns=time.perf_counter_ns(),
            )
            return self._version

    @property
    def last_update_latency_ns(self) -> int:
        """Time spent publishing the latest snapshot, useful for telemetry."""
        with self._lock:
            return self._last_publish_latency_ns
