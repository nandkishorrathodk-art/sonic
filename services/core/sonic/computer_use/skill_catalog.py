"""Runtime loader for SONIC's capability catalog (``SKILLS.md``).

The catalog is documentation *and* prompt context: it tells a reasoning agent
which computer-use skills exist, how to route to them, and — critically — which
ones are fail-closed so the agent does not assume a capability it lacks.

Design rules:

* **Fail closed.** A missing, unreadable, or malformed catalog yields an empty
  catalog. It never invents a skill or a status.
* **Advisory only.** The rendered text is routing context. It grants no
  capability and cannot widen the safety envelope; ``ActionPolicy`` still
  decides every action.
* **Cheap.** Parsing happens once per process and is cached; callers that pass a
  different path bypass the cache.

The catalog lives in a fenced ``json`` block between
``<!-- SKILL-CATALOG:START -->`` and ``<!-- SKILL-CATALOG:END -->`` so the file
stays readable as Markdown while remaining machine-parseable.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sonic.logger import get_logger

logger = get_logger(__name__)

_VALID_STATUS = {"available", "conditional", "not_wired"}
_START = "<!-- SKILL-CATALOG:START -->"
_END = "<!-- SKILL-CATALOG:END -->"
_JSON_BLOCK = re.compile(r"```json\s*(?P<body>\{.*?\})\s*```", re.DOTALL)

_DEFAULT_FILENAME = "SKILLS.md"


@dataclass(frozen=True)
class Skill:
    """One capability entry from the catalog."""

    name: str
    category: str
    status: str
    summary: str
    actions: tuple[str, ...] = ()
    evidence: str = ""
    requires: str = ""

    @property
    def available(self) -> bool:
        return self.status == "available"

    @property
    def blocked(self) -> bool:
        """A skill the agent must not rely on right now."""
        return self.status in {"conditional", "not_wired"}


@dataclass(frozen=True)
class SkillCatalog:
    """A parsed, validated capability catalog."""

    version: int
    skills: tuple[Skill, ...]

    def by_status(self, status: str) -> tuple[Skill, ...]:
        return tuple(s for s in self.skills if s.status == status)

    def get(self, name: str) -> Skill | None:
        for skill in self.skills:
            if skill.name == name:
                return skill
        return None

    def render(self, *, max_chars: int = 2000) -> str:
        """Render a compact prompt section, or "" when there is nothing to say."""
        if not self.skills:
            return ""
        lines = [
            "=== CAPABILITY CATALOG (advisory routing; does not grant permissions) ===",
            "Skills marked [available] may be used. Skills marked [conditional] or "
            "[not_wired] are NOT currently usable — choose a different path instead "
            "of assuming they work.",
        ]
        for skill in self.skills:
            actions = f" actions={','.join(skill.actions)}" if skill.actions else ""
            requires = f" requires={skill.requires}" if skill.requires else ""
            lines.append(
                f"- [{skill.status}] {skill.name} ({skill.category}): {skill.summary}"
                f"{actions}{requires}"
            )
        lines.append("=== END CAPABILITY CATALOG ===")
        text = "\n".join(lines)
        if len(text) > max_chars:
            # Truncate on a line boundary so a half-written entry is never shown.
            text = text[:max_chars].rsplit("\n", 1)[0] + "\n(truncated)"
        return text


def default_catalog_path() -> Path:
    """Locate ``SKILLS.md`` at the repository root.

    This module lives at ``services/core/sonic/computer_use/skill_catalog.py``,
    so the repository root is four levels up from the package directory.
    """
    return Path(__file__).resolve().parents[4] / _DEFAULT_FILENAME


_cache: dict[str, SkillCatalog] = {}


def _parse(raw: str) -> SkillCatalog:
    section = raw
    start = raw.find(_START)
    end = raw.find(_END)
    if start != -1 and end != -1 and end > start:
        section = raw[start + len(_START):end]
    match = _JSON_BLOCK.search(section)
    if match is None:
        logger.warning("skill_catalog_json_block_missing")
        return SkillCatalog(version=0, skills=())

    try:
        payload = json.loads(match.group("body"))
    except json.JSONDecodeError as exc:
        logger.warning("skill_catalog_invalid_json", error=str(exc))
        return SkillCatalog(version=0, skills=())

    if not isinstance(payload, dict):
        logger.warning("skill_catalog_not_an_object")
        return SkillCatalog(version=0, skills=())

    raw_skills = payload.get("skills")
    if not isinstance(raw_skills, list):
        logger.warning("skill_catalog_skills_not_a_list")
        return SkillCatalog(version=0, skills=())

    skills: list[Skill] = []
    for item in raw_skills:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip()
        status = str(item.get("status") or "").strip()
        if not name or status not in _VALID_STATUS:
            logger.warning(
                "skill_catalog_entry_rejected", name=name or "<empty>", status=status or "<empty>"
            )
            continue
        raw_actions = item.get("actions") or []
        actions = (
            tuple(str(a).strip() for a in raw_actions if str(a).strip())
            if isinstance(raw_actions, list)
            else ()
        )
        skills.append(
            Skill(
                name=name,
                category=str(item.get("category") or "general").strip() or "general",
                status=status,
                summary=str(item.get("summary") or "").strip(),
                actions=actions,
                evidence=str(item.get("evidence") or "").strip(),
                requires=str(item.get("requires") or "").strip(),
            )
        )

    version_raw: Any = payload.get("version", 0)
    try:
        version = int(version_raw)
    except (TypeError, ValueError):
        version = 0
    return SkillCatalog(version=version, skills=tuple(skills))


def load_skill_catalog(path: str | Path | None = None) -> SkillCatalog:
    """Load and cache the capability catalog.

    Returns an empty catalog (never raises) when the file is absent or invalid,
    so a missing catalog degrades to "no catalog context" rather than a crash.
    """
    try:
        target = Path(path) if path is not None else default_catalog_path()
    except (TypeError, ValueError):
        return SkillCatalog(version=0, skills=())

    cache_key = str(target)
    if path is None and cache_key in _cache:
        return _cache[cache_key]

    try:
        raw = target.read_text(encoding="utf-8")
    except (FileNotFoundError, OSError):
        logger.info("skill_catalog_unavailable", path=cache_key)
        catalog = SkillCatalog(version=0, skills=())
    else:
        catalog = _parse(raw)

    if path is None:
        _cache[cache_key] = catalog
    return catalog


def render_skill_catalog(path: str | Path | None = None, *, max_chars: int = 2000) -> str:
    """Convenience: render the catalog prompt section ("" when unavailable)."""
    return load_skill_catalog(path).render(max_chars=max_chars)


def reset_skill_catalog_cache() -> None:
    """Clear the process cache (used by tests)."""
    _cache.clear()
