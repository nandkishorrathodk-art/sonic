"""Tests for the runtime capability catalog (SKILLS.md) loader and prompt wiring."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sonic.computer_use.skill_catalog import (
    SkillCatalog,
    default_catalog_path,
    load_skill_catalog,
    render_skill_catalog,
    reset_skill_catalog_cache,
)

START = "<!-- SKILL-CATALOG:START -->"
END = "<!-- SKILL-CATALOG:END -->"


class _StubProvider:
    """Minimal provider so the agent can build a reasoning prompt offline."""

    async def terminal(self, workspace_id, cmd):
        raise AssertionError("the prompt build must not execute commands")

    async def screenshot(self, workspace_id):
        raise AssertionError("the prompt build must not capture the screen")


def _write_catalog(tmp_path: Path, payload: dict) -> Path:
    path = tmp_path / "SKILLS.md"
    path.write_text(
        "# Skills\n\n"
        f"{START}\n```json\n{json.dumps(payload)}\n```\n{END}\n",
        encoding="utf-8",
    )
    return path


def test_repository_catalog_parses_and_is_non_empty() -> None:
    """The real SKILLS.md must parse and expose the expected skill surface."""
    reset_skill_catalog_cache()
    catalog = load_skill_catalog()
    assert catalog.skills, "SKILLS.md must contain at least one skill"
    names = {s.name for s in catalog.skills}
    assert "desktop_app_control" in names
    assert "semantic_gui_grounding" in names


def test_every_status_is_one_of_the_declared_values() -> None:
    reset_skill_catalog_cache()
    catalog = load_skill_catalog()
    assert all(s.status in {"available", "conditional", "not_wired"} for s in catalog.skills)


def test_conditional_skills_name_their_requirement() -> None:
    """A conditional skill must say what it needs, or the agent cannot route."""
    reset_skill_catalog_cache()
    catalog = load_skill_catalog()
    for skill in catalog.by_status("conditional"):
        assert skill.requires, f"conditional skill {skill.name} must declare `requires`"


def test_available_skills_reference_real_code_evidence() -> None:
    """Every skill must cite a path; this keeps the catalog honest."""
    reset_skill_catalog_cache()
    catalog = load_skill_catalog()
    for skill in catalog.skills:
        assert skill.evidence, f"skill {skill.name} must cite evidence"


def test_render_lists_status_and_marks_unusable_skills() -> None:
    catalog = SkillCatalog(
        version=1,
        skills=(
            __import__(
                "sonic.computer_use.skill_catalog", fromlist=["Skill"]
            ).Skill(
                name="thing",
                category="test",
                status="conditional",
                summary="needs a checkpoint",
                actions=("GUI_CLICK",),
                requires="SOME_ENV",
            ),
        ),
    )
    text = catalog.render()
    assert "[conditional]" in text
    assert "requires=SOME_ENV" in text
    assert "actions=GUI_CLICK" in text


def test_missing_file_yields_empty_catalog_and_no_context(tmp_path: Path) -> None:
    catalog = load_skill_catalog(tmp_path / "nope.md")
    assert catalog.skills == ()
    assert catalog.render() == ""


def test_malformed_json_fails_closed(tmp_path: Path) -> None:
    path = tmp_path / "SKILLS.md"
    path.write_text(f"{START}\n```json\n{{ not json }}\n```\n{END}\n", encoding="utf-8")
    assert load_skill_catalog(path).skills == ()


def test_missing_json_block_fails_closed(tmp_path: Path) -> None:
    path = tmp_path / "SKILLS.md"
    path.write_text("# no catalog here\n", encoding="utf-8")
    assert load_skill_catalog(path).skills == ()


def test_invalid_status_entry_is_rejected_not_defaulted(tmp_path: Path) -> None:
    """An entry with an unknown status must be dropped, never silently allowed."""
    path = _write_catalog(
        tmp_path,
        {
            "version": 1,
            "skills": [
                {"name": "ok", "category": "t", "status": "available", "summary": "s"},
                {"name": "bad", "category": "t", "status": "totally-fine", "summary": "s"},
            ],
        },
    )
    catalog = load_skill_catalog(path)
    names = {s.name for s in catalog.skills}
    assert names == {"ok"}


def test_non_object_payload_fails_closed(tmp_path: Path) -> None:
    path = tmp_path / "SKILLS.md"
    path.write_text(f"{START}\n```json\n[1, 2, 3]\n```\n{END}\n", encoding="utf-8")
    assert load_skill_catalog(path).skills == ()


def test_render_truncates_on_a_line_boundary() -> None:
    from sonic.computer_use.skill_catalog import Skill

    skills = tuple(
        Skill(name=f"skill-{i}", category="c", status="available", summary="x" * 60)
        for i in range(50)
    )
    text = SkillCatalog(version=1, skills=skills).render(max_chars=300)
    assert text.endswith("(truncated)")
    assert not text.rstrip("\n(truncated)").endswith("x ")


def test_cache_returns_same_object_for_default_path() -> None:
    reset_skill_catalog_cache()
    first = load_skill_catalog()
    second = load_skill_catalog()
    assert first is second


def test_default_catalog_path_points_at_repo_root() -> None:
    path = default_catalog_path()
    assert path.name == "SKILLS.md"
    assert path.exists(), f"expected the repo catalog at {path}"


def test_render_skill_catalog_convenience_matches_catalog() -> None:
    reset_skill_catalog_cache()
    assert render_skill_catalog() == load_skill_catalog().render()


def test_agent_prompt_includes_capability_catalog() -> None:
    """The real reasoning prompt must carry the catalog so workers can route."""
    from sonic.computer_use.agent import ComputerUseAgent
    from sonic.computer_use.models import ComputerWorldObservation

    reset_skill_catalog_cache()
    agent = ComputerUseAgent(computer_provider=_StubProvider())
    _, obs_summary = agent._build_reasoning_context(
        "click the Save button", ComputerWorldObservation(), 1, "a.py", "b.py"
    )
    assert "CAPABILITY CATALOG" in obs_summary
    assert "semantic_gui_grounding" in obs_summary
    assert "not_wired" in obs_summary


def test_agent_prompt_omits_catalog_when_disabled(monkeypatch) -> None:
    from sonic.computer_use.agent import ComputerUseAgent
    from sonic.computer_use.models import ComputerWorldObservation

    monkeypatch.setenv("SONIC_SKILL_CATALOG_IN_PROMPT", "0")
    agent = ComputerUseAgent(computer_provider=_StubProvider())
    _, obs_summary = agent._build_reasoning_context(
        "click the Save button", ComputerWorldObservation(), 1, "a.py", "b.py"
    )
    assert "CAPABILITY CATALOG" not in obs_summary


def test_catalog_does_not_grant_permissions() -> None:
    """The rendered catalog must state that it is advisory, not authorizing."""
    reset_skill_catalog_cache()
    text = render_skill_catalog()
    assert "advisory" in text.lower()
    assert "does not grant" in text.lower()


@pytest.mark.parametrize("status", ["available", "conditional", "not_wired"])
def test_by_status_filtering(tmp_path: Path, status: str) -> None:
    path = _write_catalog(
        tmp_path,
        {
            "version": 1,
            "skills": [
                {"name": "a", "category": "c", "status": "available", "summary": "s"},
                {"name": "b", "category": "c", "status": "conditional", "summary": "s"},
                {"name": "d", "category": "c", "status": "not_wired", "summary": "s"},
            ],
        },
    )
    catalog = load_skill_catalog(path)
    assert [s.name for s in catalog.by_status(status)] == [
        {"available": "a", "conditional": "b", "not_wired": "d"}[status]
    ]


def test_get_returns_none_for_unknown_skill() -> None:
    reset_skill_catalog_cache()
    assert load_skill_catalog().get("does-not-exist") is None
