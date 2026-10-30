"""
SONIC-REDA — Being Craft (durable toolkit / authored artifacts)
===============================================================

The "craft" part of the AI human: durable artifacts the being authors on its
own initiative (notes, observations, reusable scripts) that persist on the HOST
filesystem, keyed by being_id — so the being builds up a personal toolkit that
survives restart, distinct from disposable in-sandbox research outputs.

Before this, evolution/dev outputs were written only into disposable sandboxes
(/home/sonic/workspace) and became durable only via git commits. There was no
host-side "notes" or "scripts" home for the being itself. A being with no
durable craft is an amnesiac worker.

Persistence: host filesystem under ``sonic_data/craft/<being_id>/`` (env
``SONIC_DATA_DIR``, mirroring the workstation state file convention). Each note
is one file; an index is kept in memory + written as ``index.json``. This is
deliberately simple and inspectable — a human can read the being's notes
directly off disk.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
import re

from sonic.logger import get_logger

logger = get_logger(__name__)


def _craft_root() -> str:
    base = os.environ.get("SONIC_DATA_DIR", "sonic_data")
    return os.path.join(base, "craft")


def _now() -> str:
    return datetime.now(UTC).isoformat()


@dataclass
class CraftNote:
    """A single durable artifact authored by the being."""
    note_id: str
    being_id: str
    title: str
    body: str
    kind: str = "note"            # note | script | observation | reflection
    created_at: str = ""
    updated_at: str = ""
    tags: list[str] = field(default_factory=list)

    @property
    def content(self) -> str:
        return self.body


CraftEntry = CraftNote


def _sanitize_craft_id(val: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_-]", "_", str(val).strip())
    return cleaned or "default"


class BeingCraft:
    """Host-side durable craft store for a being.

    Files live under ``sonic_data/craft/<being_id>/``; an ``index.json`` lists
    metadata so the being can recall what it has authored without re-scanning.
    """

    def __init__(
        self,
        being_id: str = "default",
        root: str | Path | None = None,
        base_dir: str | Path | None = None,
    ):
        self.being_id = _sanitize_craft_id(being_id)
        chosen_root = root or base_dir
        self._root = str(chosen_root) if chosen_root is not None else os.path.join(_craft_root(), self.being_id)
        self._index: dict[str, CraftNote] = {}
        self._other_entries: list[dict] = []
        os.makedirs(self._root, exist_ok=True)
        self._load_index()

    # ------------------------------------------------------------------
    def _index_path(self) -> str:
        return os.path.join(self._root, "index.json")

    def _note_path(self, note_id: str) -> str:
        safe_note = _sanitize_craft_id(note_id)
        return os.path.join(self._root, f"{safe_note}.md")

    def _load_index(self) -> None:
        path = self._index_path()
        if not os.path.exists(path):
            return
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            self._other_entries = []
            for entry in data.get("notes", []):
                n = CraftNote(
                    note_id=entry["note_id"],
                    being_id=entry.get("being_id", self.being_id),
                    title=entry.get("title", ""),
                    body=entry.get("body", ""),
                    kind=entry.get("kind", "note"),
                    created_at=entry.get("created_at", ""),
                    updated_at=entry.get("updated_at", ""),
                    tags=list(entry.get("tags") or []),
                )
                if n.being_id == self.being_id:
                    self._index[n.note_id] = n
                else:
                    self._other_entries.append(entry)
        except Exception as e:
            logger.warning("craft_index_load_failed", being_id=self.being_id, error=str(e))

    def _save_index(self) -> None:
        own_notes = [n.__dict__ for n in self._index.values()]
        with open(self._index_path(), "w", encoding="utf-8") as f:
            json.dump({"notes": self._other_entries + own_notes}, f, indent=2)

    # ------------------------------------------------------------------
    def author(
        self,
        title: str,
        body: str = "",
        kind: str = "note",
        tags: list[str] | None = None,
        content: str | None = None,
    ) -> CraftNote:
        """Author a new durable note. Write-through to disk immediately."""
        actual_body = content if content is not None else body
        note_tags = [str(t) for t in (tags or []) if str(t).strip()]
        note_id = f"note-{int(time.time() * 1000)}-{len(self._index) + 1}"
        ts = _now()
        note = CraftNote(
            note_id=note_id,
            being_id=self.being_id,
            title=title,
            body=actual_body,
            kind=kind,
            created_at=ts,
            updated_at=ts,
            tags=note_tags,
        )
        # Write the note body (markdown, human-readable on disk).
        tags_line = f"_tags: {', '.join(note_tags)}_\n" if note_tags else ""
        with open(self._note_path(note_id), "w", encoding="utf-8") as f:
            f.write(f"# {title}\n\n_kind: {kind}_\n{tags_line}_created: {ts}_\n\n{actual_body}\n")
        self._index[note_id] = note
        self._save_index()
        logger.info("craft_authored", being_id=self.being_id,
                    note_id=note_id, kind=kind, title=title[:60])
        return note

    def save_note(
        self,
        title: str,
        body: str = "",
        kind: str = "note",
        tags: list[str] | None = None,
        content: str | None = None,
    ) -> CraftNote:
        """Alias for author(...)."""
        return self.author(title=title, body=body, kind=kind, tags=tags, content=content)

    def list_notes(self) -> list[CraftNote]:
        return list(self._index.values())

    def get_note(self, note_id: str) -> CraftNote | None:
        n = self._index.get(note_id)
        if n is None:
            return None
        # Re-read body from disk (authoritative).
        try:
            with open(self._note_path(note_id), encoding="utf-8") as f:
                n.body = f.read()
        except FileNotFoundError:
            pass
        return n

    def update_note(self, note_id: str, body: str | None = None,
                    title: str | None = None) -> CraftNote | None:
        n = self._index.get(note_id)
        if n is None:
            return None
        if body is not None:
            n.body = body
        if title is not None:
            n.title = title
        n.updated_at = _now()
        with open(self._note_path(note_id), "w", encoding="utf-8") as f:
            f.write(f"# {n.title}\n\n_kind: {n.kind}_\n_created: {n.created_at}_\n_updated: {n.updated_at}_\n\n{n.body}\n")
        self._save_index()
        return n
