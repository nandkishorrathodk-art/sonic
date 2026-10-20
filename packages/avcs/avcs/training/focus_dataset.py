"""Dataset support for recorded AVCS focus-verification demonstrations.

A focus demonstration is one recorded UI interaction that labels a screen point
as either focused (text entry active at that point) or not focused.  Frames are
real workstation pixels; the label is the recorded ground truth, never inferred
from a model.  Session-level splitting keeps one app/task out of both train and
held-out evaluation.
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Union

import numpy as np
import torch
from avcs.focus_verifier import focus_patch_rgb
from PIL import Image
from torch.utils.data import Dataset

FOCUS_DEMO_SCHEMA = 1


@dataclass(frozen=True)
class FocusDemoRecord:
    session_dir: Path
    frame_path: Path
    target_label: str
    point_px: tuple[int, int]
    focused: bool
    task: str
    target_window_title: str


def _load_focus_record(
    session_dir: Path,
    raw: dict[str, Any],
    *,
    task: str,
    target_window_title: str,
) -> FocusDemoRecord:
    if raw.get("schema_version") != FOCUS_DEMO_SCHEMA:
        raise ValueError(f"Unsupported focus demonstration schema in {session_dir}")
    target_label = raw.get("target_label")
    if not isinstance(target_label, str) or not target_label.strip():
        raise ValueError(f"Focus demonstration is missing a target_label in {session_dir}")
    focused = raw.get("focused")
    if not isinstance(focused, bool):
        raise ValueError(f"Focus demonstration is missing a boolean 'focused' in {session_dir}")
    point = raw.get("point_px")
    if (
        not isinstance(point, list)
        or len(point) != 2
        or any(isinstance(v, bool) or not isinstance(v, int) for v in point)
    ):
        raise ValueError(f"Focus demonstration has an invalid point_px in {session_dir}")
    frame_name = raw.get("frame")
    if not isinstance(frame_name, str) or not frame_name:
        raise ValueError(f"Focus demonstration is missing a frame reference in {session_dir}")
    frame_path = (session_dir / frame_name).resolve()
    if not frame_path.is_file() or not frame_path.is_relative_to(session_dir.resolve()):
        raise ValueError(f"Focus demonstration frame is missing or outside its session: {frame_name}")
    return FocusDemoRecord(
        session_dir=session_dir,
        frame_path=frame_path,
        target_label=target_label.strip(),
        point_px=(int(point[0]), int(point[1])),
        focused=focused,
        task=task,
        target_window_title=target_window_title,
    )


def discover_focus_demonstrations(
    root: Union[str, Path],  # noqa: UP007
) -> dict[str, list[FocusDemoRecord]]:
    """Load every recorder session under a root directory, keyed by session name."""
    root_path = Path(root).expanduser()
    if not root_path.is_dir():
        raise ValueError(f"Focus demonstration root does not exist: {root_path}")
    sessions: dict[str, list[FocusDemoRecord]] = {}
    for session_dir in sorted(p for p in root_path.iterdir() if p.is_dir()):
        manifest_path = session_dir / "manifest.json"
        records_path = session_dir / "records.jsonl"
        if not manifest_path.is_file() or not records_path.is_file():
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") != "complete":
            raise ValueError(f"Focus session {session_dir.name} is not complete")
        task = manifest.get("task")
        window_title = manifest.get("target_window_title")
        if not isinstance(task, str) or not isinstance(window_title, str):
            raise ValueError(f"Focus session {session_dir.name} has an invalid manifest")
        records: list[FocusDemoRecord] = []
        for line in records_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            records.append(
                _load_focus_record(
                    session_dir, raw, task=task, target_window_title=window_title
                )
            )
        if records:
            sessions[session_dir.name] = records
    if not sessions:
        raise ValueError(f"No focus demonstration sessions found under {root_path}")
    return sessions


def split_focus_sessions(
    sessions: dict[str, list[FocusDemoRecord]],
    *,
    seed: int = 42,
    train_fraction: float = 0.6,
    validation_fraction: float = 0.2,
) -> tuple[
    list[FocusDemoRecord], list[FocusDemoRecord], list[FocusDemoRecord]
]:
    """Split by session so one app/task cannot leak across the three sets."""
    names = sorted(sessions)
    if len(names) < 3:
        raise ValueError("Focus training requires at least three recording sessions")
    groups: dict[tuple[str, str], list[str]] = {}
    for name in names:
        records = sessions[name]
        key = (records[0].task.casefold(), records[0].target_window_title.casefold())
        groups.setdefault(key, []).append(name)
    if len(groups) < 3:
        raise ValueError("Focus training requires at least three distinct task/window groups")
    shuffled = list(groups)
    random.Random(seed).shuffle(shuffled)
    n_train = max(1, int(len(shuffled) * train_fraction))
    n_val = max(1, int(len(shuffled) * validation_fraction))
    if n_train + n_val >= len(shuffled):
        n_train = max(1, len(shuffled) - 2)
        n_val = 1
    train_names = [n for g in shuffled[:n_train] for n in groups[g]]
    val_names = [n for g in shuffled[n_train : n_train + n_val] for n in groups[g]]
    test_names = [n for g in shuffled[n_train + n_val :] for n in groups[g]]
    return (
        [r for n in train_names for r in sessions[n]],
        [r for n in val_names for r in sessions[n]],
        [r for n in test_names for r in sessions[n]],
    )


class FocusDemoDataset(Dataset):
    """Yield (patch_tensor, query_tokens, focused_label) from recorded demos."""

    def __init__(
        self, records: list[FocusDemoRecord], *, input_size: int = 96
    ) -> None:
        if not records:
            raise ValueError("FocusDemoDataset requires at least one record")
        self.records = records
        self.input_size = input_size
        self._cache: dict[Path, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self.records)

    def _frame(self, path: Path) -> np.ndarray:
        cached = self._cache.get(path)
        if cached is None:
            with Image.open(path) as image:
                cached = np.array(image.convert("RGB"))
            self._cache[path] = cached
        return cached

    def __getitem__(self, index: int) -> tuple[torch.Tensor, str, float]:
        record = self.records[index]
        frame = self._frame(record.frame_path)
        height, width = frame.shape[:2]
        x, y = record.point_px
        if not (0 <= x < width and 0 <= y < height):
            raise ValueError(f"Focus point is outside the frame for {record.frame_path.name}")
        patch = focus_patch_rgb(frame, x, y, input_size=self.input_size)
        return patch, record.target_label, float(record.focused)


def focus_collate(
    batch: list[tuple[torch.Tensor, str, float]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Collate patches, encoded queries, and float labels."""
    from avcs.goal_grounder import encode_queries

    patches = torch.stack([item[0] for item in batch])
    queries = encode_queries([item[1] for item in batch])
    labels = torch.tensor([item[2] for item in batch], dtype=torch.float32)
    return patches, queries, labels
