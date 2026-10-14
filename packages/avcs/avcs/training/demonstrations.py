"""Dataset support for human-recorded AVCS click demonstrations."""
from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import torch
from avcs.goal_grounder import (
    DEFAULT_INPUT_SIZE,
    click_heatmap,
    encode_queries,
    letterbox_rgb,
)
from PIL import Image
from torch.utils.data import Dataset


@dataclass(frozen=True)
class DemoRecord:
    session_dir: Path
    frame_path: Path
    target_label: str
    point_norm: tuple[float, float]
    task: str
    target_window_title: str


def _load_record(
    session_dir: Path,
    raw: dict[str, Any],
    *,
    task: str,
    target_window_title: str,
) -> DemoRecord:
    if raw.get("schema_version") != 1:
        raise ValueError(f"Unsupported demonstration schema in {session_dir}")
    action = raw.get("action")
    if not isinstance(action, dict) or action.get("type") != "mouse_click":
        raise ValueError(f"Only mouse_click demonstrations are supported in {session_dir}")
    if action.get("button") != "left":
        raise ValueError(f"Only left-click demonstrations are supported in {session_dir}")
    target_label = raw.get("target_label")
    if not isinstance(target_label, str) or not target_label.strip():
        raise ValueError(f"Demonstration is missing a target_label in {session_dir}")
    point = action.get("point_norm")
    if (
        not isinstance(point, list)
        or len(point) != 2
        or any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in point)
    ):
        raise ValueError(f"Invalid point_norm in {session_dir}")
    x, y = float(point[0]), float(point[1])
    if not (math.isfinite(x) and math.isfinite(y) and 0 <= x < 1 and 0 <= y < 1):
        raise ValueError(f"point_norm must be finite and in [0, 1) in {session_dir}")
    frame_value = raw.get("frame")
    if not isinstance(frame_value, str) or not frame_value:
        raise ValueError(f"Demonstration is missing a frame path in {session_dir}")
    frame_path = (session_dir / frame_value).resolve()
    try:
        frame_path.relative_to(session_dir.resolve())
    except ValueError as exc:
        raise ValueError(f"Frame path escapes its demonstration session: {frame_value}") from exc
    if not frame_path.is_file():
        raise FileNotFoundError(f"Demonstration frame does not exist: {frame_path}")
    return DemoRecord(
        session_dir,
        frame_path,
        target_label.strip(),
        (x, y),
        task,
        target_window_title,
    )


def discover_demonstrations(
    root: Union[str, Path],  # noqa: UP007
) -> dict[Path, list[DemoRecord]]:
    """Read recorder sessions while keeping every frame inside its session."""
    root_path = Path(root).expanduser().resolve()
    if not root_path.is_dir():
        raise FileNotFoundError(f"Demonstration directory does not exist: {root_path}")
    sessions: dict[Path, list[DemoRecord]] = {}
    for samples_path in sorted(root_path.rglob("samples.jsonl")):
        manifest_path = samples_path.parent / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Recorder manifest is missing: {manifest_path}")
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid recorder manifest: {manifest_path}") from exc
        if not isinstance(manifest, dict):
            raise TypeError(f"Recorder manifest must be a JSON object: {manifest_path}")
        if (
            manifest.get("schema_version") != 1
            or manifest.get("status") != "stopped"
            or manifest.get("capture_scope") != "cropped target window only"
            or manifest.get("keyboard_recording") is not False
            or manifest.get("network_upload") is not False
            or not isinstance(manifest.get("target_window_title"), str)
            or not manifest["target_window_title"].strip()
            or not isinstance(manifest.get("task"), str)
            or not manifest["task"].strip()
        ):
            raise ValueError(f"Recorder manifest does not meet dataset requirements: {manifest_path}")
        records: list[DemoRecord] = []
        with samples_path.open(encoding="utf-8") as samples_file:
            for line_number, line in enumerate(samples_file, start=1):
                if not line.strip():
                    continue
                try:
                    raw = json.loads(line)
                    if not isinstance(raw, dict):
                        raise TypeError("sample must be a JSON object")
                    records.append(
                        _load_record(
                            samples_path.parent,
                            raw,
                            task=manifest["task"].strip(),
                            target_window_title=manifest["target_window_title"].strip(),
                        )
                    )
                except (json.JSONDecodeError, ValueError, TypeError, FileNotFoundError) as exc:
                    raise ValueError(f"{samples_path}:{line_number}: {exc}") from exc
        sample_count = manifest.get("sample_count")
        if (
            isinstance(sample_count, bool)
            or not isinstance(sample_count, int)
            or sample_count != len(records)
        ):
            raise ValueError(
                f"Recorder manifest sample_count does not match {samples_path}"
            )
        if records:
            sessions[samples_path.parent] = records
    if not sessions:
        raise ValueError(f"No labeled click demonstrations found under {root_path}")
    return sessions


def split_sessions(
    sessions: dict[Path, list[DemoRecord]],
    *,
    seed: int = 42,
) -> tuple[list[DemoRecord], list[DemoRecord], list[DemoRecord]]:
    """Split by task/window group so repeated recordings cannot leak across sets."""
    groups: dict[tuple[str, str], list[Path]] = {}
    for path, records in sessions.items():
        if not records:
            continue
        key = (
            records[0].task.casefold().strip(),
            records[0].target_window_title.casefold().strip(),
        )
        groups.setdefault(key, []).append(path)
    group_keys = list(groups)
    if len(group_keys) < 5:
        raise ValueError(
            "At least 5 distinct task/window groups are required for leakage-free "
            "train/validation/test splits with at least 3 training groups."
        )
    random.Random(seed).shuffle(group_keys)
    test_count = max(1, round(len(group_keys) * 0.15))
    validation_count = max(1, round(len(group_keys) * 0.15))
    if test_count + validation_count >= len(group_keys):
        raise ValueError("Not enough task/window groups to create all three splits.")
    test_groups = set(group_keys[:test_count])
    validation_groups = set(group_keys[test_count : test_count + validation_count])
    train_groups = set(group_keys[test_count + validation_count :])
    if (
        train_groups & validation_groups
        or train_groups & test_groups
        or validation_groups & test_groups
    ):
        raise RuntimeError("Task/window group split unexpectedly overlaps")

    def flatten(selected: set[tuple[str, str]]) -> list[DemoRecord]:
        return [
            record
            for key in sorted(selected)
            for path in sorted(groups[key])
            for record in sessions[path]
        ]

    return flatten(train_groups), flatten(validation_groups), flatten(test_groups)


class DemonstrationClickDataset(Dataset):
    """Letterboxed RGB frames paired with target labels and click-point targets."""

    def __init__(
        self,
        records: list[DemoRecord],
        *,
        input_size: int = DEFAULT_INPUT_SIZE,
        output_size: Optional[int] = None,  # noqa: UP045
    ) -> None:
        if not records:
            raise ValueError("A demonstration dataset split must not be empty")
        self.records = records
        self.input_size = input_size
        self.output_size = output_size or input_size // 2
        if self.output_size <= 0:
            raise ValueError("output_size must be positive")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        record = self.records[index]
        with Image.open(record.frame_path) as image:
            rgb = image.convert("RGB")
            image_tensor, transform = letterbox_rgb(
                np.asarray(rgb), self.input_size
            )
        canvas_point = transform.original_to_canvas(record.point_norm)
        return {
            "image": image_tensor,
            "query_tokens": encode_queries([record.target_label])[0],
            "target_heatmap": click_heatmap(canvas_point, (self.output_size, self.output_size)),
            "point_norm": torch.tensor(record.point_norm, dtype=torch.float32),
            "letterbox": torch.tensor(
                [
                    transform.left_norm,
                    transform.top_norm,
                    transform.width_norm,
                    transform.height_norm,
                ],
                dtype=torch.float32,
            ),
        }
