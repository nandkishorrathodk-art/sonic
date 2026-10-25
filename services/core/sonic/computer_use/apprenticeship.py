"""AVCS apprenticeship: the eye learns from the being's own work.

The AVCS cortex was trained once, offline, on frames an operator captured by
hand. It never improves after that. This module closes the loop so the being's
ordinary desktop work *becomes* its training data: every frame it looks at while
using a real application is a potential labeled sample, and the eye periodically
retrains and re-verifies itself on a held-out slice of that experience.

What makes the labels honest
----------------------------
A label is never guessed from pixels and never taken from a hardcoded vendor
name. It is derived from the real operating system: the active window's
``WM_CLASS`` is resolved to its freedesktop ``.desktop`` entry, whose published
``Categories=`` field (a standard: TerminalEmulator, WebBrowser, Development,
Settings, ...) is mapped to the cortex's own UI vocabulary by
:data:`_FREEDESKTOP_CATEGORY_TO_UI`. A window whose class cannot be resolved, or
whose categories map to nothing, yields **no** label -- it is skipped, not
invented. Every sample records its ``label_source`` so a later reader can see
exactly where the label came from.

The learn -> verify -> promote gate
-----------------------------------
Training a checkpoint is not the same as improving it. :meth:`learn` trains a
candidate on the training split, then measures the candidate **and** the
currently deployed checkpoint on the *same* held-out split (grouped so
near-duplicate frames never leak across the boundary). The candidate replaces
the deployed checkpoint only when it is at least as accurate and better than
chance. A regression keeps the current checkpoint and records the honest reason.
Nothing is promoted by decree.

This is deliberately distinct from the code self-evolution loop: that loop edits
source code, this one grows the eye's own model from lived experience.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Iterable, Optional

import numpy as np

from sonic.logger import get_logger

logger = get_logger(__name__)

# Cortex UI vocabulary (mirrors forge.vision.cortex.core.ventral.UI_CATEGORIES).
# Kept as a local tuple so this module imports without pulling torch in.
UI_CATEGORIES: tuple[str, ...] = (
    "desktop_canvas",
    "editor_code",
    "browser_webpage",
    "terminal_shell",
    "modal_dialog",
    "form_input",
    "media_playback",
    "settings_panel",
)

# Freedesktop categories -> cortex UI categories, most specific first so a
# window advertising several categories resolves deterministically. These keys
# are the *standard* category names any `.desktop` file may carry; they are not
# application names, and an unmapped category yields no label rather than a
# guess.
_FREEDESKTOP_CATEGORY_TO_UI: tuple[tuple[str, str], ...] = (
    ("TerminalEmulator", "terminal_shell"),
    ("WebBrowser", "browser_webpage"),
    ("Development", "editor_code"),
    ("IDE", "editor_code"),
    ("TextEditor", "editor_code"),
    ("FileManager", "form_input"),
    ("FileTools", "form_input"),
    ("Filesystem", "form_input"),
    ("Office", "form_input"),
    ("WordProcessor", "form_input"),
    ("Spreadsheet", "form_input"),
    ("Presentation", "form_input"),
    ("Settings", "settings_panel"),
    ("DesktopSettings", "settings_panel"),
    ("HardwareSettings", "settings_panel"),
    ("PersonalSettings", "settings_panel"),
    ("SystemSettings", "settings_panel"),
    ("AudioVideo", "media_playback"),
    ("Audio", "media_playback"),
    ("Video", "media_playback"),
    ("Player", "media_playback"),
    ("Graphics", "media_playback"),
    ("Viewer", "media_playback"),
    ("Photography", "media_playback"),
)

_FRAME_SIZE = (640, 360)
_CHANCE_ACCURACY = 1.0 / len(UI_CATEGORIES)


def parse_desktop_categories(desktop_text: str) -> list[str]:
    """Extract the ``Categories=`` list from a freedesktop ``.desktop`` file."""
    for raw in desktop_text.splitlines():
        line = raw.strip()
        if not line.startswith("Categories="):
            continue
        value = line.split("=", 1)[1]
        return [part.strip() for part in value.split(";") if part.strip()]
    return []


def freedesktop_categories_to_ui(categories: Iterable[str]) -> Optional[str]:
    """Map freedesktop categories to a cortex UI category, or ``None``.

    ``None`` means "this window does not advertise a category the cortex knows".
    It is deliberately not coerced into a default class: an invented label is
    worse than no sample.
    """
    present = {str(c).strip() for c in categories if str(c).strip()}
    for fd_category, ui_category in _FREEDESKTOP_CATEGORY_TO_UI:
        if fd_category in present:
            return ui_category
    return None


@dataclass
class ApprenticeshipSample:
    """One frame with its honestly-derived label."""

    frame: str  # relative path under the store's frames/ dir
    label: str
    wm_class: str
    label_source: str
    captured_at: float
    group: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame": self.frame,
            "label": self.label,
            "wm_class": self.wm_class,
            "label_source": self.label_source,
            "captured_at": self.captured_at,
            "group": self.group,
        }


class DesktopSceneLabeler:
    """Resolve the active window to an honest UI label via the real OS.

    The resolution chain is entirely OS-supplied: ``WM_CLASS`` (from X11) ->
    the matching ``.desktop`` file -> its ``Categories=`` field -> the cortex
    vocabulary. ``.desktop`` lookups are cached because the file set changes
    rarely. No application name is ever hardcoded.
    """

    def __init__(self, exec_runner: Callable[[str], Awaitable[str]] | None = None) -> None:
        self._exec = exec_runner
        # Two tiers. Identity keys (desktop-file basename, StartupWMClass) name
        # the application itself. Exec/TryExec keys only name the *binary* it
        # runs, which several desktop entries can share (an app and its settings
        # dialog both launch the same binary), so they are consulted second.
        self._identity_cache: dict[str, list[tuple[str, list[str]]]] = {}
        self._exec_cache: dict[str, list[tuple[str, list[str]]]] = {}
        self._category_map_loaded = False

    @staticmethod
    def _pick(candidates: list[tuple[str, list[str]]], key: str) -> Optional[list[str]]:
        """Choose among desktop entries sharing a key.

        Reverse-DNS application ids end in the application name
        (``org.xfce.mousepad``), so an entry whose final dotted segment equals
        the window class is the application itself rather than a sibling such as
        its settings dialog. Falls back to the shortest basename, which is the
        least-qualified (most likely the primary) entry.
        """
        if not candidates:
            return None
        for base, categories in candidates:
            if base.rsplit(".", 1)[-1] == key:
                return categories
        return min(candidates, key=lambda item: len(item[0]))[1]

    async def _load_category_map(self) -> None:
        if self._category_map_loaded or self._exec is None:
            return
        # One pass over every application directory. A window's WM_CLASS may
        # match the desktop file's basename, its StartupWMClass, or the basename
        # of its Exec/TryExec binary -- different packages advertise different
        # subsets. Keying on only the first two silently drops any app whose
        # desktop file omits StartupWMClass (e.g. org.xfce.mousepad.desktop),
        # so its frames are never labeled and the eye learns nothing from it.
        script = (
            "for d in /usr/share/applications /usr/local/share/applications "
            "~/.local/share/applications; do "
            'for f in "$d"/*.desktop; do [ -f "$f" ] || continue; '
            'b=$(basename "$f" .desktop); '
            'c=$(grep -m1 "^Categories=" "$f" | cut -d= -f2-); '
            'w=$(grep -m1 "^StartupWMClass=" "$f" | cut -d= -f2-); '
            'e=$(grep -m1 "^Exec=" "$f" | cut -d= -f2- | awk \'{print $1}\'); '
            't=$(grep -m1 "^TryExec=" "$f" | cut -d= -f2-); '
            'printf "%s\\t%s\\t%s\\t%s\\t%s\\n" "$b" "$w" "$e" "$t" "$c"; '
            "done; done 2>/dev/null"
        )
        try:
            output = await self._exec(script)
        except Exception as exc:  # pragma: no cover - environment dependent
            logger.warning("apprenticeship_desktop_scan_failed", error=str(exc))
            return
        for line in output.splitlines():
            parts = line.split("\t")
            if len(parts) < 5:
                continue
            base, wm_class, exec_field, try_exec, categories = parts[:5]
            parsed = [c.strip() for c in categories.split(";") if c.strip()]
            if not parsed:
                continue
            for raw_key in (base, wm_class):
                key = str(raw_key).strip().lower()
                if key:
                    self._identity_cache.setdefault(key, []).append((base.lower(), parsed))
            for raw_key in (exec_field, try_exec):
                # Exec/TryExec may carry a path or arguments; the window class is
                # the binary's basename.
                key = str(raw_key).strip().split("/")[-1].strip("\"'").lower()
                if key:
                    self._exec_cache.setdefault(key, []).append((base.lower(), parsed))
        self._category_map_loaded = True

    async def label_for(self, wm_class: str, has_active_window: bool) -> tuple[Optional[str], str]:
        """Return ``(ui_category_or_None, label_source)`` for a window class.

        ``has_active_window=False`` with an empty class means nothing is
        focused, which is the bare desktop -- a real observable, not a guess.
        """
        await self._load_category_map()
        normalized = (wm_class or "").strip().lower()
        if not normalized:
            if has_active_window:
                return None, "no_wm_class"
            return "desktop_canvas", "no_active_window"
        # WM_CLASS arrives as "instance.class"; the class is the second token.
        candidates = [normalized]
        if "." in normalized:
            candidates.append(normalized.split(".")[-1])
        for cache in (self._identity_cache, self._exec_cache):
            for candidate in candidates:
                categories = self._pick(cache.get(candidate, []), candidate)
                if categories:
                    ui = freedesktop_categories_to_ui(categories)
                    if ui:
                        return ui, f"freedesktop_categories:{','.join(categories)}"
        return None, "unmapped_wm_class"


@dataclass
class ApprenticeshipStatus:
    samples: int = 0
    labeled: int = 0
    rounds: int = 0
    promoted: int = 0
    last_result: str = ""
    checkpoint: str = ""
    deployed_accuracy: Optional[float] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "samples": self.samples,
            "labeled": self.labeled,
            "rounds": self.rounds,
            "promoted": self.promoted,
            "last_result": self.last_result,
            "checkpoint": self.checkpoint,
            "deployed_accuracy": self.deployed_accuracy,
            "ui_categories": list(UI_CATEGORIES),
        }


class ApprenticeshipStore:
    """Durable, inspectable store of (frame, honest label) samples.

    Mirrors the BeingCraft convention: plain files on the host filesystem under
    ``sonic_data/avcs/apprenticeship/`` so an operator can read exactly what the
    eye has learned from, without a database.
    """

    def __init__(self, root: str | Path | None = None) -> None:
        base = root or os.environ.get(
            "SONIC_AVCS_APPRENTICESHIP_DIR",
            os.path.join(os.environ.get("SONIC_DATA_DIR", "sonic_data"), "avcs", "apprenticeship"),
        )
        self.root = Path(base)
        self.frames_dir = self.root / "frames"
        self.manifest_path = self.root / "samples.json"

    def _load(self) -> list[dict[str, Any]]:
        if not self.manifest_path.is_file():
            return []
        try:
            data = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except Exception:
            return []
        samples = data.get("samples")
        return samples if isinstance(samples, list) else []

    def count(self) -> int:
        return len(self._load())

    def add(
        self,
        rgb: np.ndarray,
        label: str,
        wm_class: str,
        label_source: str,
        group: str = "",
    ) -> ApprenticeshipSample:
        """Persist one labeled frame (downscaled PNG) and return its record.

        ``group`` identifies one continuous look at a single screen state. The
        train/held-out split moves whole groups, so frames captured seconds
        apart from the same window never end up on both sides -- otherwise a
        held-out score would be measured on near-duplicates of training frames
        and mean nothing.
        """
        from PIL import Image

        self.frames_dir.mkdir(parents=True, exist_ok=True)
        samples = self._load()
        index = len(samples)
        name = f"frame_{index:05d}.png"
        image = Image.fromarray(rgb)
        if image.size != _FRAME_SIZE:
            image = image.resize(_FRAME_SIZE)
        image.save(self.frames_dir / name)
        sample = ApprenticeshipSample(
            frame=f"frames/{name}",
            label=label,
            wm_class=wm_class,
            label_source=label_source,
            captured_at=time.time(),
            group=group or f"{wm_class}:{index}",
        )
        samples.append(sample.to_dict())
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        self.manifest_path.write_text(
            json.dumps(
                {
                    "provenance": "live_desktop_apprenticeship",
                    "label_rule": "freedesktop_categories_from_wm_class",
                    "samples": samples,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return sample

    def label_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for item in self._load():
            label = str(item.get("label", ""))
            counts[label] = counts.get(label, 0) + 1
        return counts


def _grouped_split(
    samples: list[dict[str, Any]], held_out_fraction: float, seed: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split by (label, group) so near-duplicates never leak.

    Same discipline as ``scripts/split_desktop_dataset.py``: whole groups move
    to one side, and a class with a single group stays in training so every
    held-out class is also present in training. ``group`` is one continuous look
    at a screen state; older records without one fall back to their wm_class.
    """
    import random
    from collections import defaultdict

    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for item in samples:
        group = str(item.get("group") or item.get("wm_class") or "")
        groups[(str(item.get("label", "")), group)].append(item)

    by_label: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for key in groups:
        by_label[key[0]].append(key)

    rng = random.Random(seed)
    held_keys: set[tuple[str, str]] = set()
    for label in sorted(by_label):
        keys = sorted(by_label[label])
        rng.shuffle(keys)
        if len(keys) >= 2:
            n_held = max(1, round(len(keys) * held_out_fraction))
            n_held = min(n_held, len(keys) - 1)
        else:
            n_held = 0
        held_keys.update(keys[:n_held])

    train: list[dict[str, Any]] = []
    held: list[dict[str, Any]] = []
    for key, items in groups.items():
        (held if key in held_keys else train).extend(items)
    return train, held


@dataclass
class LearnResult:
    promoted: bool
    reason: str
    candidate_accuracy: float = 0.0
    deployed_accuracy: Optional[float] = None
    train_frames: int = 0
    heldout_frames: int = 0
    checkpoint: str = ""
    per_class: dict[str, dict[str, int]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "promoted": self.promoted,
            "reason": self.reason,
            "candidate_accuracy": self.candidate_accuracy,
            "deployed_accuracy": self.deployed_accuracy,
            "train_frames": self.train_frames,
            "heldout_frames": self.heldout_frames,
            "checkpoint": self.checkpoint,
            "per_class": self.per_class,
        }


class AVCSApprenticeshipLoop:
    """The eye's self-learning loop: capture -> learn -> verify -> promote.

    ``capture_round`` samples the live desktop and records honestly-labeled
    frames. ``learn`` trains a candidate cortex on the accumulated experience
    and deploys it only when it genuinely beats the current checkpoint on
    held-out frames. Both fail closed: a capture or training error leaves the
    deployed eye untouched.
    """

    def __init__(
        self,
        *,
        store: ApprenticeshipStore | None = None,
        labeler: DesktopSceneLabeler | None = None,
        checkpoint_path: str | Path | None = None,
        held_out_fraction: float = 0.25,
        seed: int = 7,
        epochs: int = 6,
        perception_engine: Any | None = None,
    ) -> None:
        self.store = store or ApprenticeshipStore()
        self.labeler = labeler or DesktopSceneLabeler()
        self.checkpoint_path = Path(
            checkpoint_path
            or os.environ.get(
                "SONIC_AVCS_APPRENTICESHIP_CHECKPOINT",
                str(self.store.root / "cortex_learned.pt"),
            )
        )
        self.held_out_fraction = held_out_fraction
        self.seed = seed
        self.epochs = epochs
        self._perception_engine = perception_engine
        self.status = ApprenticeshipStatus(checkpoint=str(self.checkpoint_path))
        self._lock = asyncio.Lock()

    # -- capture ------------------------------------------------------------

    async def capture_round(
        self,
        frame_source: Callable[[], Awaitable[tuple[Optional[np.ndarray], str, bool]]],
        max_samples: int = 8,
        interval_s: float = 0.5,
    ) -> int:
        """Sample the desktop ``max_samples`` times, keeping only labeled frames.

        ``frame_source`` returns ``(rgb_or_None, wm_class, has_active_window)``.
        A frame whose window cannot be honestly labeled is skipped -- it is not
        stored with a guessed class.

        Each call is one *group*: a continuous look at the desktop. The split
        later keeps whole groups together, so frames from this look never appear
        on both sides of the train/held-out boundary.
        """
        group = f"{time.time():.3f}"
        stored = 0
        for _ in range(max(0, max_samples)):
            try:
                rgb, wm_class, has_active_window = await frame_source()
            except Exception as exc:
                logger.warning("apprenticeship_frame_source_failed", error=str(exc))
                break
            if rgb is None:
                await asyncio.sleep(interval_s)
                continue
            label, source = await self.labeler.label_for(wm_class, has_active_window)
            if label is None:
                logger.debug(
                    "apprenticeship_sample_unlabeled", wm_class=wm_class, reason=source
                )
                await asyncio.sleep(interval_s)
                continue
            try:
                self.store.add(rgb, label, wm_class, source, group=group)
                stored += 1
            except Exception as exc:
                logger.warning("apprenticeship_store_failed", error=str(exc))
                break
            await asyncio.sleep(interval_s)
        self.status.samples = self.store.count()
        return stored

    # -- learn --------------------------------------------------------------

    async def learn(self) -> LearnResult:
        """Train a candidate on lived experience; deploy only on real gain."""
        async with self._lock:
            return await asyncio.to_thread(self._learn_sync)

    def _learn_sync(self) -> LearnResult:
        samples = self.store._load()
        self.status.samples = len(samples)
        if len(samples) < 4:
            result = LearnResult(
                promoted=False,
                reason=f"insufficient samples ({len(samples)}); need >= 4",
            )
            self.status.last_result = result.reason
            return result

        train, held = _grouped_split(samples, self.held_out_fraction, self.seed)
        if not train or not held:
            result = LearnResult(
                promoted=False,
                reason="split produced an empty train or held-out set",
                train_frames=len(train),
                heldout_frames=len(held),
            )
            self.status.last_result = result.reason
            return result
        train_labels = {str(s.get("label")) for s in train}
        held_labels = {str(s.get("label")) for s in held}
        if held_labels - train_labels:
            result = LearnResult(
                promoted=False,
                reason=f"held-out classes absent from train: {sorted(held_labels - train_labels)}",
                train_frames=len(train),
                heldout_frames=len(held),
            )
            self.status.last_result = result.reason
            return result

        split_dir = self.store.root
        split_dir.mkdir(parents=True, exist_ok=True)
        train_manifest = split_dir / "train.json"
        held_manifest = split_dir / "heldout.json"
        train_manifest.write_text(
            json.dumps({"provenance": "live_desktop_apprenticeship", "frames": train}, indent=2),
            encoding="utf-8",
        )
        held_manifest.write_text(
            json.dumps({"provenance": "live_desktop_apprenticeship", "frames": held}, indent=2),
            encoding="utf-8",
        )

        try:
            self._train_candidate(train_manifest)
        except Exception as exc:
            result = LearnResult(
                promoted=False,
                reason=f"candidate training failed: {exc}",
                train_frames=len(train),
                heldout_frames=len(held),
            )
            self.status.last_result = result.reason
            logger.warning("apprenticeship_train_failed", error=str(exc))
            return result

        candidate_path = self.store.root / "runs" / "candidate.pt"
        try:
            candidate_accuracy, per_class = self._evaluate_on_manifest(
                candidate_path, held_manifest
            )
        except Exception as exc:
            result = LearnResult(
                promoted=False,
                reason=f"candidate evaluation failed: {exc}",
                train_frames=len(train),
                heldout_frames=len(held),
            )
            self.status.last_result = result.reason
            return result

        deployed_accuracy: Optional[float] = None
        if self.checkpoint_path.is_file():
            try:
                deployed_accuracy, _ = self._evaluate_on_manifest(
                    self.checkpoint_path, held_manifest
                )
            except Exception:
                deployed_accuracy = None

        gate = candidate_accuracy > _CHANCE_ACCURACY and (
            deployed_accuracy is None or candidate_accuracy >= deployed_accuracy
        )
        self.status.rounds += 1
        self.status.deployed_accuracy = (
            candidate_accuracy if gate else deployed_accuracy
        )
        if not gate:
            reason = (
                f"candidate {candidate_accuracy:.3f} did not beat deployed "
                f"{deployed_accuracy if deployed_accuracy is None else round(deployed_accuracy, 3)}"
            )
            result = LearnResult(
                promoted=False,
                reason=reason,
                candidate_accuracy=candidate_accuracy,
                deployed_accuracy=deployed_accuracy,
                train_frames=len(train),
                heldout_frames=len(held),
                per_class=per_class,
            )
            self.status.last_result = reason
            return result

        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(candidate_path, self.checkpoint_path)
        self._write_sidecar(candidate_accuracy, per_class, len(held))
        self._reload_engine()
        self.status.promoted += 1
        reason = f"promoted candidate {candidate_accuracy:.3f} (deployed {deployed_accuracy})"
        self.status.last_result = reason
        logger.info(
            "avcs_apprenticeship_promoted",
            accuracy=candidate_accuracy,
            deployed=deployed_accuracy,
            checkpoint=str(self.checkpoint_path),
        )
        return LearnResult(
            promoted=True,
            reason=reason,
            candidate_accuracy=candidate_accuracy,
            deployed_accuracy=deployed_accuracy,
            train_frames=len(train),
            heldout_frames=len(held),
            checkpoint=str(self.checkpoint_path),
            per_class=per_class,
        )

    # -- internals ----------------------------------------------------------

    def _train_candidate(self, train_manifest: Path) -> None:
        from torch.utils.data import DataLoader

        from forge.vision.cortex.config import CortexConfig
        from forge.vision.cortex.data.labeled_dataset import LabeledFrameDataset
        from forge.vision.cortex.training.trainer import AVCSCloudTrainer

        dataset = LabeledFrameDataset(train_manifest)
        cfg = CortexConfig()
        cfg.training.epochs = self.epochs
        cfg.training.device = "cpu"
        loader = DataLoader(dataset, batch_size=min(8, len(dataset)), shuffle=True, num_workers=0)
        trainer = AVCSCloudTrainer(config=cfg, dataloader=loader)
        out = self.store.root / "runs" / "candidate.pt"
        out.parent.mkdir(parents=True, exist_ok=True)
        trainer.train(epochs=self.epochs, save_path=str(out))

    def _evaluate_on_manifest(
        self, checkpoint: Path, manifest: Path
    ) -> tuple[float, dict[str, dict[str, int]]]:
        """Real held-out category accuracy of ``checkpoint`` on real frames."""
        import torch
        from PIL import Image

        from forge.vision.cortex.config import ModelConfig
        from forge.vision.cortex.model import ArtificialVisualCortex

        data = json.loads(manifest.read_text(encoding="utf-8"))
        frames = data.get("frames", [])
        model = ArtificialVisualCortex(ModelConfig())
        state = torch.load(checkpoint, map_location="cpu")
        model.load_state_dict(state, strict=False)
        model.eval()
        model.reset_temporal_memory()

        # Frame paths are relative to the store root, which is also where the
        # split manifests are written, so the manifest's own directory is the
        # correct base.
        root = manifest.parent
        correct = 0
        total = 0
        per_class: dict[str, dict[str, int]] = {}
        with torch.no_grad():
            for item in frames:
                label = str(item.get("label", ""))
                if label not in UI_CATEGORIES:
                    continue
                target = UI_CATEGORIES.index(label)
                path = root / str(item.get("frame", ""))
                if not path.is_file():
                    continue
                img = Image.open(path).convert("RGB")
                if img.size != _FRAME_SIZE:
                    img = img.resize(_FRAME_SIZE)
                arr = np.asarray(img, dtype=np.uint8)
                frame = torch.from_numpy(arr).permute(2, 0, 1).float().unsqueeze(0) / 255.0
                view = model.retina.extract_views(frame)
                out = model.forward(view.peripheral, view.foveal)
                pred = int(out["category_logits"].argmax(dim=-1).item())
                total += 1
                bucket = per_class.setdefault(label, {"correct": 0, "total": 0})
                bucket["total"] += 1
                if pred == target:
                    correct += 1
                    bucket["correct"] += 1
        if total == 0:
            raise ValueError("held-out manifest produced no evaluable frames")
        return correct / total, per_class

    def _write_sidecar(
        self, accuracy: float, per_class: dict[str, dict[str, int]], samples: int
    ) -> None:
        sidecar = self.checkpoint_path.with_suffix(".eval.json")
        sidecar.write_text(
            json.dumps(
                {
                    "checkpoint": self.checkpoint_path.name,
                    "dataset_provenance": "live desktop apprenticeship (real frames, freedesktop labels)",
                    "samples": samples,
                    "category_accuracy": accuracy,
                    "chance_accuracy": round(_CHANCE_ACCURACY, 4),
                    "per_class": per_class,
                    "promoted_at": time.time(),
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    def _reload_engine(self) -> None:
        engine = self._perception_engine
        if engine is None:
            return
        reload_fn = getattr(engine, "reload_cortex", None)
        if callable(reload_fn):
            try:
                reload_fn()
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("avcs_apprenticeship_reload_failed", error=str(exc))


def build_provider_labeler(provider: Any) -> DesktopSceneLabeler:
    """Bind a :class:`DesktopSceneLabeler` to a provider's sandbox exec.

    The labeler needs to read ``.desktop`` files *inside* the workstation to
    resolve a window class to its real categories. Without an exec runner it can
    never build that map and would honestly label nothing -- so the runner is
    wired from the same provider that supplies the frames.
    """
    exec_fn = getattr(provider, "_docker_exec", None)
    if not callable(exec_fn):
        return DesktopSceneLabeler()

    async def _run(cmd: str) -> str:
        result = await exec_fn(cmd)
        if isinstance(result, tuple):
            return result[1] if len(result) > 1 else ""
        return str(result or "")

    return DesktopSceneLabeler(exec_runner=_run)


def build_provider_frame_source(
    provider: Any, workspace_id: str, labeler: DesktopSceneLabeler
) -> Callable[[], Awaitable[tuple[Optional[np.ndarray], str, bool]]]:
    """Adapt a live ``ComputerProvider`` into an apprenticeship frame source.

    The frame comes from the provider's own screenshot; the window class comes
    from its ``WM_CLASS`` probe. When the provider cannot report a class the
    sample is simply skipped by the loop (never guessed).
    """

    async def _source() -> tuple[Optional[np.ndarray], str, bool]:
        screen = await provider.screenshot(workspace_id)
        if screen is None or getattr(screen, "desktop_state", None) != "INTERACTIVE":
            return None, "", False
        raw = getattr(screen, "raw_rgb", b"")
        rgb: Optional[np.ndarray] = None
        if raw:
            height = int(getattr(screen, "height", 0) or 0)
            width = int(getattr(screen, "width", 0) or 0)
            if height > 0 and width > 0 and len(raw) == height * width * 3:
                rgb = np.frombuffer(raw, dtype=np.uint8).reshape(height, width, 3).copy()
        if rgb is None:
            return None, "", False
        wm_class = ""
        get_class = getattr(provider, "get_active_window_class", None)
        if callable(get_class):
            try:
                wm_class = await get_class(workspace_id)
            except Exception:
                wm_class = ""
        active_window = str(getattr(screen, "active_window", "") or "")
        has_active_window = bool(wm_class) or (
            active_window not in ("", "Desktop", "None")
        )
        return rgb, wm_class, has_active_window

    return _source
