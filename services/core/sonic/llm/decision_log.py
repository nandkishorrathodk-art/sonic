"""Durable decision log for the Laya System-1 engine.

Fine-tuning a decision head needs ``(state, question, correct answer)`` triples.
SONIC makes thousands of typed decisions a day but never persisted the ground
truth, so there was nothing to train on. This module records each decision and,
separately, its later-observed outcome.

Two files, both append-only JSONL:

* ``decisions.jsonl`` — what the engine saw and predicted.
* ``outcomes.jsonl`` — what actually happened, keyed by ``record_id``.

The split exists because the label is not known when the decision is made. A
routing choice is only correct once the chosen plane is seen to work; a safety
verdict is only correct once the action's real effect is known. Annotating in a
second file keeps the hot path a single append and never rewrites the log.

``export_training_jsonl`` joins the two into the ``state`` / ``questions`` /
``expected`` rows ``laya-train`` consumes, so the loop is
log -> annotate -> export -> train.

Recording is OFF unless ``SONIC_LAYA_DECISION_LOG=1``. States can contain
screen text, terminal output and file paths, so every record passes through the
same secret scrubber that protects log output before it is written.
"""

from __future__ import annotations

import json
import os
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Iterator

from sonic.logger import get_logger, scrub_secrets

logger = get_logger(__name__)

SCHEMA_VERSION = 1

_DEFAULT_DIR = "sonic_data/laya_decisions"
_DEFAULT_MAX_MB = 256.0
_DEFAULT_MAX_STATE_CHARS = 8000


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return default
    return value if value > 0 else default


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return default
    return value if value > 0 else default


class DecisionLog:
    """Append-only recorder for System-1 decisions and their later outcomes."""

    def __init__(
        self,
        directory: str | Path | None = None,
        *,
        enabled: bool | None = None,
        max_megabytes: float | None = None,
        max_state_chars: int | None = None,
    ) -> None:
        configured_dir = directory or os.environ.get("SONIC_LAYA_DECISION_LOG_PATH") or _DEFAULT_DIR
        self.directory = Path(configured_dir)
        self.enabled = _env_flag("SONIC_LAYA_DECISION_LOG", False) if enabled is None else bool(enabled)
        self.max_bytes = int(
            (max_megabytes if max_megabytes is not None else _env_float("SONIC_LAYA_DECISION_LOG_MAX_MB", _DEFAULT_MAX_MB))
            * 1024
            * 1024
        )
        self.max_state_chars = (
            max_state_chars
            if max_state_chars is not None
            else _env_int("SONIC_LAYA_DECISION_LOG_MAX_STATE_CHARS", _DEFAULT_MAX_STATE_CHARS)
        )
        self._lock = threading.Lock()
        self._written = 0
        self._skipped = 0
        self._rotations = 0
        self._last_error = ""

    # -- paths -------------------------------------------------------------
    @property
    def decisions_path(self) -> Path:
        return self.directory / "decisions.jsonl"

    @property
    def outcomes_path(self) -> Path:
        return self.directory / "outcomes.jsonl"

    # -- helpers -----------------------------------------------------------
    def _prepare_state(self, state: Any) -> Any:
        """Scrub secrets and bound the size of a recorded state."""
        if isinstance(state, str):
            text = state
        else:
            try:
                text = json.dumps(state, ensure_ascii=False, default=str)
            except (TypeError, ValueError):
                text = str(state)
        scrubbed = scrub_secrets(text)
        if not isinstance(scrubbed, str):  # pragma: no cover - scrubber returns str for str
            scrubbed = str(scrubbed)
        if len(scrubbed) > self.max_state_chars:
            scrubbed = scrubbed[: self.max_state_chars] + "...[truncated]"
        return scrubbed

    def _rotate_if_needed(self, path: Path) -> None:
        """Bound disk use by keeping one previous generation of a log file."""
        try:
            if not path.exists() or path.stat().st_size < self.max_bytes:
                return
            backup = path.with_suffix(path.suffix + ".1")
            if backup.exists():
                backup.unlink()
            path.rename(backup)
            self._rotations += 1
            logger.warning(
                "laya_decision_log_rotated",
                path=str(path),
                max_mb=round(self.max_bytes / (1024 * 1024), 1),
            )
        except OSError as exc:
            self._last_error = f"rotation failed: {exc}"

    def _append(self, path: Path, record: dict[str, Any]) -> bool:
        """Append one JSON line. Never raises: a logging failure must not break a decision."""
        try:
            with self._lock:
                self.directory.mkdir(parents=True, exist_ok=True)
                self._rotate_if_needed(path)
                line = json.dumps(record, ensure_ascii=False, default=str)
                with path.open("a", encoding="utf-8") as handle:
                    handle.write(line + "\n")
                self._written += 1
            return True
        except (OSError, TypeError, ValueError) as exc:
            self._skipped += 1
            self._last_error = str(exc)
            logger.warning("laya_decision_log_write_failed", path=str(path), error=str(exc))
            return False

    # -- recording ---------------------------------------------------------
    def log_decision(
        self,
        *,
        state: Any,
        questions: dict[str, dict[str, Any]],
        answers: dict[str, dict[str, Any]],
        backend: str,
        model: str = "",
        source: str = "predict",
        routing: dict[str, Any] | None = None,
    ) -> str:
        """Record one decision and return its ``record_id`` ("" when not stored).

        Returns "" when logging is disabled or the append failed, so a caller
        never annotates an outcome against a record that does not exist.
        """
        if not self.enabled:
            return ""
        record_id = uuid.uuid4().hex
        record = {
            "schema_version": SCHEMA_VERSION,
            "record_id": record_id,
            "timestamp": time.time(),
            "backend": backend,
            "model": model,
            "source": source,
            "state": self._prepare_state(state),
            "questions": scrub_secrets(questions),
            "answers": scrub_secrets(answers),
            "routing": scrub_secrets(routing or {}),
        }
        return record_id if self._append(self.decisions_path, record) else ""

    def record_outcome(
        self,
        record_id: str,
        *,
        expected: dict[str, Any] | None = None,
        correct: bool | None = None,
        note: str = "",
    ) -> bool:
        """Attach the observed outcome for an earlier decision.

        ``expected`` carries the ground-truth answer per question when it is
        known, which is what makes a row trainable. ``correct`` records a bare
        right/wrong verdict when only that is observable. Either or both may be
        supplied.
        """
        if not self.enabled:
            return False
        if not record_id:
            return False
        record = {
            "schema_version": SCHEMA_VERSION,
            "record_id": record_id,
            "timestamp": time.time(),
            "expected": scrub_secrets(expected) if expected is not None else None,
            "correct": correct,
            "note": self._prepare_state(note) if note else "",
        }
        return self._append(self.outcomes_path, record)

    # -- reading -----------------------------------------------------------
    def _iter_lines(self, path: Path) -> Iterator[dict[str, Any]]:
        if not path.exists():
            return
        try:
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        parsed = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(parsed, dict):
                        yield parsed
        except OSError as exc:
            logger.warning("laya_decision_log_read_failed", path=str(path), error=str(exc))

    def iter_decisions(self) -> Iterator[dict[str, Any]]:
        yield from self._iter_lines(self.decisions_path)

    def iter_outcomes(self) -> Iterator[dict[str, Any]]:
        yield from self._iter_lines(self.outcomes_path)

    def stats(self) -> dict[str, Any]:
        decisions = sum(1 for _ in self.iter_decisions())
        outcomes = sum(1 for _ in self.iter_outcomes())
        return {
            "enabled": self.enabled,
            "directory": str(self.directory),
            "decisions": decisions,
            "outcomes": outcomes,
            "labelled": min(decisions, outcomes),
            "unlabelled": max(0, decisions - outcomes),
            "written_this_process": self._written,
            "skipped_this_process": self._skipped,
            "rotations_this_process": self._rotations,
            "last_error": self._last_error,
        }

    # -- export ------------------------------------------------------------
    def export_training_jsonl(self, destination: str | Path, *, require_labels: bool = True) -> dict[str, Any]:
        """Join decisions with outcomes into ``laya-train``-compatible rows.

        Only rows with a real label are exported by default: a decision whose
        outcome was never observed has no ground truth, and inventing one would
        train the head on fiction.
        """
        dest = Path(destination)
        outcomes: dict[str, dict[str, Any]] = {}
        for outcome in self.iter_outcomes():
            record_id = outcome.get("record_id")
            if isinstance(record_id, str) and record_id:
                outcomes[record_id] = outcome

        written = 0
        skipped_unlabelled = 0
        try:
            dest.parent.mkdir(parents=True, exist_ok=True)
            with dest.open("w", encoding="utf-8") as handle:
                for decision in self.iter_decisions():
                    record_id = decision.get("record_id")
                    outcome = outcomes.get(record_id) if isinstance(record_id, str) else None
                    expected = (outcome or {}).get("expected")
                    if expected is None:
                        if require_labels:
                            skipped_unlabelled += 1
                            continue
                        expected = {}
                    row = {
                        "state": decision.get("state", ""),
                        "questions": decision.get("questions", {}),
                        "expected": expected,
                        "_record_id": record_id,
                        "_source": decision.get("source", ""),
                        "_backend": decision.get("backend", ""),
                    }
                    handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
                    written += 1
        except OSError as exc:
            logger.warning("laya_decision_export_failed", path=str(dest), error=str(exc))
            return {"written": 0, "skipped_unlabelled": skipped_unlabelled, "error": str(exc)}

        return {
            "written": written,
            "skipped_unlabelled": skipped_unlabelled,
            "destination": str(dest),
        }


_decision_log: DecisionLog | None = None
_decision_log_lock = threading.Lock()


def get_decision_log() -> DecisionLog:
    """Return the process-wide decision log (created lazily)."""
    global _decision_log
    if _decision_log is None:
        with _decision_log_lock:
            if _decision_log is None:
                _decision_log = DecisionLog()
    return _decision_log


def reset_decision_log() -> None:
    """Drop the cached log so the next access re-reads the environment."""
    global _decision_log
    with _decision_log_lock:
        _decision_log = None
