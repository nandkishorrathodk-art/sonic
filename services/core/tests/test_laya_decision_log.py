"""Tests for the Laya System-1 decision log.

The log exists so a fine-tuning corpus can be built from real decisions. These
tests cover the properties that matter: it is off unless asked, it never leaks a
secret to disk, it never breaks a decision when writing fails, and its export
contains only genuinely labelled rows.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sonic.llm.decision_log import DecisionLog, get_decision_log, reset_decision_log
from sonic.llm.laya_engine import LayaSystem1Engine

CHOICE_Q = {
    "plane": {
        "type": "choice",
        "instructions": "Which execution plane?",
        "criteria": {"gui": "desktop", "headless": "terminal"},
    }
}


@pytest.fixture
def log(tmp_path: Path) -> DecisionLog:
    return DecisionLog(tmp_path / "decisions", enabled=True)


def _decision(log: DecisionLog, state: str = "open the browser and click Save"):
    return log.log_decision(
        state=state,
        questions=CHOICE_Q,
        answers={"plane": {"choice": "gui", "probability": 0.8}},
        backend="laya_local",
        model="english",
    )


class TestDisabledByDefault:
    def test_environment_default_is_off(self, tmp_path, monkeypatch):
        monkeypatch.delenv("SONIC_LAYA_DECISION_LOG", raising=False)
        assert DecisionLog(tmp_path).enabled is False

    def test_disabled_log_writes_nothing(self, tmp_path):
        log = DecisionLog(tmp_path / "d", enabled=False)
        assert log.log_decision(
            state="x", questions=CHOICE_Q, answers={}, backend="b"
        ) == ""
        assert not log.decisions_path.exists()

    def test_engine_does_not_record_when_disabled(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SONIC_LAYA_DECISION_LOG_PATH", str(tmp_path / "d"))
        monkeypatch.delenv("SONIC_LAYA_DECISION_LOG", raising=False)
        reset_decision_log()
        try:
            engine = LayaSystem1Engine(enable_local_model=False, api_url=None)
            result = engine.predict("open the browser", CHOICE_Q)
            assert "record_id" not in result.routing
            assert not (tmp_path / "d" / "decisions.jsonl").exists()
        finally:
            reset_decision_log()


class TestRecording:
    def test_decision_round_trips(self, log):
        record_id = _decision(log)
        assert record_id
        rows = list(log.iter_decisions())
        assert len(rows) == 1
        row = rows[0]
        assert row["record_id"] == record_id
        assert row["backend"] == "laya_local"
        assert row["questions"] == CHOICE_Q
        assert row["answers"]["plane"]["choice"] == "gui"

    def test_outcome_attaches_by_record_id(self, log):
        record_id = _decision(log)
        assert log.record_outcome(record_id, expected={"plane": "gui"}, correct=True)
        outcomes = list(log.iter_outcomes())
        assert len(outcomes) == 1
        assert outcomes[0]["record_id"] == record_id
        assert outcomes[0]["expected"] == {"plane": "gui"}

    def test_outcome_requires_a_record_id(self, log):
        assert log.record_outcome("", expected={"plane": "gui"}) is False

    def test_stats_counts_labelled_and_unlabelled(self, log):
        first = _decision(log)
        _decision(log, "second decision")
        log.record_outcome(first, expected={"plane": "gui"})
        stats = log.stats()
        assert stats["decisions"] == 2
        assert stats["outcomes"] == 1
        assert stats["labelled"] == 1
        assert stats["unlabelled"] == 1

    def test_engine_records_and_returns_record_id(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SONIC_LAYA_DECISION_LOG", "1")
        monkeypatch.setenv("SONIC_LAYA_DECISION_LOG_PATH", str(tmp_path / "d"))
        reset_decision_log()
        try:
            engine = LayaSystem1Engine(enable_local_model=False, api_url=None)
            result = engine.predict("open the browser", CHOICE_Q)
            record_id = result.routing.get("record_id")
            assert record_id, "an enabled log must expose its record id"
            rows = list(get_decision_log().iter_decisions())
            assert len(rows) == 1
            assert rows[0]["record_id"] == record_id
            # The decision itself must be unchanged by recording.
            assert result.get_choice("plane") in {"gui", "headless"}
        finally:
            reset_decision_log()


class TestSecretScrubbing:
    def test_secret_value_is_redacted(self, log):
        _decision(log, "export GROQ_API_KEY=gsk_abcdefghijklmnopqrstuvwxyz0123 and run")
        stored = log.decisions_path.read_text(encoding="utf-8")
        assert "gsk_abcdefghijklmnopqrstuvwxyz0123" not in stored
        assert "[REDACTED]" in stored

    def test_sensitive_question_keys_are_redacted(self, log):
        log.log_decision(
            state="rotate the credential",
            questions={
                "auth": {
                    "type": "choice",
                    "instructions": "pick",
                    "criteria": {"a": "b"},
                    "api_key": "sk-super-secret-value-1234567890",
                }
            },
            answers={"auth": {"choice": "a", "probability": 0.5}},
            backend="laya_local",
        )
        stored = log.decisions_path.read_text(encoding="utf-8")
        assert "sk-super-secret-value-1234567890" not in stored

    def test_non_string_state_is_json_encoded_and_scrubbed(self, log):
        log.log_decision(
            state={"token": "sk-abcdefghijklmnopqrstuvwxyz012345", "note": "ok"},
            questions=CHOICE_Q,
            answers={"plane": {"choice": "gui", "probability": 0.5}},
            backend="laya_local",
        )
        stored = log.decisions_path.read_text(encoding="utf-8")
        assert "sk-abcdefghijklmnopqrstuvwxyz012345" not in stored


class TestResilience:
    def test_long_state_is_bounded(self, tmp_path):
        log = DecisionLog(tmp_path / "d", enabled=True, max_state_chars=50)
        _decision(log, "A" * 500)
        row = next(iter(log.iter_decisions()))
        assert len(row["state"]) <= 50 + len("...[truncated]")
        assert row["state"].endswith("...[truncated]")

    def test_write_failure_does_not_raise(self, tmp_path):
        blocker = tmp_path / "blocked"
        blocker.write_text("i am a file, not a directory")
        log = DecisionLog(blocker / "nested", enabled=True)
        # Must return a falsy id rather than raising.
        assert log.log_decision(state="x", questions=CHOICE_Q, answers={}, backend="b") == ""
        assert log.stats()["skipped_this_process"] == 1

    def test_engine_survives_a_broken_log(self, tmp_path, monkeypatch):
        blocker = tmp_path / "blocked"
        blocker.write_text("not a directory")
        monkeypatch.setenv("SONIC_LAYA_DECISION_LOG", "1")
        monkeypatch.setenv("SONIC_LAYA_DECISION_LOG_PATH", str(blocker / "nested"))
        reset_decision_log()
        try:
            engine = LayaSystem1Engine(enable_local_model=False, api_url=None)
            result = engine.predict("open the browser", CHOICE_Q)
            assert result.get_choice("plane") in {"gui", "headless"}
        finally:
            reset_decision_log()

    def test_rotation_bounds_disk_use(self, tmp_path):
        log = DecisionLog(tmp_path / "d", enabled=True, max_megabytes=0.001)
        for index in range(60):
            _decision(log, f"decision number {index} " + "x" * 100)
        assert log.decisions_path.exists()
        assert log.stats()["rotations_this_process"] >= 1

    def test_malformed_line_is_skipped(self, log):
        _decision(log)
        with log.decisions_path.open("a", encoding="utf-8") as handle:
            handle.write("{not valid json\n")
        _decision(log, "second")
        assert len(list(log.iter_decisions())) == 2


class TestExport:
    def test_export_contains_only_labelled_rows(self, log, tmp_path):
        labelled = _decision(log, "click the save button")
        _decision(log, "unlabelled decision")
        log.record_outcome(labelled, expected={"plane": "gui"})

        out = tmp_path / "train.jsonl"
        summary = log.export_training_jsonl(out)
        assert summary["written"] == 1
        assert summary["skipped_unlabelled"] == 1

        row = json.loads(out.read_text(encoding="utf-8").strip())
        assert row["state"] == "click the save button"
        assert row["expected"] == {"plane": "gui"}
        assert row["questions"] == CHOICE_Q

    def test_export_without_labels_is_opt_in(self, log, tmp_path):
        _decision(log)
        out = tmp_path / "all.jsonl"
        summary = log.export_training_jsonl(out, require_labels=False)
        assert summary["written"] == 1
        assert json.loads(out.read_text(encoding="utf-8").strip())["expected"] == {}

    def test_export_is_empty_when_no_decisions(self, log, tmp_path):
        out = tmp_path / "train.jsonl"
        assert log.export_training_jsonl(out)["written"] == 0


class TestSingleton:
    def test_reset_rereads_environment(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SONIC_LAYA_DECISION_LOG_PATH", str(tmp_path / "a"))
        reset_decision_log()
        assert get_decision_log().directory == tmp_path / "a"
        monkeypatch.setenv("SONIC_LAYA_DECISION_LOG_PATH", str(tmp_path / "b"))
        assert get_decision_log().directory == tmp_path / "a"
        reset_decision_log()
        assert get_decision_log().directory == tmp_path / "b"
