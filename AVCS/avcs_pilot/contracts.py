from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Mapping


class ActionStatus(str, Enum):
    CONFIRMED = "CONFIRMED"
    FAILED = "FAILED"
    AMBIGUOUS = "AMBIGUOUS"
    BLOCKED = "BLOCKED"
    CANCELLED = "CANCELLED"
    TIMED_OUT = "TIMED_OUT"


@dataclass(frozen=True)
class PublicTask:
    """Only these fields may cross the executor boundary. No oracle fields."""
    task_id: str
    workflow_id: str
    prompt: str
    application: str
    timeout_s: float
    max_actions: int
    risk_level: str = "LOW"


@dataclass(frozen=True)
class AgentExecution:
    status: ActionStatus
    action_count: int = 0
    duration_ms: float = 0.0
    provider: str = "unknown"
    model: str = "unknown"
    confirmation_method: str | None = None
    message: str = ""
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "AgentExecution":
        status = value.get("status", "FAILED")
        try:
            parsed_status = status if isinstance(status, ActionStatus) else ActionStatus(str(status).upper())
        except ValueError as exc:
            raise ValueError(f"Unsupported action status: {status!r}") from exc
        action_count = int(value.get("action_count", 0))
        duration_ms = float(value.get("duration_ms", 0.0))
        if action_count < 0 or duration_ms < 0:
            raise ValueError("action_count and duration_ms must be non-negative")
        diagnostics = value.get("diagnostics", {})
        if not isinstance(diagnostics, Mapping):
            raise ValueError("diagnostics must be a mapping")
        return cls(
            status=parsed_status,
            action_count=action_count,
            duration_ms=duration_ms,
            provider=str(value.get("provider", "unknown")),
            model=str(value.get("model", "unknown")),
            confirmation_method=(str(value["confirmation_method"]) if value.get("confirmation_method") else None),
            message=str(value.get("message", "")),
            diagnostics=dict(diagnostics),
        )

    def to_record(self) -> dict[str, Any]:
        # Keep audit telemetry useful without persisting arbitrary desktop text.
        allowed_diagnostics = {
            "error_code", "verification_method", "retry_count", "queue_wait_ms",
            "capture_backend", "adapter_reported_duration_ms", "confidence",
            "dispatch_accepted", "target_age_ms", "cancel_acknowledged",
        }
        safe_diagnostics = {
            str(key): json_safe(value)
            for key, value in self.diagnostics.items()
            if str(key) in allowed_diagnostics and isinstance(value, (str, int, float, bool, type(None)))
        }
        return {
            "status": self.status.value,
            "action_count": self.action_count,
            "duration_ms": self.duration_ms,
            "provider": self.provider,
            "model": self.model,
            "confirmation_method": self.confirmation_method,
            # Do not serialize arbitrary free-form messages that might contain screen text or PII.
            "message_present": bool(self.message),
            "diagnostics": safe_diagnostics,
        }


@dataclass(frozen=True)
class VerificationResult:
    passed: bool
    verifier: str
    evidence_code: str
    details: str = ""


@dataclass(frozen=True)
class PilotScenario:
    task_id: str
    workflow_id: str
    application: str
    prompt_template: str
    timeout_s: float
    max_actions: int
    risk_level: str
    output_relative_path: str
    expected_text: str
    fixture_relative_path: str | None = None
    fixture_text: str | None = None

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "PilotScenario":
        required = ("task_id", "workflow_id", "application", "prompt_template", "timeout_s",
                    "max_actions", "output_relative_path", "expected_text")
        missing = [key for key in required if key not in value]
        if missing:
            raise ValueError(f"Scenario missing required fields: {', '.join(missing)}")
        timeout_s = float(value["timeout_s"])
        max_actions = int(value["max_actions"])
        if timeout_s <= 0 or max_actions <= 0:
            raise ValueError("timeout_s and max_actions must be positive")
        risk = str(value.get("risk_level", "LOW")).upper()
        if risk not in {"LOW", "MEDIUM", "HIGH"}:
            raise ValueError(f"Unsupported risk_level: {risk}")
        fixture_text = value.get("fixture_text")
        fixture_path = value.get("fixture_relative_path")
        if (fixture_text is None) != (fixture_path is None):
            raise ValueError("fixture_relative_path and fixture_text must be supplied together")
        return cls(
            task_id=str(value["task_id"]),
            workflow_id=str(value["workflow_id"]),
            application=str(value["application"]),
            prompt_template=str(value["prompt_template"]),
            timeout_s=timeout_s,
            max_actions=max_actions,
            risk_level=risk,
            output_relative_path=str(value["output_relative_path"]),
            expected_text=str(value["expected_text"]),
            fixture_relative_path=str(fixture_path) if fixture_path is not None else None,
            fixture_text=str(fixture_text) if fixture_text is not None else None,
        )

    def public_task(self, run_dir: str, task_id: str) -> PublicTask:
        """Render prompt and intentionally omit expected output/evaluator details."""
        from pathlib import Path
        root = Path(run_dir).resolve()
        output_path = (root / self.output_relative_path).resolve()
        fixture_path = (root / self.fixture_relative_path).resolve() if self.fixture_relative_path else None
        if root not in output_path.parents:
            raise ValueError("output_relative_path must resolve inside the trial directory")
        if fixture_path is not None and root not in fixture_path.parents:
            raise ValueError("fixture_relative_path must resolve inside the trial directory")
        prompt = self.prompt_template.format(
            output_path=str(output_path),
            fixture_path=str(fixture_path) if fixture_path else "",
        )
        return PublicTask(
            task_id=task_id,
            workflow_id=self.workflow_id,
            prompt=prompt,
            application=self.application,
            timeout_s=self.timeout_s,
            max_actions=self.max_actions,
            risk_level=self.risk_level,
        )


def json_safe(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if hasattr(value, "to_record"):
        return value.to_record()
    if hasattr(value, "__dataclass_fields__"):
        return asdict(value)
    if isinstance(value, Mapping):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)
