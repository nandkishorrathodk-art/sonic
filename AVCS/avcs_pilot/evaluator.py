from __future__ import annotations

from pathlib import Path
from .contracts import PilotScenario, VerificationResult


def verify_exact_text(scenario: PilotScenario, run_dir: Path) -> VerificationResult:
    """Independent local file oracle; do not pass this scenario object to the agent."""
    target = run_dir / scenario.output_relative_path
    try:
        resolved_root = run_dir.resolve()
        resolved_target = target.resolve()
        if resolved_root not in resolved_target.parents and resolved_target != resolved_root:
            return VerificationResult(False, "file_exact_text", "PATH_ESCAPE", "Target resolved outside run directory")
        actual = target.read_text(encoding="utf-8")
    except FileNotFoundError:
        return VerificationResult(False, "file_exact_text", "OUTPUT_MISSING", "Expected output file was not created")
    except (OSError, UnicodeError) as exc:
        return VerificationResult(False, "file_exact_text", "OUTPUT_UNREADABLE", type(exc).__name__)
    if actual != scenario.expected_text:
        return VerificationResult(False, "file_exact_text", "CONTENT_MISMATCH", "Output differs from expected state")
    return VerificationResult(True, "file_exact_text", "EXACT_CONTENT_MATCH", "Expected final file state independently verified")
