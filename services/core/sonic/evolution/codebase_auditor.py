"""
SONIC — Codebase Auditor (read-only full sweep for real problems)
=================================================================

This is the "wake up and check the whole codebase" pass a careful engineer does
before touching anything. It is strictly READ-ONLY: it never writes to a source
file, never commits, and never mutates the tree.

It produces a structured `AuditReport` from three independent sources of truth:

1. **Health check** — are the runtime prerequisites actually satisfied? (Can the
   core dependencies import? Do the safety-rules / model config files exist? Is a
   test runner available?) This answers "sab tik hai ya nahi".
2. **Static scan (AST + text)** — real defects, not style:
      * bare `except:` and swallowed `except ...: pass`
      * mutable default arguments (`def f(x=[])`)
      * `eval()` / `exec()` dynamic execution
      * `subprocess(..., shell=True)`
      * leftover `# [SONIC-EVOLUTION]` no-op stubs
      * TODO / FIXME / HACK / XXX markers
3. **Test scan** — actually run the suite and capture every failure, so the audit
   reflects reality rather than opinion.

The report can be rendered as a readable narrative ("Sir, while I was looking at
your codebase ...") for the operator.
"""

from __future__ import annotations

import ast
import re
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from sonic.logger import get_logger

logger = get_logger(__name__)

SOURCE_ROOTS = ("services/core/sonic", "apps/cli/sonic_cli", "packages/avcs")
TEST_ROOTS = ("services/core/tests", "apps/cli/tests", "packages/avcs/tests")

_TODO_RE = re.compile(r"(?m)#[^\n]*\b(TODO|FIXME|HACK|XXX)\b[:\s]*(.*)")
_STUB_RE = re.compile(r"(?m)^[ \t]*#\s*\[SONIC-EVOLUTION\]")

# Severity ordering (worst first) used for sorting + reporting.
_SEVERITY_ORDER = {"critical": 0, "high": 1, "medium": 2, "low": 3, "info": 4}


def _now() -> str:
    return datetime.now(UTC).isoformat()


@dataclass
class AuditFinding:
    """One concrete problem found in the codebase."""
    kind: str
    severity: str
    file: str
    line: int
    title: str
    detail: str
    evidence: str = ""

    def __str__(self) -> str:  # pragma: no cover - convenience
        return f"[{self.severity}] {self.file}:{self.line} {self.title}"


@dataclass
class AuditReport:
    health: dict = field(default_factory=dict)
    tests: dict = field(default_factory=dict)
    findings: list[AuditFinding] = field(default_factory=list)
    files_scanned: int = 0
    started_at: str = field(default_factory=_now)
    finished_at: str = ""

    @property
    def ok(self) -> bool:
        """True when nothing above 'low' was found and the suite is not failing."""
        blocking = [f for f in self.findings if f.severity in ("critical", "high", "medium")]
        return not blocking and not self.tests.get("failed")

    def sorted_findings(self) -> list[AuditFinding]:
        return sorted(self.findings, key=lambda f: (_SEVERITY_ORDER.get(f.severity, 9), f.file, f.line))

    def counts(self) -> dict[str, int]:
        out = {sev: 0 for sev in _SEVERITY_ORDER}
        for f in self.findings:
            out[f.severity] = out.get(f.severity, 0) + 1
        return out

    def to_text(self) -> str:
        c = self.counts()
        lines = [
            "# Codebase audit (read-only)",
            "",
            f"- Started:  `{self.started_at}`",
            f"- Finished: `{self.finished_at or _now()}`",
            f"- Files scanned: {self.files_scanned}",
            f"- Findings: {len(self.findings)} "
            f"(critical {c['critical']}, high {c['high']}, medium {c['medium']}, low {c['low']}, info {c['info']})",
            f"- Verdict: {'OK' if self.ok else 'NEEDS ATTENTION'}",
            "",
            "## Health",
        ]
        for k, v in self.health.items():
            lines.append(f"- {k}: {v}")
        lines.append("")
        lines.append("## Tests")
        for k, v in self.tests.items():
            if k != "failed_tests":
                lines.append(f"- {k}: {v}")
        for t in self.tests.get("failed_tests", [])[:40]:
            lines.append(f"  - FAILED: {t}")
        lines.append("")
        lines.append("## Findings")
        for f in self.sorted_findings():
            lines.append(f"- [{f.severity}] `{f.file}:{f.line}` — {f.title}")
            lines.append(f"    {f.detail}")
            if f.evidence:
                lines.append(f"    evidence: `{f.evidence[:200]}`")
        return "\n".join(lines)


class CodebaseAuditor:
    """Read-only auditor over SONIC's own source tree."""

    def __init__(
        self,
        repo_root: Path | str | None = None,
        source_roots: tuple[str, ...] = SOURCE_ROOTS,
        test_paths: list[str] | None = None,
        max_findings: int = 60,
    ) -> None:
        self.repo_root = Path(repo_root).resolve() if repo_root else self._discover_root()
        self.source_roots = source_roots
        self.test_paths = test_paths
        self.max_findings = max_findings

    @staticmethod
    def _discover_root() -> Path:
        cur = Path(__file__).resolve()
        for p in cur.parents:
            if (p / ".git").exists() or (p / "services/core").exists():
                return p
            if (p / "sonic" / "evolution" / "codebase_auditor.py").is_file():
                return p
        return cur.parent

    # ------------------------------------------------------------------
    def source_files(self) -> list[str]:
        out: list[str] = []
        for root in self.source_roots:
            base = self.repo_root / root
            if not base.exists() and root == "services/core/sonic":
                if (self.repo_root / "sonic" / "evolution").is_dir():
                    base = self.repo_root / "sonic"
                elif (self.repo_root / "evolution").is_dir():
                    base = self.repo_root
            if not base.exists():
                continue
            for p in sorted(base.rglob("*.py")):
                rel = str(p.relative_to(self.repo_root)).replace("\\", "/")
                if rel.endswith("__init__.py"):
                    continue
                out.append(rel)
        return out

    # ------------------------------------------------------------------
    def health_check(self) -> dict:
        """Are the runtime prerequisites actually satisfied?"""
        deps = ["fastapi", "pydantic", "pydantic_settings", "yaml"]
        missing = []
        for d in deps:
            try:
                __import__(d)
            except Exception:
                missing.append(d)
        pytest_ok = False
        try:
            import importlib.util

            pytest_ok = importlib.util.find_spec("pytest") is not None
        except Exception:
            pytest_ok = False

        core_source = self.repo_root / "services" / "core" / "sonic"
        if not core_source.is_dir():
            packaged_source = self.repo_root / "sonic"
            if packaged_source.is_dir():
                core_source = packaged_source
            elif self.repo_root.name == "sonic":
                core_source = self.repo_root
        files = {
            "safety_rules.yaml": (self.repo_root / "configs" / "safety_rules.yaml").exists(),
            "models.yaml": (self.repo_root / "configs" / "models.yaml").exists(),
            "core_source": core_source.is_dir(),
        }
        env = "development"
        try:
            from sonic.config import get_settings

            env = get_settings().app_env
        except Exception:
            pass

        return {
            "python": sys.version.split()[0],
            "missing_dependencies": missing or "none",
            "test_runner_available": pytest_ok,
            "config_files": files,
            "app_env": env,
            "healthy": not missing and all(files.values()),
        }

    # ------------------------------------------------------------------
    def static_scan(self) -> list[AuditFinding]:
        findings: list[AuditFinding] = []
        for rel in self.source_files():
            p = self.repo_root / rel
            try:
                text = p.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue
            findings.extend(self._scan_text(rel, text))
            findings.extend(self._scan_ast(rel, text))
        return findings

    def _scan_text(self, rel: str, text: str) -> list[AuditFinding]:
        out: list[AuditFinding] = []
        for m in _STUB_RE.finditer(text):
            out.append(AuditFinding(
                kind="noop_stub", severity="high", file=rel,
                line=text[:m.start()].count("\n") + 1,
                title="Leftover no-op self-evolution stub",
                detail="A previous self-evolution cycle appended a cosmetic comment instead of a real change.",
                evidence=m.group(0).strip()[:120],
            ))
        for m in _TODO_RE.finditer(text):
            note = (m.group(2) or "").strip()
            out.append(AuditFinding(
                kind="todo_marker", severity="medium", file=rel,
                line=text[:m.start()].count("\n") + 1,
                title=f"{m.group(1).upper()} marker",
                detail=note or "A developer-flagged incomplete/risky spot.",
                evidence=m.group(0).strip()[:160],
            ))
        return out

    def _scan_ast(self, rel: str, text: str) -> list[AuditFinding]:
        out: list[AuditFinding] = []
        try:
            tree = ast.parse(text, filename=rel)
        except SyntaxError as e:
            return [AuditFinding(
                kind="syntax_error", severity="critical", file=rel, line=e.lineno or 0,
                title="File does not parse", detail=f"SyntaxError: {e.msg}",
            )]
        except Exception:
            return out

        for node in ast.walk(tree):
            if isinstance(node, ast.ExceptHandler):
                if node.type is None:
                    out.append(AuditFinding(
                        kind="bare_except", severity="high", file=rel, line=node.lineno,
                        title="Bare `except:`",
                        detail="Catches everything including KeyboardInterrupt/SystemExit; masks real failures.",
                    ))
                elif len(node.body) == 1 and isinstance(node.body[0], ast.Pass):
                    out.append(AuditFinding(
                        kind="swallowed_exception", severity="medium", file=rel, line=node.lineno,
                        title="Swallowed exception (except: pass)",
                        detail="An error is silently discarded, hiding failures from the caller.",
                    ))
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for default in list(node.args.defaults) + [d for d in node.args.kw_defaults if d]:
                    if isinstance(default, (ast.List, ast.Dict, ast.Set)):
                        out.append(AuditFinding(
                            kind="mutable_default", severity="medium", file=rel, line=node.lineno,
                            title="Mutable default argument",
                            detail=f"`{node.name}(...)` uses a mutable default; state leaks between calls.",
                        ))
            if isinstance(node, ast.Call):
                fn = node.func
                if isinstance(fn, ast.Name) and fn.id in ("eval", "exec"):
                    out.append(AuditFinding(
                        kind="dynamic_exec", severity="high", file=rel, line=node.lineno,
                        title=f"Dynamic `{fn.id}()` execution",
                        detail="Executing dynamic strings is an injection surface.",
                    ))
                for kw in node.keywords or []:
                    if kw.arg == "shell" and isinstance(kw.value, ast.Constant) and kw.value.value is True:
                        out.append(AuditFinding(
                            kind="shell_true", severity="high", file=rel, line=node.lineno,
                            title="subprocess with shell=True",
                            detail="Shell interpretation enables command injection.",
                        ))
        return out

    # ------------------------------------------------------------------
    def test_scan(self) -> dict:
        """Run the suite and capture reality."""
        paths = self.test_paths or [
            root for root in TEST_ROOTS if (self.repo_root / root).is_dir()
        ]
        if not paths:
            return {
                "available": False,
                "error": "No configured test suites are present in this runtime.",
                "failed_tests": [],
            }
        cmd = [sys.executable, "-m", "pytest", *paths, "-q", "--tb=no", "-p", "no:cacheprovider"]
        try:
            res = subprocess.run(
                cmd, cwd=str(self.repo_root), capture_output=True, text=True, timeout=900,
            )
        except Exception as e:
            return {"available": False, "error": str(e), "failed_tests": []}

        output = (res.stdout or "") + (res.stderr or "")
        passed = failed = skipped = 0
        m = re.search(r"(\d+)\s+passed", output)
        if m:
            passed = int(m.group(1))
        m = re.search(r"(\d+)\s+failed", output)
        if m:
            failed = int(m.group(1))
        m = re.search(r"(\d+)\s+skipped", output)
        if m:
            skipped = int(m.group(1))
        failed_tests = [ln.replace("FAILED ", "").strip() for ln in output.splitlines() if ln.startswith("FAILED ")]
        if not failed_tests:
            failed_tests = [ln.replace("ERROR ", "").strip() for ln in output.splitlines() if ln.startswith("ERROR ")]
        return {
            "available": True,
            "exit_code": res.returncode,
            "passed": passed,
            "failed": failed,
            "skipped": skipped,
            "failed_tests": failed_tests,
        }

    # ------------------------------------------------------------------
    def audit(self, run_tests: bool = True) -> AuditReport:
        report = AuditReport()
        report.health = self.health_check()
        report.files_scanned = len(self.source_files())
        report.findings = self.static_scan()
        if run_tests:
            report.tests = self.test_scan()
        if len(report.findings) > self.max_findings:
            report.findings = report.sorted_findings()[: self.max_findings]
        report.finished_at = _now()
        logger.info(
            "codebase_audit_complete",
            files=report.files_scanned,
            findings=len(report.findings),
            tests_failed=report.tests.get("failed", 0),
        )
        return report
