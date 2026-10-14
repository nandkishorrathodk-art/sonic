from __future__ import annotations

from pathlib import Path

from sonic.evolution.codebase_auditor import CodebaseAuditor


def test_auditor_scans_core_package_in_packaged_runtime_layout(tmp_path: Path) -> None:
    (tmp_path / "evolution").mkdir()
    (tmp_path / "evolution" / "codebase_auditor.py").write_text("", encoding="utf-8")
    (tmp_path / "agents").mkdir()
    (tmp_path / "agents" / "example.py").write_text("VALUE = 1\n", encoding="utf-8")

    auditor = CodebaseAuditor(
        repo_root=tmp_path,
        source_roots=("services/core/sonic", "apps/cli/sonic_cli"),
    )

    assert auditor.source_files() == ["agents/example.py", "evolution/codebase_auditor.py"]


def test_auditor_reports_tests_unavailable_when_image_omits_test_sources(tmp_path: Path) -> None:
    auditor = CodebaseAuditor(repo_root=tmp_path)

    assert auditor.test_scan() == {
        "available": False,
        "error": "No configured test suites are present in this runtime.",
        "failed_tests": [],
    }
