"""
SONIC-REDA — Nuclei Tool Adapter (Vulnerability Template Scanner)
===================================================================
Executes Nuclei inside the ComputeProvider workspace with JSON output.
"""

from __future__ import annotations

import json
import shlex
from typing import Any

from sonic.tools.base import SecurityTool, ToolRequest


class NucleiAdapter(SecurityTool):
    """Adapter for Nuclei template-based vulnerability scanning."""

    @property
    def name(self) -> str:
        return "nuclei"

    @property
    def version(self) -> str:
        return "v3.2"

    def build_command(self, request: ToolRequest) -> str:
        tags = shlex.quote(str(request.options.get("tags", "cve,misconfig,exposure")).strip())
        severity = shlex.quote(str(request.options.get("severity", "critical,high,medium")).strip())
        try:
            rate_limit = max(1, min(1000, int(request.options.get("rate_limit", 50))))
        except Exception:
            rate_limit = 50
        safe_target = shlex.quote(request.target.strip())

        return f"nuclei -u {safe_target} -tags {tags} -severity {severity} -rate-limit {rate_limit} -jsonl -silent"

    def parse_output(self, raw_stdout: str, raw_stderr: str) -> list[dict[str, Any]]:
        """Parse Nuclei NDJSON output stream into structured findings."""
        findings = []
        for line in raw_stdout.splitlines():
            line = line.strip()
            if not line or not line.startswith("{"):
                continue
            try:
                data = json.loads(line)
                info = data.get("info") or {}
                classification = info.get("classification") if isinstance(info, dict) else {}
                if not isinstance(classification, dict):
                    classification = {}

                cve_raw = classification.get("cve-id", "")
                cve_id = cve_raw[0] if isinstance(cve_raw, list) and cve_raw else (cve_raw if isinstance(cve_raw, str) else "")
                cvss_score = float(classification.get("cvss-score", 0.0) or 0.0)

                findings.append({
                    "template_id": data.get("template-id", ""),
                    "name": info.get("name", "") if isinstance(info, dict) else "",
                    "severity": str(info.get("severity", "info") if isinstance(info, dict) else "info").upper(),
                    "matched_at": data.get("matched-at", ""),
                    "description": info.get("description", "") if isinstance(info, dict) else "",
                    "cve_id": cve_id,
                    "cvss_score": cvss_score,
                    "curl_command": data.get("curl-command", ""),
                    "extracted_results": data.get("extracted-results", []),
                })
            except (json.JSONDecodeError, Exception):
                continue

        return findings
