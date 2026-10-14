"""
SONIC-REDA — HTTP Client Adapter (Raw HTTP Probing)
======================================================
Executes raw HTTP requests via curl inside the sandbox with full header and timing capture.
"""

from __future__ import annotations

import re
import shlex
from typing import Any

from sonic.tools.base import SecurityTool, ToolRequest


class HTTPClientAdapter(SecurityTool):
    """Adapter for raw HTTP probing inside the isolated container."""

    @property
    def name(self) -> str:
        return "http_client"

    @property
    def version(self) -> str:
        return "curl-8.5"

    def build_command(self, request: ToolRequest) -> str:
        raw_method = str(request.options.get("method", "GET")).upper().strip()
        method = raw_method if re.match(r"^[A-Z]{3,10}$", raw_method) else "GET"
        headers = request.options.get("headers", {})
        data = request.options.get("data")
        follow_redirects = request.options.get("follow_redirects", False)
        if not isinstance(follow_redirects, bool):
            raise ValueError("follow_redirects must be a boolean")
        if follow_redirects:
            raise ValueError(
                "Automatic redirect following is disabled because each redirect "
                "destination must be independently scope- and egress-validated."
            )

        timeout_sec = max(1, min(300, int(request.timeout_seconds or 30)))
        cmd = [
            "curl", "-i", "-s",
            "-X", method,
            "--max-time", str(timeout_sec),
        ]
        if isinstance(headers, dict):
            for k, v in headers.items():
                safe_header = f"{str(k).strip()}: {str(v).strip()}"
                cmd.extend(["-H", shlex.quote(safe_header)])

        if data is not None:
            cmd.extend(["--data-raw", shlex.quote(str(data))])

        cmd.append(shlex.quote(request.target.strip()))
        return " ".join(cmd)

    def parse_output(self, raw_stdout: str, raw_stderr: str) -> list[dict[str, Any]]:
        """Split HTTP response into headers, status code, and body, handling redirects cleanly."""
        if not raw_stdout:
            return []

        import re
        # When curl follows redirects (-L), multiple HTTP responses appear sequentially.
        # Split on boundary between responses to extract the final response block
        blocks = re.split(r"(?:\r?\n\r?\n)(?=HTTP/\d)", raw_stdout)
        final_block = blocks[-1] if blocks else raw_stdout

        parts = final_block.split("\r\n\r\n", 1) if "\r\n\r\n" in final_block else final_block.split("\n\n", 1)
        raw_headers = parts[0]
        body = parts[1] if len(parts) > 1 else ""

        status_code = 0
        header_lines = raw_headers.splitlines()
        if header_lines and "HTTP/" in header_lines[0]:
            try:
                status_code = int(header_lines[0].split()[1])
            except Exception:
                pass

        return [{
            "status_code": status_code,
            "headers_raw": raw_headers,
            "body": body[:4096],  # First 4KB preview
            "body_length": len(body),
            "redirects_followed": max(0, len(blocks) - 1),
        }]
