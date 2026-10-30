"""
SONIC-REDA — Nmap Tool Adapter (Port & Service Scanner)
==========================================================
Executes Nmap inside the ComputeProvider workspace and parses open ports and services.
"""

from __future__ import annotations

import re
import shlex
from typing import Any

from sonic.tools.base import SecurityTool, ToolRequest


class NmapAdapter(SecurityTool):
    """Adapter for Nmap port & service scanning inside isolated compute."""

    @property
    def name(self) -> str:
        return "nmap"

    @property
    def version(self) -> str:
        return "7.94"

    def build_command(self, request: ToolRequest) -> str:
        ports = str(request.options.get("ports", "top-100")).strip()
        timing_raw = str(request.options.get("timing", "T4")).strip().upper()
        extra_args_raw = str(request.options.get("extra_args", "-sV --open")).strip()

        # Validate timing: T0..T5 or digits 0..5
        timing = timing_raw if re.match(r"^T?[0-5]$", timing_raw) else "T4"
        if not timing.startswith("T"):
            timing = f"T{timing}"

        if ports == "top-100":
            port_arg = "--top-ports 100"
        elif ports == "all":
            port_arg = "-p-"
        elif re.match(r"^[0-9,\-]+$", ports):
            port_arg = f"-p {shlex.quote(ports)}"
        else:
            port_arg = "--top-ports 100"

        # Sanitize extra_args by allowlisting valid nmap flag tokens
        safe_flags = {"-sV", "--open", "-sC", "-sT", "-sU", "-Pn", "-n", "-O", "-v", "-vv", "--version-light", "--version-all"}
        tokens = shlex.split(extra_args_raw)
        validated_tokens = [t for t in tokens if t in safe_flags]
        extra_args = " ".join(validated_tokens) if validated_tokens else "-sV --open"

        safe_target = shlex.quote(request.target.strip())
        return f"nmap -{timing} {port_arg} {extra_args} {safe_target}"

    def parse_output(self, raw_stdout: str, raw_stderr: str) -> list[dict[str, Any]]:
        """Parse text/greppable Nmap output into structured port mappings."""
        open_ports = []
        port_pattern = re.compile(r"^\s*(\d+)/(tcp|udp)\s+(open(?:\|filtered)?)\s+(\S+)(?:\s+(.*))?$", re.MULTILINE)

        for match in port_pattern.finditer(raw_stdout):
            port = int(match.group(1))
            proto = match.group(2)
            state = match.group(3)
            service = match.group(4)
            version = (match.group(5) or "").strip()
            open_ports.append({
                "port": port,
                "protocol": proto,
                "service": service,
                "version": version,
                "state": state,
            })

        return open_ports

