"""Command-line executable entrypoint for AVCS MCP server.

Run as:
    python -m avcs.mcp
    python -m avcs.mcp --synthetic --virtual-motor
"""
from __future__ import annotations

import argparse
import sys

from avcs.mcp.server import AVCSMCPServer
from forge.vision.cortex.sdk import ScreenEye
from forge.vision.capture import SyntheticScreen


def main() -> None:
    parser = argparse.ArgumentParser(description="AVCS Model Context Protocol (MCP) Server")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic screen source instead of physical display")
    parser.add_argument("--virtual-motor", action="store_true", help="Use in-memory virtual cursor & keyboard driver")
    parser.add_argument("--device", default="cpu", help="PyTorch compute device (default: cpu)")
    args = parser.parse_args()

    # Configure UTF-8 encoding on standard streams
    if hasattr(sys.stdin, "reconfigure"):
        sys.stdin.reconfigure(encoding="utf-8")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    # Preserve pure stdout for JSON-RPC 2.0 frames; redirect general prints to stderr
    mcp_stdout = sys.stdout
    sys.stdout = sys.stderr

    source = SyntheticScreen() if args.synthetic else None
    eye = ScreenEye(source=source, device=args.device)
    server = AVCSMCPServer(eye=eye, stdout=mcp_stdout, force_virtual_motor=args.virtual_motor)

    try:
        server.run()
    except KeyboardInterrupt:
        pass
    finally:
        server.close()
        sys.stdout = mcp_stdout


if __name__ == "__main__":
    main()

