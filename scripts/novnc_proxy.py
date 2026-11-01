#!/usr/bin/env python3
"""Expose the local noVNC port to a remote operator browser.

The Docker workstation publishes noVNC on ``127.0.0.1:6080`` (loopback only,
by design -- raw VNC must never be on the LAN). A browser on another machine
therefore cannot reach it. This is a small TCP splice that listens on a
reachable address and forwards to the loopback noVNC port, so an operator can
watch the desktop.

It is a viewer/relay only. It never drives the desktop and never fabricates
agent progress; every pixel it carries comes from the real workstation.

Usage:
    python scripts/novnc_proxy.py --listen-host 0.0.0.0 --listen-port 12001 \
        --target-host 127.0.0.1 --target-port 6080
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib


async def _pipe(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    try:
        while True:
            chunk = await reader.read(65536)
            if not chunk:
                break
            writer.write(chunk)
            await writer.drain()
    except (ConnectionResetError, BrokenPipeError, asyncio.IncompleteReadError):
        pass
    finally:
        with contextlib.suppress(Exception):
            writer.close()


async def _handle(
    client_reader: asyncio.StreamReader,
    client_writer: asyncio.StreamWriter,
    target_host: str,
    target_port: int,
) -> None:
    try:
        upstream_reader, upstream_writer = await asyncio.open_connection(target_host, target_port)
    except OSError:
        client_writer.close()
        return
    await asyncio.gather(
        _pipe(client_reader, upstream_writer),
        _pipe(upstream_reader, client_writer),
        return_exceptions=True,
    )


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--listen-host", default="0.0.0.0")
    parser.add_argument("--listen-port", type=int, default=12001)
    parser.add_argument("--target-host", default="127.0.0.1")
    parser.add_argument("--target-port", type=int, default=6080)
    args = parser.parse_args()

    server = await asyncio.start_server(
        lambda r, w: _handle(r, w, args.target_host, args.target_port),
        host=args.listen_host,
        port=args.listen_port,
    )
    print(
        f"noVNC relay listening on {args.listen_host}:{args.listen_port} "
        f"-> {args.target_host}:{args.target_port}",
        flush=True,
    )
    async with server:
        await server.serve_forever()


if __name__ == "__main__":
    asyncio.run(main())
