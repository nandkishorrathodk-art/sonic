"""Minimal dependency-free RFB (VNC) client for remote desktop control.

Docker-OSX runs macOS inside a QEMU virtual machine.  There is no X11 display
inside the guest, so the desktop cannot be driven with ``xdotool``/``scrot``
the way the Linux workstation is.  Instead the guest exposes the QEMU
framebuffer over RFB (VNC) and a shell over SSH.  This module implements just
enough of RFB 3.8 to:

* read the framebuffer (Raw encoding) as an RGB ``numpy`` array, and
* inject pointer and keyboard events.

It uses only the standard library plus ``numpy`` (already a dependency), so it
adds no new runtime requirement and is fully unit-testable without a VM.

Security types:

``1``
    None (no authentication).
``2``
    VNC Authentication -- a DES challenge/response.  DES is implemented here so
    the client does not depend on ``cryptography`` being importable.
``33``
    Apple Remote Desktop (ARD) authentication.  Rejected with a clear error:
    it needs RSA/AES from ``cryptography``.  Configure a VNC password instead.

The client is synchronous (blocking socket with timeouts); callers run it in a
worker thread.  RFB is a strict request/response protocol, so a blocking socket
is both simpler and easier to reason about than an asyncio transport.
"""

from __future__ import annotations

import socket
import struct
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SEC_TYPE_NONE = 1
SEC_TYPE_VNC_AUTH = 2
SEC_TYPE_APPLE_ARD = 33

ENCODING_RAW = 0

# Pointer button masks (RFB 3.8).
POINTER_BUTTON_LEFT = 1
POINTER_BUTTON_MIDDLE = 2
POINTER_BUTTON_RIGHT = 4
POINTER_WHEEL_UP = 8
POINTER_WHEEL_DOWN = 16
# X11/RFB convention: buttons 6/7 are horizontal wheel left/right.
POINTER_WHEEL_LEFT = 32
POINTER_WHEEL_RIGHT = 64

# Keysyms we need for text entry and navigation.  Printable ASCII maps to
# itself in X11 keysyms; the rest are listed explicitly.
_KEYSYMS: dict[str, int] = {
    "backspace": 0xFF08,
    "tab": 0xFF09,
    "return": 0xFF0D,
    "enter": 0xFF0D,
    "escape": 0xFF1B,
    "esc": 0xFF1B,
    "insert": 0xFF63,
    "delete": 0xFFFF,
    "del": 0xFFFF,
    "home": 0xFF50,
    "end": 0xFF57,
    "pageup": 0xFF55,
    "prior": 0xFF55,
    "pgup": 0xFF55,
    "pagedown": 0xFF56,
    "next": 0xFF56,
    "pgdn": 0xFF56,
    "left": 0xFF51,
    "up": 0xFF52,
    "right": 0xFF53,
    "down": 0xFF54,
    "shift": 0xFFE1,
    "shift_l": 0xFFE1,
    "shift_r": 0xFFE2,
    "control": 0xFFE3,
    "ctrl": 0xFFE3,
    "control_l": 0xFFE3,
    "control_r": 0xFFE4,
    "alt": 0xFFE9,
    "option": 0xFFE9,
    "alt_l": 0xFFE9,
    "alt_r": 0xFFEA,
    "meta": 0xFFEB,
    "super": 0xFFEB,
    "command": 0xFFEB,
    "cmd": 0xFFEB,
    "caps_lock": 0xFFE5,
    "space": 0x0020,
}
for _i in range(1, 13):
    _KEYSYMS[f"f{_i}"] = 0xFFBE + (_i - 1)


class RFBError(RuntimeError):
    """Raised when the RFB session cannot be established or used."""


# ---------------------------------------------------------------------------
# DES (for VNC Authentication, security type 2)
# ---------------------------------------------------------------------------

_IP = [
    58, 50, 42, 34, 26, 18, 10, 2, 60, 52, 44, 36, 28, 20, 12, 4,
    62, 54, 46, 38, 30, 22, 14, 6, 64, 56, 48, 40, 32, 24, 16, 8,
    57, 49, 41, 33, 25, 17, 9, 1, 59, 51, 43, 35, 27, 19, 11, 3,
    61, 53, 45, 37, 29, 21, 13, 5, 63, 55, 47, 39, 31, 23, 15, 7,
]
_FP = [
    40, 8, 48, 16, 56, 24, 64, 32, 39, 7, 47, 15, 55, 23, 63, 31,
    38, 6, 46, 14, 54, 22, 62, 30, 37, 5, 45, 13, 53, 21, 61, 29,
    36, 4, 44, 12, 52, 20, 60, 28, 35, 3, 43, 11, 51, 19, 59, 27,
    34, 2, 42, 10, 50, 18, 58, 26, 33, 1, 41, 9, 49, 17, 57, 25,
]
_E = [
    32, 1, 2, 3, 4, 5, 4, 5, 6, 7, 8, 9, 8, 9, 10, 11, 12, 13,
    12, 13, 14, 15, 16, 17, 16, 17, 18, 19, 20, 21, 20, 21, 22, 23, 24, 25,
    24, 25, 26, 27, 28, 29, 28, 29, 30, 31, 32, 1,
]
_P = [
    16, 7, 20, 21, 29, 12, 28, 17, 1, 15, 23, 26, 5, 18, 31, 10,
    2, 8, 24, 14, 32, 27, 3, 9, 19, 13, 30, 6, 22, 11, 4, 25,
]
_PC1 = [
    57, 49, 41, 33, 25, 17, 9, 1, 58, 50, 42, 34, 26, 18,
    10, 2, 59, 51, 43, 35, 27, 19, 11, 3, 60, 52, 44, 36,
    63, 55, 47, 39, 31, 23, 15, 7, 62, 54, 46, 38, 30, 22,
    14, 6, 61, 53, 45, 37, 29, 21, 13, 5, 28, 20, 12, 4,
]
_PC2 = [
    14, 17, 11, 24, 1, 5, 3, 28, 15, 6, 21, 10,
    23, 19, 12, 4, 26, 8, 16, 7, 27, 20, 13, 2,
    41, 52, 31, 37, 47, 55, 30, 40, 51, 45, 33, 48,
    44, 49, 39, 56, 34, 53, 46, 42, 50, 36, 29, 32,
]
_SHIFTS = [1, 1, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 1]
_SBOXES = [
    [14, 4, 13, 1, 2, 15, 11, 8, 3, 10, 6, 12, 5, 9, 0, 7,
     0, 15, 7, 4, 14, 2, 13, 1, 10, 6, 12, 11, 9, 5, 3, 8,
     4, 1, 14, 8, 13, 6, 2, 11, 15, 12, 9, 7, 3, 10, 5, 0,
     15, 12, 8, 2, 4, 9, 1, 7, 5, 11, 3, 14, 10, 0, 6, 13],
    [15, 1, 8, 14, 6, 11, 3, 4, 9, 7, 2, 13, 12, 0, 5, 10,
     3, 13, 4, 7, 15, 2, 8, 14, 12, 0, 1, 10, 6, 9, 11, 5,
     0, 14, 7, 11, 10, 4, 13, 1, 5, 8, 12, 6, 9, 3, 2, 15,
     13, 8, 10, 1, 3, 15, 4, 2, 11, 6, 7, 12, 0, 5, 14, 9],
    [10, 0, 9, 14, 6, 3, 15, 5, 1, 13, 12, 7, 11, 4, 2, 8,
     13, 7, 0, 9, 3, 4, 6, 10, 2, 8, 5, 14, 12, 11, 15, 1,
     13, 6, 4, 9, 8, 15, 3, 0, 11, 1, 2, 12, 5, 10, 14, 7,
     1, 10, 13, 0, 6, 9, 8, 7, 4, 15, 14, 3, 11, 5, 2, 12],
    [7, 13, 14, 3, 0, 6, 9, 10, 1, 2, 8, 5, 11, 12, 4, 15,
     13, 8, 11, 5, 6, 15, 0, 3, 4, 7, 2, 12, 1, 10, 14, 9,
     10, 6, 9, 0, 12, 11, 7, 13, 15, 1, 3, 14, 5, 2, 8, 4,
     3, 15, 0, 6, 10, 1, 13, 8, 9, 4, 5, 11, 12, 7, 2, 14],
    [2, 12, 4, 1, 7, 10, 11, 6, 8, 5, 3, 15, 13, 0, 14, 9,
     14, 11, 2, 12, 4, 7, 13, 1, 5, 0, 15, 10, 3, 9, 8, 6,
     4, 2, 1, 11, 10, 13, 7, 8, 15, 9, 12, 5, 6, 3, 0, 14,
     11, 8, 12, 7, 1, 14, 2, 13, 6, 15, 0, 9, 10, 4, 5, 3],
    [12, 1, 10, 15, 9, 2, 6, 8, 0, 13, 3, 4, 14, 7, 5, 11,
     10, 15, 4, 2, 7, 12, 9, 5, 6, 1, 13, 14, 0, 11, 3, 8,
     9, 14, 15, 5, 2, 8, 12, 3, 7, 0, 4, 10, 1, 13, 11, 6,
     4, 3, 2, 12, 9, 5, 15, 10, 11, 14, 1, 7, 6, 0, 8, 13],
    [4, 11, 2, 14, 15, 0, 8, 13, 3, 12, 9, 7, 5, 10, 6, 1,
     13, 0, 11, 7, 4, 9, 1, 10, 14, 3, 5, 12, 2, 15, 8, 6,
     1, 4, 11, 13, 12, 3, 7, 14, 10, 15, 6, 8, 0, 5, 9, 2,
     6, 11, 13, 8, 1, 4, 10, 7, 9, 5, 0, 15, 14, 2, 3, 12],
    [13, 2, 8, 4, 6, 15, 11, 1, 10, 9, 3, 14, 5, 0, 12, 7,
     1, 15, 13, 8, 10, 3, 7, 4, 12, 5, 6, 11, 0, 14, 9, 2,
     7, 11, 4, 1, 9, 12, 14, 2, 0, 6, 10, 13, 15, 3, 5, 8,
     2, 1, 14, 7, 4, 10, 8, 13, 15, 12, 9, 0, 3, 5, 6, 11],
]


def _permute(block: int, table: list[int], in_bits: int) -> int:
    out = 0
    for position in table:
        out = (out << 1) | ((block >> (in_bits - position)) & 1)
    return out


def _des_subkeys(key: int) -> list[int]:
    key = _permute(key, _PC1, 64)
    c, d = (key >> 28) & 0xFFFFFFF, key & 0xFFFFFFF
    subkeys = []
    for shift in _SHIFTS:
        c = ((c << shift) | (c >> (28 - shift))) & 0xFFFFFFF
        d = ((d << shift) | (d >> (28 - shift))) & 0xFFFFFFF
        subkeys.append(_permute((c << 28) | d, _PC2, 56))
    return subkeys


def _des_block(block: int, subkeys: list[int]) -> int:
    block = _permute(block, _IP, 64)
    left, right = (block >> 32) & 0xFFFFFFFF, block & 0xFFFFFFFF
    for subkey in subkeys:
        expanded = _permute(right, _E, 32) ^ subkey
        substituted = 0
        for i in range(8):
            chunk = (expanded >> (42 - 6 * i)) & 0x3F
            row = ((chunk & 0x20) >> 4) | (chunk & 1)
            col = (chunk >> 1) & 0xF
            substituted = (substituted << 4) | _SBOXES[i][row * 16 + col]
        left, right = right, left ^ _permute(substituted, _P, 32)
    return _permute((right << 32) | left, _FP, 64)


def des_encrypt_block(key: bytes, data: bytes) -> bytes:
    """Encrypt one 8-byte block with DES-ECB (used by VNC Authentication)."""
    if len(key) != 8 or len(data) != 8:
        raise ValueError("DES requires an 8-byte key and an 8-byte block")
    key_int = int.from_bytes(key, "big")
    return _des_block(int.from_bytes(data, "big"), _des_subkeys(key_int)).to_bytes(8, "big")


def vnc_password_key(password: str) -> bytes:
    """Derive the DES key VNC uses: password bytes, each bit-reversed, padded."""
    raw = password.encode("ascii", errors="ignore")[:8].ljust(8, b"\x00")
    return bytes(int(f"{byte:08b}"[::-1], 2) for byte in raw)


def vnc_auth_response(password: str, challenge: bytes) -> bytes:
    """Compute the 16-byte response to a VNC Authentication challenge."""
    if len(challenge) != 16:
        raise ValueError("VNC challenge must be 16 bytes")
    key = vnc_password_key(password)
    return des_encrypt_block(key, challenge[:8]) + des_encrypt_block(key, challenge[8:])


# ---------------------------------------------------------------------------
# Keysym mapping
# ---------------------------------------------------------------------------

def keysym_for(key: str) -> int:
    """Map a key name or single character to an X11 keysym.

    Printable ASCII maps to itself; named keys (``return``, ``f5``, ``command``)
    are looked up in the table.  Raises ``ValueError`` for anything unknown so
    an unmappable key fails loudly instead of silently typing the wrong thing.
    """
    token = (key or "").strip()
    if not token:
        raise ValueError("empty key name")
    lowered = token.lower()
    if lowered in _KEYSYMS:
        return _KEYSYMS[lowered]
    if len(token) == 1 and 0x20 <= ord(token) <= 0x7E:
        return ord(token)
    raise ValueError(f"unsupported key: {key!r}")


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

class RFBClient:
    """A synchronous RFB 3.8 client supporting Raw framebuffer updates."""

    def __init__(
        self,
        host: str,
        port: int = 5900,
        *,
        password: str | None = None,
        timeout: float = 10.0,
    ) -> None:
        self.host = host
        self.port = int(port)
        self.password = password
        self.timeout = float(timeout)
        self._sock: socket.socket | None = None
        self.width = 0
        self.height = 0
        self._pixel_format: tuple[int, int, int, bool, bool, int, int, int, int, int, int, int, int] | None = None

    # -- lifecycle ---------------------------------------------------------

    @property
    def connected(self) -> bool:
        return self._sock is not None

    def connect(self) -> None:
        try:
            self._sock = socket.create_connection((self.host, self.port), timeout=self.timeout)
        except OSError as exc:
            raise RFBError(f"cannot reach VNC server at {self.host}:{self.port}: {exc}") from exc
        self._sock.settimeout(self.timeout)
        try:
            self._handshake()
        except Exception:
            self.close()
            raise

    def close(self) -> None:
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None

    def __enter__(self) -> "RFBClient":
        self.connect()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- socket helpers ----------------------------------------------------

    def _recv(self, count: int) -> bytes:
        assert self._sock is not None
        chunks = bytearray()
        while len(chunks) < count:
            try:
                chunk = self._sock.recv(count - len(chunks))
            except socket.timeout as exc:
                raise RFBError(
                    f"timed out reading {count} bytes from the VNC server"
                ) from exc
            if not chunk:
                raise RFBError("VNC server closed the connection")
            chunks.extend(chunk)
        return bytes(chunks)

    def _send(self, data: bytes) -> None:
        assert self._sock is not None
        self._sock.sendall(data)

    # -- handshake ---------------------------------------------------------

    def _handshake(self) -> None:
        version = self._recv(12)
        if not version.startswith(b"RFB "):
            raise RFBError(f"not a VNC server (greeting {version!r})")
        # Always answer with 3.8; servers downgrade to their own version.
        self._send(b"RFB 003.008\n")

        count = self._recv(1)[0]
        if count == 0:
            reason = self._recv(struct.unpack(">I", self._recv(4))[0]).decode(
                "utf-8", errors="replace"
            )
            raise RFBError(f"VNC server refused the connection: {reason}")
        offered = set(self._recv(count))
        self._authenticate(offered)

        self._send(struct.pack(">B", 1))  # ClientInit: shared session
        width, height = struct.unpack(">HH", self._recv(4))
        self.width, self.height = width, height
        self._pixel_format = struct.unpack(">BBBBHHHBBBxxx", self._recv(16))
        self._configure_pixel_format()

    def _authenticate(self, offered: set[int]) -> None:
        if SEC_TYPE_NONE in offered:
            self._send(struct.pack(">B", SEC_TYPE_NONE))
        elif SEC_TYPE_VNC_AUTH in offered:
            if not self.password:
                raise RFBError(
                    "VNC server requires a password but none is configured "
                    "(set SONIC_MACOS_VNC_PASSWORD)"
                )
            self._send(struct.pack(">B", SEC_TYPE_VNC_AUTH))
            challenge = self._recv(16)
            self._send(vnc_auth_response(self.password, challenge))
        elif SEC_TYPE_APPLE_ARD in offered:
            raise RFBError(
                "VNC server offers only Apple ARD authentication, which needs "
                "RSA/AES support. Configure a plain VNC password instead."
            )
        else:
            raise RFBError(f"no supported VNC security type (server offers {sorted(offered)})")

        result = struct.unpack(">I", self._recv(4))[0]
        if result != 0:
            if result == 1:
                raise RFBError("VNC authentication failed (wrong password)")
            if result == 2:
                raise RFBError("VNC authentication failed (too many attempts)")
            reason = self._recv(result).decode("utf-8", errors="replace")
            raise RFBError(f"VNC authentication failed: {reason}")

    def _configure_pixel_format(self) -> None:
        # 32 bpp, depth 24, little-endian, true-colour, RGB order (max 255,
        # shift 16/8/0).  This is the layout QEMU's VNC server emits natively.
        pixel_format = struct.pack(
            ">BBBBHHHBBBxxx",
            32, 24, 0, 1, 255, 255, 255, 16, 8, 0,
        )
        self._send(struct.pack(">Bxxx", 0) + pixel_format)
        self._send(struct.pack(">BBH", 2, 0, 1) + struct.pack(">i", ENCODING_RAW))

    # -- framebuffer -------------------------------------------------------

    def framebuffer(self, *, incremental: bool = False) -> np.ndarray:
        """Request and return the framebuffer as an ``(h, w, 3)`` uint8 array."""
        if self._sock is None:
            raise RFBError("not connected")
        self._send(
            struct.pack(">BBHHHH", 3, 1 if incremental else 0, 0, 0, self.width, self.height)
        )
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        drew = False
        # Skip asynchronous server messages (Bell, ServerCutText) that may
        # arrive before the FramebufferUpdate we asked for.
        while True:
            msg_type, _pad, count = struct.unpack(">BBH", self._recv(4))
            if msg_type == 0:  # FramebufferUpdate
                break
            if msg_type == 2:  # Bell
                continue
            if msg_type == 3:  # ServerCutText
                self._recv(3)
                length = struct.unpack(">I", self._recv(4))[0]
                self._recv(length)
                continue
            raise RFBError(f"unexpected server message {msg_type} while awaiting framebuffer")

        for _ in range(count):
            x, y, width, height, encoding = struct.unpack(">HHHHi", self._recv(12))
            if encoding != ENCODING_RAW:
                raise RFBError(f"server answered with unsupported encoding {encoding}")
            pixels = self._recv(width * height * 4)
            rect = np.frombuffer(pixels, dtype=np.uint8).reshape(height, width, 4)
            frame[y:y + height, x:x + width] = rect[:, :, [2, 1, 0]]
            drew = True
        if not drew:
            raise RFBError("server returned no framebuffer rectangles")
        return frame

    # -- input -------------------------------------------------------------

    def pointer(self, x: int, y: int, button_mask: int = 0) -> None:
        if self._sock is None:
            raise RFBError("not connected")
        self._send(
            struct.pack(">BBHH", 5, button_mask & 0xFF, max(0, int(x)), max(0, int(y)))
        )

    def key(self, keysym: int, down: bool) -> None:
        if self._sock is None:
            raise RFBError("not connected")
        self._send(struct.pack(">BBxxI", 4, 1 if down else 0, keysym))

    def key_press(self, key: str) -> None:
        keysym = keysym_for(key)
        self.key(keysym, True)
        self.key(keysym, False)

    def chord(self, keys: list[str]) -> None:
        """Press several keys together (e.g. ``["command", "c"]``)."""
        if not keys:
            raise ValueError("chord requires at least one key")
        keysyms = [keysym_for(k) for k in keys]
        for keysym in keysyms:
            self.key(keysym, True)
        for keysym in reversed(keysyms):
            self.key(keysym, False)

    def type_text(self, text: str) -> None:
        for character in text:
            self.key_press(character)

    def metadata(self) -> dict[str, Any]:
        return {"width": self.width, "height": self.height, "host": self.host, "port": self.port}
