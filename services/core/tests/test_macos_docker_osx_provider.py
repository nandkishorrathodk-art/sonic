"""Tests for the Docker-OSX (macOS) computer provider and its RFB client.

No live macOS VM is required: the RFB protocol is exercised against an
in-process fake VNC server, and the provider is driven with a fake RFB client
plus a fake SSH runner.  The one thing that cannot be faked -- the host's KVM
capability -- is asserted directly so the fail-closed path is real.
"""

from __future__ import annotations

import struct
import threading

import numpy as np
import pytest

from sonic.computer.macos_docker_osx import MacOSDockerComputerProvider
from sonic.computer.models import (
    ComputerWorkspaceStatus,
    GUIAction,
    GUIActionType,
)
from sonic.computer.rfb import (
    POINTER_BUTTON_LEFT,
    POINTER_BUTTON_RIGHT,
    POINTER_WHEEL_DOWN,
    POINTER_WHEEL_RIGHT,
    RFBClient,
    RFBError,
    des_encrypt_block,
    keysym_for,
    vnc_auth_response,
    vnc_password_key,
)

pytestmark = pytest.mark.no_live_infra


# ---------------------------------------------------------------------------
# DES / keysyms
# ---------------------------------------------------------------------------

class TestDesAndKeysyms:
    def test_fips_des_vector(self):
        # FIPS test vector: key 133457799BBCDFF1, plaintext 0123456789ABCDEF.
        out = des_encrypt_block(bytes.fromhex("133457799BBCDFF1"), bytes.fromhex("0123456789ABCDEF"))
        assert out.hex().upper() == "85E813540F0AB405"

    def test_vnc_password_key_bit_reversed(self):
        # 'a' = 0x61 = 0110_0001 -> bit-reversed 1000_0110 = 0x86.
        assert vnc_password_key("a")[0] == 0x86
        assert len(vnc_password_key("short")) == 8

    def test_vnc_auth_response_is_16_bytes(self):
        challenge = bytes(range(16))
        response = vnc_auth_response("secret", challenge)
        assert len(response) == 16

    def test_vnc_auth_response_requires_16_byte_challenge(self):
        with pytest.raises(ValueError):
            vnc_auth_response("secret", b"short")

    def test_keysym_mapping(self):
        assert keysym_for("a") == 0x61
        assert keysym_for("A") == 0x41
        assert keysym_for("return") == 0xFF0D
        assert keysym_for("Enter") == 0xFF0D
        assert keysym_for("command") == 0xFFEB
        assert keysym_for("f5") == 0xFFC2

    def test_unknown_key_is_rejected(self):
        with pytest.raises(ValueError):
            keysym_for("definitely-not-a-key")


# ---------------------------------------------------------------------------
# RFB protocol against a fake VNC server
# ---------------------------------------------------------------------------

class _FakeVNCServer(threading.Thread):
    """A minimal RFB 3.8 server that speaks just enough for the client."""

    def __init__(self, *, width=8, height=4, password=None, security=1):
        super().__init__(daemon=True)
        self.width, self.height = width, height
        self.password = password
        self.security = security
        self.received: list[bytes] = []
        self.pointer_events: list[tuple[int, int, int]] = []
        self.key_events: list[tuple[int, bool]] = []
        self.error: Exception | None = None
        self._server_sock = None
        self.port = None

    def run(self):
        import socket

        try:
            self._server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self._server_sock.bind(("127.0.0.1", 0))
            self.port = self._server_sock.getsockname()[1]
            self._server_sock.listen(1)
            conn, _ = self._server_sock.accept()
            with conn:
                self._serve(conn)
        except Exception as exc:  # pragma: no cover - surfaced via self.error
            self.error = exc

    def _serve(self, conn):
        conn.sendall(b"RFB 003.008\n")
        self._recv(conn, 12)  # client version
        conn.sendall(bytes([1, self.security]))  # one security type
        chosen = self._recv(conn, 1)[0]
        assert chosen == self.security
        if self.security == 2:
            challenge = bytes(range(16))
            conn.sendall(challenge)
            self._recv(conn, 16)  # client response (not validated here)
        conn.sendall(struct.pack(">I", 0))  # auth OK
        self._recv(conn, 1)  # ClientInit
        conn.sendall(struct.pack(">HH", self.width, self.height) + bytes(16))  # ServerInit
        self._recv(conn, 20)  # SetPixelFormat
        self._recv(conn, 8)  # SetEncodings
        while True:
            header = self._recv(conn, 1)
            if not header:
                return
            msg = header[0]
            if msg == 3:  # FramebufferUpdateRequest
                self._recv(conn, 9)
                pixels = bytes([10, 20, 30, 255]) * (self.width * self.height)
                conn.sendall(
                    struct.pack(">BBH", 0, 0, 1)
                    + struct.pack(">HHHHi", 0, 0, self.width, self.height, 0)
                    + pixels
                )
            elif msg == 4:  # KeyEvent
                data = self._recv(conn, 7)
                down = bool(data[0])
                keysym = struct.unpack(">I", data[3:7])[0]
                self.key_events.append((keysym, down))
            elif msg == 5:  # PointerEvent
                data = self._recv(conn, 5)
                self.pointer_events.append((data[0], *struct.unpack(">HH", data[1:5])))
            else:
                return

    @staticmethod
    def _recv(conn, count):
        buf = bytearray()
        while len(buf) < count:
            chunk = conn.recv(count - len(buf))
            if not chunk:
                return bytes(buf)
            buf.extend(chunk)
        return bytes(buf)


def _start_fake_server(**kwargs) -> _FakeVNCServer:
    server = _FakeVNCServer(**kwargs)
    server.start()
    for _ in range(200):
        if server.port is not None:
            break
        threading.Event().wait(0.01)
    assert server.port is not None, "fake VNC server did not start"
    return server


class TestRFBProtocol:
    def test_framebuffer_roundtrip_no_auth(self):
        server = _start_fake_server(width=6, height=3, security=1)
        with RFBClient("127.0.0.1", server.port, timeout=5) as client:
            frame = client.framebuffer()
            assert frame.shape == (3, 6, 3)
            # Server sent pixel (10,20,30) in BGR order -> client yields RGB.
            assert tuple(frame[0, 0]) == (30, 20, 10)

    def test_pointer_and_key_events_reach_server(self):
        server = _start_fake_server(security=1)
        with RFBClient("127.0.0.1", server.port, timeout=5) as client:
            client.pointer(12, 34, POINTER_BUTTON_LEFT)
            client.key_press("a")
        for _ in range(200):
            if server.pointer_events and len(server.key_events) >= 2:
                break
            threading.Event().wait(0.01)
        assert (POINTER_BUTTON_LEFT, 12, 34) in server.pointer_events
        assert (keysym_for("a"), True) in server.key_events
        assert (keysym_for("a"), False) in server.key_events

    def test_vnc_auth_password_accepted(self):
        server = _start_fake_server(security=2)
        with RFBClient("127.0.0.1", server.port, password="hunter2", timeout=5) as client:
            assert client.width == server.width
            client.framebuffer()

    def test_missing_password_fails_closed(self):
        server = _start_fake_server(security=2)
        with pytest.raises(RFBError, match="requires a password"):
            RFBClient("127.0.0.1", server.port, timeout=5).connect()

    def test_unreachable_server_fails_closed(self):
        with pytest.raises(RFBError, match="cannot reach VNC server"):
            RFBClient("127.0.0.1", 1, timeout=1).connect()


# ---------------------------------------------------------------------------
# Fake RFB client + fake SSH runner for the provider
# ---------------------------------------------------------------------------

class _FakeRFBClient:
    def __init__(self, host, port, *, password=None):
        self.host, self.port, self.password = host, port, password
        self.connected = False
        self.pointer_events: list[tuple[int, int, int]] = []
        self.key_events: list[tuple[int, bool]] = []
        self.width, self.height = 10, 6

    def connect(self):
        self.connected = True

    def close(self):
        self.connected = False

    def framebuffer(self, incremental=False):
        return np.zeros((self.height, self.width, 3), dtype=np.uint8)

    def pointer(self, x, y, mask=0):
        self.pointer_events.append((mask, x, y))

    def key(self, keysym, down):
        self.key_events.append((keysym, down))

    def key_press(self, key):
        self.key(keysym_for(key), True)
        self.key(keysym_for(key), False)

    def chord(self, keys):
        for key in keys:
            self.key(keysym_for(key), True)
        for key in reversed(keys):
            self.key(keysym_for(key), False)

    def type_text(self, text):
        for ch in text:
            self.key_press(ch)


class _SSHRecorder:
    """Records SSH commands and answers a few of them like a macOS shell."""

    def __init__(self, responses=None):
        self.commands: list[str] = []
        self.responses = responses or {}

    async def __call__(self, command: str, timeout: int):
        self.commands.append(command)
        for needle, reply in self.responses.items():
            if needle in command:
                return reply
        return 0, "", ""


def _provider(**kwargs) -> MacOSDockerComputerProvider:
    provider = MacOSDockerComputerProvider(
        rfb_factory=_FakeRFBClient,
        ssh_runner=kwargs.pop("ssh_runner", _SSHRecorder()),
        **kwargs,
    )
    return provider


async def _provision(provider) -> str:
    # Bypass the host KVM check: the VM lifecycle is out of scope for unit tests,
    # but everything downstream of create() is exercised for real.
    provider._workspace_owner = "acme"
    from sonic.computer.models import ComputerProfile, ComputerWorkspace, ComputerWorkspaceType

    provider._workspace = ComputerWorkspace(
        id=provider.host_workspace_id,
        tenant_id="acme",
        engagement_id="eng-1",
        workspace_type=ComputerWorkspaceType.MISSION_COMPUTER,
        profile=ComputerProfile.GENERIC_LINUX,
        provider_type="MacOSDockerComputerProvider",
        status=ComputerWorkspaceStatus.RUNNING,
    )
    return provider.host_workspace_id


class TestProviderCapability:
    def test_kvm_probe_matches_host(self):
        import pathlib

        assert MacOSDockerComputerProvider.kvm_available() == pathlib.Path("/dev/kvm").exists()

    @pytest.mark.asyncio
    async def test_create_fails_closed_without_kvm(self, monkeypatch):
        provider = _provider()
        monkeypatch.setattr(provider, "kvm_available", staticmethod(lambda: False))
        with pytest.raises(RuntimeError, match="requires hardware virtualisation"):
            await provider.create("acme", "eng-1")

    @pytest.mark.asyncio
    async def test_create_rejects_cross_tenant_reuse(self, monkeypatch):
        provider = _provider()
        monkeypatch.setattr(provider, "kvm_available", staticmethod(lambda: True))
        await provider.create("acme", "eng-1")
        with pytest.raises(RuntimeError, match="already owned by another tenant"):
            await provider.create("evil", "eng-2")

    @pytest.mark.asyncio
    async def test_get_or_create_home_provisions_through_create(self, monkeypatch):
        provider = _provider()
        monkeypatch.setattr(provider, "kvm_available", staticmethod(lambda: True))
        home = await provider.get_or_create_home("acme")
        assert home.id == provider.host_workspace_id
        assert home.status == ComputerWorkspaceStatus.RUNNING


class TestProviderScreenshotAndGui:
    @pytest.mark.asyncio
    async def test_screenshot_returns_real_frame_and_avcs_state(self):
        provider = _provider()
        ws = await _provision(provider)
        obs = await provider.screenshot(ws)
        assert obs.desktop_state == "INTERACTIVE"
        assert obs.width == 10 and obs.height == 6
        assert len(obs.raw_rgb) == 10 * 6 * 3
        assert obs.avcs_state.get("engine") == "avcs_reflex"

    @pytest.mark.asyncio
    async def test_screenshot_fails_closed_when_vnc_unavailable(self):
        def boom(*a, **k):
            raise RFBError("connection refused")

        provider = MacOSDockerComputerProvider(rfb_factory=boom, ssh_runner=_SSHRecorder())
        ws = await _provision(provider)
        obs = await provider.screenshot(ws)
        assert obs.desktop_state == "NO_DISPLAY"
        assert "unavailable" in obs.visible_text

    @pytest.mark.asyncio
    async def test_gui_click_dispatches_pointer(self):
        provider = _provider()
        ws = await _provision(provider)
        await provider.gui_action(ws, GUIAction(action=GUIActionType.CLICK, x=5, y=4))
        client = provider._rfb
        assert (POINTER_BUTTON_LEFT, 5, 4) in client.pointer_events

    @pytest.mark.asyncio
    async def test_gui_right_click_uses_right_button(self):
        provider = _provider()
        ws = await _provision(provider)
        await provider.gui_action(ws, GUIAction(action=GUIActionType.RIGHT_CLICK, x=2, y=3))
        assert (POINTER_BUTTON_RIGHT, 2, 3) in provider._rfb.pointer_events

    @pytest.mark.asyncio
    async def test_gui_type_and_hotkey(self):
        provider = _provider()
        ws = await _provision(provider)
        await provider.gui_action(ws, GUIAction(action=GUIActionType.TYPE, text="hi"))
        await provider.gui_action(ws, GUIAction(action=GUIActionType.HOTKEY, key="command+c"))
        keysyms = [k for k, _ in provider._rfb.key_events]
        assert keysym_for("h") in keysyms and keysym_for("i") in keysyms
        assert keysym_for("command") in keysyms and keysym_for("c") in keysyms

    @pytest.mark.asyncio
    async def test_gui_scroll_horizontal(self):
        provider = _provider()
        ws = await _provision(provider)
        await provider.gui_action(
            ws,
            GUIAction(action=GUIActionType.SCROLL, x=1, y=1, scroll_delta=2, scroll_axis="horizontal"),
        )
        assert (POINTER_WHEEL_RIGHT, 1, 1) in provider._rfb.pointer_events

    @pytest.mark.asyncio
    async def test_gui_scroll_vertical(self):
        provider = _provider()
        ws = await _provision(provider)
        await provider.gui_action(ws, GUIAction(action=GUIActionType.SCROLL, x=1, y=1, scroll_delta=1))
        assert (POINTER_WHEEL_DOWN, 1, 1) in provider._rfb.pointer_events

    @pytest.mark.asyncio
    async def test_gui_click_without_coordinates_fails_closed(self):
        provider = _provider()
        ws = await _provision(provider)
        with pytest.raises(RuntimeError, match="explicit screen pixel coordinates"):
            await provider.gui_action(ws, GUIAction(action=GUIActionType.CLICK))

    @pytest.mark.asyncio
    async def test_unsupported_gui_action_fails_closed(self):
        provider = _provider()
        ws = await _provision(provider)
        with pytest.raises(RuntimeError, match="no VNC equivalent"):
            await provider.gui_action(ws, GUIAction(action=GUIActionType.WINDOW_TILE))


class TestProviderSshSurfaces:
    @pytest.mark.asyncio
    async def test_terminal_runs_over_ssh(self):
        ssh = _SSHRecorder({"uname": (0, "Darwin\n", "")})
        provider = _provider(ssh_runner=ssh)
        ws = await _provision(provider)
        result = await provider.terminal(ws, "uname -a")
        assert result.exit_code == 0 and result.stdout == "Darwin\n"
        assert ssh.commands == ["uname -a"]

    @pytest.mark.asyncio
    async def test_write_file_base64_encodes_content(self):
        ssh = _SSHRecorder()
        provider = _provider(ssh_runner=ssh)
        ws = await _provision(provider)
        ok = await provider.write_file(ws, "/tmp/x.txt", "hello")
        assert ok is True
        assert any("base64 -d" in c for c in ssh.commands)

    @pytest.mark.asyncio
    async def test_application_list_parses_app_bundles(self):
        ssh = _SSHRecorder({"/Applications": (0, "Safari.app\nTerminal.app\n", "")})
        provider = _provider(ssh_runner=ssh)
        ws = await _provision(provider)
        apps = await provider.application_list(ws)
        assert apps == ["Safari", "Terminal"]

    @pytest.mark.asyncio
    async def test_launch_application_uses_open(self):
        ssh = _SSHRecorder()
        provider = _provider(ssh_runner=ssh)
        ws = await _provision(provider)
        assert await provider.launch_application(ws, "Safari") is True
        assert any(c.startswith("open -a") for c in ssh.commands)

    @pytest.mark.asyncio
    async def test_launch_application_fails_closed_on_error(self):
        ssh = _SSHRecorder({"open -a": (1, "", "no such app")})
        provider = _provider(ssh_runner=ssh)
        ws = await _provision(provider)
        assert await provider.launch_application(ws, "Nope") is False

    @pytest.mark.asyncio
    async def test_git_status_parses_porcelain(self):
        ssh = _SSHRecorder({"status --porcelain": (0, "?? new.txt\n", ""), "branch --show-current": (0, "main\n", "")})
        provider = _provider(ssh_runner=ssh)
        ws = await _provision(provider)
        info = await provider.git_action(ws, "status")
        assert info.branch == "main"
        assert info.untracked_files == ["new.txt"]

    @pytest.mark.asyncio
    async def test_status_is_degraded_without_container(self):
        provider = _provider()
        state = await provider.status(provider.host_workspace_id)
        assert state.status == ComputerWorkspaceStatus.DEGRADED
        assert state.open_applications == []


class TestProviderSelection:
    def test_get_computer_supports_macos_provider(self, monkeypatch):
        import sonic.api.routes.workstation as ws

        monkeypatch.setenv("SONIC_WORKSTATION_PROVIDER", "macos_docker_osx")
        monkeypatch.setattr(ws, "_primary_computer_instance", None)
        monkeypatch.setattr(ws, "_daytona_provider_instance", None)
        provider = ws.get_computer()
        assert isinstance(provider, MacOSDockerComputerProvider)

    def test_unknown_provider_still_rejected(self, monkeypatch):
        import sonic.api.routes.workstation as ws

        monkeypatch.setenv("SONIC_WORKSTATION_PROVIDER", "not_a_real_provider")
        monkeypatch.setattr(ws, "_primary_computer_instance", None)
        monkeypatch.setattr(ws, "_daytona_provider_instance", None)
        with pytest.raises(RuntimeError, match="SONIC_WORKSTATION_PROVIDER must be one of"):
            ws.get_computer()
