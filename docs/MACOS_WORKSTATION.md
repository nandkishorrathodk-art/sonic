# macOS Workstation (Docker-OSX)

SONIC can drive a **macOS desktop** instead of the default Linux XFCE
workstation, using [Docker-OSX](https://github.com/sickcodes/docker-osx)
(`sickcodes/docker-osx`). Docker-OSX runs macOS inside a QEMU virtual machine
that is itself packaged as a Docker container.

This is an **opt-in** body. The default `sonic-desktop-workstation` (Linux)
remains the default; nothing changes unless you select the macOS provider.

## Why it is a different kind of body

| | Linux workstation | macOS (Docker-OSX) |
|---|---|---|
| Guest OS | Debian + XFCE | macOS |
| Host requirement | Docker only | Docker **+ `/dev/kvm`** |
| GUI control | X11 (`xdotool`, `wmctrl`, `scrot`) | **VNC / RFB** (QEMU framebuffer) |
| Terminal / files / git | `docker exec` | **SSH** into the guest |
| Window tiling | `wmctrl` | AppleScript over SSH (limited) |

There is **no X11 inside the guest**, so the Linux provider's `xdotool`/`scrot`
path cannot be reused. `MacOSDockerComputerProvider` talks RFB for pixels and
pointer/keyboard, and SSH for execution.

## Requirements

- `/dev/kvm` on the host (hardware virtualisation). Without it the VM cannot
  boot, and the provider **fails closed** with a clear error — it never reports
  a desktop that does not exist.
- An x86-64 host (Docker-OSX does not emulate on ARM).
- ~16 GB RAM and a 20+ GB disk image for the guest.

Check KVM availability:

```bash
ls -l /dev/kvm        # must exist
```

## Starting it

```bash
docker compose -f docker-compose.yml -f docker-compose.macos.yml up -d macos-workstation
```

**First boot installs macOS.** Connect a VNC viewer to `127.0.0.1:5901`,
complete the installer, then enable **Remote Login** (System Settings → General
→ Sharing) so SONIC can reach the guest over SSH. The disk image lives in the
`macos_image` volume, so the install survives container restarts.

## Pointing SONIC at it

```bash
export SONIC_WORKSTATION_PROVIDER=macos_docker_osx
```

Configuration (all optional, with defaults):

| Variable | Default | Meaning |
|---|---|---|
| `SONIC_MACOS_CONTAINER` | `sonic-macos-desktop` | Docker container name |
| `SONIC_MACOS_VNC_HOST` | `127.0.0.1` | VNC host |
| `SONIC_MACOS_VNC_PORT` | `5901` | VNC port |
| `SONIC_MACOS_VNC_PASSWORD` | *(none)* | VNC password, if the server requires one |
| `SONIC_MACOS_SSH_HOST` | VNC host | SSH host |
| `SONIC_MACOS_SSH_PORT` | `50922` | SSH port |
| `SONIC_MACOS_SSH_USER` | `sonic` | SSH user |
| `SONIC_MACOS_SSH_KEY` | *(none)* | Private key for SSH auth |
| `SONIC_MACOS_SSH_PASSWORD` | *(none)* | Password auth (needs `sshpass`) |
| `SONIC_MACOS_WORKSPACE` | `/Users/sonic/workspace` | Default workspace path |

### Authentication notes

- **VNC:** security types `None` (1) and **VNC Authentication** (2) are
  supported; VNC auth uses a DES challenge/response implemented in
  `sonic/computer/rfb.py`, so no `cryptography` dependency is required. Apple
  ARD (type 33) is rejected with a clear message — configure a plain VNC
  password instead.
- **SSH:** prefer a key (`SONIC_MACOS_SSH_KEY`). Password auth requires
  `sshpass` to be installed; otherwise the provider fails closed with an
  explicit message rather than hanging.

## What works, and what does not

**Works through RFB (GUI):** `GUI_MOVE`, `GUI_CLICK`, `GUI_DOUBLE_CLICK`,
`GUI_RIGHT_CLICK`, `GUI_DRAG`, `GUI_SCROLL` (vertical and horizontal),
`GUI_TYPE`, `GUI_KEYPRESS`, `GUI_HOTKEY` (e.g. `command+c`), and screenshots
(fed into the same AVCS perception loop as the Linux body).

**Works through SSH:** terminal execution, file read/write/list, git, process
list, application list, app launch (`open -a`), app quit (AppleScript),
Homebrew install/uninstall, and service status (`launchctl`).

**Fails closed (no VNC equivalent):** `WINDOW_TILE` and window-manager actions
that only `wmctrl` could express. `maximize_window` uses AppleScript and needs a
process name; `tile_workstation` returns `False` rather than pretending.

## Honesty

- No live macOS VM was booted to develop this: the host used for development
  had no `/dev/kvm`. The RFB client is verified against an in-process fake VNC
  server (handshake, framebuffer round-trip, pointer/keyboard events, VNC auth,
  DES against the FIPS test vector), and the provider is verified with a fake
  RFB client and a fake SSH runner. Real end-to-end macOS task completion is
  **not** claimed and must be measured on a KVM-capable host.
- Semantic GUI targeting and text entry still depend on the AVCS grounding and
  focus checkpoints, exactly as on the Linux body.
