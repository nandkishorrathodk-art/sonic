#!/bin/bash
set -e

if ! command -v tesseract >/dev/null 2>&1; then
    echo "[!] tesseract-ocr is required for workstation NSI perception." >&2
    exit 1
fi

# Remove any stale X locks
rm -f /tmp/.X99-lock /tmp/.X11-unix/X99

# Start virtual framebuffer at 1280x800 with 24-bit color depth
echo "[*] Starting Xvfb on display :99..."
mkdir -p /tmp/.X11-unix /usr/local/bin /workspace /home/sonic
Xvfb :99 -screen 0 1280x800x24 &
XVFB_PID=$!
for _ in $(seq 1 20); do
    [ -S /tmp/.X11-unix/X99 ] && break
    sleep 0.2
done
kill -0 "$XVFB_PID"

export DISPLAY=:99
if [ -S /tmp/.X11-unix/X99 ]; then
    ln -sf /tmp/.X11-unix/X99 /tmp/.X11-unix/X0
fi

# Start D-Bus session daemon
if [ -z "$DBUS_SESSION_BUS_ADDRESS" ]; then
    eval $(dbus-launch --sh-syntax)
fi

# Ensure the generic Python command name exists
if [ -x /usr/bin/python3 ]; then
    ln -sf /usr/bin/python3 /usr/local/bin/python
fi

# Ensure smooth human-like mouse cursor glide helper exists for visible VNC cursor movement
cat << 'EOF' > /usr/local/bin/sonic_cursor.py
import sys, subprocess, time, math

def get_mouse():
    out = subprocess.check_output(['xdotool', 'getmouselocation'], env={'DISPLAY': ':99'}).decode()
    parts = dict(p.split(':') for p in out.strip().split() if ':' in p)
    return int(parts['x']), int(parts['y'])

def smooth_move(tx, ty, steps=25, duration=0.25):
    sx, sy = get_mouse()
    dt = max(0.005, duration / steps)
    for i in range(1, steps + 1):
        t = i / steps
        ease = 0.5 - 0.5 * math.cos(t * math.pi)
        cx = int(sx + (tx - sx) * ease)
        cy = int(sy + (ty - sy) * ease)
        subprocess.run(['xdotool', 'mousemove', str(cx), str(cy)], env={'DISPLAY': ':99'}, check=True)
        time.sleep(dt)

def click(button=1, repeat=1):
    subprocess.run(['xdotool', 'click', '--repeat', str(repeat), str(button)], env={'DISPLAY': ':99'}, check=True)

def move_and_click(tx, ty, button=1, repeat=1, steps=25, duration=0.25):
    smooth_move(tx, ty, steps, duration)
    time.sleep(0.05)
    click(button, repeat)

def drag(sx, sy, tx, ty):
    smooth_move(sx, sy)
    subprocess.run(['xdotool', 'mousedown', '1'], env={'DISPLAY': ':99'}, check=True)
    try:
        smooth_move(tx, ty)
    finally:
        subprocess.run(['xdotool', 'mouseup', '1'], env={'DISPLAY': ':99'}, check=True)

def type_text(text, delay=40):
    subprocess.run(['xdotool', 'type', '--delay', str(delay), text], env={'DISPLAY': ':99'}, check=True)

def key(k):
    subprocess.run(['xdotool', 'key', k], env={'DISPLAY': ':99'}, check=True)

if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv) > 1 else 'help'
    if cmd == 'move':
        smooth_move(int(sys.argv[2]), int(sys.argv[3]))
    elif cmd == 'click':
        if len(sys.argv) >= 4:
            move_and_click(int(sys.argv[2]), int(sys.argv[3]))
        else:
            click()
    elif cmd == 'double_click':
        if len(sys.argv) >= 4:
            move_and_click(int(sys.argv[2]), int(sys.argv[3]), repeat=2)
        else:
            click(repeat=2)
    elif cmd == 'drag':
        drag(*(int(value) for value in sys.argv[2:6]))
    elif cmd == 'type':
        type_text(sys.argv[2])
    elif cmd == 'key':
        key(sys.argv[2])
    elif cmd == 'pos':
        print(get_mouse())
EOF
chmod +x /usr/local/bin/sonic_cursor.py

# Ensure workspace and /home/sonic/workspace exist and link together
ln -sfn /workspace /home/sonic/workspace

# Start full XFCE4 desktop session
echo "[*] Starting XFCE4 desktop session..."
startxfce4 &
sleep 2

# Start x11vnc server on port 5900.  Passwordless VNC is never acceptable:
# the compose stack must provide a secret explicitly.
echo "[*] Starting x11vnc on port 5900..."
if [ -z "${SONIC_VNC_PASSWORD:-}" ]; then
  echo "[!] SONIC_VNC_PASSWORD is required; refusing to expose the desktop."
  exit 1
fi
mkdir -p /run/sonic
x11vnc -storepasswd "$SONIC_VNC_PASSWORD" /run/sonic/vnc.pass >/dev/null
chmod 600 /run/sonic/vnc.pass
# -cursor arrow -nocursorshape: render the X cursor directly into the VNC framebuffer
# so server-side programmatic mouse moves (e.g. xdotool) are visible live in noVNC.
export XCURSOR_THEME="${XCURSOR_THEME:-DMZ-White}"
export XCURSOR_SIZE="${XCURSOR_SIZE:-24}"
x11vnc -display :99 -forever -rfbauth /run/sonic/vnc.pass -shared -rfbport 5900 -bg -cursor arrow -nocursorshape

# Start noVNC websockify bridge on port 6080
echo "[*] Starting noVNC websockify bridge on port 6080..."
/usr/bin/python3 /usr/bin/websockify --web=/usr/share/novnc/ 6080 localhost:5900 &

echo "[+] SONIC Docker Workstation desktop ready on http://localhost:6080/vnc.html"

# Keep container alive
exec tail -f /dev/null
