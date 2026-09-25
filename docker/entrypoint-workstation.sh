#!/bin/bash
set -e

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

# Ensure container web browser wrapper has --no-sandbox, CDP port 9222, and First Run state initialized
if [ -x /usr/bin/google-chrome-stable ]; then
    mkdir -p /root/.config/google-chrome
    touch "/root/.config/google-chrome/First Run"
    rm -f /root/.config/google-chrome/Singleton* 2>/dev/null || true
    cat << 'EOF' > /usr/local/bin/chrome
#!/bin/bash
mkdir -p /root/.config/google-chrome
touch "/root/.config/google-chrome/First Run"
rm -f /root/.config/google-chrome/Singleton* 2>/dev/null
exec /usr/bin/google-chrome-stable --no-sandbox --disable-dev-shm-usage --no-first-run --disable-fre --no-default-browser-check --remote-debugging-port=9222 "$@"
EOF
    chmod +x /usr/local/bin/chrome
fi

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
