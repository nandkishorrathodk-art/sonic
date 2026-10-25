#!/usr/bin/env bash
# Ensure the Docker daemon + sonic workstation container are up and capturable.
# This sandbox periodically resets the daemon/socket, so every live AVCS test
# starts by calling this. Idempotent: safe to run repeatedly.
set -uo pipefail

ROOT=/workspace/project/sonic
CONTAINER=sonic-desktop-workstation
IMAGE=sonic-workstation:latest
VNC_PASS=sonicvnc123

log() { echo "[ensure-workstation] $*"; }

# 1. daemon
if ! pgrep -x dockerd >/dev/null 2>&1; then
  log "starting dockerd"
  sudo nohup dockerd > /tmp/dockerd.log 2>&1 &
  for _ in $(seq 1 30); do
    [ -S /var/run/docker.sock ] && break
    sleep 1
  done
fi
sudo chmod 666 /var/run/docker.sock 2>/dev/null || true

# 2. image
if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  log "building image $IMAGE"
  docker build -t "$IMAGE" -f "$ROOT/docker/Dockerfile.workstation" "$ROOT" \
    > /tmp/wk_build.log 2>&1 || { log "BUILD FAILED"; tail -5 /tmp/wk_build.log; exit 1; }
fi

# 3. container
if ! docker ps --format '{{.Names}}' | grep -qx "$CONTAINER"; then
  log "starting container"
  docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
  docker run -d --name "$CONTAINER" \
    -p 127.0.0.1:5900:5900 -p 127.0.0.1:6080:6080 \
    -e SONIC_VNC_PASSWORD="$VNC_PASS" \
    -v "$ROOT/docker/entrypoint-workstation.sh:/entrypoint-workstation.sh:ro" \
    --init "$IMAGE" /bin/bash /entrypoint-workstation.sh >/dev/null 2>&1
fi

# 4. wait for a real X11 frame
for _ in $(seq 1 40); do
  n=$(docker exec "$CONTAINER" bash -c \
      "DISPLAY=:99 import -window root -depth 8 rgb:- 2>/dev/null | wc -c" 2>/dev/null || echo 0)
  if [ "$n" -ge 1000000 ]; then
    log "display LIVE (${n} bytes)"
    exit 0
  fi
  sleep 2
done

log "display did NOT come up"
docker logs --tail 15 "$CONTAINER" 2>&1
exit 1
