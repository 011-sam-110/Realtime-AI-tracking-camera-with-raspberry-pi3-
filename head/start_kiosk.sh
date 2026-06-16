#!/bin/bash
# Show the camera stream full-screen on the Pi's own monitor (Chromium kiosk).
# Works from the desktop autostart, or over SSH into the running Wayland session.
: "${XDG_RUNTIME_DIR:=/run/user/1000}"
: "${WAYLAND_DISPLAY:=wayland-0}"
export XDG_RUNTIME_DIR WAYLAND_DISPLAY
until curl -sf http://localhost:8000/ >/dev/null 2>&1; do sleep 1; done
exec chromium --kiosk --noerrdialogs --disable-infobars \
  --disable-session-crashed-bubble --disable-features=Translate \
  http://localhost:8000/
