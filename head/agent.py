#!/usr/bin/env python3
"""Pi agent (offload mode) — hardware-PWM servo + raw MJPEG stream + laptop-driven overlay.

  * serves the camera as MJPEG on :8000  (browser index at /, raw stream at /stream)
  * UDP :9999  -> target servo angle (text float) from the laptop tracker
  * UDP :9998  -> overlay metadata "x,y,w,h,correcting"  (or "none") so the boxes/lines
                  the laptop computes get drawn onto THIS feed (visible in any browser)

  * servo runs on the Pi HARDWARE PWM (GPIO18 = pwmchip0 ch0) via sysfs -> silicon-timed,
    jitter-free even under load (no gpiozero/lgpio software-PWM buzz). A slew thread eases
    current -> target at ~100 Hz for smooth glides.

Needs the PWM overlay in /boot/firmware/config.txt, then a reboot:
    dtoverlay=pwm,pin=18,func=2
    dtparam=audio=off
Run by systemd (kitchen-vision.service). Heavy face detection runs on the laptop."""
import os, time, threading, socket
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import cv2

W, H = 640, 480
PORT, UDP_PORT, OVERLAY_PORT = 8000, 9999, 9998
ANGLE_MIN, ANGLE_MAX = 10.0, 170.0
START_ANGLE = 90.0
SLEW_HZ = 100.0
MAX_DEG_PER_S = 120.0            # servo glide-speed cap; lower = gentler, higher = snappier
STEP_MAX = MAX_DEG_PER_S / SLEW_HZ
CX = W // 2
OUTER_PX = 200                  # mirror track_client.py: pan-trigger guide lines
BOX_TTL = 0.4                   # s: hide the face box if no fresh overlay packet arrives
JPEG_QUALITY = 70


def clamp(v, lo, hi):
    return lo if v < lo else hi if v > hi else v


class ServoPWM:
    """SG90 on the Pi hardware PWM via sysfs. 50 Hz frame, 600-2400 us pulse."""
    CHIP = "/sys/class/pwm/pwmchip0"   # GPIO18 via 'dtoverlay=pwm,pin=18,func=2'
    CH = 0
    PERIOD_NS = 20_000_000             # 50 Hz
    MIN_US, MAX_US = 600, 2400

    def __init__(self):
        if not os.path.isdir(self.CHIP):
            raise SystemExit(f"{self.CHIP} missing - enable the pwm overlay in config.txt + reboot")
        base = f"{self.CHIP}/pwm{self.CH}"
        if not os.path.isdir(base):
            self._w(f"{self.CHIP}/export", self.CH)
            time.sleep(0.2)
        self.enable_p = f"{base}/enable"
        self.period_p = f"{base}/period"
        self.duty_p = f"{base}/duty_cycle"
        try:
            self._w(self.enable_p, 0)        # safe to re-init on a restart
        except OSError:
            pass
        self._w(self.period_p, self.PERIOD_NS)
        self._w(self.duty_p, self._us(START_ANGLE))
        self._w(self.enable_p, 1)

    def _w(self, path, val):
        with open(path, "w") as f:
            f.write(str(int(val)))

    def _us(self, angle):
        angle = clamp(angle, 0.0, 180.0)
        return int(round((self.MIN_US + angle / 180.0 * (self.MAX_US - self.MIN_US)) * 1000))

    def write_angle(self, angle):
        self._w(self.duty_p, self._us(angle))

    def release(self):
        try:
            self._w(self.duty_p, 0)          # stop driving (servo relaxes); channel stays exported
        except OSError:
            pass


servo = ServoPWM()

_state_lock = threading.Lock()
target = START_ANGLE
current = START_ANGLE
box = None              # (x, y, w, h, correcting)
box_t = 0.0

_frame_lock = threading.Lock()
_latest = None              # annotated JPEG (Pi overlay drawn) -> /stream
_latest_raw = None          # unannotated JPEG (clean frame)    -> /raw (laptop Station pulls this)


def slew():
    """Own the PWM: ease 'current' toward 'target' at a capped rate -> smooth glide."""
    global current
    period = 1.0 / SLEW_HZ
    while True:
        with _state_lock:
            tgt = target
        d = tgt - current
        if abs(d) > 0.05:
            current += clamp(d, -STEP_MAX, STEP_MAX)
            servo.write_angle(current)
        time.sleep(period)


def servo_listener():
    global target
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind(("0.0.0.0", UDP_PORT))
    while True:
        try:
            data, _ = s.recvfrom(64)
            a = clamp(float(data.decode().strip()), ANGLE_MIN, ANGLE_MAX)
            with _state_lock:
                target = a
        except (ValueError, OSError):
            pass


def overlay_listener():
    global box, box_t
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind(("0.0.0.0", OVERLAY_PORT))
    while True:
        try:
            data, _ = s.recvfrom(64)
            msg = data.decode().strip()
            if msg == "none":
                with _state_lock:
                    box = None
            else:
                x, y, w, h, c = msg.split(",")
                with _state_lock:
                    box = (int(x), int(y), int(w), int(h), int(c))
                    box_t = time.monotonic()
        except (ValueError, OSError):
            pass


def draw_overlay(frame):
    with _state_lock:
        b, bt, ang = box, box_t, current
    # static guide lines: pan-trigger edges (red-ish) + centre (blue)
    cv2.line(frame, (CX - OUTER_PX, 0), (CX - OUTER_PX, H), (60, 60, 200), 1)
    cv2.line(frame, (CX + OUTER_PX, 0), (CX + OUTER_PX, H), (60, 60, 200), 1)
    cv2.line(frame, (CX, 0), (CX, H), (255, 0, 0), 1)
    n, statetxt = 0, "hold"
    if b is not None and time.monotonic() - bt <= BOX_TTL:
        x, y, w, h, c = b
        n = 1
        col = (0, 165, 255) if c else (0, 255, 0)   # orange while recentering, green while holding
        statetxt = "recenter" if c else "hold"
        cv2.rectangle(frame, (x, y), (x + w, y + h), col, 2)
        cv2.line(frame, (x + w // 2, 0), (x + w // 2, H), col, 1)
    cv2.putText(frame, f"angle={ang:.0f}  faces={n}  {statetxt}", (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)


def open_cam():
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
    return cap


def cam():
    global _latest, _latest_raw
    cap = open_cam()
    for _ in range(5):           # warm up / let exposure settle
        cap.read()
    fails = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            fails += 1
            if fails >= 30:      # camera dropped -> reopen
                try:
                    cap.release()
                except Exception:
                    pass
                time.sleep(0.5)
                cap = open_cam()
                fails = 0
            else:
                time.sleep(0.02)
            continue
        fails = 0
        # Encode the RAW frame first (served at /raw) so the laptop Station can pull a clean
        # feed and draw its OWN boxes/names without the Pi's overlay underneath. Then draw the
        # Pi's overlay and encode /stream (what a plain browser pointed at the Pi still sees).
        ok_raw, jpg_raw = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        if ok_raw:
            with _frame_lock:
                _latest_raw = jpg_raw.tobytes()
        draw_overlay(frame)
        ok, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        if ok:
            with _frame_lock:
                _latest = jpg.tobytes()


PAGE = (b"<html><body style='margin:0;background:#111;text-align:center'>"
        b"<img src='/stream' style='max-width:100%'></body></html>")


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _serve_mjpeg(self, raw):
        """Stream the latest JPEG as multipart/x-mixed-replace. raw=True -> the clean frame
        (/raw, for the laptop Station); raw=False -> the Pi-annotated frame (/stream)."""
        self.send_response(200)
        self.send_header("Cache-Control", "no-cache, private")
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=FRAME")
        self.end_headers()
        try:
            while True:
                with _frame_lock:
                    buf = _latest_raw if raw else _latest
                if buf is None:
                    time.sleep(0.05)
                    continue
                self.wfile.write(b"--FRAME\r\nContent-Type: image/jpeg\r\n")
                self.wfile.write(("Content-Length: %d\r\n\r\n" % len(buf)).encode())
                self.wfile.write(buf)
                self.wfile.write(b"\r\n")
                time.sleep(0.04)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_GET(self):
        if self.path == "/":
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(PAGE)
        elif self.path == "/stream":
            self._serve_mjpeg(raw=False)
        elif self.path == "/raw":
            self._serve_mjpeg(raw=True)
        else:
            self.send_response(404)
            self.end_headers()


def main():
    for fn in (slew, servo_listener, overlay_listener, cam):
        threading.Thread(target=fn, daemon=True).start()
    print("agent up: MJPEG :%d  servo-UDP :%d  overlay-UDP :%d  (hardware PWM, slew %.0f deg/s)"
          % (PORT, UDP_PORT, OVERLAY_PORT, MAX_DEG_PER_S), flush=True)
    try:
        ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
    finally:
        servo.release()


if __name__ == "__main__":
    main()
