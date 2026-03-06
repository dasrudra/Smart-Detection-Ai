from ultralytics import YOLO
import cv2
import time
import os
import csv
import sqlite3
from datetime import datetime
from collections import defaultdict

# ----------------------------
# SETTINGS
# ----------------------------
MODEL_PATH = "models/yolov8n.pt"
CAMERA_INDEX = 0

RESIZE_W = None

CONF_THRES = 0.25
SHOW_ALL_CLASSES = True

ZONE_HALF_HEIGHT = 18
COOLDOWN_SEC = 0.8
LINE_MOVE_STEP = 8

ROI_MOVE_STEP = 12
ROI_RESIZE_STEP = 16

# Dashboard overlay
DASH_TOP_N = 8
PANEL_W = 420

# Daily / hourly logs (CSV backup)
LOG_ROOT = "logs"
DAILY_DIR = os.path.join(LOG_ROOT, "daily")
SNAPS_ROOT = os.path.join(DAILY_DIR, "snaps")       # snaps/YYYY-MM-DD/
EVENTS_ROOT = os.path.join(DAILY_DIR, "events")     # events/events_YYYY-MM-DD.csv
HOURLY_ROOT = os.path.join(DAILY_DIR, "hourly")     # hourly/hourly_YYYY-MM-DD.csv

os.makedirs(SNAPS_ROOT, exist_ok=True)
os.makedirs(EVENTS_ROOT, exist_ok=True)
os.makedirs(HOURLY_ROOT, exist_ok=True)

# SQLite
DB_DIR = "database"
DB_PATH = os.path.join(DB_DIR, "gate_events.db")
os.makedirs(DB_DIR, exist_ok=True)

# ----------------------------
# HELPERS
# ----------------------------
def today_str():
    return datetime.now().strftime("%Y-%m-%d")

def hour_str():
    return datetime.now().strftime("%H")  # 00..23

def side_of_zone(y, line_y, half_h):
    if y < (line_y - half_h):
        return "top"
    if y > (line_y + half_h):
        return "bottom"
    return "zone"

def in_roi(cx, cy, roi):
    x1, y1, x2, y2 = roi
    return (x1 <= cx <= x2) and (y1 <= cy <= y2)

def get_daily_events_path(day):
    return os.path.join(EVENTS_ROOT, f"events_{day}.csv")

def get_daily_hourly_path(day):
    return os.path.join(HOURLY_ROOT, f"hourly_{day}.csv")

def ensure_csv_header(path, header):
    new = not os.path.exists(path)
    f = open(path, "a", newline="", encoding="utf-8")
    w = csv.writer(f)
    if new:
        w.writerow(header)
        f.flush()
    return f, w

def save_snapshot(frame, day, direction, label, track_id, conf):
    day_dir = os.path.join(SNAPS_ROOT, day)
    os.makedirs(day_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"{ts}_{direction}_{label}_id{track_id}_{conf:.2f}.jpg"
    path = os.path.join(day_dir, filename)
    cv2.imwrite(path, frame)
    return path

def draw_dashboard(frame, fps, line_y, roi, in_total, out_total, in_by_class, out_by_class):
    h, w = frame.shape[:2]
    panel_w = min(PANEL_W, w)
    x0, y0 = 10, 10
    x1, y1 = x0 + panel_w, 10 + 28 * (DASH_TOP_N + 4)

    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x1, y1), (20, 20, 20), -1)
    alpha = 0.55
    frame[:] = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

    net = in_total - out_total
    cv2.putText(frame, "Gate Counter Dashboard", (x0 + 12, y0 + 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
    cv2.putText(frame, f"IN: {in_total}   OUT: {out_total}   NET: {net}",
                (x0 + 12, y0 + 58), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(frame, f"FPS: {fps:.1f}   LineY: {line_y}",
                (x0 + 12, y0 + 84), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
    cv2.putText(frame, f"ROI: {roi}", (x0 + 12, y0 + 108),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

    all_classes = set(in_by_class.keys()) | set(out_by_class.keys())
    ranked = sorted(
        [(c, in_by_class[c], out_by_class[c]) for c in all_classes],
        key=lambda t: (t[1] + t[2]),
        reverse=True
    )[:DASH_TOP_N]

    cv2.putText(frame, "Class-wise (Top)", (x0 + 12, y0 + 136),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

    yy = y0 + 162
    for cls, inc, outc in ranked:
        cv2.putText(frame, f"{cls:>12}  IN:{inc:<4} OUT:{outc:<4}",
                    (x0 + 12, yy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        yy += 24

# ----------------------------
# SQLITE SETUP
# ----------------------------
conn = sqlite3.connect(DB_PATH, check_same_thread=False)
cur = conn.cursor()

cur.execute("""
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    date TEXT NOT NULL,
    hour TEXT NOT NULL,
    direction TEXT NOT NULL,
    label TEXT NOT NULL,
    track_id INTEGER,
    conf REAL,
    roi TEXT,
    in_total INTEGER,
    out_total INTEGER,
    net_total INTEGER,
    snapshot_path TEXT
)
""")

cur.execute("""
CREATE INDEX IF NOT EXISTS idx_events_date_hour
ON events(date, hour)
""")

cur.execute("""
CREATE TABLE IF NOT EXISTS hourly_summary (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    date TEXT NOT NULL,
    hour TEXT NOT NULL,
    label TEXT NOT NULL,
    direction TEXT NOT NULL,
    count INTEGER NOT NULL,
    UNIQUE(date, hour, label, direction)
)
""")

conn.commit()

def db_insert_event(ts_str, direction, label, track_id, conf, roi, in_total, out_total, net_total, snapshot_path):
    day = ts_str.split(" ")[0]
    hr = ts_str.split(" ")[1][:2]

    cur.execute("""
    INSERT INTO events(ts, date, hour, direction, label, track_id, conf, roi, in_total, out_total, net_total, snapshot_path)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (ts_str, day, hr, direction, label, track_id, conf, roi, in_total, out_total, net_total, snapshot_path))
    conn.commit()

def db_upsert_hourly(date, hour, label, direction, inc=1):
    cur.execute("""
    INSERT INTO hourly_summary(date, hour, label, direction, count)
    VALUES (?, ?, ?, ?, ?)
    ON CONFLICT(date, hour, label, direction)
    DO UPDATE SET count = count + ?
    """, (date, hour, label, direction, inc, inc))
    conn.commit()

# ----------------------------
# INIT MODEL + CAMERA
# ----------------------------
model = YOLO(MODEL_PATH)
cap = cv2.VideoCapture(CAMERA_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
if not cap.isOpened():
    print("Error: Could not open camera.")
    raise SystemExit

WIN = "Smart Office Gate Counter - Rudra"
cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)

LINE_Y = None
ROI = None

track_state = {}  # track_id -> {"enter_side": None, "in_zone": False, "last_count_time": 0.0}

# totals
in_total = 0
out_total = 0
in_by_class = defaultdict(int)
out_by_class = defaultdict(int)

# open today's CSV files (backup)
current_day = today_str()
events_path = get_daily_events_path(current_day)
hourly_path = get_daily_hourly_path(current_day)

events_file, events_writer = ensure_csv_header(
    events_path,
    ["timestamp", "direction", "label", "track_id", "conf", "roi", "in_total", "out_total", "net_total", "snapshot_path"]
)

hourly_file, hourly_writer = ensure_csv_header(
    hourly_path,
    ["date", "hour", "label", "direction", "count"]
)

prev_time = time.time()

print("Controls:")
print("  Q = quit")
print("  W/S = move counting line up/down")
print("  I/K/J/L = move ROI up/down/left/right")
print("  [ / ] = resize ROI smaller/bigger")

try:
    while True:
        # Rotate daily CSVs if date changed (DB continues normally)
        day_now = today_str()
        if day_now != current_day:
            events_file.close()
            hourly_file.close()

            current_day = day_now
            events_path = get_daily_events_path(current_day)
            hourly_path = get_daily_hourly_path(current_day)

            events_file, events_writer = ensure_csv_header(
                events_path,
                ["timestamp", "direction", "label", "track_id", "conf", "roi", "in_total", "out_total", "net_total", "snapshot_path"]
            )
            hourly_file, hourly_writer = ensure_csv_header(
                hourly_path,
                ["date", "hour", "label", "direction", "count"]
            )

        ret, frame = cap.read()
        if not ret:
            break

        # resize for speed
        h, w = frame.shape[:2]
        if RESIZE_W is not None and w != RESIZE_W:
            new_h = int(h * (RESIZE_W / w))
            frame = cv2.resize(frame, (RESIZE_W, new_h))
            h, w = frame.shape[:2]

        if LINE_Y is None:
            LINE_Y = h // 2

        if ROI is None:
            roi_w = int(w * 0.60)
            roi_h = int(h * 0.65)
            x1 = (w - roi_w) // 2
            y1 = (h - roi_h) // 2
            ROI = [x1, y1, x1 + roi_w, y1 + roi_h]

        zone_top = LINE_Y - ZONE_HALF_HEIGHT
        zone_bot = LINE_Y + ZONE_HALF_HEIGHT

        results = model.track(frame, persist=True, verbose=False)

        if results and results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes
            ids = boxes.id.int().tolist()
            classes = boxes.cls.int().tolist()
            confs = boxes.conf.tolist()
            xyxy_list = boxes.xyxy.tolist()

            now_ts = time.time()

            for xyxy, track_id, cls, conf in zip(xyxy_list, ids, classes, confs):
                if conf < CONF_THRES:
                    continue

                label = model.names[int(cls)]

                x1, y1, x2, y2 = map(int, xyxy)
                cy = int((y1 + y2) / 2)
                cx = int((x1 + x2) / 2)

                # draw detections anywhere
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)
                cv2.putText(frame, f"{label} ID:{track_id} {conf:.2f}",
                            (x1, max(20, y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                st = track_state.get(track_id)
                if st is None:
                    st = {"enter_side": None, "in_zone": False, "last_count_time": 0.0}
                    track_state[track_id] = st

                curr_side = side_of_zone(cy, LINE_Y, ZONE_HALF_HEIGHT)

                if curr_side == "zone" and not st["in_zone"]:
                    st["enter_side"] = "top" if cy < LINE_Y else "bottom"
                    st["in_zone"] = True

                if st["in_zone"] and curr_side in ("top", "bottom"):
                    exit_side = curr_side
                    enter_side = st["enter_side"]

                    can_count = (now_ts - st["last_count_time"]) >= COOLDOWN_SEC
                    direction = None

                    if can_count and enter_side is not None and exit_side != enter_side:
                        if in_roi(cx, cy, ROI):
                            if enter_side == "top" and exit_side == "bottom":
                                direction = "IN"
                            elif enter_side == "bottom" and exit_side == "top":
                                direction = "OUT"

                            if direction == "IN":
                                in_total += 1
                                in_by_class[label] += 1
                            elif direction == "OUT":
                                out_total += 1
                                out_by_class[label] += 1

                            if direction:
                                st["last_count_time"] = now_ts

                                net = in_total - out_total
                                ts_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                roi_str = str(ROI)

                                # Snapshot (organized by date)
                                snap_path = save_snapshot(frame, today_str(), direction, label, track_id, conf)

                                # CSV backup: events
                                events_writer.writerow([ts_str, direction, label, track_id, f"{conf:.2f}",
                                                       roi_str, in_total, out_total, net, snap_path])
                                events_file.flush()

                                # CSV backup: hourly (write per event, simple)
                                d = today_str()
                                hr = hour_str()
                                hourly_writer.writerow([d, hr, label, direction, 1])
                                hourly_file.flush()

                                # SQLite: events
                                db_insert_event(ts_str, direction, label, track_id, float(conf), roi_str,
                                                in_total, out_total, net, snap_path)

                                # SQLite: hourly summary (upsert)
                                db_upsert_hourly(d, hr, label, direction, 1)

                    st["in_zone"] = False
                    st["enter_side"] = None

        # Draw ROI
        rx1, ry1, rx2, ry2 = ROI
        cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), (255, 0, 0), 2)
        cv2.putText(frame, "ROI Gate Area (Counting Only Here)",
                    (rx1, max(20, ry1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        # Draw zone + line
        cv2.rectangle(frame, (0, zone_top), (w, zone_bot), (0, 0, 255), 2)
        cv2.line(frame, (0, LINE_Y), (w, LINE_Y), (0, 0, 255), 1)

        # FPS
        now = time.time()
        fps = 1.0 / max(1e-6, (now - prev_time))
        prev_time = now

        draw_dashboard(frame, fps, LINE_Y, ROI, in_total, out_total, in_by_class, out_by_class)

        # Scale to window size
        try:
            x, y, win_w, win_h = cv2.getWindowImageRect(WIN)
            disp = cv2.resize(frame, (win_w, win_h)) if win_w > 0 and win_h > 0 else frame
        except:
            disp = frame

        cv2.imshow(WIN, disp)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), ord("Q")):
            break
        elif key in (ord("w"), ord("W")):
            LINE_Y = max(ZONE_HALF_HEIGHT + 5, LINE_Y - LINE_MOVE_STEP)
        elif key in (ord("s"), ord("S")):
            LINE_Y = min(h - (ZONE_HALF_HEIGHT + 5), LINE_Y + LINE_MOVE_STEP)

        elif key in (ord("i"), ord("I")):
            ROI[1] = max(0, ROI[1] - ROI_MOVE_STEP); ROI[3] = max(ROI[1] + 50, ROI[3] - ROI_MOVE_STEP)
        elif key in (ord("k"), ord("K")):
            ROI[1] = min(h - 50, ROI[1] + ROI_MOVE_STEP); ROI[3] = min(h, ROI[3] + ROI_MOVE_STEP)
        elif key in (ord("j"), ord("J")):
            ROI[0] = max(0, ROI[0] - ROI_MOVE_STEP); ROI[2] = max(ROI[0] + 50, ROI[2] - ROI_MOVE_STEP)
        elif key in (ord("l"), ord("L")):
            ROI[0] = min(w - 50, ROI[0] + ROI_MOVE_STEP); ROI[2] = min(w, ROI[2] + ROI_MOVE_STEP)

        elif key == ord(']'):
            ROI[0] = max(0, ROI[0] - ROI_RESIZE_STEP)
            ROI[1] = max(0, ROI[1] - ROI_RESIZE_STEP)
            ROI[2] = min(w, ROI[2] + ROI_RESIZE_STEP)
            ROI[3] = min(h, ROI[3] + ROI_RESIZE_STEP)
        elif key == ord('['):
            ROI[0] = min(ROI[2] - 50, ROI[0] + ROI_RESIZE_STEP)
            ROI[1] = min(ROI[3] - 50, ROI[1] + ROI_RESIZE_STEP)
            ROI[2] = max(ROI[0] + 50, ROI[2] - ROI_RESIZE_STEP)
            ROI[3] = max(ROI[1] + 50, ROI[3] - ROI_RESIZE_STEP)

finally:
    cap.release()
    cv2.destroyAllWindows()
    events_file.close()
    hourly_file.close()
    conn.close()