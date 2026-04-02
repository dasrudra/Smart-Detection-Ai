from ultralytics import YOLO
import cv2
import time
import os
import csv
import sqlite3
from datetime import datetime
from collections import defaultdict, Counter
from urllib.parse import quote

# ----------------------------
# SETTINGS
# ----------------------------
MODEL_PATH = "models/yolov8n.pt"

# ----------------------------
# IP CAMERA SETTINGS
# ----------------------------
IP_CAM_IP = "10.203.90.207"
IP_CAM_USER = "tvl"
IP_CAM_PASSWORD = "235@YngTvl"

ENC_USER = quote(IP_CAM_USER, safe="")
ENC_PASS = quote(IP_CAM_PASSWORD, safe="")

# Try lower stream first for speed, then fallback
CAMERA_CANDIDATES = [
    f"rtsp://{ENC_USER}:{ENC_PASS}@{IP_CAM_IP}:554/profile3/media.smp",
    f"rtsp://{ENC_USER}:{ENC_PASS}@{IP_CAM_IP}:554/profile4/media.smp",
    f"rtsp://{ENC_USER}:{ENC_PASS}@{IP_CAM_IP}:554/profile2/media.smp",
]

# ----------------------------
# DETECTION SETTINGS
# ----------------------------
RESIZE_W = 960             # faster than 1280
MODEL_IMGSZ = 640          # faster inference
CONF_THRES = 0.14          # lower so far/small objects can still be detected
PROCESS_EVERY_N_FRAMES = 2 # detect every 2nd frame for speed

# COCO class IDs:
# 0=person, 2=car, 3=motorcycle, 5=bus, 7=truck
TARGET_CLASSES = [0, 2, 3, 5, 7]

DISPLAY_LABELS = {
    "person": "person",
    "car": "car",
    "motorcycle": "motorcycle",
    "bus": "bus",
    "truck": "truck",
    "micro": "micro",
}

MIN_BOX_W = 12
MIN_BOX_H = 16

# Ignore static false-person detections
PERSON_MIN_MOVE_PX = 10
PERSON_MIN_ASPECT_RATIO = 0.45
PERSON_NEAR_LINE_MARGIN = 40

# Overlap filtering for duplicate vehicle labels
DUPLICATE_IOU_THRES = 0.45

# Stable relabeling for van-like vehicles using current COCO model
ENABLE_MICRO_ALIAS = True
MICRO_MIN_AR = 1.45      # width/height lower bound
MICRO_MAX_AR = 3.40      # width/height upper bound
MICRO_MIN_H = 28         # ignore tiny boxes
MICRO_HISTORY_LEN = 8    # number of recent labels to smooth

# Label priority for overlapping same-object detections
CLASS_PRIORITY = {
    "person": 1,
    "motorcycle": 2,
    "car": 3,
    "truck": 4,
    "bus": 5,
    "micro": 6,
}

ZONE_HALF_HEIGHT = 18
COOLDOWN_SEC = 0.8
LINE_MOVE_STEP = 8
LINE_ROTATE_STEP = 10

# Detection drawing style
BOX_THICKNESS = 1
TEXT_SCALE = 0.45
TEXT_THICKNESS = 1
DOT_RADIUS = 3

ROI_MOVE_STEP = 12
ROI_RESIZE_STEP = 16

# Dashboard overlay
DASH_TOP_N = 4
PANEL_W = 300

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

def clip_roi(roi, w, h):
    x1, y1, x2, y2 = roi
    x1 = max(0, min(x1, w - 50))
    y1 = max(0, min(y1, h - 50))
    x2 = max(x1 + 50, min(x2, w))
    y2 = max(y1 + 50, min(y2, h))
    return [x1, y1, x2, y2]

def point_line_signed_distance(px, py, p1, p2):
    x1, y1 = p1
    x2, y2 = p2

    dx = x2 - x1
    dy = y2 - y1
    line_len = (dx * dx + dy * dy) ** 0.5
    if line_len < 1e-6:
        return 0.0

    # signed perpendicular distance
    return ((px - x1) * dy - (py - y1) * dx) / line_len

def side_of_zone_diagonal(px, py, p1, p2, half_h):
    d = point_line_signed_distance(px, py, p1, p2)

    if d < -half_h:
        return "neg"
    if d > half_h:
        return "pos"
    return "zone"

def move_line(line_p1, line_p2, dx, dy, w, h):
    x1 = min(max(line_p1[0] + dx, 0), w - 1)
    y1 = min(max(line_p1[1] + dy, 0), h - 1)
    x2 = min(max(line_p2[0] + dx, 0), w - 1)
    y2 = min(max(line_p2[1] + dy, 0), h - 1)
    return [x1, y1], [x2, y2]

def rotate_line(line_p1, line_p2, delta_y_left=0, delta_y_right=0, w=0, h=0):
    x1, y1 = line_p1
    x2, y2 = line_p2

    y1 = min(max(y1 + delta_y_left, 0), h - 1)
    y2 = min(max(y2 + delta_y_right, 0), h - 1)

    return [x1, y1], [x2, y2]

def get_parallel_lines(p1, p2, offset):
    x1, y1 = p1
    x2, y2 = p2

    dx = x2 - x1
    dy = y2 - y1
    length = (dx * dx + dy * dy) ** 0.5
    if length < 1e-6:
        return (p1, p2), (p1, p2)

    nx = -dy / length
    ny = dx / length

    p1a = (int(x1 + nx * offset), int(y1 + ny * offset))
    p2a = (int(x2 + nx * offset), int(y2 + ny * offset))
    p1b = (int(x1 - nx * offset), int(y1 - ny * offset))
    p2b = (int(x2 - nx * offset), int(y2 - ny * offset))

    return (p1a, p2a), (p1b, p2b)

def box_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0, x2 - x1)
    inter_h = max(0, y2 - y1)
    inter_area = inter_w * inter_h

    area1 = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
    area2 = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])

    union_area = area1 + area2 - inter_area
    if union_area <= 0:
        return 0.0

    return inter_area / union_area

def choose_better_detection(a, b):
    """
    Keep the better one between two overlapping detections.
    Higher class priority wins first, then higher confidence.
    """
    pa = CLASS_PRIORITY.get(a["label"], 0)
    pb = CLASS_PRIORITY.get(b["label"], 0)

    if pa != pb:
        return a if pa > pb else b

    return a if a["conf"] >= b["conf"] else b

def dedupe_overlapping_detections(detections, iou_thres=DUPLICATE_IOU_THRES):
    """
    Remove duplicate overlapping detections of the same physical object.
    Do not suppress person detections.
    """
    kept = []

    for det in detections:
        replaced = False
        for i, prev in enumerate(kept):
            # Never merge away a person
            if det["label"] == "person" or prev["label"] == "person":
                continue

            iou = box_iou(det["box"], prev["box"])
            if iou >= iou_thres:
                kept[i] = choose_better_detection(det, prev)
                replaced = True
                break

        if not replaced:
            kept.append(det)

    return kept

def relabel_vehicle_as_micro(label, bw, bh, conf):
    """
    Convert van-like car/bus/truck detections into 'micro' using shape heuristics.
    This does not change the model, only stabilizes display/output.
    """
    if not ENABLE_MICRO_ALIAS:
        return label

    if label not in ("car", "bus", "truck"):
        return label

    if bh < MICRO_MIN_H:
        return label

    ar = bw / max(1, bh)

    # Van / microbus-like proportions
    if MICRO_MIN_AR <= ar <= MICRO_MAX_AR:
        return "micro"

    return label


def smooth_track_label(track_id, label, track_label_history, history_len=MICRO_HISTORY_LEN):
    """
    Stabilize class name per track ID across frames.
    Keeps the most frequent recent label; on ties, prefers the most recent.
    """
    if track_id <= 0:
        return label

    hist = track_label_history.setdefault(track_id, [])
    hist.append(label)

    if len(hist) > history_len:
        hist.pop(0)

    counts = Counter(hist)
    max_count = max(counts.values())

    # If tie, prefer the most recent label among the winners
    for recent_label in reversed(hist):
        if counts[recent_label] == max_count:
            return recent_label

    return label

def moved_enough(track_id, cx, cy, track_motion, min_move_px):
    prev = track_motion.get(track_id)
    track_motion[track_id] = (cx, cy)

    if prev is None:
        return False

    px, py = prev
    dx = cx - px
    dy = cy - py
    return (dx * dx + dy * dy) ** 0.5 >= min_move_px

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

def draw_dashboard(frame, fps, line_info, roi, in_total, out_total, in_by_class, out_by_class):
    h, w = frame.shape[:2]
    panel_w = min(PANEL_W, w)
    x0, y0 = 10, 10
    x1, y1 = x0 + panel_w, y0 + 165

    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x1, y1), (20, 20, 20), -1)
    alpha = 0.40
    frame[:] = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

    net = in_total - out_total

    cv2.putText(frame, "Gate Counter", (x0 + 10, y0 + 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.putText(frame, f"IN:{in_total}  OUT:{out_total}  NET:{net}",
                (x0 + 10, y0 + 48),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.putText(frame, f"FPS:{fps:.1f}",
                (x0 + 10, y0 + 72),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 2)

    cv2.putText(frame, f"ROI:{roi}",
                (x0 + 10, y0 + 94),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

    all_classes = set(in_by_class.keys()) | set(out_by_class.keys())
    ranked = sorted(
        [(c, in_by_class[c], out_by_class[c]) for c in all_classes],
        key=lambda t: (t[1] + t[2]),
        reverse=True
    )[:DASH_TOP_N]

    yy = y0 + 120
    for cls, inc, outc in ranked:
        cv2.putText(frame, f"{cls}: {inc}/{outc}",
                    (x0 + 10, yy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        yy += 18

def open_ip_camera(candidates, warmup_sec=2.5):
    for url in candidates:
        print(f"[INFO] Trying camera stream: {url}")
        cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)

        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        if not cap.isOpened():
            print("[WARN] OpenCV could not open this stream.")
            cap.release()
            continue

        start = time.time()
        ok = False
        while time.time() - start < warmup_sec:
            ret, frame = cap.read()
            if ret and frame is not None and frame.size > 0:
                ok = True
                break

        if ok:
            print(f"[INFO] Connected successfully to: {url}")
            return cap, url

        print("[WARN] Stream opened but no valid frame received.")
        cap.release()

    return None, None

    #if not ok:
        #print("[ERROR] Stream opened but frame read failed.")
        #cap.release()
        #return None

    #print("[INFO] Connected successfully.")
    #return cap

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

cap, ACTIVE_CAMERA_SOURCE = open_ip_camera(CAMERA_CANDIDATES)
if cap is None:
    print("Error: Could not open any IP camera stream.")
    raise SystemExit

WIN = "Smart Office Gate Counter - Rudra"
cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)

LINE_P1 = None
LINE_P2 = None
ROI = None

track_state = {}          # track_id -> {"enter_side": None, "in_zone": False, "last_count_time": 0.0}
track_motion = {}         # track_id -> (prev_cx, prev_cy)
track_label_history = {}  # track_id -> recent labels for smoothing

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
frame_count = 0
last_results = None
last_roi_origin = (0, 0)

print("Controls:")
print("  Q = quit")
print("  W/S = move counting line up/down")
print("  A/D = move counting line left/right")
print("  R/F = tilt counting line diagonally")
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
        if not ret or frame is None:
            print("[WARN] Camera frame lost. Reconnecting...")
            cap.release()
            time.sleep(2)
            cap, ACTIVE_CAMERA_SOURCE = open_ip_camera(CAMERA_CANDIDATES)
            if cap is None:
                print("[ERROR] Reconnect failed. Exiting.")
                break
            continue

        # resize for speed
        h, w = frame.shape[:2]
        if RESIZE_W is not None and w != RESIZE_W:
            new_h = int(h * (RESIZE_W / w))
            frame = cv2.resize(frame, (RESIZE_W, new_h))
            h, w = frame.shape[:2]

        if LINE_P1 is None or LINE_P2 is None:
            y_mid = h // 2
            LINE_P1 = [0, y_mid - 30]
            LINE_P2 = [w - 1, y_mid + 30]

        if ROI is None:
            roi_w = int(w * 0.52)
            roi_h = int(h * 0.52)
            x1 = int(w * 0.20)
            y1 = int(h * 0.14)
            ROI = [x1, y1, x1 + roi_w, y1 + roi_h]

        ROI = clip_roi(ROI, w, h)



        # ----------------------------
        # DETECT ONLY INSIDE ROI
        # ----------------------------
        frame_count += 1
        rx1, ry1, rx2, ry2 = ROI
        roi_frame = frame[ry1:ry2, rx1:rx2]

        if roi_frame.size > 0 and frame_count % PROCESS_EVERY_N_FRAMES == 0:
            last_results = model.track(
                roi_frame,
                persist=True,
                verbose=False,
                imgsz=MODEL_IMGSZ,
                conf=CONF_THRES,
                classes=TARGET_CLASSES,
                tracker="bytetrack.yaml"
            )
            last_roi_origin = (rx1, ry1)

        results = last_results
        ox, oy = last_roi_origin

        if results and results[0].boxes is not None:
            boxes = results[0].boxes
            classes = boxes.cls.int().tolist()
            confs = boxes.conf.tolist()
            xyxy_list = boxes.xyxy.tolist()

            if boxes.id is not None:
                ids = boxes.id.int().tolist()
            else:
                # fallback temporary IDs so detections are still processed/drawn
                ids = [-(i + 1) for i in range(len(classes))]

            now_ts = time.time()

            # ----------------------------
            # Build detection list first
            # ----------------------------
            detections = []

            for xyxy, track_id, cls, conf in zip(xyxy_list, ids, classes, confs):
                if conf < CONF_THRES:
                    continue

                raw_label = model.names[int(cls)]
                label = DISPLAY_LABELS.get(raw_label, raw_label)

                # map ROI coordinates back to full frame
                x1, y1, x2, y2 = map(int, xyxy)
                x1 += ox
                x2 += ox
                y1 += oy
                y2 += oy

                bw = x2 - x1
                bh = y2 - y1
                if bw < MIN_BOX_W or bh < MIN_BOX_H:
                    continue

                label = relabel_vehicle_as_micro(label, bw, bh, conf)

                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)

                if not in_roi(cx, cy, ROI):
                    continue

                detections.append({
                    "track_id": track_id,
                    "cls": cls,
                    "label": label,
                    "conf": float(conf),
                    "box": [x1, y1, x2, y2],
                    "cx": cx,
                    "cy": cy,
                    "bw": bw,
                    "bh": bh,
                })

            # ----------------------------
            # Remove duplicate overlapping vehicle labels
            # ----------------------------
            detections = dedupe_overlapping_detections(detections)

            # ----------------------------
            # Process final detections
            # ----------------------------
            for det in detections:
                track_id = det["track_id"]
                label = det["label"]
                conf = det["conf"]
                x1, y1, x2, y2 = det["box"]
                cx = det["cx"]
                cy = det["cy"]
                bw = det["bw"]
                bh = det["bh"]

                if label != "person":
                    label = smooth_track_label(track_id, label, track_label_history)

                # Ignore static false persons like fire hose pipe
                if label == "person":
                    aspect_ratio = bh / max(1, bw)
                    dist_to_line = abs(point_line_signed_distance(cx, cy, LINE_P1, LINE_P2))
                    near_count_line = dist_to_line <= (ZONE_HALF_HEIGHT + 40)

                    # Keep close and medium persons much more easily
                    if bh < 14:
                        continue
                    if aspect_ratio < 0.35:
                        continue

                    # Apply motion filtering only to large persons that are far from the line
                    # and only if the detection already has a stable positive tracker ID
                    if track_id > 0 and (not near_count_line) and bh >= 40:
                        if not moved_enough(track_id, cx, cy, track_motion, PERSON_MIN_MOVE_PX):
                            continue

                # draw detections
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), BOX_THICKNESS)
                cv2.circle(frame, (cx, cy), DOT_RADIUS, (0, 255, 0), -1)
                cv2.putText(frame, f"{label} ID:{track_id} {conf:.2f}",
                            (x1, max(20, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, TEXT_SCALE, (0, 255, 0), TEXT_THICKNESS)

                st = track_state.get(track_id)
                if st is None:
                    st = {"enter_side": None, "in_zone": False, "last_count_time": 0.0}
                    track_state[track_id] = st

                curr_side = side_of_zone_diagonal(cx, cy, LINE_P1, LINE_P2, ZONE_HALF_HEIGHT)

                if curr_side == "zone" and not st["in_zone"]:
                    d = point_line_signed_distance(cx, cy, LINE_P1, LINE_P2)
                    st["enter_side"] = "neg" if d < 0 else "pos"
                    st["in_zone"] = True

                if st["in_zone"] and curr_side in ("neg", "pos"):
                    exit_side = curr_side
                    enter_side = st["enter_side"]

                    can_count = (now_ts - st["last_count_time"]) >= COOLDOWN_SEC
                    direction = None

                    if can_count and enter_side is not None and exit_side != enter_side:
                        # Your scene mapping:
                        # outside -> inside = IN
                        # inside -> outside = OUT
                        if enter_side == "neg" and exit_side == "pos":
                            direction = "OUT"
                        elif enter_side == "pos" and exit_side == "neg":
                            direction = "IN"

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

                            snap_path = save_snapshot(frame, today_str(), direction, label, track_id, conf)

                            events_writer.writerow([
                                ts_str, direction, label, track_id, f"{conf:.2f}",
                                roi_str, in_total, out_total, net, snap_path
                            ])
                            events_file.flush()

                            d = today_str()
                            hr = hour_str()
                            hourly_writer.writerow([d, hr, label, direction, 1])
                            hourly_file.flush()

                            db_insert_event(ts_str, direction, label, track_id, float(conf), roi_str,
                                            in_total, out_total, net, snap_path)

                            db_upsert_hourly(d, hr, label, direction, 1)

                    st["in_zone"] = False
                    st["enter_side"] = None



                    #st["in_zone"] = False
                    #st["enter_side"] = None

        # Draw ROI
        rx1, ry1, rx2, ry2 = ROI
        cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), (255, 0, 0), 2)
        cv2.putText(frame, "ROI Gate Area (Counting Only Here)",
                    (rx1, max(20, ry1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        # Draw zone + line
        (line_a1, line_a2), (line_b1, line_b2) = get_parallel_lines(LINE_P1, LINE_P2, ZONE_HALF_HEIGHT)

        cv2.line(frame, tuple(LINE_P1), tuple(LINE_P2), (0, 0, 255), 1)
        cv2.line(frame, line_a1, line_a2, (0, 0, 255), 2)
        cv2.line(frame, line_b1, line_b2, (0, 0, 255), 2)

        # FPS
        now = time.time()
        fps = 1.0 / max(1e-6, (now - prev_time))
        prev_time = now

        line_info = f"{LINE_P1}->{LINE_P2}"
        draw_dashboard(frame, fps, line_info, ROI, in_total, out_total, in_by_class, out_by_class)

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
            LINE_P1, LINE_P2 = move_line(LINE_P1, LINE_P2, 0, -LINE_MOVE_STEP, w, h)
        elif key in (ord("s"), ord("S")):
            LINE_P1, LINE_P2 = move_line(LINE_P1, LINE_P2, 0, LINE_MOVE_STEP, w, h)
        elif key in (ord("a"), ord("A")):
            LINE_P1, LINE_P2 = move_line(LINE_P1, LINE_P2, -LINE_MOVE_STEP, 0, w, h)
        elif key in (ord("d"), ord("D")):
            LINE_P1, LINE_P2 = move_line(LINE_P1, LINE_P2, LINE_MOVE_STEP, 0, w, h)

        # rotate / tilt the line diagonally
        elif key in (ord("r"), ord("R")):
            LINE_P1, LINE_P2 = rotate_line(LINE_P1, LINE_P2, -LINE_ROTATE_STEP, LINE_ROTATE_STEP, w, h)
        elif key in (ord("f"), ord("F")):
            LINE_P1, LINE_P2 = rotate_line(LINE_P1, LINE_P2, LINE_ROTATE_STEP, -LINE_ROTATE_STEP, w, h)

        elif key in (ord("i"), ord("I")):
            ROI[1] = max(0, ROI[1] - ROI_MOVE_STEP)
            ROI[3] = max(ROI[1] + 50, ROI[3] - ROI_MOVE_STEP)
        elif key in (ord("k"), ord("K")):
            ROI[1] = min(h - 50, ROI[1] + ROI_MOVE_STEP)
            ROI[3] = min(h, ROI[3] + ROI_MOVE_STEP)
        elif key in (ord("j"), ord("J")):
            ROI[0] = max(0, ROI[0] - ROI_MOVE_STEP)
            ROI[2] = max(ROI[0] + 50, ROI[2] - ROI_MOVE_STEP)
        elif key in (ord("l"), ord("L")):
            ROI[0] = min(w - 50, ROI[0] + ROI_MOVE_STEP)
            ROI[2] = min(w, ROI[2] + ROI_MOVE_STEP)

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

        ROI = clip_roi(ROI, w, h)

finally:
    cap.release()
    cv2.destroyAllWindows()
    events_file.close()
    hourly_file.close()
    conn.close()