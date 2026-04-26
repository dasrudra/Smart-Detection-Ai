from ultralytics import YOLO
import cv2
import time
import os
import csv
import sqlite3
from datetime import datetime
from collections import Counter
from camera_config import CAMERAS
from camera_processor import CameraProcessor

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
    "rtsp_transport;tcp|"
    "fflags;nobuffer|"
    "flags;low_delay|"
    "max_delay;500000|"
    "stimeout;5000000"
)

# ----------------------------
# SETTINGS
# ----------------------------
MODEL_PATH = "models/yolov8s.pt"

RESIZE_W = 960
MODEL_IMGSZ = 416
CONF_THRES = 0.30
PROCESS_EVERY_N_FRAMES = 6

TARGET_CLASSES = [0, 2, 3, 5, 7]

DISPLAY_LABELS = {
    "person": "person",
    "car": "car",
    "motorcycle": "motorcycle",
    "bus": "bus",
    "truck": "truck",
    "micro": "micro",
}

MIN_BOX_W = 22
MIN_BOX_H = 28

PERSON_MIN_MOVE_PX = 10
DUPLICATE_IOU_THRES = 0.45

ENABLE_MICRO_ALIAS = False
MICRO_MIN_AR = 1.45
MICRO_MAX_AR = 3.40
MICRO_MIN_H = 28
MICRO_HISTORY_LEN = 8

CLASS_PRIORITY = {
    "person": 1,
    "motorcycle": 2,
    "car": 3,
    "truck": 4,
    "bus": 5,
    "micro": 6,
}

ZONE_HALF_HEIGHT = 18
COOLDOWN_SEC = 1.2
MIN_TRACK_FRAMES_FOR_COUNT = 5
TRACK_STALE_SEC = 4.0
CONTROL_FREEZE_SEC = 0.6
LINE_MOVE_STEP = 8
LINE_ROTATE_STEP = 10
LINE_RESIZE_STEP = 20
ZONE_RESIZE_STEP = 2

BOX_THICKNESS = 1
TEXT_SCALE = 0.40
TEXT_THICKNESS = 1
DOT_RADIUS = 3

DISPLAY_TRACK_TTL_SEC = 0.85
DISPLAY_SMOOTH_ALPHA = 0.55
DISPLAY_MAX_PREDICT_SEC = 0.15
DISPLAY_RESET_JUMP_PX = 160

ROI_MOVE_STEP = 12
ROI_RESIZE_STEP = 16

DASH_TOP_N = 3
PANEL_W = 220

PLATE_MIN_BOX_W = 110
PLATE_MIN_BOX_H = 70
PLATE_SUBMIT_SEC = 2.5

PLATE_NEAR_LINE_PX = 90
PLATE_MAX_TRACK_CROPS = 4

LOG_ROOT = "logs"
DAILY_DIR = os.path.join(LOG_ROOT, "daily")
SNAPS_ROOT = os.path.join(DAILY_DIR, "snaps")
EVENTS_ROOT = os.path.join(DAILY_DIR, "events")
HOURLY_ROOT = os.path.join(DAILY_DIR, "hourly")

PLATE_SNAPS_ROOT = os.path.join(DAILY_DIR, "plate_snaps")
os.makedirs(PLATE_SNAPS_ROOT, exist_ok=True)

os.makedirs(SNAPS_ROOT, exist_ok=True)
os.makedirs(EVENTS_ROOT, exist_ok=True)
os.makedirs(HOURLY_ROOT, exist_ok=True)

DB_DIR = "database"
DB_PATH = os.path.join(DB_DIR, "gate_events.db")
os.makedirs(DB_DIR, exist_ok=True)


def today_str():
    return datetime.now().strftime("%Y-%m-%d")


def hour_str():
    return datetime.now().strftime("%H")


def in_roi(cx, cy, roi):
    x1, y1, x2, y2 = roi
    return (x1 <= cx <= x2) and (y1 <= cy <= y2)


def clip_roi(roi, w, h):
    x1, y1, x2, y2 = roi

    x1 = max(0, min(int(x1), w - 50))
    y1 = max(0, min(int(y1), h - 50))
    x2 = max(x1 + 50, min(int(x2), w))
    y2 = max(y1 + 50, min(int(y2), h))

    return [x1, y1, x2, y2]


def move_roi(roi, dx, dy, w, h):
    x1, y1, x2, y2 = roi
    roi_w = x2 - x1
    roi_h = y2 - y1

    nx1 = min(max(0, x1 + dx), max(0, w - roi_w))
    ny1 = min(max(0, y1 + dy), max(0, h - roi_h))

    return [nx1, ny1, nx1 + roi_w, ny1 + roi_h]


def point_line_signed_distance(px, py, p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1
    line_len = (dx * dx + dy * dy) ** 0.5
    if line_len < 1e-6:
        return 0.0
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

def resize_line_length(line_p1, line_p2, delta, w, h):
    x1, y1 = line_p1
    x2, y2 = line_p2

    dx = x2 - x1
    dy = y2 - y1
    length = (dx * dx + dy * dy) ** 0.5
    if length < 1e-6:
        return line_p1, line_p2

    ux = dx / length
    uy = dy / length

    new_x1 = int(min(max(x1 - ux * delta / 2, 0), w - 1))
    new_y1 = int(min(max(y1 - uy * delta / 2, 0), h - 1))
    new_x2 = int(min(max(x2 + ux * delta / 2, 0), w - 1))
    new_y2 = int(min(max(y2 + uy * delta / 2, 0), h - 1))

    return [new_x1, new_y1], [new_x2, new_y2]


def change_zone_half_height(current_value, delta):
    return max(6, current_value + delta)


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

def plate_view_score(x1, y1, x2, y2, w, h, dist_to_line, conf):
    bw = x2 - x1
    bh = y2 - y1
    ar = bw / max(1.0, bh)

    area_score = bw * bh

    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0

    center_dx = abs(cx - (w / 2.0)) / max(1.0, w / 2.0)
    center_dy = abs(cy - (h / 2.0)) / max(1.0, h / 2.0)
    center_penalty = (center_dx + center_dy) * 0.30

    line_penalty = dist_to_line * 3.0

    # side-view penalty: very wide boxes are usually bad for plate OCR
    side_penalty = 0.0
    if ar > 2.2:
        side_penalty = (ar - 2.2) * 9000.0

    return area_score + (conf * 5000.0) - line_penalty - (center_penalty * 1000.0) - side_penalty


def build_plate_candidate_crops_from_original(orig_frame, x1, y1, x2, y2, w, h, scale_x, scale_y):
    bw = x2 - x1
    bh = y2 - y1
    candidates = []

    regions = [
        # center lower
        (0.18, 0.56, 0.82, 0.94),
        # left lower
        (0.00, 0.52, 0.38, 0.94),
        # right lower
        (0.62, 0.52, 1.00, 0.94),
        # narrow lower center
        (0.22, 0.68, 0.78, 0.92),
    ]

    for rx1, ry1, rx2, ry2 in regions:
        px1 = x1 + int(bw * rx1)
        py1 = y1 + int(bh * ry1)
        px2 = x1 + int(bw * rx2)
        py2 = y1 + int(bh * ry2)

        px1 = max(0, px1)
        py1 = max(0, py1)
        px2 = min(w, px2)
        py2 = min(h, py2)

        if px2 <= px1 or py2 <= py1:
            continue

        ox1 = max(0, int(px1 * scale_x))
        oy1 = max(0, int(py1 * scale_y))
        ox2 = min(orig_frame.shape[1], int(px2 * scale_x))
        oy2 = min(orig_frame.shape[0], int(py2 * scale_y))

        if (ox2 - ox1) < 70 or (oy2 - oy1) < 30:
            continue

        crop = orig_frame[oy1:oy2, ox1:ox2].copy()
        if crop.size > 0:
            candidates.append(crop)

    return candidates

def remember_best_plate_crops(processor, track_id, candidate_crops, score, max_keep=4):
    if not candidate_crops or track_id <= 0:
        return

    prev = processor.best_plate_crops.get(track_id)

    prev_score = -1.0
    if isinstance(prev, dict):
        prev_score = float(prev.get("score", -1.0))

    if prev is None or score > prev_score:
        processor.best_plate_crops[track_id] = {
            "score": float(score),
            "crops": [c.copy() for c in candidate_crops[:max_keep]]
        }

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
    pa = CLASS_PRIORITY.get(a["label"], 0)
    pb = CLASS_PRIORITY.get(b["label"], 0)
    if pa != pb:
        return a if pa > pb else b
    return a if a["conf"] >= b["conf"] else b


def dedupe_overlapping_detections(detections, iou_thres=DUPLICATE_IOU_THRES):
    kept = []
    for det in detections:
        replaced = False
        for i, prev in enumerate(kept):
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

def update_display_track(processor, track_id, label, conf, box, cx, cy, now_ts):
    if box is None:
        return

    x1, y1, x2, y2 = box

    if track_id > 0:
        key = f"id_{track_id}"
    else:
        # temporary display key for untracked boxes
        key = f"tmp_{label}_{int(cx // 80)}_{int(cy // 80)}"

    old = processor.display_tracks.get(key)

    # reset smoothing if the object jumps too much
    if old is not None:
        old_cx, old_cy = old["center"]
        jump = ((cx - old_cx) ** 2 + (cy - old_cy) ** 2) ** 0.5
        if jump > DISPLAY_RESET_JUMP_PX or old.get("label") != label:
            old = None

    if old is None:
        smooth_box = [float(x1), float(y1), float(x2), float(y2)]
        smooth_center = [float(cx), float(cy)]
        velocity = [0.0, 0.0]
    else:
        alpha = DISPLAY_SMOOTH_ALPHA
        prev_box = old["box"]
        prev_center = old["center"]
        prev_updated = old.get("updated", now_ts)

        smooth_box = [
            prev_box[0] * (1 - alpha) + x1 * alpha,
            prev_box[1] * (1 - alpha) + y1 * alpha,
            prev_box[2] * (1 - alpha) + x2 * alpha,
            prev_box[3] * (1 - alpha) + y2 * alpha,
        ]

        smooth_center = [
            prev_center[0] * (1 - alpha) + cx * alpha,
            prev_center[1] * (1 - alpha) + cy * alpha,
        ]

        dt = max(1e-6, now_ts - prev_updated)
        velocity = [
            (smooth_center[0] - prev_center[0]) / dt,
            (smooth_center[1] - prev_center[1]) / dt,
        ]

    processor.display_tracks[key] = {
        "track_id": track_id,
        "label": label,
        "conf": float(conf),
        "box": smooth_box,
        "center": smooth_center,
        "velocity": velocity,
        "updated": now_ts,
    }


def draw_display_tracks(frame, processor, now_ts):
    h, w = frame.shape[:2]
    stale_keys = []

    for key, tr in list(processor.display_tracks.items()):
        age = now_ts - tr.get("updated", 0)

        if age > DISPLAY_TRACK_TTL_SEC:
            stale_keys.append(key)
            continue

        x1, y1, x2, y2 = tr["box"]
        cx, cy = tr["center"]
        vx, vy = tr.get("velocity", [0.0, 0.0])

        predict_t = min(age, DISPLAY_MAX_PREDICT_SEC)

        dx = vx * predict_t
        dy = vy * predict_t

        x1 = int(round(x1 + dx))
        y1 = int(round(y1 + dy))
        x2 = int(round(x2 + dx))
        y2 = int(round(y2 + dy))
        cx = int(round(cx + dx))
        cy = int(round(cy + dy))

        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(0, min(x2, w - 1))
        y2 = max(0, min(y2, h - 1))
        cx = max(0, min(cx, w - 1))
        cy = max(0, min(cy, h - 1))

        if x2 <= x1 or y2 <= y1:
            continue

        label = tr["label"]
        track_id = tr["track_id"]
        conf = tr["conf"]

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), BOX_THICKNESS)
        cv2.circle(frame, (cx, cy), DOT_RADIUS, (0, 255, 0), -1)
        cv2.putText(
            frame,
            f"{label} ID:{track_id} {conf:.2f}",
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            TEXT_SCALE,
            (0, 255, 0),
            TEXT_THICKNESS
        )

    for key in stale_keys:
        processor.display_tracks.pop(key, None)

def relabel_vehicle_as_micro(label, bw, bh, conf):
    if not ENABLE_MICRO_ALIAS:
        return label
    if label not in ("car", "bus", "truck"):
        return label
    if bh < MICRO_MIN_H:
        return label

    ar = bw / max(1, bh)
    if MICRO_MIN_AR <= ar <= MICRO_MAX_AR:
        return "micro"
    return label


def smooth_track_label(track_id, label, track_label_history, history_len=MICRO_HISTORY_LEN):
    if track_id <= 0:
        return label

    hist = track_label_history.setdefault(track_id, [])
    hist.append(label)

    if len(hist) > history_len:
        hist.pop(0)

    counts = Counter(hist)
    max_count = max(counts.values())

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


def save_snapshot(frame, camera_id, day, direction, label, track_id, conf):
    day_dir = os.path.join(SNAPS_ROOT, day, camera_id)
    os.makedirs(day_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"{ts}_{direction}_{label}_id{track_id}_{conf:.2f}.jpg"
    path = os.path.join(day_dir, filename)
    cv2.imwrite(path, frame)
    return path

def save_plate_snapshot(vehicle_crop, camera_id, day, event_id, track_id, label):
    if vehicle_crop is None or vehicle_crop.size == 0:
        return None

    day_dir = os.path.join(PLATE_SNAPS_ROOT, day, camera_id)
    os.makedirs(day_dir, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"{ts}_{label}_id{track_id}_event{event_id}.jpg"
    path = os.path.join(day_dir, filename)
    cv2.imwrite(path, vehicle_crop)
    return path

def draw_dashboard(frame, fps, roi, in_total, out_total, in_by_class, out_by_class, cam_name):
    h, w = frame.shape[:2]
    panel_w = min(PANEL_W, w)
    x0, y0 = 10, 10
    x1, y1 = x0 + panel_w, y0 + 132

    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x1, y1), (20, 20, 20), -1)
    alpha = 0.40
    frame[:] = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

    net = in_total + out_total

    cv2.putText(frame, cam_name, (x0 + 8, y0 + 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1)
    cv2.putText(frame, "Gate Counter", (x0 + 8, y0 + 34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    cv2.putText(frame, f"IN:{in_total}  OUT:{out_total}  NET:{net}",
                (x0 + 8, y0 + 56),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 255, 255), 1)
    cv2.putText(frame, f"FPS:{fps:.1f}",
                (x0 + 8, y0 + 74),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1)
    cv2.putText(frame, f"ROI:{roi}",
                (x0 + 8, y0 + 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.34, (200, 200, 200), 1)

    all_classes = set(in_by_class.keys()) | set(out_by_class.keys())
    ranked = sorted(
        [(c, in_by_class[c], out_by_class[c]) for c in all_classes],
        key=lambda t: (t[1] + t[2]),
        reverse=True
    )[:DASH_TOP_N]

    yy = y0 + 106
    for cls, inc, outc in ranked:
        cv2.putText(frame, f"{cls}: {inc}/{outc}",
                    (x0 + 8, yy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1)
        yy += 14


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


conn = sqlite3.connect(DB_PATH, check_same_thread=False)
cur = conn.cursor()

conn.execute("PRAGMA journal_mode=WAL")
conn.execute("PRAGMA busy_timeout=30000")
conn.commit()

def ensure_column(conn, table_name, column_name, column_type):
    cols = [r[1] for r in conn.execute(f"PRAGMA table_info({table_name})").fetchall()]
    if column_name not in cols:
        conn.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_type}")
        conn.commit()


cur.execute("""
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    camera_id TEXT,
    camera_name TEXT,
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
    snapshot_path TEXT,
    plate_text TEXT,
    plate_score REAL,
    plate_image_path TEXT
)
""")

cur.execute("""
CREATE INDEX IF NOT EXISTS idx_events_date_hour
ON events(date, hour)
""")

cur.execute("""
CREATE TABLE IF NOT EXISTS hourly_summary (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    camera_id TEXT,
    camera_name TEXT,
    date TEXT NOT NULL,
    hour TEXT NOT NULL,
    label TEXT NOT NULL,
    direction TEXT NOT NULL,
    count INTEGER NOT NULL,
    UNIQUE(camera_id, date, hour, label, direction)
)
""")
conn.commit()

ensure_column(conn, "events", "camera_id", "TEXT")
ensure_column(conn, "events", "camera_name", "TEXT")
ensure_column(conn, "events", "plate_text", "TEXT")
ensure_column(conn, "events", "plate_score", "REAL")
ensure_column(conn, "hourly_summary", "camera_id", "TEXT")
ensure_column(conn, "hourly_summary", "camera_name", "TEXT")
ensure_column(conn, "events", "plate_image_path", "TEXT")
ensure_column(conn, "hourly_summary", "plate_image_saved_count", "INTEGER DEFAULT 0")
ensure_column(conn, "hourly_summary", "plate_ocr_success_count", "INTEGER DEFAULT 0")


def db_insert_event(camera_id, camera_name, ts_str, direction, label, track_id, conf, roi, in_total, out_total, net_total, snapshot_path):
    day = ts_str.split(" ")[0]
    hr = ts_str.split(" ")[1][:2]

    cur.execute("""
    INSERT INTO events(
        camera_id, camera_name, ts, date, hour, direction, label,
        track_id, conf, roi, in_total, out_total, net_total, snapshot_path
    )
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        camera_id, camera_name, ts_str, day, hr, direction, label,
        track_id, conf, roi, in_total, out_total, net_total, snapshot_path
    ))
    conn.commit()
    return cur.lastrowid

def db_update_plate_image(event_id, image_path):
    cur.execute(
        "UPDATE events SET plate_image_path=? WHERE id=?",
        (image_path, event_id)
    )
    conn.commit()

def db_upsert_hourly(camera_id, camera_name, date, hour, label, direction,
                     inc=1, plate_image_saved_inc=0, plate_ocr_success_inc=0):
    cur.execute("""
        UPDATE hourly_summary
        SET count = COALESCE(count, 0) + ?,
            camera_name = ?,
            plate_image_saved_count = COALESCE(plate_image_saved_count, 0) + ?,
            plate_ocr_success_count = COALESCE(plate_ocr_success_count, 0) + ?
        WHERE camera_id = ?
          AND date = ?
          AND hour = ?
          AND label = ?
          AND direction = ?
    """, (
        inc,
        camera_name,
        plate_image_saved_inc,
        plate_ocr_success_inc,
        camera_id,
        date,
        hour,
        label,
        direction
    ))

    if cur.rowcount == 0:
        cur.execute("""
            INSERT INTO hourly_summary(
                camera_id, camera_name, date, hour, label, direction, count,
                plate_image_saved_count, plate_ocr_success_count
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            camera_id,
            camera_name,
            date,
            hour,
            label,
            direction,
            inc,
            plate_image_saved_inc,
            plate_ocr_success_inc
        ))

    conn.commit()


helpers = {
    "open_ip_camera": open_ip_camera,
    "clip_roi": clip_roi,
}

processors = []
for cam_cfg in CAMERAS:
    cam_model = YOLO(MODEL_PATH)   # separate model/tracker per camera
    processor = CameraProcessor(cam_cfg, cam_model, DB_PATH, helpers)
    if processor.open_camera():
        processors.append(processor)
    else:
        print(f"[WARN] Could not open camera: {cam_cfg['camera_name']}")

active_processor_index = 0
window_to_index = {}


def make_mouse_callback(idx):
    def _mouse(event, x, y, flags, param):
        global active_processor_index
        if event == cv2.EVENT_LBUTTONDOWN:
            active_processor_index = idx
            try:
                cam_name = processors[idx].cam_cfg["camera_name"]
                print(f"[INFO] Active camera selected: {cam_name}")
            except Exception:
                pass
    return _mouse

if not processors:
    print("Error: Could not open any camera streams.")
    raise SystemExit

for idx, processor in enumerate(processors):
    window_to_index[processor.win_name] = idx
    cv2.setMouseCallback(processor.win_name, make_mouse_callback(idx))

current_day = today_str()
events_path = get_daily_events_path(current_day)
hourly_path = get_daily_hourly_path(current_day)

events_file, events_writer = ensure_csv_header(
    events_path,
    [
        "event_id",
        "camera_id",
        "camera_name",
        "timestamp",
        "direction",
        "label",
        "track_id",
        "conf",
        "roi",
        "in_total",
        "out_total",
        "net_total",
        "snapshot_path",
        "plate_text",
        "plate_score",
        "plate_image_path"
    ]
)

hourly_file, hourly_writer = ensure_csv_header(
    hourly_path,
    [
        "camera_id",
        "camera_name",
        "date",
        "hour",
        "label",
        "direction",
        "count",
        "plate_image_saved_count",
        "plate_ocr_success_count"
    ]
)

print("Controls:")
print("  Q = quit all")
print("  Click a camera window to make it active")
print("  P = print active camera ROI/line config")
print("  W/S = move counting line up/down")
print("  A/D = move counting line left/right")
print("  R/F = tilt counting line diagonally")
print("  I/K/J/L = move ROI up/down/left/right")
print("  [ / ] = resize ROI smaller/bigger")
print("  Z/X = make line shorter/longer")
print("  C/V = make red zone thinner/thicker")

#active_processor_index = 0

try:
    while True:
        day_now = today_str()
        if day_now != current_day:
            events_file.close()
            hourly_file.close()

            current_day = day_now
            events_path = get_daily_events_path(current_day)
            hourly_path = get_daily_hourly_path(current_day)

            events_file, events_writer = ensure_csv_header(
                events_path,
                [
                    "event_id",
                    "camera_id",
                    "camera_name",
                    "timestamp",
                    "direction",
                    "label",
                    "track_id",
                    "conf",
                    "roi",
                    "in_total",
                    "out_total",
                    "net_total",
                    "snapshot_path",
                    "plate_text",
                    "plate_score",
                    "plate_image_path"
                ]
            )

            hourly_file, hourly_writer = ensure_csv_header(
                hourly_path,
                [
                    "camera_id",
                    "camera_name",
                    "date",
                    "hour",
                    "label",
                    "direction",
                    "count",
                    "plate_image_saved_count",
                    "plate_ocr_success_count"
                ]
            )

        for processor in processors:
            cam_cfg = processor.cam_cfg
            WIN = processor.win_name

            ret, frame = processor.read_latest_frame(max_age=2.0)
            if not ret or frame is None:
                print(f"[WARN] No fresh frame available yet: {cam_cfg['camera_name']}")
                continue

            orig_frame = frame.copy()
            orig_h, orig_w = frame.shape[:2]

            resize_w = cam_cfg.get("resize_w", RESIZE_W)
            model_imgsz = cam_cfg.get("model_imgsz", MODEL_IMGSZ)
            conf_thres = cam_cfg.get("conf_thres", CONF_THRES)
            process_n = cam_cfg.get("process_every_n_frames", PROCESS_EVERY_N_FRAMES)
            zone_half_h = cam_cfg.get("zone_half_height", ZONE_HALF_HEIGHT)
            min_track_frames = cam_cfg.get("min_track_frames_for_count", MIN_TRACK_FRAMES_FOR_COUNT)

            h, w = frame.shape[:2]
            if resize_w is not None and w != resize_w:
                new_h = int(h * (resize_w / w))
                frame = cv2.resize(frame, (resize_w, new_h))
                h, w = frame.shape[:2]

            scale_x = orig_w / float(w)
            scale_y = orig_h / float(h)

            processor.frame_w = w
            processor.frame_h = h

            processor.ensure_geometry(w, h)
            LINE_P1 = processor.line_p1
            LINE_P2 = processor.line_p2
            ROI = processor.roi

            track_state = processor.track_state
            track_motion = processor.track_motion
            track_label_history = processor.track_label_history
            in_total = processor.in_total
            out_total = processor.out_total
            in_by_class = processor.in_by_class
            out_by_class = processor.out_by_class
            prev_time = processor.prev_time
            frame_count = processor.frame_count
            last_results = processor.last_results
            last_roi_origin = processor.last_roi_origin

            frame_count += 1
            processor.frame_count = frame_count

            rx1, ry1, rx2, ry2 = ROI
            roi_frame = frame[ry1:ry2, rx1:rx2]

            fresh_results = False

            if roi_frame.size > 0 and frame_count % process_n == 0:
                last_results = processor.model.track(
                    roi_frame,
                    persist=True,
                    verbose=False,
                    imgsz=model_imgsz,
                    conf=conf_thres,
                    classes=TARGET_CLASSES,
                    tracker="bytetrack.yaml"
                )
                last_roi_origin = (rx1, ry1)
                processor.last_results = last_results
                processor.last_roi_origin = last_roi_origin
                fresh_results = True

            results = last_results
            ox, oy = last_roi_origin

            if fresh_results and results and results[0].boxes is not None:
                boxes = results[0].boxes
                classes = boxes.cls.int().tolist()
                confs = boxes.conf.tolist()
                xyxy_list = boxes.xyxy.tolist()

                if boxes.id is not None:
                    ids = boxes.id.int().tolist()
                else:
                    ids = [-(i + 1) for i in range(len(classes))]

                now_ts = time.time()
                detections = []

                for xyxy, track_id, cls, conf in zip(xyxy_list, ids, classes, confs):
                    if conf < conf_thres:
                        continue

                    raw_label = processor.model.names[int(cls)]
                    label = DISPLAY_LABELS.get(raw_label, raw_label)

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

                detections = dedupe_overlapping_detections(detections)

                for det in detections:
                    track_id = det["track_id"]
                    label = det["label"]
                    conf = det["conf"]
                    x1, y1, x2, y2 = det["box"]
                    cx = det["cx"]
                    cy = det["cy"]
                    bw = det["bw"]
                    bh = det["bh"]

                    signed_dist = point_line_signed_distance(cx, cy, LINE_P1, LINE_P2)
                    dist_to_line = abs(signed_dist)

                    plate_min_box_w = cam_cfg.get("plate_min_box_w", PLATE_MIN_BOX_W)
                    plate_min_box_h = cam_cfg.get("plate_min_box_h", PLATE_MIN_BOX_H)
                    plate_near_line_px = cam_cfg.get("plate_near_line_px", PLATE_NEAR_LINE_PX)
                    plate_max_track_crops = cam_cfg.get("plate_max_track_crops", PLATE_MAX_TRACK_CROPS)

                    if (
                            track_id > 0
                            and cam_cfg.get("enable_plate", False)
                            and label in {"car", "micro", "truck", "bus"}
                            and bw >= plate_min_box_w
                            and bh >= plate_min_box_h
                            and conf >= 0.45
                            and dist_to_line <= (zone_half_h + plate_near_line_px)
                    ):
                        candidate_crops = build_plate_candidate_crops_from_original(
                            orig_frame, x1, y1, x2, y2, w, h, scale_x, scale_y
                        )

                        if candidate_crops:
                            score = plate_view_score(x1, y1, x2, y2, w, h, dist_to_line, conf)
                            remember_best_plate_crops(
                                processor,
                                track_id,
                                candidate_crops,
                                score,
                                plate_max_track_crops
                            )

                    if label != "person":
                        label = smooth_track_label(track_id, label, track_label_history)

                    if label == "person":
                        aspect_ratio = bh / max(1, bw)
                        near_count_line = dist_to_line <= (zone_half_h + 40)

                        if bh < 22:
                            continue
                        if aspect_ratio < 0.45:
                            continue

                        if track_id > 0 and (not near_count_line) and bh >= 40:
                            if not moved_enough(track_id, cx, cy, track_motion, PERSON_MIN_MOVE_PX):
                                continue

                    update_display_track(
                        processor,
                        track_id,
                        label,
                        conf,
                        [x1, y1, x2, y2],
                        cx,
                        cy,
                        now_ts
                    )

                    # Draw unstable detections, but never count them
                    if track_id <= 0:
                        continue

                    st = track_state.get(track_id)
                    if st is None:
                        st = {
                            "prev_side": None,
                            "prev_dist": None,
                            "in_zone": False,
                            "last_count_time": 0.0,
                            "seen_frames": 0,
                            "last_seen_time": now_ts,
                            "last_center": None,
                        }
                        track_state[track_id] = st

                    st["seen_frames"] += 1
                    st["last_seen_time"] = now_ts

                    curr_dist = signed_dist
                    curr_side = side_of_zone_diagonal(cx, cy, LINE_P1, LINE_P2, zone_half_h)

                    prev_side = st["prev_side"]
                    prev_dist = st.get("prev_dist")

                    if curr_side == "zone":
                        st["in_zone"] = True
                        st["prev_dist"] = curr_dist
                        st["last_center"] = (cx, cy)
                        can_count = False

                    elif curr_side in ("neg", "pos"):
                        direct_cross = (
                                prev_dist is not None
                                and (
                                        (prev_dist < -zone_half_h and curr_dist > zone_half_h)
                                        or
                                        (prev_dist > zone_half_h and curr_dist < -zone_half_h)
                                )
                        )

                        can_count = (
                                time.time() >= processor.control_freeze_until
                                and prev_side in ("neg", "pos")
                                and prev_side != curr_side
                                and (st["in_zone"] or direct_cross)
                                and st["seen_frames"] >= min_track_frames
                                and (now_ts - st["last_count_time"]) >= COOLDOWN_SEC
                        )

                        direction = None

                        if can_count:
                            if prev_side == "neg" and curr_side == "pos":
                                direction = cam_cfg["neg_to_pos"]
                            elif prev_side == "pos" and curr_side == "neg":
                                direction = cam_cfg["pos_to_neg"]

                            if direction == "IN":
                                in_total += 1
                                in_by_class[label] += 1
                            elif direction == "OUT":
                                out_total += 1
                                out_by_class[label] += 1

                            processor.in_total = in_total
                            processor.out_total = out_total

                            if direction:
                                st["last_count_time"] = now_ts

                                print(
                                    f'[COUNT] {cam_cfg["camera_name"]} '
                                    f'{label} ID:{track_id} {prev_side}->{curr_side} => {direction}'
                                )

                                net = in_total + out_total
                                ts_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                d = today_str()
                                hr = hour_str()
                                roi_str = str(ROI)

                                snap_path = save_snapshot(
                                    frame, cam_cfg["camera_id"], d, direction, label, track_id, conf
                                )

                                # 1) Insert DB row first -> get event_id
                                event_id = db_insert_event(
                                    cam_cfg["camera_id"],
                                    cam_cfg["camera_name"],
                                    ts_str,
                                    direction,
                                    label,
                                    track_id,
                                    float(conf),
                                    roi_str,
                                    in_total,
                                    out_total,
                                    net,
                                    snap_path
                                )

                                # 2) Default plate fields
                                plate_image_path = None
                                plate_image_saved_count = 0

                                # 3) Get best saved crops for this track
                                track_entry = processor.best_plate_crops.get(track_id)
                                if isinstance(track_entry, dict):
                                    track_crops = track_entry.get("crops")
                                else:
                                    track_crops = track_entry

                                # 4) Fallback crop build if not cached yet
                                plate_min_box_w = cam_cfg.get("plate_min_box_w", PLATE_MIN_BOX_W)
                                plate_min_box_h = cam_cfg.get("plate_min_box_h", PLATE_MIN_BOX_H)
                                plate_submit_sec = cam_cfg.get("plate_submit_sec", PLATE_SUBMIT_SEC)

                                if (
                                        not track_crops
                                        and cam_cfg.get("enable_plate", False)
                                        and label in {"car", "micro", "truck", "bus"}
                                        and bw >= plate_min_box_w
                                        and bh >= plate_min_box_h
                                        and conf >= 0.45
                                ):
                                    track_crops = build_plate_candidate_crops_from_original(
                                        orig_frame, x1, y1, x2, y2, w, h, scale_x, scale_y
                                    )

                                # 5) Save official plate snapshot + queue OCR
                                if cam_cfg.get("enable_plate", False) and track_crops:
                                    plate_image_path = save_plate_snapshot(
                                        track_crops[0],
                                        cam_cfg["camera_id"],
                                        d,
                                        event_id,
                                        track_id,
                                        label
                                    )

                                    if plate_image_path:
                                        db_update_plate_image(event_id, plate_image_path)
                                        plate_image_saved_count = 1

                                    if cam_cfg.get("enable_plate_ocr", False):
                                        last_plate_ts = processor.plate_recent.get(track_id, 0.0)
                                        if (now_ts - last_plate_ts) >= plate_submit_sec:
                                            processor.plate_recent[track_id] = now_ts
                                            processor.plate_worker.submit(event_id, track_crops)

                                # 6) Write CSV only once
                                events_writer.writerow([
                                    event_id,
                                    cam_cfg["camera_id"],
                                    cam_cfg["camera_name"],
                                    ts_str,
                                    direction,
                                    label,
                                    track_id,
                                    f"{conf:.2f}",
                                    roi_str,
                                    in_total,
                                    out_total,
                                    net,
                                    snap_path,
                                    "",  # plate_text will be filled later in DB by OCR worker
                                    "",  # plate_score will be filled later in DB by OCR worker
                                    plate_image_path or ""
                                ])
                                events_file.flush()

                                hourly_writer.writerow([
                                    cam_cfg["camera_id"],
                                    cam_cfg["camera_name"],
                                    d,
                                    hr,
                                    label,
                                    direction,
                                    1,
                                    plate_image_saved_count,
                                    0
                                ])
                                hourly_file.flush()

                                db_upsert_hourly(
                                    cam_cfg["camera_id"],
                                    cam_cfg["camera_name"],
                                    d,
                                    hr,
                                    label,
                                    direction,
                                    1,
                                    plate_image_saved_count,
                                    0
                                )

                                # prevent immediate re-count
                                st["in_zone"] = False

                        st["prev_side"] = curr_side
                        st["prev_dist"] = curr_dist
                        st["last_center"] = (cx, cy)

            draw_display_tracks(frame, processor, time.time())
            rx1, ry1, rx2, ry2 = ROI
            cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), (255, 0, 0), 2)
            cv2.putText(frame, "ROI Gate Area (Counting Only Here)",
                        (rx1, max(20, ry1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

            (line_a1, line_a2), (line_b1, line_b2) = get_parallel_lines(LINE_P1, LINE_P2, zone_half_h)
            cv2.line(frame, tuple(LINE_P1), tuple(LINE_P2), (0, 0, 255), 1)
            cv2.line(frame, line_a1, line_a2, (0, 0, 255), 2)
            cv2.line(frame, line_b1, line_b2, (0, 0, 255), 2)

            now = time.time()
            fps = 1.0 / max(1e-6, (now - prev_time))
            prev_time = now
            processor.prev_time = prev_time

            draw_dashboard(
                frame,
                fps,
                ROI,
                in_total,
                out_total,
                in_by_class,
                out_by_class,
                cam_cfg["camera_name"]
            )

            disp = frame.copy()

            if processors[active_processor_index] is processor:
                cv2.putText(
                    disp,
                    "ACTIVE",
                    (disp.shape[1] - 100, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 255),
                    2
                )

            cv2.imshow(WIN, disp)

        for processor in processors:
            stale_ids = [
                tid for tid, st in processor.track_state.items()
                if (time.time() - st.get("last_seen_time", 0)) > TRACK_STALE_SEC
            ]
            for tid in stale_ids:
                processor.track_state.pop(tid, None)
                processor.track_motion.pop(tid, None)
                processor.track_label_history.pop(tid, None)
                processor.plate_recent.pop(tid, None)
                processor.best_plate_crops.pop(tid, None)
                processor.display_tracks.pop(f"id_{tid}", None)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), ord("Q")):
            break

        if not processors:
            continue

        active = processors[active_processor_index]

        w = getattr(active, "frame_w", 960)
        h = getattr(active, "frame_h", 540)

        if active.line_p1 is None or active.line_p2 is None or active.roi is None:
            active.ensure_geometry(w, h)

        LINE_P1 = active.line_p1.copy()
        LINE_P2 = active.line_p2.copy()
        ROI = active.roi.copy()

        changed = False

        if key in (ord("p"), ord("P")):
            cam = active.cam_cfg
            print("\n# Paste these values into camera_config.py")
            print(f'# {cam["camera_name"]}')
            print(f'"roi": {active.roi},')
            print(f'"line_p1": {active.line_p1},')
            print(f'"line_p2": {active.line_p2},')
            print(f'"neg_to_pos": "{cam["neg_to_pos"]}",')
            print(f'"pos_to_neg": "{cam["pos_to_neg"]}",')
            print(f'"enable_plate": {cam.get("enable_plate", False)},')
            print(f'"resize_w": {cam.get("resize_w", RESIZE_W)},')
            print(f'"model_imgsz": {cam.get("model_imgsz", MODEL_IMGSZ)},')
            print(f'"conf_thres": {cam.get("conf_thres", CONF_THRES)},')
            print(f'"process_every_n_frames": {cam.get("process_every_n_frames", PROCESS_EVERY_N_FRAMES)},')
            print(f'"zone_half_height": {cam.get("zone_half_height", ZONE_HALF_HEIGHT)},')
            print()

        elif key in (ord("w"), ord("W")):
            LINE_P1, LINE_P2 = move_line(LINE_P1, LINE_P2, 0, -LINE_MOVE_STEP, w, h)
            changed = True
        elif key in (ord("s"), ord("S")):
            LINE_P1, LINE_P2 = move_line(LINE_P1, LINE_P2, 0, LINE_MOVE_STEP, w, h)
            changed = True
        elif key in (ord("a"), ord("A")):
            LINE_P1, LINE_P2 = move_line(LINE_P1, LINE_P2, -LINE_MOVE_STEP, 0, w, h)
            changed = True
        elif key in (ord("d"), ord("D")):
            LINE_P1, LINE_P2 = move_line(LINE_P1, LINE_P2, LINE_MOVE_STEP, 0, w, h)
            changed = True
        elif key in (ord("r"), ord("R")):
            LINE_P1, LINE_P2 = rotate_line(LINE_P1, LINE_P2, -LINE_ROTATE_STEP, LINE_ROTATE_STEP, w, h)
            changed = True
        elif key in (ord("f"), ord("F")):
            LINE_P1, LINE_P2 = rotate_line(LINE_P1, LINE_P2, LINE_ROTATE_STEP, -LINE_ROTATE_STEP, w, h)
            changed = True
        elif key in (ord("z"), ord("Z")):
            LINE_P1, LINE_P2 = resize_line_length(LINE_P1, LINE_P2, -LINE_RESIZE_STEP, w, h)
            changed = True
        elif key in (ord("x"), ord("X")):
            LINE_P1, LINE_P2 = resize_line_length(LINE_P1, LINE_P2, LINE_RESIZE_STEP, w, h)
            changed = True
        elif key in (ord("c"), ord("C")):
            active.cam_cfg["zone_half_height"] = change_zone_half_height(
                active.cam_cfg.get("zone_half_height", ZONE_HALF_HEIGHT),
                -ZONE_RESIZE_STEP
            )
            changed = True
        elif key in (ord("v"), ord("V")):
            active.cam_cfg["zone_half_height"] = change_zone_half_height(
                active.cam_cfg.get("zone_half_height", ZONE_HALF_HEIGHT),
                ZONE_RESIZE_STEP
            )
            changed = True

        elif key in (ord("i"), ord("I")):
            ROI = move_roi(ROI, 0, -ROI_MOVE_STEP, w, h)
            changed = True
        elif key in (ord("k"), ord("K")):
            ROI = move_roi(ROI, 0, ROI_MOVE_STEP, w, h)
            changed = True
        elif key in (ord("j"), ord("J")):
            ROI = move_roi(ROI, -ROI_MOVE_STEP, 0, w, h)
            changed = True
        elif key in (ord("l"), ord("L")):
            ROI = move_roi(ROI, ROI_MOVE_STEP, 0, w, h)
            changed = True

        elif key == ord("["):
            ROI[0] = min(ROI[2] - 50, ROI[0] + ROI_RESIZE_STEP)
            ROI[1] = min(ROI[3] - 50, ROI[1] + ROI_RESIZE_STEP)
            ROI[2] = max(ROI[0] + 50, ROI[2] - ROI_RESIZE_STEP)
            ROI[3] = max(ROI[1] + 50, ROI[3] - ROI_RESIZE_STEP)
            changed = True

        elif key == ord("]"):
            ROI[0] = max(0, ROI[0] - ROI_RESIZE_STEP)
            ROI[1] = max(0, ROI[1] - ROI_RESIZE_STEP)
            ROI[2] = min(w, ROI[2] + ROI_RESIZE_STEP)
            ROI[3] = min(h, ROI[3] + ROI_RESIZE_STEP)
            changed = True

        if changed:
            active.line_p1 = LINE_P1
            active.line_p2 = LINE_P2
            active.roi = clip_roi(ROI, w, h)
            active.control_freeze_until = time.time() + CONTROL_FREEZE_SEC

            print(
                f'[CONTROL] {active.cam_cfg["camera_name"]} '
                f'LINE:{active.line_p1}->{active.line_p2} ROI:{active.roi}'
            )

finally:
    for processor in processors:
        processor.close()
    events_file.close()
    hourly_file.close()
    conn.close()