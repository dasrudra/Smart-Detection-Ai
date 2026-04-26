import os
import re
import cv2
import sqlite3
import queue
import threading
import torch
import easyocr
from ultralytics import YOLO
import numpy as np

CUSTOM_PLATE_MODEL_PATH = "models/bd_plate.pt"   # optional: if you later add a BD plate detector, it will be used automatically
PLATE_DEBUG_DIR = "logs/plate_debug"
os.makedirs(PLATE_DEBUG_DIR, exist_ok=True)

BANGLA_DIGITS = "০১২৩৪৫৬৭৮৯"
ASCII_DIGITS = "0123456789"
BANGLA_PLATE_CHARS = (
    "অআইঈউঊঋএঐওঔকখগঘঙচছজঝঞটঠডঢণতথদধনপফবভমযরলশষসহড়ঢ়য়"
    "ািীুূৃেৈোৌঁংঃ্"
)
PLATE_ALLOWLIST = (
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    + ASCII_DIGITS
    + BANGLA_DIGITS
    + BANGLA_PLATE_CHARS
    + " -"
)

BD_REGION_HINTS = [
    "ঢাকা", "চট্টগ্রাম", "চট্ট", "কুমিল্লা", "রাজশাহী", "খুলনা",
    "বরিশাল", "সিলেট", "রংপুর", "ময়মনসিংহ", "গাজীপুর", "নারায়ণগঞ্জ"
]

BANGLA_WORD_RE = re.compile(r"[অ-হড়ঢ়য়]")
DIGIT_RE = re.compile(r"[0-9০-৯]")

BANGLA_TO_ASCII = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")

BN_PLATE_RE = re.compile(r"[অ-হড়ঢ়য়]{1,4}[- ]?[০-৯]{1,2}[- ]?[০-৯]{3,4}")
EN_PLATE_RE = re.compile(r"[A-Za-z]{1,3}[- ]?\d{1,2}[- ]?\d{3,4}")

def normalize_plate_text(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\n", " ")
    text = text.replace("|", " ")
    text = text.replace("—", "-").replace("–", "-")
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s*-\s*", "-", text)

    cleaned = []
    for ch in text:
        if ch in PLATE_ALLOWLIST:
            cleaned.append(ch)

    text = "".join(cleaned).strip()
    text = re.sub(r"\s+", " ", text)
    return text


def score_bd_plate_text(text: str, conf: float) -> float:
    if not text:
        return -1.0

    score = float(conf)

    digits = len(DIGIT_RE.findall(text))
    has_bangla = bool(BANGLA_WORD_RE.search(text))

    if has_bangla:
        score += 1.2

    if 4 <= digits <= 6:
        score += 1.2
    elif digits >= 2:
        score += 0.5

    if "-" in text or " " in text:
        score += 0.2

    if 6 <= len(text) <= 22:
        score += 0.3

    if any(hint in text for hint in BD_REGION_HINTS):
        score += 0.8

    return score

def is_likely_bd_plate(text: str) -> bool:
    if not text:
        return False

    text = normalize_plate_text(text)
    ascii_text = text.translate(BANGLA_TO_ASCII)

    digit_count = len(re.findall(r"\d", ascii_text))
    has_bangla = bool(BANGLA_WORD_RE.search(text))
    has_english = bool(re.search(r"[A-Za-z]", text))

    if digit_count < 4 or digit_count > 8:
        return False

    if not (has_bangla or has_english):
        return False

    if "-" not in text and " " not in text:
        return False

    if BN_PLATE_RE.search(text):
        return True

    if EN_PLATE_RE.search(ascii_text):
        return True

    return False

def preprocess_variants(img):
    variants = [img]

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    gray = cv2.bilateralFilter(gray, 7, 50, 50)
    variants.append(gray)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_clahe = clahe.apply(gray)
    variants.append(gray_clahe)

    up = cv2.resize(gray_clahe, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)
    variants.append(up)

    sharp = cv2.filter2D(
        up,
        -1,
        np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    )
    variants.append(sharp)

    th = cv2.adaptiveThreshold(
        sharp, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 9
    )
    variants.append(th)

    return variants


def heuristic_plate_regions(vehicle_img):
    h, w = vehicle_img.shape[:2]
    regions = [vehicle_img]

    # lower half
    regions.append(vehicle_img[int(h * 0.40):h, :])

    # center-lower wide band
    regions.append(vehicle_img[int(h * 0.48):int(h * 0.92), int(w * 0.08):int(w * 0.92)])

    # center-lower tight band
    regions.append(vehicle_img[int(h * 0.58):int(h * 0.88), int(w * 0.18):int(w * 0.82)])

    # left lower corner
    regions.append(vehicle_img[int(h * 0.52):int(h * 0.92), 0:int(w * 0.38)])

    # right lower corner
    regions.append(vehicle_img[int(h * 0.52):int(h * 0.92), int(w * 0.62):w])

    # bottom center narrow strip
    regions.append(vehicle_img[int(h * 0.68):int(h * 0.92), int(w * 0.20):int(w * 0.80)])

    return [r for r in regions if r.size > 0]


def detector_plate_regions(plate_model, vehicle_img):
    if plate_model is None:
        return []

    try:
        res = plate_model(vehicle_img, imgsz=320, conf=0.25, verbose=False)[0]
    except Exception:
        return []

    if res.boxes is None:
        return []

    h, w = vehicle_img.shape[:2]
    regions = []

    for xyxy in res.boxes.xyxy.tolist():
        x1, y1, x2, y2 = map(int, xyxy)
        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(x1 + 1, min(x2, w))
        y2 = max(y1 + 1, min(y2, h))

        if (x2 - x1) < 20 or (y2 - y1) < 10:
            continue

        crop = vehicle_img[y1:y2, x1:x2]
        if crop.size > 0:
            regions.append(crop)

    return regions


def run_ocr(reader, img):
    best_text = ""
    best_score = -1.0

    for variant in preprocess_variants(img):
        try:
            results = reader.readtext(
                variant,
                detail=1,
                paragraph=False,
                allowlist=PLATE_ALLOWLIST
            )
        except TypeError:
            results = reader.readtext(
                variant,
                detail=1,
                paragraph=False
            )

        if not results:
            continue

        lines = []
        confs = []

        for item in results:
            if len(item) < 3:
                continue

            box, text, conf = item[0], item[1], float(item[2])
            text = normalize_plate_text(text)

            if len(text) < 2:
                continue

            y_top = min(p[1] for p in box)
            lines.append((y_top, text))
            confs.append(conf)

        if not lines:
            continue

        lines.sort(key=lambda x: x[0])
        merged = " ".join(t for _, t in lines).strip()
        avg_conf = sum(confs) / len(confs)

        s = score_bd_plate_text(merged, avg_conf)
        if s > best_score:
            best_text = merged
            best_score = s

    return best_text, best_score


def read_bd_plate_from_vehicle(reader, vehicle_img, plate_model=None):
    regions = detector_plate_regions(plate_model, vehicle_img)
    if not regions:
        regions = heuristic_plate_regions(vehicle_img)

    best_text = ""
    best_score = -1.0

    for region in regions:
        text, score = run_ocr(reader, region)
        if score > best_score:
            best_text, best_score = text, score

    if best_score < 1.20:
        return None, 0.0

    if not is_likely_bd_plate(best_text):
        return None, 0.0

    return best_text, float(best_score)


def update_event_plate(db_path, event_id, text=None, score=None, image_path=None):
    conn = sqlite3.connect(db_path, timeout=30)
    try:
        cur = conn.cursor()
        cur.execute(
            """
            UPDATE events
            SET plate_text = COALESCE(?, plate_text),
                plate_score = COALESCE(?, plate_score),
                plate_image_path = COALESCE(?, plate_image_path)
            WHERE id = ?
            """,
            (text, score, image_path, event_id)
        )
        conn.commit()
    finally:
        conn.close()

def save_plate_debug_image(event_id, vehicle_crop, text=None):
    try:
        safe_text = (text or "none").replace("/", "_").replace("\\", "_").replace(" ", "_")
        path = os.path.join(PLATE_DEBUG_DIR, f"event_{event_id}_{safe_text}.jpg")
        cv2.imwrite(path, vehicle_crop)
        return path
    except Exception:
        return None

class PlateWorker:
    def __init__(self, db_path, enabled=True):
        self.db_path = db_path
        self.enabled = enabled
        self.q = queue.Queue(maxsize=64)
        self.stop_token = object()
        self.thread = None
        self.reader = None
        self.plate_model = None

        if self.enabled:
            self.thread = threading.Thread(target=self._run, daemon=True)
            self.thread.start()

    def submit(self, event_id, vehicle_crops):
        if not self.enabled or vehicle_crops is None:
            return

        if not isinstance(vehicle_crops, list):
            vehicle_crops = [vehicle_crops]

        clean_crops = []
        for crop in vehicle_crops:
            if crop is not None and hasattr(crop, "size") and crop.size > 0:
                clean_crops.append(crop.copy())

        if not clean_crops:
            return

        try:
            self.q.put_nowait((event_id, clean_crops))
        except queue.Full:
            pass

    def _run(self):
        gpu = bool(torch.cuda.is_available())
        self.reader = easyocr.Reader(["bn", "en"], gpu=gpu)

        if os.path.exists(CUSTOM_PLATE_MODEL_PATH):
            try:
                self.plate_model = YOLO(CUSTOM_PLATE_MODEL_PATH)
                print(f"[INFO] Using custom BD plate detector: {CUSTOM_PLATE_MODEL_PATH}")
            except Exception as e:
                print(f"[WARN] Could not load custom plate detector: {e}")
                self.plate_model = None
        else:
            print("[INFO] Custom BD plate detector not found. Using OCR crop fallback.")

        while True:
            item = self.q.get()
            try:
                if item is self.stop_token:
                    return

                event_id, vehicle_crops = item

                best_text = None
                best_score = 0.0
                best_crop = None

                for crop in vehicle_crops:
                    text, score = read_bd_plate_from_vehicle(
                        self.reader,
                        crop,
                        self.plate_model
                    )
                    if score > best_score:
                        best_text = text
                        best_score = score
                        best_crop = crop

                if best_crop is None:
                    best_crop = vehicle_crops[0]

                image_path = save_plate_debug_image(
                    event_id,
                    best_crop,
                    best_text if best_text else "none"
                )

                if best_text:
                    print(f"[PLATE] event_id={event_id} text={best_text} score={best_score:.2f}")
                    update_event_plate(self.db_path, event_id, best_text, best_score, None)
                else:
                    update_event_plate(self.db_path, event_id, None, None, None)

            except Exception as e:
                print(f"[WARN] Plate OCR failed: {e}")
            finally:
                self.q.task_done()

    def stop(self):
        if not self.enabled or self.thread is None:
            return

        self.q.put(self.stop_token)
        self.thread.join(timeout=2.0)