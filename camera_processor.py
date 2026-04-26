import cv2
import time
import threading
from collections import defaultdict

from plate_utils import PlateWorker


class CameraProcessor:
    def __init__(self, cam_cfg, model, db_path, helpers):
        self.cam_cfg = cam_cfg
        self.model = model
        self.db_path = db_path
        self.h = helpers

        self.cap = None
        self.active_source = None
        self.win_name = f'Smart Office Gate Counter - {cam_cfg["camera_name"]}'

        self.line_p1 = None
        self.line_p2 = None
        self.roi = None

        self.track_state = {}
        self.track_motion = {}
        self.track_label_history = {}
        self.plate_recent = {}
        self.best_plate_crops = {}
        self.display_tracks = {}

        self.in_total = 0
        self.out_total = 0
        self.in_by_class = defaultdict(int)
        self.out_by_class = defaultdict(int)

        self.prev_time = time.time()
        self.frame_count = 0
        self.last_results = None
        self.last_roi_origin = (0, 0)

        self.frame_w = 960
        self.frame_h = 540
        self.control_freeze_until = 0.0

        self.capture_thread = None
        self.capture_stop = False
        self.latest_frame = None
        self.latest_frame_time = 0.0
        self.frame_lock = threading.Lock()
        self.read_fail_count = 0

        self.plate_worker = PlateWorker(
            db_path,
            enabled=cam_cfg.get("enable_plate_ocr", False)
        )

    def open_camera(self):
        self.cap, self.active_source = self.h["open_ip_camera"](self.cam_cfg["sources"])
        if self.cap is None:
            return False

        cv2.namedWindow(self.win_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.win_name, 1280, 720)

        self.start_capture_thread()
        return True

    def close(self):
        self.capture_stop = True

        if self.capture_thread is not None:
            self.capture_thread.join(timeout=2.0)

        if self.cap is not None:
            self.cap.release()

        self.plate_worker.stop()

        try:
            cv2.destroyWindow(self.win_name)
        except cv2.error:
            pass

    def _clip_line_to_frame(self, w, h):
        if self.line_p1 is None or self.line_p2 is None:
            return

        self.line_p1 = [
            max(0, min(int(self.line_p1[0]), w - 1)),
            max(0, min(int(self.line_p1[1]), h - 1)),
        ]
        self.line_p2 = [
            max(0, min(int(self.line_p2[0]), w - 1)),
            max(0, min(int(self.line_p2[1]), h - 1)),
        ]

    def ensure_geometry(self, w, h):
        if self.line_p1 is None or self.line_p2 is None:
            self.line_p1 = self.cam_cfg["line_p1"].copy()
            self.line_p2 = self.cam_cfg["line_p2"].copy()

        if self.roi is None:
            self.roi = self.cam_cfg["roi"].copy()

        self.roi = self.h["clip_roi"](self.roi, w, h)

        self.line_p1[0] = max(0, min(int(self.line_p1[0]), w - 1))
        self.line_p1[1] = max(0, min(int(self.line_p1[1]), h - 1))
        self.line_p2[0] = max(0, min(int(self.line_p2[0]), w - 1))
        self.line_p2[1] = max(0, min(int(self.line_p2[1]), h - 1))

    def reconnect(self):
        if self.cap is not None:
            self.cap.release()
        time.sleep(2)
        self.cap, self.active_source = self.h["open_ip_camera"](self.cam_cfg["sources"])
        return self.cap is not None

    def start_capture_thread(self):
        self.capture_stop = False
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()

    def _capture_loop(self):
        while not self.capture_stop:
            if self.cap is None:
                time.sleep(0.1)
                continue

            ret, frame = self.cap.read()

            if ret and frame is not None and frame.size > 0:
                with self.frame_lock:
                    self.latest_frame = frame
                    self.latest_frame_time = time.time()

                self.read_fail_count = 0
                continue

            self.read_fail_count += 1

            if self.read_fail_count >= 20:
                print(f"[WARN] Camera frame lost: {self.cam_cfg['camera_name']}. Reconnecting...")

                try:
                    if self.cap is not None:
                        self.cap.release()
                except Exception:
                    pass

                time.sleep(1.0)
                self.cap, self.active_source = self.h["open_ip_camera"](self.cam_cfg["sources"])
                self.read_fail_count = 0
            else:
                time.sleep(0.02)

    def read_latest_frame(self, max_age=2.0):
        with self.frame_lock:
            if self.latest_frame is None:
                return False, None

            if time.time() - self.latest_frame_time > max_age:
                return False, None

            return True, self.latest_frame.copy()