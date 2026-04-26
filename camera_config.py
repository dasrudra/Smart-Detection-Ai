from urllib.parse import quote

CAMERA_USER = "tvl"
CAMERA_PASSWORD = "235@YngTvl"

ENC_USER = quote(CAMERA_USER, safe="")
ENC_PASS = quote(CAMERA_PASSWORD, safe="")

CAMERAS = [
    {
        "camera_id": "cam_71",
        "camera_name": "KSI Main Gate",
        "ip": "10.203.71.111",
        "sources": [
            f"rtsp://{ENC_USER}:{ENC_PASS}@10.203.71.111:554/profile2/media.smp",
        ],
        "roi": [0, 0, 768, 432],
        "line_p1": [0, 431],
        "line_p2": [767, 178],
        "neg_to_pos": "IN",
        "pos_to_neg": "OUT",

        "enable_plate": True,
        "enable_plate_ocr": False,   # keep OCR OFF on wide camera
        "resize_w": 768,
        "model_imgsz": 384,
        "conf_thres": 0.35,
        "process_every_n_frames": 4,
        "zone_half_height": 24,
        "min_track_frames_for_count": 3,
    },
    {
        "camera_id": "cam_90",
        "camera_name": "TVL Main Gate",
        "ip": "10.203.90.207",
        "sources": [
            f"rtsp://{ENC_USER}:{ENC_PASS}@10.203.90.207:554/profile2/media.smp",
        ],
        "roi": [0, 0, 800, 450],
        "line_p1": [0, 193],
        "line_p2": [799, 266],
        "neg_to_pos": "OUT",
        "pos_to_neg": "IN",

        "enable_plate": False,
        "enable_plate_ocr": False,
        "resize_w": 800,
        "model_imgsz": 384,
        "conf_thres": 0.30,
        "process_every_n_frames": 3,
        "zone_half_height": 18,
        "min_track_frames_for_count": 3,
    },
]