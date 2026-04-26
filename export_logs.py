import os
import csv
import sqlite3
from datetime import datetime

DB_PATH = "database/gate_events.db"
EVENTS_ROOT = os.path.join("logs", "daily", "events")
HOURLY_ROOT = os.path.join("logs", "daily", "hourly")

os.makedirs(EVENTS_ROOT, exist_ok=True)
os.makedirs(HOURLY_ROOT, exist_ok=True)

def export_day(day):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    events_path = os.path.join(EVENTS_ROOT, f"events_{day}.csv")
    hourly_path = os.path.join(HOURLY_ROOT, f"hourly_{day}.csv")

    cur.execute("""
        SELECT
            id,
            camera_id,
            camera_name,
            ts,
            direction,
            label,
            track_id,
            conf,
            roi,
            in_total,
            out_total,
            net_total,
            snapshot_path,
            plate_text,
            plate_score,
            plate_image_path
        FROM events
        WHERE date = ?
        ORDER BY id ASC
    """, (day,))
    rows = cur.fetchall()

    with open(events_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
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
            "plate_image_path",
        ])
        w.writerows(rows)

    cur.execute("""
        SELECT
            camera_id,
            camera_name,
            date,
            hour,
            label,
            direction,
            COUNT(*) AS count,
            SUM(CASE WHEN plate_image_path IS NOT NULL AND plate_image_path != '' THEN 1 ELSE 0 END) AS plate_image_saved_count,
            SUM(CASE WHEN plate_text IS NOT NULL AND plate_text != '' THEN 1 ELSE 0 END) AS plate_ocr_success_count
        FROM events
        WHERE date = ?
        GROUP BY camera_id, camera_name, date, hour, label, direction
        ORDER BY hour ASC, camera_id ASC, label ASC, direction ASC
    """, (day,))
    rows = cur.fetchall()

    with open(hourly_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "camera_id",
            "camera_name",
            "date",
            "hour",
            "label",
            "direction",
            "count",
            "plate_image_saved_count",
            "plate_ocr_success_count",
        ])
        w.writerows(rows)

    conn.close()
    print(f"Exported {events_path}")
    print(f"Exported {hourly_path}")

if __name__ == "__main__":
    export_day(datetime.now().strftime("%Y-%m-%d"))