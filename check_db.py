import sqlite3

conn = sqlite3.connect("database/gate_events.db")
cur = conn.cursor()

cur.execute("SELECT COUNT(*) FROM events")
print("Total events:", cur.fetchone()[0])

cur.execute("""
SELECT ts, label, direction, track_id, conf, plate_text, plate_score, plate_image_path, snapshot_path
FROM events
WHERE label IN ('car', 'truck', 'bus', 'micro', 'motorcycle')
ORDER BY id DESC
LIMIT 20
""")

rows = cur.fetchall()

print("\nLatest events:")
for r in rows:
    print(r)

conn.close()