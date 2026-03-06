import sqlite3

conn = sqlite3.connect("database/gate_events.db")
cur = conn.cursor()

cur.execute("SELECT COUNT(*) FROM events")
print("Total events:", cur.fetchone()[0])

cur.execute("""
SELECT date, hour, label, direction, count
FROM hourly_summary
ORDER BY date DESC, hour DESC, count DESC
LIMIT 10
""")
rows = cur.fetchall()
print("Top hourly rows:")
for r in rows:
    print(r)

conn.close()