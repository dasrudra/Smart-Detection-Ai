from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import sqlite3
from datetime import datetime

DB_PATH = "../database/gate_events.db"

app = FastAPI(title="Office Gate AI Dashboard")

templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

def db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    today = datetime.now().strftime("%Y-%m-%d")

    conn = db()
    cur = conn.cursor()

    cur.execute("SELECT COUNT(*) AS c FROM events WHERE date=?", (today,))
    total_events = cur.fetchone()["c"]

    cur.execute("SELECT COUNT(*) AS c FROM events WHERE date=? AND direction='IN'", (today,))
    in_total = cur.fetchone()["c"]

    cur.execute("SELECT COUNT(*) AS c FROM events WHERE date=? AND direction='OUT'", (today,))
    out_total = cur.fetchone()["c"]

    # class wise today
    cur.execute("""
        SELECT label,
               SUM(CASE WHEN direction='IN' THEN 1 ELSE 0 END) AS in_count,
               SUM(CASE WHEN direction='OUT' THEN 1 ELSE 0 END) AS out_count
        FROM events
        WHERE date=?
        GROUP BY label
        ORDER BY (in_count + out_count) DESC
        LIMIT 20
    """, (today,))
    class_rows = cur.fetchall()

    conn.close()

    return templates.TemplateResponse("index.html", {
        "request": request,
        "today": today,
        "total_events": total_events,
        "in_total": in_total,
        "out_total": out_total,
        "net": in_total - out_total,
        "class_rows": class_rows
    })

@app.get("/events", response_class=HTMLResponse)
def events(request: Request, limit: int = 200):
    conn = db()
    cur = conn.cursor()
    cur.execute("""
        SELECT ts, date, hour, direction, label, track_id, conf, snapshot_path
        FROM events
        ORDER BY id DESC
        LIMIT ?
    """, (limit,))
    rows = cur.fetchall()
    conn.close()

    return templates.TemplateResponse("events.html", {
        "request": request,
        "rows": rows,
        "limit": limit
    })

@app.get("/hourly", response_class=HTMLResponse)
def hourly(request: Request, date: str = None):
    if date is None:
        date = datetime.now().strftime("%Y-%m-%d")

    conn = db()
    cur = conn.cursor()
    cur.execute("""
        SELECT date, hour, label, direction, count
        FROM hourly_summary
        WHERE date=?
        ORDER BY hour ASC, label ASC, direction ASC
    """, (date,))
    rows = cur.fetchall()
    conn.close()

    return templates.TemplateResponse("hourly.html", {
        "request": request,
        "date": date,
        "rows": rows
    })