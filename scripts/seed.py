#!/usr/bin/env python3
"""Buddy - seed Supabase from the local data files.

Reads the four source files under data/ and upserts them into the three tables
created by supabase/schema.sql. Idempotent: re-running replaces the context rows
and upserts the two memory rows.

Usage:
    export SUPABASE_URL="https://<project>.supabase.co"
    export SUPABASE_SERVICE_KEY="<service_role key>"   # NOT the anon key
    python scripts/seed.py

The service_role key is required because RLS blocks anon writes. Keep it server
side only - never ship it to the GUI.
"""
import json
import os
import sys
from pathlib import Path

try:
    import requests
except ImportError:
    sys.exit("pip install requests")

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"  # calendar/todo JSON read at runtime, plus the memory seeds

URL = os.environ.get("SUPABASE_URL", "").rstrip("/")
KEY = os.environ.get("SUPABASE_SERVICE_KEY", "")
if not URL or not KEY:
    sys.exit("Set SUPABASE_URL and SUPABASE_SERVICE_KEY env vars first.")

H = {
    "apikey": KEY,
    "Authorization": f"Bearer {KEY}",
    "Content-Type": "application/json",
}


def rest(method, table, params=None, body=None, prefer=None):
    headers = dict(H)
    if prefer:
        headers["Prefer"] = prefer
    r = requests.request(
        method, f"{URL}/rest/v1/{table}", headers=headers, params=params, json=body, timeout=30
    )
    if not r.ok:
        sys.exit(f"{method} {table} -> {r.status_code} {r.text}")
    return r


def wipe(table):
    # delete-all: id greater-than-or-equal 0 (bigint pk) matches every row
    rest("DELETE", table, params={"id": "gte.0"})


# ---- memory (short + long) ----
short = (DATA / "short-memory.md").read_text(encoding="utf-8").strip()
long_ = (DATA / "long-memory.md").read_text(encoding="utf-8").strip()
rest(
    "POST", "memory",
    body=[{"scope": "short", "content": short}, {"scope": "long", "content": long_}],
    prefer="resolution=merge-duplicates",
)
print("memory: upserted short + long")

# ---- calendar ----
cal = json.loads((DATA / "calendar.json").read_text(encoding="utf-8"))
events = [{"date": e["date"], "title": e["title"]} for e in cal["weekly_events"]]
wipe("calendar_events")
rest("POST", "calendar_events", body=events)
print(f"calendar_events: inserted {len(events)}")

# ---- todo ----
todo = json.loads((DATA / "todo-list.json").read_text(encoding="utf-8"))
rows = []
for i, t in enumerate(todo.get("pending_tasks", [])):
    rows.append({"task": t, "status": "pending", "ord": i})
for i, t in enumerate(todo.get("completed_tasks", [])):
    rows.append({"task": t, "status": "completed", "ord": i})
wipe("todo_tasks")
rest("POST", "todo_tasks", body=rows)
print(f"todo_tasks: inserted {len(rows)}")

print("\nDone. Buddy's Supabase is seeded.")
