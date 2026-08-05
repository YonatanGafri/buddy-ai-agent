"""Buddy - all four graded endpoints.

Vercel's Python runtime loads ONE top-level `app` from api/index.py and routes
everything through it; it does not map one file per route. So all four endpoints
live here rather than in separate modules.
https://vercel.com/docs/functions/runtimes/python

Every response from /api/execute carries exactly four top-level fields:
{status, error, response, steps}.
"""
import json
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from .agent import llm, loop, prompts

app = FastAPI()

# The GUI is served as a static file, but Postman and the grader may call from
# anywhere, and the spec forbids auth guards of any kind.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Vercel resolves relative paths against the project root, not this file, so
# anchor on __file__ instead of assuming a working directory.
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"


@app.get("/")
async def gui():
    """The GUI. On Vercel this is never reached - vercel.json routes / to the
    static file first. It exists so `uvicorn api.index:app` serves the whole
    thing from one origin, which is what makes the bundle's relative
    /api/execute work locally."""
    return FileResponse(ROOT / "public" / "index.html", media_type="text/html")


def _error(message: str) -> dict:
    return {"status": "error", "error": message, "response": None, "steps": []}


@app.post("/api/execute")
async def execute(body: dict | None = None):
    prompt = (body or {}).get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        return _error("Missing 'prompt'. Send {\"prompt\": \"...\"}.")

    try:
        result, steps = loop.run(prompt)
    except llm.LLMError as exc:
        return _error(str(exc))

    # The agent itself decides a prompt carries no browsing event, so the error
    # arrives from the loop rather than from a regex in front of it. The steps
    # are kept either way - the reasoning that reached "no tab here" is as much
    # part of the trace as a decision is.
    if "error" in result:
        return {"status": "error", "error": result["error"], "response": None, "steps": steps}

    return {"status": "ok", "error": None, "response": result, "steps": steps}



from .agent import memory
from .agent import tools
import re
from datetime import datetime
from zoneinfo import ZoneInfo

@app.post("/api/close_tab")
async def close_tab():
    now = datetime.now(ZoneInfo("Asia/Jerusalem"))
    now_str = now.strftime("%H:%M")
    
    short = memory.read("short")
    short_content = short.get("content", "")
    
    activity_match = re.search(r'\[Current Activity\]\s*- Started (.*?) at (\d{2}:\d{2})\.', short_content, flags=re.DOTALL)
    if not activity_match:
        return {"status": "ok", "message": "Already offline or time not parsable"}
        
    activity_desc = activity_match.group(1).lower()
    start_time_str = activity_match.group(2)
    
    now_time = datetime.strptime(now_str, "%H:%M")
    start_time = datetime.strptime(start_time_str, "%H:%M")
    
    if now_time < start_time:
        elapsed_mins = ((now_time.hour + 24) * 60 + now_time.minute) - (start_time.hour * 60 + start_time.minute)
    else:
        elapsed_mins = (now_time.hour * 60 + now_time.minute) - (start_time.hour * 60 + start_time.minute)
        
    if elapsed_mins < 0:
        elapsed_mins = 0
        
    category_match = re.search(r'\[(Study|Entertainment|Errands)\]', activity_desc, re.IGNORECASE)
    if category_match:
        category = category_match.group(1).capitalize()
    else:
        category = "Study"
        if any(k in activity_desc for k in ["entertainment", "music", "video", "facebook", "youtube"]):
            category = "Entertainment"
        elif any(k in activity_desc for k in ["errand", "mail", "bank"]):
            category = "Errands"
        
    pattern = rf'- {category}: (?:(\d+) hours? )?(\d+) mins'
    
    def replacer(match):
        hours = int(match.group(1)) if match.group(1) else 0
        mins = int(match.group(2))
        total_mins = hours * 60 + mins + elapsed_mins
        new_hours = total_mins // 60
        new_mins = total_mins % 60
        if new_hours > 0:
            return f"- {category}: {new_hours} hour{'s' if new_hours > 1 else ''} {new_mins} mins"
        return f"- {category}: {new_mins} mins"
            
    new_memory = re.sub(pattern, replacer, short_content)
    new_memory = re.sub(r'\[Current Activity\].*', f'[Current Activity]\n- Offline (no active tracked tabs).', new_memory, flags=re.DOTALL)
    
    memory.write("short", new_memory)
    return {"status": "ok", "message": f"Closed tab, added {elapsed_mins} mins to {category}"}

@app.post("/api/complete_task")
async def complete_task(body: dict | None = None):
    task_name = (body or {}).get("task_name")
    if not task_name:
        return _error("Missing 'task_name'. Send {\"task_name\": \"...\"}.")
        
    res = tools.update_todo_status(task_name, "completed")
    return {"status": "ok", "result": res}


@app.get("/api/team_info")
async def team_info():
    return {
        "group_batch_order_number": "1_7",
        "team_name": "Buddy",
        "students": [
            {"name": "Rotem Lanyado", "email": "lanyado98@gmail.com"},
            {"name": "Yonatan Gafri", "email": "yonatangafri@gmail.com"},
        ],
    }


@app.get("/api/agent_info")
async def agent_info():
    # Examples are recorded fixtures, not live runs: a grader hitting this
    # repeatedly costs nothing and gets an identical answer. They capture the
    # envelope and the decision shape, which is what the spec grades against
    # /api/execute - not the exact wording a model happens to produce.
    try:
        examples = json.loads((DATA / "agent_examples.json").read_text(encoding="utf-8"))
    except Exception:
        examples = []

    return {
        "description": (
            "An AI study buddy. The student says what they just opened, in their "
            "own words; the agent works out the site itself and may read their "
            "calendar, to-do list and its own short and long term memory before "
            "allowing the tab, locking it, or nudging the student."
        ),
        "purpose": (
            "Help the student succeed in their studies. Context-aware judgment "
            "instead of a static domain blocklist - nudge before lock, and follow "
            "a nudge with a callback to see whether it landed."
        ),
        "prompt_template": {"template": prompts.TEMPLATE},
        "prompt_examples": examples,
    }


@app.get("/api/model_architecture")
async def model_architecture():
    return FileResponse(DATA / "architecture.png", media_type="image/png")
