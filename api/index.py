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
from .agent.parser import FORMAT_HINT, parse_event

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

    # Plain regex, before the loop - a malformed prompt costs no LLM call.
    event = parse_event(prompt)
    if event is None:
        return _error(FORMAT_HINT)

    domain, title = event
    try:
        decision, steps = loop.run(prompt, domain, title)
    except llm.LLMError as exc:
        return _error(str(exc))

    return {"status": "ok", "error": None, "response": decision, "steps": steps}


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
    # repeatedly costs nothing and gets an identical answer. Re-record with
    # scripts/record_examples.py whenever the prompt or decision shape changes -
    # the spec grades consistency between these and /api/execute.
    try:
        examples = json.loads((DATA / "agent_examples.json").read_text(encoding="utf-8"))
    except Exception:
        examples = []

    return {
        "description": (
            "An AI study buddy. The browser reports tab activity; the agent can decide to reads "
            "the student's calendar, todo list and its own short and long term memories"
            "eventually allow the browser tab, lock it, or just nudge the student ."
        ),
        "purpose": (
            "Help the student succeed in their studies. Context-aware judgment "
            "instead of a static domain blocklist - nudge before lock, and follow "
            "every his nudges with callbacks."
        ),
        "prompt_template": {"template": prompts.TEMPLATE},
        "prompt_examples": examples,
    }


@app.get("/api/model_architecture")
async def model_architecture():
    return FileResponse(DATA / "architecture.png", media_type="image/png")
