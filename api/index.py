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
