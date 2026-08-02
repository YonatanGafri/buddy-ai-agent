"""The four tools the agent can call.

Calendar and todo are two separate reads, not one combined call. Which the agent
calls, in what order, or whether it calls both, is its own choice - no prompt
rule forces an order. The honest cost is an extra reasoning turn versus a
combined tool; the split buys autonomy and a clearer trace.

There is deliberately no `classify` tool: the agent already has the URL, title,
calendar and memory in context, so classification happens inside the main
reasoning turn rather than in a second round-trip.
"""
import json
from pathlib import Path

from . import memory

DATA = Path(__file__).resolve().parent.parent.parent / "data"


def _read_json(name: str) -> dict:
    try:
        return json.loads((DATA / name).read_text(encoding="utf-8"))
    except Exception as exc:
        return {"error": f"could not read {name}: {exc}"}


def read_calendar() -> dict:
    """Mock MCP read - the student's calendar."""
    return _read_json("calendar.json")


def read_todo_list() -> dict:
    """Mock MCP read - the student's todo list."""
    return _read_json("todo-list.json")


def read_memory(scope: str = "short") -> dict:
    return memory.read(scope)


def update_memory(scope: str = "short", text: str = "") -> dict:
    return memory.write(scope, text)


TOOLS = {
    "read_calendar": read_calendar,
    "read_todo_list": read_todo_list,
    "read_memory": read_memory,
    "update_memory": update_memory,
}


def run_tool(name: str, args: dict | None) -> dict:
    """Dispatch by name. An unknown tool or bad args is an observation, not a
    crash - the model gets told and can correct itself on the next turn."""
    fn = TOOLS.get(name)
    if fn is None:
        return {"error": f"unknown tool {name!r}. Available: {', '.join(TOOLS)}"}
    try:
        return fn(**(args or {}))
    except TypeError as exc:
        return {"error": f"bad arguments for {name}: {exc}"}
    except Exception as exc:
        return {"error": f"{name} failed: {str(exc)[:200]}"}
