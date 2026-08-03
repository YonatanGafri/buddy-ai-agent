"""The four tools the agent can call.

Calendar and todo are two separate reads, not one combined call. Which the agent
calls, in what order, or whether it calls both, is its own choice - no prompt
rule forces an order. The honest cost is an extra reasoning turn versus a
combined tool; the split buys autonomy and a clearer trace.

There is deliberately no `classify` tool: the agent already has the URL, title,
calendar and memory in context, so classification happens inside the main
reasoning turn rather than in a second round-trip.

All four read from Supabase, which is also what the GUI's context panel shows.
They used to read data/*.json while the panel read the database, so a row edited
in Supabase was visible to the student and invisible to the agent - the panel
claimed to show "what Buddy knows" and did not. The JSON files are now seed input
for scripts/seed.py, nothing more.
"""
from . import memory


def read_calendar() -> dict:
    """Mock MCP read - the student's calendar."""
    rows = memory.select("calendar_events", {"select": "date,title", "order": "date.asc"})
    if isinstance(rows, dict):
        return rows  # {"error": ...}
    return {"weekly_events": rows}


def read_todo_list() -> dict:
    """Mock MCP read - the student's todo list."""
    rows = memory.select("todo_tasks", {"select": "task,status,ord", "order": "ord.asc"})
    if isinstance(rows, dict):
        return rows
    return {
        "pending_tasks": [r["task"] for r in rows if r.get("status") == "pending"],
        "completed_tasks": [r["task"] for r in rows if r.get("status") == "completed"],
    }


def read_memory(scope: str = "short") -> dict:
    return memory.read(scope)


def rewrite_memory(scope: str = "short", text: str = "") -> dict:
    return memory.write(scope, text)


TOOLS = {
    "read_calendar": read_calendar,
    "read_todo_list": read_todo_list,
    "read_memory": read_memory,
    "rewrite_memory": rewrite_memory,
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
