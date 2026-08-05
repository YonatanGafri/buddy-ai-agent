"""The four tools the agent can call.

Calendar and todo are two separate reads, not one combined call. Which the agent
calls, in what order, or whether it calls both, is its own choice - no prompt
rule forces an order. The honest cost is an extra reasoning turn versus a
combined tool; the split buys autonomy and a clearer trace.

There is deliberately no `classify` tool: the agent already has the URL, title,
calendar and memory in context, so classification happens inside the main
reasoning turn rather than in a second round-trip.

All four read from Supabase, which is also what the GUI's context panel shows -
one source, so the panel's claim to show "what Buddy knows" is true. The JSON
files under data/ are seed input for the seeding script, nothing more.
"""
import html as _html
import re
import urllib.request

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


def read_long_memory() -> dict:
    """Long memory only, and deliberately no scope argument.

    Short memory is inlined in the system prompt on every turn, so a read of it
    could only spend a turn handing back what the model can already see. Given
    the choice the model takes it anyway, on most runs. Removing the option is
    more reliable than asking it not to, and it makes the remaining tool
    self-describing.
    """
    return memory.read("long")


def rewrite_memory(scope: str = "short", text: str = "") -> dict:
    return memory.write(scope, text)


_TITLE_RE = re.compile(r"<title[^>]*>(.*?)</title>", re.I | re.S)
_META_DESC_RE = re.compile(
    r'<meta[^>]+(?:name|property)=["\'](?:og:)?description["\'][^>]*>', re.I)
_CONTENT_RE = re.compile(r'content=["\']([^"\']*)["\']', re.I)


def read_website(url: str = "") -> dict:
    """Fetch a page and return only its <title> and meta description.

    Page text never reaches the model - the whole HTML stays in here and two
    short strings come out. What comes out is still web content, not the
    student, so the loop's OBSERVATION prefix applies to it like any other
    tool result.
    """
    if not url:
        return {"error": "read_website needs a url"}
    if not re.match(r"^https?://", url, re.I):
        url = "https://" + url
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Buddy)"})
        with urllib.request.urlopen(req, timeout=5) as resp:
            # ponytail: first 64KB only - title/meta live in <head>; full pages can be MBs
            raw = resp.read(65536).decode("utf-8", errors="replace")
    except Exception as exc:
        return {"error": f"could not fetch {url}: {str(exc)[:200]}"}

    title = _TITLE_RE.search(raw)
    desc = None
    meta = _META_DESC_RE.search(raw)
    if meta:
        content = _CONTENT_RE.search(meta.group(0))
        desc = content.group(1) if content else None

    clean = lambda s: _html.unescape(re.sub(r"\s+", " ", s)).strip()[:300]
    return {
        "title": clean(title.group(1)) if title else None,
        "description": clean(desc) if desc else None,
    }


TOOLS = {
    "read_calendar": read_calendar,
    "read_todo_list": read_todo_list,
    "read_long_memory": read_long_memory,
    "rewrite_memory": rewrite_memory,
    "read_website": read_website,
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
