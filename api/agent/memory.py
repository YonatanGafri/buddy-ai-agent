"""Two memory rows in Supabase, over PostgREST.

Plain `requests` rather than supabase-py: scripts/seed.py already talks to
PostgREST this way, and the SDK is cold-start weight for two queries.

Vercel's filesystem is ephemeral, so file-backed memory would silently vanish in
production and the learning loop would do nothing.
"""
import os
from datetime import date, datetime, timezone

import requests

URL = os.environ.get("SUPABASE_URL", "").rstrip("/")
KEY = os.environ.get("SUPABASE_SERVICE_KEY", "") or os.environ.get("SUPABASE_KEY", "")
TIMEOUT = 10

# Memory is a nice-to-have per turn, not a precondition for deciding. If
# Supabase is unreachable the agent should still judge the tab in front of it,
# so failures degrade to empty rather than raising. In-process only - a Vercel
# invocation is short-lived, so this is a per-request cache, not shared state.
_FALLBACK: dict[str, str] = {"short": "", "long": ""}


def _headers() -> dict:
    return {
        "apikey": KEY,
        "Authorization": f"Bearer {KEY}",
        "Content-Type": "application/json",
    }


def configured() -> bool:
    return bool(URL and KEY)


def select(table: str, params: dict) -> list | dict:
    """GET rows from a table. Returns {"error": ...} instead of raising, so a
    tool call can hand the failure to the model as an observation."""
    if not configured():
        return {"error": "Supabase is not configured on this server."}
    try:
        r = requests.get(
            f"{URL}/rest/v1/{table}", headers=_headers(), params=params, timeout=TIMEOUT
        )
        r.raise_for_status()
        return r.json()
    except Exception as exc:
        return {"error": f"could not read {table}: {str(exc)[:200]}"}



def update(table: str, match: dict, data: dict) -> dict:
    """PATCH rows in a table."""
    if not configured():
        return {"error": "Supabase is not configured on this server."}
    
    params = {k: f"eq.{v}" for k, v in match.items()}
    try:
        r = requests.patch(
            f"{URL}/rest/v1/{table}",
            headers={**_headers(), "Prefer": "return=minimal"},
            params=params,
            json=data,
            timeout=TIMEOUT
        )
        r.raise_for_status()
        return {"ok": True}
    except Exception as exc:
        return {"error": f"could not update {table}: {str(exc)[:200]}"}

def read(scope: str) -> dict:
    """{content, updated_at} for a scope. Missing row reads as empty."""
    if scope not in ("short", "long"):
        return {"content": "", "updated_at": None, "error": f"unknown scope {scope!r}"}

    if not configured():
        return {"content": _FALLBACK[scope], "updated_at": None}

    try:
        r = requests.get(
            f"{URL}/rest/v1/memory",
            headers=_headers(),
            params={"scope": f"eq.{scope}", "select": "content,updated_at"},
            timeout=TIMEOUT,
        )
        r.raise_for_status()
        rows = r.json()
    except Exception as exc:
        # Surfaced to the model as an observation - it can decide without memory.
        return {"content": _FALLBACK[scope], "updated_at": None, "error": str(exc)[:200]}

    if not rows:
        return {"content": "", "updated_at": None}
    return {"content": rows[0].get("content") or "", "updated_at": rows[0].get("updated_at")}


def write(scope: str, text: str) -> dict:
    """Overwrite a scope. Never appends - the agent rewrites its own summary
    each time so memory stays prompt-sized."""
    if scope not in ("short", "long"):
        return {"ok": False, "error": f"unknown scope {scope!r}"}

    _FALLBACK[scope] = text

    if not configured():
        return {"ok": True, "note": "no Supabase configured - kept in memory for this request only"}

    body = {
        "scope": scope,
        "content": text,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    try:
        r = requests.post(
            f"{URL}/rest/v1/memory",
            headers={**_headers(), "Prefer": "resolution=merge-duplicates,return=minimal"},
            json=body,
            timeout=TIMEOUT,
        )
        r.raise_for_status()
    except Exception as exc:
        return {"ok": False, "error": str(exc)[:200]}
    return {"ok": True}


def day_of(stamp: str | None) -> str | None:
    """Date part of an updated_at, for end-of-day promotion.

    The agent compares this against today (both are in the system prompt) to
    notice a new day and consolidate short into long before deciding anything.
    """
    if not stamp:
        return None
    try:
        return datetime.fromisoformat(stamp.replace("Z", "+00:00")).date().isoformat()
    except Exception:
        return str(stamp)[:10] or None


def age_of(stamp: str | None) -> str | None:
    """How long ago a note was written, in words - "74 minutes ago".

    Computed here rather than left to the model, for the same reason day_of is:
    the agent writes clock times into its own notes ("Locked netflix.com at
    20:25") and `now` is printed at the top of every prompt, so everything
    needed to notice an hour has passed is already in front of it. It does not
    do the subtraction. A note reading "exam in an hour" was read back 65
    minutes later and asserted to a student opening an unrelated site.

    One age for the whole note, because that is what the row stores - a rewrite
    restamps every line in it. The agent's own clock times inside the text are
    what distinguish lines within a note; this catches the common case, where
    the whole note is stale.
    """
    if not stamp:
        return None
    try:
        written = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
    except Exception:
        return None

    seconds = (datetime.now(timezone.utc) - written).total_seconds()
    # Clock skew between Supabase and the runtime can make a fresh note look
    # slightly future-dated. "in -2 minutes" is worse than saying nothing.
    if seconds < 0:
        return None
    minutes = int(seconds // 60)
    if minutes < 1:
        return "just now"
    if minutes < 60:
        return f"{minutes} minute{'s' if minutes > 1 else ''} ago"
    hours = minutes // 60
    if hours < 24:
        return f"{hours} hour{'s' if hours > 1 else ''} ago"
    days = hours // 24
    return f"{days} day{'s' if days > 1 else ''} ago"


def today() -> str:
    return date.today().isoformat()
