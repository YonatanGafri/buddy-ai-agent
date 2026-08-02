"""Patch the bundled GUI for the real agent's response shape.

    python3 scripts/patch_gui.py

gui.html is generated output with no source tree in the repo, so edits go
through this script rather than by hand - re-runnable and reviewable as a diff
of intent. Idempotent: patching an already-patched file is a no-op.

Three changes:
  1. executeUrl defaults to a relative /api/execute, so the GUI talks to
     whatever host serves it. It was pinned to the production URL, which meant
     a local server could never be exercised.
  2. runExecute unwraps {status, error, response, steps} into the flat shape
     the components already render, and surfaces status:"error" instead of
     silently falling back to the mock.
  3. The offline FALLBACK seeds mirror data/*.json. They shipped with the
     author's real calendar and medical appointments, which this is a public
     repo now, so they are replaced with generic Open University coursework.
"""
import base64
import gzip
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TARGETS = [ROOT / "public" / "index.html", ROOT / "gui.html"]
ASSET = "5abae978-442a-488a-8b9e-9eb9966dcb17"  # the data/config module

OLD_URL = "  executeUrl: 'https://buddy-study-agent.vercel.app/api/execute',"
NEW_URL = """  // relative by default: the GUI talks to whatever host serves it, so the same
  // build works locally, on a preview deploy and in production. Override with
  // window.BUDDY_CONFIG.executeUrl to point at another backend.
  executeUrl: '/api/execute',"""

OLD_EXEC = """/* POST a navigation to /api/execute. Hardcoded mock returns fixed text for now;
   swap the endpoint for the real agent later without touching the caller. */
async function runExecute(prompt) {
  if (!BUDDY_CONFIG.executeUrl) return { ...MOCK_EXECUTE, source: 'local' };
  try {
    const r = await fetch(BUDDY_CONFIG.executeUrl, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ prompt }),
    });
    if (!r.ok) throw new Error(`execute -> ${r.status}`);
    return { ...(await r.json()), source: 'remote' };
  } catch (e) {
    console.warn('execute failed, using local mock:', e.message);
    return { ...MOCK_EXECUTE, source: 'local' };
  }
}"""

NEW_EXEC = """/* POST a navigation to /api/execute.

   The API returns exactly four top-level fields - {status, error, response,
   steps} - where `response` is the decision object. The components render a
   flat shape, so unwrap here and leave them alone.

   A status:"error" reply is a real answer, not a transport failure: the agent
   is telling the caller the prompt carried no browsing event. Surfacing it
   beats falling back to the mock, which would look like a working decision. */
async function runExecute(prompt) {
  if (!BUDDY_CONFIG.executeUrl) return { ...MOCK_EXECUTE, source: 'local' };
  try {
    const r = await fetch(BUDDY_CONFIG.executeUrl, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ prompt }),
    });
    if (!r.ok) throw new Error(`execute -> ${r.status}`);
    const body = await r.json();

    if (body.status === 'error' || !body.response) {
      return {
        action: 'allow',
        message: body.error || 'The agent returned no decision.',
        steps: body.steps || [],
        source: 'remote',
        isError: true,
      };
    }
    return { ...body.response, steps: body.steps || [], source: 'remote' };
  } catch (e) {
    console.warn('execute failed, using local mock:', e.message);
    return { ...MOCK_EXECUTE, source: 'local' };
  }
}"""


# The offline seeds, kept in sync with data/calendar.json + data/todo-list.json
# by hand - they are two literals in a generated bundle, and a loader would be
# more machinery than the duplication costs.
#
# Matched by the region between `calendar: [` and the end of the todo block
# rather than by its previous contents, because those were the author's real
# appointments and this repo is public. The bundle already carries the new
# seeds; the regex is what keeps this script re-runnable if it changes again.
SEEDS_RE = re.compile(r"  calendar: \[\n.*?\n  \},\n", re.S)

NEW_SEEDS = """  calendar: [
    { date: '2026-08-03', title: 'מצגת 10 דקות - סוכן בינה מלאכותית' },
    { date: '2026-08-05', title: 'ממן 16 - מבוא לבינה מלאכותית' },
    { date: '2026-08-09', title: 'מפגש הנחיה - אלגוריתמים (זום)' },
    { date: '2026-08-12', title: 'בחינת סמסטר - ניהול פיננסי' },
    { date: '2026-08-19', title: 'בחינת בית 24 שעות - ניהול אסטרטגי' },
  ],
  todo: {
    pending: ['לסיים שקפים למצגת הסוכן', 'ממן 16 - שאלה 3 ו-4',
      'לקרוא פרק 7 - למידה מונחית', 'תרגול שאלות מבחן - ניהול פיננסי',
      'להירשם לקורסי סמסטר ב\\'', 'לתאם פגישה עם מנחה הסמינר'],
    completed: ['ממן 15 - הוגש', 'לצפות בהקלטת מפגש 4', 'סיכום פרקים 1-6',
      'להזמין ספר קורס מהספרייה', 'תרגיל רשות - אלגוריתמים'],
  },"""


def patch_source(src: str) -> tuple[str, list[str]]:
    applied = []
    if NEW_URL in src and NEW_EXEC in src and NEW_SEEDS in src:
        return src, applied

    if NEW_SEEDS not in src:
        src, n = SEEDS_RE.subn(NEW_SEEDS + "\n", src, count=1)
        if not n:
            raise SystemExit("FALLBACK seeds not found - the bundle changed shape")
        applied.append("FALLBACK seeds -> generic coursework")

    if OLD_URL in src:
        src = src.replace(OLD_URL, NEW_URL, 1)
        applied.append("executeUrl -> relative")
    elif NEW_URL not in src:
        raise SystemExit("executeUrl line not found - the bundle changed shape")

    if OLD_EXEC in src:
        src = src.replace(OLD_EXEC, NEW_EXEC, 1)
        applied.append("runExecute -> unwraps the four-field envelope")
    elif NEW_EXEC not in src:
        raise SystemExit("runExecute block not found - the bundle changed shape")

    return src, applied


def main() -> None:
    html = TARGETS[0].read_text(encoding="utf-8")
    match = re.search(r'(<script type="__bundler/manifest"[^>]*>)(.*?)(</script>)', html, re.S)
    if not match:
        raise SystemExit("no bundler manifest in public/index.html")

    manifest = json.loads(match.group(2).strip())
    entry = manifest[ASSET]

    src = gzip.decompress(base64.b64decode(entry["data"])).decode("utf-8")
    patched, applied = patch_source(src)
    if not applied:
        print("Already patched - nothing to do.")
        return

    entry["data"] = base64.b64encode(
        gzip.compress(patched.encode("utf-8"), mtime=0)  # mtime=0 keeps it reproducible
    ).decode("ascii")

    new_html = html[: match.start(2)] + json.dumps(manifest) + html[match.end(2) :]
    for target in TARGETS:
        target.write_text(new_html, encoding="utf-8")

    for line in applied:
        print(f"  patched: {line}")
    print(f"Wrote {len(TARGETS)} file(s).")


if __name__ == "__main__":
    main()
