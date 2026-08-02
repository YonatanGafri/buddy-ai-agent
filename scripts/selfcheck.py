"""Self-check - no LLM key, no network, no test framework.

    python3 scripts/selfcheck.py

Covers the parser, all three loop exits, and the response envelope. The LLM is
stubbed with a scripted reply list, so this costs $0 and runs in under a second.
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from api.agent import llm, loop, tools  # noqa: E402
from api.agent.parser import parse_event  # noqa: E402

failures = []


def check(label, got, want):
    if got != want:
        failures.append(f"{label}\n    got:  {got!r}\n    want: {want!r}")


# ---------------------------------------------------------------- parser

check("bare domain + title",
      parse_event("Opened youtube.com - 'lo-fi beats to study to'"),
      ("youtube.com", "lo-fi beats to study to"))

check("full URL with scheme and path",
      parse_event("Opened https://www.youtube.com/watch?v=abc - 'lecture 4'"),
      ("youtube.com", "lecture 4"))

check("wake prompt",
      parse_event("Waking up - you asked to check back. Student is now on youtube.com - 'lo-fi beats'."),
      ("youtube.com", "lo-fi beats"))

check("domain with no title",
      parse_event("switched to stackoverflow.com"),
      ("stackoverflow.com", ""))

check("hebrew title, hebrew domain",
      parse_event("Opened ynet.co.il - 'חדשות הבוקר'"),
      ("ynet.co.il", "חדשות הבוקר"))

check("subdomain kept",
      parse_event("Opened docs.google.com - 'AI presentation'"),
      ("docs.google.com", "AI presentation"))

# Must NOT parse - these have to fail before the loop, not after a paid run.
check("conversational prompt rejected", parse_event("what should I work on?"), None)
check("empty prompt rejected", parse_event(""), None)
check("whitespace prompt rejected", parse_event("   "), None)
check("filename is not a domain", parse_event("check todo-list.json please"), None)


# ------------------------------------------------------------------ loop

class Stub:
    """Replaces llm.invoke with a scripted list of replies."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = 0

    def __call__(self, messages):
        self.calls += 1
        return self.replies.pop(0) if self.replies else '{"type":"tool_call","tool":"read_memory","args":{}}'


real_invoke = llm.invoke

# Exit 1: an immediate decision.
llm.invoke = Stub(['{"type":"decision","action":"lock","url":"youtube.com","message":"Later.","callback_delay_seconds":300}'])
decision, steps = loop.run("Opened youtube.com - 'x'", "youtube.com", "x")
check("exit 1 action", decision["action"], "lock")
check("exit 1 callback kept", decision["callback_delay_seconds"], 300)
check("exit 1 step count", len(steps), 1)
check("exit 1 module", steps[0]["module"], "ReAct.LLM")
check("exit 1 step keys", sorted(steps[0]), ["module", "prompt", "response"])
check("exit 1 prompt is the message array", "messages" in steps[0]["prompt"], True)

# A tool call, then a decision - the trace must show both, in order.
llm.invoke = Stub([
    '{"type":"tool_call","tool":"read_calendar","args":{}}',
    '{"type":"decision","action":"nudge","url":"youtube.com","message":"Soon."}',
])
decision, steps = loop.run("Opened youtube.com - 'x'", "youtube.com", "x")
check("tool run: modules in order",
      [s["module"] for s in steps],
      ["ReAct.LLM", "Tools.read_calendar", "ReAct.LLM"])
check("tool run: no phantom callback", "callback_delay_seconds" in decision, False)

# An unknown tool is an observation, not a crash.
llm.invoke = Stub([
    '{"type":"tool_call","tool":"read_brain","args":{}}',
    '{"type":"decision","action":"allow","url":"youtube.com","message":"ok"}',
])
decision, steps = loop.run("Opened youtube.com - 'x'", "youtube.com", "x")
check("unknown tool traced", steps[1]["module"], "Tools.unknown")
check("unknown tool recovered", decision["action"], "allow")

# Prose instead of JSON, then a decision - must not derail the loop.
llm.invoke = Stub([
    "I think I should look at the calendar first.",
    '{"type":"decision","action":"allow","url":"youtube.com","message":"ok"}',
])
decision, steps = loop.run("Opened youtube.com - 'x'", "youtube.com", "x")
check("prose reply survived", decision["action"], "allow")

# JSON wrapped in a code fence - some servers do this even in JSON mode.
check("fenced JSON parsed",
      llm.parse_reply('```json\n{"type":"decision","action":"allow"}\n```'),
      {"type": "decision", "action": "allow"})
check("garbage parses to None", llm.parse_reply("no json here at all"), None)

# A decision is returned as-is - the loop never second-guesses it. A nudge
# without a memory write is the agent's call to make, and its own problem.
llm.invoke = Stub(['{"type":"decision","action":"nudge","url":"twitch.tv","message":"Later?"}'])
decision, steps = loop.run("Opened twitch.tv - 'x'", "twitch.tv", "x")
check("decision returned unchallenged", len(steps), 1)
check("nudge passes through", decision["action"], "nudge")

# Exit 2: deadline passes, forced turn returns a decision.
llm.invoke = Stub([
    '{"type":"tool_call","tool":"read_memory","args":{"scope":"short"}}',
    '{"type":"decision","action":"nudge","url":"youtube.com","message":"Time."}',
])
real_deadline = loop.DEADLINE_SECONDS
loop.DEADLINE_SECONDS = -1  # already expired
decision, steps = loop.run("Opened youtube.com - 'x'", "youtube.com", "x")
check("exit 2 summarize module", steps[-1]["module"], "ReAct.Summarize")
check("exit 2 decision", decision["action"], "nudge")

# Exit 3: the forced turn still isn't a decision - allow, no timer.
llm.invoke = Stub(["still thinking", "still thinking"])
decision, steps = loop.run("Opened youtube.com - 'x'", "youtube.com", "x")
check("exit 3 falls back to allow", decision["action"], "allow")
check("exit 3 sets no timer", "callback_delay_seconds" in decision, False)
check("exit 3 url preserved", decision["url"], "youtube.com")
loop.DEADLINE_SECONDS = real_deadline
llm.invoke = real_invoke


# -------------------------------------------------------------- envelope

from fastapi.testclient import TestClient  # noqa: E402

from api.index import app  # noqa: E402

client = TestClient(app)

FIELDS = ["error", "response", "status", "steps"]

r = client.post("/api/execute", json={"prompt": "what should I work on?"})
body = r.json()
check("no-URL is a 200 with status error", (r.status_code, body["status"]), (200, "error"))
check("no-URL envelope fields", sorted(body), FIELDS)
check("no-URL response is null", body["response"], None)
check("no-URL steps empty", body["steps"], [])
check("error names the format", "e.g." in body["error"], True)

r = client.post("/api/execute", json={})
check("missing prompt errors", r.json()["status"], "error")
check("missing prompt envelope", sorted(r.json()), FIELDS)

llm.invoke = Stub(['{"type":"decision","action":"allow","url":"youtube.com","message":"go"}'])
r = client.post("/api/execute", json={"prompt": "Opened youtube.com - 'x'"})
body = r.json()
check("ok envelope fields", sorted(body), FIELDS)
check("ok status", body["status"], "ok")
check("ok error is null", body["error"], None)
check("decision fields", sorted(body["response"]), ["action", "message", "url"])

# A dead LLM endpoint must surface as status:error, not a 500.
def _dead(messages):
    raise llm.LLMError("LLM request failed: connection refused")

llm.invoke = _dead
r = client.post("/api/execute", json={"prompt": "Opened youtube.com - 'x'"})
check("dead LLM is a clean error", (r.status_code, r.json()["status"]), (200, "error"))
check("dead LLM envelope", sorted(r.json()), FIELDS)
llm.invoke = real_invoke

r = client.get("/api/team_info")
check("team_info fields", sorted(r.json()), ["group_batch_order_number", "students", "team_name"])

r = client.get("/api/agent_info")
info = r.json()
check("agent_info fields", sorted(info),
      ["description", "prompt_examples", "prompt_template", "purpose"])
check("prompt_template is an object with template", "template" in info["prompt_template"], True)

r = client.get("/api/model_architecture")
check("architecture is a PNG", r.headers["content-type"], "image/png")


# --------------------------------------------------------------- report

if failures:
    print(f"FAIL - {len(failures)} check(s):\n")
    for f in failures:
        print(f"  {f}\n")
    sys.exit(1)
print("All checks passed.")
