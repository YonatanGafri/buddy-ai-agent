"""Hand-written ReAct loop.

No step limit - the agent takes as many tool calls as it needs. The only ceiling
is wall clock: 240s, leaving 60s of headroom under Vercel's 300s hard cap.

Four exits:
  1. The model emits a decision - the normal one.
  2. The model emits an error - it read the prompt and found no tab to judge.
  3. The deadline passes - stop feeding tool results, force one summarizing turn.
  4. That forced turn still isn't a decision - return allow, no callback.

Exit 4 is what makes exit 3 safe. Without it, a model that keeps requesting
tools loops forever and burns the budget.

Nothing here second-guesses the decision itself. The loop runs turns and stops;
what to do about a tab - and whether the prompt even contains one - is the
agent's call, not the harness's.

The one thing it does finish on the agent's behalf is bookkeeping, and only when
the agent armed a timer it left nothing to read - see _ensure_short_note. That
is not a second guess: the action, the message and the callback come back
exactly as written.
"""
import json
import time
from datetime import datetime
from zoneinfo import ZoneInfo

from . import llm, memory, prompts, tools

DEADLINE_SECONDS = 240
ACTIONS = {"allow", "nudge", "lock"}

# There are five tools and no reason to call more than a few at once. The cap is
# a runaway guard, not a budget: a model that asks for the same read eight times
# gets the first four and an observation it can correct from.
MAX_BATCH = 4


def _is_decision(reply: dict | None) -> bool:
    return bool(reply) and reply.get("type") == "decision" and reply.get("action") in ACTIONS


def _calls(reply: dict | None) -> list[tuple[str, dict]]:
    """The tool calls this reply asks for, one or many.

    Batched form is {"type":"tool_call","tools":[{"tool":..,"args":{..}}, ..]}.
    The single form is untouched and still the fallback, so every few-shot
    example, and any model that ignores batching entirely, behaves exactly as
    before - this widens what is accepted rather than replacing it.

    Duplicates are dropped. Asking for the same read twice in one batch is not a
    second opinion, just a second identical payload resent on every later turn.
    An unparseable reply has no tool name and lands here as [(None, {})], which
    run_tool turns into the same "unknown tool" observation it always did.
    """
    reply = reply or {}
    batch = reply.get("tools")
    if not isinstance(batch, list) or not batch:
        return [(reply.get("tool"), reply.get("args") or {})]

    out: list[tuple[str, dict]] = []
    seen: set[str] = set()
    for call in batch:
        if not isinstance(call, dict):
            continue
        name = call.get("tool")
        args = call.get("args") or {}
        if not isinstance(args, dict):
            args = {}
        key = json.dumps([name, args], sort_keys=True, ensure_ascii=False)
        if key in seen:
            continue
        seen.add(key)
        out.append((name, args))
        if len(out) == MAX_BATCH:
            break
    return out or [(reply.get("tool"), reply.get("args") or {})]


def _bad_url(reply: dict) -> str | None:
    """Why this decision's url is not a site, or None if it is fine.

    A nudge and a lock both render a browser tab in the GUI built from this
    field, so a value that is not a site draws a tab for a place that does not
    exist. Live runs returned action:"nudge" with url:"calendar", and another
    with url:"" - one drew a tab called "calendar", the other an empty one.

    Deliberately NOT a parser. It answers "is this string a domain", never
    "which domain did the student mean" - working that out from free text is the
    agent's whole job, and doing it here would move the classification out of
    the agent and into the harness. The agent picks the action, writes the
    message and supplies the corrected url; this only reports that the field it
    sent cannot be a site.

    allow is exempt: a wake that finds nothing to follow up on returns an empty
    url by design, and allow renders no tab.
    """
    if reply.get("action") == "allow":
        return None
    url = (reply.get("url") or "").strip()
    if not url:
        return "you returned no url"
    # A bare label - "calendar", "social media" - has no dot in it. Anything
    # dotted is taken at face value: judging whether a real domain is the RIGHT
    # one is the agent's call, not this function's.
    if "." not in url.strip("."):
        return f"{url!r} is not a site"
    return None


def _clean_decision(reply: dict) -> dict:
    """Keep only the sanctioned fields, and only a sane callback.

    message is dropped on allow. The student only ever reads a message through a
    nudge toast, so an allow message is written, sent and never displayed - and a
    model asked for one every turn starts narrating ("Good, keep going!"), which
    reads as surveillance for the one action that is supposed to be silent.
    Enforced here rather than trusted to the prompt: it is one line, and it makes
    the contract true regardless of what the model returns.
    """
    action = reply.get("action")
    out = {"action": action, "url": (reply.get("url") or "").strip(), "message": (reply.get("message") or "").strip()}
    callback = reply.get("callback")
    if isinstance(callback, (int, float)) and callback > 0:
        out["callback"] = int(callback)
    elif action == "nudge":
        # A nudge without a callback is a wish: no wake ever fires and the
        # chain dies silently. The prompt says every nudge sets one; when the
        # model skips it anyway, arm a default so the contract stays true.
        out["callback"] = 600
    return out


def _wrote_short(steps: list, url: str) -> bool:
    """Did this run leave itself a note in short memory ABOUT THIS SITE?

    Any short write used to count, so a housekeeping write early in the run
    (stale-day cleanup, say) satisfied the check and the wake then read a note
    that never mentioned the site being nudged. Requiring the judged domain in
    the text closes that; with no url, any non-empty write still counts.
    """
    domain = (url or "").strip().lower()
    for s in steps:
        if s["module"] != "Tools.rewrite_memory":
            continue
        args = s["prompt"].get("args") or {}
        text = str(args.get("text") or "").strip()
        if (args.get("scope") == "short" and text and s["response"].get("ok")
                and (not domain or domain in text.lower())):
            return True
    return False



def run(prompt: str) -> tuple[dict, list]:
    """Returns (result, steps). result is a decision dict, or {"error": ...} when
    the agent judged there was no browsing event to act on.

    Never raises for model behaviour - only for a dead LLM endpoint, which the
    caller turns into status:"error".
    """
    deadline = time.time() + DEADLINE_SECONDS
    now = datetime.now(ZoneInfo("Asia/Jerusalem"))

    short = memory.read("short")
    system = prompts.build_system(
        now=now.strftime("%Y-%m-%d %H:%M"),
        weekday=now.strftime("%A"),
        today=memory.today(),
        short_written=memory.day_of(short["updated_at"]),
        short_age=memory.age_of(short["updated_at"]),
        short_memory=short["content"],
    )

    # The prompt goes to the model verbatim. No pre-parse, no [parsed: ...] hint -
    # working out the site and title from free text is part of the agent's job.
    #
    # One exception, and it is not student text: the wake sentinel below is a
    # fixed string the client sends itself, so matching it is not classification.
    # A bare wake was allowed even at "nudged 2x" through eleven prompt rewrites.
    # The cause is shape, not wording - every few-shot wake carries its note
    # INSIDE the user turn, while the live wake arrived bare with the note only
    # up in the system block, so the examples never matched and the model read
    # the note as background. Same note, moved into the user turn, escalates.
    # This does not tell the agent anything it did not write itself.
    messages = (
        [{"role": "system", "content": system}]
        + prompts.FEW_SHOT
        + [{"role": "user", "content": prompts.inject_context(prompt, short["content"], now.strftime("%H:%M"))}]
    )
    steps: list[dict] = []
    asked_for_url = False  # the malformed-url nudge below fires at most once

    while True:
        raw = llm.invoke(messages)
        reply = llm.parse_reply(raw)
        # prompt is the full array sent this turn, verbatim - it repeats across
        # steps by design, because that is what was actually sent.
        steps.append({
            "module": "ReAct.LLM",
            "prompt": {"messages": list(messages)},
            "response": reply if reply is not None else {"raw": raw},
        })

        if _is_decision(reply):
            # One chance to name the site it judged. Handed back the same way an
            # unknown tool is - as an observation it can correct from - rather
            # than rejected or repaired here. Once only: a model that sends a
            # second bad url is not going to find a good one on the third try,
            # and the decision itself is still worth returning.
            problem = _bad_url(reply)
            if problem and not asked_for_url and time.time() < deadline:
                asked_for_url = True
                messages = messages + [
                    {"role": "assistant", "content": raw},
                    {"role": "user", "content": prompts.observation(json.dumps({
                        "error": f"{problem}. A {reply.get('action')} draws a browser "
                                 "tab from that field, so it has to be the site you "
                                 "judged, as a bare domain. Which site was it? Resend "
                                 "your decision - same action, same message - with the "
                                 "url filled in. If you cannot name one, return an "
                                 "error instead.",
                    }, ensure_ascii=False))},
                ]
                continue
            decision = _clean_decision(reply)
            url = decision.get("url", "")
            if not _wrote_short(steps, url) and time.time() < deadline:
                messages = messages + [
                    {"role": "assistant", "content": raw},
                    {"role": "user", "content": prompts.observation(json.dumps({
                        "error": "You returned a decision without writing to short memory first. Every decision (including allow) MUST be documented in short memory BEFORE you return it, otherwise you will lose track of the student's timeline. Please call rewrite_memory to document your decision (and copy existing memory if needed), and then return the decision again."
                    }, ensure_ascii=False))}
                ]
                continue
            
            return decision, steps

        # Exit 2: the agent read the prompt and found nothing to judge.
        if reply and reply.get("type") == "error":
            return {"error": (reply.get("message") or "").strip() or prompts.TEMPLATE}, steps

        if time.time() >= deadline:
            messages = messages + [
                {"role": "assistant", "content": raw},
                {"role": "user", "content": prompts.SUMMARIZE},
            ]
            raw = llm.invoke(messages)
            final = llm.parse_reply(raw)
            steps.append({
                "module": "ReAct.Summarize",
                "prompt": {"messages": list(messages)},
                "response": final if final is not None else {"raw": raw},
            })
            if _is_decision(final):
                decision = _clean_decision(final)
                return decision, steps
            # Exit 4: a stuck agent must never hang the browser.
            return {"action": "allow", "url": ""}, steps

        # Not a decision and time remains: run the tools it asked for. An
        # unparseable reply has no tool name, so it lands in run_tool's unknown
        # branch and comes back as an observation the model can correct from.
        #
        # One step per tool, even when they arrived in one batch. The steps list
        # is graded surface and the module names match the architecture diagram,
        # so a batch must not collapse into a step whose module is "Tools.three
        # of them". What changes is how many reasoning turns are paid for, not
        # what the trace looks like.
        batched = _calls(reply)
        results = []
        for name, args in batched:
            result = tools.run_tool(name, args)
            # Keyed by tool name, but read_website can legitimately appear twice
            # with different urls - so the url rides in the key. Without this the
            # second fetch silently overwrites the first in the observation and
            # the model sees one result for two questions it asked.
            key = f"{name}({args['url']})" if name == "read_website" and args.get("url") else name
            results.append((key, result))
            steps.append({
                "module": f"Tools.{name}" if name in tools.TOOLS else "Tools.unknown",
                "prompt": {"tool": name, "args": args},
                "response": result,
            })

        # Labelled, because the role cannot distinguish it. A tool result and the
        # student both arrive as "user" - so an unlabelled {"ok":true} is the
        # last thing the model sees, and it answers that instead of the prompt.
        # One live trace returned "No browsing tab here" for a prompt reading
        # "opened youtube.com": two writes had buried it under two bare acks.
        #
        # A batch comes back as ONE observation keyed by tool name. Separate
        # user turns per result would be indistinguishable from the student
        # typing several times, and an unkeyed list makes the model match
        # results to calls by position - which it gets wrong the moment one
        # errors. The budget is per observation, not per tool, so a batch of
        # four cannot crowd out the student's own line.
        if len(results) == 1:
            payload = json.dumps(results[0][1], ensure_ascii=False)[:4000]
        else:
            payload = json.dumps(
                {name: res for name, res in results}, ensure_ascii=False)[:4000]
        messages = messages + [
            {"role": "assistant", "content": raw},
            {"role": "user", "content": prompts.observation(payload)},
        ]
