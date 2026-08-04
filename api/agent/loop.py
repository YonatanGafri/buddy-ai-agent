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

from . import llm, memory, prompts, tools

DEADLINE_SECONDS = 240
ACTIONS = {"allow", "nudge", "lock"}


def _is_decision(reply: dict | None) -> bool:
    return bool(reply) and reply.get("type") == "decision" and reply.get("action") in ACTIONS


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
    out = {"action": action, "url": (reply.get("url") or "").strip()}
    if action != "allow":
        out["message"] = (reply.get("message") or "").strip()
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


def _ensure_short_note(decision: dict, steps: list, short_before: str) -> list:
    """Arming a callback with nothing written is a follow-up aimed at nothing.

    A wake is told only that the timer fired - never which site it was for. The
    note in short memory is the entire briefing, and the prompt says so at
    length. In production it got written on half the nudges: 5 of 10 set a
    callback and wrote nothing. That is not a cosmetic miss. One measured run
    nudged twitter.com, wrote no note, and when its callback fired the agent
    read the PREVIOUS site's leftover note and followed up about wikipedia.org -
    a site the student had already left. The chain does not just go quiet, it
    reattaches to the wrong tab, which is worse than silence.

    Rewriting the prompt is the move that has already failed here - this exact
    instruction is spelled out with reasons and it still lands about half the
    time. So this is enforced where it is deterministic instead. What is written
    is only what the loop watched happen: the action, the domain the agent
    itself returned, and the delay it chose. No count, because the honest count
    lives in the note the agent did not write, and inventing "nudged 1x" over an
    unknown history is the same lie the prompt warns about. No deadline, no task
    - nothing that would need a tool read to be true.

    Deliberately narrow. It fires only when a callback is armed AND nothing was
    written, so an agent that keeps its own books is never touched, and a
    decision without a timer is left completely alone.
    """
    url = (decision.get("url") or "").strip()
    if not decision.get("callback") or _wrote_short(steps, url):
        return steps

    text = (
        f"{decision['action']} {url}".strip()
        + f" - callback in {decision['callback']}s to check whether it landed."
        " (Written by the loop: the decision set a timer but left no note, so"
        " this records only what was decided - no nudge count, and nothing"
        " about deadlines or tasks.)"
    )
    # Anything the agent knew before this run still belongs to it: appended, not
    # overwritten, or a wake loses the history it was actually counting on.
    if short_before.strip():
        text = f"{short_before.strip()}\n{text}"

    result = tools.run_tool("rewrite_memory", {"scope": "short", "text": text})
    # Traced like any other tool call - the module name says who called it, so
    # the step list stays an honest record rather than a write appearing from
    # nowhere. Spec-shaped: {module, prompt, response}, same as every other step.
    return steps + [{
        "module": "Tools.rewrite_memory",
        "prompt": {"tool": "rewrite_memory",
                   "args": {"scope": "short", "text": text},
                   "note": "written by the loop - callback armed with no note"},
        "response": result,
    }]


def run(prompt: str) -> tuple[dict, list]:
    """Returns (result, steps). result is a decision dict, or {"error": ...} when
    the agent judged there was no browsing event to act on.

    Never raises for model behaviour - only for a dead LLM endpoint, which the
    caller turns into status:"error".
    """
    deadline = time.time() + DEADLINE_SECONDS
    now = datetime.now()

    short = memory.read("short")
    system = prompts.build_system(
        now=now.strftime("%Y-%m-%d %H:%M"),
        weekday=now.strftime("%A"),
        today=memory.today(),
        short_written=memory.day_of(short["updated_at"]),
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
        + [{"role": "user", "content": prompts.wake_prompt(prompt, short["content"])}]
    )
    steps: list[dict] = []

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
            decision = _clean_decision(reply)
            return decision, _ensure_short_note(decision, steps, short["content"])

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
                return decision, _ensure_short_note(decision, steps, short["content"])
            # Exit 4: a stuck agent must never hang the browser.
            return {"action": "allow", "url": ""}, steps

        # Not a decision and time remains: run the tool it asked for. An
        # unparseable reply has no tool name, so it lands in run_tool's unknown
        # branch and comes back as an observation the model can correct from.
        name = (reply or {}).get("tool")
        args = (reply or {}).get("args") or {}
        result = tools.run_tool(name, args)
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
        messages = messages + [
            {"role": "assistant", "content": raw},
            {"role": "user", "content": prompts.observation(
                json.dumps(result, ensure_ascii=False)[:4000])},
        ]
