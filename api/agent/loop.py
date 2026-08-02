"""Hand-written ReAct loop.

No step limit - the agent takes as many tool calls as it needs. The only ceiling
is wall clock: 240s, leaving 60s of headroom under Vercel's 300s hard cap.

Three exits:
  1. The model emits a decision - the normal one.
  2. The deadline passes - stop feeding tool results, force one summarizing turn.
  3. That forced turn still isn't a decision - return allow, no timer.

Exit 3 is what makes exit 2 safe. Without it, a model that keeps requesting
tools loops forever and burns the budget.

Nothing here second-guesses the decision itself. The loop runs turns and stops;
what to do about a tab is the agent's call, not the harness's.
"""
import json
import time
from datetime import datetime

from . import llm, memory, prompts, tools

DEADLINE_SECONDS = 240
ACTIONS = {"allow", "nudge", "lock"}


def _is_decision(reply: dict | None) -> bool:
    return bool(reply) and reply.get("type") == "decision" and reply.get("action") in ACTIONS


def _clean_decision(reply: dict, fallback_url: str) -> dict:
    """Keep only the four sanctioned fields, and only a sane callback."""
    out = {
        "action": reply.get("action"),
        "url": reply.get("url") or fallback_url,
        "message": (reply.get("message") or "").strip(),
    }
    delay = reply.get("callback_delay_seconds")
    if isinstance(delay, (int, float)) and delay > 0:
        out["callback_delay_seconds"] = int(delay)
    return out


def run(prompt: str, domain: str, title: str) -> tuple[dict, list]:
    """Returns (decision, steps). Never raises for model behaviour - only for a
    dead LLM endpoint, which the caller turns into status:"error"."""
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

    event = f"Opened {domain}" + (f" - '{title}'" if title else "")
    messages = (
        [{"role": "system", "content": system}]
        + prompts.FEW_SHOT
        + [{"role": "user", "content": f"{prompt}\n\n[parsed: {event}]"}]
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
            return _clean_decision(reply, domain), steps

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
                return _clean_decision(final, domain), steps
            # Exit 3: a stuck agent must never hang the browser.
            return {"action": "allow", "url": domain, "message": ""}, steps

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

        messages = messages + [
            {"role": "assistant", "content": raw},
            {"role": "user", "content": json.dumps(result, ensure_ascii=False)[:4000]},
        ]
