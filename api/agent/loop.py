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
    return out


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
    messages = (
        [{"role": "system", "content": system}]
        + prompts.FEW_SHOT
        + [{"role": "user", "content": prompt}]
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
            return _clean_decision(reply), steps

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
                return _clean_decision(final), steps
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

        messages = messages + [
            {"role": "assistant", "content": raw},
            {"role": "user", "content": json.dumps(result, ensure_ascii=False)[:4000]},
        ]
