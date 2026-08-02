"""Record /api/agent_info fixtures from real agent runs.

    OPENAI_BASE_URL=http://127.0.0.1:1234/v1 \
    BUDDY_MODEL=qwen3-coder-30b-a3b-instruct-mlx \
    python3 scripts/record_examples.py

Writes data/agent_examples.json. /api/agent_info serves that file and never
calls the LLM, so a grader hitting it repeatedly costs nothing and gets an
identical answer.

These are REAL runs, not invented traces: the spec grades consistency between
these examples and what /api/execute actually returns. Re-record whenever the
prompt or the decision shape changes - stale fixtures are worse than none.

Memory is stubbed per scenario rather than read from Supabase. A recording run
would otherwise depend on whatever state the database happens to be in, and the
lock example needs a prior nudge count to escalate from - that is the scenario,
not a side effect to hope for.
"""
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from api.agent import llm, loop, memory  # noqa: E402
from api.agent.parser import parse_event  # noqa: E402

OUT = ROOT / "data" / "agent_examples.json"

# (label, prompt, seeded short memory). The three the spec names, plus a wake
# where the student left and the no-URL rejection - between them they show every
# action and both failure modes.
SCENARIOS = [
    # Domains here must not appear in prompts.FEW_SHOT. An earlier recording
    # used arxiv.org, which is example 1, and the model returned that example's
    # message word for word.
    (
        "on-task tab -> allow",
        "Opened overleaf.com - 'AI presentation slides'",
        "Working on the AI presentation due tomorrow.",
    ),
    (
        "distraction with work pending -> nudge + callback",
        "Opened tiktok.com - 'for you'",
        "Working on the AI presentation due tomorrow. Nothing nudged yet today.",
    ),
    (
        "repeat distraction after nudges -> lock",
        "Waking up - you asked to check back. Student is now on tiktok.com - 'for you'",
        "AI presentation due tomorrow. Nudged tiktok.com 2x - still there both times.",
    ),
    (
        "wake, student left on their own -> allow",
        "Waking up - you asked to check back. Student is now on docs.google.com - 'presentation draft'",
        "AI presentation due tomorrow. Nudged tiktok.com 1x - check whether they left.",
    ),
    (
        "no browsing event -> error, no LLM call",
        "what should I work on?",
        "",
    ),
]


def main() -> None:
    examples = []

    for label, prompt, seed in SCENARIOS:
        print(f"- {label} ... ", end="", flush=True)

        # Seed memory in-process. memory.write falls back to _FALLBACK when
        # Supabase is unconfigured, and read serves it back.
        memory._FALLBACK["short"] = seed
        memory._FALLBACK["long"] = "Best learning window 08:00-11:00 - strictest then."

        event = parse_event(prompt)
        if event is None:
            from api.agent.parser import FORMAT_HINT
            full = {"status": "error", "error": FORMAT_HINT, "response": None, "steps": []}
            print("error (no LLM call)")
        else:
            domain, title = event
            try:
                decision, steps = loop.run(prompt, domain, title)
            except llm.LLMError as exc:
                print(f"FAILED: {exc}")
                sys.exit(1)
            full = {"status": "ok", "error": None, "response": decision, "steps": steps}
            print(f"{decision['action']} ({len(steps)} steps)")

        examples.append({
            "prompt": prompt,
            "full_response": full,
            "steps": full["steps"],
        })

    OUT.write_text(json.dumps(examples, ensure_ascii=False, indent=2), encoding="utf-8")

    size = OUT.stat().st_size
    print(f"\nWrote {OUT.relative_to(ROOT)} - {len(examples)} examples, {size / 1024:.0f} KB")
    if size > 1_000_000:
        print("WARNING: over 1 MB. Every step carries the full message array "
              "verbatim, so this grows fast. Consider fewer or shorter examples.")


if __name__ == "__main__":
    main()
