"""Self-check for data/agent_examples.json - the fixtures /api/agent_info serves.

    python3 data/check_examples.py

Two things are asserted, both of them things the spec grades:

  * every module the loop can emit for a normal run appears in some example, so
    the names in /api/agent_info match the names in a real /api/execute trace;
  * every example carries the exact envelope /api/execute returns, and every
    step the exact {module, prompt, response} shape.

Tools.unknown and ReAct.Summarize are excluded deliberately - the first is the
unrecognized-tool-name fallback, the second only fires on the 240s deadline
path. Neither belongs in a set of examples of normal operation.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from api.agent import tools  # noqa: E402

EXAMPLES = json.loads((ROOT / "data" / "agent_examples.json").read_text(encoding="utf-8"))

# What a normal run can emit: the reasoning module plus one per registered tool.
expected = {"ReAct.LLM"} | {f"Tools.{name}" for name in tools.TOOLS}

documented = {s["module"] for e in EXAMPLES for s in e["steps"]}

missing = expected - documented
assert not missing, (
    f"modules a live trace can show but no example documents: {sorted(missing)}. "
    "Add an example that exercises them, or the names in /api/agent_info will not "
    "match the names in /api/execute."
)

unknown = documented - expected - {"ReAct.Summarize"}
assert not unknown, f"examples document modules the agent cannot emit: {sorted(unknown)}"

for i, e in enumerate(EXAMPLES):
    assert set(e) == {"prompt", "full_response", "steps"}, f"example {i}: keys {sorted(e)}"
    assert isinstance(e["prompt"], str) and e["prompt"].strip(), f"example {i}: empty prompt"

    # steps live alongside full_response, not inside it - the fixture would
    # otherwise carry every trace twice.
    full = e["full_response"]
    assert set(full) == {"status", "error", "response"}, \
        f"example {i}: full_response keys {sorted(full)}"
    if full["status"] == "ok":
        assert full["error"] is None and full["response"] is not None, f"example {i}: bad ok envelope"
    else:
        assert full["status"] == "error" and full["response"] is None, f"example {i}: bad error envelope"
        assert isinstance(full["error"], str) and full["error"], f"example {i}: empty error"

    assert isinstance(e["steps"], list) and e["steps"], f"example {i}: no steps"
    for j, s in enumerate(e["steps"]):
        assert set(s) == {"module", "prompt", "response"}, f"example {i} step {j}: keys {sorted(s)}"
        assert isinstance(s["module"], str) and s["module"], f"example {i} step {j}: bad module"
        assert isinstance(s["prompt"], dict), f"example {i} step {j}: prompt not an object"
        assert isinstance(s["response"], dict), f"example {i} step {j}: response not an object"

print(f"agent_examples selfcheck: {len(EXAMPLES)} examples, "
      f"{len(documented)} modules, all assertions passed")
