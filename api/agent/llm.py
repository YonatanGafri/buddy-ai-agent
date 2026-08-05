"""OpenAI-compatible chat client.

Raw HTTP rather than an SDK, for two reasons. The trace must carry the exact
message array that was sent, so nothing may reshape it on the way out. And the
provider swap - LM Studio locally, LLMod.ai at submission - has to be a base_url
change and nothing else.
"""
import json
import os

import requests

BASE_URL = os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:1234/v1").rstrip("/")
API_KEY = os.environ.get("OPENAI_API_KEY", "lm-studio")  # local servers ignore it
MODEL = os.environ.get("BUDDY_MODEL", "qwen3-coder-30b-a3b-instruct-mlx")
TIMEOUT = 120


class LLMError(RuntimeError):
    pass


def invoke(messages: list[dict]) -> str:
    """Send a message array, return the raw reply text.

    response_format asks for JSON, but not every OpenAI-compatible server honors
    it, so the caller must still handle prose. Servers that reject the field
    outright get one retry without it.
    """
    # No temperature. A low one keeps the JSON shape stable, but the gpt-5
    # family accepts only the default and rejects the field outright, so sending
    # it makes every call a 400. The default it is.
    body = {
        "model": MODEL,
        "messages": messages,
        "response_format": {"type": "json_object"},
    }

    def post():
        return requests.post(
            f"{BASE_URL}/chat/completions",
            headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
            json=body,
            timeout=TIMEOUT,
        )

    try:
        r = post()
        if r.status_code == 400 and "response_format" in r.text:
            body.pop("response_format")
            r = post()
        # The server's own words, not just the status line. A 400 saying which
        # parameter it rejected is the whole diagnosis; "400 Bad Request" alone
        # leaves the key and the model name equally suspect.
        if not r.ok:
            raise LLMError(f"LLM request failed: {r.status_code} - {r.text[:300]}")
        return r.json()["choices"][0]["message"]["content"] or ""
    except requests.RequestException as exc:
        raise LLMError(f"LLM request failed: {str(exc)[:200]}") from exc
    except (KeyError, IndexError, ValueError) as exc:
        raise LLMError(f"unexpected LLM response shape: {str(exc)[:200]}") from exc


def parse_reply(text: str) -> dict | None:
    """The model's JSON object, or None if there isn't one.

    Some servers wrap JSON in prose or a code fence even in JSON mode, so fall
    back to the outermost braces. None means "not a decision" to the loop, which
    keeps going - the deadline bounds it.
    """
    if not text:
        return None
    text = text.strip()

    if text.startswith("```"):
        text = text.split("```")[1] if "```" in text[3:] else text[3:]
        text = text.removeprefix("json").strip()

    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except ValueError:
        pass

    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        obj = json.loads(text[start : end + 1])
        return obj if isinstance(obj, dict) else None
    except ValueError:
        return None
