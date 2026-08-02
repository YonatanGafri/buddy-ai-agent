"""Pull the browsing event out of a free-text prompt.

This runs BEFORE the loop and costs no LLM call, so a malformed prompt is free.
The agent handles tab events, not open conversation - "what should I work on?"
must fail here rather than burn a run discovering it has no URL to judge.
"""
import re

# A domain is two-or-more dot-separated labels ending in an alphabetic TLD.
# Deliberately generic: no TLD allowlist, no domain list of any kind - a static
# list is the exact failure this project exists to fix.
_DOMAIN = re.compile(
    r"(?:https?://)?(?:www\.)?"
    r"((?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.)+[a-z]{2,24})"
    r"(?:/\S*)?",
    re.IGNORECASE,
)

# Filenames look exactly like domains (`index.py`, `todo-list.json`), and prompts
# quote them often enough to matter. This is about file extensions, not sites.
_NOT_A_DOMAIN = {
    "py", "json", "html", "js", "css", "md", "txt", "png", "jpg", "jpeg",
    "gif", "svg", "pdf", "csv", "yml", "yaml", "sh", "ts", "tsx", "jsx",
}

# Titles arrive quoted - straight, curly, or the Hebrew gershayim a mixed
# keyboard produces.
_TITLE = re.compile(r"['\"‘’“”׳״]([^'\"‘’“”׳״]{2,200})['\"‘’“”׳״]")

FORMAT_HINT = (
    "No URL found. Send a browsing event, e.g. "
    "\"Opened youtube.com - 'lo-fi beats to study to'\"."
)


def parse_event(prompt: str) -> tuple[str, str] | None:
    """(domain, title) from a browsing event, or None if there is no URL.

    Title is "" when the prompt names a site but no tab title - that is a valid
    event, just a thinner one.
    """
    if not prompt or not prompt.strip():
        return None

    for match in _DOMAIN.finditer(prompt):
        domain = match.group(1).lower().rstrip(".")
        if domain.rsplit(".", 1)[-1] in _NOT_A_DOMAIN:
            continue
        title_match = _TITLE.search(prompt)
        return domain, (title_match.group(1).strip() if title_match else "")

    return None
