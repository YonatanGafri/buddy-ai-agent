"""Rebuild public/index.html from the sources in this folder.

    python3 gui-src/build.py

The GUI arrived as generated output: one HTML file with its JS modules gzipped
and base64'd inside a <script type="__bundler/manifest"> blob, and no source
tree. Editing that meant regex surgery on decompressed strings.

So the two modules we own live here as real files, and this script installs
them - decompress nothing, patch nothing, just replace each manifest entry's
payload with the file's bytes. The vendored modules (React, Babel, the icon
set) are left exactly as they were.

Styles are not touched. The rules this app needs beyond the generated
stylesheet ship inside app.jsx, which mounts them itself.

Idempotent: running it twice produces the same file.
"""
import base64
import gzip
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "gui-src"
TARGET = ROOT / "public" / "index.html"

# Asset ids are the <script src="..."> values in the HTML body; they are what
# ties a manifest entry to a load order, so they stay fixed.
MODULES = {
    "5abae978-442a-488a-8b9e-9eb9966dcb17": SRC / "data.jsx",  # config + backend client
    "5b52d3b1-900b-4ff2-b543-63ed73b4a3d8": SRC / "app.jsx",   # the React app
}


def main() -> None:
    html = TARGET.read_text(encoding="utf-8")

    m = re.search(r'(<script type="__bundler/manifest"[^>]*>)(.*?)(</script>)', html, re.S)
    if not m:
        raise SystemExit("no bundler manifest in public/index.html")
    manifest = json.loads(m.group(2).strip())

    for asset, path in MODULES.items():
        if asset not in manifest:
            raise SystemExit(f"asset {asset} missing from the manifest")
        manifest[asset]["data"] = base64.b64encode(
            gzip.compress(path.read_text(encoding="utf-8").encode("utf-8"), mtime=0)
        ).decode("ascii")  # mtime=0 keeps the output byte-identical across runs

    html = html[: m.start(2)] + json.dumps(manifest) + html[m.end(2):]
    TARGET.write_text(html, encoding="utf-8")
    print(f"built {len(MODULES)} module(s) into {TARGET.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
