"""Sync the deposit with the thesis-med working tree.

The scripts in ``deposit/`` are copies of ``thesis-med/_paper/method-paper/scripts``
with the package import renamed (``geodesic_decay`` -> ``pulse_ep_decay``; the
working tree still uses the development name).  This script refreshes every
file the deposit already holds (scripts, manuscript and supplement source),
applies the rename, strips LaTeX comments (working notes) from the .tex
sources, and reports what changed.  It never adds files: new
scripts or result folders are added by hand, so that clinical data cannot
slip in.

    python reproduction/sync_from_worktree.py [--root /path/to/IdeaProjects] [--check]

``--check`` only reports (exit code 1 when the deposit is out of date).
"""

from __future__ import annotations

import argparse
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DEPOSIT = os.path.join(HERE, "deposit")
METHOD = "thesis-med/_paper/method-paper"
RENAME = [
    (re.compile(r"\bgeodesic_decay\b"), "pulse_ep_decay"),
    (re.compile(r"/Users/\w+/IdeaProjects/"), ""),  # absolute developer paths -> relative to GD_ROOT
]
SYNC_DIRS = [METHOD + "/scripts", METHOD + "/revision_2026-10"]  # files already present there
SYNC_EXT = (".py", ".tex")


COMMENT_LINE = re.compile(r"^\s*%")
TRAILING_COMMENT = re.compile(r"(?<!\\)%.*$")


def strip_tex_comments(text: str) -> str:
    """Drop LaTeX comment lines and trailing comments (working notes stay in the
    working tree); a trailing comment becomes a bare ``%`` so that the
    end-of-line semantics of the source are unchanged."""
    out = []
    for line in text.split("\n"):
        if COMMENT_LINE.match(line):
            continue
        out.append(TRAILING_COMMENT.sub("%", line))
    return "\n".join(out)


def transform(text: str, name: str = "") -> str:
    for pat, rep in RENAME:
        text = pat.sub(rep, text)
    if name.endswith(".tex"):
        text = strip_tex_comments(text)
    return text


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument(
        "--root",
        default=os.path.abspath(os.path.join(HERE, "..", "..")),
        help="folder that contains thesis-med/",
    )
    p.add_argument("--check", action="store_true", help="report only")
    a = p.parse_args(argv)
    changed, missing = [], []
    for rel in SYNC_DIRS:
        ddir = os.path.join(DEPOSIT, rel)
        for name in sorted(os.listdir(ddir)):
            if not name.endswith(SYNC_EXT):
                continue
            dst = os.path.join(ddir, name)
            src = os.path.join(a.root, rel, name)
            if not os.path.isfile(src):
                missing.append(os.path.join(rel, name))
                continue
            new = transform(open(src, encoding="utf-8").read(), name)
            old = open(dst, encoding="utf-8").read()
            if new != old:
                changed.append(os.path.join(rel, name))
                if not a.check:
                    open(dst, "w", encoding="utf-8").write(new)
    for f in changed:
        print(("would update " if a.check else "updated ") + f)
    for f in missing:
        print("MISSING in working tree: " + f, file=sys.stderr)
    print(f"{len(changed)} changed, {len(missing)} missing")
    return 1 if (a.check and changed) or missing else 0


if __name__ == "__main__":
    sys.exit(main())
