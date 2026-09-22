#!/usr/bin/env python3
"""Redact credentials from streamed pytest output without hiding failures.

The test runner sends all test output through this filter before it reaches the
terminal or a report log.  It intentionally leaves traceback structure and
ordinary local variables intact while removing the value part of common
credential forms.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path


KEY_VALUE = re.compile(
    r"(?i)\b(token|password|secret|authorization|api[_-]?key|cookie)"
    r"(\s*[=:]\s*)([^\s,'\"}]+)"
)
URL_CREDENTIALS = re.compile(r"(https?://)([^\s/@:]+):([^\s/@]+)@")
BEARER = re.compile(r"(?i)(bearer\s+)([^\s,'\"]+)")


def redact(text: str) -> str:
    text = URL_CREDENTIALS.sub(r"\1<redacted>:<redacted>@", text)
    text = BEARER.sub(r"\1<redacted>", text)
    return KEY_VALUE.sub(lambda match: f"{match.group(1)}{match.group(2)}<redacted>", text)


def redact_file(path: Path) -> None:
    """Replace a report in place after pytest has closed it."""

    if path.exists():
        path.write_text(redact(path.read_text(encoding="utf-8", errors="replace")), encoding="utf-8")


def main() -> None:
    if sys.argv[1:2] == ["--file"]:
        for raw_path in sys.argv[2:]:
            redact_file(Path(raw_path))
        return
    for line in sys.stdin:
        sys.stdout.write(redact(line))
        sys.stdout.flush()


if __name__ == "__main__":
    main()
