#!/usr/bin/env python3
"""Create human-readable diagnostics from pytest-json-report output."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re


SECRET_PATTERN = re.compile(
    r"(?i)(token|password|secret|authorization|api[_-]?key|cookie)(\s*[=:]\s*)(\S+)"
)
URL_CREDENTIALS = re.compile(r"(https?://)([^\s/@:]+):([^\s/@]+)@")
BEARER = re.compile(r"(?i)(bearer\s+)([^\s,'\"]+)")

CATEGORIES = (
    ("timeout", ("timeout", "timed out")),
    ("subprocess", ("subprocess", "calledprocesserror", "returncode")),
    ("network", ("network", "connection", "urlopen", "huggingface", "socket")),
    ("dependency", ("modulenotfounderror", "importerror", "optional dependency")),
    ("configuration", ("config", "yaml", "omegaconf", "schema")),
    ("data/annotation", ("annotation", "manifest", "json", "parquet", "h5")),
    ("dataset loading", ("dataset", "dataloader", "collate")),
    ("checkpoint", ("checkpoint", "state_dict", "weights", "resume")),
    ("forward pass", ("forward", "logits", "tensor shape")),
    ("loss/backward", ("loss", "backward", "gradient")),
    ("optimizer/scheduler", ("optimizer", "scheduler", "learning rate")),
    ("prediction format", ("prediction", "save_predictions", "serialization")),
    ("evaluation", ("evaluate", "evaluation", "metric")),
    ("inference", ("infer", "inference", "generate")),
    ("model construction", ("model", "backbone", "head", "forward")),
    ("environment", ("cuda", "gpu", "environment", "python version")),
)


def redact(value: str) -> str:
    value = URL_CREDENTIALS.sub(r"\1<redacted>:<redacted>@", value)
    value = BEARER.sub(r"\1<redacted>", value)
    return SECRET_PATTERN.sub(lambda match: f"{match.group(1)}{match.group(2)}<redacted>", value)


def classify(text: str) -> str:
    lowered = text.lower()
    for category, needles in CATEGORIES:
        if any(needle in lowered for needle in needles):
            return category
    return "unknown"


def phase_for(test: dict) -> str:
    for phase in ("setup", "call", "teardown"):
        if test.get(phase, {}).get("outcome") == "failed":
            return phase
    return "collection"


def failure_text(test: dict) -> str:
    phase = phase_for(test)
    detail = test.get(phase, {})
    return str(detail.get("longrepr") or detail.get("crash", {}).get("message") or "No exception text recorded")


def component(nodeid: str) -> str:
    for name in ("classification", "localization", "vqa", "config", "data", "models", "api", "tools"):
        if name in nodeid.lower():
            return name
    return "package/core"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", required=True, type=Path)
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--failed", required=True, type=Path)
    parser.add_argument("--tier", required=True)
    parser.add_argument("--log", required=True)
    parser.add_argument("--junit", required=True)
    parser.add_argument("--exit-code", required=True, type=int)
    args = parser.parse_args()

    if not args.json.exists():
        args.failed.write_text("pytest did not create a JSON report\n", encoding="utf-8")
        args.summary.write_text(
            f"# {args.tier.title()} test report\n\nPytest did not create its JSON report. "
            f"Inspect `{args.log}` for collection or startup errors.\n",
            encoding="utf-8",
        )
        return 1

    try:
        report = json.loads(args.json.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        args.failed.write_text(f"report-generation: {type(exc).__name__}: {exc}\n", encoding="utf-8")
        args.summary.write_text(
            f"# {args.tier.title()} test report\n\n- Result: FAILED\n"
            f"- Classification: report generation\n- Error: `{redact(str(exc))}`\n"
            f"- Full log: `{args.log}`\n",
            encoding="utf-8",
        )
        return 1
    failures = [test for test in report.get("tests", []) if test.get("outcome") in {"failed", "error"}]
    collectors = [item for item in report.get("collectors", []) if item.get("outcome") == "failed"]
    failed_run = args.exit_code != 0 or bool(failures or collectors)
    lines = [f"# {args.tier.title()} test report", "", f"- Result: {'FAILED' if failed_run else 'PASSED'}"]
    summary = report.get("summary", {})
    lines.append("- Counts: " + ", ".join(f"{key}={value}" for key, value in sorted(summary.items())))
    lines.extend([f"- Full log: `{args.log}`", f"- JUnit: `{args.junit}`", ""])

    failed_ids = []
    for test in failures:
        nodeid = test.get("nodeid", "unknown")
        text = redact(failure_text(test))
        first_error = next((line.strip() for line in reversed(text.splitlines()) if line.strip()), "Unknown error")
        failed_ids.append(nodeid)
        lines.extend([
            f"## {nodeid}", "",
            f"- Component: {component(nodeid)}",
            f"- Tier: {args.tier}",
            f"- Phase: {phase_for(test)}",
            f"- Classification: {classify(nodeid + ' ' + text)}",
            f"- Error: `{first_error[:500]}`",
            "",
        ])
    for collector in collectors:
        nodeid = collector.get("nodeid", "collection")
        failed_ids.append(nodeid)
        lines.extend([f"## Collection failure: {nodeid}", "", "See the full log for the original traceback.", ""])

    if args.exit_code != 0 and not failures and not collectors:
        failed_ids.append(f"pytest-exit-code-{args.exit_code}")
        lines.extend([
            "## Suite-level failure", "",
            f"- Pytest exit code: {args.exit_code}",
            "- Classification: coverage, collection, interruption, or pytest infrastructure",
            "- Error: Tests did not record an individual failure; inspect the full log.",
            "",
        ])

    args.failed.write_text("".join(f"{nodeid}\n" for nodeid in failed_ids), encoding="utf-8")
    args.summary.write_text("\n".join(lines), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
