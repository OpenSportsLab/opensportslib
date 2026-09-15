#!/usr/bin/env bash

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)-$$"
REPORT_ROOT="$REPO_ROOT/.test-reports/$RUN_ID"
mkdir -p "$REPORT_ROOT"
ln -sfn "$RUN_ID" "$REPO_ROOT/.test-reports/latest"

interrupted() {
  local signal="$1"
  {
    echo "# Test run interrupted"
    echo
    echo "- Classification: interruption"
    echo "- Signal: $signal"
    echo "- Reports retained at: $REPORT_ROOT"
  } >"$REPORT_ROOT/summary.md"
  echo "interrupted: $signal" >"$REPORT_ROOT/failed-tests.txt"
  echo "Test run interrupted by $signal; reports retained at $REPORT_ROOT" >&2
  exit 130
}
trap 'interrupted INT' INT
trap 'interrupted TERM' TERM

PYTHON_BIN="${OSL_TEST_PYTHON:-python}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  PYTHON_BIN=python3
fi

ENV_REPORT="$REPORT_ROOT/environment.txt"
{
  echo "OpenSportsLib test environment"
  echo "run_id=$RUN_ID"
  echo "repository=$REPO_ROOT"
  echo "command=bash scripts/run_tests.sh"
  echo "utc_started=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "release_mode=${RUN_OSL_RELEASE_TESTS:-0}"
  echo "python_command=$PYTHON_BIN"
  for name in OSL_RELEASE_CACHE_DIR OSL_RELEASE_DATA_DIR OSL_RELEASE_EPOCHS \
    OSL_RELEASE_MAX_CLIPS OSL_RELEASE_MAX_GAMES OSL_RELEASE_CLS_NUM_FRAMES \
    OSL_RELEASE_CLS_REPO OSL_RELEASE_CLS_REVISION OSL_RELEASE_LOC_E2E_REPO \
    OSL_RELEASE_LOC_FEATURES_REPO OSL_RELEASE_VQA_REPO OSL_RELEASE_VQA_REVISION; do
    if [[ -n "${!name:-}" ]]; then
      echo "env.$name=${!name}"
    fi
  done
  "$PYTHON_BIN" --version 2>&1 || true
  uname -a || true
  git rev-parse HEAD 2>/dev/null || true
  "$PYTHON_BIN" - <<'PY' 2>&1 || true
import importlib.metadata
import platform
import sys

print(f"python_executable={sys.executable}")
print(f"platform={platform.platform()}")
for package in ("opensportslib", "pytest", "pytest-cov", "pytest-json-report", "pytest-timeout", "torch", "transformers"):
    try:
        print(f"package.{package}={importlib.metadata.version(package)}")
    except importlib.metadata.PackageNotFoundError:
        print(f"package.{package}=NOT_INSTALLED")
try:
    import torch
    print(f"cuda.available={torch.cuda.is_available()}")
    print(f"cuda.device_count={torch.cuda.device_count()}")
    for index in range(torch.cuda.device_count()):
        print(f"cuda.device.{index}={torch.cuda.get_device_name(index)}")
except Exception as exc:
    print(f"cuda.error={type(exc).__name__}: {exc}")
PY
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=index,name,driver_version,memory.total,memory.free --format=csv,noheader || true
  fi
} >"$ENV_REPORT"

finish() {
  local exit_code="$1"
  echo
  echo "============================================================"
  if [[ "$exit_code" -eq 0 ]]; then
    echo "OVERALL RESULT: PASS"
  else
    echo "OVERALL RESULT: FAIL (exit $exit_code)"
  fi
  echo "DEBUG REPORTS: $REPORT_ROOT"
  echo "SUMMARY      : $REPORT_ROOT/summary.md"
  echo "ENVIRONMENT  : $ENV_REPORT"
  echo "============================================================"
}

bootstrap_failure() {
  local classification="$1"
  local message="$2"
  local remediation="$3"
  printf '%s\n' "$message" >"$REPORT_ROOT/fast.log"
  printf '%s: %s\n' "$classification" "$message" >"$REPORT_ROOT/failed-tests.txt"
  cat >"$REPORT_ROOT/fast-report.json" <<EOF
{"created": "$(date -u +%Y-%m-%dT%H:%M:%SZ)", "exitcode": 2, "summary": {"failed": 1, "total": 1}, "tests": []}
EOF
  cat >"$REPORT_ROOT/fast-junit.xml" <<EOF
<?xml version="1.0" encoding="utf-8"?><testsuites><testsuite name="environment" tests="1" failures="1"><testcase classname="environment" name="$classification"><failure message="$classification">See summary.md and environment.txt</failure></testcase></testsuite></testsuites>
EOF
  cat >"$REPORT_ROOT/summary.md" <<EOF
# Test bootstrap failure

- Result: FAILED
- Classification: $classification
- Error: $message
- Remediation: $remediation
- Full log: $REPORT_ROOT/fast.log
- JUnit: $REPORT_ROOT/fast-junit.xml
- JSON: $REPORT_ROOT/fast-report.json
- Diagnostics: $ENV_REPORT
EOF
}

if ! "$PYTHON_BIN" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 12) else 1)'; then
  bootstrap_failure "environment" "OpenSportsLib requires Python 3.12 or newer (selected: $PYTHON_BIN)." "Activate the supported OpenSportsLib environment."
  finish 2
  exit 2
fi

if ! "$PYTHON_BIN" -c 'import pytest, pytest_cov, pytest_jsonreport, pytest_timeout' >/dev/null 2>&1; then
  bootstrap_failure "dependency" "Required pytest plugins are not installed." "Install the project test extra in the active environment."
  finish 2
  exit 2
fi

COVERAGE_BASELINE="$(tr -d '[:space:]' < "$REPO_ROOT/scripts/coverage-baseline.txt")"
FAST_LOG="$REPORT_ROOT/fast.log"
FAST_JSON="$REPORT_ROOT/fast-report.json"
FAST_JUNIT="$REPORT_ROOT/fast-junit.xml"

FAST_ARGS=(
  tests/unit tests/smoke tests/integration
  -vv -ra --tb=long --durations=25
  --log-cli-level=INFO
  --json-report --json-report-file="$FAST_JSON"
  --junitxml="$FAST_JUNIT"
  --cov=opensportslib --cov-report=term-missing
  --cov-report="xml:$REPORT_ROOT/coverage.xml"
  --cov-fail-under="$COVERAGE_BASELINE"
)

echo "Running fast OpenSportsLib tests; full log: $FAST_LOG"
set +e
"$PYTHON_BIN" -m pytest "${FAST_ARGS[@]}" 2>&1 | tee "$FAST_LOG"
FAST_STATUS=${PIPESTATUS[0]}

"$PYTHON_BIN" scripts/summarize_test_report.py \
  --json "$FAST_JSON" --summary "$REPORT_ROOT/summary.md" \
  --failed "$REPORT_ROOT/failed-tests.txt" --tier fast \
  --log "$FAST_LOG" --junit "$FAST_JUNIT" --exit-code "$FAST_STATUS"
SUMMARY_STATUS=$?
if [[ "$SUMMARY_STATUS" -ne 0 && "$FAST_STATUS" -eq 0 ]]; then
  FAST_STATUS="$SUMMARY_STATUS"
fi

if [[ "$FAST_STATUS" -ne 0 ]]; then
  finish "$FAST_STATUS"
  exit "$FAST_STATUS"
fi

if [[ "${RUN_OSL_RELEASE_TESTS:-0}" == "1" ]]; then
  RELEASE_REPORT="$REPORT_ROOT/release"
  mkdir -p "$RELEASE_REPORT"
  RELEASE_LOG="$RELEASE_REPORT/release.log"
  RELEASE_JSON="$RELEASE_REPORT/release-report.json"
  RELEASE_JUNIT="$RELEASE_REPORT/release-junit.xml"
  echo "Running GPU release verification; full log: $RELEASE_LOG"
  set +e
  "$PYTHON_BIN" -m pytest tests/release -vv -ra -s --tb=long --maxfail=1 \
    --timeout="${OSL_RELEASE_TEST_TIMEOUT:-7200}" --durations=0 \
    --log-cli-level=INFO --json-report --json-report-file="$RELEASE_JSON" \
    --junitxml="$RELEASE_JUNIT" 2>&1 | tee "$RELEASE_LOG"
  RELEASE_STATUS=${PIPESTATUS[0]}
  "$PYTHON_BIN" scripts/summarize_test_report.py \
    --json "$RELEASE_JSON" --summary "$RELEASE_REPORT/summary.md" \
    --failed "$RELEASE_REPORT/failed-tests.txt" --tier release \
    --log "$RELEASE_LOG" --junit "$RELEASE_JUNIT" --exit-code "$RELEASE_STATUS"
  SUMMARY_STATUS=$?
  if [[ "$SUMMARY_STATUS" -ne 0 && "$RELEASE_STATUS" -eq 0 ]]; then
    RELEASE_STATUS="$SUMMARY_STATUS"
  fi
  RELEASE_CACHE="${OSL_RELEASE_CACHE_DIR:-$REPO_ROOT/.release_test_cache}"
  find "$RELEASE_CACHE/configs" "$RELEASE_CACHE/outputs" -type f 2>/dev/null \
    | sort >"$RELEASE_REPORT/artifacts-index.txt" || true
  printf '\n\n' >>"$REPORT_ROOT/summary.md"
  cat "$RELEASE_REPORT/summary.md" >>"$REPORT_ROOT/summary.md"
  if [[ "$RELEASE_STATUS" -ne 0 ]]; then
    finish "$RELEASE_STATUS"
    exit "$RELEASE_STATUS"
  fi
fi

finish 0
