#!/usr/bin/env bash

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)-$$"
REPORT_ROOT="$REPO_ROOT/.test-reports/$RUN_ID"
mkdir -p "$REPORT_ROOT"
ln -sfn "$RUN_ID" "$REPO_ROOT/.test-reports/latest"

write_release_artifact_index() {
  local release_report="$1"
  local release_cache="${OSL_RELEASE_CACHE_DIR:-$REPO_ROOT/.release_test_cache}"
  [[ -n "$release_report" ]] || return 0
  mkdir -p "$release_report"
  {
    echo "# Release artifacts"
    find "$release_cache/configs" "$release_cache/outputs" -type f 2>/dev/null | sort
    if [[ -f "$release_report/release-metadata.jsonl" ]]; then
      echo "# Structured provenance: $release_report/release-metadata.jsonl"
    fi
  } >"$release_report/artifacts-index.txt"
}

interrupted() {
  local signal="$1"
  local release_report="${RELEASE_REPORT:-}"
  if [[ -n "$release_report" ]]; then
    write_release_artifact_index "$release_report"
  fi
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
  echo "release_scale=${OSL_RELEASE_SCALE:-full}"
  echo "release_profile=${OSL_RELEASE_PROFILE:-qwen}"
  echo "test_vqa_profile=${OSL_TEST_VQA_PROFILE:-qwen}"
  echo "auto_setup=${OSL_TEST_AUTO_SETUP:-1}"
  echo "deterministic_seed=42"
  echo "fast_markers=unit,smoke,integration"
  echo "release_markers=release,gpu,slow,network,pretrained,classification,localization,vqa"
  echo "python_command=$PYTHON_BIN"
  for name in OSL_RELEASE_SCALE OSL_RELEASE_PROFILE OSL_RELEASE_CACHE_DIR OSL_RELEASE_DATA_DIR OSL_RELEASE_EPOCHS \
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
  echo "SETUP LOG    : ${SETUP_LOG:-not-run}"
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

MISSING_TEST_MODULES="$("$PYTHON_BIN" - <<'PY'
import importlib.util

modules = ("pytest", "pytest_cov", "pytest_jsonreport", "pytest_timeout")
print(", ".join(module for module in modules if importlib.util.find_spec(module) is None))
PY
)"
if [[ -n "$MISSING_TEST_MODULES" ]]; then
  bootstrap_failure \
    "dependency" \
    "Required pytest modules are missing from $PYTHON_BIN: $MISSING_TEST_MODULES." \
    "Install the project test extra with: $PYTHON_BIN -m pip install -e '.[test]'"
  finish 2
  exit 2
fi

# Optional test coverage is provisioned in the same interpreter that runs
# pytest. Qwen is the default VQA profile. X-VARS uses incompatible pinned
# Transformers dependencies and must be selected in a separate environment.
TEST_VQA_PROFILE="${OSL_TEST_VQA_PROFILE:-qwen}"
if [[ "$TEST_VQA_PROFILE" != "qwen" && "$TEST_VQA_PROFILE" != "xvars" && "$TEST_VQA_PROFILE" != "none" ]]; then
  bootstrap_failure "configuration" "Unsupported OSL_TEST_VQA_PROFILE=$TEST_VQA_PROFILE." "Use qwen (default), xvars, or none."
  finish 2
  exit 2
fi

SETUP_LOG="$REPORT_ROOT/setup.log"
if [[ "${OSL_TEST_AUTO_SETUP:-1}" == "1" ]]; then
  if [[ "${RUN_OSL_RELEASE_TESTS:-0}" == "1" && "${OSL_RELEASE_PROFILE:-qwen}" == "gar" ]]; then
    SETUP_ARGS=(setup --pyg)
  elif [[ "${RUN_OSL_RELEASE_TESTS:-0}" == "1" && "${OSL_RELEASE_PROFILE:-qwen}" == "xvars" ]]; then
    SETUP_ARGS=(setup --vqa_xvars)
  else
    SETUP_ARGS=(setup --pyg --dali)
  fi
  if [[ "$TEST_VQA_PROFILE" == "qwen" && "${OSL_RELEASE_PROFILE:-qwen}" != "gar" ]]; then
    SETUP_ARGS+=(--vqa_qwen)
  elif [[ "$TEST_VQA_PROFILE" == "xvars" && "${OSL_RELEASE_PROFILE:-qwen}" != "xvars" ]]; then
    SETUP_ARGS+=(--vqa_xvars)
  fi
  echo "Provisioning OpenSportsLib test profile ($TEST_VQA_PROFILE); full log: $SETUP_LOG"
  set +e
  "$PYTHON_BIN" -m opensportslib.cli "${SETUP_ARGS[@]}" 2>&1 \
    | "$PYTHON_BIN" scripts/redact_test_stream.py | tee "$SETUP_LOG"
  SETUP_STATUS=${PIPESTATUS[0]}
  if [[ "$SETUP_STATUS" -ne 0 ]]; then
    bootstrap_failure \
      "dependency" \
      "OpenSportsLib optional test-profile setup failed (profile=$TEST_VQA_PROFILE)." \
      "Inspect $SETUP_LOG, resolve the reported package/CUDA issue, then re-run bash scripts/run_tests.sh."
    cat "$SETUP_LOG" >>"$REPORT_ROOT/fast.log"
    finish "$SETUP_STATUS"
    exit "$SETUP_STATUS"
  fi
else
  printf '%s\n' "Automatic optional-profile setup disabled (OSL_TEST_AUTO_SETUP=0)." >"$SETUP_LOG"
fi

COVERAGE_BASELINE="$(tr -d '[:space:]' < "$REPO_ROOT/scripts/coverage-baseline.txt")"
FAST_LOG="$REPORT_ROOT/fast.log"
FAST_JSON="$REPORT_ROOT/fast-report.json"
FAST_JUNIT="$REPORT_ROOT/fast-junit.xml"

# Unit and integration tests are organized under subsystem directories. Build
# the collection list from those directories rather than collecting a stale
# loose test left behind by an older checkout. Smoke tests intentionally live
# directly in their tier.
FAST_TEST_PATHS=(tests/smoke)
while IFS= read -r test_dir; do
  FAST_TEST_PATHS+=("$test_dir")
done < <(find tests/unit tests/integration -mindepth 1 -maxdepth 1 -type d | sort)

FAST_ARGS=(
  "${FAST_TEST_PATHS[@]}"
  -vv -ra --tb=long --showlocals --durations=25 --strict-markers --import-mode=importlib
  --log-cli-level=INFO
  --json-report --json-report-file="$FAST_JSON"
  --junitxml="$FAST_JUNIT"
  --cov=opensportslib --cov-report=term-missing
  --cov-report="xml:$REPORT_ROOT/coverage.xml"
  --cov-fail-under="$COVERAGE_BASELINE"
)

if [[ "$TEST_VQA_PROFILE" == "qwen" ]]; then
  FAST_ARGS+=(-m "not vqa_xvars")
elif [[ "$TEST_VQA_PROFILE" == "xvars" ]]; then
  FAST_ARGS+=(-m "not vqa_qwen")
fi

echo "Running fast OpenSportsLib tests; full log: $FAST_LOG"
set +e
"$PYTHON_BIN" -m pytest "${FAST_ARGS[@]}" 2>&1 \
  | "$PYTHON_BIN" scripts/redact_test_stream.py | tee "$FAST_LOG"
FAST_STATUS=${PIPESTATUS[0]}
REDACTION_STATUS=0
"$PYTHON_BIN" scripts/redact_test_stream.py --file "$FAST_JSON" "$FAST_JUNIT" || REDACTION_STATUS=$?
if [[ "$REDACTION_STATUS" -ne 0 && "$FAST_STATUS" -eq 0 ]]; then
  FAST_STATUS="$REDACTION_STATUS"
fi

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
  RELEASE_PROFILE="${OSL_RELEASE_PROFILE:-qwen}"
  if [[ "$RELEASE_PROFILE" != "qwen" && "$RELEASE_PROFILE" != "xvars" && "$RELEASE_PROFILE" != "gar" ]]; then
    bootstrap_failure "configuration" "Unsupported OSL_RELEASE_PROFILE=$RELEASE_PROFILE." "Use qwen, xvars, or gar."
    finish 2
    exit 2
  fi
  RELEASE_SCALE="${OSL_RELEASE_SCALE:-full}"
  if [[ "$RELEASE_SCALE" != "full" && "$RELEASE_SCALE" != "bounded" ]]; then
    bootstrap_failure "configuration" "Unsupported OSL_RELEASE_SCALE=$RELEASE_SCALE." "Use full (default) or bounded."
    finish 2
    exit 2
  fi
  RELEASE_REPORT="$REPORT_ROOT/release"
  mkdir -p "$RELEASE_REPORT"
  RELEASE_LOG="$RELEASE_REPORT/release.log"
  RELEASE_JSON="$RELEASE_REPORT/release-report.json"
  RELEASE_JUNIT="$RELEASE_REPORT/release-junit.xml"
  export OSL_RELEASE_REPORT_DIR="$RELEASE_REPORT"
  echo "Running GPU release verification (profile=$RELEASE_PROFILE scale=$RELEASE_SCALE); full log: $RELEASE_LOG"
  set +e
  case "$RELEASE_PROFILE" in
    qwen) RELEASE_MARKER="not vqa_xvars and not release_gar" ;;
    xvars) RELEASE_MARKER="vqa_xvars or gpu" ;;
    gar) RELEASE_MARKER="release_gar or gpu" ;;
  esac
  "$PYTHON_BIN" -m pytest tests/release -vv -ra -s --tb=long --showlocals --strict-markers --import-mode=importlib -m "$RELEASE_MARKER" --maxfail=1 \
    --timeout="${OSL_RELEASE_TEST_TIMEOUT:-7200}" --durations=0 \
    --log-cli-level=INFO --json-report --json-report-file="$RELEASE_JSON" \
    --junitxml="$RELEASE_JUNIT" 2>&1 \
    | "$PYTHON_BIN" scripts/redact_test_stream.py | tee "$RELEASE_LOG"
  RELEASE_STATUS=${PIPESTATUS[0]}
  REDACTION_STATUS=0
  "$PYTHON_BIN" scripts/redact_test_stream.py --file "$RELEASE_JSON" "$RELEASE_JUNIT" || REDACTION_STATUS=$?
  if [[ "$REDACTION_STATUS" -ne 0 && "$RELEASE_STATUS" -eq 0 ]]; then
    RELEASE_STATUS="$REDACTION_STATUS"
  fi
  "$PYTHON_BIN" scripts/summarize_test_report.py \
    --json "$RELEASE_JSON" --summary "$RELEASE_REPORT/summary.md" \
    --failed "$RELEASE_REPORT/failed-tests.txt" --tier release \
    --log "$RELEASE_LOG" --junit "$RELEASE_JUNIT" --exit-code "$RELEASE_STATUS"
  SUMMARY_STATUS=$?
  if [[ "$SUMMARY_STATUS" -ne 0 && "$RELEASE_STATUS" -eq 0 ]]; then
    RELEASE_STATUS="$SUMMARY_STATUS"
  fi
  write_release_artifact_index "$RELEASE_REPORT"
  "$PYTHON_BIN" scripts/write_release_manifest.py \
    --report-dir "$RELEASE_REPORT" --profile "$RELEASE_PROFILE" --scale "$RELEASE_SCALE" \
    --exit-code "$RELEASE_STATUS" || RELEASE_STATUS=$?
  printf '\n\n' >>"$REPORT_ROOT/summary.md"
  cat "$RELEASE_REPORT/summary.md" >>"$REPORT_ROOT/summary.md"
  if [[ "$RELEASE_STATUS" -ne 0 ]]; then
    finish "$RELEASE_STATUS"
    exit "$RELEASE_STATUS"
  fi
fi

finish 0
