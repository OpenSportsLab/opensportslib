#!/usr/bin/env bash

set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

WITH_INTEGRATION=0
LOG_FILE="scripts/smoke-test-report.log"

for arg in "$@"; do
  case "$arg" in
    --with-integration)
      WITH_INTEGRATION=1
      ;;
    *)
      LOG_FILE="$arg"
      ;;
  esac
done

: > "$LOG_FILE"

FAILED=0

run_check() {
  local title="$1"
  shift

  echo "" | tee -a "$LOG_FILE"
  echo "==================================================" | tee -a "$LOG_FILE"
  echo "TEST AREA: $title" | tee -a "$LOG_FILE"
  echo "COMMAND  : $*" | tee -a "$LOG_FILE"
  echo "==================================================" | tee -a "$LOG_FILE"

  if "$@" 2>&1 | tee -a "$LOG_FILE"; then
    echo "RESULT   : PASS" | tee -a "$LOG_FILE"
  else
    echo "RESULT   : FAIL" | tee -a "$LOG_FILE"
    FAILED=1
  fi
}

echo "OpenSportsLib Smoke Test Report" | tee -a "$LOG_FILE"
echo "Repository: $ROOT_DIR" | tee -a "$LOG_FILE"
echo "Log file  : $LOG_FILE" | tee -a "$LOG_FILE"
if [[ "$WITH_INTEGRATION" -eq 1 ]]; then
  echo "Mode      : smoke + integration subset train/infer" | tee -a "$LOG_FILE"
else
  echo "Mode      : smoke only" | tee -a "$LOG_FILE"
fi

run_check \
  "Package imports and lazy module exposure" \
  pytest -q tests/test_package_smoke.py --tb=short

run_check \
  "Public APIs (classification/localization) basic initialization" \
  pytest -q tests/test_public_apis_smoke.py --tb=short

run_check \
  "Config utilities (path expansion, JSON read/write, class mapping)" \
  pytest -q tests/test_config_utils_smoke.py --tb=short

# if [[ "$WITH_INTEGRATION" -eq 1 ]]; then
#   run_check \
#     "Classification + Localization train/infer on tiny real-data subsets" \
#     env RUN_OSL_SUBSET_INTEGRATION=1 OSL_PRETRAINED_WEIGHTS=0 pytest -q tests/test_subset_train_infer_integration.py -m integration --tb=short
# fi

echo "" | tee -a "$LOG_FILE"
echo "==================================================" | tee -a "$LOG_FILE"
if [[ "$FAILED" -eq 0 ]]; then
  echo "OVERALL RESULT: PASS" | tee -a "$LOG_FILE"
  echo "All smoke test areas passed." | tee -a "$LOG_FILE"
  exit 0
else
  echo "OVERALL RESULT: FAIL" | tee -a "$LOG_FILE"
  echo "One or more smoke test areas failed." | tee -a "$LOG_FILE"
  exit 1
fi
