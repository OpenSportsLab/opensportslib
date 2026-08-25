#!/usr/bin/env bash
#
# Convenience wrapper for tests/release/ -- extensive real-dataset,
# real-training verification, meant to be run manually after a big release.
# See tests/release/README.md for full documentation.
#
# This does NOT run automatically as part of any other script; it must be
# invoked directly, and requires RUN_OSL_RELEASE_TESTS=1 to actually execute
# anything (the tests skip themselves otherwise).

set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ "${RUN_OSL_RELEASE_TESTS:-}" != "1" ]]; then
  echo "RUN_OSL_RELEASE_TESTS is not set to 1 -- refusing to run."
  echo "This suite downloads real (sometimes large) datasets and runs real"
  echo "training. Set RUN_OSL_RELEASE_TESTS=1 to proceed. See"
  echo "tests/release/README.md for prerequisites and tuning."
  exit 1
fi

echo "OpenSportsLib Release Verification"
echo "Repository: $ROOT_DIR"
echo "Cache dir : ${OSL_RELEASE_CACHE_DIR:-$ROOT_DIR/.release_test_cache}"
echo ""

exec pytest tests/release -v -s "$@"
