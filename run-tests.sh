#!/usr/bin/env bash
# Test runner. Uses the venv interpreter deliberately: the dependencies
# (mcp, torch, sentence-transformers) are installed there and not on the
# system python, and the tests this replaced failed for exactly that reason.
#
#   ./run-tests.sh              # everything
#   ./run-tests.sh units        # hermetic only - no vault, no model, no network
#
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="$REPO/.venv/bin/python"

if [ ! -x "$PY" ]; then
  echo "no interpreter at $PY - create it with:" >&2
  echo "  python3 -m venv $REPO/.venv && $REPO/.venv/bin/pip install -r $REPO/requirements.txt" >&2
  exit 1
fi

export OBSIDIAN_VAULT_PATH="${OBSIDIAN_VAULT_PATH:-$HOME/obsidian/vault-obsidian}"

case "${1:-all}" in
  units) target="tests.test_server_units" ;;
  paths) target="tests.test_server_paths" ;;
  live)  target="tests.test_server_live" ;;
  all)   target="" ;;
  *)     echo "usage: $0 [all|units|paths|live]" >&2; exit 2 ;;
esac

# When live tests were asked for, a missing store is a failure rather than a
# silent skip. Green because nothing ran is the exact lie the tests this suite
# replaced were telling.
if [ "${1:-all}" != "units" ] && [ "${1:-all}" != "paths" ] \
   && [ "${SCMCP_ALLOW_NO_STORE:-0}" != "1" ]; then
  export SCMCP_REQUIRE_LIVE=1
fi

echo "vault: $OBSIDIAN_VAULT_PATH"
if [ -n "$target" ]; then
  "$PY" -m unittest "$target" -v
else
  "$PY" -m unittest discover -s "$REPO/tests" -t "$REPO" -v
fi
rc=$?

echo
[ $rc -eq 0 ] && echo "TESTS GREEN" || echo "TESTS RED"
exit $rc
