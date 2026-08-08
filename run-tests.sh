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
  live)  target="tests.test_server_live" ;;
  all)   target="" ;;
  *)     echo "usage: $0 [all|units|live]" >&2; exit 2 ;;
esac

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
