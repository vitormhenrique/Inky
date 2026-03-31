#!/usr/bin/env bash
# run_daily.sh — wrapper for cron / systemd / launchd
#
# Usage:
#   ./scripts/run_daily.sh
#   ./scripts/run_daily.sh --skip-display
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

echo "[$(date)] Starting daily stylisation run"

# Prefer uv run (handles venv activation automatically)
if command -v uv &> /dev/null; then
    uv run python -m src.cli run "$@"
else
    # Fallback: activate virtualenv manually
    if [[ -f ".venv/bin/activate" ]]; then
        source .venv/bin/activate
    fi
    python -m src.cli run "$@"
fi

echo "[$(date)] Daily run complete"
echo "[$(date)] Daily run complete"
