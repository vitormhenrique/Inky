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

# Activate virtualenv if present
if [[ -f ".venv/bin/activate" ]]; then
    source .venv/bin/activate
fi

echo "[$(date)] Starting daily stylisation run"
python -m src.cli run "$@"
echo "[$(date)] Daily run complete"
