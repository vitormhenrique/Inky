#!/usr/bin/env bash
# setup_mac.sh — macOS (Apple Silicon) setup script
set -euo pipefail

echo "═══════════════════════════════════════════════"
echo "  Inky Stylisation System — macOS Setup"
echo "═══════════════════════════════════════════════"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# ── Python venv ──────────────────────────────────────────────────
echo "Creating Python virtual environment…"
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools wheel

# ── Install core dependencies ────────────────────────────────────
echo "Installing core Python dependencies…"
pip install -r requirements.txt

# ── Install diffusion extras (macOS has enough power) ────────────
echo "Installing diffusion dependencies…"
pip install "diffusers>=0.25" "transformers>=4.35" "accelerate>=0.25" "safetensors>=0.4"

# ── Install dev dependencies ─────────────────────────────────────
echo "Installing dev dependencies…"
pip install "pytest>=7.4" "pytest-cov>=4.1" "ruff>=0.1"

# ── .env file ────────────────────────────────────────────────────
if [[ ! -f .env ]]; then
    cp .env.example .env
    echo "Created .env from .env.example — please edit it with your settings."
fi

# ── Create data directories ──────────────────────────────────────
echo "Creating data directories…"
python3 -c "from src.config import get_settings; from src.utils.files import ensure_dirs; ensure_dirs(get_settings())"

echo ""
echo "Setup complete!"
echo ""
echo "Quick start:"
echo "  source .venv/bin/activate"
echo "  python -m src.cli run --skip-sync --skip-display -i path/to/test_image.jpg"
echo ""
echo "To schedule daily runs, create a launchd plist or cron entry:"
echo "  crontab -e"
echo "  0 6 * * * $PROJECT_DIR/scripts/run_daily.sh >> $PROJECT_DIR/data/logs/cron.log 2>&1"
echo ""
echo "For diffusion on Apple Silicon, MPS acceleration is auto-detected."
