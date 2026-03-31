#!/usr/bin/env bash
# setup_mac.sh — macOS (Apple Silicon) setup script
# Requires: uv (https://docs.astral.sh/uv/)
set -euo pipefail

echo "═══════════════════════════════════════════════"
echo "  Inky Stylisation System — macOS Setup"
echo "═══════════════════════════════════════════════"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# ── Check for uv ─────────────────────────────────────────────────
if ! command -v uv &> /dev/null; then
    echo "Installing uv…"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    echo "Please restart your shell or run: source \$HOME/.local/bin/env"
    exit 1
fi

# ── Create venv & install core deps ─────────────────────────────
echo "Creating Python 3.14 virtual environment with uv…"
uv venv --python 3.14
echo "Installing core dependencies…"
uv sync

# ── Install diffusion extras (macOS has enough power) ────────────
echo "Installing diffusion dependencies…"
uv pip install "diffusers>=0.25" "transformers>=4.35" "accelerate>=0.25" "safetensors>=0.4"

# ── Install dev dependencies ─────────────────────────────────────
echo "Installing dev dependencies…"
uv sync --group dev

# ── .env file ────────────────────────────────────────────────────
if [[ ! -f .env ]]; then
    cp .env.example .env
    echo "Created .env from .env.example — please edit it with your settings."
fi

# ── Create data directories ──────────────────────────────────────
echo "Creating data directories…"
uv run python -c "from src.config import get_settings; from src.utils.files import ensure_dirs; ensure_dirs(get_settings())"

echo ""
echo "Setup complete!"
echo ""
echo "Quick start:"
echo "  source .venv/bin/activate"
echo "  uv run python -m src.cli run --skip-sync --skip-display -i path/to/test_image.jpg"
echo ""
echo "To schedule daily runs, create a launchd plist or cron entry:"
echo "  crontab -e"
echo "  0 6 * * * $PROJECT_DIR/scripts/run_daily.sh >> $PROJECT_DIR/data/logs/cron.log 2>&1"
echo ""
echo "For diffusion on Apple Silicon, MPS acceleration is auto-detected."
