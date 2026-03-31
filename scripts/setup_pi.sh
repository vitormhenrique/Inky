#!/usr/bin/env bash
# setup_pi.sh — Raspberry Pi setup script
#
# Run on a fresh Raspberry Pi OS (Bookworm or later, 64-bit recommended).
# Installs system dependencies, creates a uv venv, and installs Python packages.
# Requires: uv (https://docs.astral.sh/uv/)
set -euo pipefail

echo "═══════════════════════════════════════════════"
echo "  Inky Stylisation System — Raspberry Pi Setup"
echo "═══════════════════════════════════════════════"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# ── System packages ──────────────────────────────────────────────
echo "Installing system packages…"
sudo apt-get update
sudo apt-get install -y \
    python3-dev \
    libopenblas-dev \
    libjpeg-dev \
    zlib1g-dev \
    libffi-dev \
    git \
    curl

# ── Install uv ───────────────────────────────────────────────────
if ! command -v uv &> /dev/null; then
    echo "Installing uv…"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

# ── Create venv & install deps ───────────────────────────────────
echo "Creating Python 3.14 virtual environment with uv…"
uv venv --python 3.14
echo "Installing Python dependencies…"
uv sync

# ── Install Inky display library ─────────────────────────────────
echo "Installing Inky display driver…"
uv pip install "inky[rpi]>=2.0"

# ── .env file ────────────────────────────────────────────────────
if [[ ! -f .env ]]; then
    cp .env.example .env
    echo "Created .env from .env.example — please edit it with your settings."
fi

# ── Create data directories ──────────────────────────────────────
echo "Creating data directories…"
uv run python -c "from src.config import get_settings; from src.utils.files import ensure_dirs; ensure_dirs(get_settings())"

# ── Cron job suggestion ──────────────────────────────────────────
echo ""
echo "Setup complete!"
echo ""
echo "To schedule daily runs, add a cron entry:"
echo "  crontab -e"
echo "  # Add this line (runs at 6:00 AM):"
echo "  0 6 * * * $PROJECT_DIR/scripts/run_daily.sh >> $PROJECT_DIR/data/logs/cron.log 2>&1"
echo ""
echo "Or create a systemd timer — see README.md for instructions."
echo ""
echo "Test the pipeline with:"
echo "  uv run python -m src.cli run --skip-sync --skip-display -i path/to/test_image.jpg"
