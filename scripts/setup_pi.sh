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
if command -v uv &> /dev/null; then
    echo "uv already installed: $(uv --version)"
else
    echo "Installing uv…"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
    if ! command -v uv &> /dev/null; then
        echo "ERROR: uv install succeeded but is not on PATH."
        echo "Please restart your shell or run: source \$HOME/.local/bin/env"
        exit 1
    fi
    echo "uv installed: $(uv --version)"
fi

# ── Create venv (skip if already exists with correct Python) ────
if [[ -d ".venv" ]] && .venv/bin/python --version 2>/dev/null | grep -q "3.14"; then
    echo "Python 3.14 venv already exists — skipping creation."
else
    echo "Creating Python 3.14 virtual environment with uv…"
    uv venv --python 3.14
fi

# ── Install / sync deps ─────────────────────────────────────────
echo "Syncing dependencies…"
uv sync

# ── Install Inky display library ─────────────────────────────────
echo "Installing Inky display driver…"
uv pip install "inky[rpi]>=2.0"

# ── .env file ────────────────────────────────────────────────────
if [[ ! -f .env ]]; then
    cp .env.example .env
    echo "Created .env from .env.example — please edit it with your settings."
else
    echo ".env already exists — skipping."
fi

# ── Create data directories ──────────────────────────────────────
echo "Creating data directories…"
uv run python -c "from src.config import get_settings; from src.utils.files import ensure_dirs; ensure_dirs(get_settings())"

# ── Download reference paintings ─────────────────────────────────
if [[ -f "scripts/download_references.py" ]]; then
    EXISTING=$(find data/styles -type f \( -name "*.jpg" -o -name "*.png" \) 2>/dev/null | wc -l | tr -d ' ')
    if [[ "$EXISTING" -lt 50 ]]; then
        echo "Downloading reference style paintings…"
        uv run python scripts/download_references.py
    else
        echo "Reference paintings already present ($EXISTING files) — skipping download."
    fi
fi

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
