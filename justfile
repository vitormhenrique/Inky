# Inky Stylisation System — Justfile
# Usage: just <recipe> [args...]

set dotenv-load

# directories
raw_dir     := "data/raw"
display_dir := "data/display"
styles_dir  := "data/styles"

# list available recipes
default:
    @just --list

# ── Stylisation ──────────────────────────────────────────────

# stylize a single image with a given style
[group('stylize')]
stylize image style:
    uv run python -m src.cli run \
        --input "{{image}}" \
        --style "{{style}}" \
        --skip-sync --skip-display --skip-upload

# stylize a single image, auto-finding it by partial name in data/raw
[group('stylize')]
stylize-by-name name style:
    #!/usr/bin/env bash
    set -euo pipefail
    match=$(find "{{raw_dir}}" -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" -o -iname "*.bmp" -o -iname "*.tiff" -o -iname "*.webp" \) | grep -i "{{name}}" | head -1)
    if [[ -z "$match" ]]; then
        echo "Error: no image matching '{{name}}' found in {{raw_dir}}"
        exit 1
    fi
    echo "Found: $match"
    uv run python -m src.cli run \
        --input "$match" \
        --style "{{style}}" \
        --skip-sync --skip-display --skip-upload

# stylize ALL images in data/raw with a given style
[group('stylize')]
stylize-all style:
    #!/usr/bin/env bash
    set -euo pipefail
    shopt -s nullglob nocaseglob
    files=()
    for ext in jpg jpeg png bmp tiff webp; do
        files+=("{{raw_dir}}"/*."$ext")
    done
    if [[ ${#files[@]} -eq 0 ]]; then
        echo "No images found in {{raw_dir}}"
        exit 1
    fi
    echo "Processing ${#files[@]} image(s) with style '{{style}}'..."
    for f in "${files[@]}"; do
        echo "──── $(basename "$f") ────"
        uv run python -m src.cli run \
            --input "$f" \
            --style "{{style}}" \
            --skip-sync --skip-display --skip-upload
    done
    echo "Done! All outputs saved to {{display_dir}}"

# stylize ALL images in data/raw with a random style each
[group('stylize')]
stylize-all-random:
    #!/usr/bin/env bash
    set -euo pipefail
    shopt -s nullglob nocaseglob
    files=()
    for ext in jpg jpeg png bmp tiff webp; do
        files+=("{{raw_dir}}"/*."$ext")
    done
    if [[ ${#files[@]} -eq 0 ]]; then
        echo "No images found in {{raw_dir}}"
        exit 1
    fi
    styles=($(ls -1 "{{styles_dir}}"))
    echo "Processing ${#files[@]} image(s) with random styles..."
    for f in "${files[@]}"; do
        s="${styles[$((RANDOM % ${#styles[@]}))]}"
        echo "──── $(basename "$f")  →  $s ────"
        uv run python -m src.cli run \
            --input "$f" \
            --style "$s" \
            --skip-sync --skip-display --skip-upload
    done
    echo "Done! All outputs saved to {{display_dir}}"

# ── Info ─────────────────────────────────────────────────────

# list available styles
[group('info')]
styles:
    @uv run python -m src.cli styles

# list images in data/raw ready to process
[group('info')]
raw:
    #!/usr/bin/env bash
    shopt -s nullglob nocaseglob
    files=()
    for ext in jpg jpeg png bmp tiff webp; do
        files+=("{{raw_dir}}"/*."$ext")
    done
    if [[ ${#files[@]} -eq 0 ]]; then
        echo "No images in {{raw_dir}}"
    else
        echo "${#files[@]} image(s) in {{raw_dir}}:"
        for f in "${files[@]}"; do
            echo "  $(basename "$f")"
        done
    fi

# list reference images for a style
[group('info')]
refs style:
    @ls -1 "{{styles_dir}}/{{style}}/" 2>/dev/null || echo "Style '{{style}}' not found. Run: just styles"

# show current configuration
[group('info')]
config:
    @uv run python -m src.cli config

# ── Setup ────────────────────────────────────────────────────

# download reference paintings from Wikimedia Commons
[group('setup')]
download-refs:
    uv run python scripts/download_references.py

# install project dependencies
[group('setup')]
install:
    uv sync
