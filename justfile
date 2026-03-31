# Inky Stylisation System — Justfile
# Usage: just <recipe> [args...]

set dotenv-load

# directories (absolute paths for shebang recipe compatibility)
raw_dir     := justfile_directory() / "data/raw"
display_dir := justfile_directory() / "data/display"
styles_dir  := justfile_directory() / "data/styles"

# list available recipes
default:
    @just --list

# ── Stylisation ──────────────────────────────────────────────

# stylize a single image with a given style
# style can be a name (e.g. cubism) or style/reference.jpg (e.g. cubism/picasso_girl.jpg)
[group('stylize')]
stylize image style method='nst' intensity='':
    #!/usr/bin/env bash
    set -euo pipefail

    style_arg="{{style}}"
    ref_flag=""
    # If style contains a /, split into style name + reference image
    if [[ "$style_arg" == */* ]]; then
        style_name="${style_arg%%/*}"
        ref_image="${style_arg}"
        ref_flag="--reference $ref_image"
    else
        style_name="$style_arg"
    fi

    # Resolve input: if not an existing file, search data/raw
    input_arg="{{image}}"
    if [[ ! -f "$input_arg" ]]; then
        match=$(find "{{raw_dir}}" -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" -o -iname "*.bmp" -o -iname "*.tiff" -o -iname "*.webp" \) | grep -i "$input_arg" | head -1)
        if [[ -z "$match" ]]; then
            echo "Error: no image matching '$input_arg' found in {{raw_dir}}"
            exit 1
        fi
        input_arg="$match"
        echo "Found: $input_arg"
    fi

    intensity_flag=""
    if [[ -n "{{intensity}}" ]]; then
        intensity_flag="--style-intensity {{intensity}}"
    fi

    uv run python -m src.cli run \
        --input "$input_arg" \
        --style "$style_name" \
        --algorithm "{{method}}" \
        $ref_flag $intensity_flag \
        --skip-sync --skip-display --skip-upload

# stylize a single image, auto-finding it by partial name in data/raw
[group('stylize')]
stylize-by-name name style method='nst' intensity='':
    #!/usr/bin/env bash
    set -euo pipefail
    match=$(find "{{raw_dir}}" -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" -o -iname "*.bmp" -o -iname "*.tiff" -o -iname "*.webp" \) | grep -i "{{name}}" | head -1)
    if [[ -z "$match" ]]; then
        echo "Error: no image matching '{{name}}' found in {{raw_dir}}"
        exit 1
    fi
    echo "Found: $match"
    intensity_flag=""
    if [[ -n "{{intensity}}" ]]; then
        intensity_flag="--style-intensity {{intensity}}"
    fi
    uv run python -m src.cli run \
        --input "$match" \
        --style "{{style}}" \
        --algorithm "{{method}}" \
        $intensity_flag \
        --skip-sync --skip-display --skip-upload

# stylize ALL images in data/raw with a given style
[group('stylize')]
stylize-all style method='nst' intensity='':
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
    intensity_flag=""
    if [[ -n "{{intensity}}" ]]; then
        intensity_flag="--style-intensity {{intensity}}"
    fi
    echo "Processing ${#files[@]} image(s) with style '{{style}}' ({{method}})..."
    for f in "${files[@]}"; do
        echo "──── $(basename "$f") ────"
        uv run python -m src.cli run \
            --input "$f" \
            --style "{{style}}" \
            --algorithm "{{method}}" \
            $intensity_flag \
            --skip-sync --skip-display --skip-upload
    done
    echo "Done! All outputs saved to {{display_dir}}"

# batch-generate: match images by name, apply a style N times per match
[group('stylize')]
generate name style count method='nst' intensity='':
    #!/usr/bin/env bash
    set -euo pipefail
    mapfile -t matches < <(find "{{raw_dir}}" -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" -o -iname "*.bmp" -o -iname "*.tiff" -o -iname "*.webp" \) | grep -i "{{name}}")
    if [[ ${#matches[@]} -eq 0 ]]; then
        echo "Error: no images matching '{{name}}' in {{raw_dir}}"
        exit 1
    fi
    echo "Found ${#matches[@]} image(s) matching '{{name}}'"
    echo "Generating {{count}} variation(s) per image with style '{{style}}' ({{method}})..."
    intensity_flag=""
    if [[ -n "{{intensity}}" ]]; then
        intensity_flag="--style-intensity {{intensity}}"
    fi
    for f in "${matches[@]}"; do
        if [[ ! -f "$f" ]]; then
            echo "Skipping missing file: $f"
            continue
        fi
        for i in $(seq 1 {{count}}); do
            echo "──── $(basename "$f") [${i}/{{count}}] ────"
            uv run python -m src.cli run \
                --input "$f" \
                --style "{{style}}" \
                --algorithm "{{method}}" \
                $intensity_flag \
                --skip-sync --skip-display --skip-upload
        done
    done
    echo "Done! Generated $((${#matches[@]} * {{count}})) image(s) in {{display_dir}}"

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
            --algorithm nst \
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
