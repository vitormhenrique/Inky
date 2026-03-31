"""Image selection logic — picks the next source image to stylise."""

from __future__ import annotations

import random
from pathlib import Path

from src.config import Settings
from src.logging_utils import get_logger
from src.utils.files import list_images, newest_file

log = get_logger("selector")


def select_image(
    settings: Settings,
    *,
    explicit_path: Path | None = None,
    mode_override: str | None = None,
) -> Path:
    """Choose a source image according to the selection hierarchy.

    Priority:
    1. *explicit_path* — a specific file supplied via ``--input``.
    2. *mode_override* or ``settings.selection_mode``:
       - ``latest_parsed``:  newest file in ``cache/parsed/``
       - ``random_raw``:     random file from ``cache/raw/``
       - ``random_any``:     random file from raw + parsed

    Raises ``FileNotFoundError`` if no suitable image is found.
    """
    if explicit_path is not None:
        if not explicit_path.is_file():
            raise FileNotFoundError(f"Explicit input not found: {explicit_path}")
        log.info("Using explicit input: %s", explicit_path)
        return explicit_path

    cache = settings.resolve_path(settings.local_cache_dir)
    mode = mode_override or settings.selection_mode

    if mode == "latest_parsed":
        parsed_dir = cache / "parsed"
        chosen = newest_file(parsed_dir)
        if chosen:
            log.info("Selected latest parsed image: %s", chosen.name)
            return chosen
        log.warning("No parsed images found — falling back to random_raw")
        mode = "random_raw"

    if mode == "random_raw":
        raw_dir = cache / "raw"
        images = list_images(raw_dir)
        if images:
            chosen = random.choice(images)
            log.info("Selected random raw image: %s", chosen.name)
            return chosen
        raise FileNotFoundError(f"No raw images found in {raw_dir}")

    if mode == "random_any":
        raw_dir = cache / "raw"
        parsed_dir = cache / "parsed"
        images = list_images(raw_dir) + list_images(parsed_dir)
        if images:
            chosen = random.choice(images)
            log.info("Selected random image: %s", chosen.name)
            return chosen
        raise FileNotFoundError("No images found in raw or parsed directories")

    raise ValueError(f"Unknown selection mode: {mode}")
