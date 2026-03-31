"""Image preprocessing — prepare source images for the stylisation pipeline."""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from src.config import Settings
from src.logging_utils import get_logger
from src.utils.image_ops import composition_safe_crop, load_image, resize_long_edge

log = get_logger("preprocess")


def preprocess(
    image_path: Path,
    settings: Settings,
    *,
    max_long_edge: int | None = None,
    target_ratio: float | None = None,
) -> Image.Image:
    """Load and prepare an image for stylisation.

    Steps:
    1. Load and convert to RGB.
    2. Apply composition-safe crop to *target_ratio* if provided.
    3. Resize so the longest edge ≤ *max_long_edge* (defaults to config value).

    Returns the prepared ``PIL.Image.Image``.
    """
    log.info("Preprocessing: %s", image_path.name)
    img = load_image(image_path)
    log.debug("Original size: %dx%d", img.width, img.height)

    # Composition crop (e.g., to match display aspect ratio)
    if target_ratio is not None:
        img = composition_safe_crop(img, target_ratio)
        log.debug("After crop: %dx%d (ratio %.2f)", img.width, img.height, target_ratio)

    # Resize
    long_edge = max_long_edge or settings.nst_output_long_edge
    img = resize_long_edge(img, long_edge)
    log.debug("After resize: %dx%d", img.width, img.height)

    return img


def preprocess_for_display(
    image_path: Path,
    settings: Settings,
) -> Image.Image:
    """Preprocess specifically for the display aspect ratio."""
    display_ratio = settings.display_width / settings.display_height
    return preprocess(
        image_path,
        settings,
        target_ratio=display_ratio,
    )
