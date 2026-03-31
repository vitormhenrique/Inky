"""Post-processing — prepare stylised images for display and archival."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

from src.config import Settings
from src.logging_utils import get_logger
from src.utils.files import safe_filename
from src.utils.image_ops import add_title_bar, fit_to_display

log = get_logger("postprocess")


def prepare_display_image(
    stylised: Image.Image,
    settings: Settings,
    *,
    style_name: str = "",
) -> Image.Image:
    """Resize / letterbox the stylised image for the Inky Impression display."""
    display_img = fit_to_display(
        stylised,
        width=settings.display_width,
        height=settings.display_height,
        border_px=settings.display_border_px,
    )

    if settings.display_add_title and style_name:
        display_img = add_title_bar(display_img, style_name)

    return display_img


def save_outputs(
    stylised: Image.Image,
    display_img: Image.Image,
    settings: Settings,
    *,
    source_name: str,
    style_name: str,
    algorithm: str,
) -> tuple[Path, Path]:
    """Save high-resolution output and display-ready image. Return their paths."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    base = safe_filename(f"{ts}_{source_name}_{style_name}_{algorithm}")

    output_dir = settings.resolve_path(settings.local_output_dir)
    display_dir = settings.resolve_path(settings.local_display_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    display_dir.mkdir(parents=True, exist_ok=True)

    hires_path = output_dir / f"{base}_hires.png"
    display_path = display_dir / f"{base}_display.png"

    stylised.save(hires_path, "PNG")
    display_img.save(display_path, "PNG")

    log.info("Saved hi-res: %s", hires_path)
    log.info("Saved display: %s", display_path)

    return hires_path, display_path
