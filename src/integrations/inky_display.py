"""Inky Impression 13.3" e-paper display driver.

This module is only functional on Raspberry Pi with the ``inky`` library installed.
On other platforms it provides a *simulation* mode that saves PNGs instead.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from src.config import Settings
from src.logging_utils import get_logger

log = get_logger("inky_display")

_INKY_AVAILABLE: bool | None = None


def is_inky_available() -> bool:
    """Check whether the Inky library is installed and a display is attached."""
    global _INKY_AVAILABLE
    if _INKY_AVAILABLE is None:
        try:
            from inky.auto import auto  # noqa: F401

            _INKY_AVAILABLE = True
        except (ImportError, RuntimeError):
            _INKY_AVAILABLE = False
    return _INKY_AVAILABLE


def update_display(image_path: Path, settings: Settings) -> None:
    """Push an image to the Inky Impression display.

    Falls back to saving a simulation file on non-Pi hardware.
    """
    img = Image.open(image_path).convert("RGB")

    if is_inky_available():
        _send_to_inky(img, settings)
    else:
        _simulate_display(img, image_path, settings)


def _send_to_inky(img: Image.Image, settings: Settings) -> None:
    """Send image to Inky Impression hardware."""
    from inky.auto import auto

    display = auto()
    log.info(
        "Inky display detected: %s (%dx%d)",
        display.resolution,
        display.width,
        display.height,
    )

    # Resize to exact display resolution if needed
    if (img.width, img.height) != (display.width, display.height):
        img = img.resize((display.width, display.height), Image.LANCZOS)

    display.set_image(img)
    display.show()
    log.info("Display updated successfully")


def _simulate_display(img: Image.Image, source_path: Path, settings: Settings) -> None:
    """Save a 'simulated' display image for preview on non-Pi systems."""
    sim_dir = settings.resolve_path(settings.local_display_dir)
    sim_dir.mkdir(parents=True, exist_ok=True)
    sim_path = sim_dir / "simulated_display.png"

    target = (settings.display_width, settings.display_height)
    if (img.width, img.height) != target:
        img = img.resize(target, Image.LANCZOS)

    img.save(sim_path, "PNG")
    log.info("Simulated display saved: %s (Inky hardware not available)", sim_path)
